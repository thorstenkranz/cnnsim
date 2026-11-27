"""Fused Triton kernels for the Euler-integrated CNN: forward and discrete adjoint.

One Triton program integrates one grid for all steps. The state x stays in registers; per
step only y = f(x) goes through a per-grid scratch buffer (L1/L2 resident) so that cells can
read their neighbours. Boundaries are index arithmetic, so no padded tensors are created.

Backward (discrete adjoint of x_{t+1} = (1-dt) x_t + dt (A ⊛ f(x_t) + bias)):
    g_t      = (1-dt) g_{t+1} + dt f'(x_t) ⊙ (Aᵀ g_{t+1})
    dL/dA[o] = dt Σ_t Σ_p g_{t+1}(p) f(x_t)(nb(p, o))
    dL/dbias = dt Σ_t g_{t+1}
where Aᵀ is the transpose of the boundary-padded correlation. The forward pass stores x_t
for all steps when gradients are required (N · steps · H · W floats).

Limits: Euler integrator, odd k ≤ 7, grids with H·W ≤ 16384 and H, W ≥ k; otherwise use the
"stencil" backend.
"""

from __future__ import annotations

import importlib.util
import os

import torch

from .boundary import Boundary
from .outputs import Output

_BOUND = {"fixed": 0, "zeroflux": 1, "periodic": 2, "spiral": 3}
_OUT = {"pwl": 0, "sigmoid": 1, "tanh": 2, "identity": 3, "sign": 4}
MAX_CELLS = 16384
# Training (backward): the dA reductions grow with k². For k > 3 the compiled stencil path is
# faster only with the spiral boundary (its padding compiles well); with fixed / zeroflux /
# periodic padding the compiled k = 5 path is 2.4-16x slower than fused (RTX 5060 Ti, 64x64).
MAX_K_BACKWARD = 3  # applies to the boundaries in COMPILED_PREFERRED only
COMPILED_PREFERRED = frozenset({"spiral"})


def prefer_for_training(boundary: Boundary, k: int) -> bool:
    """Whether the fused backward is the faster choice for training with this boundary and k."""
    return k <= MAX_K_BACKWARD or boundary.kind not in COMPILED_PREFERRED


# --------------------------------------------------------------------------- host side


HAVE_TRITON = importlib.util.find_spec("triton") is not None


def _k():
    from . import _fused_kernels

    return _fused_kernels


def _next_pow2(n: int) -> int:
    return 1 << (n - 1).bit_length()


def supported(
    boundary: Boundary,
    out: Output,
    k: int,
    h: int,
    w: int,
    device: torch.device | str = "cuda",
    dtype: torch.dtype = torch.float32,
) -> bool:
    """Whether the fused kernels handle this configuration (the kernels compute in float32 only)."""
    return (
        HAVE_TRITON
        and torch.device(device).type == "cuda"
        and dtype == torch.float32
        and boundary.kind in _BOUND
        and out.kind in _OUT
        and k % 2 == 1
        and k <= 7
        and h * w <= MAX_CELLS
        and h >= k
        and w >= k
    )


def _launch_cfg(h: int, w: int, k: int = 3, kernel: str = "fwd"):
    BH, BW = _next_pow2(h), _next_pow2(w)
    cells = BH * BW
    env = os.environ.get(f"CNNSIM_WARPS_{kernel.upper()}")
    if env:
        return BH, BW, int(env)
    # few cells per thread: the unrolled k×k neighbourhood otherwise spills registers
    warps = max(4, min(32, cells // 128))
    return BH, BW, warps


def _check_float32(**tensors: torch.Tensor) -> None:
    bad = {n: t.dtype for n, t in tensors.items() if t.dtype != torch.float32}
    if bad:
        raise TypeError(f"the fused backend computes in float32 only, got {bad}; use the stencil backend")


def _consts(boundary: Boundary, out: Output):
    return _BOUND[boundary.kind], _OUT[out.kind], float(out.beta), float(boundary.value)


def forward(
    x0: torch.Tensor,
    bias: torch.Tensor,
    A: torch.Tensor,
    boundary: Boundary,
    out: Output,
    steps: int,
    dt: float,
    store_history: bool = False,
):
    """Integrate `steps` Euler steps. x0, bias (N,1,H,W); A (1|N, k, k). Returns (x_T, history)."""
    _check_float32(x0=x0, bias=bias, A=A)
    N, _, H, W = x0.shape
    k = A.shape[-1]
    x0 = x0.contiguous()
    bias = bias.expand(N, 1, H, W).contiguous()
    A = A.contiguous()
    a_stride = 0 if A.shape[0] == 1 else k * k
    xout = torch.empty_like(x0)
    scratch = torch.empty(N, H * W, device=x0.device, dtype=torch.float32)
    hist = torch.empty(N, steps, H * W, device=x0.device, dtype=torch.float32) if store_history else scratch
    bnd, outc, beta, bval = _consts(boundary, out)
    BH, BW, warps = _launch_cfg(H, W, k, "fwd")
    _k()._fwd_kernel[(N,)](
        x0,
        bias,
        A,
        a_stride,
        xout,
        hist,
        scratch,
        H,
        W,
        steps,
        float(dt),
        beta,
        bval,
        K=k,
        BOUND=bnd,
        OUT=outc,
        STORE=store_history,
        BH=BH,
        BW=BW,
        num_warps=warps,
    )
    return xout, (hist if store_history else None)


def backward(
    gT: torch.Tensor,
    hist: torch.Tensor,
    A: torch.Tensor,
    boundary: Boundary,
    out: Output,
    dt: float,
    grad_A: bool = True,
):
    """Adjoint pass. Returns (d bias (N,1,H,W), d A (N,k,k), d x0 (N,1,H,W))."""
    N, _, H, W = gT.shape
    k = A.shape[-1]
    steps = hist.shape[1]
    _check_float32(gT=gT, hist=hist, A=A)
    gT = gT.contiguous()
    A = A.contiguous()
    a_stride = 0 if A.shape[0] == 1 else k * k
    sy = torch.empty(N, H * W, device=gT.device, dtype=torch.float32)
    sg = torch.empty_like(sy)
    gb = torch.empty_like(gT)
    g0 = torch.empty_like(gT)
    BH, BW, warps = _launch_cfg(H, W, k, "bwd")
    # per-step, per-row partial sums of dA (summed on the host); written completely by the kernel
    gA = torch.empty(N, steps, k * k, BH, device=gT.device, dtype=torch.float32) if grad_A else gT.new_zeros(1)
    bnd, outc, beta, bval = _consts(boundary, out)
    _k()._bwd_kernel[(N,)](
        gT,
        hist,
        A,
        a_stride,
        sy,
        sg,
        gb,
        gA,
        g0,
        H,
        W,
        steps,
        float(dt),
        beta,
        bval,
        K=k,
        BOUND=bnd,
        OUT=outc,
        GRAD_A=grad_A,
        BH=BH,
        BW=BW,
        num_warps=warps,
    )
    gA = gA.sum((1, 3)).view(N, k, k) if grad_A else torch.zeros(N, k, k, device=gT.device)
    return gb, gA, g0


class _FusedEuler(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x0, bias, A, boundary, out, steps, dt):
        need = ctx.needs_input_grad[0] or ctx.needs_input_grad[1] or ctx.needs_input_grad[2]
        xT, hist = forward(x0, bias, A, boundary, out, steps, dt, store_history=need)
        if need:
            ctx.save_for_backward(hist, A)
            ctx.meta = (boundary, out, dt, x0.shape[0], bias.shape[0], A.shape[0])
        return xT

    @staticmethod
    def backward(ctx, gxT):
        hist, A = ctx.saved_tensors
        boundary, out, dt, n_x, n_b, n_a = ctx.meta
        gb, gA, g0 = backward(gxT, hist, A, boundary, out, dt, grad_A=ctx.needs_input_grad[2])
        if n_b != gb.shape[0]:
            gb = gb.sum(0, keepdim=True)
        if n_a != gA.shape[0]:
            gA = gA.sum(0, keepdim=True)
        if n_x != g0.shape[0]:
            g0 = g0.sum(0, keepdim=True)
        return g0, gb, gA, None, None, None, None


def integrate(
    x0: torch.Tensor, bias: torch.Tensor, A: torch.Tensor, boundary: Boundary, out: Output, steps: int, dt: float
) -> torch.Tensor:
    """Differentiable fused Euler integration: returns x after `steps` steps (N,1,H,W)."""
    N = max(x0.shape[0], bias.shape[0], A.shape[0])
    x0 = x0.expand(N, *x0.shape[1:])
    return _FusedEuler.apply(x0, bias, A, boundary, out, steps, dt)
