"""Functional core of the Chua–Yang dynamics. Pure, differentiable tensor functions.

    dx/dt = -x + A ⊛ f(x) + B ⊛ u + z

Shapes: states/inputs (N, 1, H, W); templates (1|N, k, k) with odd k (A and B may differ
in size); z broadcastable to (N, 1, H, W). The control term B ⊛ u + z is constant for a
static input and computed once ("bias map").
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Literal

import torch
import torch.nn.functional as F

from .boundary import Boundary
from .outputs import Output

Backend = Literal["stencil", "conv", "fused"]
Integrator = Literal["euler", "rk4"]


def as_template_batch(t: torch.Tensor) -> torch.Tensor:
    """(k,k) -> (1,k,k); (N,k,k) stays. k must be odd."""
    if t.dim() == 2:
        t = t[None]
    if t.dim() != 3 or t.shape[-1] != t.shape[-2] or t.shape[-1] % 2 == 0:
        raise ValueError(f"templates must be odd k×k (optionally batched), got {tuple(t.shape)}")
    return t


def correlate(t: torch.Tensor, w: torch.Tensor, boundary: Boundary, backend: Backend = "stencil") -> torch.Tensor:
    """Same-size correlation of `t` with the templates `w`, boundary-padded.

    Args:
        t: Tensor of shape `(N, 1, H, W)`.
        w: Templates of shape `(1, k, k)` (shared) or `(N, k, k)` (one per sample), odd k.
        boundary: How the virtual cells around the grid are filled.
        backend: `"stencil"`: k² shifted multiply-adds; under `torch.compile` they fuse with
            the padding and the surrounding elementwise operations into one kernel.
            `"conv"`: `F.conv2d` (cuDNN); fast on its own but cannot fuse.

    Returns:
        Tensor of shape `(N, 1, H, W)`.
    """
    n, _, h, wd = t.shape
    k = w.shape[-1]
    r = k // 2
    tp = boundary.pad(t, r)
    if backend == "conv":
        if w.shape[0] == 1:
            return F.conv2d(tp, w[:, None])
        # per-sample templates -> grouped conv over the batch dimension
        out = F.conv2d(tp.reshape(1, n, h + 2 * r, wd + 2 * r), w[:, None], groups=n)
        return out.reshape(n, 1, h, wd)

    wk = w.reshape(w.shape[0], 1, k, k, 1, 1)  # broadcast over (N,1,H,W)
    acc = None
    for i in range(k):
        for j in range(k):
            term = wk[:, :, i, j] * tp[..., i : i + h, j : j + wd]
            acc = term if acc is None else acc + term
    return acc


def bias_map(
    u: torch.Tensor, B: torch.Tensor, z: torch.Tensor, boundary: Boundary, backend: Backend = "stencil"
) -> torch.Tensor:
    """The control term `B ⊛ u + z`, constant in time for a static input."""
    return correlate(u, B, boundary, backend) + z


def derivative(
    x: torch.Tensor, bias: torch.Tensor, A: torch.Tensor, boundary: Boundary, out: Output, backend: Backend = "stencil"
) -> torch.Tensor:
    """Right-hand side `dx/dt = -x + A ⊛ f(x) + bias` of the state equation."""
    return -x + correlate(out(x), A, boundary, backend) + bias


def make_steps(
    boundary: Boundary,
    out: Output,
    k: int,
    integrator: Integrator = "euler",
    backend: Backend = "stencil",
) -> Callable[[torch.Tensor, torch.Tensor, torch.Tensor, float], torch.Tensor]:
    """Build a function `steps(x, bias, A, dt)` that performs `k` integration steps.

    The returned function is differentiable with respect to all its tensor arguments and can
    be wrapped with `torch.compile` (see `compile_steps`).
    """

    def f(x, bias, A):
        return derivative(x, bias, A, boundary, out, backend)

    def steps(x, bias, A, dt: float):
        for _ in range(k):
            if integrator == "euler":
                x = x + dt * f(x, bias, A)
            else:
                k1 = f(x, bias, A)
                k2 = f(x + 0.5 * dt * k1, bias, A)
                k3 = f(x + 0.5 * dt * k2, bias, A)
                k4 = f(x + dt * k3, bias, A)
                x = x + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        return x

    return steps


def compile_steps(fn: Callable) -> Callable:
    """torch.compile a step function (static shapes).

    Every grid shape / boundary / output / step count gets its own specialised kernel, and all of
    them share the code object of `make_steps`' inner function. torch's default recompile limit
    (8) is easily hit when sweeping sizes, after which it silently falls back to eager mode, so
    this raises the limit to at least 256. This changes a process-wide torch setting; it is done
    only when compilation is requested (`compile=True`), not on import.
    """
    import torch._dynamo

    cfg = torch._dynamo.config
    for name in ("cache_size_limit", "recompile_limit"):
        if hasattr(cfg, name):
            setattr(cfg, name, max(getattr(cfg, name), 256))
    return torch.compile(fn, dynamic=False)
