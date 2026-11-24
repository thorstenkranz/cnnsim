"""Chua–Yang cellular neural network simulator in PyTorch (fast inference path).

State equation (per cell, k×k neighbourhood, unit time constant):

    dx/dt = -x + A ⊛ y + B ⊛ u + z,      y = f(x)   (default f = clip(x, -1, 1))

Conventions (as in Chua & Roska and the SZTAKI template library):
  * black = +1, white = -1
  * ⊛ is a correlation: A[k, l] multiplies y(i+k-r, j+l-r), same as torch's conv2d
  * B ⊛ u + z is constant in time for a static input, so it is computed once ("bias map")

Tensors are kept as (N, 1, H, W). N can be a batch of images, and optionally each
sample can carry its own template (A, B with shape (N, k, k), z with shape (N,)).
This class runs without gradients, with optional torch.compile and CUDA graphs. For
training, use the differentiable functions in `cnnsim.dynamics`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Literal

import torch
import torch._dynamo

from .boundary import Boundary
from .dynamics import Backend, Integrator, as_template_batch, bias_map, correlate, make_steps
from .outputs import PWL, Output

# Each grid shape / boundary / chunk gets its own specialised kernel. The default limit
# (8) is easily hit when sweeping sizes, after which torch silently falls back to eager.
torch._dynamo.config.cache_size_limit = max(torch._dynamo.config.cache_size_limit, 256)
torch._dynamo.config.recompile_limit = max(torch._dynamo.config.recompile_limit, 256)

X0Kind = Literal["zero", "input", "ones", "minus_ones"]


# --------------------------------------------------------------------------- template


@dataclass
class Template:
    """A CNN template plus the run conventions that belong to it.

    A, B: odd k×k (sizes may differ); z: bias (scalar)
    x0: initial state, "zero" | "input" | "ones" | "minus_ones"
    boundary: boundary for the feedback term (and for the input unless boundary_u is set)
    output: cell output function
    """

    A: list[list[float]]
    B: list[list[float]]
    z: float
    name: str = ""
    x0: X0Kind = "zero"
    boundary: Boundary = field(default_factory=Boundary)
    boundary_u: Boundary | None = None
    output: Output = PWL
    description: str = ""


# --------------------------------------------------------------------------- helpers


def _as_nchw(t: torch.Tensor) -> torch.Tensor:
    if t.dim() == 2:
        return t[None, None]
    if t.dim() == 3:
        return t[:, None]
    if t.dim() == 4:
        return t
    raise ValueError(f"expected 2-, 3- or 4-D tensor, got shape {tuple(t.shape)}")


def _as_bias(z, dev, dt) -> torch.Tensor:
    """z: float | (N,) per sample | (H,W) map | (N,H,W) | (N,1,H,W) -> broadcastable to NCHW."""
    z = torch.as_tensor(0.0 if z is None else z, device=dev, dtype=dt)
    if z.dim() <= 1:
        return z.reshape(-1, 1, 1, 1)
    return _as_nchw(z)


def correlate3x3(t, w, boundary: Boundary, backend: Backend = "stencil"):
    """Backwards-compatible name for `cnnsim.dynamics.correlate` (any odd k)."""
    return correlate(t, w, boundary, backend)


def output(x: torch.Tensor) -> torch.Tensor:
    """Standard piecewise-linear CNN output nonlinearity."""
    return PWL(x)


def initial_state(kind: X0Kind | torch.Tensor | None, u: torch.Tensor) -> torch.Tensor:
    if kind is None or isinstance(kind, str):
        return {
            "zero": lambda: torch.zeros_like(u),
            "input": lambda: u.clone(),
            "ones": lambda: torch.ones_like(u),
            "minus_ones": lambda: -torch.ones_like(u),
        }[kind or "zero"]()
    return _as_nchw(torch.as_tensor(kind, device=u.device, dtype=u.dtype))


# --------------------------------------------------------------------------- simulator


@dataclass
class Prepared:
    """Everything the integrator needs."""

    x: torch.Tensor
    bias: torch.Tensor
    A: torch.Tensor
    boundary: Boundary
    output: Output


class CNNSimulator:
    """Simulates a (batch of) Chua–Yang CNNs.

    Args:
        device: "cuda", "cpu", ... (default: cuda if available)
        dtype: float32 is plenty for CNN dynamics
        backend: "stencil" (fusable, default) or "conv"
        integrator: "euler" (classic for CNN, dt ~ 0.1-0.2) or "rk4"
        compile: wrap the inner step loop with torch.compile
        cuda_graph: capture `chunk` steps as one CUDA graph (GPU only). Removes almost
            all launch overhead, which dominates on small grids.
        chunk: number of steps executed per compiled call / graph replay
    """

    def __init__(
        self,
        device: str | torch.device | None = None,
        dtype: torch.dtype = torch.float32,
        backend: Backend = "stencil",
        integrator: Integrator = "euler",
        compile: bool = False,
        cuda_graph: bool = False,
        chunk: int = 20,
    ):
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.dtype = dtype
        self.backend = backend
        self.integrator = integrator
        self.compile = compile
        self.cuda_graph = cuda_graph and self.device.type == "cuda"
        self.chunk = chunk
        self._compiled: dict = {}

    # ---------------------------------------------------------------- setup

    def prepare(
        self,
        template: Template | None = None,
        u: torch.Tensor | None = None,
        *,
        A: torch.Tensor | None = None,
        B: torch.Tensor | None = None,
        z: torch.Tensor | float | None = None,
        x0: torch.Tensor | str | None = None,
        boundary: Boundary | None = None,
        boundary_u: Boundary | None = None,
        output: Output | None = None,
        shape: tuple[int, ...] | None = None,
    ) -> Prepared:
        """Turn template + input into tensors. Keyword arguments override the template."""
        dev, dt = self.device, self.dtype
        if template is not None:
            A = template.A if A is None else A
            B = template.B if B is None else B
            z = template.z if z is None else z
            x0 = template.x0 if x0 is None else x0
            boundary = template.boundary if boundary is None else boundary
            boundary_u = template.boundary_u if boundary_u is None else boundary_u
            output = template.output if output is None else output
        boundary = boundary or Boundary()
        boundary_u = boundary_u or boundary
        output = output or PWL
        A = as_template_batch(torch.as_tensor(A, device=dev, dtype=dt))
        B = as_template_batch(torch.as_tensor(B if B is not None else torch.zeros(1, 1), device=dev, dtype=dt))
        z = _as_bias(z, dev, dt)

        if u is not None:
            u = _as_nchw(torch.as_tensor(u, device=dev, dtype=dt))
        elif shape is not None:
            u = torch.zeros(_as_nchw(torch.empty(shape)).shape, device=dev, dtype=dt)
        else:
            raise ValueError("need an input u or a shape")

        n = max(u.shape[0], A.shape[0], B.shape[0], z.shape[0])
        if u.shape[0] != n:
            u = u.expand(n, -1, -1, -1)
        bias = bias_map(u, B, z, boundary_u, self.backend)
        bias = bias.expand(n, -1, -1, -1).contiguous()
        x = initial_state(x0, u).expand(n, -1, -1, -1).contiguous()
        return Prepared(x, bias, A, boundary, output)

    # ---------------------------------------------------------------- dynamics

    def _get_steps(self, boundary: Boundary, out: Output, k: int) -> Callable:
        key = (boundary, out, k)
        if key not in self._compiled:
            fn = make_steps(boundary, out, k, self.integrator, self.backend)
            if self.compile:
                fn = torch.compile(fn, dynamic=False)
            self._compiled[key] = fn
        return self._compiled[key]

    def _graph_runner(self, p: Prepared, dt: float) -> Callable[[], None]:
        """Capture `chunk` steps into a CUDA graph operating on static buffers."""
        fn = self._get_steps(p.boundary, p.output, self.chunk)
        sx, sb, sa = p.x.clone(), p.bias.clone(), p.A.clone()
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(3):  # warm-up (compilation, autotuning, cuDNN plans)
                sx.copy_(fn(sx, sb, sa, dt))
        torch.cuda.current_stream().wait_stream(s)
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            sx.copy_(fn(sx, sb, sa, dt))
        sx.copy_(p.x)

        def replay():
            g.replay()

        replay.state = sx  # type: ignore[attr-defined]
        return replay

    # ---------------------------------------------------------------- public API

    @torch.no_grad()
    def run(
        self,
        template: Template | None = None,
        u: torch.Tensor | None = None,
        *,
        steps: int = 200,
        dt: float = 0.1,
        tol: float | None = None,
        check_every: int = 100,
        record_every: int | None = None,
        **prepare_kw,
    ) -> dict:
        """Integrate the network.

        steps: maximum number of integration steps (rounded up to a multiple of chunk)
        tol: stop early when max |dx/dt| < tol (checked every `check_every` steps)
        record_every: keep a snapshot of y every this many steps (multiple of chunk)

        Returns dict with "y" (output), "x" (state), "steps" (steps taken),
        "history" (list of y snapshots, if requested).
        """
        p = self.prepare(template, u, **prepare_kw)
        return self.integrate(p, steps=steps, dt=dt, tol=tol, check_every=check_every, record_every=record_every)

    @torch.no_grad()
    def integrate(self, p: Prepared, *, steps=200, dt=0.1, tol=None, check_every=100, record_every=None) -> dict:
        k = self.chunk
        n_chunks = -(-steps // k)
        check_chunks = max(1, check_every // k)
        record_chunks = max(1, record_every // k) if record_every else None
        history = [p.output(p.x).clone()] if record_chunks else []

        advance = self.stepper(p, dt)
        taken = 0
        for c in range(1, n_chunks + 1):
            prev = advance.state().clone() if tol is not None and c % check_chunks == 0 else None
            cur = advance()
            taken += k
            if record_chunks and c % record_chunks == 0:
                history.append(p.output(cur).clone())
            if prev is not None:
                rate = (cur - prev).abs().max().item() / (k * dt)
                if rate < tol:
                    break
        x = advance.state().clone()
        return {"y": p.output(x), "x": x, "steps": taken, "history": history}

    def stepper(self, p: Prepared, dt: float) -> Callable[[], torch.Tensor]:
        """Return advance() which performs `chunk` steps and returns the new state.

        advance.state() gives the current state. Building the stepper does the
        compilation / graph capture once, so repeated advance() calls are cheap.
        """
        if self.cuda_graph:
            replay = self._graph_runner(p, dt)
            st = replay.state

            def advance():
                replay()
                return st

            advance.state = lambda: st  # type: ignore[attr-defined]
            return advance

        fn = self._get_steps(p.boundary, p.output, self.chunk)
        cur = [p.x]
        bias, A = p.bias, p.A

        def advance():
            cur[0] = fn(cur[0], bias, A, dt)
            return cur[0]

        advance.state = lambda: cur[0]  # type: ignore[attr-defined]
        return advance


def run_template(template: Template, u: torch.Tensor, **kw) -> torch.Tensor:
    """Convenience: run a template with sensible defaults, return output y."""
    sim_kw = {k: kw.pop(k) for k in list(kw) if k in ("device", "backend", "integrator", "compile", "cuda_graph", "chunk")}
    return CNNSimulator(**sim_kw).run(template, u, **kw)["y"]
