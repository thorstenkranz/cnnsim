"""Chua–Yang cellular neural network simulator in PyTorch.

State equation (per cell, 3x3 neighbourhood, unit time constant):

    dx/dt = -x + A ⊛ y + B ⊛ u + z,      y = f(x) = 0.5 (|x+1| - |x-1|) = clip(x, -1, 1)

Conventions (as in Chua & Roska and the SZTAKI template library):
  * black = +1, white = -1
  * ⊛ is a correlation: A[k, l] multiplies y(i+k-1, j+l-1), same as torch's conv2d
  * B ⊛ u + z is constant in time for a static input, so it is computed once ("bias map")

Tensors are kept as (N, 1, H, W). N can be a batch of images, and optionally each
sample can carry its own template (A, B with shape (N, 3, 3), z with shape (N,)).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Literal

import torch
import torch._dynamo
import torch.nn.functional as F

# Each grid shape / boundary / chunk gets its own specialised kernel. The default limit
# (8) is easily hit when sweeping sizes, after which torch silently falls back to eager.
torch._dynamo.config.cache_size_limit = max(torch._dynamo.config.cache_size_limit, 256)
torch._dynamo.config.recompile_limit = max(torch._dynamo.config.recompile_limit, 256)

Backend = Literal["stencil", "conv"]
Integrator = Literal["euler", "rk4"]


# --------------------------------------------------------------------------- boundary


@dataclass(frozen=True)
class Boundary:
    """Boundary condition for the virtual cells around the grid.

    kind: "fixed"    -> virtual cells have constant value `value` (Dirichlet)
          "zeroflux" -> virtual cells copy their nearest edge cell (Neumann)
          "periodic" -> torus
    """

    kind: Literal["fixed", "zeroflux", "periodic"] = "fixed"
    value: float = 0.0

    def pad(self, t: torch.Tensor) -> torch.Tensor:
        if self.kind == "fixed":
            return F.pad(t, (1, 1, 1, 1), mode="constant", value=self.value)
        if self.kind == "zeroflux":
            return F.pad(t, (1, 1, 1, 1), mode="replicate")
        if self.kind == "periodic":
            return F.pad(t, (1, 1, 1, 1), mode="circular")
        raise ValueError(f"unknown boundary kind {self.kind!r}")


# --------------------------------------------------------------------------- template


@dataclass
class Template:
    """A CNN template plus the run conventions that belong to it.

    x0: initial state, "zero" | "input" | "ones" | "minus_ones"
    boundary: boundary used for both y (feedback) and u (control)
    """

    A: list[list[float]]
    B: list[list[float]]
    z: float
    name: str = ""
    x0: Literal["zero", "input", "ones", "minus_ones"] = "zero"
    boundary: Boundary = field(default_factory=Boundary)
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


def _as_template_batch(t: torch.Tensor) -> torch.Tensor:
    """(3,3) -> (1,3,3); (N,3,3) stays."""
    if t.dim() == 2:
        t = t[None]
    if t.shape[-2:] != (3, 3):
        raise ValueError("only 3x3 templates are supported")
    return t


def correlate3x3(
    t: torch.Tensor, w: torch.Tensor, boundary: Boundary, backend: Backend = "stencil"
) -> torch.Tensor:
    """3x3 correlation of t (N,1,H,W) with w (1|N, 3, 3), boundary-padded, 'same' size.

    backend="stencil": nine shifted multiply-adds. Under torch.compile these fuse with
        the padding and the surrounding elementwise ops into a single kernel.
    backend="conv": F.conv2d (cuDNN / oneDNN). Fast on its own but cannot fuse.
    """
    n, _, h, wd = t.shape
    tp = boundary.pad(t)
    if backend == "conv":
        if w.shape[0] == 1:
            return F.conv2d(tp, w[:, None])
        # per-sample templates -> grouped conv over the batch dimension
        out = F.conv2d(tp.reshape(1, n, h + 2, wd + 2), w[:, None], groups=n)
        return out.reshape(n, 1, h, wd)

    wk = w.reshape(w.shape[0], 1, 3, 3, 1, 1)  # broadcast over (N,1,H,W)
    acc = wk[:, :, 0, 0] * tp[..., 0:h, 0:wd]
    for i in range(3):
        for j in range(3):
            if i == 0 and j == 0:
                continue
            acc = acc + wk[:, :, i, j] * tp[..., i : i + h, j : j + wd]
    return acc


def output(x: torch.Tensor) -> torch.Tensor:
    """Standard piecewise-linear CNN output nonlinearity."""
    return x.clamp(-1.0, 1.0)


# --------------------------------------------------------------------------- simulator


class CNNSimulator:
    """Simulates a (batch of) Chua–Yang CNNs with 3x3 templates.

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
        shape: tuple[int, ...] | None = None,
    ):
        """Turn template + input into tensors: (x0, bias_map, A, boundary)."""
        dev, dt = self.device, self.dtype
        if template is not None:
            A = template.A if A is None else A
            B = template.B if B is None else B
            z = template.z if z is None else z
            x0 = template.x0 if x0 is None else x0
            boundary = template.boundary if boundary is None else boundary
        boundary = boundary or Boundary()
        A = _as_template_batch(torch.as_tensor(A, device=dev, dtype=dt))
        B = _as_template_batch(torch.as_tensor(B if B is not None else torch.zeros(3, 3), device=dev, dtype=dt))
        z = torch.as_tensor(0.0 if z is None else z, device=dev, dtype=dt).reshape(-1, 1, 1, 1)

        if u is not None:
            u = _as_nchw(torch.as_tensor(u, device=dev, dtype=dt))
        elif shape is not None:
            u = torch.zeros(_as_nchw(torch.empty(shape)).shape, device=dev, dtype=dt)
        else:
            raise ValueError("need an input u or a shape")

        n = max(u.shape[0], A.shape[0], B.shape[0], z.shape[0])
        if u.shape[0] != n:
            u = u.expand(n, -1, -1, -1)

        bias = correlate3x3(u, B, boundary, self.backend) + z
        if bias.shape[0] != n:
            bias = bias.expand(n, -1, -1, -1)
        bias = bias.contiguous()

        if x0 is None or isinstance(x0, str):
            kind = x0 or "zero"
            x = {
                "zero": lambda: torch.zeros_like(u),
                "input": lambda: u.clone(),
                "ones": lambda: torch.ones_like(u),
                "minus_ones": lambda: -torch.ones_like(u),
            }[kind]()
        else:
            x = _as_nchw(torch.as_tensor(x0, device=dev, dtype=dt))
        x = x.expand(n, -1, -1, -1).contiguous()
        return x, bias, A, boundary

    # ---------------------------------------------------------------- dynamics

    def _make_steps(self, boundary: Boundary, k: int) -> Callable:
        backend, integrator = self.backend, self.integrator

        def deriv(x, bias, A):
            return -x + correlate3x3(output(x), A, boundary, backend) + bias

        def steps(x, bias, A, dt: float):
            for _ in range(k):
                if integrator == "euler":
                    x = x + dt * deriv(x, bias, A)
                else:
                    k1 = deriv(x, bias, A)
                    k2 = deriv(x + 0.5 * dt * k1, bias, A)
                    k3 = deriv(x + 0.5 * dt * k2, bias, A)
                    k4 = deriv(x + dt * k3, bias, A)
                    x = x + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
            return x

        return steps

    def _get_steps(self, boundary: Boundary, k: int) -> Callable:
        key = (boundary, k)
        if key not in self._compiled:
            fn = self._make_steps(boundary, k)
            if self.compile:
                fn = torch.compile(fn, dynamic=False)
            self._compiled[key] = fn
        return self._compiled[key]

    def _graph_runner(self, x, bias, A, boundary, dt) -> Callable[[], None]:
        """Capture `chunk` steps into a CUDA graph operating on static buffers."""
        fn = self._get_steps(boundary, self.chunk)
        sx, sb, sa = x.clone(), bias.clone(), A.clone()
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(3):  # warm-up (compilation, autotuning, cuDNN plans)
                sx.copy_(fn(sx, sb, sa, dt))
        torch.cuda.current_stream().wait_stream(s)
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            sx.copy_(fn(sx, sb, sa, dt))
        sx.copy_(x)

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
        x, bias, A, boundary = self.prepare(template, u, **prepare_kw)
        return self.integrate(x, bias, A, boundary, steps=steps, dt=dt, tol=tol,
                              check_every=check_every, record_every=record_every)

    @torch.no_grad()
    def integrate(self, x, bias, A, boundary, *, steps=200, dt=0.1, tol=None,
                  check_every=100, record_every=None) -> dict:
        k = self.chunk
        n_chunks = -(-steps // k)
        check_chunks = max(1, check_every // k)
        record_chunks = max(1, record_every // k) if record_every else None
        history = [output(x).clone()] if record_chunks else []

        advance = self.stepper(x, bias, A, boundary, dt)
        taken = 0
        for c in range(1, n_chunks + 1):
            prev = advance.state().clone() if tol is not None and c % check_chunks == 0 else None
            cur = advance()
            taken += k
            if record_chunks and c % record_chunks == 0:
                history.append(output(cur).clone())
            if prev is not None:
                rate = (cur - prev).abs().max().item() / (k * dt)
                if rate < tol:
                    break
        x = advance.state().clone()
        return {"y": output(x), "x": x, "steps": taken, "history": history}

    def stepper(self, x, bias, A, boundary, dt) -> Callable[[], torch.Tensor]:
        """Return advance() which performs `chunk` steps and returns the new state.

        advance.state() gives the current state. Building the stepper does the
        compilation / graph capture once, so repeated advance() calls are cheap.
        """
        if self.cuda_graph:
            replay = self._graph_runner(x, bias, A, boundary, dt)
            st = replay.state

            def advance():
                replay()
                return st

            advance.state = lambda: st  # type: ignore[attr-defined]
            return advance

        fn = self._get_steps(boundary, self.chunk)
        cur = [x]

        def advance():
            cur[0] = fn(cur[0], bias, A, dt)
            return cur[0]

        advance.state = lambda: cur[0]  # type: ignore[attr-defined]
        return advance


def run_template(template: Template, u: torch.Tensor, **kw) -> torch.Tensor:
    """Convenience: run a template with sensible defaults, return output y."""
    sim_kw = {k: kw.pop(k) for k in list(kw) if k in ("device", "backend", "integrator", "compile", "cuda_graph", "chunk")}
    return CNNSimulator(**sim_kw).run(template, u, **kw)["y"]
