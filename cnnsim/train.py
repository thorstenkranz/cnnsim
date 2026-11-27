"""Training CNN templates by backpropagation through the dynamics.

The Euler/RK4 integration is an ordinary differentiable computation, so the gradient of
any loss on the final output with respect to (A, B, z) is exact (backpropagation through
time). Memory is bounded with gradient checkpointing: only the state at chunk boundaries
is stored, chunks are recomputed in the backward pass.

Population × data batching: templates have a population dimension P (independent
candidates, e.g. multi-start), the data a batch dimension D. Every candidate is run on
every sample in one batched simulation of P·D grids; outputs come back as (P, D, 1, H, W)
and losses per candidate as (P,), so the candidates are trained simultaneously and
independently.
"""

from __future__ import annotations

import math
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Literal

import torch
from torch.utils.checkpoint import checkpoint

from . import fused
from .boundary import Boundary
from .dynamics import Integrator, bias_map, compile_steps, derivative, make_steps
from .outputs import PWL, Output
from .params import TemplateParams

X0 = Literal["input", "zero", "ones", "minus_ones"]


@dataclass(frozen=True)
class Dynamics:
    """Everything about a run except the templates and the data.

    Attributes:
        boundary: Boundary for the feedback term (and the input unless `boundary_u` is set).
        boundary_u: Boundary for the input term, if it differs.
        output: Cell output function.
        steps: Number of integration steps.
        dt: Time step.
        integrator: `"euler"` or `"rk4"`.
        x0: Initial state: `"input"`, `"zero"`, `"ones"` or `"minus_ones"`.
        checkpoint_every: Steps per gradient-checkpointed chunk; 0 disables checkpointing.
        compile: `torch.compile` the step chunks (forward and backward).
        backend: `"stencil"`, or `"fused"` for the Triton kernels (Euler only; see
            `cnnsim.fused`). Fused is used for all forward passes and for training where
            `fused.prefer_for_training` says so; otherwise the stencil path (then set
            `compile=True`).
    """

    boundary: Boundary = Boundary("fixed")
    boundary_u: Boundary | None = None
    output: Output = PWL
    steps: int = 200
    dt: float = 0.2
    integrator: Integrator = "euler"
    x0: X0 = "input"
    checkpoint_every: int = 20
    compile: bool = False
    backend: Literal["stencil", "fused"] = "stencil"


_STEP_CACHE: dict = {}


def _steps_fn(dyn: Dynamics, n: int):
    key = (dyn.boundary, dyn.output, n, dyn.integrator, dyn.compile)
    if key not in _STEP_CACHE:
        fn = make_steps(dyn.boundary, dyn.output, n, dyn.integrator)
        _STEP_CACHE[key] = compile_steps(fn) if dyn.compile else fn
    return _STEP_CACHE[key]


def _initial(kind: X0 | torch.Tensor, u: torch.Tensor) -> torch.Tensor:
    if isinstance(kind, torch.Tensor):
        return kind
    return {"input": u, "zero": torch.zeros_like(u), "ones": torch.ones_like(u), "minus_ones": -torch.ones_like(u)}[
        kind
    ]


def simulate(
    templates: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    u: torch.Tensor,
    dyn: Dynamics,
    x0: torch.Tensor | None = None,
    return_state: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Differentiable simulation of P candidates on D samples.

    Args:
        templates: `(A, B, z)` with shapes `(P, kA, kA)`, `(P, kB, kB)`, `(P,)`, e.g. the
            output of a `TemplateParams` module.
        u: Inputs of shape `(D, 1, H, W)`.
        dyn: Boundary, output function, steps, integrator and backend.
        x0: Initial states of shape `(D, 1, H, W)`; default: `dyn.x0`.
        return_state: Also return the final state and the bias map.

    Returns:
        The outputs y of shape `(P, D, 1, H, W)`; with `return_state`, the tuple
            `(y, x, bias)`, all of that shape.
    """
    A, B, z = templates
    P, D = A.shape[0], u.shape[0]
    shp = u.shape[1:]
    uu = u.unsqueeze(0).expand(P, D, *shp).reshape(P * D, *shp)
    xx = (x0 if x0 is not None else _initial(dyn.x0, u)).unsqueeze(0).expand(P, D, *shp).reshape(P * D, *shp)
    rep = lambda t: t.repeat_interleave(D, dim=0)
    bias = bias_map(uu, rep(B), rep(z).reshape(-1, 1, 1, 1), dyn.boundary_u or dyn.boundary)
    AA = rep(A)

    use_fused = (
        dyn.backend == "fused"
        and dyn.integrator == "euler"
        and fused.supported(dyn.boundary, dyn.output, AA.shape[-1], *xx.shape[-2:], xx.device, xx.dtype)
    )
    grad = torch.is_grad_enabled() and (A.requires_grad or B.requires_grad or z.requires_grad)
    if use_fused and grad and not fused.prefer_for_training(dyn.boundary, AA.shape[-1]):
        use_fused = False  # training with large templates: the compiled stencil path is faster
    if use_fused:
        x = fused.integrate(xx, bias, AA, dyn.boundary, dyn.output, dyn.steps, dyn.dt)
        y = dyn.output(x).reshape(P, D, *shp)
        if return_state:
            return y, x.reshape(P, D, *shp), bias.reshape(P, D, *shp)
        return y

    k = dyn.checkpoint_every or dyn.steps
    x = xx
    done = 0
    while done < dyn.steps:
        n = min(k, dyn.steps - done)
        fn = _steps_fn(dyn, n)
        if dyn.checkpoint_every and torch.is_grad_enabled():
            x = checkpoint(fn, x, bias, AA, dyn.dt, use_reentrant=False)
        else:
            x = fn(x, bias, AA, dyn.dt)
        done += n
    y = dyn.output(x).reshape(P, D, *shp)
    if return_state:
        return y, x.reshape(P, D, *shp), bias.reshape(P, D, *shp)
    return y


def settle_rate(x: torch.Tensor, bias: torch.Tensor, A: torch.Tensor, dyn: Dynamics) -> torch.Tensor:
    """Mean |dx/dt| at state x per candidate: x, bias (P,D,1,H,W), A (P,k,k) -> (P,).
    Small values mean the network has converged; usable as a regulariser."""
    P, D = x.shape[:2]
    shp = x.shape[2:]
    dx = derivative(
        x.reshape(P * D, *shp), bias.reshape(P * D, *shp), A.repeat_interleave(D, 0), dyn.boundary, dyn.output
    )
    return dx.abs().reshape(P, -1).mean(1)


# --------------------------------------------------------------------------- fitting

LossFn = Callable[[torch.Tensor, dict], torch.Tensor]  # (y (P,D,1,H,W), batch) -> (P,)


@dataclass
class FitResult:
    """Result of `fit`. `best()` returns the best templates `(A, B, z)` per candidate."""

    params: TemplateParams  # parameters at the end of training
    best: TemplateParams  # per member: parameters with the lowest monitored loss
    best_loss: torch.Tensor  # (P,)
    history: dict[str, list] = field(default_factory=dict)  # "train": [(P,)...], "val": [...]
    seconds: float = 0.0


def _batches(n: int, batch_size: int | None, gen: torch.Generator | None):
    if not batch_size or batch_size >= n:
        yield torch.arange(n)
        return
    perm = torch.randperm(n, generator=gen)
    for i in range(0, n, batch_size):
        yield perm[i : i + batch_size]


def _select(data: dict, idx: torch.Tensor) -> dict:
    return {k: (v[idx.to(v.device)] if isinstance(v, torch.Tensor) and v.dim() > 0 else v) for k, v in data.items()}


def evaluate(
    params: TemplateParams, loss_fn: LossFn, data: dict, dyn: Dynamics, batch_size: int | None = None
) -> torch.Tensor:
    """Loss per candidate (P,) on `data` without gradients (sample-weighted over batches)."""
    n = data["u"].shape[0]
    total = 0.0
    with torch.no_grad():
        for idx in _batches(n, batch_size, None):
            b = _select(data, idx)
            total = total + loss_fn(simulate(params(), b["u"], dyn, b.get("x0")), b) * len(idx)
    return total / n


def fit(
    params: TemplateParams,
    loss_fn: LossFn,
    train: dict,
    dyn: Dynamics,
    *,
    val: dict | None = None,
    epochs: int = 100,
    lr: float = 0.05,
    weight_decay: float = 0.0,
    batch_size: int | None = None,
    grad_clip: float | None = 1.0,
    schedule: Literal["cosine", "constant"] = "cosine",
    settle_weight: float = 0.0,
    eval_every: int = 1,
    seed: int = 0,
    callback: Callable[[int, dict], None] | None = None,
    verbose: bool = False,
) -> FitResult:
    """Train all P candidates with AdamW.

    The best parameters per candidate are tracked on the validation loss if `val` is given (on
    the epochs where it is evaluated, see `eval_every`), otherwise on the training loss.

    Args:
        params: The candidates; trained in place.
        loss_fn: `loss_fn(y, batch) -> (P,)`, one loss per candidate, with `y` of shape
            `(P, D, 1, H, W)`. Their sum is minimised; since the candidates do not share
            parameters, each one follows its own gradient.
        train: Dict of tensors with a leading sample dimension D. Must contain `"u"`
            `(D, 1, H, W)`, optionally `"x0"`; everything else (e.g. `"target"`) is passed
            to `loss_fn`, sliced to the minibatch.
        dyn: The dynamics.
        val: Validation data, like `train`.
        epochs: Number of passes over the training data.
        lr: Learning rate.
        weight_decay: AdamW weight decay.
        batch_size: Minibatch size; None for full batches.
        grad_clip: Clip each candidate's gradient to this norm; None to disable.
        schedule: Learning-rate schedule, `"cosine"` (decay to 0) or `"constant"`.
        settle_weight: Adds `settle_weight × mean |dx/dt|` at the final state to the loss
            (a convergence penalty).
        eval_every: Evaluate the validation loss every this many epochs (and in the last).
        seed: Seed for the minibatch order.
        callback: Called as `callback(epoch, info)` after every epoch, with `info` holding
            `"epoch"`, `"train"`, `"val"` and `"best"` losses.
        verbose: Print progress about ten times per run.

    Returns:
        The final and the best parameters, the best losses and the loss history.
    """
    gen = torch.Generator().manual_seed(seed)
    opt = torch.optim.AdamW([p for p in params.parameters() if p.requires_grad], lr=lr, weight_decay=weight_decay)
    n_batches = math.ceil(train["u"].shape[0] / (batch_size or train["u"].shape[0]))
    total_steps = epochs * n_batches
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, (lambda s: 0.5 * (1 + math.cos(math.pi * s / total_steps))) if schedule == "cosine" else (lambda s: 1.0)
    )
    best = params.select(torch.arange(params.P))
    best_loss = torch.full((params.P,), float("inf"))
    hist: dict[str, list] = {"train": [], "val": []}
    t0 = time.time()

    for ep in range(epochs):
        ep_loss, n_seen = 0.0, 0
        for idx in _batches(train["u"].shape[0], batch_size, gen):
            b = _select(train, idx)
            tpl = params()
            if settle_weight:
                y, x, bias = simulate(tpl, b["u"], dyn, b.get("x0"), return_state=True)
                loss = loss_fn(y, b) + settle_weight * settle_rate(x, bias, tpl[0], dyn)
            else:
                loss = loss_fn(simulate(tpl, b["u"], dyn, b.get("x0")), b)
            opt.zero_grad(set_to_none=True)
            loss.sum().backward()
            if grad_clip:
                for p in params.parameters():
                    if p.grad is not None:  # clip per candidate
                        g = p.grad.reshape(params.P, -1)
                        norm = g.norm(dim=1, keepdim=True).clamp_min(1e-12)
                        g.mul_((grad_clip / norm).clamp(max=1.0))
            opt.step()
            sched.step()
            ep_loss = ep_loss + loss.detach() * len(idx)
            n_seen += len(idx)
        tr = (ep_loss / n_seen).cpu()
        hist["train"].append(tr)

        # The best parameters are tracked on one loss only: val if given (on the epochs where it
        # is evaluated), else train. A train loss is never compared with a best val loss.
        monitor = tr if val is None else None
        if val is not None and (ep % eval_every == 0 or ep == epochs - 1):
            v = evaluate(params, loss_fn, val, dyn, batch_size).cpu()
            hist["val"].append(v)
            monitor = v
        if monitor is not None and (improved := monitor < best_loss).any():
            best_loss = torch.where(improved, monitor, best_loss)
            with torch.no_grad():
                cur = params.to_vector()
                new = best.to_vector()
                new[improved.to(new.device)] = cur[improved.to(cur.device)]
                best.load_vector(new)
        info = {"epoch": ep, "train": tr, "val": hist["val"][-1] if hist["val"] else None, "best": best_loss}
        if callback:
            callback(ep, info)
        if verbose and (ep % max(1, epochs // 10) == 0 or ep == epochs - 1):
            print(f"epoch {ep:4d}  train min {tr.min():.4g} median {tr.median():.4g}  best {best_loss.min():.4g}")

    return FitResult(params, best, best_loss, hist, time.time() - t0)
