"""Gradient-free template optimisers (baselines), with batched evaluation.

Both take an objective f: (Q, n) parameter vectors -> (Q,) losses, evaluated for the whole
population in one batched simulation (see `objective`), so a GPU evaluates many candidates
at the cost of roughly one.

- simulated_annealing: P independent chains with Gaussian proposals whose width shrinks with
  the temperature, Metropolis acceptance ("iterative annealing" as in SCNN).
- cmaes: Covariance Matrix Adaptation Evolution Strategy (Hansen), (mu/mu_w, lambda) with
  rank-one and rank-mu updates and cumulative step-size adaptation.
"""

from __future__ import annotations

import math
import time
from collections.abc import Callable
from dataclasses import dataclass, field

import torch

from .params import TemplateParams
from .train import Dynamics, LossFn, simulate

Objective = Callable[[torch.Tensor], torch.Tensor]


def objective(
    params: TemplateParams, loss_fn: LossFn, data: dict, dyn: Dynamics, chunk: int | None = None
) -> Objective:
    """f(v) = loss of the templates encoded by vectors v (Q, n_free), evaluated without gradients."""

    @torch.no_grad()
    def f(v: torch.Tensor) -> torch.Tensor:
        out = []
        for i in range(0, v.shape[0], chunk or v.shape[0]):
            tpl = params.templates_from_vector(v[i : i + (chunk or v.shape[0])])
            out.append(loss_fn(simulate(tpl, data["u"], dyn, data.get("x0")), data))
        return torch.cat(out)

    return f


@dataclass
class OptResult:
    best_x: torch.Tensor  # (n,) overall best, or (P, n) per chain for SA
    best_f: torch.Tensor
    history: list = field(default_factory=list)  # best loss after each iteration
    evaluations: int = 0
    seconds: float = 0.0


def simulated_annealing(
    f: Objective,
    x0: torch.Tensor,
    iters: int = 1000,
    step0: float = 0.5,
    step_min: float = 1e-3,
    T0: float | None = None,
    seed: int = 0,
) -> OptResult:
    """Parallel simulated annealing from x0 (P, n): P independent chains.

    Temperature and proposal width decay geometrically from (T0, step0) to (T0 * step_min/step0,
    step_min) over `iters` iterations. T0 defaults to the median initial loss.
    """
    gen = torch.Generator(device=x0.device).manual_seed(seed)
    x = x0.clone()
    fx = f(x)
    T0 = float(fx.median()) if T0 is None else T0
    best_x, best_f = x.clone(), fx.clone()
    decay = (step_min / step0) ** (1.0 / max(1, iters - 1))
    hist, t0 = [], time.time()
    for it in range(iters):
        step, T = step0 * decay**it, T0 * decay**it
        y = x + step * torch.randn(x.shape, generator=gen, device=x.device, dtype=x.dtype)
        fy = f(y)
        accept = (fy < fx) | (
            torch.rand(fx.shape, generator=gen, device=x.device) < torch.exp(-(fy - fx) / max(T, 1e-300))
        )
        x = torch.where(accept[:, None], y, x)
        fx = torch.where(accept, fy, fx)
        better = fx < best_f
        best_x = torch.where(better[:, None], x, best_x)
        best_f = torch.where(better, fx, best_f)
        hist.append(best_f.min().item())
    return OptResult(best_x, best_f, hist, (iters + 1) * x0.shape[0], time.time() - t0)


def cmaes(
    f: Objective,
    mean0: torch.Tensor,
    sigma0: float = 0.5,
    iters: int = 200,
    popsize: int | None = None,
    seed: int = 0,
    tol_f: float = 0.0,
) -> OptResult:
    """Minimise f with CMA-ES starting from mean0 (n,)."""
    dev, dt = mean0.device, torch.float64
    n = mean0.numel()
    lam = popsize or 4 + int(3 * math.log(n))
    mu = lam // 2
    w = torch.log(torch.tensor(mu + 0.5, dtype=dt)) - torch.log(torch.arange(1, mu + 1, dtype=dt))
    w = (w / w.sum()).to(dev)
    mueff = 1.0 / (w**2).sum().item()
    cc = (4 + mueff / n) / (n + 4 + 2 * mueff / n)
    cs = (mueff + 2) / (n + mueff + 5)
    c1 = 2 / ((n + 1.3) ** 2 + mueff)
    cmu = min(1 - c1, 2 * (mueff - 2 + 1 / mueff) / ((n + 2) ** 2 + mueff))
    damps = 1 + 2 * max(0.0, math.sqrt((mueff - 1) / (n + 1)) - 1) + cs
    chiN = math.sqrt(n) * (1 - 1 / (4 * n) + 1 / (21 * n * n))

    gen = torch.Generator(device=dev).manual_seed(seed)
    m = mean0.to(dt).clone()
    sigma = sigma0
    C = torch.eye(n, dtype=dt, device=dev)
    pc = torch.zeros(n, dtype=dt, device=dev)
    ps = torch.zeros(n, dtype=dt, device=dev)
    best_x, best_f = m.clone(), torch.tensor(float("inf"), dtype=dt)
    hist, evals, t0 = [], 0, time.time()
    for g in range(iters):
        evals_, Bm = torch.linalg.eigh(C)
        D = evals_.clamp_min(1e-20).sqrt()
        z = torch.randn(lam, n, generator=gen, device=dev, dtype=dt)
        y = (z * D) @ Bm.T  # ~ N(0, C)
        X = m + sigma * y
        fx = f(X.to(mean0.dtype)).to(dt).cpu()
        evals += lam
        order = fx.argsort()
        if fx[order[0]] < best_f:
            best_f, best_x = fx[order[0]].clone(), X[order[0]].clone()
        hist.append(best_f.item())
        if best_f.item() <= tol_f:
            break
        ysel = y[order[:mu].to(dev)]
        yw = (w[:, None] * ysel).sum(0)
        m = m + sigma * yw
        invsqrtC = Bm @ torch.diag(1 / D) @ Bm.T
        ps = (1 - cs) * ps + math.sqrt(cs * (2 - cs) * mueff) * (invsqrtC @ yw)
        hsig = float(ps.norm() / math.sqrt(1 - (1 - cs) ** (2 * (g + 1))) / chiN < 1.4 + 2 / (n + 1))
        pc = (1 - cc) * pc + hsig * math.sqrt(cc * (2 - cc) * mueff) * yw
        C = (1 - c1 - cmu) * C + c1 * (torch.outer(pc, pc) + (1 - hsig) * cc * (2 - cc) * C) + cmu * (ysel.T * w) @ ysel
        C = 0.5 * (C + C.T)
        sigma *= math.exp((cs / damps) * (ps.norm().item() / chiN - 1))
    return OptResult(best_x.to(mean0.dtype), best_f.to(mean0.dtype), hist, evals, time.time() - t0)
