# Changelog

## 0.3.1
- Fused training policy depends on the boundary (`fused.prefer_for_training`): for k > 3 the
  fused backward is now used with fixed / zeroflux / periodic boundaries (2.4–16× faster than
  the compiled stencil path there); spiral keeps the compiled path.

## 0.3.0
- Fused Triton backend (`cnnsim.fused`, `backend="fused"`): forward kernel with the state on chip
  and a discrete-adjoint backward kernel; 4× faster training (3×3), 20–30× faster evaluation.
  Optional dependency on Triton (bundled with CUDA builds of PyTorch); CPU and unsupported
  configurations fall back to the stencil path.

## 0.2.0
- Generalised model: odd k×k templates (A and B independent), output functions
  (`Output`: pwl, sigmoid(β), tanh(β), sign with surrogate gradient, identity),
  closed-spiral boundary, separate input boundary `boundary_u`, bias maps.
- Differentiable functional core `cnnsim.dynamics`; the simulator is built on it.
- Training: `cnnsim.params` (TemplateParams / TemplateSpec with symmetry groups, masks,
  population dimension), `cnnsim.train` (differentiable `simulate` with checkpointing and
  optional torch.compile, `fit` with multi-start, `evaluate`), `cnnsim.losses`,
  `cnnsim.optim` (simulated annealing, CMA-ES).
- API: `prepare()` returns a `Prepared` dataclass; `integrate(p, ...)` and `stepper(p, dt)`
  take it instead of the (x, bias, A, boundary) tuple.

## 0.1.0
- Chua–Yang simulator (3×3, pwl output), fused stencil kernel, torch.compile, CUDA graphs,
  classic templates, benchmark.
