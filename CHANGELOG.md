# Changelog

## 0.4.0
Bug fixes; the changes marked (breaking) can affect existing code.
- `fit` with `val` and `eval_every > 1`: the best parameters were tracked on a mix of train and
  val losses (on epochs without validation, the train loss was compared with the best val loss).
  They are now tracked on the val loss only. Results of such runs may change.
- `backend="fused"` with a dtype other than float32 silently computed and returned float32; it now
  falls back to the stencil path (`fused.supported` takes a `dtype`), and direct calls to
  `fused.forward` / `fused.integrate` with non-float32 tensors raise `TypeError`.
- `CNNSimulator.run` / `integrate` take exactly `steps` steps; before, `steps` was rounded up to a
  multiple of `chunk` (the remainder now runs once at the end, uncompiled).
- (breaking) `Template` is frozen and stores A and B as tuples (tensors are accepted and
  converted), so templates can no longer be modified in place or share state (the bundled
  THRESHOLD, CCD_HORIZONTAL and AVERAGE shared one list for B). Use `dataclasses.replace`.
- `Boundary`, `Output` and `TemplateSpec` validate their fields on construction (unknown kinds,
  symmetries, even or non-positive sizes, wrong mask shapes) instead of failing later in a run.
  `TemplateSpec.maps()` no longer caches, so it follows changes to the fields.
- (breaking) Importing cnnsim no longer changes torch's process-wide `torch._dynamo` recompile limits;
  they are raised to 256 only when compilation is used (`compile=True`, via
  `cnnsim.dynamics.compile_steps`).
- MIT license; GitHub Actions workflow running the CPU tests on every push to main.

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
