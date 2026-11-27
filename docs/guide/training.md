# Training templates

Templates can be learned from examples by backpropagation through the dynamics: gradients of the
loss with respect to A, B and z are exact gradients of the Euler (or RK4) integration, and
gradient checkpointing keeps the memory bounded for long runs. Every learnable template carries a
**population** dimension P: P independent candidates (a multi-start) are trained simultaneously,
in one batched simulation of P × D grids for D training samples.

## Example: learning an edge detector

```python
import torch
from cnnsim import Boundary, CNNSimulator, Output, templates as T
from cnnsim.losses import image_loss
from cnnsim.params import TemplateParams, TemplateSpec
from cnnsim.train import Dynamics, fit

# training data: random blobs and the output of the EDGE template
g = torch.Generator().manual_seed(0)
u = torch.where(torch.rand(64, 1, 16, 16, generator=g) > 0.6, 1.0, -1.0)
target = CNNSimulator(device="cpu").run(T.EDGE, u, steps=200)["y"]

dyn = Dynamics(boundary=Boundary("fixed", -1.0), output=Output("pwl"), steps=40, dt=0.2, x0="zero")
spec = TemplateSpec(sym_A="isotropic", sym_B="isotropic")  # 3 + 3 + 1 free parameters
params = TemplateParams(spec, population=8, generator=g)
res = fit(params, image_loss("mse"), {"u": u, "target": target}, dyn, epochs=150, lr=0.05)

A, B, z = res.best()   # (8,3,3), (8,3,3), (8,): best parameters per candidate
i = res.best_loss.argmin()
print(res.best_loss[i].item(), A[i], B[i], z[i])
```

## The pieces

* [`Dynamics`][cnnsim.train.Dynamics]: everything about a run except the templates and the
  data — boundary, output function, number of steps, dt, integrator, initial state,
  checkpointing, `compile` and the backend.
* [`TemplateSpec`][cnnsim.params.TemplateSpec]: template sizes, symmetry (`none`, `central`,
  `mirror_lr`, `mirror_ud`, `mirror`, `rot90`, `isotropic`), masks, and which of A, B and z are
  learned. There is one free parameter per orbit of template cells under the symmetry group, so
  the constraints hold exactly during training.
* [`TemplateParams`][cnnsim.params.TemplateParams]: the learnable parameters of P candidates
  (an `nn.Module`); calling it returns `(A, B, z)` with shapes `(P, k, k)`, `(P, k, k)`, `(P,)`.
* [`simulate`][cnnsim.train.simulate]: the differentiable simulation of P candidates on D
  samples, returning outputs of shape `(P, D, 1, H, W)`.
* [`fit`][cnnsim.train.fit]: AdamW with a cosine schedule, per-candidate gradient clipping,
  minibatches, an optional convergence penalty (`settle_weight` × mean |dx/dt| at the end),
  and best-per-candidate tracking on validation data (or on the training loss without it).

## Losses

Any function `loss_fn(y, batch) -> (P,)` works, where `y` has shape `(P, D, 1, H, W)` and
`batch` is the dict of tensors for the current minibatch. [`cnnsim.losses`][cnnsim.losses] has
building blocks: per-image `mse`, `msen` (the normalised MSE of SCNN, in [0, 1]), `mae`,
`pixel_error_rate`, the read-out `mean_output`, and aggregations over the samples
(`mean`, `max`, `rms`, `harmonic`). `image_loss("mse", "mean")` combines them into a loss that
compares `y` with `batch["target"]`.

## Gradient-free baselines

[`cnnsim.optim`][cnnsim.optim] has simulated annealing (parallel chains) and CMA-ES. Both take
an objective f: `(Q, n) -> (Q,)` that evaluates a whole population in one batched simulation;
`objective(params, loss_fn, data, dyn)` turns a training problem into one:

```python
from cnnsim.losses import aggregate, msen
from cnnsim.optim import cmaes, objective

p = TemplateParams(spec)
f = objective(p, lambda y, b: aggregate(msen(y, b["target"])), {"u": u, "target": target}, dyn)
best = cmaes(f, torch.zeros(p.n_free), sigma0=1.0, iters=100, seed=0)
A, B, z = p.templates_from_vector(best.best_x[None])
```
