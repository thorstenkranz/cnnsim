# cnnsim: Chua–Yang cellular neural networks in PyTorch

```
dx/dt = -x + A ⊛ y + B ⊛ u + z,     y = f(x)   (default f = clip(x, -1, 1); black = +1, white = -1)
```

## Setup

```bash
uv sync                      # torch from the CUDA 13 index (RTX 50xx needs CUDA >= 12.8), dev tools
uv run python -m pytest -q
uv build --wheel             # dist/cnnsim-<version>-py3-none-any.whl for use in other projects
```

## Usage

```python
import torch
from cnnsim import CNNSimulator, Template, Boundary, templates as T

sim = CNNSimulator(compile=True, cuda_graph=True)          # fastest GPU mode
r = sim.run(T.HOLE_FILL, u, steps=5000, dt=0.1, tol=1e-3)  # u: (H,W), (N,H,W) or (N,1,H,W) in [-1, 1]
y = r["y"]                                                 # output, plus r["x"], r["steps"], r["history"]

# own template
my = Template(A=[[0,0,0],[0,2,0],[0,0,0]], B=[[0,0,0],[0,1,0],[0,0,0]], z=0.5,
              x0="input", boundary=Boundary("zeroflux"))

# per-sample templates (e.g. a parameter sweep): A, B of shape (N,k,k), z of shape (N,)
r = sim.run(None, u, A=A_batch, B=B_batch, z=z_batch, boundary=Boundary("fixed", -1))

# generalised model: any odd k×k (A and B may differ), output functions, closed-spiral
# boundary, separate input boundary, bias map z of shape (H,W)
from cnnsim import Output
r = sim.run(None, u, A=A5x5, B=B3x3, z=zmap, x0="input",
            boundary=Boundary("spiral"), boundary_u=Boundary("fixed", 0.0),
            output=Output("sigmoid", beta=1.0))
```

### Model options

| | options |
|---|---|
| boundary | `fixed` (Dirichlet, `value`), `zeroflux` (Neumann), `periodic` (torus), `spiral` (closed spiral: row-major ring, neighbour of flat index p at (di,dj) is p + di·W + dj mod HW) |
| output `Output(kind, beta)` | `pwl` clip(x) · `sigmoid` 2/(1+e^(−βx))−1 · `tanh` tanh(βx) · `sign` (surrogate gradient of clip(βx)) · `identity` |
| templates | odd k×k, A and B independent; shared or per sample |
| bias z | scalar, per sample (N,), or map (H,W) / (N,1,H,W) |
| initial state | `zero`, `input`, `ones`, `minus_ones`, or a tensor |

### Differentiable core

`cnnsim.dynamics` (`correlate`, `bias_map`, `derivative`, `make_steps`) consists of pure tensor
functions; gradients flow through the whole integration (checked by `torch.autograd.gradcheck`
in the tests). `CNNSimulator` is the fast no-grad path on top of the same functions.

## Training templates

Templates are learned by backpropagation through the dynamics (exact gradients of the
Euler/RK4 integration, memory bounded by gradient checkpointing). Every template carries a
population dimension P: P independent candidates (multi-start) are trained simultaneously
in one batched simulation of P × D grids.

```python
from cnnsim import Boundary, Output
from cnnsim.params import TemplateParams, TemplateSpec
from cnnsim.train import Dynamics, fit, simulate
from cnnsim.losses import image_loss

dyn = Dynamics(boundary=Boundary("fixed", -1.0), output=Output("tanh"), steps=60, dt=0.2,
               x0="zero", compile=True)
spec = TemplateSpec(size_A=3, size_B=3, sym_A="isotropic", sym_B="none")   # 3 + 9 + 1 free parameters
params = TemplateParams(spec, population=16, device="cuda")
res = fit(params, image_loss("mse"), {"u": u, "target": y_ref}, dyn,
          val={"u": u_val, "target": y_val}, epochs=300, lr=0.05)
A, B, z = res.best()           # (16,3,3), (16,3,3), (16,): best parameters per candidate
```

* `TemplateSpec`: template sizes, symmetry (`none`, `central`, `mirror_lr`, `mirror_ud`,
  `mirror`, `rot90`, `isotropic`), masks, and which of A, B, z are learned. One free parameter
  per orbit of template cells under the symmetry group, so constraints hold exactly.
* `fit`: AdamW, cosine schedule, per-candidate gradient clipping, minibatches, optional
  convergence penalty (`settle_weight` × mean |dx/dt| at the end), best-per-candidate tracking
  on validation data. Any loss `loss_fn(y (P,D,1,H,W), batch) -> (P,)` works.
* `cnnsim.losses`: per-image `mse`, `msen` (SCNN normalised MSE), `mae`, `pixel_error_rate`,
  read-out `mean_output`, aggregations over samples (`mean`, `max`, `rms`, `harmonic`).
* `cnnsim.optim`: gradient-free baselines with batched evaluation — `simulated_annealing`
  (parallel chains) and `cmaes`; `objective(params, loss_fn, data, dyn)` turns a training
  problem into f: (Q, n_free) -> (Q,).

* `backend="stencil"` (default) writes the 3x3 correlation as 9 shifted multiply-adds; with
  `compile=True` the whole Euler step (pad + stencil + clip + update) becomes **one fused kernel**.
  `backend="conv"` uses cuDNN `conv2d`.
* `cuda_graph=True` records `chunk` (default 20) steps as one CUDA graph, removing launch overhead.
* `B ⊛ u + z` is computed once (constant for a static input).
* `integrator="rk4"` is available; Euler with dt 0.1–0.2 is the classic choice.
* Boundaries: `fixed` (Dirichlet, with value), `zeroflux`, `periodic`.

Files: `cnnsim/dynamics.py` (differentiable core), `cnnsim/boundary.py`, `cnnsim/outputs.py`,
`cnnsim/simulator.py` (fast simulator), `cnnsim/templates.py` (edge, edge_gray, corner, inversion,
threshold, hole_fill, shadow_left, ccd_horizontal, average), `demo.py` → `output/templates.png`,
`benchmark.py`, `tests/`.

Run `python -m pytest tests` to check templates against reference results and all modes against each other.

## Benchmark (RTX 5060 Ti, Ryzen 7 5700X, WSL2, torch 2.14, float32)

Time per Euler step, single image (`python benchmark.py`); in brackets: Gcell-updates/s.

| config | 64² | 256² | 1024² | 2048² | 4096² |
|---|---|---|---|---|---|
| cpu-eager-conv | 47 µs | 486 µs | 10.5 ms | 51 ms | – |
| cpu-compiled (8 thr) | 17 µs | 296 µs | 2.5 ms [0.42] | 12.0 ms | – |
| gpu-eager-conv | 158 µs | 162 µs | 134 µs [7.8] | 607 µs | 3.6 ms |
| gpu-compiled | 22 µs | 15 µs | 32 µs [33] | 143 µs | 645 µs |
| gpu-conv-graph | 9 µs | 13 µs | 78 µs [13] | 607 µs | 3.6 ms |
| **gpu-compiled-graph** | **1.6 µs** | **3.1 µs** | **31 µs [34]** | **142 µs [30]** | **645 µs [26]** |

Batch of 64 images, gpu-compiled-graph: 128² → 29 µs/step (36 Gcell/s), 512² → 624 µs (27 Gcell/s).

Takeaways:
* Fused + CUDA graph vs eager GPU (conv): ~100x faster at 64², ~50x at 256², 4–6x at ≥1024².
  vs compiled CPU: ~10x at 64², 80–95x at 256²–2048².
* At 4096² it sustains ~300 GB/s effective (read x, bias; write x), i.e. close to the memory bandwidth limit.
* Eager PyTorch on the GPU is launch-bound (~150 µs/step regardless of size) – always use compile + graph.
* A hand-written multithreaded Rust CPU version would be bounded by ~50 GB/s DRAM bandwidth,
  so at best a few Gcell/s – still well below the GPU path.

Note: `torch.compile` specialises on shape; the first run of a new size compiles for a few seconds.
