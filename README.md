# cnnsim: Chua–Yang cellular neural networks in PyTorch

```
dx/dt = -x + A ⊛ y + B ⊛ u + z,     y = clip(x, -1, 1)      (black = +1, white = -1)
```

## Setup

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv torch --index-url https://download.pytorch.org/whl/cu130   # RTX 50xx needs CUDA >= 12.8
uv pip install --python .venv numpy matplotlib pillow pytest
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

# per-sample templates (e.g. a parameter sweep): A, B of shape (N,3,3), z of shape (N,)
r = sim.run(None, u, A=A_batch, B=B_batch, z=z_batch, boundary=Boundary("fixed", -1))
```

* `backend="stencil"` (default) writes the 3x3 correlation as 9 shifted multiply-adds; with
  `compile=True` the whole Euler step (pad + stencil + clip + update) becomes **one fused kernel**.
  `backend="conv"` uses cuDNN `conv2d`.
* `cuda_graph=True` records `chunk` (default 20) steps as one CUDA graph, removing launch overhead.
* `B ⊛ u + z` is computed once (constant for a static input).
* `integrator="rk4"` is available; Euler with dt 0.1–0.2 is the classic choice.
* Boundaries: `fixed` (Dirichlet, with value), `zeroflux`, `periodic`.

Files: `cnnsim/simulator.py` (core), `cnnsim/templates.py` (edge, edge_gray, corner, inversion,
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
