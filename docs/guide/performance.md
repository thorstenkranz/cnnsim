# Performance

Cellular neural networks are cheap per step — a k×k stencil and an elementwise update — but
need hundreds or thousands of steps. Without care, the GPU spends its time launching kernels,
not computing. `cnnsim` has several options against that.

## Simulator options

```python
from cnnsim import CNNSimulator

sim = CNNSimulator(compile=True, cuda_graph=True)  # fastest general GPU mode
sim = CNNSimulator(backend="fused")                # Triton kernels: fastest for many small grids
```

* `backend="stencil"` (default) writes the correlation as k² shifted multiply-adds; with
  `compile=True` the whole Euler step (padding, stencil, output function, update) becomes
  **one fused kernel**. `backend="conv"` uses cuDNN `conv2d` instead, which cannot fuse.
* `cuda_graph=True` records `chunk` steps (default 20) as one CUDA graph, removing almost all
  launch overhead, which dominates on small grids.
* B ⊛ u + z is constant for a static input and computed once.
* `integrator="rk4"` is available; Euler with dt 0.1–0.2 is the classic choice.

`torch.compile` specialises on the shape: the first run of a new grid size compiles for a few
seconds.

## Fused backend (Triton)

`backend="fused"` (in `CNNSimulator` and in `train.Dynamics`) integrates with hand-written
Triton kernels: one program per grid, the state kept in registers for all steps, the boundaries
(including the spiral) as index arithmetic. Training uses a hand-written discrete adjoint — a
backward pass through the Euler steps — instead of autograd. Values and gradients agree with the
stencil path up to float32 rounding.

Limits: NVIDIA GPU, float32, Euler only, odd k ≤ 7, H·W ≤ 16384. Other configurations fall back
to the stencil path automatically. For training,
[`fused.prefer_for_training`][cnnsim.fused.prefer_for_training] decides per boundary and
template size which path is faster.

Training a typical research workload — 16 candidates × 200 series of 64×64, minibatches of 50,
200 Euler steps, spiral boundary, sigmoid output (RTX 5060 Ti):

| | compiled stencil | fused |
|---|---|---|
| 3×3, training epoch | 0.74 s | **0.18 s** (4.1×) |
| 3×3, evaluating 16 × 200 series | 1.44 s | **0.05 s** (29×) |
| 5×5, training epoch | 0.84 s | (stencil used for spiral) |
| 5×5, evaluating 16 × 200 series | 2.53 s | **0.12 s** (21×) |

## Benchmark

Time per Euler step for a single image (`python benchmark.py` in the repository; RTX 5060 Ti,
Ryzen 7 5700X, WSL2, torch 2.14, float32). In brackets: billions of cell updates per second.

| config | 64² | 256² | 1024² | 2048² | 4096² |
|---|---|---|---|---|---|
| cpu-eager-conv | 47 µs | 486 µs | 10.5 ms | 51 ms | – |
| cpu-compiled (8 threads) | 17 µs | 296 µs | 2.5 ms [0.42] | 12.0 ms | – |
| gpu-eager-conv | 158 µs | 162 µs | 134 µs [7.8] | 607 µs | 3.6 ms |
| gpu-compiled | 22 µs | 15 µs | 32 µs [33] | 143 µs | 645 µs |
| gpu-conv-graph | 9 µs | 13 µs | 78 µs [13] | 607 µs | 3.6 ms |
| **gpu-compiled-graph** | **1.6 µs** | **3.1 µs** | **31 µs [34]** | **142 µs [30]** | **645 µs [26]** |

* Compiled + CUDA graph is about 100× faster than eager GPU code at 64², 50× at 256², and 4–6×
  from 1024² on; compared with compiled CPU code it is 10× faster at 64² and 80–95× at
  256²–2048².
* At 4096² it sustains about 300 GB/s of effective memory traffic, close to the bandwidth limit.
* Eager PyTorch on the GPU is launch-bound (about 150 µs per step regardless of size): use
  `compile=True` and `cuda_graph=True`.
