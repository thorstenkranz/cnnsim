"""Benchmark CNN simulation speed across devices, backends and grid sizes.

Measures time per Euler step of the full CNN dynamics
(dx/dt = -x + A*y + bias, y = clip(x)) with a random dense 3x3 A template.

    python benchmark.py                       # default sweep
    python benchmark.py --sizes 256 1024 --batch 16
    python benchmark.py --configs gpu-compiled-graph cpu-compiled
"""

from __future__ import annotations

import argparse
import time
import warnings

import torch

from cnnsim import Boundary, CNNSimulator

warnings.filterwarnings("ignore")

CONFIGS = {
    # name: (device, simulator kwargs)
    "cpu-eager-conv": ("cpu", dict(backend="conv")),
    "cpu-eager-stencil": ("cpu", dict(backend="stencil")),
    "cpu-compiled": ("cpu", dict(backend="stencil", compile=True)),
    "gpu-eager-conv": ("cuda", dict(backend="conv")),
    "gpu-eager-stencil": ("cuda", dict(backend="stencil")),
    "gpu-compiled": ("cuda", dict(backend="stencil", compile=True)),
    "gpu-conv-graph": ("cuda", dict(backend="conv", cuda_graph=True)),
    "gpu-compiled-graph": ("cuda", dict(backend="stencil", compile=True, cuda_graph=True)),
}


def sync(device):
    if device == "cuda":
        torch.cuda.synchronize()


def bench_one(cfg: str, size: int, batch: int, min_time: float, chunk: int) -> float:
    """Return seconds per step."""
    device, kw = CONFIGS[cfg]
    sim = CNNSimulator(device=device, chunk=chunk, **kw)
    g = torch.Generator().manual_seed(0)
    u = torch.rand(batch, size, size, generator=g) * 2 - 1
    A = torch.randn(3, 3, generator=g) * 0.3
    B = torch.randn(3, 3, generator=g) * 0.3
    x, bias, A, boundary = sim.prepare(None, u, A=A, B=B, z=0.0, boundary=Boundary("zeroflux"))

    advance = sim.stepper(x, bias, A, boundary, dt=0.1)
    for _ in range(3):  # warm-up: compilation, autotune, cuDNN algorithm selection
        advance()
    sync(device)

    n = 1
    while True:
        t0 = time.perf_counter()
        for _ in range(n):
            advance()
        sync(device)
        dt = time.perf_counter() - t0
        if dt >= min_time:
            return dt / (n * chunk)
        n = max(n * 2, int(n * min_time / max(dt, 1e-6) * 1.2))


def fmt_time(s: float) -> str:
    if s < 1e-3:
        return f"{s * 1e6:7.1f} µs"
    return f"{s * 1e3:7.2f} ms"


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--sizes", type=int, nargs="+", default=[64, 128, 256, 512, 1024, 2048, 4096])
    p.add_argument("--batch", type=int, default=1)
    p.add_argument("--configs", nargs="+", default=list(CONFIGS), choices=list(CONFIGS))
    p.add_argument("--chunk", type=int, default=20, help="steps per compiled call / graph replay")
    p.add_argument("--min-time", type=float, default=0.5, help="seconds of measurement per cell")
    p.add_argument("--threads", type=int, default=None, help="CPU threads (default: torch default)")
    p.add_argument("--max-cpu-size", type=int, default=2048, help="skip larger sizes on CPU")
    args = p.parse_args()

    if args.threads:
        torch.set_num_threads(args.threads)
    configs = [c for c in args.configs if CONFIGS[c][0] != "cuda" or torch.cuda.is_available()]

    gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none"
    print(f"torch {torch.__version__} | GPU: {gpu} | CPU threads: {torch.get_num_threads()} "
          f"| batch {args.batch} | chunk {args.chunk}")
    print("time per Euler step (whole batch); in brackets: throughput in Gcell-updates/s\n")

    w = 22
    print(f"{'config':<20}" + "".join(f"{f'{s}x{s}':>{w}}" for s in args.sizes))
    for cfg in configs:
        print(f"  running {cfg} ...", end="\r", flush=True)
        row = f"{cfg:<20}"
        for s in args.sizes:
            if CONFIGS[cfg][0] == "cpu" and s > args.max_cpu_size:
                row += f"{'-':>{w}}"
            else:
                try:
                    t = bench_one(cfg, s, args.batch, args.min_time, args.chunk)
                    rate = args.batch * s * s / t / 1e9
                    row += f"{fmt_time(t) + f' [{rate:6.2f}]':>{w}}"
                except torch.OutOfMemoryError:
                    row += f"{'OOM':>{w}}"
                    torch.cuda.empty_cache()
        print(row)


if __name__ == "__main__":
    main()
