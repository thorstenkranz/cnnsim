# cnn — the `cnnsim` package

Generic, standalone library for Chua–Yang cellular neural networks in PyTorch: GPU simulation and
template learning. It must stay independent of any particular application: nothing thesis-specific
(Lyapunov exponents, time series, EEG) belongs here. Applications consume it as a wheel
(`../thesis` vendors it via `scripts/update-cnnsim.sh`).

- Env: `uv sync`, tests `uv run python -m pytest -q`, benchmark `uv run python benchmark.py`.
- Conventions: +1 = black, −1 = white; templates are correlations (like torch conv2d); tensors (N,1,H,W).
- Every feature gets tests, including gradient checks for anything differentiable, and keeps the
  fast inference path (compile + CUDA graph) working; check `benchmark.py` for regressions.
- Public API changes: update README.md, bump `version` in pyproject.toml (semver), note in CHANGELOG.md.
