# cnn — the `cnnsim` package

Generic, standalone library for Chua–Yang cellular neural networks in PyTorch: GPU simulation and
template learning. It must stay independent of any particular application: nothing thesis-specific
(Lyapunov exponents, time series, EEG) belongs here. Applications consume it as a wheel
(`../thesis` vendors it via `scripts/update-cnnsim.sh`).

- Env: `uv sync`, tests `uv run python -m pytest -q`, benchmark `uv run python benchmark.py`,
  lint `uv run ruff check . && uv run ruff format .` (CI fails on either).
- Conventions: +1 = black, −1 = white; templates are correlations (like torch conv2d); tensors (N,1,H,W).
- Every feature gets tests, including gradient checks for anything differentiable, and keeps the
  fast inference path (compile + CUDA graph) working; check `benchmark.py` for regressions.
- Docs: user guide in `docs/`, API reference from Google-style docstrings (mkdocs-material +
  mkdocstrings, `mkdocs.yml`); `uv run --group docs mkdocs build --strict` must pass. Deployed to
  GitHub Pages from main by `.github/workflows/docs.yml`.
- Public API changes: update docstrings, README.md and `docs/`, add an entry under `[Unreleased]` in CHANGELOG.md
  (Keep a Changelog format). Releases (version bump, tag `vX.Y.Z`, PyPI via the release
  workflow) follow RELEASING.md; work goes through branches and PRs (main is protected).
- Supported: Python >= 3.11, PyTorch >= 2.8 (2.6/2.7 fail to compile the spiral padding on CPU).
