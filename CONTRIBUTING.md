# Contributing to cnnsim

Thanks for your interest! Bug reports, questions, ideas and pull requests are all welcome.

## Questions, bugs and ideas

- **Bug:** open an [issue](https://github.com/thorstenkranz/cnnsim/issues/new/choose) with a
  minimal example, what you expected, and the output of
  `python -c "import cnnsim, torch, sys; print(cnnsim.__version__, torch.__version__, torch.version.cuda, sys.version)"`.
- **Feature or larger change:** please open an issue first to discuss it, so that no work is
  wasted on something that doesn't fit the library.
- **Security problem:** don't open a public issue; see [SECURITY.md](SECURITY.md).

## Scope

cnnsim is a generic library for Chua–Yang cellular neural networks: simulation, the model's
variants (templates, boundaries, output functions), and template learning. Application-specific
code (a particular data set, analysis or experiment) belongs in the project that uses cnnsim.

## Development setup

You need [uv](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/thorstenkranz/cnnsim && cd cnnsim
uv sync                      # PyTorch from the CUDA 13 index (Linux / Windows, NVIDIA GPU)
uv run python -m pytest -q
```

Without an NVIDIA GPU (or on macOS), install a CPU build of PyTorch instead, as CI does:

```bash
uv venv
uv pip install torch --index-url https://download.pytorch.org/whl/cpu
uv pip install -e . pytest numpy matplotlib pillow
uv run --no-sync python -m pytest -q   # the CUDA/Triton tests are skipped
```

Tests marked `slow` can be skipped with `-m "not slow"`.

Code style is enforced by [ruff](https://docs.astral.sh/ruff/) (configured in `pyproject.toml`):

```bash
uv run ruff check --fix .
uv run ruff format .
```

Optionally, run both on every commit with [pre-commit](https://pre-commit.com/):
`uvx pre-commit install`.

## Pull requests

1. Fork the repository and create a branch from `main`.
2. Make your change, with tests:
   - every feature or fix gets a test; anything differentiable also gets a gradient check
     (`torch.autograd.gradcheck` in float64);
   - keep the fast inference path working (`compile=True`, `cuda_graph=True`, the fused
     backend); if you touch the simulator or the kernels and have a GPU, compare
     `uv run python benchmark.py` before and after.
3. If the public API or behaviour changes, update `README.md` and add an entry under
   `[Unreleased]` in [CHANGELOG.md](CHANGELOG.md) (Added / Changed / Fixed / Removed).
4. Open a pull request against `main`. CI runs ruff and the tests on CPU for Python 3.11–3.14
   (oldest and newest supported PyTorch) and checks the package build; it must pass before
   merging.

GitHub's CI runners have no GPU, so the CUDA and Triton tests only run on a maintainer's
machine. If your change affects GPU code and you couldn't test it, say so in the pull request.

## Conventions

- Tensors are `(N, 1, H, W)`; templates are correlations (like `torch.nn.functional.conv2d`),
  odd `k×k`; +1 is black, −1 is white.
- Supported versions: Python ≥ 3.11, PyTorch ≥ 2.8.
- Versioning and releases: see [RELEASING.md](RELEASING.md).

By contributing you agree that your contribution is licensed under the [MIT License](LICENSE).
