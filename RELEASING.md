# Releasing cnnsim

Releases are annotated git tags `vX.Y.Z` on `main`. Pushing a tag runs
`.github/workflows/release.yml`: it checks that the tag, `pyproject.toml` and `CHANGELOG.md`
agree, runs the full CI, builds the sdist and wheel, publishes them, and creates a GitHub release
with the changelog section as release notes.

| tag | goes to |
|---|---|
| `v0.5.0` | PyPI (final release) |
| `v0.5.0rc1`, `v0.5.0b1`, `v0.5.0a1` | TestPyPI (pre-release / release candidate) |
| no tag: *Actions → release → Run workflow* | TestPyPI, dry run of the current version (no GitHub release) |

TestPyPI, like PyPI, accepts each version only once, so a dry run uses up that version number
there (not on PyPI).

## Branching

`main` is always releasable; work happens on short-lived branches merged by pull request (CI must
pass). There are no long-lived development or release branches. To fix an old minor version
after `main` has moved on, branch `maint/0.4.x` from the `v0.4.0` tag, fix, bump to 0.4.1 there
and tag on that branch.

## Release checklist

1. On a branch, prepare the release in one commit:
   - `version = "X.Y.Z"` in `pyproject.toml` (semantic versioning; before 1.0 a minor bump may
     break the API, a patch bump only fixes bugs);
   - in `CHANGELOG.md`, move the entries from `[Unreleased]` into `## [X.Y.Z] - YYYY-MM-DD`
     and update the compare links at the bottom;
   - `version` and `date-released` in `CITATION.cff`;
   - `uv lock` (updates the version in `uv.lock`).
2. Run the GPU tests locally (`uv run python -m pytest -q`); CI only covers the CPU paths. For
   changes to the kernels or the simulator, also check `uv run python benchmark.py`.
3. Open a pull request, wait for CI, merge.
4. Tag the merge commit and push the tag:
   ```bash
   git checkout main && git pull
   git tag -a vX.Y.Z -m "cnnsim X.Y.Z"
   git push origin vX.Y.Z
   ```
5. Watch the `release` workflow (`gh run watch`). Then check
   https://pypi.org/project/cnnsim/ and `pip install cnnsim==X.Y.Z` in a fresh environment.

Never move or delete a pushed tag, and never reuse a version number: PyPI does not allow
re-uploading a version. A broken release is *yanked* on PyPI (still installable when pinned
exactly, hidden otherwise) and fixed by a new patch release.

## One-time setup (trusted publishing)

PyPI accepts uploads from this repository's release workflow without any API token, once the
project is registered as a *trusted publisher*:

1. On https://pypi.org (account with 2FA): *Your account → Publishing → Add a new pending
   publisher* → GitHub, with
   - PyPI project name: `cnnsim`
   - Owner: `thorstenkranz`, repository: `cnnsim`
   - Workflow name: `release.yml`
   - Environment name: `pypi`
2. The same on https://test.pypi.org (separate account), environment name `testpypi`.
3. Optional: in the GitHub repository settings, *Environments → pypi → Required reviewers*, so
   that every PyPI upload waits for a click of approval.

The pending publisher becomes a normal one with the first upload, which also creates the
project on PyPI.
