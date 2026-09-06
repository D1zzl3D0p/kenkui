# Contributing, testing, and release checks

## Locked setup

Kenkui supports CPython 3.11 through 3.13. Install
[uv](https://docs.astral.sh/uv/), then use the committed lock as authority:

```console
uv lock --check
uv sync --frozen --all-groups
```

Do not refresh `uv.lock` as a side effect of an unrelated change. When a
dependency update is intentional, explain the pin/lock change and rerun the full
matrix-relevant gates.

## Deterministic default gates

```console
uv run ruff format --check .
uv run ruff check .
uv run mypy
uv run pytest
uv run mkdocs build --strict
```

Mypy is strict. Pytest enables branch coverage and fails below 90%; default test
configuration excludes the `native` marker. Tests must not need network,
credentials, user caches, model assets, or gated terms. Add/update deterministic
tests with implementation changes and keep imports from the documented public
facade in public API examples/smoke tests.

## Native FFmpeg tier

Install/check host `ffmpeg` and `ffprobe` first, then opt in explicitly:

```console
KENKUI_RUN_NATIVE=1 uv run pytest --no-cov -m native tests/test_native_ffmpeg.py
```

This runs generated fake PCM through the production FFmpeg shell and
independently probes/full-decodes the resulting M4B. It is native FFmpeg
acceptance, not real Pocket inference.

## Documentation

```console
uv run mkdocs build --strict
```

Strict mode must be warning-free. A theme package may print its own upstream
informational notice; do not suppress project warnings to hide broken links/nav.

## Distribution and release candidate

Start from a clean checkout when preparing a release candidate. Build does not
need network after the locked environment is present:

```console
uv build
uv run twine check dist/*
uv run check-wheel-contents dist/*.whl
```

Inspect both archives before publishing. The sdist intentionally includes source,
documentation, README, contribution guide, MkDocs config, Apache-2.0
`LICENSE`/`NOTICE`, project metadata, and the lock; it excludes tests and local
spikes/evidence. The wheel includes only the package/public code and typing marker
plus required distribution metadata/licenses. Neither archive may contain tests,
virtual environments, temporary/build/site/cache files, credentials, `.env`
files, auth tokens, model/voice assets, or generated evidence.

Install the built wheel (not the source tree) in a fresh compatible Python 3.11,
3.12, or 3.13 environment with dependencies, import only `kenkui` public names,
and exercise constructor/validation/EPUB inspection. An isolated installation
without provisioned model/voice assets must reject production writing with a
stable resource error. Real Pocket inference is a separate opt-in acceptance
check; it is not a runtime approval switch.

Publishing itself is intentionally not automated by the CI workflow. Before a
release, confirm version/changelog policy, exact archive listing, all six
OS/Python CI cells, both native OS jobs, metadata checks, and explicit approval
state. Never add Pocket secrets/assets merely to make CI green.

## DCO and license

Contributions require a Developer Certificate of Origin 1.1 sign-off:

```console
git commit --signoff
```

The trailer must use your real name and an email address you control. By
contributing, you agree that your contribution is licensed under Apache-2.0.
Model/voice materials are not accepted without separate provenance and rights
review.
