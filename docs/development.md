# Contributing, testing, and release checks

The published documentation is exactly the pages listed in `mkdocs.yml`. The
guide and the generated API reference are the authority on the public
interface. Planning notes and experiments (`docs/superpowers/`, `.superpowers/`,
`spikes/`) are git-ignored and are never versioned or published.

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

The scripts in `examples/` are part of the documentation and are held to the
same gates: `ruff` lints them and strict `mypy` type-checks them against the
public API, so an API change that breaks an example fails CI. Update the
matching example and guide section whenever public behavior changes.

## Opt-in tiers

Two further tiers need local resources and never run in ordinary CI:

- **Real Pocket inference.** `KENKUI_RUN_PROVISIONING_REAL=1 uv run pytest
  --no-cov tests/test_voice_provisioning_real.py` downloads real assets and
  renders a real M4B.
- **Corpus.** `KENKUI_RUN_CORPUS=1 uv run pytest --no-cov -m corpus` runs
  property tests over a local EPUB library, found at `~/Calibre Library` or
  the directory named by `KENKUI_CORPUS_LIBRARY`.

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

## Distribution checks

Build does not need network after the locked environment is present:

```console
uv build
uv run twine check dist/*
uv run check-wheel-contents dist/*.whl
```

Inspect both archives before publishing. The sdist intentionally includes source,
documentation, examples, README, changelog, contribution guide, MkDocs config,
Apache-2.0 `LICENSE`/`NOTICE`, project metadata, and the lock; it excludes
tests. The wheel includes only the package/public code and typing marker
plus required distribution metadata/licenses. Neither archive may contain tests,
virtual environments, temporary/build/site/cache files, credentials, `.env`
files, auth tokens, model/voice assets, or generated evidence.

Install the built wheel (not the source tree) in a fresh compatible Python 3.11,
3.12, or 3.13 environment with dependencies, import only `kenkui` public names,
and exercise constructor/validation/EPUB inspection. An isolated installation
without provisioned model/voice assets must reject production writing with a
stable resource error. Real Pocket inference is a separate opt-in acceptance
check; it is not a runtime approval switch.

CI runs these checks, including the isolated wheel smoke, on every push.
Never add Pocket secrets or assets merely to make CI green.

## Releasing

Versions follow [Semantic Versioning](https://semver.org/), and every
user-visible change gets a line under `## [Unreleased]` in `CHANGELOG.md` in
the same pull request.

1. Move the `Unreleased` entries under a new `## [X.Y.Z] - YYYY-MM-DD`
   heading, and update the comparison links at the bottom of the file.
2. Set `version` in `pyproject.toml` and `__version__` in
   `src/kenkui/__init__.py`, then run `uv lock`. `tests/test_package.py`
   asserts that the two agree.
3. Merge to `main` and wait for CI to pass.
4. Tag the merge commit and push the tag:

   ```console
   git tag -s vX.Y.Z -m "Kenkui X.Y.Z"
   git push origin vX.Y.Z
   ```

The `Release` workflow checks that the tag matches the package version, builds
the sdist and wheel, publishes them to PyPI through trusted publishing (no API
token is stored anywhere), and creates a GitHub release whose notes are that
version's changelog section. The `Docs` workflow republishes the documentation
site on every push to `main`.

## DCO and license

Contributions require a Developer Certificate of Origin 1.1 sign-off:

```console
git commit --signoff
```

The trailer must use your real name and an email address you control. By
contributing, you agree that your contribution is licensed under Apache-2.0.
Model/voice materials are not accepted without separate provenance and rights
review.
