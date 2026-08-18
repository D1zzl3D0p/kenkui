# Contributing to Kenkui

Thank you for contributing.

## Development setup

Kenkui supports CPython 3.11 through 3.13 and uses uv for reproducible
environments:

```console
uv lock --check
uv sync --frozen --all-groups
uv run ruff format --check .
uv run ruff check .
uv run mypy
uv run pytest
uv run mkdocs build --strict
KENKUI_RUN_NATIVE=1 uv run pytest --no-cov -m native tests/test_native_ffmpeg.py
uv build
uv run twine check dist/*
uv run check-wheel-contents dist/*.whl
```

Add or update tests before implementation changes. Keep the public API typed,
and update documentation when behavior changes. Default tests are deterministic,
offline, branch-covered at 90% or higher, and exclude the explicitly opted-in
native marker. Native FFmpeg acceptance uses generated fake audio and is not real
Pocket inference. Gated model/voice assets, credentials, and real Pocket tests
must not be added to ordinary CI.

See [the development and release guide](docs/development.md) for archive contents,
isolated wheel smoke, strict documentation, and release-candidate checks. The
committed `uv.lock` is authoritative; do not refresh it in an unrelated change.

## Developer Certificate of Origin

Contributions require certification under the
[Developer Certificate of Origin, Version 1.1](https://developercertificate.org/).
By adding a `Signed-off-by` trailer, you certify that you have the right to
submit the contribution under this project's license. Sign each commit with:

```console
git commit --signoff
```

The trailer must use your real name and an email address you control:

```text
Signed-off-by: Your Name <your.email@example.com>
```

## License

By contributing, you agree that your contribution is licensed under the
Apache License, Version 2.0.
