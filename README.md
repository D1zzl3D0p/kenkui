# Kenkui

Kenkui is a typed Python toolkit for deterministic, security-bounded EPUB-to-M4B
audiobook production. It exposes an immutable pipeline API for inspection,
validation, selection, metadata, synthesis intent, progress, cancellation, and
atomic publication.

> **Production status:** source inspection, planning, spawned execution, caching,
> and FFmpeg assembly are implemented and tested. The public production renderer
> remains fail-closed until an approved local Pocket-TTS model manifest and an
> authorized voice prompt are supplied and the separate real-inference gate passes.
> No gated model or voice is bundled or downloaded by Kenkui.

## Requirements and support matrix

- CPython 3.11, 3.12, or 3.13 (other versions are outside the declared range).
- Ubuntu 24.04 and macOS 14 are exercised in CI with every supported Python.
- M4B writing requires `ffmpeg` and `ffprobe` 5 or newer on `PATH`, including an
  AAC encoder. These system tools are not bundled.

| Platform | Python 3.11 | Python 3.12 | Python 3.13 |
| --- | --- | --- | --- |
| Ubuntu 24.04 (`ubuntu-24.04`) | CI | CI | CI |
| macOS 14 (`macos-14`) | CI | CI | CI |

The pure-Python wheel is platform-independent, but the table is the exact tier-1
CI matrix, not a claim about untested operating systems or architectures.

## Install

Base installation (EPUB inspection and public API):

```console
python -m pip install kenkui
```

Optional, pinned Pocket-TTS adapter dependencies:

```console
python -m pip install "kenkui[pocket]"
```

The project is not yet published. For a local wheel, use `uv build` and install
`dist/kenkui-0.1.0-py3-none-any.whl`. The `[pocket]` extra installs only the
adapter package; it does not activate production, fetch assets, or grant rights
to a model or voice.

Install FFmpeg separately:

```console
# Ubuntu 24.04
sudo apt-get update && sudo apt-get install -y ffmpeg

# macOS with Homebrew
brew install ffmpeg
```

## Immutable API quickstart

Constructors and fluent methods record intent and return new frozen pipelines;
they do not mutate the original. `validate()` is inexpensive and does not parse
the EPUB. `inspect()` securely parses it and returns frozen metadata/chapters.

```python
from kenkui import CancellationToken, ErrorCode, KenkuiError, epub, load_voice

load_voice("eponine")  # one-time: downloads and hashes the voice

base = epub("book.epub")
job = (
    base.select_chapter_range("chapter-start", "chapter-end")
    .normalize_text()
    .assign_voice("eponine")
    .tts()
    .metadata(title="Example", author="Author", cover="source")
)
assert base.operations == ()  # branching did not mutate base

validation = job.validate()
for issue in validation.issues:
    print(issue.code, issue.message)

inspection = job.inspect()
for chapter in inspection.chapters:
    print(chapter.id, chapter.title, chapter.speech_characters)

token = CancellationToken()
try:
    result = job.write_m4b(
        "book.m4b",
        workers="auto",
        overwrite=False,
        cancel=token,
        on_event=lambda event: print(event),
    )
    print(result.output, result.stats.duration_ms)
except KenkuiError as error:
    if error.code is ErrorCode.RENDERER_UNAVAILABLE:
        print("production assets are not approved/activated")
    else:
        raise
```

`write()` is an alias for `write_m4b()`. Output must end in `.m4b` and its parent
must exist. Existing output is rejected unless `overwrite=True`; publication is
atomic, and cancellation or failure does not publish a candidate. `workers` is a
positive integer or `"auto"` and is conservatively capped at two. Callbacks
receive immutable ordered events (`Started`, stage events, `Warning`, and
`Completed`). Any callback failure before publication commit becomes stable
`callback_failed` and publishes nothing. After commit, publication
`StageCompleted` and terminal `Completed` are best-effort notifications: callback
errors cannot revoke the output. Cancellation is cooperative and raises
`CancelledError` with `cancelled`.

See the [documentation](docs/index.md), especially [usage](docs/usage.md),
[Pocket adapter deployment](docs/pocket-tts-adapter.md), and
[troubleshooting](docs/troubleshooting.md).

## Development and release gates

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

The committed `uv.lock` is the authority for CI/development dependencies. Exact
runtime pins protect reviewed parser/adapter semantics; rationale is documented
in [installation](docs/installation.md). See [CONTRIBUTING.md](CONTRIBUTING.md)
for the complete workflow and DCO requirement.

## License

Kenkui source is licensed under the Apache License, Version 2.0. See
[LICENSE](LICENSE) and [NOTICE](NOTICE). Model, dataset, and voice rights are
separate and must be reviewed for the exact local assets used.
