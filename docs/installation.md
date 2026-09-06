# Installation

## Supported matrix

Kenkui requires CPython `>=3.11,<3.14`. The complete tier-1 CI matrix is:

| GitHub runner platform | 3.11 | 3.12 | 3.13 |
| --- | --- | --- | --- |
| Ubuntu 24.04 (`ubuntu-24.04`) | tested | tested | tested |
| macOS 14 (`macos-14`) | tested | tested | tested |

The distribution is a `py3-none-any` wheel. That does not make unlisted host
platforms supported: native M4B behavior also depends on the host FFmpeg build.
The runner labels above deliberately state the platform exactly; they do not
promise every Linux distribution, macOS release, CPU architecture, PyPy, or
Windows.

## Install size

`pocket-tts==2.1.0` is a **required** dependency, and it pulls `torch`,
`scipy`, `sentencepiece`, `numpy`, and `pydantic`. Expect a torch-scale
install of well over a gigabyte, not the few hundred kilobytes a pure-Python
EPUB library would suggest.

Provisioning downloads more on top of that, on first use only: roughly 225 MB
per language engine plus roughly 6.5 MB per voice. See
[Usage](usage.md#provisioning-voices).

The `[pocket]` extra is retained as an empty alias so existing
`kenkui[pocket]` installs keep resolving. It no longer adds anything.

## Base and Pocket installs

Install the library, including its required synthesis dependencies:

```console
python -m pip install kenkui
```

The adapter dependencies are already included. This extra is a no-op, kept
for compatibility with older installs that requested Pocket separately:

```console
python -m pip install "kenkui[pocket]"
```

To build and install from this checkout:

```console
uv build
python -m pip install dist/kenkui-0.1.0-py3-none-any.whl
```

`pocket-tts==2.1.0` is pinned exactly: the adapter depends on the inspected
upstream API and rejects any other installed version.
The wheel does not contain model weights or voice recordings. Explicit
`load_voice()` provisioning downloads the necessary assets and records them in
the local manifest; rendering verifies that manifest before synthesis.
See [Pocket-TTS adapter](pocket-tts-adapter.md).

## Optional offline character discovery with spaCy

The base install uses your configured LiteLLM model for character discovery.
To discover characters locally instead, install the optional spaCy dependency
and its English pipeline into the same Python environment:

```console
python -m pip install "kenkui[spacy]"
python -m spacy download en_core_web_lg
```

`infer_characters("spacy")` selects `en_core_web_lg`. The download is explicit;
Kenkui loads the installed pipeline and does not fetch it during resolution.
To choose a smaller pipeline, install `en_core_web_sm` and use
`infer_characters("spacy:en_core_web_sm")`. Different pipeline sizes can change
the discovered roster. Kenkui uses dependency parsing and English speech/name
heuristics, so this path is intended for English books.

```python
import kenkui as kk

if __name__ == "__main__":
    review = kk.epub("book.epub").infer_characters("spacy").resolve(until="characters")
    print(review.inspect().roster)
```

This replaces character discovery only. `attribute_quotes(model)` still uses
a LiteLLM provider/model identifier and may make network requests; `"spacy"`
is not a quote-attribution provider. See
[character review](usage.md#review-characters-before-attribution) to correct the
roster before continuing. Missing dependencies raise `spacy_package_missing`
or `spacy_pipeline_missing`; install the missing component in the environment
running Kenkui.

## FFmpeg prerequisite

M4B production requires both `ffmpeg` and `ffprobe` version 5 or newer on
`PATH`, with MP4/M4B muxing, AAC encoding, probing, and full decoding available.
They are external executables and are never bundled in the wheel.

```console
# Ubuntu 24.04
sudo apt-get update
sudo apt-get install -y ffmpeg

# macOS with Homebrew
brew install ffmpeg

ffmpeg -version
ffprobe -version
ffmpeg -hide_banner -encoders
```

Kenkui performs a capability preflight rather than trusting a version string.
Missing executables report `ffmpeg_not_found` or `ffprobe_not_found`; incomplete
builds report `ffmpeg_unsupported` or `ffprobe_unsupported`. Encoding, probing,
decoding, semantic validation, and publication have distinct stable errors. No
candidate is published after a failed check.

## Why dependencies are pinned

- Base runtime pins `defusedxml==0.7.1`: XML hardening is part of the reviewed
  EPUB parser boundary, so silent behavior drift is avoided.
- `pocket-tts==2.1.0` is pinned exactly: the adapter depends on the inspected
  package API and rejects any other installed version.
- Development/docs tools are resolved transitively and locked in `uv.lock`.
  `uv sync --frozen --all-groups` makes that lock authoritative in CI without
  rewriting it.

For a locked contributor environment:

```console
uv lock --check
uv sync --frozen --all-groups
```
