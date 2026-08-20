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

Install only the base parser/public API dependencies:

```console
python -m pip install kenkui
```

The adapter dependencies are already included. This extra is a no-op, kept
for compatibility with installs that requested it when Pocket
deployment is available:

```console
python -m pip install "kenkui[pocket]"
```

The project is not yet published. Build and install locally instead:

```console
uv build
python -m pip install dist/kenkui-0.1.0-py3-none-any.whl
```

`pocket-tts==2.1.0` is pinned exactly: the adapter depends on the inspected
upstream API and rejects any other installed version.
It does **not** include model weights, a voice, accepted upstream terms, or an
online downloader, and it does not make the fail-closed public renderer active.
See [Pocket-TTS adapter](pocket-tts-adapter.md).

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
