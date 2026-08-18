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

## Base and Pocket installs

Install only the base parser/public API dependencies:

```console
python -m pip install kenkui
```

Install the optional adapter dependencies when an approved local Pocket
deployment is available:

```console
python -m pip install "kenkui[pocket]"
```

The project is not yet published. Build and install locally instead:

```console
uv build
python -m pip install dist/kenkui-0.1.0-py3-none-any.whl
```

`[pocket]` pins `pocket-tts==2.1.0`, the exact upstream API inspected and adapted.
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
- The optional extra pins `pocket-tts==2.1.0`: the adapter depends on the
  inspected package API and rejects any other installed version.
- Development/docs tools are resolved transitively and locked in `uv.lock`.
  `uv sync --frozen --all-groups` makes that lock authoritative in CI without
  rewriting it.

For a locked contributor environment:

```console
uv lock --check
uv sync --frozen --all-groups
```
