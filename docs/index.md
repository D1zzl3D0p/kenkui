# Kenkui

Kenkui turns EPUB books into M4B audiobooks. Speech is synthesized locally with
[Pocket-TTS](https://github.com/kyutai-labs/pocket-tts), by one narrator or by
a full cast, with a voice for each character that stays with them across a
series.

```python
import kenkui as kk

if __name__ == "__main__":
    kk.load_voice("eponine")
    kk.magic_run("book.epub", narrator="eponine")  # writes book.m4b
```

## Start here

- **[Getting started](getting-started.md)**: install Kenkui, render your first
  book, then add a cast, one step at a time.
- **[Examples](https://github.com/D1zzl3D0p/kenkui/tree/main/examples)**:
  runnable scripts, from inspecting a book to batch-rendering a library.
- **[Guide](usage.md)**: every feature in depth, including casting, series,
  pronunciation, pauses, and the dial-in loop.
- **[API reference](api.md)**: every public name and signature.

## How it behaves

- **Explicit downloads.** `load_voice()` is the only call that downloads
  anything: a language engine (about 225 MB) and the voice (about 6.5 MB).
  Rendering reads only what is already on disk.
- **Network use you choose.** Character discovery can run locally with spaCy.
  Quote attribution calls the [LiteLLM](https://docs.litellm.ai/) model you
  name, and its result is stored, so rendering the same book again makes no
  further calls. Synthesis workers never use the network.
- **Immutable pipelines.** Every method returns a new pipeline, so one book can
  be branched into variants without them affecting each other.
- **Safe output.** A render is published atomically: a failure or cancellation
  never replaces an existing audiobook with a partial one.
- **Stable errors.** Every failure carries a machine-readable `ErrorCode`; see
  [troubleshooting](troubleshooting.md).

## The Kenkui suite

| Project | What it is |
| --- | --- |
| **kenkui** | This library (Apache-2.0). |
| [kenkui-server](https://github.com/D1zzl3D0p/kenkui-server) | A self-hostable API server that runs audiobook jobs on this library (AGPL-3.0-only). |
| [kenkui-studio](https://github.com/D1zzl3D0p/kenkui-studio) | A web and desktop app for the server (AGPL-3.0-only). |
| [kenkui-voices](https://huggingface.co/datasets/D1zzl3D0p/kenkui-voices) | The precompiled voices in the built-in catalog. |

## Reference

- [Installation](installation.md): supported platforms, the spaCy extra, and FFmpeg
- [Models and voices](models-and-voices.md): the catalog, gated models, and voice rights
- [Troubleshooting](troubleshooting.md): what each error code means
- [Architecture](architecture.md): planning, caching, process isolation, and publication
- [Security boundaries](security.md): EPUB limits and local asset verification
- [Pocket-TTS adapter](pocket-tts-adapter.md): how synthesis stays offline
- [Development](development.md): tests, quality gates, and releasing
