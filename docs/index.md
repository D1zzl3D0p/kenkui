# Kenkui documentation

Kenkui is a typed, immutable EPUB-to-M4B toolkit with deterministic planning,
spawned synthesis isolation, security-bounded input handling, and atomic output
publication.

## Current status

EPUB inspection, semantic planning, process isolation, private caching, FFmpeg
assembly, events, cancellation, and stable errors are implemented and tested.
The ordinary public write path intentionally reports `renderer_unavailable`
until approved local production bindings exist. Pocket-TTS package compatibility
has been implemented without gated assets, but **real Pocket inference is not
approved or verified**. Kenkui does not silently discover/download a model or
voice.

Start here:

- [Installation and exact support matrix](installation.md)
- [Immutable API and execution controls](usage.md)
- [Architecture, determinism, isolation, and cache](architecture.md)
- [EPUB and local-asset security boundaries](security.md)
- [Models, voices, rights, and current evidence](models-and-voices.md)
- [Pocket-TTS adapter and offline activation gate](pocket-tts-adapter.md)
- [Stable errors and troubleshooting](troubleshooting.md)
- [Contributing, testing, and release checks](development.md)
