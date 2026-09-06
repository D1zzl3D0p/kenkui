# Kenkui documentation

Kenkui is a typed, immutable EPUB-to-M4B toolkit with deterministic planning,
spawned synthesis isolation, security-bounded input handling, and atomic output
publication.

## Current status

EPUB inspection, semantic planning, process isolation, private caching, FFmpeg
assembly, events, cancellation, and stable errors are implemented and tested.
Writing requires one explicit provisioning call. `kk.load_voice("eponine")`
downloads and hashes a voice from the built-in catalog; after that the ordinary
public write path works. Without it, `write()` reports `voice_not_provisioned`
or `renderer_unavailable`. Kenkui never silently downloads a model or voice
during rendering. Character analysis in `resolve()` or `write()` may make
LiteLLM requests; synthesis workers themselves remain offline.

**Real Pocket inference has not been verified in CI.** The end-to-end test that
downloads real assets and renders a real M4B is opt-in. See
[Models and voices](models-and-voices.md).

Start here:

- [Installation and exact support matrix](installation.md)
- [Immutable API and execution controls](usage.md)
- [Architecture, determinism, isolation, and cache](architecture.md)
- [EPUB and local-asset security boundaries](security.md)
- [Models, voices, rights, and current evidence](models-and-voices.md)
- [Pocket-TTS adapter and offline activation gate](pocket-tts-adapter.md)
- [Stable errors and troubleshooting](troubleshooting.md)
- [Contributing, testing, and release checks](development.md)
