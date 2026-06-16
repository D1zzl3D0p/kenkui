# Architecture

`kenkui` is a Python library core for turning ebooks into audiobooks. It also
owns a typed in-process application service API that clients can use as their
stable integration contract. HTTP is an optional adapter over that service, not
the source of core behavior.

## Boundaries

- Library core: parsing, config models, readers, NLP/cache logic, voice catalog
  and assignment, rendering, workers, post-processing, and typed errors.
- Application service: typed queue/task/job/config/book/voice/series/auth
  methods that compose the reusable core behavior for clients.
- HTTP adapter: optional FastAPI transport for non-Python clients and process
  isolation.
- Clients: prompts, screens, notifications, and local workflow policy.
- Extensions: remote NLP/TTS providers may be registered through provider
  boundaries, but cloud control-plane concerns stay outside this package.

## Public Surface

- `kenkui.__init__` re-exports the supported public API.
- `kenkui.api` contains convenience functions such as `load_config`,
  `parse_book`, `fast_scan`, `full_analysis`, `list_voices`, `suggest_cast`,
  `authenticate_huggingface`, and `run_job`.
- `kenkui.models` re-exports dataclasses and enums from the split
  `kenkui.models.*` package.
- `kenkui.services.application_service.KenkuiService` is the canonical
  in-process API for queue/task-style application workflows.
- `kenkui.server` exposes the same service over HTTP when `kenkui[server]` is
  installed.
- `kenkui.errors` defines typed exception categories for new actionable errors.

## Runtime Flow

1. A client gathers user choices and builds library/API models.
2. Book parsing runs through `services.book_service` and `readers`.
3. Optional NLP analysis runs through `services.nlp_service` and
   `nlp.pipeline`.
4. Voice assignment runs through `services.voice_service`.
5. Queued application workflows resolve `JobConfig` to `ProcessingConfig`
   through `services.job_service`.
6. Rendering runs through `AudioBuilder`, worker subprocesses, and
   post-processing.
7. Output artifacts are written to the configured destination.

## State

- Config uses XDG paths and TOML helpers in `config.py`.
- NLP caches use JSON under the configured cache directory.
- App-level saved jobs or queues are managed by `KenkuiService` when callers use
  the built-in application API, and remain serializable through public models.
- Secrets are environment-first. Any local credential support is a convenience
  for local clients.

## Logging

Library modules use Python logging and should not require in-app logging setup.
Clients decide whether logs go to stdout, stderr, files, or a structured sink.

## Compatibility

Serialized config, cache, and job shapes should remain compatible unless a
tested migration is part of the change. Import compatibility should be preserved
with package re-exports where practical.
