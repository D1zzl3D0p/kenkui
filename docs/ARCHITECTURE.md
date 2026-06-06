# Architecture

`kenkui` is a Python library core for turning ebooks into audiobooks. It is not
the CLI, server, or frontend. Those clients should call the public API and
service layer in this package.

## Boundaries

- Library core: parsing, config models, readers, NLP/cache logic, voice catalog
  and assignment, rendering, workers, post-processing, and typed errors.
- Clients: prompts, screens, HTTP routes, queues, notifications, deployment
  policy, and remote execution decisions.
- Extensions: remote NLP/TTS providers may be registered through provider
  boundaries, but cloud control-plane concerns stay outside this package.

## Public Surface

- `kenkui.__init__` re-exports the supported public API.
- `kenkui.api` contains convenience functions such as `load_config`,
  `parse_book`, `fast_scan`, `full_analysis`, `list_voices`, `suggest_cast`,
  `authenticate_huggingface`, and `run_job`.
- `kenkui.models` re-exports dataclasses and enums from the split
  `kenkui.models.*` package.
- `kenkui.errors` defines typed exception categories for new actionable errors.

## Runtime Flow

1. A client gathers user choices and builds library models.
2. Book parsing runs through `services.book_service` and `readers`.
3. Optional NLP analysis runs through `services.nlp_service` and
   `nlp.pipeline`.
4. Voice assignment runs through `services.voice_service`.
5. Rendering runs through `AudioBuilder`, worker subprocesses, and
   post-processing.
6. Output artifacts are written to the configured destination.

## State

- Config uses XDG paths and TOML helpers in `config.py`.
- NLP caches use JSON under the configured cache directory.
- App-level saved jobs or queues are client-owned unless represented only as
  serializable library models.
- Secrets are environment-first. Any local credential support is a convenience
  for local clients.

## Logging

Library modules use Python logging and should not require in-app logging setup.
Clients decide whether logs go to stdout, stderr, files, or a structured sink.

## Compatibility

Serialized config, cache, and job shapes should remain compatible unless a
tested migration is part of the change. Import compatibility should be preserved
with package re-exports where practical.

