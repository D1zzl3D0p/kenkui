# AI Guide for `kenkui`

This file is the working contract for AI agents editing this repository.

## Product Direction

`kenkui` is the reusable Python library and canonical local HTTP runtime for
ebook-to-audiobook conversion. Interactive clients such as `kentui` and `kengui`
should consume this package instead of maintaining separate service wrappers.

- The library owns ebook parsing, chapter filtering, config models, NLP pipeline
  orchestration, voice catalog logic, rendering, workers, post-processing, cache
  schemas, queueing, and the local HTTP API contract.
- Clients own screens, prompts, notifications, shell integration, and
  deployment-specific policy.
- Local execution is the only built-in execution path today.
- Remote execution remains an extension hook for external packages; do not add
  cloud auth, billing, farm scheduling, or remote runtime concerns here.
- Preserve serialized config/cache/job compatibility unless a migration is
  intentional and tested.

## Purpose

The main library flow is:

1. A caller parses an ebook and chooses config through its own UI or API.
2. The caller builds library dataclasses such as `ProcessingConfig` or calls
   service functions such as `fast_scan`, `full_analysis`, and `suggest_cast`.
3. The library reads the ebook through `readers/`.
4. Optional NLP analysis writes roster and attribution caches under XDG paths.
5. `AudioBuilder` dispatches chapter rendering to subprocess-safe workers.
6. Post-processing and M4B assembly produce the final artifact on disk.

## Architecture Map

- `src/kenkui/api.py`
  Public convenience API for callers. Keep this as a thin facade over models and
  services.
- `src/kenkui/__init__.py`
  Package export facade only.
- `src/kenkui/models/`
  Dataclasses, enums, and Pydantic config models split by domain. The package
  re-exports the supported `kenkui.models` surface.
- `src/kenkui/config.py`
  XDG config/cache resolution and TOML load/save helpers.
- `src/kenkui/readers/`
  Format-specific ebook readers behind the `EbookReader` abstraction.
- `src/kenkui/services/`
  Library service layer for book parsing/cache, NLP orchestration, voice
  catalog/assignment, downloads, series manifests, setup, and auth.
- `src/kenkui/nlp/`
  Quote extraction, entity discovery, provider adapters, chunking, attribution,
  roster generation, and NLP cache helpers.
- `src/kenkui/parsing.py`
  Rendering orchestration and compatibility wrapper for `AudioBuilder`.
- `src/kenkui/workers.py`
  Import-safe multiprocessing TTS worker helpers and per-process model caches.
- `src/kenkui/post_processing.py`
  Audio cleanup and mastering.
- `tests/`
  Behavioral regression suite. Treat it as the contract.

## Layer Rules

- Keep UI out of this repository. Do not add prompts, terminal flows, screens,
  or queue dashboards to core modules.
- Put reusable decisions in services or domain modules, not facades.
- Keep `api.py` and `__init__.py` thin.
- Keep worker entry points subprocess-safe and free of hidden client state.
- Prefer dataclasses/enums/Pydantic settings models as state boundaries.
- Keep extension hooks concrete and local to current needs.

## State, Config, And Caches

- Use existing XDG path helpers in `config.py`.
- Persist config through TOML helpers.
- Store NLP roster/attribution caches under the configured cache directory.
- Keep secrets environment-first. Credential files are only local convenience.
- Logging should be 12-factor friendly: stdout/stderr by default, optional file
  logging only when configured.

## Error Handling

- Prefer typed exceptions from `kenkui.errors` for new actionable failures.
- Do not silently swallow broad exceptions unless the module already treats that
  path as a documented best-effort fallback.
- Missing or corrupt caches may fall back only where the existing UX relies on
  that recovery path.

## Testing Expectations

Add or update tests when changing:

- public API exports
- config serialization or environment precedence
- cache schema compatibility
- NLP provider factory hooks or attribution behavior
- voice assignment, exclusion, or review formatting
- rendering worker messages and pause/resume-adjacent behavior
- no-interactive-I/O guarantees in core modules

Unit tests must not load real TTS, Ollama, BookNLP, HuggingFace, or network
services. Mark real model/network/audio-heavy tests as integration or slow.

## Practical Rules

- Read the owning module before editing a subsystem.
- Reuse existing dataclasses, enums, and service functions before adding new
  abstractions.
- Keep structural moves backed by focused tests and `ruff`.
- Do not couple external clients (`kentui`, `kengui`) back into the core.
- When splitting modules, preserve stable import paths with package re-exports
  where practical.
- When the user asks for implementation, commit completed code changes after
  verification.
