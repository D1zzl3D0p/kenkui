# AI Guide for `kenkui`

This file is the working contract for AI agents editing this repository.

## Product Direction

Near-term, `kenkui` is a thin-client system with a local server as the main product boundary.

- The local server is the source of truth for queueing, job lifecycle, config, NLP orchestration, voice logic, and rendering workflows.
- Local execution is the default path for CPU-friendly models such as `pocket-tts`.
- The expected baseline product experience is: client talks to a local `kenkui server`, and that local server performs the work on the user's machine when feasible.
- The CLI is a client of that server, not the owner of business rules.
- Future web, iOS, and Android clients should also target the same server API.
- Remote execution is a planned extension and fallback path, but it should be added as a separate execution provider / control-plane concern, not by coupling cloud-only concerns into the local server runtime.
- Remote work should be treated as explicitly distinct from local work in both architecture and product policy: different cost model, different storage/notification concerns, and potentially different deployment/runtime concerns.
- Capability detection and benchmarking are explicitly out of scope for the current implementation phase.

## Purpose

`kenkui` converts ebooks into audiobooks. The main runtime flow is:

1. CLI accepts a book and user choices.
2. CLI talks to the local FastAPI worker server over HTTP.
3. The server persists queue state and app config.
4. The worker server turns queued jobs into `ProcessingConfig` objects.
5. `AudioBuilder` parses chapters, optionally enriches them with NLP annotations, and dispatches chapter rendering to subprocess workers.
6. Worker functions render chapter audio, post-processing runs, and the final M4B is written to disk.

## Core Principles

- Keep changes inside the existing layer boundaries.
- Prefer small, explicit functions over abstract frameworks.
- Use dataclasses and typed models as the source of truth for app state.
- Preserve current CLI and API behavior unless the task explicitly changes it.
- Put business logic in services or core modules, not in thin transport wrappers.
- Add tests for behavior changes, especially around queueing, NLP, voice selection, and worker flow.

## Architecture Map

### Entry points

- `src/kenkui/__main__.py`
  Dispatches CLI commands, auto-starts the local server when needed, and routes into CLI handlers.
- `src/kenkui/cli/`
  User-facing terminal flows built with `argparse`, `InquirerPy`, and `rich`.

### Client and server boundary

- `src/kenkui/api_client.py`
  HTTP client used by CLI code.
- `src/kenkui/server/api.py`
  FastAPI routes and request/response models.
- `src/kenkui/server/worker.py`
  In-process queue manager, queue persistence, and job lifecycle control.
- `src/kenkui/server/tasks.py`
  Background task registry for async service operations such as scans, downloads, and previews.

### Domain models and config

- `src/kenkui/models.py`
  Shared dataclasses and enums for jobs, chapters, config, NLP results, queue items, and processing state.
- `src/kenkui/config.py`
  XDG config resolution plus TOML load/save helpers.

### Ebook ingestion and parsing

- `src/kenkui/readers/`
  Format-specific readers behind the `EbookReader` abstraction.
- `src/kenkui/services/book_service.py`
  High-level parse and chapter-filter operations, backed by cache.
- `src/kenkui/chapter_filter.py`
  Chapter selection rules and filtering primitives.
- `src/kenkui/chapter_classifier.py`
  Chapter tagging and classification logic.

### NLP pipeline

- `src/kenkui/nlp/`
  Quote extraction, entity discovery, chunking, attribution, roster generation, and JSON cache helpers.
- Multi-voice mode depends on cached or freshly generated annotated chapters from this layer.

### Voice system

- `src/kenkui/voice_registry.py`
  Voice metadata lookup and filtering.
- `src/kenkui/voice_loader.py`
  Resolves a selected voice into audio prompt data.
- `src/kenkui/services/voice_service.py`
  Voice listing, preview generation, exclusion pool management, and cast suggestion.
- `src/kenkui/services/download_service.py` and `src/kenkui/voice_download.py`
  Voice download and fetch flows.

### Rendering pipeline

- `src/kenkui/parsing.py`
  `AudioBuilder`, chapter loading, ETA tracking, stitching, output naming, and overall processing orchestration.
- `src/kenkui/workers.py`
  Subprocess-safe TTS chapter rendering helpers and model caching.
- `src/kenkui/post_processing.py`
  Audio cleanup and mastering steps.

### Tests

- `tests/`
  Behavioral tests grouped by subsystem rather than by test style. Treat this directory as the regression contract.

## Data and Control Flow

### Standard single-voice flow

1. CLI gathers inputs.
2. CLI sends a queue request through `APIClient`.
3. Server converts request payloads into `JobConfig` and persists queue state.
4. Worker server starts the next pending job.
5. `AudioBuilder` reads the ebook through `readers/`.
6. Chapters are rendered in worker subprocesses.
7. Post-processing and final M4B assembly run.
8. Queue state is updated to completed or failed.

### Multi-voice flow

1. CLI or server triggers fast scan / NLP analysis.
2. NLP cache files are written under the config directory.
3. Job config stores `annotated_chapters_path`, `roster_cache_path`, and `speaker_voices`.
4. `AudioBuilder` loads annotated chapters instead of plain chapter text when available.
5. Worker rendering switches from paragraph narration to segment-based speaker rendering.

### Future execution modes

The architecture should prepare for these modes even if only local execution is implemented today:

- `local_only`
- `prefer_local`
- `prefer_remote`
- `remote_only`

Current implementation work should keep models and APIs compatible with those modes, without adding remote-specific complexity prematurely.

### Local vs Remote demarcation

- Local server:
  owns queueing, config, orchestration, local assets, local TTS/NLP execution, and the main REST API used by clients.
- Remote control/execution path:
  is optional and future-facing; it should exist behind a separate execution-provider boundary and should not redefine the local server contract.
- Shared domain logic:
  should be reusable by both paths, but the local server remains the primary integration target during the current phase.

## Style Guide

### Python style

- Target Python 3.12.
- Keep type hints on public functions and meaningful internal helpers.
- Prefer dataclasses for internal state containers.
- Use enums for fixed status or mode values.
- Keep imports explicit and local only when needed to avoid circular imports or heavy startup cost.
- Follow existing Ruff settings in `pyproject.toml`.

### Module design

- Keep transport layers thin:
  `cli/` handles prompts and display.
  `api_client.py` handles HTTP.
  `server/api.py` handles route translation.
  `services/` handles business operations.
  Core processing modules handle rendering/parsing/NLP internals.
- Put serialization logic close to the model that owns the data.
- Prefer adding a focused helper in an existing module over creating a new generic utility module.

### State and persistence

- Persist config and queue state through the existing TOML helpers.
- Preserve XDG-based paths and current cache locations.
- Keep job-level overrides optional and explicit; `None` means inherit from app config.

### Error handling

- Fail with plain, actionable errors.
- Preserve current sentinel-style recovery paths where they already exist, such as missing NLP cache handling.
- Do not swallow exceptions unless the module already treats that path as best-effort fallback.

### CLI and UX

- Maintain current command names, queue semantics, and wizard behavior.
- Preserve rich terminal output patterns instead of replacing them with raw prints unless the file already uses a simple console fallback.
- Back-navigation and noninteractive headless mode are part of the product surface, not incidental details.

### Concurrency and workers

- Respect the existing multiprocessing boundary.
- Keep worker entry points import-safe and subprocess-safe.
- Avoid introducing hidden global state beyond the existing per-process caches.
- If changing progress reporting, keep queue and live dashboard behavior coherent.

### Testing expectations

- Add or update tests when changing:
  queue lifecycle
  config serialization
  API payload shape
  voice assignment behavior
  NLP cache behavior
  output naming
  worker progress / pause / resume behavior
- Prefer small focused tests near the subsystem already covering the behavior.

## Where To Put Changes

- New CLI flag or wizard interaction: `src/kenkui/__main__.py` and `src/kenkui/cli/`
- New HTTP endpoint: `src/kenkui/server/api.py`, then `src/kenkui/api_client.py`, then tests
- Queue or lifecycle change: `src/kenkui/server/worker.py` and related queue tests
- New persistent config field: `src/kenkui/models.py` and `src/kenkui/config.py`
- Book parsing or chapter selection change: `readers/`, `chapter_filter.py`, `services/book_service.py`
- Voice metadata or auto-assignment change: `voice_registry.py`, `services/voice_service.py`, possibly `cli/add.py`
- NLP analysis change: `src/kenkui/nlp/`
- Audio rendering or synthesis change: `src/kenkui/parsing.py`, `src/kenkui/workers.py`, `post_processing.py`

## Practical Rules For AI Agents

- Read the owning module before editing a subsystem.
- Reuse existing dataclasses, enums, and service functions before introducing new ones.
- Preserve backward compatibility for serialized queue/config/job data unless migration is intentional.
- Do not move user-facing logic into the worker layer.
- Do not bypass the API client from CLI code when server-backed behavior already exists.
- Keep new abstractions justified by repeated use, not by hypothetical reuse.
- Prefer early architectural cleanup when it removes likely client/server foot-guns, even before all user-facing features are finished.
- When adding abstractions for future remote execution, keep them concrete and close to current needs: execution provider boundaries, job packaging boundaries, and stable REST contracts are desirable; speculative distributed orchestration code is not.

## Current Implementation Plan

The active roadmap is Phase 1 plus Phase 2 preparation only.

### Phase 1: Finish the local-server boundary

Goal: make the local server the single owner of product behavior.

- Move remaining business logic out of `cli/` and into service or server-owned modules.
- Keep `cli/` focused on prompts, display, local process startup, and API calls.
- Consolidate duplicated helper logic where it materially reduces drift between CLI, services, NLP, and workers.
- Add or refine server endpoints only when they represent durable product capabilities rather than terminal-specific flows.
- Preserve current local queue semantics and serialized job/config compatibility.

### Phase 2: Prepare execution-provider boundaries

Goal: make local execution pluggable without implementing remote execution yet.

- Introduce an explicit execution policy model in shared models.
- Define a local execution provider boundary around job start, progress, completion, artifact location, and failure reporting.
- Preserve local-first assumptions in current UX and code paths; remote is a fallback, not the baseline.
- Keep queue/job state and API payloads extensible for future remote artifact references and notification metadata.
- Avoid introducing cloud auth, billing, benchmarking, or farm orchestration in this phase.

### Implementation priorities

1. Remove duplicated business logic that causes drift between CLI and services.
2. Clarify which modules are transport, service, domain-model, and execution layers.
3. Normalize queue and job state so future execution backends can share the same contract.
4. Add tests that lock in the new boundaries.

## Local Structure Reference

- `src/kenkui/`: application code
- `tests/`: regression suite
- `docs/`: human documentation and AI-facing notes
- `build/`, `dist/`, `.venv/`, `*.egg-info`, cache dirs: environment or generated artifacts, not architecture

For observed mismatches between this guide and the current codebase, see `docs/AI_NOTES.md`.
