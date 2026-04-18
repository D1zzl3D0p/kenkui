# Implementation Plan

This plan covers the currently approved work only:

- Phase 1: complete the local-server boundary
- Phase 2: prepare execution-provider boundaries

It does not include remote farm implementation, capability detection, or benchmarking.

## Goals

- Make the local FastAPI server the owner of product behavior.
- Reduce rule duplication between CLI and service layers.
- Prepare the codebase for future local vs remote execution without adding remote infrastructure yet.
- Preserve current functionality while tightening module boundaries.

## Non-Goals

- No cloud control plane
- No farm scheduler
- No capability benchmarking
- No client-specific UI rewrites
- No speculative general framework extraction

## Workstreams

### 1. Move remaining product logic behind the server boundary

Target:
- `src/kenkui/cli/add.py`
- `src/kenkui/cli/queue.py`
- `src/kenkui/server/api.py`
- `src/kenkui/services/voice_service.py`
- `src/kenkui/services/book_service.py`

Tasks:
- Identify logic in CLI modules that is not presentation or API-calling logic.
- Move reusable decision logic into service modules.
- Keep CLI modules responsible for prompts, rendering, and request composition only.
- Add API endpoints only where the behavior should be shared by future GUI clients.

Expected result:
- The CLI becomes a thin client over stable server-side behavior.

### 2. Remove concrete duplication that creates drift risk

Target:
- `src/kenkui/cli/add.py`
- `src/kenkui/services/voice_service.py`
- `src/kenkui/workers.py`
- `src/kenkui/nlp/__init__.py`

Tasks:
- Unify pronoun-to-gender mapping in one shared location.
- Unify scene-break detection in one shared location if both call sites should stay aligned.
- Remove duplicated voice-assignment helper logic from CLI where practical.

Expected result:
- One source of truth for shared rules.

### 3. Clarify layer ownership

Target:
- `src/kenkui/models.py`
- `src/kenkui/server/worker.py`
- `src/kenkui/services/`
- `src/kenkui/parsing.py`

Tasks:
- Define which models are transport-facing, queue-facing, and processing-facing.
- Keep queue persistence and lifecycle in `server/worker.py` or a closely related queue module.
- Keep rendering orchestration in `parsing.py` and subprocess rendering in `workers.py`.
- Add minimal documentation or exports where package boundaries are otherwise unclear.

Expected result:
- Easier reasoning for future AI and human contributors.

### 4. Prepare execution-provider boundaries

Target:
- `src/kenkui/models.py`
- `src/kenkui/server/worker.py`
- possibly a new module under `src/kenkui/server/` or `src/kenkui/services/`

Tasks:
- Add an execution policy enum or equivalent shared model.
- Separate job orchestration concerns from execution concerns.
- Define the interface a future execution backend must satisfy:
  submit/start
  progress updates
  pause/resume support or explicit non-support
  completion/failure
  artifact reporting

Expected result:
- Local execution remains the only implementation, but the queue/job model no longer assumes that forever.

### 5. Lock in behavior with tests

Target:
- queue tests
- API tests
- CLI tests
- voice-service tests
- worker wiring tests

Tasks:
- Add tests before or alongside moves that change ownership boundaries.
- Prefer regression tests for API payload shape and job lifecycle behavior.
- Add focused tests for any shared helper extracted from duplicated logic.

Expected result:
- Refactors stay safe while functionality remains stable.

## Suggested Sequence

1. Extract duplicated helper rules into shared code with tests.
2. Move CLI-side decision logic into services.
3. Simplify CLI modules so they only orchestrate prompts and API calls.
4. Introduce execution policy models and local execution-provider boundaries.
5. Normalize queue and artifact reporting models for future remote compatibility.

## Definition of Done for This Phase

- CLI no longer owns reusable business rules that future clients would need.
- Shared logic no longer exists in multiple drifting copies.
- Job and queue models can plausibly support local or remote execution later.
- Existing local flows still work.
- Tests cover the moved boundaries.
