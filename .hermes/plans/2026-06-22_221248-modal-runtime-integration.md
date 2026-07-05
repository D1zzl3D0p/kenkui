# Optional Modal Runtime Integration Implementation Plan

> **For Hermes:** Use subagent-driven-development skill to implement this plan task-by-task.

**Goal:** Add optional Modal execution for every heavy on-device workload so local clients can run with minimal dependencies while TTS, spaCy/BookNLP, and other NLP-heavy steps execute in isolated, scalable Modal functions when configured.

**Architecture:** Keep `kenkui` as the canonical library/runtime, but make Modal a first-class optional execution context behind existing provider boundaries. The default remains local and backwards-compatible; Modal is selected via env/config/job fields and registered at startup only when `kenkui[modal]` is installed and Modal is configured.

**Tech Stack:** Python 3.12, Pydantic settings, existing `TTSExecutionProvider` and NLP provider protocols, optional `modal` extra, stdout/stderr logging, XDG cache/config paths, pytest.

---

## Current Context

- `pyproject.toml` already declares `modal = ["modal>=0.73.0", "boto3>=1.35.0"]`, while heavy local dependencies (`pocket-tts`, `spacy`, `booknlp`) are still in core dependencies.
- TTS execution has an explicit extension boundary in `src/kenkui/services/execution_service.py`:
  - `TTSExecutionMode.LOCAL | MODAL`
  - `register_tts_execution_provider()`
  - `LocalTTSProvider` calls `AudioBuilder` in-process.
- NLP execution already has a provider-context split in `src/kenkui/nlp/providers/_base.py` and `src/kenkui/nlp/providers/_factory.py`:
  - local wrappers in `src/kenkui/nlp/providers/local.py`
  - non-local modes route through `register_nlp_extension()`.
- Heavy TTS code is concentrated in `src/kenkui/workers.py` (`pocket_tts.TTSModel.load_model`, per-worker cache, chapter rendering).
- spaCy/BookNLP-heavy discovery is currently reachable through `src/kenkui/nlp/entities.py`, `src/kenkui/nlp/booknlp_roster.py`, and `src/kenkui/nlp/providers/booknlp.py`.
- Runtime status currently probes local spaCy in `KenkuiService.multivoice_status()` (`src/kenkui/services/application_service.py:1393`).
- AGENTS.md says remote execution should remain extension-hook based and config/secrets must be 12-factor friendly. This plan keeps Modal optional and env-driven, with no UI or provider-specific policy in clients.

## Guiding Principles

1. **Default local behavior must not regress.** Existing jobs, caches, config files, and tests should pass when Modal is not installed.
2. **Heavy dependencies become optional where practical.** Core should import `pocket_tts`, `spacy`, and `booknlp` only inside local adapters/functions or Modal image definitions.
3. **12-factor configuration.** Use `KENKUI_*` environment variables and Pydantic settings. No checked-in credentials, local-only credential files, or implicit host state.
4. **Provider boundaries over conditionals.** Avoid sprinkling `if modal` throughout business logic; route through `TTSExecutionProvider`, `ExtractionProvider`, and `AttributionProvider`.
5. **Serializable contracts.** Modal calls should exchange dataclasses/Pydantic JSON/dicts and artifact URIs, not live objects or local process state.
6. **Small deployable silos.** TTS, spaCy roster extraction, BookNLP roster/attribution, and optional post-processing should be separate Modal functions/images so specs can scale independently.
7. **Clean failure modes.** If Modal is selected but unavailable, raise typed/actionable errors before work begins.

---

## Proposed Architecture

### Runtime modes

- `local`: current behavior.
- `modal`: run heavy stage on Modal.
- Future-compatible shape: preserve enum-backed modes, but keep provider registration open for additional extension packages.

### Modal silos

Create `src/kenkui/modal_runtime/` as an optional package that is imported only when explicitly registered.

Suggested modules:

- `src/kenkui/modal_runtime/__init__.py`
  - exports `register_modal_runtime()`.
- `src/kenkui/modal_runtime/settings.py`
  - `ModalRuntimeConfig(BaseSettings)` sourced from `KENKUI_MODAL_*`.
- `src/kenkui/modal_runtime/app.py`
  - Modal app, image builders, function declarations.
- `src/kenkui/modal_runtime/tts.py`
  - `ModalTTSProvider` implementing `TTSExecutionProvider`.
  - Modal function(s) for chapter rendering via `worker_process_chapter` or a smaller shared render primitive.
- `src/kenkui/modal_runtime/nlp.py`
  - `ModalExtractionProvider` and `ModalAttributionProvider` implementing NLP protocols.
  - Modal functions for spaCy/BookNLP/LLM-local-compatible stages.
- `src/kenkui/modal_runtime/artifacts.py`
  - upload/download abstraction for job inputs/outputs.
  - start with Modal volumes/dicts or mounted local file bytes; keep S3/R2 as future adapter if needed.
- `src/kenkui/modal_runtime/errors.py`
  - actionable Modal-specific exceptions that can map to `ExecutionOutcome.error_message`.

### Runtime registration

Add one non-invasive startup hook:

- `src/kenkui/services/runtime_service.py`
  - `register_configured_runtimes(app_config: AppConfig) -> None`
  - If `app_config.modal_enabled` or `KENKUI_MODAL_ENABLED=true`, import `kenkui.modal_runtime` and call `register_modal_runtime()`.
  - If not enabled, do nothing.

Call this from service/runtime initialization, likely `KenkuiService.__init__()` before queue processing starts. Do not import Modal at module import time.

### Configuration model

Add env-first config to `AppConfig` and a dedicated Modal config object:

- `KENKUI_MODAL_ENABLED=false`
- `KENKUI_MODAL_APP_NAME=kenkui`
- `KENKUI_MODAL_ENVIRONMENT=`
- `KENKUI_MODAL_TTS_GPU=` (e.g. `A10G`, blank means Modal/image default)
- `KENKUI_MODAL_TTS_CPU=`
- `KENKUI_MODAL_TTS_MEMORY_MB=`
- `KENKUI_MODAL_TTS_TIMEOUT_S=`
- `KENKUI_MODAL_NLP_CPU=`
- `KENKUI_MODAL_NLP_MEMORY_MB=`
- `KENKUI_MODAL_NLP_TIMEOUT_S=`
- `KENKUI_MODAL_ARTIFACT_BACKEND=modal_volume`
- `KENKUI_MODAL_ARTIFACT_VOLUME=kenkui-artifacts`
- `KENKUI_MODAL_KEEP_ARTIFACTS=false`

Keep existing per-job fields `modal_endpoint` and `modal_environment` for compatibility, but prefer structured env/config defaults.

### Artifact strategy

Start simple and testable:

1. For TTS jobs, serialize `ProcessingConfig`, selected chapters, voice IDs/assets, and source ebook bytes or parsed chapter payloads into a job artifact bundle.
2. Modal renders chapter WAVs and returns either:
   - final assembled artifact URI, or
   - per-chapter WAV bundle URI that local post-processing/assembly consumes.
3. Prefer rendering and post-processing fully remote when `tts_execution_mode=modal`, so the local host only downloads final output. This best satisfies “bare minimum specs”.
4. Preserve `ExecutionOutcome.artifact_uri`, `remote_job_id`, `provider_status`, and `output_path` semantics.

### Dependency split target

Move heavy packages out of core dependencies once Modal/local extras are wired:

- Core: parsing, models, config, services, API, lightweight HTTP runtime.
- `local-tts`: `pocket-tts`, `scipy`, `pydub`, audio deps as needed.
- `local-nlp`: `spacy`, `booknlp`, `ollama` if local NLP is desired.
- `modal`: `modal`, artifact backend deps.
- `dev`: test/lint only.

This can be a later compatibility step; first add lazy imports and optional Modal calls without breaking installs.

---

## Step-by-Step Plan

### Phase 1: Formalize runtime configuration and registration

#### Task 1: Add Modal settings to config

**Objective:** Make Modal env/config discoverable without importing Modal.

**Files:**
- Modify: `src/kenkui/models/config.py`
- Test: `tests/test_models.py` or new `tests/test_modal_runtime_config.py`

**Steps:**
1. Add Pydantic fields to `AppConfig` for Modal enablement and default resources.
2. Keep defaults local/off.
3. Add validators for positive integers/timeouts and non-empty strings.
4. Add tests proving `KENKUI_MODAL_ENABLED=true` and nested resource env vars load correctly.
5. Run: `uv run pytest tests/test_models.py tests/test_modal_runtime_config.py -q`.

#### Task 2: Add runtime registration service

**Objective:** Centralize optional runtime registration and keep imports lazy.

**Files:**
- Create: `src/kenkui/services/runtime_service.py`
- Modify: `src/kenkui/services/application_service.py`
- Test: `tests/test_runtime_service.py`

**Steps:**
1. Implement `register_configured_runtimes(app_config)`.
2. If Modal disabled, return without side effects.
3. If Modal enabled, import `kenkui.modal_runtime.register_modal_runtime` inside the function.
4. If import fails, raise an actionable error: “Install kenkui[modal] and configure Modal auth”.
5. Call the registration from `KenkuiService.__init__()`.
6. Test disabled path does not import Modal.
7. Test enabled path invokes a monkeypatched register function.
8. Run: `uv run pytest tests/test_runtime_service.py -q`.

#### Task 3: Add Modal runtime package skeleton

**Objective:** Create optional provider package with no import-time Modal requirement.

**Files:**
- Create: `src/kenkui/modal_runtime/__init__.py`
- Create: `src/kenkui/modal_runtime/settings.py`
- Create: `src/kenkui/modal_runtime/errors.py`
- Test: `tests/test_modal_runtime_registration.py`

**Steps:**
1. Implement `register_modal_runtime(app_config=None)`.
2. Inside it, register `ModalTTSProvider` under `"modal"` and NLP extension factories for modal modes.
3. Keep concrete providers as lightweight classes initially; defer real Modal calls to later tasks.
4. Test that `get_tts_execution_provider()` resolves modal after registration.
5. Test `get_extraction_provider(NlpExecutionMode.MODAL)` returns the modal extraction provider after registration.
6. Run: `uv run pytest tests/test_modal_runtime_registration.py tests/test_nlp_factory.py -q`.

---

### Phase 2: Define serializable contracts for remote calls

#### Task 4: Add DTOs for TTS remote jobs

**Objective:** Make TTS Modal calls independent of local live objects.

**Files:**
- Create: `src/kenkui/modal_runtime/contracts.py`
- Test: `tests/test_modal_contracts.py`

**Steps:**
1. Add Pydantic/dataclass contracts such as:
   - `RemoteTTSRequest`
   - `RemoteTTSResult`
   - `RemoteChapterRenderRequest`
   - `RemoteChapterRenderResult`
2. Include JSON-safe forms of `ProcessingConfig`, chapters, voice mapping, quality settings, output basename, and artifact hints.
3. Add round-trip tests using a minimal `Chapter` and `ProcessingConfig`.
4. Run: `uv run pytest tests/test_modal_contracts.py -q`.

#### Task 5: Add DTOs for NLP remote jobs

**Objective:** Define stable payloads for spaCy/BookNLP/attribution offload.

**Files:**
- Modify: `src/kenkui/modal_runtime/contracts.py`
- Test: `tests/test_modal_contracts.py`

**Steps:**
1. Add `RemoteExtractionRequest`, `RemoteExtractionResult`, `RemoteAttributionRequest`, `RemoteAttributionResult`.
2. Ensure `CharacterRoster` and `AttributionResult` round-trip through their existing JSON/dict shapes or add helper methods if missing.
3. Preserve book hash/provider/model/method in payload metadata for cache compatibility.
4. Run: `uv run pytest tests/test_modal_contracts.py -q`.

#### Task 6: Add artifact abstraction

**Objective:** Decouple remote compute from storage transport.

**Files:**
- Create: `src/kenkui/modal_runtime/artifacts.py`
- Test: `tests/test_modal_artifacts.py`

**Steps:**
1. Define an `ArtifactStore` protocol with `put_bytes`, `get_bytes`, `put_path`, `get_path`.
2. Add a local filesystem test implementation used by unit tests.
3. Add a Modal Volume implementation behind lazy `import modal`.
4. Keep credentials/env names outside code; read only config/env.
5. Run: `uv run pytest tests/test_modal_artifacts.py -q`.

---

### Phase 3: Silo pocket-tts rendering into Modal

#### Task 7: Extract chapter rendering primitive

**Objective:** Reuse current `worker_process_chapter` logic locally and remotely without duplicating TTS behavior.

**Files:**
- Modify: `src/kenkui/workers.py`
- Possibly modify: `src/kenkui/parsing.py`
- Test: `tests/test_workers.py`

**Steps:**
1. Identify the smallest pure function inside `worker_process_chapter()` that renders one chapter from a `Chapter`, config dict, and temp dir.
2. Extract it without changing output messages or retry behavior.
3. Keep `worker_process_chapter()` as the subprocess-safe wrapper used by local `ProcessPoolExecutor`.
4. Add tests for the extracted function using existing mocks.
5. Run: `uv run pytest tests/test_workers.py -q`.

#### Task 8: Implement Modal TTS provider control flow

**Objective:** Have `tts_execution_mode=modal` submit remote TTS work and return `ExecutionOutcome`.

**Files:**
- Create/modify: `src/kenkui/modal_runtime/tts.py`
- Modify: `src/kenkui/modal_runtime/__init__.py`
- Test: `tests/test_modal_tts_provider.py`

**Steps:**
1. Implement `ModalTTSProvider.execute()` using contract DTOs and artifact store.
2. For the first implementation, call an injectable `RemoteTTSClient` interface so unit tests do not require Modal.
3. Map remote lifecycle to metadata callback fields:
   - `provider_status="queued" | "running" | "completed" | "failed"`
   - `remote_job_id`
   - `artifact_uri`
4. Implement `cancel()` as best-effort; if Modal cancellation is unavailable, mark as unsupported but non-fatal.
5. Unit-test success, failure, cancel/pause mapping, and metadata updates.
6. Run: `uv run pytest tests/test_modal_tts_provider.py tests/test_application_service.py -q`.

#### Task 9: Implement Modal TTS functions/images

**Objective:** Actually run pocket-tts in a Modal image silo.

**Files:**
- Modify: `src/kenkui/modal_runtime/app.py`
- Modify: `src/kenkui/modal_runtime/tts.py`
- Test: unit tests with monkeypatch; optional integration marked slow.

**Steps:**
1. Define a TTS Modal image that installs only needed TTS/audio deps: `pocket-tts`, `scipy`, `pydub`, `imageio-ffmpeg`, audio post-processing deps if remote assembly is included.
2. Define a GPU-backed `render_tts_job` Modal function with resource values from `ModalRuntimeConfig`.
3. Inside the function, deserialize `RemoteTTSRequest`, render chapters, post-process/assemble final M4B if configured, write to artifact store, return `RemoteTTSResult`.
4. Avoid reading local config files inside Modal; pass all config explicitly in request/env.
5. Add an integration test file `tests/integration/test_modal_tts.py` marked `modal`/`slow` and skipped unless `KENKUI_MODAL_INTEGRATION=1`.
6. Run unit tests: `uv run pytest tests/test_modal_tts_provider.py -q`.
7. Optional real test: `KENKUI_MODAL_INTEGRATION=1 uv run pytest tests/integration/test_modal_tts.py -q`.

---

### Phase 4: Silo spaCy/BookNLP NLP into Modal

#### Task 10: Add Modal NLP provider classes

**Objective:** Route extraction/attribution modal modes through provider protocols.

**Files:**
- Create/modify: `src/kenkui/modal_runtime/nlp.py`
- Test: `tests/test_modal_nlp_provider.py`

**Steps:**
1. Implement `ModalExtractionProvider.build_roster()`.
2. Implement `ModalAttributionProvider.attribute_chapter()`.
3. Use injectable remote client methods for tests:
   - `run_extraction(RemoteExtractionRequest) -> RemoteExtractionResult`
   - `run_attribution(RemoteAttributionRequest) -> RemoteAttributionResult`
4. Preserve progress callbacks with synthetic stage messages before/after remote calls.
5. Test extraction success, attribution success, remote failure, and callbacks.
6. Run: `uv run pytest tests/test_modal_nlp_provider.py tests/test_nlp_factory.py -q`.

#### Task 11: Implement spaCy Modal extraction function

**Objective:** Remove the need for local spaCy when discovery method is `spacy` or when heuristic fallback needs spaCy.

**Files:**
- Modify: `src/kenkui/modal_runtime/app.py`
- Modify: `src/kenkui/modal_runtime/nlp.py`
- Possibly modify: `src/kenkui/nlp/providers/_factory.py` only if routing needs finer granularity.
- Test: `tests/test_modal_nlp_provider.py`, integration skipped by default.

**Steps:**
1. Define a CPU Modal image with `spacy` and `en_core_web_sm` installed.
2. Add `run_spacy_extraction` function that calls existing roster logic with a loaded spaCy model.
3. Pass `chapters`, `method`, `provider`, `model`, `book_hash`, and `series_roster` explicitly.
4. Return a `CharacterRoster` JSON payload.
5. Ensure no local `import spacy` occurs when modal mode is selected.
6. Add tests that monkeypatch local `spacy` as unavailable but Modal provider still returns remote roster.
7. Run: `uv run pytest tests/test_modal_nlp_provider.py -q`.

#### Task 12: Implement BookNLP Modal extraction/attribution functions

**Objective:** Silo BookNLP and its transformer/model footprint into Modal.

**Files:**
- Modify: `src/kenkui/modal_runtime/app.py`
- Modify: `src/kenkui/modal_runtime/nlp.py`
- Test: `tests/test_modal_nlp_provider.py`, integration skipped by default.

**Steps:**
1. Define a CPU/high-memory Modal image with `booknlp`, `spacy`, and compatible transformer deps.
2. Add `run_booknlp_extraction` using existing `build_roster_from_booknlp()`.
3. Add `run_booknlp_attribution` using existing `_run_booknlp()`/adapter behavior.
4. Keep model size and memory timeout configurable via `KENKUI_MODAL_NLP_*`.
5. Add integration tests skipped unless Modal credentials and `KENKUI_MODAL_INTEGRATION=1` are present.
6. Run: `uv run pytest tests/test_modal_nlp_provider.py tests/test_nlp_booknlp_adapters.py -q`.

---

### Phase 5: Wire job-level execution modes cleanly

#### Task 13: Propagate per-job NLP execution modes through service APIs

**Objective:** Ensure user/client choices can select Modal for extraction and attribution independently.

**Files:**
- Modify: `src/kenkui/services/nlp_service.py`
- Modify: `src/kenkui/api.py` if public facade lacks parameters
- Modify: `src/kenkui/models/api.py` only if HTTP request/response shape needs fields
- Test: `tests/test_api_attribution_options.py`, `tests/test_nlp_service.py`

**Steps:**
1. Audit `fast_scan()`, `full_analysis()`, `suggest_cast()`, and queue-job paths.
2. Ensure `job_nlp_execution_mode` maps to `NLPConfig.extraction_mode`.
3. Ensure `job_attribution_execution_mode` maps to `NLPConfig.attribution_mode`.
4. Preserve defaults from `AppConfig` when job overrides are absent.
5. Test local default, modal extraction only, modal attribution only, and both modal.
6. Run: `uv run pytest tests/test_nlp_service.py tests/test_api_attribution_options.py -q`.

#### Task 14: Update status reporting to be runtime-aware

**Objective:** Avoid local spaCy/Ollama probes when Modal mode is configured.

**Files:**
- Modify: `src/kenkui/services/application_service.py:1393`
- Modify: `src/kenkui/models/api.py` if response needs additional fields
- Test: `tests/test_application_service.py`

**Steps:**
1. Change `multivoice_status()` to report local availability for local mode and configured remote availability for modal mode.
2. Avoid importing `spacy` if `nlp_execution_mode=modal` and local status is not requested.
3. Add fields only if backward-compatible, e.g. `execution_mode`, `remote_ok`, `remote_message`; otherwise encode in existing `message`.
4. Test local behavior remains unchanged.
5. Test Modal-enabled config does not require local spaCy package.
6. Run: `uv run pytest tests/test_application_service.py -q`.

---

### Phase 6: Dependency cleanup and 12-factor hardening

#### Task 15: Split optional dependencies

**Objective:** Let bare installs avoid local heavy ML/TTS packages.

**Files:**
- Modify: `pyproject.toml`
- Test: package metadata/import smoke tests.

**Steps:**
1. Move `pocket-tts`, `spacy`, `booknlp`, and possibly local audio-heavy deps into extras after verifying lazy imports.
2. Suggested extras:
   - `local-tts`
   - `local-nlp`
   - `local-audio`
   - `modal`
   - `all-local`
3. Keep a compatibility extra if needed, e.g. `full = [local-tts, local-nlp, modal]`.
4. Add tests that `import kenkui` and `from kenkui.models import AppConfig` work without optional extras mocked as missing.
5. Run: `uv run pytest tests/test_library_contracts.py tests/test_models.py -q`.

#### Task 16: Add typed errors for missing local extras

**Objective:** Give actionable messages when users choose local heavy modes without installed deps.

**Files:**
- Modify: `src/kenkui/errors.py`
- Modify: local adapters/import sites:
  - `src/kenkui/workers.py`
  - `src/kenkui/nlp/providers/booknlp.py`
  - `src/kenkui/nlp/entities.py`
- Test: `tests/test_workers.py`, `tests/test_nlp_booknlp_adapters.py`

**Steps:**
1. Add `MissingOptionalDependencyError` or similar typed exception.
2. Wrap local `pocket_tts`, `spacy`, and `booknlp` imports with clear install guidance.
3. Ensure Modal mode never triggers these local import errors.
4. Run focused tests.

#### Task 17: Audit 12-factor behavior

**Objective:** Ensure Modal implementation is environment-driven and stateless.

**Files:**
- Modify as needed:
  - `src/kenkui/modal_runtime/settings.py`
  - `src/kenkui/config.py`
  - `src/kenkui/log.py`
- Test: `tests/test_modal_runtime_config.py`

**Checklist:**
1. No credentials in config files or repo.
2. All credentials supplied by Modal CLI/env/secrets or `KENKUI_*` env vars.
3. Logs go to stdout/stderr unless `KENKUI_LOG_FILE` is configured.
4. Processes are disposable; all durable state is in artifact store/XDG cache/final output path.
5. Concurrency/resource values are config, not code constants.
6. Local and remote stages pass config explicitly rather than reading ambient local files.

---

### Phase 7: Documentation and verification

#### Task 18: Document runtime modes

**Objective:** Explain how clients and operators opt into Modal without UI-specific policy.

**Files:**
- Modify: `README.md` or docs file if present
- Possibly add: `docs/modal-runtime.md`

**Content:**
1. Install options:
   - bare/core
   - `kenkui[local-tts,local-nlp]`
   - `kenkui[modal]`
2. Env vars for Modal runtime.
3. Example local vs Modal job config.
4. 12-factor deployment notes.
5. Known limitations: cancellation semantics, artifact retention, integration tests require credentials.

#### Task 19: Full test pass

**Objective:** Verify the library contract after integration.

**Commands:**
1. `uv run pytest tests/test_modal_runtime_config.py tests/test_modal_runtime_registration.py tests/test_modal_contracts.py tests/test_modal_artifacts.py tests/test_modal_tts_provider.py tests/test_modal_nlp_provider.py -q`
2. `uv run pytest tests/test_nlp_factory.py tests/test_workers.py tests/test_application_service.py tests/test_nlp_service.py tests/test_api_attribution_options.py -q`
3. `uv run pytest -q`
4. If available: `uv run ruff check src tests`
5. Optional integration: `KENKUI_MODAL_INTEGRATION=1 uv run pytest tests/integration -q`

---

## Files Likely to Change

- `pyproject.toml`
- `src/kenkui/models/config.py`
- `src/kenkui/models/common.py` only if additional runtime enum values are needed later
- `src/kenkui/models/job.py` if job-specific Modal resource overrides are added
- `src/kenkui/models/api.py`
- `src/kenkui/services/application_service.py`
- `src/kenkui/services/execution_service.py` only if the provider protocol needs richer progress/cancel metadata
- `src/kenkui/services/runtime_service.py` (new)
- `src/kenkui/services/nlp_service.py`
- `src/kenkui/nlp/providers/_factory.py`
- `src/kenkui/nlp/providers/_base.py` only if protocol payloads need extension
- `src/kenkui/workers.py`
- `src/kenkui/parsing.py`
- `src/kenkui/modal_runtime/` (new package)
- `tests/test_modal_*.py` (new)
- Existing focused tests listed above

## Risks and Tradeoffs

- **Remote artifact complexity:** Uploading/downloading full books, voice assets, WAVs, and M4Bs is the hardest contract. Start with explicit artifact DTOs and test with a local fake store.
- **Dependency split is breaking if rushed:** Make imports lazy first; move dependencies to extras only after import-smoke tests pass.
- **Cancellation semantics:** Modal cancellation may not match local process cancellation. Model this as best-effort and expose provider status clearly.
- **Cache compatibility:** NLP roster/attribution caches must continue using existing XDG helpers and schemas. Remote calls should return data for local cache writers rather than writing arbitrary remote caches.
- **Cost/latency:** Per-chapter Modal calls may incur startup overhead. Prefer job-level remote orchestration with warm model caches inside Modal functions/images; use per-chapter parallelism only after measuring.
- **AGENTS.md remote-extension rule:** Keep Modal isolated behind registration and optional extras. Avoid cloud auth, billing, or dashboard policy in core clients.

## Open Questions Before Implementation

1. Should `tts_execution_mode=modal` render the entire audiobook remotely, or only synthesize chapter WAVs and assemble locally? Recommended: entire audiobook remotely for minimum local specs.
2. Which artifact backend should be first-class: Modal Volume only, or S3/R2 via `boto3` from the existing modal extra? Recommended: Modal Volume first, S3/R2 as a later `ArtifactStore` implementation.
3. Should local heavy dependencies be moved out of core immediately or after Modal is functionally verified? Recommended: after lazy imports and Modal unit tests, before final release.
4. Is Modal intended for all NLP providers, or only local-heavy tools (`spacy`, `booknlp`, `pocket-tts`)? Recommended: silo local-heavy tools first; keep SaaS LLM providers (`openrouter`, `openai`, etc.) as direct API calls unless there is a security reason to proxy through Modal.

## Suggested Implementation Order

1. Config + registration skeleton.
2. Contract DTOs and artifact fake store.
3. Modal TTS provider with mocked remote client.
4. Real Modal TTS function.
5. Modal NLP provider with mocked remote client.
6. Real spaCy/BookNLP Modal functions.
7. Dependency split and typed optional-dependency errors.
8. Docs + full verification.
