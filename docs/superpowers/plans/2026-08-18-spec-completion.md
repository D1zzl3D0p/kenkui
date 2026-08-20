# Kenkui Specification Completion Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Deliver the complete specified Kenkui system: a production-capable EPUB-to-single-voice-M4B core, durable local and hosted server modes, the browser client, then the specified character-voice and cloud deployment slices.

**Architecture:** Preserve the three-layer boundary: `kenkui-web` uses versioned HTTP/SSE from `kenkui-server`; the server reconstructs a `kenkui.Pipeline` and owns Jobs, billing, assets, and dispatch; `kenkui` alone parses, plans, synthesizes, encodes, and emits interface-neutral events. Complete the core acceptance path before introducing durable application state, then replace only infrastructure adapters for hosted operation.

**Tech Stack:** Python 3.11–3.13, dataclasses, pytest, Ruff, mypy, Pocket-TTS, FFmpeg/ffprobe; FastAPI, Pydantic, SQLite/PostgreSQL, SSE; TypeScript, React, Vite, generated OpenAPI types, Vitest, React Testing Library, Playwright; Cloudflare static hosting/R2, Railway/PostgreSQL, Modal, WorkOS/AuthKit, Stripe.

**Spec:** `../kenkui-suite-v2/specs/00 Kenkui System Specification - Final.md`, `../kenkui-suite-v2/specs/01 Kenkui Core Specification - Final.md`, `../kenkui-suite-v2/specs/02 Kenkui Server Specification - Final.md`, `../kenkui-suite-v2/specs/03 Kenkui Web and Client Specification - Final.md`, `../kenkui-suite-v2/specs/04 Kenkui Cloud and Deployment Specification - Final.md`, and `../kenkui-suite-v2/specs/05 Kenkui Implementation Plan - Final.md`.

## Global Constraints

- `kenkui` stays an independent Python library: no HTTP, users, billing, cloud provider, queue, or server import.
- Pipeline construction is lazy and immutable; `inspect()` and `write()` are the only public effect boundaries.
- EPUB input, single voice, and M4B output are the first complete slice; no breadth work may pre-empt it.
- Pipeline operations contain audiobook semantics only; worker/CPU/GPU/retry/provider policy remains outside Pipeline and `JobSpec`.
- A `JobSpec` is not a serialized Pipeline. Server code reconstructs a Pipeline explicitly.
- A Job snapshot is authoritative. Server events are append-only history/SSE delivery, not event sourcing.
- All public errors have stable codes; logs are structured and never contain source text, credentials, payment data, voice samples, presigned URLs, or raw provider diagnostics.
- Core cache is private implementation state; server Job/billing data never enters it.
- One credit is exactly 1,000 normalized speech characters; successful Jobs settle one reservation, failed/cancelled Jobs charge zero.
- Do not add Redis, Celery, broker infrastructure, WebSockets, GraphQL, pipeline serialization, LiteLLM Proxy, Cloudflare control-plane primitives, or Kubernetes.
- Readable sources only. DRM removal, retailer integrations, and public sharing are out of scope.

---

## Current assessment and required cutover

`kenkui` is materially aligned with the **core architecture**, not yet with the whole system. It has a lazy frozen `Pipeline`, `book()`/`epub()`, stable EPUB inspection IDs, chapter selection, normalization, single-voice intent, validation, an internal planner/cache, spawned worker execution, FFmpeg M4B assembly, events, cancellation, stable errors, and a 300-pass test suite. Its implemented tree is intentionally narrower than the historic proposed tree and does not contain server/UI code, which is architecturally correct.

The v1 core acceptance path is nevertheless blocked: `README.md` declares the production Pocket-TTS renderer fail-closed until an approved local model manifest and authorized voice prompt are supplied; the real-inference test is skipped. `Pipeline.write()` also requires voice and TTS intent before `inspect()`, while the final core specification defines inspection as a source-derived terminal and does not require rendering intent. The current core has no public voice registry/listing, no `assign_voices`/character operations, no LiteLLM adapter, and no persisted book/voice/run ownership model. `kenkui-server` and `kenkui-web` do not exist in the active workspace; therefore API, durable Jobs, local E2E, hosted adapters, billing, auth, and cloud deployment are wholly unimplemented.

The final suite supersedes `Kenkui_vNext_Architecture_Findings(1).md`. Use the historic document for the core boundary, but use the final suite as the implementation contract. “100%” below means all normative V1 through hosted/cloud and character-voice requirements. The `additional formats`, custom voices, persistent library, and native packaging entries are explicitly demand-driven post-V1 candidates, not completion blockers.

### Task 1: Close the core production-renderer gate

**Files:**

- Modify: `pyproject.toml`, `src/kenkui/_tts/production.py`, `src/kenkui/_tts/pocket.py`, `src/kenkui/voices.py`, `README.md`, `docs/pocket-tts-adapter.md`
- Modify: `tests/test_production_manifest_branches.py`, `tests/test_pocket_tts_real.py`
- Create: an approved, versioned local production-manifest fixture under `tests/fixtures/voices/` (metadata only; never commit unlicensed model/prompt assets)

**Interfaces:**

- Consumes: approved Pocket-TTS model revision, redistributable manifest metadata, and an authorized narrator prompt supplied by the project owner.
- Produces: `production_bindings_from_environment(voice_id) -> ExecutionBindings` that accepts only manifest-authorized local assets and a real local smoke-test configuration.

- [ ] **Step 1: Obtain and record the rights decision**

Record the exact Pocket-TTS model revision, asset checksums, license/provenance, permitted usage, prompt authorization, and whether CI may access the assets. Reject activation if any field is unknown; the fail-closed behavior is correct until this decision exists.

- [ ] **Step 2: Write failing activation tests**

```python
def test_approved_manifest_and_authorized_prompt_enable_bindings(monkeypatch, tmp_path):
    monkeypatch.setenv("KENKUI_POCKET_MANIFEST", str(tmp_path / "manifest.json"))
    monkeypatch.setenv("KENKUI_POCKET_VOICE_PROMPT", str(tmp_path / "prompt.wav"))
    assert (
        production_bindings_from_environment("approved-narrator").renderer is not None
    )


@pytest.mark.pocket_real
def test_real_fixture_epub_renders_playable_m4b(tmp_path):
    result = (
        epub(FIXTURE_EPUB)
        .assign_voice("approved-narrator")
        .tts()
        .write(tmp_path / "book.m4b")
    )
    assert result.output.exists()
    assert result.stats.rendered_chapters > 0
```

- [ ] **Step 3: Implement manifest validation and rights-bearing voice metadata**

Require an exact manifest schema/version, hashes, model revision, voice ID, provenance, license ID, and commercial-use flag. Resolve only explicit local paths/environment values; do not download models or prompts. Keep rejected, missing, mismatched, or unapproved assets mapped to the existing stable unavailable error.

- [ ] **Step 4: Run bounded renderer tests**

Run: `uv run pytest tests/test_production_manifest_branches.py tests/test_pocket_tts_real.py -m pocket_real --no-cov`

Expected: the manifest tests pass and the real test passes only in an authorized local environment; ordinary CI retains deterministic fake-renderer coverage and does not need licensed assets.

- [ ] **Step 5: Commit**

```bash
git add pyproject.toml src/kenkui/_tts src/kenkui/voices.py README.md docs/pocket-tts-adapter.md tests
git commit -m "feat: activate approved Pocket-TTS renderer"
```

### Task 2: Finish the specified core V1 public contract

**Files:**

- Modify: `src/kenkui/pipeline.py`, `src/kenkui/api.py`, `src/kenkui/inspection.py`, `src/kenkui/errors.py`, `src/kenkui/__init__.py`
- Create: `src/kenkui/validation.py`, `src/kenkui/observability.py`, `src/kenkui/voices/registry.py`
- Modify: `tests/test_pipeline.py`, `tests/test_epub.py`, `tests/test_planning.py`, `tests/test_execution.py`

**Interfaces:**

- Consumes: immutable source and operation values already used by the planner.
- Produces: `Pipeline.inspect() -> BookInspection` independent of rendering intent; public `list_voices()`/`get_voice()`; authoritative validation with stable codes; logger helpers that never configure global handlers.

- [ ] **Step 1: Write failing contract tests**

```python
def test_source_only_pipeline_can_inspect_fixture_epub():
    inspection = epub(FIXTURE_EPUB).inspect()
    assert inspection.chapters


def test_voice_registry_returns_enabled_licensed_metadata():
    voice = get_voice("fixture-voice")
    assert voice.id == "fixture-voice"
    assert voice.provenance is not None
```

- [ ] **Step 2: Make inspection source-derived**

Keep source readability/format and selected-chapter validation in `inspect()`, but remove `assign_voice`/`tts` preconditions from the inspection path. Preserve lazy construction and stable chapter IDs. Move reusable inexpensive checks into `validation.py`; `write()` still requires a complete renderable single-voice intent.

- [ ] **Step 3: Add public registry and structured logging boundary**

Expose immutable `Voice` records from a bundled/local registry without provider/network discovery. Add module-level loggers and structured context fields at parse, planning, cache, rendering, encoding, and terminal-error boundaries. Never call `basicConfig()` or emit source text/paths.

- [ ] **Step 4: Run core contract tests**

Run: `uv run pytest tests/test_pipeline.py tests/test_epub.py tests/test_planning.py tests/test_execution.py`

Expected: source-only inspection, immutable branching, stable IDs, validation, deterministic plans, events, cancellation, and `ExecutionStats` all pass.

- [ ] **Step 5: Commit**

```bash
git add src/kenkui tests README.md docs
git commit -m "feat: complete core v1 inspection and voice contract"
```

### Task 3: Prove the local single-voice vertical slice

**Files:**

- Modify: `src/kenkui/_execution/coordinator.py`, `src/kenkui/_execution/process_pool.py`, `src/kenkui/_audio/m4b.py`, `src/kenkui/_execution/cache.py`
- Modify: `tests/test_process_execution.py`, `tests/test_ffmpeg_audio.py`, `tests/test_cache.py`, `tests/test_cache_branches.py`, `tests/test_native_ffmpeg.py`
- Create: `tests/integration/test_single_voice_pipeline.py`

**Interfaces:**

- Consumes: `Pipeline.write(output, on_event, cancel, workers)` and approved production bindings.
- Produces: one valid, atomically published M4B with ordered chapters, events, cancellation-safe failure behavior, cache equivalence, and non-inflated statistics.

- [ ] **Step 1: Write the end-to-end acceptance test**

```python
def test_fixture_epub_to_single_voice_m4b(tmp_path):
    events = []
    result = (
        epub(FIXTURE_EPUB)
        .assign_voice("fixture-voice")
        .tts()
        .write(tmp_path / "fixture.m4b", on_event=events.append, workers=1)
    )
    assert probe_m4b(result.output).chapter_titles == ("One", "Two")
    assert result.stats.normalized_speech_characters > 0
    assert type(events[-1]).__name__ == "Completed"
```

- [ ] **Step 2: Verify failure and concurrency boundaries**

Add tests for `workers=1` versus `workers=2` preserving chapter/audio order, cancellation before encoding leaving no output, callback failure pre-commit leaving no output, and a warm cache producing equivalent audio metadata/statistics to a cold cache.

- [ ] **Step 3: Correct only observed orchestration gaps**

Retain spawned processes, plan-order emission, atomic output publication, bounded diagnostics, and private cache ownership. Do not add a public cache path, server ID, or remote scheduling control to `Pipeline`.

- [ ] **Step 4: Run the vertical suite**

Run: `KENKUI_RUN_NATIVE=1 uv run pytest --no-cov tests/integration/test_single_voice_pipeline.py tests/test_process_execution.py tests/test_ffmpeg_audio.py tests/test_cache.py tests/test_native_ffmpeg.py`

Expected: real fixture output passes FFmpeg probe/decode, failure paths publish nothing, and cached/uncached results agree semantically.

- [ ] **Step 5: Commit**

```bash
git add src/kenkui tests
git commit -m "test: prove EPUB single-voice M4B vertical slice"
```

### Task 4: Establish cross-repository contracts and create `kenkui-server`

**Files:**

- Create repository: `../kenkui-server/`
- Create: `pyproject.toml`, `src/kenkui_server/main.py`, `src/kenkui_server/app.py`, `src/kenkui_server/config.py`, `src/kenkui_server/observability.py`, `tests/conftest.py`
- Create: checked OpenAPI contract artifact consumed by `kenkui-web`

**Interfaces:**

- Consumes: public `kenkui` package only.
- Produces: versioned `/v1` OpenAPI document, shared normalized error envelope, capabilities DTO, correlation/logging conventions, and repeatable test/tooling commands.

- [ ] **Step 1: Create server package and CI gates**

Configure Python 3.11–3.13, FastAPI/ASGI, Pydantic, pytest, Ruff, mypy, migrations, and a local-only deterministic test setup. Use AGPL-3.0-only licensing and a minimal README that links the normative specifications.

- [ ] **Step 2: Write failing health/capability contract tests**

```python
def test_local_capabilities_are_versioned(client):
    response = client.get("/v1/capabilities")
    assert response.status_code == 200
    assert response.json()["apiVersion"] == "1"
    assert response.json()["billing"]["mode"] == "unmetered"
    assert response.json()["sourceFormats"] == ["epub"]
```

- [ ] **Step 3: Implement app factory and normalized errors**

Implement an app factory with `/v1/health`, `/v1/capabilities`, request-ID propagation, one `{ "error": { "code", "message", "requestId", "details" } }` failure schema, and structured module logging. Local mode binds loopback by default and advertises `auth.mode=none`, `billing.mode=unmetered`, EPUB, M4B, and single casting.

- [ ] **Step 4: Run server foundation tests**

Run: `uv run pytest tests/test_health.py tests/test_capabilities.py && uv run mypy && uv run ruff check .`

Expected: OpenAPI exposes only `/v1` routes, health and capabilities work, and all failures carry a request ID.

- [ ] **Step 5: Commit**

```bash
git add .
git commit -m "feat: scaffold versioned kenkui server contract"
```

### Task 5: Implement durable local server state and pure Job semantics

**Files:**

- Create: `src/kenkui_server/jobs/models.py`, `src/kenkui_server/jobs/transitions.py`, `src/kenkui_server/jobs/pipeline.py`
- Create: `src/kenkui_server/storage/database.py`, `src/kenkui_server/storage/repositories.py`, `src/kenkui_server/storage/migrations/`
- Create: `tests/test_transitions.py`, `tests/test_job_spec.py`, `tests/test_sqlite_repositories.py`, `tests/test_pipeline_reconstruction.py`

**Interfaces:**

- Consumes: server DTOs and public `kk.Pipeline` methods.
- Produces: immutable `JobSpec`, `Job`, `JobEvent`, `Progress`, pure `transition(job, event)`, SQLite/WAL durable repositories, and `pipeline_from_job(spec, source) -> kk.Pipeline`.

- [ ] **Step 1: Write pure transition/reconstruction tests**

```python
def test_queued_cancel_is_terminal():
    assert transition(queued_job(), CancelRequested()).status is JobStatus.CANCELLED


def test_job_spec_reconstructs_single_voice_pipeline(source_path):
    pipeline = pipeline_from_job(single_voice_spec(), source_path)
    assert pipeline.inspect().chapters
```

- [ ] **Step 2: Define immutable models and validators**

Implement `JobSpec(source_id, chapters, casting, tts, output)` without owner, billing, retry, compute, provider, storage, or credential fields. Persist stable chapter IDs, never UI indexes. Add deterministic invalid-transition codes and separate row mapping from domain values.

- [ ] **Step 3: Add SQLite/WAL repositories and migrations**

Persist assets, inspections, jobs, job events, dispatches, and artifacts. Make snapshot rows authoritative, `(job_id, sequence)` unique, and every update guarded against stale concurrent writes. Do not use an in-memory Job dictionary or queue as durable truth.

- [ ] **Step 4: Run repository tests**

Run: `uv run pytest tests/test_transitions.py tests/test_job_spec.py tests/test_sqlite_repositories.py tests/test_pipeline_reconstruction.py`

Expected: restart-safe rows round-trip, invalid transitions fail stably, and a `JobSpec` reconstructs the same public pipeline semantics.

- [ ] **Step 5: Commit**

```bash
git add src tests
git commit -m "feat: add durable local Job domain and repositories"
```

### Task 6: Deliver the local server API, dispatcher, worker, and artifact lifecycle

**Files:**

- Create: `src/kenkui_server/api/assets.py`, `src/kenkui_server/api/voices.py`, `src/kenkui_server/api/jobs.py`, `src/kenkui_server/api/billing.py`, `src/kenkui_server/api/schemas.py`
- Create: `src/kenkui_server/compute/base.py`, `src/kenkui_server/compute/local.py`, `src/kenkui_server/jobs/dispatcher.py`, `src/kenkui_server/worker.py`, `src/kenkui_server/storage/assets.py`
- Create: API/integration tests for upload, inspection, preflight, Jobs, SSE, cancellation, artifact authorization, and restart.

**Interfaces:**

- Consumes: local repositories, `pipeline_from_job`, `kk.Pipeline.write()`, and the unmetered admission policy.
- Produces: all local `/v1` endpoints, durable dispatch rows, `LocalJobRunner`, SSE snapshots/history, authorized local artifact access.

- [ ] **Step 1: Write API acceptance tests**

```python
def test_local_epub_job_lifecycle(client, fixture_epub):
    asset = upload_asset(client, fixture_epub)
    inspection = client.get(f"/v1/assets/{asset['id']}/book").json()
    job = client.post(
        "/v1/jobs", json=single_voice_spec(asset["id"], inspection)
    ).json()
    assert wait_for_terminal_job(client, job["id"])["status"] == "completed"
    assert client.get(f"/v1/jobs/{job['id']}/artifact").status_code == 200
```

- [ ] **Step 2: Implement assets, inspection, voices, and preflight**

Store source/artifact bytes behind a local AssetStore with private paths. Inspect via public `Pipeline.inspect()` in a bounded executor. Return only registry-derived voices. Preflight validates asset ownership, stable chapter IDs, voice policy, and normalized character count; it creates neither a Job nor billing reservation.

- [ ] **Step 3: Implement Job admission, local execution, SSE, and cancellation**

Support idempotency keys on create. Commit queued Job plus dispatch record before process start. The worker claims an attempt, polls durable cancellation, maps Kenkui events to monotonic server events/snapshot progress, writes the artifact once, and terminalizes through conditional transactions. SSE is incremental; reconnect correctness comes from `GET /v1/jobs/{id}`.

- [ ] **Step 4: Run local end-to-end tests**

Run: `uv run pytest tests/api tests/integration/test_local_job_lifecycle.py tests/integration/test_restart_recovery.py`

Expected: upload → inspection → preflight → submitted Job → SSE → M4B artifact works; cancellation is idempotent; restart loses no Job truth; duplicate create does not create a second Job.

- [ ] **Step 5: Commit**

```bash
git add src tests
git commit -m "feat: deliver local durable audiobook server"
```

### Task 7: Create `kenkui-web` and prove local browser completion

**Files:**

- Create repository: `../kenkui-web/`
- Create: `src/main.tsx`, `src/app.tsx`, `src/router.tsx`, `src/api/client.ts`, `src/api/generated/`, `src/api/errors.ts`, `src/api/events.ts`
- Create: `src/pages/{jobs,new-job,job,billing,sign-in}.tsx`, `src/components/{file-upload,chapter-selection,casting,voice-select,tts-settings,output-settings,job-progress,error-message}.tsx`
- Create: unit/component and Playwright tests; configure server static-bundle hook.

**Interfaces:**

- Consumes: generated `/v1/openapi.json` types and `KenkuiServerClient` only.
- Produces: a static React/Vite SPA that renders server-authoritative Jobs/assets/billing and works both as a standalone trusted-origin bundle and when served by local server.

- [ ] **Step 1: Generate types and write API client tests**

```ts
it("refetches the Job snapshot after SSE reconnect", async () => {
  const stream = client.events("job-1");
  await stream.onDisconnect();
  expect(mock.getJob).toHaveBeenCalledWith("job-1");
});
```

- [ ] **Step 2: Implement the single network boundary and capability shell**

Generate OpenAPI DTO types and route all HTTP/SSE through `KenkuiServerClient`. Load capabilities before feature UI; do not branch on local/cloud hostname. Keep only routes, form drafts, display preferences, and SSE display state in the browser.

- [ ] **Step 3: Implement the linear creation and Job-progress flow**

Build Source → Chapters → Casting → Synthesis → Output → Review. Upload source, render server inspection, retain stable chapter IDs, select one voice, request server preflight, reuse one idempotency key through retries, then navigate to Job detail. Job detail refetches snapshot after bounded SSE reconnect, exposes cancellation honestly, and obtains artifact access only from the server.

- [ ] **Step 4: Run browser and local E2E tests**

Run: `npm run test && npx playwright test`

Expected: a local server-hosted SPA uploads the fixture EPUB, preflights, submits, observes progress, downloads the M4B, recovers from an SSE disconnect, and never imports Kenkui/Pocket-TTS into browser tests.

- [ ] **Step 5: Commit**

```bash
git add .
git commit -m "feat: add capability-driven Kenkui web client"
```

### Task 8: Add hosted persistence, storage, retention, auth, and prepaid billing

**Files:**

- Modify: `kenkui-server/src/kenkui_server/storage/repositories.py`, `kenkui-server/src/kenkui_server/storage/assets.py`, `kenkui-server/src/kenkui_server/config.py`
- Create: `kenkui-server/src/kenkui_server/auth/{base,workos}.py`, `kenkui-server/src/kenkui_server/billing/{models,pricing,service,stripe}.py`, PostgreSQL migrations, S3/R2 adapter, retention worker
- Create: PostgreSQL, fake WorkOS/Stripe, idempotency, authorization, and retention tests

**Interfaces:**

- Consumes: provider-neutral `AuthBackend`, `PaymentProvider`, repository, AssetStore, and JobRunner protocols.
- Produces: PostgreSQL/R2 hosted adapters, WorkOS session-backed identity mapping, one-credit-per-1,000-character reservation/ledger flow, verified Stripe webhooks, idempotent retention.

- [ ] **Step 1: Write transaction and payment tests**

```python
def test_duplicate_completed_attempt_settles_one_authorization(postgres):
    authorization = reserve(job, credits=12)
    finalize_success(job)
    finalize_success(job)
    assert settlements_for(authorization.id) == 1


def test_duplicate_stripe_event_credits_account_once(client, signed_event):
    client.post("/v1/billing/webhooks/stripe", content=signed_event)
    client.post("/v1/billing/webhooks/stripe", content=signed_event)
    assert account_balance() == PURCHASED_CREDITS
```

- [ ] **Step 2: Implement PostgreSQL/R2 and retention adapters**

Use ordinary SQL migrations and repository row mapping. Store sources/artifacts in R2 through an S3-compatible adapter; never expose permanent keys. Make worker final upload direct. Implement idempotent retention decisions: temporary files within 24h of terminal state, sources within 24h after dependent Jobs terminal, artifacts after 30 days by default.

- [ ] **Step 3: Implement hosted auth and billing policy**

Use WorkOS/AuthKit behind `AuthBackend`, map provider identities to internal UUIDs, and keep browser secrets out of localStorage. Implement credit calculation as `ceil(normalized_speech_characters / 1000)`, reserve before queued admission, settle only success, release failure/cancel, append ledger entries, and authenticate/idempotently persist Stripe events.

- [ ] **Step 4: Run hosted-adapter tests**

Run: `uv run pytest tests/postgres tests/billing tests/auth tests/storage tests/retention`

Expected: concurrent/replayed finalization makes one artifact and settlement, account/resource authorization is enforced, expired data is removed once, and all provider tests use fakes by default.

- [ ] **Step 5: Commit**

```bash
git add src tests migrations
git commit -m "feat: add hosted storage identity and credit billing adapters"
```

### Task 9: Implement Modal dispatch and deploy the cloud control plane

**Files:**

- Create/modify: `kenkui-server/src/kenkui_server/compute/modal.py`, dispatch retry tests, deployment configuration in a private `kenkui-infra` repository
- Configure: Cloudflare DNS/TLS/WAF/static SPA/R2; Railway FastAPI/PostgreSQL; Modal worker; WorkOS; Stripe
- Create: staging smoke-test/runbook automation and backup/restore verification

**Interfaces:**

- Consumes: persisted `job_id`/attempt context plus secret-injected database/object-store configuration.
- Produces: provider-neutral `HostedJobRunner.submit(job_id)`, ephemeral Modal workers, staging/production deployment, operational correlation across services.

- [ ] **Step 1: Write stale-attempt and retry tests**

```python
def test_stale_modal_attempt_cannot_replace_newer_terminal_attempt():
    mark_attempt_retrying(job_id, attempt=1)
    complete_attempt(job_id, attempt=2)
    assert (
        finalize_attempt(job_id, attempt=1, outcome="completed") is FinalizationIgnored
    )
```

- [ ] **Step 2: Implement durable claim/submit/retry behavior**

Claim dispatch rows with database concurrency controls, submit idempotently by Job/attempt, use bounded infrastructure retry without a new billing authorization, and reject stale worker finalization. The Modal worker reconstructs the Pipeline, bridges cancellation, uploads one artifact, and finalizes through server repositories.

- [ ] **Step 3: Configure provider resources without leaking names into models**

Deploy static web and R2 at Cloudflare, FastAPI/PostgreSQL at Railway, and worker at Modal. Use WorkOS and Stripe secrets only in provider secret stores. Keep local, staging, and production databases, buckets, identities, keys, and Modal configuration separate. Enable backups/PITR and TLS/WAF/rate limits.

- [ ] **Step 4: Run staging operations checks**

Run the staging end-to-end scenario: signed-in user buys test credits, uploads EPUB, preflights, submits a Job, observes SSE, downloads an authorized artifact, then repeats webhook/cancel/retry paths. Verify request/job/attempt correlation across Railway and Modal logs and validate backup restore plus retention cleanup.

- [ ] **Step 5: Commit deployment code and record private operations state**

```bash
git add src tests
# Commit provider configuration only to the private infrastructure repository.
git commit -m "feat: add Modal job runner and cloud deployment adapters"
```

### Task 10: Add character-voice mode only after the hosted single-voice slice is stable

**Files:**

- Create: `kenkui/src/kenkui/characters/{__init__,llm,entities,infer,quotes,attribution}.py`, `kenkui/src/kenkui/voices/assign.py`
- Modify: `kenkui/src/kenkui/pipeline.py`, `kenkui/src/kenkui/_domain/operations.py`, planner/execution modules, `kenkui-server` JobSpec mapping/capabilities, `kenkui-web` casting UI
- Create: fake LiteLLM adapter tests, character normalization tests, server allowlist tests, browser capability tests

**Interfaces:**

- Consumes: `Pipeline.infer_characters(model)`, `Pipeline.attribute_quotes(model)`, immutable Book/Segment values, LiteLLM, and server-provided allowed model IDs.
- Produces: one shared `VoicePlan` and renderer for single and character casting; `castingModes` includes `characters` only for enabled deployments.

- [ ] **Step 1: Write deterministic fake-provider tests**

```python
def test_character_pipeline_normalizes_fake_llm_output():
    book = infer_characters(fixture_book(), model="fake/model", llm=FakeLiteLLM())
    assert [character.id for character in book.characters] == ["elizabeth-bennet"]


def test_character_casting_reuses_one_voice_plan_renderer():
    plan = assign_voices(character_book, narrator="narrator", characters="auto")
    assert plan.narrator_voice_id == "narrator"
```

- [ ] **Step 2: Implement direct, narrow LiteLLM integration**

Call LiteLLM directly with model identifiers in Pipeline semantic intent and credentials resolved at execution. Add bounded retry/backoff and structured JSON validation in `characters/llm.py`; normalize provider output into immutable deterministic values. Do not add a provider framework or LiteLLM Proxy.

- [ ] **Step 3: Implement inference, attribution, and shared rendering**

Implement mention extraction/clustering, quote extraction, bounded-context speaker attribution, explicit Unknown fallback, and automatic casting. Materialize a `VoicePlan` consumed by the existing segmentation/render/encode path; do not add a second multi-voice workflow, Job type, or billing model.

- [ ] **Step 4: Expose only capability-approved character mode**

Validate server model/voice allowlists, map character `JobSpec` to public Pipeline calls, advertise capability only when configured, and conditionally show the web casting controls. The browser never runs inference or sends credentials.

- [ ] **Step 5: Run character acceptance tests and commit**

Run: `uv run pytest tests/characters tests/test_pipeline.py && uv run pytest tests/server_character tests/web_character`

Expected: fake providers cover all default tests; single voice remains unchanged; character Jobs use the same events, cancellation, renderer, output, and credit character accounting.

```bash
git add src tests
git commit -m "feat: add capability-gated character voice mode"
```

## Coverage review

- Core requirements map to Tasks 1–3: renderer activation, source-only inspection, public voice metadata, immutable Pipeline, EPUB/single-voice/M4B, events, cancellation, cache, logs, stable errors, and the required core vertical test.
- Server requirements map to Tasks 4–6 and 8–9: `/v1`, capabilities, durable Jobs/dispatch/assets/artifacts, preflight, SSE, cancellation, authorization, restart/idempotency, hosted adapters, retention, billing, and observability.
- Web requirements map to Task 7 and Task 10: generated OpenAPI types, one client boundary, capability-driven SPA, creation/progress/cancellation/artifact/billing behavior, and character UI gating.
- Cloud requirements map to Tasks 8–9: PostgreSQL/R2 adapters, WorkOS, Stripe, Modal, Cloudflare/Railway deployment, correlation, backups, retention, and provider-neutral domain models.
- The plan intentionally excludes explicitly demand-driven post-V1 formats, custom voices, permanent library, and native packaging. Add them only after their triggering demand and a separate approved design/plan.

## Consistency review

- `Pipeline` and `JobSpec` remain distinct in every task; Task 5 and Task 6 use explicit reconstruction rather than serialization.
- Render parallelism remains inside `kenkui`; many-Job scheduling/retry remains server-owned in Tasks 6 and 9.
- The 1,000-character credit definition is used only by server billing in Task 8; core exposes technical counts and does not know prices.
- Cloud provider names occur only in adapter/deployment tasks, never in Pipeline or JobSpec interfaces.
- The core production activation task identifies the only external prerequisite that cannot be implemented without project-owner rights/asset decisions.
