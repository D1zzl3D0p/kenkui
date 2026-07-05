# K’enkui Cloud Runtime Repo Split Architecture Plan

> **For Hermes:** Planning only. Do not move files, create repos, push, or implement cloud runtime until the open questions are answered and the plan is explicitly approved.

**Goal:** Move the Modal/cloud runtime work out of `kenkui` core into a separate repository/package while preserving `kenkui` as the clean local library/runtime and creating a path toward deployable Modal + Cloudflare R2 + OAuth infrastructure.

**Architecture:** Use a separate cloud-extension repo that depends on `kenkui`, registers execution providers through `kenkui` extension hooks, and owns all Modal/R2/OAuth/control-plane concerns. Keep only generic provider hooks and stable library models in `kenkui`; place deployable cloud code in the new repo.

**Tech Stack:** Python 3.12, `uv`, pytest, ruff; Modal for GPU/CPU compute; Cloudflare R2 for artifacts; likely Cloudflare Workers + D1/Durable Objects for OAuth/job control if hosted multi-user mode is required; GitHub Actions for CI; entry points for plugin registration.

---

## 0. Current State Observed

- `kenkui` is on `main`, ahead of `origin/main` by one commit:
  - `f3631fd feat: add optional modal runtime hooks`
- `.hermes/` is untracked and contains plan files.
- `src/kenkui/modal_runtime/` currently exists in core and contains scaffolding only:
  - `__init__.py`
  - `artifacts.py`
  - `contracts.py`
  - `errors.py`
  - `nlp.py`
  - `settings.py`
  - `tts.py`
- `src/kenkui/services/runtime_service.py` currently imports `kenkui.modal_runtime` lazily when `modal_enabled` is true.
- `AppConfig` currently has Modal-specific fields in `src/kenkui/models/config.py`.
- AGENTS.md explicitly says:
  - local execution is the only built-in path today;
  - remote execution should remain extension-hook based;
  - do not add cloud auth, billing, farm scheduling, or remote runtime concerns to core.

That means the current scaffolding is useful, but it should be extracted before real Modal/R2/OAuth work begins.

---

## 1. The Central Product Question

Before architecture, answer this:

**Are we building a bring-your-own-cloud extension, a hosted multi-user cloud service, or both?**

This one answer changes almost every tool choice.

### Option A — BYO Cloud Extension

Users install `kenkui-modal`, configure their own Modal account and R2 bucket, then run cloud jobs from their local `kenkui`/`kengui` runtime.

**Pros:**
- Much simpler.
- Minimal OAuth or no OAuth.
- User owns data/costs/secrets.
- Good first milestone.
- Easier to align with `kenkui` as local-first library.

**Cons:**
- Setup burden on users.
- Harder for non-technical users.
- No centralized billing/quotas/job dashboard.
- Desktop app still needs cloud credentials or a local helper.

### Option B — Hosted Multi-User Cloud Runtime

You operate a service. Users sign in via OAuth, upload ebooks/voice assets to R2, jobs run on your Modal app, and clients poll/stream status from your control plane.

**Pros:**
- Best UX.
- Centralized job control, status, quota, retention, support.
- Easier clients: login + submit job.
- Supports future billing.

**Cons:**
- Requires OAuth, user/job database, R2 authorization model, audit/delete, cost controls, abuse prevention.
- More operational responsibility.
- Higher security/privacy burden because ebooks and voice samples are sensitive.

### Option C — Hybrid

A shared codebase supports both:
- BYO mode: local client talks directly to Modal/R2 with user credentials.
- Hosted mode: client talks to control plane; control plane talks to R2/Modal.

**Pros:**
- Best long-term flexibility.
- BYO can ship first, hosted can follow.

**Cons:**
- More abstractions up front.
- Risk of over-engineering if hosted service is not imminent.

### Current Recommendation

Build the repo as a **hybrid-capable monorepo**, but implement in this order:

1. Extract cloud runtime package into a new repo with BYO-safe boundaries.
2. Add R2 artifact store in that package.
3. Add Modal functions and direct Modal client.
4. Only then add hosted OAuth/control-plane service if still needed.

This avoids trapping cloud auth and storage inside `kenkui`, but does not force us to build a full hosted platform before proving Modal execution works.

---

## 2. Architecture Iteration and Critique

### Iteration 1 — Keep everything in `kenkui`

**Idea:** Leave `src/kenkui/modal_runtime/` in core and add real Modal, R2, and OAuth there.

**Why tempting:**
- Fastest path from current code.
- Fewer repos.
- Fewer packaging concerns.

**Critique:**
- Directly violates AGENTS.md product direction.
- Couples `kenkui` core to Modal, R2, OAuth, job control, credentials, and deployment policy.
- Makes bare/local installs harder.
- Creates a dependency/config surface that external clients inherit whether they want it or not.
- OAuth and R2 retention policy do not belong in an ebook conversion library.

**Verdict:** Reject.

---

### Iteration 2 — Create `kenkui-modal` as a single Python extension repo

**Idea:** Move `src/kenkui/modal_runtime/` into `kenkui-modal`, a Python package that depends on `kenkui` and registers providers.

**Why good:**
- Clean extension boundary.
- Easy packaging and testing.
- Good for BYO Modal mode.
- Keeps real Modal SDK/images out of core.

**Critique:**
- Name is too narrow if OAuth/R2/control-plane code grows.
- R2 is not strictly “Modal”; it is cloud artifact infrastructure.
- OAuth/control plane should not be buried inside a Modal compute package.
- If hosted multi-user is planned, this package will become a grab-bag.

**Verdict:** Good implementation package, but not sufficient as the top-level architecture if OAuth/R2/control-plane are real requirements.

---

### Iteration 3 — Create `kenkui-cloud` monorepo with multiple packages/services

**Idea:** New repo `kenkui-cloud` contains separate components:

```text
kenkui-cloud/
  packages/
    kenkui-modal/          # Python Modal provider package
    kenkui-cloud-client/   # optional Python client for hosted control plane
  services/
    control-plane/         # OAuth, job state, presigned R2 URLs
  infra/
    cloudflare/            # R2/D1/DO setup notes or Terraform/OpenTofu later
  docs/
```

**Why good:**
- Separates compute provider from control-plane/auth concerns.
- Can start with only `packages/kenkui-modal` and leave `services/control-plane` as docs/stub.
- Gives room for hosted mode without contaminating `kenkui` core.
- Makes OAuth/R2/security docs first-class.

**Critique:**
- More complex repository layout.
- Python + TypeScript toolchains if Workers are used.
- Overkill if you only want a local BYO Modal plugin.
- Requires disciplined boundaries to avoid monorepo sprawl.

**Verdict:** Best long-term shape if OAuth/R2 are in scope, but implementation should be staged so only one package is real at first.

---

### Iteration 4 — Final staged architecture

**Decision:** Create a new repo named `kenkui-cloud` as the umbrella, with `packages/kenkui-modal` as the first implemented package. Do not implement OAuth/control-plane immediately, but reserve clear paths for it.

**Why this survives critique:**
- Avoids cloud code in core.
- Avoids forcing hosted control-plane work before proving Modal/R2 execution.
- Avoids renaming later when `kenkui-modal` becomes more than Modal.
- Allows `kenkui-modal` to be published independently if desired.
- Keeps `kenkui` core extensible through entry points.

---

## 3. Proposed Repository Split

### New repo

Recommended name: `kenkui-cloud`

Initial tree:

```text
kenkui-cloud/
  README.md
  LICENSE
  .gitignore
  .github/
    workflows/
      ci.yml
  docs/
    architecture.md
    local-byo-modal.md
    hosted-control-plane.md
    security-and-retention.md
  packages/
    kenkui-modal/
      pyproject.toml
      README.md
      src/
        kenkui_modal/
          __init__.py
          artifacts.py
          contracts.py
          errors.py
          nlp.py
          settings.py
          tts.py
          # later:
          # app.py
          # client.py
          # r2.py
      tests/
        test_modal_artifacts.py
        test_modal_contracts.py
        test_modal_providers.py
        test_modal_runtime_config.py
        test_modal_runtime_registration.py
  services/
    control-plane/
      README.md
      ADR-0001-control-plane-options.md
  infra/
    README.md
```

### `kenkui` core after split

Target state:

```text
kenkui/
  src/kenkui/services/runtime_service.py       # generic plugin loading only
  src/kenkui/services/execution_service.py     # provider hook remains
  src/kenkui/nlp/providers/_factory.py         # NLP hook remains
  src/kenkui/models/common.py                  # TTS/NLP enum values remain for compatibility
  src/kenkui/models/config.py                  # keep only minimal remote/plugin config if possible
```

Remove from core:

```text
src/kenkui/modal_runtime/
```

Migrate to extension:

```text
src/kenkui/modal_runtime/*
→ kenkui-cloud/packages/kenkui-modal/src/kenkui_modal/*
```

### Naming rationale

- Top-level repo: `kenkui-cloud` because OAuth/R2/control-plane are not Modal-specific.
- Python package: `kenkui-modal` distribution, `kenkui_modal` import name, because the first concrete runtime is Modal.
- Avoid `kenkui.modal_runtime` namespace in the external repo to prevent namespace-package confusion and accidental import collisions with core.

---

## 4. Core Extension Mechanism Recommendation

### Current mechanism

`src/kenkui/services/runtime_service.py` currently does:

```python
from kenkui.modal_runtime import register_modal_runtime
```

This works only while Modal code lives in core.

### Recommended mechanism

Use Python package entry points.

In `kenkui-cloud/packages/kenkui-modal/pyproject.toml`:

```toml
[project.entry-points."kenkui.runtime_providers"]
modal = "kenkui_modal:register_modal_runtime"
```

In `kenkui` core, `runtime_service.py` should discover installed providers with `importlib.metadata.entry_points()` and call the provider whose name matches configured runtime mode.

### Why entry points

- No hard import path from core to extension.
- Works with pip/uv-installed packages.
- Allows future `kenkui-vllm`, `kenkui-runpod`, etc.
- Keeps `kenkui` open/closed: new providers do not require editing core.

### Critique of entry points

- Slightly more complex than explicit imports.
- Debugging missing entry points can be confusing.
- Editable installs must be correct.

### Mitigation

- Add `kenkui runtime providers` or `kenkui doctor` later to list discovered providers.
- Good error message:
  - “Runtime mode 'modal' requested but provider entry point 'modal' was not found. Install `kenkui-modal` or disable modal mode.”

---

## 5. Tool Choices and Why

### Python package/dependency management: `uv`

**Use for:** Python package builds, lockfiles, editable installs, test runs.

**Why:**
- Already used by `kenkui`.
- Fast and deterministic.
- Good workspace support if we keep multiple Python packages.

**Critique:**
- If the repo also includes Cloudflare Workers TypeScript, `uv` does not manage that side.

**Mitigation:**
- Use `uv` for Python packages and `pnpm` for Workers only if/when Workers are added.

---

### Compute: Modal

**Use for:**
- GPU TTS rendering with `pocket-tts`.
- CPU/high-memory spaCy and BookNLP processing.
- Optional post-processing/M4B assembly in remote workers.

**Why:**
- Good fit for bursty ML/audio workloads.
- Container images can isolate heavy local dependencies.
- GPU resources can be configured per function.
- Avoids requiring users’ laptops to have local ML/audio stacks.

**Critique:**
- Long-running audiobook jobs may stress timeout/cancellation semantics.
- Cold starts and model downloads can dominate small jobs.
- Modal is not the full job-control plane.

**Mitigation:**
- Keep job manifests/status outside Modal.
- Cache models in Modal images/volumes where possible.
- Use job-level remote orchestration rather than thousands of tiny function calls.

---

### Artifact storage: Cloudflare R2

**Use for:**
- input job bundles
- source ebooks if needed
- custom voice assets
- intermediate WAV bundles if retained
- final M4B artifacts
- job manifests/log summaries if no DB yet

**Why:**
- S3-compatible API.
- No egress fees in many common patterns.
- Good fit with Cloudflare Workers if hosted mode is needed.
- Better durable artifact store than Modal volumes for cross-service access.

**Critique:**
- R2 is eventually consistent enough that manifest design matters.
- Multipart upload and presigned URL handling add complexity.
- Object key authorization must be designed carefully to avoid cross-user leaks.

**Mitigation:**
- Use strict object key layout:
  - `env/{env}/users/{user_id}/jobs/{job_id}/input/...`
  - `env/{env}/users/{user_id}/jobs/{job_id}/output/...`
- Never let clients choose arbitrary object keys.
- Store job manifest with checksums and expected keys.

---

### R2 Python access: `boto3` or `aioboto3`

**Recommendation:** Start with `boto3`.

**Why:**
- Stable S3-compatible API.
- Already in the previous `modal` extra.
- Works inside Modal functions and local provider clients.

**Critique:**
- Synchronous; not ideal for high-throughput API servers.

**Mitigation:**
- For Python Modal workers, sync is fine.
- For Cloudflare Workers control plane, use R2 bindings or JS signing instead.

---

### Hosted control plane: Cloudflare Workers + Hono + D1/Durable Objects

**Use for hosted mode only.**

**Why Workers:**
- Natural pairing with R2.
- Easy edge OAuth redirects and signed upload/download endpoints.
- Low operational overhead.
- D1/Durable Objects can track job state without running a server.

**Why Hono:**
- Lightweight TypeScript web framework for Workers.
- Good middleware story.
- Simpler than building a large API framework.

**D1 vs Durable Objects:**
- D1 for durable job/user tables and history.
- Durable Object per job if we need live progress streams, cancellation coordination, or WebSocket/SSE state.
- Start with D1 + polling unless the UI needs realtime.

**Critique:**
- Mixed Python/TypeScript monorepo.
- Workers cannot run Python NLP/TTS; all compute remains in Modal.
- OAuth correctness/security burden is non-trivial.

**Mitigation:**
- Do not implement this first.
- Keep `services/control-plane/README.md` and ADRs until product shape is confirmed.

---

### OAuth provider

Possible choices:

1. **Clerk**
   - Great hosted auth UX, frontend-friendly, supports OAuth providers.
   - Good if `kengui`/web UI need polished login quickly.
   - Vendor dependency.

2. **Auth0**
   - Mature, enterprise-friendly.
   - More configuration overhead.

3. **Cloudflare Access**
   - Great for private/internal access.
   - Poor fit for consumer OAuth unless this is only for you/team.

4. **Supabase Auth**
   - Good if Supabase Postgres is also used.
   - Less natural if we choose Cloudflare D1/R2 stack.

5. **Roll our own OAuth with Workers**
   - Maximum control.
   - Highest risk; avoid unless necessary.

**Recommendation pending answers:**
- For private/personal deployment: Cloudflare Access or no OAuth initially.
- For public/multi-user: Clerk or Auth0; do not roll our own.

---

### Schema/contracts

**Recommendation:** Keep Python Pydantic models in `kenkui-modal` for provider-to-Modal contracts. If a TypeScript control plane is added, emit JSON Schema/OpenAPI and generate TS types.

**Why:**
- Current contracts are already Pydantic.
- Modal functions and provider clients are Python.
- Avoid hand-maintaining two schemas at the start.

**Critique:**
- Generated TS types can lag if CI does not enforce regeneration.

**Mitigation:**
- CI check: generated schema/types are up-to-date.
- Keep Worker API contracts small: job metadata and artifact URLs, not full internal chapter payloads unless needed.

---

### CI/CD

**Use:** GitHub Actions.

Initial Python CI:

```bash
uv run pytest -q
uvx ruff check packages/kenkui-modal/src packages/kenkui-modal/tests
uv build packages/kenkui-modal
```

Later Workers CI:

```bash
pnpm install --frozen-lockfile
pnpm --filter control-plane test
pnpm --filter control-plane wrangler deploy --dry-run
```

Modal integration tests should be opt-in:

```bash
KENKUI_MODAL_INTEGRATION=1 uv run pytest tests/integration -q
```

---

### Infrastructure-as-code

**Start:** documented `wrangler`/Cloudflare setup steps and Modal deploy commands.

**Later:** Terraform/OpenTofu if environments multiply.

**Why not Terraform first:**
- Premature for initial extraction.
- Modal app is code-defined.
- R2/D1/Workers can be bootstrapped manually or via Wrangler initially.

**When to add Terraform/OpenTofu:**
- staging/prod split
- multiple buckets
- strict IAM policies
- GitHub OIDC/deploy automation
- team handoff

---

## 6. Security and Data Questions

These need explicit answers because ebooks and custom voices are sensitive.

### Questions for you

1. Is this cloud runtime only for your personal use, private beta, or public users?
2. Who owns the Modal account and bill: you, each user, or an organization?
3. Who owns the R2 bucket: you, each user, or an organization?
4. Should uploaded ebooks be deleted immediately after final M4B generation?
5. Should intermediate WAVs be retained for resume/debug, or deleted by default?
6. How long should final M4Bs remain downloadable?
7. Do we need per-user storage quotas?
8. Do we need per-user compute quotas or max GPU spend per job/day/month?
9. Are custom voice samples stored, or only used transiently per job?
10. Should users be able to bring their own Hugging Face token / provider keys?
11. Should hosted mode support multiple OAuth providers or just one?
12. Do you already have a domain for OAuth redirects/control plane?
13. Should clients support offline/local-only mode forever as a first-class path?
14. What is the largest expected ebook/audio output size?
15. Is resumability required after process crash/container timeout?
16. Is cancellation best-effort acceptable, or must it stop billing quickly?
17. Do we need streaming progress events, or is polling enough?
18. Should generated audiobooks be encrypted at rest with per-user keys, or is R2 bucket encryption sufficient?
19. Do we need audit logs for artifact access/deletion?
20. Is this intended to integrate primarily with `kengui`, `kentui`, CLI, or all of them?

### Questions I am asking myself

1. What is the minimum extraction that keeps `kenkui` clean without breaking the just-added hooks?
2. Are Modal-specific config fields acceptable in core, or should core only expose generic plugin config?
3. Should `TTSExecutionMode.MODAL` stay in core, or should modes become stringly plugin IDs?
4. Where should remote contracts live if both local provider and control plane need them?
5. Can we support BYO and hosted without designing two separate clients?
6. Should Modal call R2 directly, or should it call the control plane for signed URLs?
7. How do we prevent users from submitting arbitrary R2 keys?
8. Is OAuth needed for BYO mode at all?
9. If we add Cloudflare Workers, are we comfortable with a Python + TypeScript monorepo?
10. How do we test Modal/R2 without making CI expensive or flaky?
11. How do we keep job metadata compatible with local queue semantics?
12. How do we preserve privacy and deletion guarantees under failure?
13. Should the first deployable path require R2, or can it use Modal volumes for a smoke test?
14. Is full remote render mandatory, or should initial Modal only render chapter WAVs?
15. Should post-processing/M4B assembly run on Modal to minimize local specs? Current recommendation: yes.

---

## 7. Final Proposed Plan

### Phase 0 — Decide product shape before moving files

**Objective:** Confirm the target deployment mode and repo naming so we do not design the wrong split.

**Questions to answer:**
- BYO, hosted, or hybrid?
- Repo name: `kenkui-cloud`, `kenkui-modal`, or another name?
- Private or public repo initially?
- GitHub owner/org?
- OAuth provider preference?
- R2 ownership/retention policy?

**Recommendation:**
- New repo: `kenkui-cloud`
- Private initially
- First package: `packages/kenkui-modal`
- Hosted control plane: planned but not implemented yet

**Exit criteria:**
- Written decision summary in `docs/architecture.md`.

---

### Phase 1 — Extract Modal scaffold into new repo, no real cloud implementation

**Objective:** Move existing scaffold out of `kenkui` without adding Modal SDK/R2/OAuth behavior yet.

**New repo files:**

```text
kenkui-cloud/packages/kenkui-modal/src/kenkui_modal/__init__.py
kenkui-cloud/packages/kenkui-modal/src/kenkui_modal/artifacts.py
kenkui-cloud/packages/kenkui-modal/src/kenkui_modal/contracts.py
kenkui-cloud/packages/kenkui-modal/src/kenkui_modal/errors.py
kenkui-cloud/packages/kenkui-modal/src/kenkui_modal/nlp.py
kenkui-cloud/packages/kenkui-modal/src/kenkui_modal/settings.py
kenkui-cloud/packages/kenkui-modal/src/kenkui_modal/tts.py
```

**Move tests:**

```text
kenkui/tests/test_modal_artifacts.py
→ kenkui-cloud/packages/kenkui-modal/tests/test_modal_artifacts.py

kenkui/tests/test_modal_contracts.py
→ kenkui-cloud/packages/kenkui-modal/tests/test_modal_contracts.py

kenkui/tests/test_modal_providers.py
→ kenkui-cloud/packages/kenkui-modal/tests/test_modal_providers.py

kenkui/tests/test_modal_runtime_registration.py
→ kenkui-cloud/packages/kenkui-modal/tests/test_modal_runtime_registration.py
```

**Core `kenkui` cleanup files:**

```text
Remove: src/kenkui/modal_runtime/
Modify: src/kenkui/services/runtime_service.py
Modify: tests/test_modal_runtime_registration.py or replace with generic runtime provider tests
```

**Important design choice:** Use entry points rather than `from kenkui.modal_runtime import ...`.

**Validation:**
- In `kenkui`: `uv run pytest tests/test_nlp_factory.py tests/test_application_service.py tests/test_models.py -q`
- In `kenkui-cloud/packages/kenkui-modal`: `uv run pytest -q`

**Critique:**
- This phase does not make anything deployable.
- But it prevents cloud implementation from growing in the wrong repo.

---

### Phase 2 — Define the cloud artifact contract before R2 implementation

**Objective:** Design the R2 object model and job manifest before writing R2 code.

**Docs to create first:**

```text
kenkui-cloud/docs/artifact-layout.md
kenkui-cloud/docs/job-manifest.md
kenkui-cloud/docs/security-and-retention.md
```

**Proposed R2 key layout:**

```text
env/{environment}/users/{user_id}/jobs/{job_id}/input/job.json
env/{environment}/users/{user_id}/jobs/{job_id}/input/source.{ext}
env/{environment}/users/{user_id}/jobs/{job_id}/input/voices/{voice_id}.safetensors
env/{environment}/users/{user_id}/jobs/{job_id}/work/chapters/{chapter_index}.wav
env/{environment}/users/{user_id}/jobs/{job_id}/output/final.m4b
env/{environment}/users/{user_id}/jobs/{job_id}/manifest.json
env/{environment}/users/{user_id}/jobs/{job_id}/logs/summary.json
```

**Questions:**
- Is `user_id` available in BYO mode? If not, use `users/local` or `accounts/{account_id}`.
- Do we need object encryption metadata?
- Do we need content hashes for every artifact?

**Critique:**
- If we overfit the key layout now, migrations are painful.
- If we under-specify it, auth bugs happen.

**Recommendation:**
- Define manifest schema with a `schema_version` field from day one.

---

### Phase 3 — Add R2 artifact store to `kenkui-modal`

**Objective:** Implement artifact transport in the extension package, not core.

**Files later:**

```text
packages/kenkui-modal/src/kenkui_modal/r2.py
packages/kenkui-modal/tests/test_r2_artifacts.py
packages/kenkui-modal/tests/integration/test_r2_artifacts.py
```

**Tool:** `boto3` against R2 S3-compatible endpoint.

**Env vars:**

```text
KENKUI_R2_ACCOUNT_ID
KENKUI_R2_BUCKET
KENKUI_R2_ACCESS_KEY_ID
KENKUI_R2_SECRET_ACCESS_KEY
KENKUI_R2_ENDPOINT_URL optional override
KENKUI_R2_ENVIRONMENT dev|staging|prod
```

**Testing:**
- Unit tests use `botocore.stub.Stubber` or fake store.
- Integration tests skipped unless `KENKUI_R2_INTEGRATION=1`.

**Critique:**
- R2 credentials in a local desktop process are risky for hosted mode.

**Mitigation:**
- BYO mode can use local credentials.
- Hosted mode should use control-plane presigned URLs or scoped service credentials, not user-visible bucket secrets.

---

### Phase 4 — Add Modal app/functions to `kenkui-modal`

**Objective:** Make deployable Modal compute exist, still without OAuth/control plane.

**Files later:**

```text
packages/kenkui-modal/src/kenkui_modal/app.py
packages/kenkui-modal/src/kenkui_modal/client.py
packages/kenkui-modal/tests/test_modal_client.py
packages/kenkui-modal/tests/integration/test_modal_tts.py
packages/kenkui-modal/tests/integration/test_modal_nlp.py
```

**Functions:**
- `render_tts_job`
- `run_spacy_extraction`
- `run_booknlp_extraction`
- `run_booknlp_attribution`
- `healthcheck`

**Images:**
- TTS image: `pocket-tts`, audio deps, ffmpeg tooling, possibly HF model cache.
- spaCy image: `spacy`, `en_core_web_sm`.
- BookNLP image: `booknlp`, compatible transformer deps.

**Critique:**
- Multiple images increase deployment time and maintenance.

**Mitigation:**
- Start with two images:
  - `tts_image`
  - `nlp_image`
- Split BookNLP later if dependency conflicts force it.

---

### Phase 5 — Decide hosted control plane only after BYO works

**Objective:** Avoid implementing OAuth before the core remote compute path is proven.

**Control plane candidates:**

1. Cloudflare Workers + D1 + R2 + optional Durable Objects
2. FastAPI + Postgres + R2
3. Supabase + Edge Functions + R2/Storage

**Current recommendation:** Cloudflare Workers + D1, because R2 is already Cloudflare and the API can remain lightweight.

**Files later if chosen:**

```text
services/control-plane/package.json
services/control-plane/wrangler.toml
services/control-plane/src/index.ts
services/control-plane/src/auth.ts
services/control-plane/src/jobs.ts
services/control-plane/src/r2.ts
services/control-plane/migrations/*.sql
```

**OAuth provider:** choose Clerk/Auth0 unless this is private-only.

**Critique:**
- Workers + Modal requires cross-platform status/cancel coordination.

**Mitigation:**
- D1 job table is source of truth.
- Modal job writes progress events to control plane or manifests.
- Clients poll initially; add SSE/WebSocket only if required.

---

## 8. Concrete Next-Step Plan, Once Approved

### Task 1: Create the new repo skeleton

**No implementation beyond repo scaffolding.**

**Commands, once approved:**

```bash
cd /Users/dizzler/Projects/Repos
gh repo create kenkui-cloud --private --clone
cd kenkui-cloud
mkdir -p packages/kenkui-modal/src/kenkui_modal packages/kenkui-modal/tests docs services/control-plane infra .github/workflows
```

**Files:**
- `README.md`
- `docs/architecture.md`
- `packages/kenkui-modal/pyproject.toml`

**Commit:**

```bash
git add .
git commit -m "chore: bootstrap kenkui cloud repo"
```

### Task 2: Move scaffold files from `kenkui` to `kenkui-cloud`

**Files moved:**
- From `src/kenkui/modal_runtime/*`
- To `packages/kenkui-modal/src/kenkui_modal/*`

**Import changes:**
- `kenkui.modal_runtime` → `kenkui_modal`

**Commit in new repo:**

```bash
git commit -m "chore: extract modal provider scaffold"
```

### Task 3: Replace core import with plugin discovery

**In `kenkui`:**

Modify:
- `src/kenkui/services/runtime_service.py`

Remove:
- `src/kenkui/modal_runtime/`

Add/update tests:
- `tests/test_runtime_service.py`

**Commit in `kenkui`:**

```bash
git commit -m "refactor: load runtime providers from plugins"
```

### Task 4: Add editable local dev workflow

**Docs:**

```bash
cd /Users/dizzler/Projects/Repos/kenkui
uv pip install -e ../kenkui-cloud/packages/kenkui-modal
```

**Validation:**
- `kenkui` discovers `kenkui-modal` entry point.
- Modal modes resolve providers.
- No actual Modal call is made yet.

### Task 5: Stop and reassess before cloud implementation

Before R2/Modal/OAuth code:
- Confirm artifact layout.
- Confirm BYO vs hosted.
- Confirm auth provider.
- Confirm retention/deletion policy.

---

## 9. What I Would Not Do Yet

- Do not add OAuth code.
- Do not add R2 code.
- Do not create Modal functions.
- Do not decide billing/quota logic.
- Do not move local TTS/NLP dependencies out of core in the same PR as repo extraction.
- Do not introduce Terraform/OpenTofu yet.
- Do not make `kenkui` depend on `kenkui-cloud`.

Reason: the cleanest first move is extraction and plugin loading. Cloud implementation should be a second architectural milestone.

---

## 10. Approval Questions Blocking Implementation

Please answer these before I move files or create a repo:

1. Should the new repo be named `kenkui-cloud`, `kenkui-modal`, or something else?
2. Should it be private initially?
3. Which GitHub owner/org should own it?
4. Is the first target BYO Modal/R2, hosted multi-user, or hybrid?
5. Do you want the control plane in the same repo from day one as docs/stubs, or should the first repo be Python-only?
6. Are you okay with entry-point plugin discovery in `kenkui` core?
7. Should the current local commit `f3631fd` be pushed before extraction, or should extraction happen as a follow-up local commit first?
8. Should `modal_*` fields remain in `AppConfig`, or should most Modal config move entirely into `kenkui-modal` settings?
9. For hosted auth, do you prefer Clerk, Auth0, Cloudflare Access, Supabase Auth, or undecided?
10. Should default artifact retention be delete-on-success, retain-final-only, or retain-everything-for-debug?

---

## 11. My Current Best Answer

If I had to choose with current information:

- Repo: `kenkui-cloud`
- Visibility: private until OAuth/R2/security story is reviewed
- First package: `packages/kenkui-modal`
- Import package: `kenkui_modal`
- Core integration: Python entry points, not direct imports
- First deliverable: extraction only, no real Modal/R2/OAuth
- Artifact backend later: R2 via `boto3`
- Hosted control plane later: Cloudflare Workers + Hono + D1, OAuth via Clerk/Auth0 depending on product audience
- Default retention: delete source/intermediates on success, retain final artifact for a configurable short window
- Progress: polling first; Durable Objects/SSE later only if UX demands it

This plan is intentionally conservative: it prioritizes clean boundaries before cloud functionality.

---

## 12. Decision Update After User Feedback — 2026-06-23

### Decisions now accepted

1. **Repo shape:** Use `kenkui-cloud` as the umbrella repo, with a Modal plugin/package inside it.
2. **Visibility:** Private initially.
3. **GitHub owner:** Default GitHub credentials / `D1zzl3D0p`.
4. **Product target:** Hosted multi-user is the long-term goal, but BYO-style environment configuration remains the right 12-factor implementation pattern for internal services and deployable functions.
5. **Architecture docs:** The new repo should include rigorous architecture docs from day one so AI agents can build within the defined boundaries.
6. **Core integration:** Use entry-point/plugin discovery. The user does not need to understand this mechanism; it is the cleanest approach because it keeps `kenkui` from importing cloud packages directly.
7. **Core cleanup:** Move all Modal-specific implementation and most Modal-specific config out of `kenkui` into `kenkui-cloud/packages/kenkui-modal`.
8. **OAuth:** Undecided; needs recommendation.
9. **Retention:** Default to retaining everything for now, but design retention as a policy with lifecycle controls, not as hardcoded permanence.

### Updated OAuth recommendation

For the stated goal — hosted multi-user audiobook generation, likely with a desktop/client app plus possibly a web control surface — the best default recommendation is:

**Use Clerk for the first hosted version unless enterprise/B2B SSO is a near-term requirement; use Auth0 if enterprise SSO/compliance is central. Do not roll our own OAuth. Cloudflare Access is best only for private/admin/internal access.**

#### Clerk

**Pros**
- Fastest path to polished user auth.
- Good hosted login UI and account management.
- Works well with modern web apps and API backends.
- Supports social OAuth providers and session/JWT workflows.
- Lower implementation burden, which matters because Modal/R2/job orchestration is already complex.

**Cons**
- Vendor dependency.
- Pricing/limits need review before public launch.
- If we have a pure desktop-first app, we need to design the browser-based login/device-token flow carefully.

**Best fit**
- Consumer/prosumer hosted product.
- Private beta evolving into public hosted service.
- Fast iteration with minimal auth engineering.

#### Auth0

**Pros**
- Mature and enterprise-friendly.
- Strong SSO/SAML/enterprise identity support.
- Flexible token/tenant/rule/action model.
- Familiar to many security teams.

**Cons**
- More configuration and conceptual overhead than Clerk.
- Can feel heavy for a small hosted product.
- Pricing can become relevant as users/organizations grow.

**Best fit**
- B2B/enterprise from the beginning.
- Organizations, SAML, enterprise SSO, strict tenant separation.

#### Cloudflare Access / Zero Trust

**Pros**
- Excellent for private/internal tools.
- Integrates naturally with Cloudflare-hosted control plane.
- Can protect admin dashboards and internal endpoints easily.

**Cons**
- Not ideal as the primary consumer auth product layer.
- Account UX is not what most public users expect.
- Better for operator/admin access than user accounts.

**Best fit**
- Private alpha for you/team.
- Admin surfaces even if Clerk/Auth0 handles user auth.

#### Supabase Auth

**Pros**
- Good if we choose Supabase/Postgres for job state.
- Integrated auth + database story.
- Open-source-ish ecosystem and straightforward APIs.

**Cons**
- Less natural if the control plane is Cloudflare Workers + D1 + R2.
- Adds another platform alongside Cloudflare and Modal.

**Best fit**
- If we abandon Cloudflare D1/Workers for Supabase/Postgres as the app backend.

#### Roll our own OAuth/session service

**Pros**
- Maximum control.
- No auth vendor dependency.

**Cons**
- Highest security risk.
- Slower.
- Easy to get token/session/account-linking edge cases wrong.
- Bad use of effort for this project.

**Best fit**
- Almost never for v1.

### Updated retention recommendation

Cloudflare R2 pricing docs currently indicate:
- Standard storage: `$0.015 / GB-month`.
- Infrequent Access storage: `$0.01 / GB-month`.
- Standard egress/data transfer to Internet: free.
- Infrequent Access has data retrieval processing fees.
- Operations are billed separately as Class A / Class B requests.
- Free tier includes 10 GB-month/month standard storage and operation allowances.

Therefore, retaining everything is financially plausible for early/private usage, but it is still not automatically the best product/security default.

**Recommended policy:**

- **Default for private alpha:** retain everything.
- **Design from day one:** every artifact has retention metadata and lifecycle class.
- **Future hosted default:** retain final output and manifests; retain sources/intermediates only while user policy allows.
- **Required from day one:** deletion API and per-user/job purge capability, even if not used by default.

Why not hardcode permanent retention:
- Ebooks and voice samples may be copyrighted/private/sensitive.
- Some users will expect deletion.
- Retaining custom voice samples increases privacy obligations.
- Storage is cheap, but support/security risk is not free.
- Large generated WAV intermediates can accumulate quickly.

### Updated architecture consequence

`kenkui-cloud` should include docs/stubs from day one:

```text
docs/
  architecture.md
  auth-options.md
  artifact-retention.md
  r2-object-layout.md
  hosted-control-plane.md
  ai-agent-contract.md
```

The first repo bootstrap should prioritize these documents before implementation so AI agents have a stable contract to follow.

---

## 13. Final Pre-Bootstrap Stack Decision — 2026-06-23

### User direction incorporated

- The project should be **hosted multi-user**, not merely BYO cloud.
- BYO-style env/secrets are still desirable internally because they preserve 12-factor deployability.
- The user is open to Supabase, Cloudflare, AWS, or other providers if justified.
- The user wants decisions set before repo bootstrap so AI agents can build against a stable architecture.
- The user is questioning whether Modal remains the right remote workload platform.

### Final recommended v1 stack

Use this as the repo bootstrap contract unless explicitly superseded:

```text
Auth / users / DB / realtime / control-plane functions:
  Supabase

Large object/artifact storage:
  Cloudflare R2

Remote heavy compute v1:
  Modal

Future compute extension candidates:
  RunPod serverless, AWS Batch GPU, AWS ECS GPU, GCP Cloud Run GPU

Core library:
  kenkui remains local-first and cloud-provider agnostic

Cloud repo:
  kenkui-cloud
```

### Why Supabase for auth/control plane

Supabase is the best fit for the hosted multi-user control plane because it bundles:

- OAuth/social login and email auth.
- User identity.
- Postgres for relational job/user/artifact metadata.
- Row Level Security for user-owned job data.
- Realtime subscriptions for progress updates.
- Edge Functions for signed upload URLs, job submission, cancellation, and privileged service-role actions.

This avoids building a custom OAuth/session system. It also gives a richer relational database than Cloudflare D1 for job state, manifests, quotas, users, and future billing metadata.

#### Supabase critique

- It adds another vendor alongside R2 and Modal.
- Edge Functions are Deno/TypeScript rather than Python.
- Logs are not in the same Cloudflare dashboard as R2.
- Supabase Storage overlaps with R2, so architecture must clearly say: Supabase stores metadata; R2 stores large artifacts.

#### Why this critique is acceptable

The hosted product needs good auth, DB, and realtime progress more than it needs single-vendor purity. Supabase reduces auth/control-plane implementation risk enough to justify the extra vendor.

### Why Cloudflare is not the whole stack

Cloudflare remains excellent for R2 object storage, and possibly for future edge/API surfaces. But Cloudflare alone is not the best full stack here because:

- Workers are constrained for heavy compute: 128 MB memory and CPU-time limits unsuitable for TTS/NLP rendering.
- Cloudflare Containers are useful for CPU containers, but current documented instance sizes top out around 4 vCPU / 12 GiB memory / 20 GB disk unless limits are increased.
- Cloudflare Containers are not a GPU compute product for arbitrary Pocket-TTS workloads.
- Workers AI is a model catalog/API product, not a way to run arbitrary `pocket-tts`, spaCy, or BookNLP code.

Therefore Cloudflare should own **R2 artifacts** in v1, not the TTS/NLP compute plane.

### Why not AWS as v1 default

AWS can absolutely run this system end-to-end:

```text
Auth: Cognito
DB: DynamoDB / RDS / Aurora
Artifacts: S3
Queue: SQS / EventBridge
Compute: AWS Batch GPU / ECS on EC2 GPU / SageMaker Async
Logs: CloudWatch
```

It is a strong future enterprise option. However, it is not the best v1 default because:

- GPU batch orchestration on AWS requires more infrastructure: ECR, ECS/Batch compute environments, IAM, networking, instance quotas, AMIs, CloudWatch, scaling policies.
- Lambda is unsuitable for the heavy path because it has no GPU and a 15-minute timeout.
- SageMaker Async is good for model inference, but its documented long-processing use case is up to one hour and is more endpoint-oriented than arbitrary audiobook pipeline orchestration.
- AWS Batch GPU is powerful, but much more ops-heavy than Modal.

AWS should remain a future `kenkui-aws` runtime provider if enterprise, single-cloud, or very high-volume economics justify it.

### Why Modal remains the v1 remote workload choice

Modal is still the best v1 choice for heavy remote workloads because it matches the actual compute shape:

- Arbitrary Python/container workloads, not just model endpoints.
- GPU functions with specific GPU selection.
- CPU functions for spaCy/BookNLP.
- Function timeouts configurable up to 24 hours, suitable for long audiobook jobs.
- Secrets support.
- Cloud bucket mounts support Cloudflare R2 directly.
- Volumes for model/cache sharing where appropriate.
- Per-second billing and scale-to-zero behavior.
- Very low operational overhead compared with AWS Batch/ECS.

#### Modal critique

- It is another vendor outside Supabase/R2.
- Cost may be higher than raw RunPod/AWS GPU capacity at scale.
- It is not the durable job database or user control plane.
- Long jobs still need our own manifest/status/cancellation model.

#### Why this critique is acceptable

The biggest v1 risk is implementation complexity, not theoretical GPU unit cost. Modal minimizes infrastructure work while preserving enough control over Python, containers, secrets, R2 mounts, and long-running jobs. We keep compute pluggable so RunPod/AWS can be added later if costs or product requirements demand it.

### Why not RunPod as v1 default

RunPod Serverless is a serious alternative and may be cheaper for GPU-heavy workloads. It supports custom Docker workers, serverless endpoints, model caching, network volumes, and per-second pricing.

However, for v1:

- It is less Python-native than Modal for function-style orchestration.
- It pushes us toward Docker/image/endpoint management earlier.
- It is better when the workload is a stable model-serving endpoint; our workload is a multi-stage audiobook job pipeline.
- It still requires separate auth, job state, and artifacts.

RunPod should be designed as a future runtime provider, not the first implementation target.

### Final v1 architecture diagram

```text
Client / desktop / web
  |
  | Supabase Auth session
  v
Supabase Edge Functions  ---------------------+
  | create job / sign uploads / cancel / auth  |
  v                                           |
Supabase Postgres + RLS + Realtime            |
  | job rows / progress / artifact metadata   |
  v                                           |
Cloudflare R2 <-------------------------- Modal Functions
  | input bundles / voices / WAVs / M4B        |
  +--------------------------------------------+
```

### Responsibilities by component

#### `kenkui`

- Ebook/audio/NLP core library.
- Local runtime.
- Stable models and provider hooks.
- No Modal, Supabase, R2, OAuth, or hosted control-plane implementation.

#### `kenkui-cloud/packages/kenkui-modal`

- Modal app/functions.
- Modal provider implementation for `kenkui` hooks.
- R2 artifact client used by Modal functions.
- Runtime contracts for TTS/NLP cloud execution.
- Modal deploy/doctor tooling.

#### `kenkui-cloud/services/control-plane`

- Supabase Edge Functions.
- SQL migrations.
- RLS policies.
- Job creation/status/cancel APIs.
- R2 signed upload/download flows.
- Optional future billing/quota logic.

#### `kenkui-cloud/docs`

- Architecture contract for AI agents.
- ADRs for Supabase, R2, Modal.
- Artifact retention policy.
- Job lifecycle and schema.
- Security model.

### Decisions now set for bootstrap

- Auth/control plane: **Supabase**.
- Artifact storage: **Cloudflare R2**.
- Remote compute v1: **Modal**.
- Repo: **private `D1zzl3D0p/kenkui-cloud`**.
- First plugin package: **`packages/kenkui-modal`**.
- Keep cloud-specific implementation out of `kenkui` core.
- Use plugin/entry-point discovery from `kenkui` to installed cloud runtimes.
- Retention default: **retain everything in private alpha**, with metadata and purge APIs from day one.

### Still open, but not blocking bootstrap

These should be documented as TODOs/ADRs in `kenkui-cloud`, but they do not block repo creation:

1. Exact Supabase OAuth providers enabled first: GitHub, Google, Apple, email magic link, etc.
2. Whether the first user-facing client is desktop-only, web, CLI, or all three.
3. Whether progress is initially polling or Supabase Realtime. Recommendation: write DB rows and support polling first; add Realtime subscriptions immediately after.
4. Detailed quota/billing policy.
5. Whether custom voice assets require special encryption or consent flows.

### Bootstrap order updated

1. Create private `kenkui-cloud` repo.
2. Add docs/ADRs first:
   - `docs/architecture.md`
   - `docs/adr/0001-supabase-control-plane.md`
   - `docs/adr/0002-cloudflare-r2-artifacts.md`
   - `docs/adr/0003-modal-v1-compute.md`
   - `docs/adr/0004-runtime-plugin-boundary.md`
   - `docs/ai-agent-contract.md`
3. Add package skeleton for `packages/kenkui-modal`.
4. Move Modal scaffold out of `kenkui`.
5. Convert `kenkui` runtime registration to entry-point plugin discovery.
6. Stop again before implementing Supabase/R2/Modal real behavior.
