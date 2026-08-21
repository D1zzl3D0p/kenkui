# Multi-voice casting: characters, attribution, and the shared renderer

Date: 2026-08-20
Status: Approved for planning

## 1. Problem

Kenkui renders one voice. `Pipeline.assign_voice()` stores a single ID
(`pipeline.py:93`, `:235`), `ExecutionPlan.voice` is one `VoicePlan` documented
as *"Resolved single-voice content"* (`_domain/planning.py:110`), and
`SpeechSegment` carries no speaker (`:124`). Segmentation is length-driven
chunking with no notion of dialogue (`:298-370`). There is no `characters`
package, `litellm` is not a dependency, and `voices/assign.py` does not exist.
The server declares `SingleVoiceCasting` — *"The only supported local casting
intent"* (`jobs/models.py:17`) — behind `CastingCapabilities.mode:
Literal["single"]` (`config.py:41`), and the web casting component is a single
sentence.

This is deliberate. The voice-provisioning design (2026-08-19) lists *"Character
voices and multi-voice assignment"* under out of scope, and the spec-completion
plan defers them to Task 10, gated behind a stable hosted single-voice slice.
Tasks 4 through 8 are complete; Tasks 9 and 10 have not started.

This spec covers Task 10 in full: character inference, quote attribution,
automatic casting, the renderer changes that let one engine speak in many
voices, and the server and browser surface that exposes it.

## 2. Facts that shaped this design

Verified against the working tree, not recalled.

**Engines are per-language; voices are cheap.** A language engine is roughly
225 MB of weights against roughly 6.5 MB per speaker embedding
(voice-provisioning design section 4). `PocketTTSEngine._voice_state()` already
derives one conditioning state and reuses it for every segment, deliberately —
*"Workers hold a reusable engine and process a batch serially, so deriving
per-segment state is pure waste"* (`_tts/pocket.py:766-782`). The cost of a
second voice is therefore one 6.5 MB state, not a second model.

**But the config conflates them.** `PocketEngineConfig` carries
`voice_asset_path` and `voice_asset_sha256` inline (`:62-69`), and
`_make_snapshot` copies that single asset into the per-engine snapshot
(`:590-618`). One engine means one voice purely as an artifact of that shape.

**One engine per worker is enforced.** `_parse_result` rejects any result whose
metadata reports `engine_initializations != 1` (`_execution/process_pool.py:505`).
Any design that instantiates an engine per voice fails validation.

**Determinism is claimed for planning, not for audio.** `docs/architecture.md`
promises the same *plan* regardless of output path, callback, worker count,
cache location, or run ID, and states that *"the source file and resolved
voice/model inputs — not any cached row — remain the authority for every run"*
(`:31-33`). It separately states that *"segment PCM caching is a private
execution optimization, never a public pipeline parameter or source of semantic
truth"* (`:74`).

**Resolution already happens in the shell.** `compile_execution_plan` receives
`resolved_voice: Voice | None` as a finished value (`_domain/planning.py:183-188`).
The effectful part — manifest read, hashing, state checks — runs in
`pipeline.py:247` before the pure planner is called. The comment at
`pipeline.py:213-217` records why: *"an unprovisioned or unknown voice must fail
before any worker spawns, and that ordering should be legible."*

**Operations are unordered.** `append_unique` enforces type uniqueness and one
rule only: nothing may be appended after `SynthesizeSpeech`
(`_domain/operations.py:70-74`). The planner reads operations by type through
`has_operation`, never by position. Order-independence is a property worth
preserving.

**Network is already partitioned.** `sys.addaudithook(_network_audit)` and
`HF_HUB_OFFLINE=1` are installed in `_enter_spawned_worker` only
(`_tts/pocket.py:510-518`, `:679-685`). Workers deny the entire `socket.*` audit
namespace. All existing network — voice provisioning — happens in the parent.

**The cache is fail-open.** `_execution/cache.py` treats *"any unsafe state,
race, lock timeout, corruption, or I/O failure"* as a miss, and evicts against
bounded budgets (4096 entries, 512 MB). Losing a cached segment costs time.
State whose loss would change output cannot live there.

**Preflight is meant to be free.** `_preflight` builds its own minimal pipeline
rather than reusing `pipeline_from_job` (`kenkui-server-v2/.../api/jobs.py:73-79`),
and the route docstring is *"Validate executable local intent without creating a
Job or reservation."*

**The catalog is 21 English voices**, plus one each in Italian, Spanish, German,
Portuguese, and French (`voices/registry.py:87-175`). Reserving the narrator leaves 20 castable, or 19 when a
distinct unknown voice is chosen. VCTK ships documented speaker gender for
the twelve `p###` voices; the other nine English entries have no shipped
metadata.

## 3. Decisions

1. **Multi-voice is a composition, not an architecture.** One pipeline shape,
   one `VoicePlan`, one renderer. `assign_voice(x)` is defined as
   `assign_voices(narrator=x)` with an empty cast. This is the vNext
   requirement (`Kenkui_vNext_Architecture_Findings(1).md:98`, `:601`) and it is
   the load-bearing constraint on everything below.

2. **Attribution is a resolved input to the pure planner**, a peer of
   `resolved_voice`. The shell resolves it; the planner never calls a model and
   never reads the store. This is the existing pattern, not a new one.

3. **Attribution is treated as deterministic.** Kenkui already reuses a neural
   model's output keyed by its inputs without re-verification — that is what the
   PCM cache does. Extending the same treatment to a model call at fixed
   parameters with a pinned prompt version is consistent with the established
   posture. Two runs *can* differ after a store miss; that is a reproducibility
   cost, not a correctness bug, and it is bounded by pinning parameters and
   versioning prompts.

4. **Casting is pure.** Given attribution, a voice pool, a method, and explicit
   assignments, casting does no I/O. It is therefore directly unit-testable and
   deterministic, which is what the plan fingerprint requires.

5. **Both attribution and casting are persisted**, in a store separate from the
   fail-open PCM cache and separately named for operator legibility:
   `<cache_root>/casting.sqlite3`. Persisting the cast is what stops `add_voice`
   from silently re-shuffling every character in a book already rendered.

6. **The voice manifest stays JSON.** The vNext doc assigns voice data to
   SQLite, but that document is explicitly superseded
   (`docs/superpowers/plans/2026-08-18-spec-completion.md:35`) and the operative
   voice-provisioning design chose `manifest.json`. `KENKUI_POCKET_MANIFEST`
   exists to be hand-authored, and `_read_manifest` (`_tts/production.py:230`) is
   the security boundary deciding which model weights load. Neither is improved
   by moving to SQLite. `cast_assignments.voice_id` references voices by string
   ID; the stale-voice policy in section 5.4 is required regardless, so a
   foreign key would save no code.

7. **Casting methods are candidate filters over a shared solver.** A method
   answers only *"which voices is this character eligible for?"*. Coloring,
   least-used weighting, and tie-breaking are shared. Adding an LLM-driven or
   description-and-keyword method later means one filter function, not a second
   solver.

8. **No RNG.** `"random"` means *unconstrained pool*; the solver's deterministic
   ordering supplies the variation. A seeded generator would be one more thing
   to pin in the fingerprint for no benefit. Same book, method, and pool always
   yield the same cast.

9. **Narrator and unknown are reserved roles**, excluded from every method's
   candidate pool. The narrator speaks in every chapter, so as a graph vertex it
   is adjacent to every character; excluding its voice is pre-coloring a
   universally adjacent vertex rather than a special case. The exclusion is
   method-independent and therefore applies to all methods.

10. **Unknown speech has its own voice**, defaulting to the narrator's. It is a
    named role, not a fallback, so "I could not tell who said this" can be made
    audibly distinct or identical at the caller's choice.

11. **Cast collisions are minimized and logged, never surfaced.** When list
    coloring admits no conflict-free assignment the solver minimizes collision
    weight and emits `log_event(..., "cast_collision", level=WARNING, ...)`
    through `observability.py`. It does not emit the public `Warning` execution
    event, which is sequenced into `on_event` and would reach the browser.

12. **`inspect()` never spends money and takes no flag.** It reports every fact
    derivable for free and marks the rest `pending`. Implicit
    evaluation-on-access was considered and rejected: Django's QuerySet
    laziness is the widely documented cautionary case, where implicit triggering
    is repeatedly described as a double-edged sword. A call whose cost depends
    on which operations you happened to chain is exactly that hazard, and
    `inspect()` is called by server preflight, which is contractually free.

13. **`resolve()` steps the pipeline forward and returns a `Pipeline`.** Same
    type in, same type out — the pattern Dask's `persist()` and Spark's
    `materialize()`/`cache()` both settled on, and the reason no second object
    type is needed. It is *optional*: `write()` resolves internally, so
    `resolve()` only lets a caller pay early. This mirrors Terraform, where
    `apply` runs without a prior `plan` and the preview verb is additive rather
    than a stage.

    `resolve()` is functional in shape but is not pure. It performs network
    access and writes the store, so it is an effect verb like `write()`. What it
    guarantees is immutability, idempotence, and intent preservation:

    - `p.resolve()` returns a new `Pipeline`; `p` is unchanged.
    - `p.resolve().operations == p.operations`.
    - `p.resolve().resolve()` performs no further work.
    - `p.resolve().write(x)` and `p.write(x)` produce identical output.
    - Resolved values live in a private field that never enters the semantic
      fingerprint, equality of intent, or the plan.

14. **Appending an operation drops resolved values.** Any change of intent
    invalidates resolution, with no per-operation reasoning about which appends
    survive. This is conservative and free, because re-resolution against a
    populated store is a lookup.

15. **The resolved cast is emitted as an execution event.** `CastResolved`
    reaches `on_event` before rendering begins, so a caller can observe the cast
    and abort through the existing `CancellationToken`. The server streams it
    over SSE for free, which is how the browser will eventually display casting
    without a metered preflight.

16. **Order-independence is preserved.** `attribute_quotes` requiring a roster is
    a *presence* check in `render_intent_errors`, not an ordering rule. The only
    ordering constraint remains `before_tts`.

17. **Single-language casts in v1, plural structure.** Validation rejects a cast
    spanning languages, but `PocketEngineConfig` and the plan model engines as a
    keyed collection, so a second language is later a change to validation and
    scheduling rather than a redesign. No multi-engine scheduling code is written
    now.

18. **Single-voice output must not change.** A chapter with no attributed
    dialogue is one span, so its chunks are byte-identical to today,
    `tts-chunks-v2` does not bump, and the existing PCM cache is not
    invalidated. This is an asserted regression test, not an aspiration.

## 4. Public API

### 4.1 Pipeline operations

```python
Pipeline.infer_characters(model: str) -> Pipeline
```

Records intent to derive a character roster. Appends `InferCharacters`; performs
no work.

```python
Pipeline.attribute_quotes(model: str) -> Pipeline
```

Records intent to assign a speaker to each quote span. Appends
`AttributeQuotes`; performs no work. Requires `InferCharacters` to be present,
checked by presence and not by position.

```python
Pipeline.assign_voices(
    *,
    narrator: str | Voice,
    unknown: str | Voice | None = None,
    cast: Mapping[str, str | Voice] | None = None,
    method: CastingMethod = "gendered",
) -> Pipeline
```

Records casting intent. `unknown` defaults to the narrator's voice. `cast`
pins specific characters; pinned entries are constraints on the solver, never
suggestions. Stores IDs only, keeping `Pipeline` intent-only.

`assign_voice(voice)` is retained and defined as `assign_voices(narrator=voice)`
with an empty cast. Both produce one `VoicePlan`.

### 4.2 Inspection

```python
Pipeline.inspect() -> BookInspection
```

Never performs network access and never calls a model. Signature is unchanged
from today. It returns source facts as before, plus the pipeline's casting
intent, plus attribution and cast **when already resolved** — either carried on
the pipeline by `resolve()`, or present in the store. Because casting is pure, a
stored attribution yields the full cast for free. Otherwise those fields report
`pending`.

`BookInspection` gains optional fields with defaults, so existing consumers —
`assets.py:66` and `jobs.py:79` — are unaffected.

### 4.3 Resolution

```python
Pipeline.resolve() -> Pipeline
```

Performs voice-pool, attribution, and cast resolution and returns a new
`Pipeline` carrying the results. Optional: `write()` resolves internally, so
`resolve()` exists only to pay early and inspect the outcome.

```python
p = (kk.epub(path)
      .infer_characters(model=m)
      .attribute_quotes(model=m)
      .assign_voices(narrator="eponine", method="gendered"))

p.inspect()                 # free. attribution: pending
resolved = p.resolve()      # effect: model calls on a store miss, store write
resolved.inspect()          # free, complete
resolved.write("out.m4b")   # reuses; no further model spend

p.write("out.m4b")          # equally valid — resolves internally
```

Per decision 13 it is immutable, idempotent, and intent-preserving, and per
decision 14 any subsequent operation append discards the carried results.

### 4.4 Store management

```python
list_castings() -> tuple[Casting, ...]
remove_casting(casting_id: str) -> None
remove_attribution(attribution_id: str) -> None
```

Module-level verbs mirroring the voice provisioning surface. The pipeline
operation stays pure intent; regeneration is remove-then-rerun. This follows
`load_voice`'s precedent — a forced refetch is `unload_voice` then `load_voice`,
not a flag — and voice-provisioning decision 11's rejection of booleans on verbs.
There are no bulk verbs; filtering composes over `list_castings()`, per
voice-provisioning decision 10.

The two removal verbs are distinct because the two artifacts have very different
costs. `remove_casting` discards one cast, which the pure solver rebuilds for
free. `remove_attribution` discards the model-derived roster and spans **and
cascades to every cast beneath it**, forcing a fresh model pass. Collapsing them
into one verb would hide the difference between a free operation and a billable
one.

`Casting` exposes the identity of both layers, so composition can target either:

```python
@dataclass(frozen=True, slots=True)
class Casting:
    id: str                       # cast_id
    attribution_id: str
    book_id: str
    model_id: str
    prompt_version: str
    method: CastingMethod
    narrator_voice_id: str
    unknown_voice_id: str
    assignments: tuple[VoiceAssignment, ...]
    created_at: str
```

### 4.5 Voice traits

`Voice` gains `perceived_gender: Literal["feminine", "masculine"] | None`.

```python
add_voice(..., perceived_gender: Literal["feminine", "masculine"] | None = None)
```

The trait is **sourced where sourceable and `None` otherwise**. VCTK publishes
speaker gender for the twelve `p###` voices; those are populated from that
metadata. The remaining nine English catalog entries have no shipped metadata
and are `None` until reviewed. Kenkui's display names (`Anna`, `Charles`) are
its own inventions and are never used to infer the trait.

**`None` never silently joins a gendered pool.** A voice without a trait is
eligible only under methods that do not filter on it.

### 4.6 Events

```python
@dataclass(frozen=True, slots=True)
class CastResolved:
    """The character-to-voice assignment resolved for this run."""

    sequence: int
    stage: str
    narrator_voice_id: str
    unknown_voice_id: str
    assignments: tuple[VoiceAssignment, ...]
```

Emitted through `on_event` after resolution and before any worker spawns, so a
caller may inspect the cast and abort through the `CancellationToken` already
passed to `write()`. It joins the existing `ExecutionEvent` union and
`__all__`. Single-voice runs emit it with an empty `assignments` tuple, keeping
one event shape for both castings.

## 5. Casting

### 5.1 The model

Characters are vertices. An edge joins two characters that speak in the same
chapter. Voices are colors, and each character's admissible colors are supplied
by the method. This is list coloring on a co-occurrence graph.

Chapter granularity is both what is wanted and what is available: kenkui has no
scene model, and `ChapterInspection` is what the planner already consumes.

### 5.2 Methods

```python
CastingMethod = Literal["random", "gendered"]

def candidates(method, character, pool) -> tuple[Voice, ...]:
    "random"   -> pool
    "gendered" -> [v for v in pool if v.perceived_gender == character.gender]
                  # unknown-gender character falls back to the full pool
```

The pool is every **loaded** voice matching the cast language, minus the
narrator and unknown voices (decision 9). Provisioning more voices with
`add_voice` therefore directly improves casting quality, which is the intended
remedy for a shallow pool.

Future methods — LLM-driven casting, character-description against voice
keywords — are added as filters. The solver does not change.

### 5.3 The solver

Deterministic greedy list coloring, DSATUR-ordered:

1. Pin explicit `cast={...}` entries. They are constraints.
2. Order remaining characters by saturation descending, then
   `spoken_characters` descending, then character ID.
3. From the character's candidate list, drop voices used by a same-chapter
   neighbour. Among survivors take the lowest **weighted usage** — the sum of
   `spoken_characters` already assigned to that voice, not the count of
   assignments — tie-breaking on voice ID. Weighting by speech volume is what
   keeps a lead from sharing with a walk-on. Characters are measured in
   normalized speech characters throughout, the same unit chunking, progress,
   and billing already use.
4. If no survivor exists, take the candidate minimizing collision weight and log
   `cast_collision`.

Every ordering and tie-break is total and content-derived, so the solver is a
pure function of its inputs.

### 5.4 Stale voices

On reuse, each stored assignment's `voice_id` is checked against the current
manifest for `loaded` state. Stale assignments are dropped and re-solved against
the current pool; every other character keeps its voice. Unloading one voice
re-casts one character, not the book.

## 6. Store schema

`<cache_root>/casting.sqlite3`, alongside `manifest.json` and `cache.sqlite3`
under the root resolved by `default_cache_root()` (`_tts/production.py:64-72`).

```sql
books(book_id PK, title, author, created_at)
    -- book_id = source_bytes_sha256

attributions(attribution_id PK, book_id, model_id, prompt_version,
             params_json, created_at)
    -- attribution_id = sha256(book_id|model_id|prompt_version|params)

characters(attribution_id, character_id, display_name, gender,
           spoken_characters, span_count,
           PRIMARY KEY (attribution_id, character_id))
    -- gender is the inferred trait, same domain as Voice.perceived_gender
    --   plus NULL; spoken_characters is the prominence weight

character_chapters(attribution_id, character_id, chapter_id)
    -- the co-occurrence edges

quote_spans(attribution_id, chapter_id, start, end, character_id)
    -- character_id NULL means Unknown

casts(cast_id PK, attribution_id, method, narrator_voice_id,
      unknown_voice_id, created_at)
    -- cast_id = sha256(attribution_id|method|explicit|narrator|unknown)

cast_assignments(cast_id, character_id, voice_id, pinned,
                 PRIMARY KEY (cast_id, character_id))
```

`casts` is keyed **downstream** of `attributions`, so exploring many castings of
one book costs exactly one model pass. That is what makes branching cheap:

```python
base = kk.epub(path).infer_characters(model=m).attribute_quotes(model=m)
a = base.assign_voices(narrator="eponine", method="gendered")
b = base.assign_voices(narrator="eponine", cast={"darcy": "michael"}, method="gendered")
a.inspect(); b.inspect()   # one attribution, two casts
```

The store is written through a temporary file, `fsync`, and `rename` at mode
`0600` inside a `0700` directory, matching `voices/manifest.py`. Corruption is
recoverable by recompute, so reads fail soft to a miss; writes fail loud.

## 7. Resolution boundary

Attribution and cast resolution join voice resolution at `pipeline.py:218`,
before `execute_sequential` and before any worker exists:

```python
validation = self.validate()          # cheap: source, presence, operation rules
...                                   # workers, output path, overwrite checks
cancel.raise_if_cancelled()
pool        = _resolved_voice_pool(...)              # manifest read
attribution = _resolve_attribution(...)              # store hit, else model, then store
cast        = _resolve_cast(attribution, pool, ...)  # pure
return execute_sequential(self, output_path, bindings=..., cast=cast, ...)
```

`compile_execution_plan` receives `pool`, `attribution`, and `cast` as finished
values beside `resolved_voice`. It calls no model and reads no store.

`Pipeline.resolve()` performs exactly these three steps and returns a new
`Pipeline` carrying the results, so `write()` on a resolved pipeline skips
straight to `execute_sequential`. There is one resolution implementation with
two entry points, not two code paths: `write()` calls `resolve()` internally
when the pipeline carries no resolved values.

`CastResolved` is emitted immediately after resolution and before the worker
pool exists, so a caller aborting from the callback pays for attribution but
never for rendering.

**Cancellation.** Attribution over a long book is a sequence of model calls, and
the only pre-render cancel check today is `pipeline.py:211`. Resolution performs
cooperative `cancel.raise_if_cancelled()` checks between chapter-level calls, so
cancellation is honored during attribution rather than deferred until rendering
starts.

## 8. Segmentation

Attribution produces spans that **partition** the normalized chapter text. The
existing frozen `tts-chunks-v2` chunker then runs *within* each span. The
invariant `"".join(chunks) == text` is preserved by construction, since a
partition of a partition is a partition.

A chapter with no attributed dialogue is a single span, so its chunks are
byte-identical to today. `CHUNKING_SCHEMA_VERSION` does not bump, single-voice
renders are unchanged, and no existing cache entry is invalidated. Multi-span
chapters yield segments whose identity includes the speaker, so they are
naturally distinct keys rather than collisions.

`SpeechSegment` gains `speaker_id: str | None` — `None` meaning narration — and
`voice_id: str`.

## 9. Renderer

`PocketEngineConfig` splits into engine identity (weights, config path,
revision, sample rate, device, file manifest) plus `voices: tuple[VoiceAsset, ...]`,
where `VoiceAsset` carries the path, SHA-256, variety, and rights fields
currently inlined. `_make_snapshot` copies every asset in the collection.

`PocketTTSEngine._voice_state()` becomes a mapping keyed by asset SHA-256,
derived lazily on first use per voice and reused thereafter. One model is
constructed per worker, so `engine_initializations == 1`
(`_execution/process_pool.py:505`) holds unchanged.

Per decision 17, the plan models engines as a keyed collection while validation
rejects more than one entry.

## 10. Cache

`CacheStore.key_for` currently mixes `_voice_material(plan)` at plan level
(`_execution/cache.py:162`). It moves to per-segment voice material, so two
characters speaking identical text produce different keys. `semantic_material()`
in `_tts/pocket.py:80` narrows to engine identity; voice identity moves to the
per-segment contribution.

## 11. Error codes

New stable codes:

- `attribution_unavailable` — a cast was requested but no attribution resolved.
- `model_call_failed` — provider error after bounded retry and backoff.
- `model_response_invalid` — response failed strict JSON validation.
- `cast_language_mixed` — assigned voices span languages.
- `cast_pool_empty` — a method's candidate pool holds no loaded voices.
- `casting_method_unknown` — unrecognized method name.
- `character_unknown` — explicit `cast` names a character absent from the roster.

Reused unchanged: `voice_not_provisioned`, `voice_unknown`, `voice_disabled`,
`duplicate_operation`.

`model_not_allowed` is a **server** code, defined with the allowlist in section
12. Core does not know about allowlists.

`validate()` is documented as inexpensive, so it checks operation presence and
uniqueness, method names, voice existence, loaded state, and language agreement.
It cannot check that an explicit cast names real characters, because that
requires attribution; `character_unknown` therefore surfaces at plan time.

## 12. Server

`CastingCapabilities.mode: Literal["single"]` becomes
`modes: list[Literal["single", "characters"]]` defaulting to `["single"]`. This
is a breaking OpenAPI change and regenerates
`kenkui-web-v2/src/api/generated/v1.ts`. `"characters"` is advertised only when
the deployment has both a non-empty LLM model allowlist and at least two loaded
voices.

`SingleVoiceCasting` becomes a union with `CharacterCasting(narrator_voice_id,
unknown_voice_id, cast, method, model_id)`. `CastingRequest` gains the
corresponding optional fields and retains `voiceId`. Row mapping
(`storage/repositories.py:44,56`) and `pipeline_from_job`
(`jobs/pipeline.py:17`) branch on the casting type.

The server validates `model_id` against its configured allowlist and rejects
others with `model_not_allowed`. Provider credentials live in the server
process environment; the browser never holds or sends them.

Billing is unchanged: credits remain `ceil(normalized_speech_characters / 1000)`
and multi-voice renders the same characters. The provider cost of attribution is
a genuinely new expense that this formula does not capture; metering it is out
of scope for v1 and noted in section 17.

## 13. Web

`casting.tsx` becomes a real component gated on `capabilities.casting.modes`:
narrator select, unknown select, and method select.

The `CastResolved` event streams over the existing SSE channel, so the browser
**can display the cast read-only while a job runs**, without any metered
preflight. That is the v1 casting visibility story.

**Per-character overrides are out of scope for v1**, for a structural reason:
the roster does not exist until attribution runs, and `_preflight` is
contractually free of Jobs and reservations
(`kenkui-server-v2/.../api/jobs.py:86-88`). Offering overrides requires a
preflight that resolves attribution, which means metering it. Section 17 records
this as the follow-on.

## 14. Network and credentials

All model traffic occurs in the parent during resolution, beside the network
that voice provisioning already performs. The render path is unchanged: the
socket audit hook and offline environment variables remain worker-only
(`_tts/pocket.py:510-518`, `:679-685`), and no credential crosses a process
boundary.

LiteLLM reads provider credentials from the environment at execution. They are
never stored, never logged, and never placed in the plan. The plan records the
model identifier, prompt version, and parameters — which is precisely what makes
them legitimate fingerprint inputs.

## 15. Testing

The load-bearing regression, protecting every shipped behavior:

- **Single-voice output is byte-identical.** Assert the plan fingerprint and
  segment IDs are unchanged for a single-voice pipeline, and that
  `tts-chunks-v2` does not bump.

Then:

- Solver purity: coloring correctness, weighted least-used selection, pin
  preservation, reserved-role exclusion, collision minimization, and identical
  output across repeated runs. No model, no I/O.
- `quotes.py` is model-free and tested directly.
- Inference and attribution normalization against a fake LiteLLM adapter, per
  spec-completion Task 10 step 1.
- Store: round trip, key derivation, cast branching sharing one attribution,
  incremental stale-voice re-solve, corruption falling back to recompute.
- `inspect()` performs no network under any pipeline, including one carrying
  character operations, asserted with a socket-denying audit hook rather than by
  inspection of call sites.
- `resolve()` is immutable, idempotent, and intent-preserving: the receiver is
  unchanged, `p.resolve().operations == p.operations`, a second `resolve()` does
  no work, and `p.resolve().write(x)` equals `p.write(x)` byte for byte.
- Appending an operation after `resolve()` discards the carried values, and the
  subsequent `write()` still produces correct output.
- Resolved values never reach the semantic fingerprint: `p` and `p.resolve()`
  compile to plans with identical fingerprints.
- `CastResolved` is emitted before any worker spawns, and cancelling from the
  callback aborts without rendering.
- Cancellation during attribution, not merely during rendering.
- Engine: N conditioning states still report `engine_initializations == 1`.
- Cache: two characters with identical text produce different keys.
- Server: allowlist enforcement, capability gating, character `JobSpec` round
  trip through row mapping.
- Real provider calls and real renders stay opt-in behind explicit environment
  variables, matching `tests/test_pocket_tts_real.py`.

## 16. Documentation

- `models-and-voices.md`: the `perceived_gender` trait, its sourcing, and why
  `None` is not guessed from display names.
- `usage.md`: casting methods, the reserved roles, and correcting the existing
  "Multi-voice work composes over `list_voices()`" line at `:163`, which is
  about bulk provisioning and reads today as a casting claim.
- `architecture.md`: attribution as a resolved input, the unchanged worker
  network posture, and the span-then-chunk segmentation.
- `pocket-tts-adapter.md`: the engine/voice split and the multi-state engine.
- `README.md`: a multi-voice example beside the single-voice one.

## 17. Open risks

**Attribution quality is unmeasured.** No baseline exists for speaker
attribution accuracy on real prose. The fake-adapter tests prove normalization,
not correctness. A wrong attribution is audible in a way a wrong cache entry is
not.

**Pool depth is shallow.** Nineteen or twenty castable English voices,
splitting to roughly ten and nine by gender, against novels that routinely exceed thirty
speaking characters. Collisions will be common, and the designed remedy —
provisioning more voices via `add_voice` — requires gated cloning weights for
WAV sources.

**Nine catalog voices lack a sourced gender trait** and are therefore excluded
from gendered pools until someone reviews them, which shrinks the effective pool
further than the numbers above suggest.

**Attribution provider cost is unmetered.** Server billing counts speech
characters only. A character job's model spend is invisible to the credit
ledger.

**Per-character overrides need a metered preflight** before the browser can
offer them, per section 13.

**Single-voice quality is still unassessed.** The voice-provisioning design
section 13 records that real inference runs only in the opt-in suite, and that
the platform matrix, determinism, cancellation under real load, timing, peak RSS,
and perceptual quality are all unmeasured. Multi-voice multiplies the surface
before that baseline exists.

## 18. Out of scope

- Per-character voice overrides in the browser.
- Metering provider spend in server billing.
- Multi-language casts, beyond the plural structure of decision 17.
- Migrating the voice manifest to SQLite.
- LLM-driven or keyword-matching casting methods, beyond the filter seam that
  admits them.
- Scene-level co-occurrence granularity.
- A CLI, per voice-provisioning decision 8.
