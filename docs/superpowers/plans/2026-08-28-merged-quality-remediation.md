# Merged Quality Remediation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Restore the source-authority, structural pause, voice lifecycle/cache identity, and deterministic quality-gate contracts violated by the recent mainline merges.

**Architecture:** Keep public pipeline values immutable. A resolved pipeline becomes valid only for the exact source snapshot it resolved; planning receives parser-provided structural coordinates rather than reconstructing them from text; voice records preserve all declared metadata through loading; cache keys encode every synthesis-relevant rights field. Test seams must be explicit injection points, never exception-driven production fallbacks.

**Tech Stack:** Python 3.11–3.13, dataclasses, defusedxml 0.7.1, pytest/pytest-cov, mypy, Ruff, MkDocs.

**Spec:** `docs/architecture.md`; `docs/development.md`; `docs/models-and-voices.md`; review findings recorded 2026-08-28.

## Global Constraints

- Preserve the functional-core/imperative-shell boundary: planners receive finished values and perform no I/O.
- Current EPUB bytes are authoritative for every render; stale resolved state must never enter planning.
- Keep `ChapterInspection`, plans, results, and events immutable.
- Preserve `tts-chunks-v2`/`tts-chunks-v3` concatenation and single-voice identity compatibility unless a new feature is active.
- Cache keys include voice content **and rights** identities; cache remains fail-open for rendering.
- EPUB 3 bare DOCTYPE and EPUB 2 public DOCTYPE are accepted; entity declarations and external entities remain rejected.
- Use no new runtime dependencies.
- The completion gate is `uv run ruff format --check . && uv run ruff check . && uv run mypy && uv run pytest && uv run mkdocs build --strict`.

---

### Task 1: Bind resolved state to an immutable source snapshot

**Files:**
- Modify: `src/kenkui/pipeline.py:77-90, 333-343, 444-495, 692-833`
- Modify: `tests/test_series_resolution.py`
- Modify: `tests/test_resolve.py` or `tests/test_pipeline.py`

**Interfaces:**
- Consumes: `_source_digest(path: Path) -> str`, `Pipeline.inspect() -> BookInspection`, and existing `Resolved` values.
- Produces: `Resolved.source_digest: str`; `_resolve_all()` records the digest used for attribution; `write_m4b()` reuses resolved values only when their digest equals the current source digest.

- [ ] **Step 1: Add a focused stale-resolution regression test**

Create an EPUB, configure attribution/casting with the existing fake attribution client and binding factory, call `.resolve()`, then replace the source at the same path with different valid content. Assert that `.write_m4b()` does not pass the old `SpeakerSpan` tuple to execution: either it calls `_resolve_all()` again and uses spans for the replacement content, or it raises a stable validation error before execution. Also test unchanged source reuses the existing resolved values.

```python
resolved = pipeline.resolve()
source.write_bytes(replacement.read_bytes())
with patch("kenkui.pipeline.execute_sequential") as execute:
    resolved.write_m4b(tmp_path / "out.m4b")
assert execute.call_args.kwargs["spans"] == replacement_spans
```

- [ ] **Step 2: Run the new regression test and confirm the current implementation fails**

Run: `uv run pytest --no-cov tests/test_resolve.py -k resolved_source`

Expected: FAIL because `write_m4b()` selects `self._resolved` without checking source bytes.

- [ ] **Step 3: Extend the resolved value with its source digest**

Add a required immutable field and populate it from the exact digest already calculated in `_resolve_all()`:

```python
@dataclass(frozen=True, slots=True)
class Resolved:
    source_digest: str
    cast_assignments: Mapping[str, str]
    unknown_voice_id: str
    spans: tuple[SpeakerSpan, ...]
    collisions: tuple[Collision, ...]
    bindings: ExecutionBindings
```

For the no-attribution path, calculate `_source_digest(pipeline.source.path)` before constructing `Resolved`; do not use a path, mtime, or cached inspection as the identity.

- [ ] **Step 4: Revalidate before execution**

In `write_m4b()`, calculate the current digest after validation and cancellation checks. Reuse `_resolved` only when `self._resolved.source_digest == current_digest`; otherwise call `_resolve_all(self, cancel)`. Pass the resulting resolved values unchanged to execution. Ensure `resolve()` remains idempotent only while its snapshot is current.

- [ ] **Step 5: Run focused behavior tests**

Run: `uv run pytest --no-cov tests/test_resolve.py tests/test_series_resolution.py tests/test_write_preflight.py`

Expected: PASS.

- [ ] **Step 6: Commit the isolated fix**

```bash
git add src/kenkui/pipeline.py tests/test_resolve.py tests/test_series_resolution.py
git commit -m "fix: invalidate stale resolved pipeline state"
```

### Task 2: Preserve source structural positions for heading pauses

**Files:**
- Modify: `src/kenkui/inspection.py:15-27`
- Modify: `src/kenkui/_epub/parser.py:82-104, 196-211, 238-276, 347-399`
- Modify: `src/kenkui/_domain/structure.py:119-150`
- Modify: `src/kenkui/_domain/planning.py:452-470`
- Modify: `tests/test_epub.py`
- Modify: `tests/test_structure.py`
- Modify: `tests/test_pauses_render.py`

**Interfaces:**
- Consumes: parser emission order and canonical normalized chapter text.
- Produces: `ChapterInspection.heading_block_indices: tuple[int, ...] = ()`, identifying canonical structural blocks that originate from visible `h1`–`h6` elements; `split_structural(text, heading_blocks, pauses)` uses block indices, never heading strings.

- [ ] **Step 1: Add duplicate-heading regression coverage**

Construct XHTML with an `h1` containing `Prologue`, an ordinary paragraph containing only `Prologue`, and a following ordinary paragraph. Request nonzero `heading_after_ms` and `heading_before_ms`. Assert only boundaries adjacent to the actual heading receive heading pause reasons; paragraph pauses remain unchanged.

```python
chapter = inspect_epub(source).chapters[0]
pieces = split_structural(
    chapter.text, frozenset(chapter.heading_block_indices), pauses
)
assert [piece.reasons for piece in pieces] == [
    frozenset({HEADING_AFTER}),
    frozenset({PARAGRAPH}),
    frozenset(),
]
```

Also retain the existing public `chapter.headings == ("Chapter One", "A Section")` assertion.

- [ ] **Step 2: Run the focused structural tests and confirm failure**

Run: `uv run pytest --no-cov tests/test_epub.py tests/test_structure.py tests/test_pauses_render.py -k 'heading or pause'`

Expected: FAIL because `split_structural()` currently classifies `body in frozenset(chapter.headings)`.

- [ ] **Step 3: Carry heading identity through canonical emission**

Extend `_TextEmitter` to record canonical block sequence and which emitted blocks are visible heading elements. During `_emit_element`, record block-open/block-close events alongside text/boundaries; normalize the emitted stream once, derive its `\n\n` structural blocks, and emit the indices whose source block is `h1`–`h6`. Do not locate headings with `str.find`, a text set, or text equality: repeated text is the regression.

Add `heading_block_indices` as a defaulted final field in `ChapterInspection` so existing direct test fixtures remain valid. Populate it in `_chapter_text()` and pass it through `_spine_chapters()`.

- [ ] **Step 4: Change the structural splitter contract**

Replace the text-based parameter with positional identity:

```python
def split_structural(
    text: str,
    heading_blocks: frozenset[int],
    pauses: PauseSpec,
) -> tuple[Piece, ...]:
    ...
    if pauses.heading_after_ms and index in heading_blocks and not last:
        reasons.add(HEADING_AFTER)
    if pauses.heading_before_ms and not last and index + 1 in heading_blocks:
        reasons.add(HEADING_BEFORE)
```

Update `_structural_pieces()` to pass `frozenset(chapter.heading_block_indices)`. Keep paragraphs and line splitting unchanged.

- [ ] **Step 5: Verify semantic invariants**

Run: `uv run pytest --no-cov tests/test_epub.py tests/test_structure.py tests/test_pauses_render.py tests/test_planning.py`

Expected: PASS, including chunk concatenation and no-pause identity regressions.

- [ ] **Step 6: Commit the isolated fix**

```bash
git add src/kenkui/inspection.py src/kenkui/_epub/parser.py src/kenkui/_domain/structure.py src/kenkui/_domain/planning.py tests/test_epub.py tests/test_structure.py tests/test_pauses_render.py
git commit -m "fix: retain heading structure for pause planning"
```

### Task 3: Preserve local voice gender through load lifecycle

**Files:**
- Modify: `src/kenkui/voices/provision.py:423-440`
- Modify: `tests/test_load_voice_local.py`
- Modify: `tests/test_casting_solver.py` only if an end-to-end gendered-pool assertion does not already exist

**Interfaces:**
- Consumes: `VoiceRecord.perceived_gender` supplied by `add_voice()`.
- Produces: loaded records and `load_voice()` results retain exactly that trait for local `wav` and `pre-compiled` voices.

- [ ] **Step 1: Add lifecycle regression tests**

Register a local `safetensors` voice with `perceived_gender="feminine"`, stub the engine/materialization dependencies using the existing local-load fixture, call `load_voice()`, then read the manifest and assert both returned `Voice` and stored `VoiceRecord` retain `"feminine"`. Add the equivalent `None` case to prove no value is inferred.

```python
voice = load_voice("local-f", manifest=manifest)
assert voice.perceived_gender == "feminine"
assert ManifestStore(manifest).read()[1]["local-f"].perceived_gender == "feminine"
```

- [ ] **Step 2: Run the lifecycle test and confirm failure**

Run: `uv run pytest --no-cov tests/test_load_voice_local.py -k perceived_gender`

Expected: FAIL because `_materialize()` omits the field.

- [ ] **Step 3: Preserve the declared trait during materialization**

Add the missing constructor field:

```python
return VoiceRecord(
    ...
    compatible_model_revisions=(engine.model_revision,),
    perceived_gender=record.perceived_gender,
)
```

Use the source record exactly; do not infer from voice names, assets, or catalog data.

- [ ] **Step 4: Run focused voice tests**

Run: `uv run pytest --no-cov tests/test_add_voice.py tests/test_load_voice_local.py tests/test_casting_solver.py tests/test_voice_traits.py`

Expected: PASS.

- [ ] **Step 5: Commit the isolated fix**

```bash
git add src/kenkui/voices/provision.py tests/test_load_voice_local.py tests/test_casting_solver.py
git commit -m "fix: retain local voice gender after loading"
```

### Task 4: Make `voice_rights` invalidate cache entries

**Files:**
- Modify: `src/kenkui/voices/types.py:28-44`
- Modify: `src/kenkui/voices/provision.py:309-326`
- Modify: `src/kenkui/_tts/production.py:89-152`
- Modify: `src/kenkui/_domain/planning.py:132-141, 364-404`
- Modify: `src/kenkui/_execution/cache.py:970-988`
- Modify: `tests/test_cache.py:109-148`
- Modify: voice construction fixtures in `tests/test_planning.py`, `tests/test_cache.py`, and production-manifest tests

**Interfaces:**
- Consumes: `Voice.voice_rights: str | None` loaded from the managed manifest and the effective `VoicePlan.voice_rights: str` selected for a segment.
- Produces: two otherwise equivalent plans with different rights statements produce different `CacheStore.key_for()` values.

- [ ] **Step 1: Extend the resolved voice path and add a cache-key regression assertion**

Add optional `voice_rights` to the public `Voice` value and required non-empty `voice_rights` to `VoicePlan`. Project the field in both `voices.provision._loaded_view()` and `_tts.production.production_bindings_from_environment()`. In `_resolve_voice()`, include it in `required_strings` and create the `VoicePlan` with its stripped value.

Beside the existing content-fingerprint mutation assertion, replace only `voice_rights` and assert a new key:

```python
changed_voice = replace(plan.cast.narrator, voice_rights="withdrawn consent")
assert (
    store.key_for(
        replace(plan, cast=CastPlan.single(changed_voice)), segment, task, spec
    )
    != baseline
)
```

Add an assertion that the resulting canonical key material contains only a SHA-256 digest of the statement, never its plaintext.

- [ ] **Step 2: Run the focused test and confirm failure**

Run: `uv run pytest --no-cov tests/test_cache.py -k key`

Expected: FAIL because neither the resolved public `Voice`/`VoicePlan` path nor the current material carries `voice_rights`.

- [ ] **Step 3: Add rights material to the complete semantic path**

Make the field available at every boundary:

```python
# voices/types.py
voice_rights: str | None = None

# _domain/planning.py
voice_rights: str

# _execution/cache.py
"voice_rights_sha256": hashlib.sha256(voice.voice_rights.encode()).hexdigest(),
```

Update all direct `Voice`/`VoicePlan` fixture constructors with explicit rights. Do not add cache location, callbacks, run IDs, or raw rights text.

- [ ] **Step 4: Run cache behavior tests**

Run: `uv run pytest --no-cov tests/test_cache.py tests/test_cache_branches.py tests/test_planning_multi_voice.py`

Expected: PASS.

- [ ] **Step 5: Commit the isolated fix**

git add src/kenkui/voices/types.py src/kenkui/voices/provision.py src/kenkui/_tts/production.py src/kenkui/_domain/planning.py src/kenkui/_execution/cache.py tests
git commit -m "fix: include voice rights in cache identity"
```

### Task 5: Remove exception-driven binding compatibility and stabilize EPUB coverage tests

**Files:**
- Modify: `src/kenkui/pipeline.py:957-977`
- Modify: tests that monkeypatch `kenkui.pipeline._execution_bindings`, including `tests/test_cache.py`, `tests/test_execution.py`, `tests/test_native_ffmpeg.py`, and `tests/test_series_resolution.py`
- Modify: `tests/test_epub.py` and, only after root-cause confirmation, the production XML setup in `src/kenkui/_epub/parser.py:123-167`
- Modify: `docs/security.md:25-28` if the accepted-DOCTYPE policy remains intentional

**Interfaces:**
- Consumes: `_execution_bindings(voice_id: str, *, also: Sequence[str] = ()) -> ExecutionBindings`.
- Produces: production `TypeError`s propagate; test doubles match the exact factory signature. The full coverage-enabled suite accepts bare/public DOCTYPEs and rejects entity declarations consistently.

- [ ] **Step 1: Update binding test doubles before changing production behavior**

Replace zero-argument lambdas with a signature-compatible helper in each affected test module:

```python
def _bindings_stub(
    _voice_id: str,
    *,
    also: Sequence[str] = (),
) -> ExecutionBindings:
    assert isinstance(also, tuple)
    return bindings
```

Where the test needs to observe a cast, record `_voice_id` and `also` rather than relying on fallback logging.

- [ ] **Step 2: Add a production-error propagation regression test**

Monkeypatch `_execution_bindings` with a same-signature function that raises `TypeError("invalid manifest field")`. Assert `write_m4b()` raises that error once and does not call the factory with alternate argument lists or log `cast_binding_fallback`.

```python
calls: list[tuple[str, tuple[str, ...]]] = []


def boom(voice_id: str, *, also: Sequence[str] = ()) -> ExecutionBindings:
    calls.append((voice_id, tuple(also)))
    raise TypeError("invalid manifest field")


with pytest.raises(TypeError, match="invalid manifest field"):
    pipeline.write_m4b(output)
assert calls == [("eponine", ())]
```

- [ ] **Step 3: Remove the fallback**

Make `_resolved_execution_bindings()` one direct call:

```python
def _resolved_execution_bindings(
    voice_id: str, *, also: Sequence[str] = ()
) -> ExecutionBindings:
    return _execution_bindings(voice_id, also=also)
```

Remove `cast_binding_fallback` logging and its stale test-only compatibility rationale.

- [ ] **Step 4: Diagnose the coverage-only DOCTYPE regression without changing policy first**

Run these commands in a clean process and preserve the first divergent traceback/configuration in the commit or PR notes:

```bash
uv run pytest --no-cov -q tests/test_epub.py::test_doctype_without_internal_subset_is_read
uv run pytest -q tests/test_epub.py::test_doctype_without_internal_subset_is_read
uv run pytest -q
```

Inspect the actual `safe_iterparse` callable and parser flags at the failing call. Change only the layer proven to override `forbid_dtd=False`; retain `forbid_entities=True` and `forbid_external=True`.

- [ ] **Step 5: Add deterministic coverage-mode regressions**

Keep parametrized EPUB 2 and EPUB 3 DOCTYPE tests, and add an entity-declaration test in the same coverage-enabled invocation:

```python
assert inspect_epub(bare_doctype_epub).chapters[0].text == "He woke."
with pytest.raises(SourceError, match=ErrorCode.MALFORMED_EPUB.value):
    inspect_epub(entity_declaration_epub)
```

Run them both with and without coverage. Update `docs/security.md` to say that entity declarations/external entities are forbidden while declaration-only EPUB DOCTYPEs are permitted; the current blanket statement that DTDs are forbidden contradicts the intended parser policy.

- [ ] **Step 6: Run focused seam and parser tests**

Run: `uv run pytest tests/test_epub.py tests/test_execution.py tests/test_cache.py tests/test_series_resolution.py`

Expected: PASS under coverage, with no binding-fallback behavior.

- [ ] **Step 7: Commit the isolated fixes**

```bash
git add src/kenkui/pipeline.py src/kenkui/_epub/parser.py docs/security.md tests
git commit -m "fix: make bindings and EPUB parsing deterministic"
```

### Task 6: Repair static test typing and certify the branch

**Files:**
- Modify: `tests/test_pipeline.py`
- Modify: `tests/test_series_resolution.py`
- Modify: `tests/test_spoken_lexicon_files.py`
- Modify: `tests/test_series_validation.py`
- Modify: `tests/test_voice_traits.py`
- Modify: all Python files named by `uv run ruff format --check .`

**Interfaces:**
- Consumes: current operation union types, optional `Pipeline._resolved`, and `VoiceVariety` literals.
- Produces: zero mypy diagnostics and Ruff-formatted source/tests without weakening type checking or coverage thresholds.

- [ ] **Step 1: Write no new behavior tests; repair test type contracts directly**

For operation tuples, narrow before attribute access:

```python
operation = pipeline.operations[0]
assert isinstance(operation, SpokenForm)
assert operation.lexicon_id == "lexicon-v1"
```

For resolved state, establish non-`None` locally before access:

```python
resolved_state = resolved._resolved  # noqa: SLF001
assert resolved_state is not None
assert resolved_state.cast_assignments["javert"] == expected
```

For validation fixtures, annotate operation sequences as `tuple[Operation, ...]`. Replace invalid `"safetensors"` `VoiceRecord.variety` fixture data with the declared `"pre-compiled"` variety.

- [ ] **Step 2: Verify the typing gate is initially failing, then make it clean**

Run: `uv run mypy`

Expected before repairs: the existing 60 errors. Expected after repairs: `Success: no issues found`.

- [ ] **Step 3: Format only after all source edits are complete**

Run:

```bash
uv run ruff format .
uv run ruff format --check .
uv run ruff check .
```

Expected: both checks pass; inspect formatter changes to ensure they are mechanical only.

- [ ] **Step 4: Run the full documented gate in one clean environment**

Run:

```bash
uv run ruff format --check . && \
uv run ruff check . && \
uv run mypy && \
uv run pytest && \
uv run mkdocs build --strict
```

Expected: every command exits zero. Then run `KENKUI_RUN_NATIVE=1 uv run pytest --no-cov -m native tests/test_native_ffmpeg.py` when host FFmpeg/FFprobe are installed, to preserve the documented native acceptance tier.

- [ ] **Step 5: Commit certification repairs**

```bash
git add tests pyproject.toml src docs
git commit -m "test: restore strict quality gates"
```

## Plan Self-Review

- **Spec coverage:** Task 1 covers source-authority and resolved-state invariants; Task 2 covers source-structural pauses; Tasks 3–4 cover metadata and rights lifecycle/cache identity; Task 5 covers the TypeError seam, EPUB behavior, and security documentation; Task 6 restores all documented gates.
- **No scope expansion:** No new renderer, cache backend, public provisioning API, provider, or dependency is proposed.
- **Ordering:** Each behavioral fix lands with a focused regression before the global gate is repaired; Task 6 is last so formatter/type work does not conceal functional regressions.
- **Open diagnostic constraint:** The coverage-only DOCTYPE failure must be root-caused in Task 5 before changing parser behavior. The plan intentionally forbids speculative XML hardening changes.
