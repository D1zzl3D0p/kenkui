# Multi-Voice Casting Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let one Kenkui render speak in many voices — inferring characters, attributing dialogue, casting voices automatically, and rendering the result through the existing single renderer.

**Architecture:** Attribution is a *resolved input* to the pure planner, exactly like voice resolution already is, so multi-voice adds operations rather than a second pipeline. Casting is a pure list-coloring pass over a chapter co-occurrence graph, with methods supplying candidate voices to one shared solver. One engine holds N cheap conditioning states, so a cast costs ~6.5 MB per voice rather than a second 225 MB model.

**Tech Stack:** Python 3.11+, pocket-tts 2.1.0, LiteLLM, SQLite, PyTorch (transitive), FastAPI + Pydantic (server), React + TypeScript + Vite (web).

**Spec:** `docs/superpowers/specs/2026-08-20-multi-voice-design.md`

## Global Constraints

- **Run everything through `uv run`.** A bare `python3` resolves `kenkui` to an empty namespace package, because the repo root contains a `kenkui/` directory with no `__init__.py`. The symptom is `AttributeError: module 'kenkui' has no attribute 'book'`, not an ImportError.
- **Any script reaching `Pipeline.write()` needs an `if __name__ == "__main__":` guard** — rendering uses `spawn` workers that re-import the main module.
- Ruff runs `select = ["ALL"]` at line length 88, target `py311`. New per-file ignores go in `pyproject.toml` `[tool.ruff.lint.per-file-ignores]` and must be justified.
- Mypy runs at `python_version = 3.12`. Everything is fully annotated; `Any` requires a comment.
- `pocket-tts==2.1.0` is pinned exactly. Do not bump it.
- Public values are frozen dataclasses with `slots=True`.
- `tests/conftest.py` redirects the cache root per test via `isolated_cache_root`. **Any new store must be redirected there too**, or tests will read and mutate the developer's real cache.
- Error codes are `ErrorCode` enum members with snake_case string values, added to `src/kenkui/errors.py`.
- **Single-voice output must stay byte-identical.** `CHUNKING_SCHEMA_VERSION` (`tts-chunks-v2`) must not change. Task 3 asserts this and every later task must keep it passing.
- Full gate before any commit: `uv run ruff check . && uv run mypy && uv run pytest`.

---

## File Structure

**Renderer substrate (Tasks 1–3)**

| File | Responsibility |
|---|---|
| `src/kenkui/_tts/pocket.py` | MOD — split `PocketEngineConfig` into engine identity + `VoiceAsset` collection; `_voice_state` becomes a per-voice mapping |
| `src/kenkui/_tts/protocols.py` | MOD — `SynthesisTask` gains `voice_asset_sha256` so a worker knows which state to use |
| `src/kenkui/_domain/planning.py` | MOD — `VoicePlan` becomes a cast; `SpeechSegment` gains `speaker_id` and `voice_id`; span-aware chunking |
| `src/kenkui/_execution/cache.py` | MOD — voice material moves from plan level to per-segment |

**Casting (Tasks 4–5)**

| File | Responsibility |
|---|---|
| `src/kenkui/voices/types.py` | MOD — `Voice.perceived_gender` |
| `src/kenkui/voices/registry.py` | MOD — sourced gender per catalog entry |
| `src/kenkui/voices/provision.py` | MOD — `add_voice(..., perceived_gender=...)` |
| `src/kenkui/_domain/casting.py` | NEW — method candidate filters and the shared list-coloring solver. Pure: no I/O, no model |

**Characters (Tasks 6–8)**

| File | Responsibility |
|---|---|
| `src/kenkui/_characters/store.py` | NEW — `casting.sqlite3` read/write |
| `src/kenkui/_characters/quotes.py` | NEW — deterministic quote-span extraction, model-free |
| `src/kenkui/_characters/prompts.py` | NEW — versioned prompt text and `PROMPT_VERSION` |
| `src/kenkui/_characters/llm.py` | NEW — LiteLLM call, bounded retry, strict JSON validation |
| `src/kenkui/_characters/infer.py` | NEW — character roster from model output |
| `src/kenkui/_characters/attribution.py` | NEW — speaker per span, Unknown fallback |
| `src/kenkui/_characters/__init__.py` | NEW — `resolve_attribution`, the single shell entry point |

**Wiring and surface (Tasks 9–11)**

| File | Responsibility |
|---|---|
| `src/kenkui/_domain/operations.py` | MOD — `InferCharacters`, `AttributeQuotes`, `AssignVoices` |
| `src/kenkui/pipeline.py` | MOD — the three operations, `resolve()`, `inspect()` enrichment |
| `src/kenkui/events.py` | MOD — `CastResolved` |
| `src/kenkui/inspection.py` | MOD — optional casting fields |
| `src/kenkui/__init__.py` | MOD — exports |
| `kenkui-server-v2/src/kenkui_server/...` | MOD — capabilities, `CharacterCasting`, allowlist |
| `kenkui-web-v2/src/...` | MOD — casting UI, regenerated OpenAPI types |

---

## Task 1: Split voice identity out of the engine config

**Files:**
- Modify: `src/kenkui/_tts/pocket.py:62-99` (`PocketEngineConfig`, `semantic_material`), `:132-183` (`_validate_fields`), `:440-470` (`_manifest`), `:590-620` (`_make_snapshot`)
- Modify: `src/kenkui/_tts/production.py:86-130` (manifest → config construction)
- Test: `tests/test_pocket_engine_config.py` (new), `tests/test_production_manifest_v2.py` (extend)

**Interfaces:**
- Produces: `VoiceAsset(path, sha256, variety, provenance, license_id, rights, commercial_use_allowed)`; `PocketEngineConfig(model_root, config_path, model_revision, package_version, files, voices: tuple[VoiceAsset, ...], cloning_capable, sample_rate_hz, device, timeout_seconds)`; `PocketEngineConfig.voice_by_sha(sha256) -> VoiceAsset`.
- Consumed by: Tasks 2, 3.

Today `PocketEngineConfig` inlines exactly one voice (`voice_asset_path`, `voice_asset_sha256`, `voice_variety`, `voice_provenance`, `voice_license_id`, `voice_rights`, `commercial_use_allowed`). That single-voice shape is the only reason one engine cannot speak twice.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_pocket_engine_config.py
from __future__ import annotations

import pytest

from kenkui._tts.pocket import PocketEngineConfig, PocketManifestFile, VoiceAsset

_SHA_A = "a" * 64
_SHA_B = "b" * 64


def _asset(sha256: str, name: str) -> VoiceAsset:
    return VoiceAsset(
        path=f"/abs/voices/{name}.safetensors",
        sha256=sha256,
        variety="built-in",
        provenance="kyutai/pocket-tts-without-voice-cloning",
        license_id="CC-BY-4.0",
        rights="Review the VCTK terms before commercial use.",
        commercial_use_allowed=False,
    )


def _config(*assets: VoiceAsset) -> PocketEngineConfig:
    return PocketEngineConfig(
        model_root="/abs/engines/english",
        config_path="/abs/engines/english/english.yaml",
        model_revision="c" * 40,
        package_version="2.1.0",
        files=(PocketManifestFile("english.yaml", 10, "d" * 64),),
        voices=assets,
        cloning_capable=False,
        sample_rate_hz=24000,
    )


def test_config_holds_many_voices_and_resolves_by_digest():
    config = _config(_asset(_SHA_A, "eponine"), _asset(_SHA_B, "charles"))
    assert config.voice_by_sha(_SHA_B).path.endswith("charles.safetensors")


def test_unknown_digest_is_rejected():
    config = _config(_asset(_SHA_A, "eponine"))
    with pytest.raises(KeyError):
        config.voice_by_sha(_SHA_B)


def test_semantic_material_is_order_independent():
    forward = _config(_asset(_SHA_A, "eponine"), _asset(_SHA_B, "charles"))
    reverse = _config(_asset(_SHA_B, "charles"), _asset(_SHA_A, "eponine"))
    assert forward.semantic_material() == reverse.semantic_material()


def test_semantic_material_separates_engine_from_voices():
    material = _config(_asset(_SHA_A, "eponine")).semantic_material()
    assert "voices" in material
    assert "voice_asset_sha256" not in material
```

Order independence matters because the cache key is derived from this material. Two runs that cast the same voices in a different order must hit the same cache.

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_pocket_engine_config.py -v`
Expected: FAIL with `ImportError: cannot import name 'VoiceAsset'`

- [ ] **Step 3: Write the minimal implementation**

```python
# src/kenkui/_tts/pocket.py — replace the single-voice fields
@dataclass(frozen=True, slots=True)
class VoiceAsset:
    """One renderable speaker embedding and the rights recorded against it."""

    path: str
    sha256: str
    variety: str
    provenance: str
    license_id: str
    rights: str
    commercial_use_allowed: bool

    def semantic_material(self) -> dict[str, object]:
        """Return semantic identity without machine-specific absolute paths."""
        return {
            "commercial_use_allowed": self.commercial_use_allowed,
            "license_id": self.license_id,
            "provenance": self.provenance,
            "rights": self.rights,
            "sha256": self.sha256,
            "variety": self.variety,
        }


@dataclass(frozen=True, slots=True)
class PocketEngineConfig:
    model_root: str
    config_path: str
    model_revision: str
    package_version: str
    files: tuple[PocketManifestFile, ...]
    voices: tuple[VoiceAsset, ...]
    cloning_capable: bool
    sample_rate_hz: int
    device: str = "cpu"
    timeout_seconds: float = 300.0

    def voice_by_sha(self, sha256: str) -> VoiceAsset:
        """Resolve one voice asset by content digest."""
        for voice in self.voices:
            if voice.sha256 == sha256:
                return voice
        raise KeyError(sha256)

    def semantic_material(self) -> dict[str, object]:
        """Return semantic identity without machine-specific absolute paths."""
        return {
            "config_manifest_path": _selected_config_identity(self),
            "device": self.device,
            "files": tuple(
                (item.relative_path, item.size, item.sha256) for item in self.files
            ),
            "model_revision": self.model_revision,
            "package_version": self.package_version,
            "sample_rate_hz": self.sample_rate_hz,
            # Sorted so cast ordering cannot change the cache key.
            "voices": tuple(
                voice.semantic_material()
                for voice in sorted(self.voices, key=lambda item: item.sha256)
            ),
        }
```

Update `_validate_fields` to validate every entry in `voices` with the existing per-field rules, rejecting an empty tuple and duplicate digests. Update `_manifest` and `_make_snapshot` to iterate `config.voices`, copying each asset into the snapshot as `voice-{sha256[:16]}.safetensors` and rebuilding the config with the snapshot paths.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/test_pocket_engine_config.py tests/test_production_manifest_v2.py tests/test_pocket_tts.py -v`
Expected: PASS

- [ ] **Step 5: Run the full gate**

Run: `uv run ruff check . && uv run mypy && uv run pytest`
Expected: PASS. `production.py` builds a one-element `voices` tuple, so behaviour is unchanged.

- [ ] **Step 6: Commit**

```bash
git add src/kenkui/_tts/pocket.py src/kenkui/_tts/production.py tests/test_pocket_engine_config.py tests/test_production_manifest_v2.py
git commit -m "refactor: separate voice assets from engine identity in pocket config"
```

---

## Task 2: One engine, many conditioning states

**Files:**
- Modify: `src/kenkui/_tts/pocket.py:766-800` (`_voice_state`, `synthesize`)
- Modify: `src/kenkui/_tts/protocols.py:9-17` (`SynthesisTask`)
- Modify: `src/kenkui/_tts/fake.py` (honour the new field)
- Test: `tests/test_pocket_multi_voice_engine.py` (new)

**Interfaces:**
- Consumes: `PocketEngineConfig.voice_by_sha` from Task 1.
- Produces: `SynthesisTask(segment_id, chapter_id, text, sample_rate_hz, channels, max_output_bytes, voice_asset_sha256)`; `PocketTTSEngine._voice_state(sha256) -> Any` backed by a per-digest cache.
- Consumed by: Tasks 3, 9.

The engine already derives conditioning state once and reuses it, deliberately (`pocket.py:766-782`). Making that a mapping is the whole change. `_parse_result` rejects `engine_initializations != 1` (`_execution/process_pool.py:505`), so one model per worker must be preserved — N states, one model.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_pocket_multi_voice_engine.py
from __future__ import annotations

from typing import Any

from kenkui._tts import pocket
from kenkui._tts.protocols import SynthesisTask

_SHA_A = "a" * 64
_SHA_B = "b" * 64


class _RecordingModel:
    """Counts state derivations so reuse and per-voice separation are visible."""

    def __init__(self) -> None:
        self.derived: list[str] = []

    def get_state_for_audio_prompt(self, path: Any) -> str:
        self.derived.append(str(path))
        return f"state:{path}"


def _engine(monkeypatch: Any, config: pocket.PocketEngineConfig) -> Any:
    engine = object.__new__(pocket.PocketTTSEngine)
    engine._config = config          # noqa: SLF001
    engine._model = _RecordingModel()  # noqa: SLF001
    engine._states = {}              # noqa: SLF001
    engine._snapshot = None          # noqa: SLF001
    engine._patched = []             # noqa: SLF001
    return engine


def test_each_voice_derives_its_own_state(monkeypatch, two_voice_config):
    engine = _engine(monkeypatch, two_voice_config)
    first = engine._voice_state(_SHA_A)   # noqa: SLF001
    second = engine._voice_state(_SHA_B)  # noqa: SLF001
    assert first != second
    assert len(engine._model.derived) == 2  # noqa: SLF001


def test_repeated_use_of_one_voice_derives_once(monkeypatch, two_voice_config):
    engine = _engine(monkeypatch, two_voice_config)
    engine._voice_state(_SHA_A)  # noqa: SLF001
    engine._voice_state(_SHA_A)  # noqa: SLF001
    assert len(engine._model.derived) == 1  # noqa: SLF001


def test_task_carries_the_voice_digest():
    task = SynthesisTask(
        segment_id="seg-1",
        chapter_id="ch-1",
        text="Hello.",
        sample_rate_hz=24000,
        channels=1,
        max_output_bytes=1024,
        voice_asset_sha256=_SHA_A,
    )
    assert task.voice_asset_sha256 == _SHA_A
```

Add a `two_voice_config` fixture to `tests/conftest.py` returning the Task 1 `PocketEngineConfig` with the `_SHA_A` and `_SHA_B` assets.

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_pocket_multi_voice_engine.py -v`
Expected: FAIL — `SynthesisTask` takes no `voice_asset_sha256`, and `_voice_state` takes no argument.

- [ ] **Step 3: Write the minimal implementation**

```python
# src/kenkui/_tts/protocols.py
@dataclass(frozen=True, slots=True)
class SynthesisTask:
    """Spawn-safe exact input and hard output-byte limit for one synthesis unit."""

    segment_id: str
    chapter_id: str
    text: str
    sample_rate_hz: int
    channels: int
    max_output_bytes: int
    voice_asset_sha256: str
```

```python
# src/kenkui/_tts/pocket.py — in __init__, replace `self._state = None`
self._states: dict[str, Any] = {}

def _voice_state(self, voice_asset_sha256: str) -> Any:
    """Derive one conditioning state per voice and reuse each for every segment.

    Workers hold a reusable engine and process a batch serially, so deriving
    this per segment was pure waste. The model is constructed once regardless
    of how many voices a cast holds: a language engine is ~225 MB of weights
    against ~6.5 MB per speaker embedding. The argument is a Path, never a
    str: get_state_for_audio_prompt calls download_if_necessary only on str,
    so a Path cannot reach the network even before _deny_remote intervenes.
    """
    cached = self._states.get(voice_asset_sha256)
    if cached is not None:
        return cached
    try:
        asset = self._config.voice_by_sha(voice_asset_sha256)
        state = self._model.get_state_for_audio_prompt(Path(asset.path))
    except Exception:
        self.close()
        raise VoiceError(ErrorCode.POCKET_VOICE_LOAD_FAILED) from None
    self._states[voice_asset_sha256] = state
    return state

def synthesize(self, task: SynthesisTask) -> SynthesizedAudio:
    state = self._voice_state(task.voice_asset_sha256)
    ...  # body otherwise unchanged
```

`close()` must also clear `self._states`.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/test_pocket_multi_voice_engine.py tests/test_pocket_tts.py tests/test_process_execution.py -v`
Expected: PASS

- [ ] **Step 5: Verify one model per worker still holds**

Run: `uv run pytest tests/test_process_execution.py -k engine_initializations -v`
Expected: PASS — `_parse_result` still sees exactly one initialization.

- [ ] **Step 6: Commit**

```bash
git add src/kenkui/_tts/pocket.py src/kenkui/_tts/protocols.py src/kenkui/_tts/fake.py tests/test_pocket_multi_voice_engine.py tests/conftest.py
git commit -m "feat: hold one conditioning state per voice on a single engine"
```

---

## Task 3: Per-segment voice in the plan and cache key

**Files:**
- Modify: `src/kenkui/_domain/planning.py:109-170` (`VoicePlan`, `SpeechSegment`, `ExecutionPlan`), `:298-400` (`_compile_segments`, `_segment`)
- Modify: `src/kenkui/_execution/cache.py:138-165` (`key_for`), `:967-976` (`_voice_material`)
- Modify: `src/kenkui/_execution/coordinator.py:342-400` (`_render`, task construction)
- Test: `tests/test_planning_multi_voice.py` (new), `tests/test_single_voice_regression.py` (new)

**Interfaces:**
- Consumes: `SynthesisTask.voice_asset_sha256` from Task 2.
- Produces: `SpeechSegment(id, chapter_id, ordinal, text, character_count, content_hash, speaker_id, voice_id)`; `CastPlan(narrator, unknown, voices: tuple[VoicePlan, ...], assignments: Mapping[str, str])`; `ExecutionPlan.cast: CastPlan`; `SpeakerSpan(chapter_id, start, end, character_id)`.
- Consumed by: Tasks 5, 9.

This is the task that protects everything already shipped. Attribution supplies spans that *partition* chapter text; the frozen `tts-chunks-v2` chunker runs within each span. A chapter with no dialogue is one span, so its chunks are byte-identical to today.

- [ ] **Step 1: Write the regression test first — it must pass before and after**

```python
# tests/test_single_voice_regression.py
from __future__ import annotations

from kenkui._domain.planning import CHUNKING_SCHEMA_VERSION, compile_execution_plan


def test_chunking_schema_version_is_frozen():
    assert CHUNKING_SCHEMA_VERSION == "tts-chunks-v2"


def test_single_voice_segments_are_unchanged(single_voice_plan_inputs):
    """A pipeline with no attribution must produce the exact prior segments."""
    plan = compile_execution_plan(**single_voice_plan_inputs)
    assert [segment.id for segment in plan.segments] == [
        # Paste the exact IDs printed by the pre-change run in Step 2.
    ]
    assert all(segment.speaker_id is None for segment in plan.segments)


def test_single_voice_fingerprint_is_unchanged(single_voice_plan_inputs):
    plan = compile_execution_plan(**single_voice_plan_inputs)
    assert plan.semantic_fingerprint == ""  # Paste from Step 2.
```

- [ ] **Step 2: Capture the current values before changing anything**

Run:
```bash
uv run python -c "
from tests.conftest import *  # noqa
# Build the same inputs the fixture uses and print the baseline.
from kenkui._domain.planning import compile_execution_plan
plan = compile_execution_plan(**INPUTS)
print([s.id for s in plan.segments])
print(plan.semantic_fingerprint)
"
```
Paste both outputs into the test literals above. This is the *only* way the regression has teeth — a test asserting against post-change output proves nothing.

- [ ] **Step 3: Run the regression test to verify it passes on unmodified code**

Run: `uv run pytest tests/test_single_voice_regression.py -v`
Expected: PASS. If it fails now, the captured values are wrong; fix them before proceeding.

- [ ] **Step 4: Write the failing multi-voice test**

```python
# tests/test_planning_multi_voice.py
from __future__ import annotations

from kenkui._domain.planning import SpeakerSpan, compile_execution_plan


def test_spans_partition_chapter_text_exactly(two_speaker_inputs):
    plan = compile_execution_plan(**two_speaker_inputs)
    chapter = two_speaker_inputs["inspection"].chapters[0]
    rebuilt = "".join(
        segment.text for segment in plan.segments if segment.chapter_id == chapter.id
    )
    assert rebuilt == chapter.text


def test_each_segment_carries_its_speaker_and_voice(two_speaker_inputs):
    plan = compile_execution_plan(**two_speaker_inputs)
    speakers = {segment.speaker_id for segment in plan.segments}
    assert speakers == {None, "javert"}
    narration = [s for s in plan.segments if s.speaker_id is None]
    dialogue = [s for s in plan.segments if s.speaker_id == "javert"]
    assert narration[0].voice_id == "eponine"
    assert dialogue[0].voice_id == "charles"


def test_identical_text_from_two_speakers_yields_distinct_segments(same_line_inputs):
    plan = compile_execution_plan(**same_line_inputs)
    first, second = plan.segments[0], plan.segments[1]
    assert first.text == second.text
    assert first.id != second.id
```

- [ ] **Step 5: Run it to verify it fails**

Run: `uv run pytest tests/test_planning_multi_voice.py -v`
Expected: FAIL — `SpeakerSpan` does not exist and `SpeechSegment` has no `speaker_id`.

- [ ] **Step 6: Write the implementation**

```python
# src/kenkui/_domain/planning.py
@dataclass(frozen=True, slots=True)
class SpeakerSpan:
    """One contiguous run of a chapter's normalized text with a single speaker."""

    chapter_id: str
    start: int
    end: int
    character_id: str | None  # None means narration


@dataclass(frozen=True, slots=True)
class SpeechSegment:
    """One exact ordered synthesis input for an M1 spine chapter."""

    id: str
    chapter_id: str
    ordinal: int
    text: str
    character_count: int
    content_hash: str
    speaker_id: str | None = None
    voice_id: str = ""


@dataclass(frozen=True, slots=True)
class CastPlan:
    """Resolved narrator, unknown, and per-character voices for one run."""

    narrator: VoicePlan
    unknown: VoicePlan
    voices: tuple[VoicePlan, ...]
    assignments: Mapping[str, str]  # character_id -> voice_id

    def voice_for(self, speaker_id: str | None) -> VoicePlan:
        """Resolve the voice a speaker renders in, narrating anything unassigned."""
        if speaker_id is None:
            return self.narrator
        voice_id = self.assignments.get(speaker_id)
        if voice_id is None:
            return self.unknown
        return next(voice for voice in self.voices if voice.id == voice_id)
```

`_compile_segments` gains a `spans` parameter defaulting to `()`. With no spans it builds one full-chapter narration span, so the existing path is literally unchanged:

```python
def _compile_segments(
    chapters: tuple[ChapterInspection, ...],
    spans: tuple[SpeakerSpan, ...],
    cast: CastPlan,
) -> tuple[SpeechSegment, ...]:
    """Split each span with the frozen chunker, numbering across the whole plan."""
    result: list[SpeechSegment] = []
    for chapter in chapters:
        for span in _spans_for(chapter, spans):
            text = chapter.text[span.start : span.end]
            voice = cast.voice_for(span.character_id)
            for chunk_index, chunk in enumerate(_chunk_span(chapter, text)):
                result.append(
                    _segment(
                        chapter,
                        len(result),
                        chunk_index,
                        chunk,
                        span.character_id,
                        voice.id,
                    )
                )
    return tuple(result)


def _spans_for(
    chapter: ChapterInspection, spans: tuple[SpeakerSpan, ...]
) -> tuple[SpeakerSpan, ...]:
    """Return a chapter's spans, or one narration span covering all of it."""
    owned = tuple(span for span in spans if span.chapter_id == chapter.id)
    if not owned:
        return (SpeakerSpan(chapter.id, 0, len(chapter.text), None),)
    return tuple(sorted(owned, key=lambda span: span.start))
```

`_segment` adds `speaker_id` and `voice_id` to the identity JSON **only when `speaker_id` is not None**, so single-voice digests are bit-identical:

```python
identity_fields: dict[str, object] = {
    "chapter_id": _string_identity(chapter.id),
    "chunk_index": chunk_index,
    "chunking_schema": CHUNKING_SCHEMA_VERSION,
    "content_hash": content_hash,
    "normalization": NORMALIZATION_SCHEMA_VERSION,
    "ordinal": ordinal,
    "segment_id_version": _SEGMENT_ID_VERSION,
}
if speaker_id is not None:
    identity_fields["speaker_id"] = _string_identity(speaker_id)
    identity_fields["voice_id"] = _string_identity(voice_id)
```

In `cache.py`, `_voice_material` takes the segment's voice rather than the plan's, and `key_for` uses it:

```python
def _voice_material(plan: ExecutionPlan, segment: SpeechSegment) -> dict[str, object]:
    voice = plan.cast.voice_for(segment.speaker_id)
    return {
        "commercial_use_allowed": voice.commercial_use_allowed,
        "content_fingerprint": voice.content_fingerprint,
        "id_sha256": hashlib.sha256(voice.id.encode()).hexdigest(),
        "language_sha256": hashlib.sha256(voice.language.encode()).hexdigest(),
        "license_sha256": hashlib.sha256(voice.license_id.encode()).hexdigest(),
        "provenance_sha256": hashlib.sha256(voice.provenance.encode()).hexdigest(),
    }
```

The `record_run` call site at `cache.py:108` keeps plan-level material by passing `plan.cast.narrator`.

`_render` in `_execution/coordinator.py` builds each `SynthesisTask`, so it is where a segment's `voice_id` becomes the digest the worker resolves. `VoicePlan.content_fingerprint` already holds the asset SHA-256, so the bridge is a lookup, not new state:

```python
voice = plan.cast.voice_for(segment.speaker_id)
task = SynthesisTask(
    segment_id=segment.id,
    chapter_id=segment.chapter_id,
    text=segment.text,
    sample_rate_hz=sample_rate_hz,
    channels=channels,
    max_output_bytes=max_output_bytes,
    voice_asset_sha256=voice.content_fingerprint,
)
```

Add a test asserting a dialogue segment's task carries the character's digest and a narration segment's carries the narrator's — without it, every segment would silently render in whichever voice the engine derived first.

- [ ] **Step 7: Run both suites**

Run: `uv run pytest tests/test_planning_multi_voice.py tests/test_single_voice_regression.py tests/test_planning.py tests/test_cache.py -v`
Expected: PASS — including the regression captured in Step 2 against unmodified code.

- [ ] **Step 8: Run the full gate**

Run: `uv run ruff check . && uv run mypy && uv run pytest`
Expected: PASS

- [ ] **Step 9: Commit**

```bash
git add src/kenkui/_domain/planning.py src/kenkui/_execution/cache.py tests/test_planning_multi_voice.py tests/test_single_voice_regression.py
git commit -m "feat: carry speaker and voice per segment through plan and cache"
```

---

## Task 4: Voice traits

**Files:**
- Modify: `src/kenkui/voices/types.py:24-38` (`Voice`)
- Modify: `src/kenkui/voices/registry.py:36-46` (`CatalogEntry`), `:49-185` (catalog)
- Modify: `src/kenkui/voices/provision.py` (`add_voice` signature and manifest write)
- Test: `tests/test_voice_traits.py` (new)

**Interfaces:**
- Produces: `Voice.perceived_gender: PerceivedGender`, `PerceivedGender = Literal["feminine", "masculine"] | None`; `add_voice(..., perceived_gender=None)`.
- Consumed by: Task 5.

The trait is **sourced or `None`** — never inferred from the display name, which Kenkui invented. VCTK publishes speaker gender for the twelve `p###` voices; the other nine English entries have no shipped metadata.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_voice_traits.py
from __future__ import annotations

from kenkui.voices.registry import CATALOG


def test_vctk_voices_carry_sourced_gender():
    """VCTK ships speaker-info.txt, so these twelve are documented, not guessed."""
    assert CATALOG["eponine"].perceived_gender == "feminine"   # p262, F
    assert CATALOG["charles"].perceived_gender == "masculine"  # p254, M


def test_unsourced_voices_are_none_not_guessed():
    """LibriVox and donation voices ship no metadata; a name is not evidence."""
    assert CATALOG["bill_boerst"].perceived_gender is None
    assert CATALOG["caro_davy"].perceived_gender is None
    assert CATALOG["alba"].perceived_gender is None


def test_every_entry_declares_the_field_explicitly():
    for entry in CATALOG.values():
        assert entry.perceived_gender in {"feminine", "masculine", None}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_voice_traits.py -v`
Expected: FAIL — `CatalogEntry` has no `perceived_gender`.

- [ ] **Step 3: Write the implementation**

Add `perceived_gender: PerceivedGender` to `CatalogEntry` and `Voice`. Populate the twelve VCTK entries from the VCTK `speaker-info.txt` gender column, keyed by the `p###` ID already encoded in each `_vctk(...)` filename:

| Voice | VCTK ID | Gender |
|---|---|---|
| anna | p228 | feminine |
| vera | p229 | feminine |
| fantine | p244 | feminine |
| charles | p254 | masculine |
| paul | p259 | masculine |
| eponine | p262 | feminine |
| azelma | p303 | feminine |
| george | p315 | masculine |
| mary | p333 | feminine |
| jane | p339 | feminine |
| michael | p360 | masculine |
| eve | p361 | feminine |

Verify each against the corpus before writing it down; do not transcribe this table on faith. Every other catalog entry gets `perceived_gender=None`.

`_vctk` gains a `gender` parameter. `add_voice` gains `perceived_gender: PerceivedGender = None`, persisted in the manifest voice entry and surfaced on `Voice`.

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/test_voice_traits.py tests/test_voice_registry.py tests/test_add_voice.py tests/test_list_voices.py -v`
Expected: PASS

- [ ] **Step 5: Run the full gate and commit**

```bash
uv run ruff check . && uv run mypy && uv run pytest
git add src/kenkui/voices tests/test_voice_traits.py
git commit -m "feat: record sourced perceived-gender voice traits"
```

---

## Task 5: The casting solver

**Files:**
- Create: `src/kenkui/_domain/casting.py`
- Test: `tests/test_casting_solver.py` (new)

**Interfaces:**
- Consumes: `Voice.perceived_gender` from Task 4.
- Produces: `CastingMethod = Literal["random", "gendered"]`; `CharacterProfile(id, display_name, gender, spoken_characters, chapter_ids)`; `CastingRequest(characters, pool, explicit, narrator_voice_id, unknown_voice_id, method)`; `CastingOutcome(assignments: Mapping[str, str], collisions: tuple[Collision, ...])`; `solve(request) -> CastingOutcome`; `candidates(method, character, pool) -> tuple[Voice, ...]`.
- Consumed by: Tasks 8, 9.

Pure module. No I/O, no model, no clock, no RNG. The whole file must be testable with plain values.

The model: characters are vertices, an edge joins two characters sharing a chapter, voices are colours, and a method supplies each character's admissible colours. This is list colouring on a co-occurrence graph.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_casting_solver.py
from __future__ import annotations

from kenkui._domain.casting import (
    CastingRequest,
    CharacterProfile,
    candidates,
    solve,
)
from kenkui.voices.types import Voice


def _voice(voice_id: str, gender: str | None) -> Voice:
    return Voice(
        id=voice_id,
        name=voice_id.title(),
        enabled=True,
        provenance="test",
        license_id="CC-BY-4.0",
        commercial_use_allowed=False,
        language="english",
        perceived_gender=gender,
        state="loaded",
    )


def _character(
    character_id: str, gender: str | None, chars: int, chapters: tuple[str, ...]
) -> CharacterProfile:
    return CharacterProfile(character_id, character_id.title(), gender, chars, chapters)


POOL = (
    _voice("anna", "feminine"),
    _voice("vera", "feminine"),
    _voice("charles", "masculine"),
    _voice("paul", "masculine"),
)


def _request(characters, **kwargs):
    defaults = {
        "characters": characters,
        "pool": POOL,
        "explicit": {},
        "narrator_voice_id": "eponine",
        "unknown_voice_id": "eponine",
        "method": "gendered",
    }
    return CastingRequest(**{**defaults, **kwargs})


def test_gendered_candidates_filter_by_trait():
    character = _character("darcy", "masculine", 100, ("ch1",))
    assert {v.id for v in candidates("gendered", character, POOL)} == {
        "charles",
        "paul",
    }


def test_unknown_gender_character_falls_back_to_the_whole_pool():
    character = _character("voice-in-the-dark", None, 10, ("ch1",))
    assert len(candidates("gendered", character, POOL)) == len(POOL)


def test_random_method_admits_the_whole_pool():
    character = _character("darcy", "masculine", 100, ("ch1",))
    assert len(candidates("random", character, POOL)) == len(POOL)


def test_same_chapter_characters_never_share_a_voice():
    outcome = solve(
        _request((
            _character("darcy", "masculine", 400, ("ch1",)),
            _character("bingley", "masculine", 300, ("ch1",)),
        ))
    )
    assert outcome.assignments["darcy"] != outcome.assignments["bingley"]
    assert outcome.collisions == ()


def test_characters_in_different_chapters_may_share():
    outcome = solve(
        _request((
            _character("darcy", "masculine", 400, ("ch1",)),
            _character("wickham", "masculine", 300, ("ch2",)),
        ))
    )
    assert outcome.assignments["darcy"] == outcome.assignments["wickham"]


def test_least_used_prefers_the_quietest_voice_by_speech_volume():
    """A lead must not land on the voice a talkative character already holds."""
    outcome = solve(
        _request((
            _character("darcy", "masculine", 5000, ("ch1",)),
            _character("collins", "masculine", 50, ("ch2",)),
            _character("wickham", "masculine", 4000, ("ch3",)),
        ))
    )
    assert outcome.assignments["wickham"] == outcome.assignments["collins"]
    assert outcome.assignments["wickham"] != outcome.assignments["darcy"]


def test_explicit_pins_are_constraints_not_suggestions():
    outcome = solve(
        _request(
            (
                _character("darcy", "masculine", 400, ("ch1",)),
                _character("bingley", "masculine", 300, ("ch1",)),
            ),
            explicit={"darcy": "paul"},
        )
    )
    assert outcome.assignments["darcy"] == "paul"
    assert outcome.assignments["bingley"] == "charles"


def test_reserved_roles_are_excluded_from_every_method():
    """The narrator speaks in every chapter, so it is adjacent to everyone."""
    outcome = solve(
        _request(
            (_character("darcy", "masculine", 400, ("ch1",)),),
            narrator_voice_id="charles",
            unknown_voice_id="paul",
            method="random",
        )
    )
    assert outcome.assignments["darcy"] not in {"charles", "paul"}


def test_exhausted_pool_collides_minimally_and_reports_it():
    characters = tuple(
        _character(f"man{index}", "masculine", 100 - index, ("ch1",))
        for index in range(3)
    )
    outcome = solve(_request(characters))
    assert len(set(outcome.assignments.values())) == 2
    assert len(outcome.collisions) == 1
    quietest = outcome.collisions[0]
    assert "man2" in {quietest.first, quietest.second}


def test_solving_is_deterministic_across_runs():
    characters = tuple(
        _character(f"c{index}", "feminine", 500 - index * 10, ("ch1", "ch2"))
        for index in range(4)
    )
    first = solve(_request(characters))
    second = solve(_request(tuple(reversed(characters))))
    assert first.assignments == second.assignments
```

The determinism test reverses the input order deliberately: a solver that depends on input ordering rather than content is not reproducible, and the plan fingerprint depends on reproducibility.

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_casting_solver.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'kenkui._domain.casting'`

- [ ] **Step 3: Write the implementation**

```python
# src/kenkui/_domain/casting.py
"""Pure character-to-voice casting. No I/O, no model, no randomness."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

from kenkui.errors import ErrorCode, ValidationError

if TYPE_CHECKING:
    from collections.abc import Mapping

    from kenkui.voices.types import Voice

CastingMethod = Literal["random", "gendered"]
_METHODS: frozenset[str] = frozenset({"random", "gendered"})


@dataclass(frozen=True, slots=True)
class CharacterProfile:
    """One speaking character, measured in normalized speech characters."""

    id: str
    display_name: str
    gender: str | None
    spoken_characters: int
    chapter_ids: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class Collision:
    """Two characters sharing a voice inside one chapter."""

    chapter_id: str
    first: str
    second: str
    voice_id: str


@dataclass(frozen=True, slots=True)
class CastingRequest:
    characters: tuple[CharacterProfile, ...]
    pool: tuple[Voice, ...]
    explicit: Mapping[str, str]
    narrator_voice_id: str
    unknown_voice_id: str
    method: CastingMethod


@dataclass(frozen=True, slots=True)
class CastingOutcome:
    assignments: Mapping[str, str]
    collisions: tuple[Collision, ...]


def candidates(
    method: str, character: CharacterProfile, pool: tuple[Voice, ...]
) -> tuple[Voice, ...]:
    """Return the voices a method admits for one character.

    A method only answers eligibility. Colouring, weighting, and tie-breaking
    belong to the shared solver, so a future LLM or keyword method is one more
    filter here rather than a second solver.
    """
    if method not in _METHODS:
        raise ValidationError(ErrorCode.CASTING_METHOD_UNKNOWN)
    if method == "random" or character.gender is None:
        return pool
    matched = tuple(
        voice for voice in pool if voice.perceived_gender == character.gender
    )
    return matched or pool


def solve(request: CastingRequest) -> CastingOutcome:
    """Assign voices by deterministic greedy list colouring over co-occurrence."""
    reserved = {request.narrator_voice_id, request.unknown_voice_id}
    pool = tuple(voice for voice in request.pool if voice.id not in reserved)
    if not pool:
        raise ValidationError(ErrorCode.CAST_POOL_EMPTY)

    by_id = {character.id: character for character in request.characters}
    for character_id in request.explicit:
        if character_id not in by_id:
            raise ValidationError(ErrorCode.CHARACTER_UNKNOWN)

    neighbours = _neighbours(request.characters)
    assignments: dict[str, str] = dict(request.explicit)
    load: dict[str, int] = {voice.id: 0 for voice in pool}
    for character_id, voice_id in assignments.items():
        load[voice_id] = load.get(voice_id, 0) + by_id[character_id].spoken_characters

    collisions: list[Collision] = []
    remaining = [c for c in request.characters if c.id not in assignments]
    while remaining:
        character = _next_character(remaining, neighbours, assignments)
        remaining.remove(character)
        admissible = candidates(request.method, character, pool)
        taken = {
            assignments[other]
            for other in neighbours[character.id]
            if other in assignments
        }
        free = tuple(voice for voice in admissible if voice.id not in taken)
        chosen = min(
            free or admissible, key=lambda voice: (load.get(voice.id, 0), voice.id)
        )
        if not free:
            collisions.extend(
                _collisions(character, chosen.id, neighbours, assignments, by_id)
            )
        assignments[character.id] = chosen.id
        load[chosen.id] = load.get(chosen.id, 0) + character.spoken_characters
    return CastingOutcome(assignments, tuple(collisions))


def _neighbours(
    characters: tuple[CharacterProfile, ...],
) -> dict[str, frozenset[str]]:
    """Build the co-occurrence graph: an edge per shared chapter."""
    edges: dict[str, set[str]] = {character.id: set() for character in characters}
    for index, first in enumerate(characters):
        for second in characters[index + 1 :]:
            if set(first.chapter_ids) & set(second.chapter_ids):
                edges[first.id].add(second.id)
                edges[second.id].add(first.id)
    return {key: frozenset(value) for key, value in edges.items()}


def _next_character(
    remaining: list[CharacterProfile],
    neighbours: dict[str, frozenset[str]],
    assignments: Mapping[str, str],
) -> CharacterProfile:
    """Pick the most constrained character, breaking ties on speech volume then ID.

    Every key is content-derived and totally ordered, so the result cannot
    depend on the order the caller supplied.
    """
    return min(
        remaining,
        key=lambda character: (
            -len(
                {
                    assignments[other]
                    for other in neighbours[character.id]
                    if other in assignments
                }
            ),
            -character.spoken_characters,
            character.id,
        ),
    )


def _collisions(
    character: CharacterProfile,
    voice_id: str,
    neighbours: dict[str, frozenset[str]],
    assignments: Mapping[str, str],
    by_id: Mapping[str, CharacterProfile],
) -> list[Collision]:
    """Record every same-chapter clash the forced assignment introduces."""
    found: list[Collision] = []
    for other in sorted(neighbours[character.id]):
        if assignments.get(other) != voice_id:
            continue
        shared = sorted(set(character.chapter_ids) & set(by_id[other].chapter_ids))
        found.extend(
            Collision(chapter_id, character.id, other, voice_id)
            for chapter_id in shared
        )
    return found
```

Add `CASTING_METHOD_UNKNOWN`, `CAST_POOL_EMPTY`, and `CHARACTER_UNKNOWN` to `ErrorCode`.

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_casting_solver.py -v`
Expected: PASS, all eleven

- [ ] **Step 5: Run the full gate and commit**

```bash
uv run ruff check . && uv run mypy && uv run pytest
git add src/kenkui/_domain/casting.py src/kenkui/errors.py tests/test_casting_solver.py
git commit -m "feat: add deterministic list-colouring casting solver"
```

---

## Task 6: The casting store

**Files:**
- Create: `src/kenkui/_characters/__init__.py`, `src/kenkui/_characters/store.py`
- Modify: `tests/conftest.py` (redirect the store path)
- Test: `tests/test_casting_store.py` (new)

**Interfaces:**
- Produces: `default_store_path() -> Path`; `AttributionRecord(attribution_id, book_id, model_id, prompt_version, params_json, characters, spans)`; `CastRecord(cast_id, attribution_id, method, narrator_voice_id, unknown_voice_id, assignments)`; `read_attribution(key) -> AttributionRecord | None`; `write_attribution(record)`; `read_cast(key)`, `write_cast(record)`; `list_castings()`, `remove_casting(id)`, `remove_attribution(id)`.
- Consumed by: Tasks 8, 9.

Lives at `<cache_root>/casting.sqlite3`, beside `manifest.json` and `cache.sqlite3`. Named for what it holds, unlike the opaque `cache.sqlite3`.

**This is not the fail-open PCM cache.** Reads fail soft to a miss so corruption is recoverable by recompute; writes fail loud, because silently losing a cast means a re-render silently re-casts the book.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_casting_store.py
from __future__ import annotations

import pytest

from kenkui._characters import store


def test_attribution_round_trips(attribution_record):
    store.write_attribution(attribution_record)
    loaded = store.read_attribution(attribution_record.attribution_id)
    assert loaded == attribution_record


def test_missing_attribution_reads_as_none():
    assert store.read_attribution("f" * 64) is None


def test_two_casts_share_one_attribution(attribution_record, cast_record_factory):
    """Cast keys hang below attribution keys, so exploring casts is free."""
    store.write_attribution(attribution_record)
    gendered = cast_record_factory(method="gendered")
    random_cast = cast_record_factory(method="random")
    store.write_cast(gendered)
    store.write_cast(random_cast)
    assert gendered.cast_id != random_cast.cast_id
    assert len(store.list_castings()) == 2
    assert store.read_attribution(attribution_record.attribution_id) is not None


def test_remove_casting_keeps_the_attribution(attribution_record, cast_record_factory):
    store.write_attribution(attribution_record)
    cast = cast_record_factory(method="gendered")
    store.write_cast(cast)
    store.remove_casting(cast.cast_id)
    assert store.list_castings() == ()
    assert store.read_attribution(attribution_record.attribution_id) is not None


def test_remove_attribution_cascades_to_its_casts(
    attribution_record, cast_record_factory
):
    store.write_attribution(attribution_record)
    store.write_cast(cast_record_factory(method="gendered"))
    store.remove_attribution(attribution_record.attribution_id)
    assert store.read_attribution(attribution_record.attribution_id) is None
    assert store.list_castings() == ()


def test_corrupt_database_reads_as_a_miss(isolated_cache_root):
    (isolated_cache_root / "casting.sqlite3").write_bytes(b"not a database")
    assert store.read_attribution("a" * 64) is None


def test_corrupt_database_fails_loud_on_write(isolated_cache_root, attribution_record):
    (isolated_cache_root / "casting.sqlite3").write_bytes(b"not a database")
    with pytest.raises(OSError):
        store.write_attribution(attribution_record)


def test_store_lives_beside_the_manifest(isolated_cache_root):
    assert store.default_store_path() == isolated_cache_root / "casting.sqlite3"
```

Add fixtures `attribution_record` and `cast_record_factory` to `tests/conftest.py`, and extend `isolated_cache_root` to redirect `store.default_store_path` the same way it already redirects the manifest.

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_casting_store.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'kenkui._characters'`

- [ ] **Step 3: Write the schema and implementation**

```sql
CREATE TABLE IF NOT EXISTS books(
    book_id TEXT PRIMARY KEY, title TEXT, author TEXT, created_ns INTEGER NOT NULL);

CREATE TABLE IF NOT EXISTS attributions(
    attribution_id TEXT PRIMARY KEY,
    book_id TEXT NOT NULL REFERENCES books(book_id),
    model_id TEXT NOT NULL, prompt_version TEXT NOT NULL,
    params_json TEXT NOT NULL, created_ns INTEGER NOT NULL);

CREATE TABLE IF NOT EXISTS characters(
    attribution_id TEXT NOT NULL REFERENCES attributions(attribution_id)
        ON DELETE CASCADE,
    character_id TEXT NOT NULL, display_name TEXT NOT NULL, gender TEXT,
    spoken_characters INTEGER NOT NULL, span_count INTEGER NOT NULL,
    PRIMARY KEY (attribution_id, character_id));

CREATE TABLE IF NOT EXISTS character_chapters(
    attribution_id TEXT NOT NULL REFERENCES attributions(attribution_id)
        ON DELETE CASCADE,
    character_id TEXT NOT NULL, chapter_id TEXT NOT NULL,
    PRIMARY KEY (attribution_id, character_id, chapter_id));

CREATE TABLE IF NOT EXISTS quote_spans(
    attribution_id TEXT NOT NULL REFERENCES attributions(attribution_id)
        ON DELETE CASCADE,
    chapter_id TEXT NOT NULL, start INTEGER NOT NULL, end INTEGER NOT NULL,
    character_id TEXT,
    PRIMARY KEY (attribution_id, chapter_id, start));

CREATE TABLE IF NOT EXISTS casts(
    cast_id TEXT PRIMARY KEY,
    attribution_id TEXT NOT NULL REFERENCES attributions(attribution_id)
        ON DELETE CASCADE,
    method TEXT NOT NULL, narrator_voice_id TEXT NOT NULL,
    unknown_voice_id TEXT NOT NULL, created_ns INTEGER NOT NULL);

CREATE TABLE IF NOT EXISTS cast_assignments(
    cast_id TEXT NOT NULL REFERENCES casts(cast_id) ON DELETE CASCADE,
    character_id TEXT NOT NULL, voice_id TEXT NOT NULL,
    pinned INTEGER NOT NULL,
    PRIMARY KEY (cast_id, character_id));
```

Open with `PRAGMA foreign_keys = ON` so the cascades fire. Write through a temporary file, `fsync`, and `rename` at mode `0600` inside a `0700` directory, matching `voices/manifest.py`. Key derivation:

```python
def attribution_key(
    book_id: str, model_id: str, prompt_version: str, params: Mapping[str, object]
) -> str:
    """Key attribution by everything that determines its content."""
    return _sha256_json(
        {
            "book_id": book_id,
            "model_id": model_id,
            "params": params,
            "prompt_version": prompt_version,
        }
    )


def cast_key(
    attribution_id: str,
    method: str,
    explicit: Mapping[str, str],
    narrator_voice_id: str,
    unknown_voice_id: str,
) -> str:
    """Key a cast below its attribution, so cast variants share one model pass."""
    return _sha256_json(
        {
            "attribution_id": attribution_id,
            "explicit": dict(sorted(explicit.items())),
            "method": method,
            "narrator": narrator_voice_id,
            "unknown": unknown_voice_id,
        }
    )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_casting_store.py -v`
Expected: PASS

- [ ] **Step 5: Verify test isolation actually holds**

Run: `uv run pytest tests/test_casting_store.py -v && ls ~/Library/Caches/kenkui/v1/ 2>/dev/null | grep casting`
Expected: PASS, and **no** `casting.sqlite3` in the real user cache. If one appears, `isolated_cache_root` is not redirecting the store.

- [ ] **Step 6: Run the full gate and commit**

```bash
uv run ruff check . && uv run mypy && uv run pytest
git add src/kenkui/_characters tests/test_casting_store.py tests/conftest.py
git commit -m "feat: add casting store for attributions and cast assignments"
```

---

## Task 7: Model-free quote extraction

**Files:**
- Create: `src/kenkui/_characters/quotes.py`
- Test: `tests/test_quote_extraction.py` (new)

**Interfaces:**
- Produces: `extract_spans(chapter_id, text) -> tuple[SpeakerSpan, ...]` returning spans that exactly partition `text`, with `character_id=None` everywhere (speakers are assigned in Task 8).
- Consumed by: Task 8.

Deterministic and model-free, so it is directly testable and its correctness never depends on a provider. It only finds *where* dialogue is — not *who* speaks it.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_quote_extraction.py
from __future__ import annotations

import pytest

from kenkui._characters.quotes import extract_spans

CASES = [
    pytest.param('He said, "Go away." She left.', 3, id="straight-quotes"),
    pytest.param("He said, “Go away.” She left.", 3, id="curly-quotes"),
    pytest.param('"Go away."', 1, id="quote-only"),
    pytest.param("No dialogue at all.", 1, id="narration-only"),
    pytest.param('"A," said B, "and C."', 5, id="interrupted-quote"),
]


@pytest.mark.parametrize(("text", "expected"), CASES)
def test_spans_partition_the_text_exactly(text: str, expected: int):
    spans = extract_spans("ch1", text)
    assert len(spans) == expected
    assert "".join(text[span.start : span.end] for span in spans) == text


@pytest.mark.parametrize(("text", "expected"), CASES)
def test_spans_are_ordered_and_non_overlapping(text: str, expected: int):
    spans = extract_spans("ch1", text)
    assert all(
        earlier.end == later.start for earlier, later in zip(spans, spans[1:])
    )


def test_empty_text_yields_no_spans():
    assert extract_spans("ch1", "") == ()


def test_apostrophes_are_not_quotes():
    """A possessive must not open a span, or every sentence becomes dialogue."""
    spans = extract_spans("ch1", "Darcy's horse and Bingley's carriage.")
    assert len(spans) == 1


def test_extraction_is_deterministic():
    text = '"A," said B, "and C." Then D spoke.'
    assert extract_spans("ch1", text) == extract_spans("ch1", text)
```

The partition assertion is the load-bearing one: Task 3's segmentation depends on spans exactly reconstructing chapter text, and a gap or overlap silently drops or duplicates audio.

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_quote_extraction.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Write the implementation**

Scan for balanced quote pairs across `"` and the curly pair `“`/`”`. Emit a dialogue span for each matched pair including its delimiters, and a narration span for every gap between them, including leading and trailing text. An apostrophe between word characters is never a delimiter. An unbalanced quote is treated as narration rather than swallowing the rest of the chapter.

```python
def extract_spans(chapter_id: str, text: str) -> tuple[SpeakerSpan, ...]:
    """Partition chapter text into alternating narration and dialogue spans.

    Speakers are not assigned here; every span carries character_id=None. The
    partition is exact, because segmentation reconstructs chapter text by
    concatenating span chunks.
    """
    if not text:
        return ()
    spans: list[SpeakerSpan] = []
    cursor = 0
    for start, end in _quote_ranges(text):
        if start > cursor:
            spans.append(SpeakerSpan(chapter_id, cursor, start, None))
        spans.append(SpeakerSpan(chapter_id, start, end, None))
        cursor = end
    if cursor < len(text):
        spans.append(SpeakerSpan(chapter_id, cursor, len(text), None))
    return tuple(spans)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_quote_extraction.py -v`
Expected: PASS

- [ ] **Step 5: Run the full gate and commit**

```bash
uv run ruff check . && uv run mypy && uv run pytest
git add src/kenkui/_characters/quotes.py tests/test_quote_extraction.py
git commit -m "feat: add deterministic model-free quote span extraction"
```

---

## Task 8: Character inference and attribution

**Files:**
- Create: `src/kenkui/_characters/prompts.py`, `llm.py`, `infer.py`, `attribution.py`
- Modify: `src/kenkui/_characters/__init__.py` (`resolve_attribution`)
- Modify: `pyproject.toml` (add `litellm`)
- Test: `tests/test_character_llm.py`, `tests/test_attribution.py` (new)

**Interfaces:**
- Consumes: `extract_spans` (Task 7); `store.read_attribution` / `write_attribution` / `attribution_key` (Task 6).
- Produces: `PROMPT_VERSION: str`; `complete_json(model, prompt, schema, *, client) -> dict`; `resolve_attribution(inspection, source_hash, model_id, *, client=None, cancel=None) -> AttributionRecord`.
- Consumed by: Task 9.

Every default test uses a fake client. **No test in the default suite may touch a provider.**

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_character_llm.py
from __future__ import annotations

import pytest

from kenkui._characters.llm import complete_json
from kenkui.errors import ErrorCode, ModelError


class FakeClient:
    """Deterministic stand-in; records calls so retry behaviour is observable."""

    def __init__(self, *responses: str) -> None:
        self.responses = list(responses)
        self.calls: list[str] = []

    def complete(self, model: str, prompt: str) -> str:
        self.calls.append(prompt)
        if not self.responses:
            raise RuntimeError("exhausted")
        return self.responses.pop(0)


SCHEMA = {"characters": list}


def test_valid_json_is_returned():
    client = FakeClient('{"characters": ["elizabeth-bennet"]}')
    assert complete_json("fake/model", "p", SCHEMA, client=client) == {
        "characters": ["elizabeth-bennet"]
    }


def test_malformed_json_retries_then_fails():
    client = FakeClient("not json", "still not json", "nope")
    with pytest.raises(ModelError) as error:
        complete_json("fake/model", "p", SCHEMA, client=client)
    assert error.value.code is ErrorCode.MODEL_RESPONSE_INVALID
    assert len(client.calls) == 3


def test_transient_failure_is_retried_then_succeeds():
    class Flaky(FakeClient):
        def complete(self, model: str, prompt: str) -> str:
            self.calls.append(prompt)
            if len(self.calls) == 1:
                raise TimeoutError
            return '{"characters": []}'

    client = Flaky()
    assert complete_json("fake/model", "p", SCHEMA, client=client) == {
        "characters": []
    }
    assert len(client.calls) == 2


def test_schema_violation_is_rejected_not_coerced():
    client = FakeClient('{"characters": "elizabeth"}')
    with pytest.raises(ModelError) as error:
        complete_json("fake/model", "p", SCHEMA, client=client)
    assert error.value.code is ErrorCode.MODEL_RESPONSE_INVALID
```

```python
# tests/test_attribution.py
from __future__ import annotations

import pytest

from kenkui._characters import resolve_attribution
from kenkui.cancellation import CancellationToken
from kenkui.errors import CancelledError


def test_roster_is_normalised_to_stable_slugs(two_chapter_inspection, fake_client):
    record = resolve_attribution(
        two_chapter_inspection, "a" * 64, "fake/model", client=fake_client
    )
    assert [c.id for c in record.characters] == ["elizabeth-bennet", "fitzwilliam-darcy"]


def test_unattributed_spans_stay_none(two_chapter_inspection, fake_client):
    """Unknown is explicit, never a guess."""
    record = resolve_attribution(
        two_chapter_inspection, "a" * 64, "fake/model", client=fake_client
    )
    assert any(span.character_id is None for span in record.spans)


def test_spans_still_partition_each_chapter(two_chapter_inspection, fake_client):
    record = resolve_attribution(
        two_chapter_inspection, "a" * 64, "fake/model", client=fake_client
    )
    for chapter in two_chapter_inspection.chapters:
        owned = [s for s in record.spans if s.chapter_id == chapter.id]
        rebuilt = "".join(chapter.text[s.start : s.end] for s in owned)
        assert rebuilt == chapter.text


def test_second_call_hits_the_store_and_makes_no_model_call(
    two_chapter_inspection, fake_client
):
    resolve_attribution(
        two_chapter_inspection, "a" * 64, "fake/model", client=fake_client
    )
    before = len(fake_client.calls)
    resolve_attribution(
        two_chapter_inspection, "a" * 64, "fake/model", client=fake_client
    )
    assert len(fake_client.calls) == before


def test_cancellation_is_honoured_between_chapters(
    two_chapter_inspection, fake_client
):
    """A long book must not ignore Ctrl-C for the whole attribution pass."""
    token = CancellationToken()
    token.cancel()
    with pytest.raises(CancelledError):
        resolve_attribution(
            two_chapter_inspection,
            "a" * 64,
            "fake/model",
            client=fake_client,
            cancel=token,
        )
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_character_llm.py tests/test_attribution.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Add the dependency**

```toml
# pyproject.toml
dependencies = [
    "defusedxml==0.7.1",
    "litellm>=1.0",
    "pocket-tts==2.1.0",
    "pyyaml>=6.0",
]
```

Run: `uv lock && uv sync`

- [ ] **Step 4: Write the implementation**

`prompts.py` holds the prompt text and `PROMPT_VERSION = "characters-v1"`. Bump it whenever prompt text changes — it is an attribution key input, so a stale key would silently reuse output from a different prompt.

`llm.py` exposes `complete_json(model, prompt, schema, *, client=None)`, defaulting to a thin LiteLLM wrapper. Bounded retry: three attempts, exponential backoff, no retry on schema violations after the final attempt. Raise `ModelError(ErrorCode.MODEL_CALL_FAILED)` for transport failures and `ModelError(ErrorCode.MODEL_RESPONSE_INVALID)` for unparseable or schema-violating output. Never log prompt or response bodies — they contain book text.

`infer.py` normalises model output into `CharacterProfile` values: slugify display names to stable IDs (`Elizabeth Bennet` → `elizabeth-bennet`), cluster aliases, sort by ID for determinism, and drop anything without a usable name.

`attribution.py` walks each chapter's spans from `extract_spans`, sends bounded context around each dialogue span, and assigns `character_id` or leaves it `None`.

`__init__.py` wires it together:

```python
def resolve_attribution(
    inspection: BookInspection,
    source_hash: str,
    model_id: str,
    *,
    client: object | None = None,
    cancel: CancellationToken | None = None,
) -> AttributionRecord:
    """Return stored attribution, else derive it and store it.

    Called only from the shell. The pure planner receives the finished record,
    exactly as it receives resolved_voice today.
    """
    params = {"temperature": 0.0}
    key = store.attribution_key(source_hash, model_id, PROMPT_VERSION, params)
    cached = store.read_attribution(key)
    if cached is not None:
        return cached
    characters = infer_characters(inspection, model_id, client=client, cancel=cancel)
    spans: list[SpeakerSpan] = []
    for chapter in inspection.chapters:
        if cancel is not None:
            cancel.raise_if_cancelled()
        spans.extend(
            attribute_chapter(chapter, characters, model_id, client=client)
        )
    record = AttributionRecord(
        attribution_id=key,
        book_id=source_hash,
        model_id=model_id,
        prompt_version=PROMPT_VERSION,
        params_json=_canonical_json(params),
        characters=tuple(characters),
        spans=tuple(spans),
    )
    store.write_attribution(record)
    return record
```

Add `MODEL_CALL_FAILED`, `MODEL_RESPONSE_INVALID`, and `ATTRIBUTION_UNAVAILABLE` to `ErrorCode`.

- [ ] **Step 5: Run tests to verify they pass**

Run: `uv run pytest tests/test_character_llm.py tests/test_attribution.py -v`
Expected: PASS

- [ ] **Step 6: Verify the default suite reaches no provider**

Run: `uv run pytest tests/ -p no:randomly -x -q 2>&1 | tail -5`
Expected: PASS with no network. Confirm no `LITELLM_*` or provider key is read by running with the environment cleared.

- [ ] **Step 7: Run the full gate and commit**

```bash
uv run ruff check . && uv run mypy && uv run pytest
git add src/kenkui/_characters pyproject.toml uv.lock tests/test_character_llm.py tests/test_attribution.py
git commit -m "feat: add character inference and quote attribution"
```

---

## Task 9: Pipeline wiring

**Files:**
- Modify: `src/kenkui/_domain/operations.py` (three operations), `src/kenkui/pipeline.py`, `src/kenkui/events.py`, `src/kenkui/inspection.py`, `src/kenkui/validation.py`, `src/kenkui/__init__.py`, `src/kenkui/_execution/coordinator.py`
- Test: `tests/test_pipeline_multi_voice.py`, `tests/test_resolve.py` (new)

**Interfaces:**
- Consumes: everything from Tasks 3, 5, 6, 8.
- Produces: `Pipeline.infer_characters(model)`, `.attribute_quotes(model)`, `.assign_voices(narrator=..., unknown=None, cast=None, method="gendered")`, `.resolve()`; `CastResolved` event; `list_castings`, `remove_casting`, `remove_attribution` exports.

This is where the spec's central claim gets tested: multi-voice is a composition, not an architecture.

- [ ] **Step 1: Write the failing tests for `resolve()`**

```python
# tests/test_resolve.py
from __future__ import annotations

import kenkui as kk


def _pipeline(epub_path, model="fake/model"):
    return (
        kk.epub(epub_path)
        .infer_characters(model=model)
        .attribute_quotes(model=model)
        .assign_voices(narrator="eponine", method="gendered")
    )


def test_building_a_pipeline_performs_no_work(epub_path, fake_client_spy):
    _pipeline(epub_path)
    assert fake_client_spy.calls == []


def test_resolve_returns_a_pipeline_and_leaves_the_receiver_alone(epub_path):
    original = _pipeline(epub_path)
    resolved = original.resolve()
    assert isinstance(resolved, kk.Pipeline)
    assert resolved is not original
    assert original.inspect().casting.state == "pending"


def test_resolve_preserves_intent(epub_path):
    original = _pipeline(epub_path)
    assert original.resolve().operations == original.operations


def test_resolve_is_idempotent(epub_path, fake_client_spy):
    resolved = _pipeline(epub_path).resolve()
    before = len(fake_client_spy.calls)
    resolved.resolve()
    assert len(fake_client_spy.calls) == before


def test_resolved_values_never_enter_the_fingerprint(epub_path):
    """resolve() must be an optimisation, never a semantic change."""
    original = _pipeline(epub_path)
    assert _fingerprint(original) == _fingerprint(original.resolve())


def test_appending_an_operation_drops_resolved_values(epub_path):
    resolved = _pipeline(epub_path).resolve()
    assert resolved.tts().inspect().casting.state == "resolved"  # store still hits


def test_inspect_never_reaches_the_network(epub_path, fake_client_spy):
    _pipeline(epub_path).inspect()
    assert fake_client_spy.calls == []


def test_write_resolves_internally_without_a_prior_resolve(epub_path, tmp_path):
    output = tmp_path / "out.m4b"
    _pipeline(epub_path).tts().write(output)
    assert output.exists()
```

```python
# tests/test_pipeline_multi_voice.py
from __future__ import annotations

import kenkui as kk


def test_assign_voice_is_assign_voices_with_an_empty_cast(epub_path):
    """One VoicePlan and one renderer for both castings."""
    single = kk.epub(epub_path).assign_voice("eponine").tts()
    plural = kk.epub(epub_path).assign_voices(narrator="eponine").tts()
    assert _fingerprint(single) == _fingerprint(plural)


def test_unknown_defaults_to_the_narrator_voice(epub_path):
    pipeline = kk.epub(epub_path).assign_voices(narrator="eponine")
    assert _cast(pipeline).unknown.id == "eponine"


def test_unknown_can_be_set_independently(epub_path):
    pipeline = kk.epub(epub_path).assign_voices(narrator="eponine", unknown="paul")
    assert _cast(pipeline).unknown.id == "paul"


def test_operation_order_does_not_matter(epub_path):
    forward = (
        kk.epub(epub_path)
        .infer_characters(model="fake/model")
        .attribute_quotes(model="fake/model")
        .assign_voices(narrator="eponine")
    )
    reverse = (
        kk.epub(epub_path)
        .assign_voices(narrator="eponine")
        .attribute_quotes(model="fake/model")
        .infer_characters(model="fake/model")
    )
    assert _fingerprint(forward) == _fingerprint(reverse)


def test_cast_resolved_event_precedes_any_rendering(epub_path, tmp_path):
    seen: list[object] = []
    (
        kk.epub(epub_path)
        .assign_voices(narrator="eponine")
        .tts()
        .write(tmp_path / "out.m4b", on_event=seen.append)
    )
    kinds = [type(event).__name__ for event in seen]
    assert kinds.index("CastResolved") < kinds.index("StageStarted")


def test_cancelling_from_the_cast_event_renders_nothing(epub_path, tmp_path):
    token = kk.CancellationToken()
    output = tmp_path / "out.m4b"

    def watch(event: object) -> None:
        if type(event).__name__ == "CastResolved":
            token.cancel()

    with pytest.raises(kk.CancelledError):
        (
            kk.epub(epub_path)
            .assign_voices(narrator="eponine")
            .tts()
            .write(output, on_event=watch, cancel=token)
        )
    assert not output.exists()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_resolve.py tests/test_pipeline_multi_voice.py -v`
Expected: FAIL — `Pipeline` has no `infer_characters`

- [ ] **Step 3: Add the operations**

```python
# src/kenkui/_domain/operations.py
@dataclass(frozen=True, slots=True)
class InferCharacters:
    """Derive a character roster with the named model."""

    model_id: str


@dataclass(frozen=True, slots=True)
class AttributeQuotes:
    """Assign a speaker to each quote span with the named model."""

    model_id: str


@dataclass(frozen=True, slots=True)
class AssignVoices:
    """Cast narrator, unknown, and characters to voices."""

    narrator_voice_id: str
    unknown_voice_id: str
    cast: tuple[tuple[str, str], ...]  # sorted pairs — hashable and deterministic
    method: str
```

`AssignVoice` is retained as an alias constructed by `assign_voice`, or `assign_voice` simply builds `AssignVoices` directly. Prefer the latter: one operation type, one planner branch.

- [ ] **Step 4: Add the pipeline methods and `resolve()`**

```python
# src/kenkui/pipeline.py
@dataclass(frozen=True, slots=True)
class Pipeline:
    source: Source
    operations: tuple[Operation, ...] = ()
    _resolved: Resolved | None = None  # not intent; never in the fingerprint

    def infer_characters(self, model: str) -> Pipeline:
        """Add character inference intent."""
        return self._append(InferCharacters(model), before_tts=True)

    def attribute_quotes(self, model: str) -> Pipeline:
        """Add dialogue attribution intent."""
        return self._append(AttributeQuotes(model), before_tts=True)

    def assign_voices(
        self,
        *,
        narrator: str | Voice,
        unknown: str | Voice | None = None,
        cast: Mapping[str, str | Voice] | None = None,
        method: str = "gendered",
    ) -> Pipeline:
        """Assign narrator, unknown, and per-character voices."""
        narrator_id = _voice_id(narrator)
        return self._append(
            AssignVoices(
                narrator_voice_id=narrator_id,
                unknown_voice_id=_voice_id(unknown) if unknown else narrator_id,
                cast=tuple(sorted((k, _voice_id(v)) for k, v in (cast or {}).items())),
                method=method,
            ),
            before_tts=True,
        )

    def resolve(self) -> Pipeline:
        """Resolve voices, attribution, and cast, returning a new Pipeline.

        Optional: write() resolves internally. This exists so a caller can pay
        the model cost early and inspect the outcome. It is an effect verb —
        it reaches the network and writes the store — but it is immutable,
        idempotent, and intent-preserving.
        """
        if self._resolved is not None:
            return self
        return Pipeline(self.source, self.operations, _resolve_all(self))

    def _append(self, operation: Operation, *, before_tts: bool = False) -> Pipeline:
        """Create a branch with one pure validated operation append.

        Resolved values are dropped: changing intent invalidates them, and
        re-resolution against a populated store is a lookup, so the
        conservative rule costs nothing.
        """
        return Pipeline(
            self.source, append_unique(self.operations, operation, before_tts=before_tts)
        )
```

In `write_m4b`, replace the single binding line with resolution that reuses carried values:

```python
resolved = self._resolved if self._resolved is not None else _resolve_all(self)
return execute_sequential(
    self, output_path, resolved=resolved, on_event=on_event, cancel=cancel, ...
)
```

`_resolve_all` performs pool resolution, `resolve_attribution`, and `casting.solve`, in that order, with `cancel.raise_if_cancelled()` between. **One implementation, two entry points** — `resolve()` and `write()` both call it.

- [ ] **Step 5: Add the event and emit it before rendering**

```python
# src/kenkui/events.py
@dataclass(frozen=True, slots=True)
class CastResolved:
    """The character-to-voice assignment resolved for this run."""

    sequence: int
    stage: str
    narrator_voice_id: str
    unknown_voice_id: str
    assignments: tuple[tuple[str, str], ...] = ()
```

Add it to the `ExecutionEvent` union and `__all__`. In `_execution/coordinator.py`, emit it from `_Emitter` immediately after `emit_started()` and before the first `emit_stage_started()`, then check cancellation so a callback that cancels aborts before workers spawn. Single-voice runs emit it with an empty `assignments`.

- [ ] **Step 6: Enrich inspection and validation**

`BookInspection` gains `casting: CastingInspection | None = None`, where `CastingInspection` carries `state: Literal["pending", "resolved"]`, the roster, and the assignments. `inspect()` fills it from `self._resolved` or a store read, and never resolves.

`render_intent_errors` gains a presence check: `AttributeQuotes` without `InferCharacters` yields `ATTRIBUTION_UNAVAILABLE`. This is a presence rule, not an ordering rule — `append_unique`'s only ordering constraint stays `before_tts`.

`validate_voices` gains the single-language check from spec decision 17. Every voice named by an `AssignVoices` operation — narrator, unknown, and each explicit cast entry — must share a `language`, or `CAST_LANGUAGE_MIXED` is raised. Add `CAST_LANGUAGE_MIXED` to `ErrorCode`. The plural engine structure from Task 1 stays; only validation is singular, so a second language later is a validation change rather than a redesign.

```python
def test_mixing_languages_is_rejected(epub_path):
    pipeline = kk.epub(epub_path).assign_voices(
        narrator="eponine", cast={"giovanni-character": "giovanni"}
    )
    issue = pipeline.validate().issues[0]
    assert issue.code is kk.ErrorCode.CAST_LANGUAGE_MIXED
```

- [ ] **Step 6b: Log collisions without surfacing them**

Per spec decision 11, `_resolve_all` logs every collision the solver reports and emits nothing to the caller:

```python
for collision in outcome.collisions:
    log_event(
        _LOGGER,
        "cast_collision",
        level=logging.WARNING,
        context={
            "boundary": "casting",
            "chapter_id": collision.chapter_id,
            "first": collision.first,
            "second": collision.second,
            "voice_id": collision.voice_id,
        },
    )
```

This uses `observability.log_event`, **not** the public `Warning` execution event — that one is sequenced into `on_event` and would reach the browser.

```python
def test_collisions_are_logged_and_not_surfaced(epub_path, caplog, tmp_path):
    seen: list[object] = []
    _exhausted_pool_pipeline(epub_path).write(
        tmp_path / "out.m4b", on_event=seen.append
    )
    assert any(record.msg == "cast_collision" for record in caplog.records)
    assert not any(type(event).__name__ == "Warning" for event in seen)
```

- [ ] **Step 7: Run the tests**

Run: `uv run pytest tests/test_resolve.py tests/test_pipeline_multi_voice.py tests/test_pipeline.py tests/test_single_voice_regression.py -v`
Expected: PASS, including the Task 3 regression

- [ ] **Step 8: Render a real two-voice book end to end**

Run a short EPUB through a real render with two loaded voices and a fake attribution client, then confirm both voices are audible and chapter markers are intact. Watch `%CPU` per worker during the run: roughly 100% each is right; well above that means the thread cap is not taking effect, which multi-voice makes easier to trip because each worker now holds more state.

- [ ] **Step 9: Run the full gate and commit**

```bash
uv run ruff check . && uv run mypy && uv run pytest
git add src/kenkui tests/test_resolve.py tests/test_pipeline_multi_voice.py
git commit -m "feat: wire multi-voice casting through the pipeline"
```

---

## Task 10: Server character casting

**Files:**
- Modify: `kenkui-server-v2/src/kenkui_server/config.py:41-62`, `jobs/models.py:17-24,50-67`, `api/schemas.py:43-58`, `api/jobs.py:53,72`, `jobs/pipeline.py:17`, `storage/repositories.py:44,56`
- Modify: `kenkui-server-v2/openapi/v1.json` (regenerated)
- Test: `kenkui-server-v2/tests/test_character_casting.py` (new)

**Interfaces:**
- Consumes: `kk.Pipeline.assign_voices`, `kk.CastResolved` from Task 9.
- Produces: `CastingCapabilities.modes: list[Literal["single", "characters"]]`; `CharacterCasting(narrator_voice_id, unknown_voice_id, cast, method, model_id)`.

- [ ] **Step 1: Write the failing tests**

```python
# kenkui-server-v2/tests/test_character_casting.py
from __future__ import annotations


def test_default_deployment_advertises_single_only(client):
    modes = client.get("/v1/capabilities").json()["casting"]["modes"]
    assert modes == ["single"]


def test_characters_advertised_only_when_configured(character_client):
    modes = character_client.get("/v1/capabilities").json()["casting"]["modes"]
    assert modes == ["single", "characters"]


def test_character_job_requires_an_allowlisted_model(character_client, source_id):
    response = character_client.post(
        "/v1/jobs/preflight",
        json={
            "sourceId": source_id,
            "chapters": ["ch1"],
            "casting": {
                "narratorVoiceId": "eponine",
                "method": "gendered",
                "modelId": "evil/model",
            },
            "tts": {"normalizeText": True},
            "output": {"format": "m4b"},
        },
    )
    assert response.status_code == 422
    assert response.json()["error"]["code"] == "model_not_allowed"


def test_character_casting_round_trips_through_row_mapping(character_spec):
    from kenkui_server.storage import repositories

    row = repositories.spec_to_row(character_spec)
    assert repositories.row_to_spec(row) == character_spec


def test_single_voice_jobs_are_unaffected(client, source_id):
    response = client.post(
        "/v1/jobs/preflight",
        json={
            "sourceId": source_id,
            "chapters": ["ch1"],
            "casting": {"voiceId": "eponine"},
            "tts": {"normalizeText": True},
            "output": {"format": "m4b"},
        },
    )
    assert response.status_code == 200


def test_preflight_makes_no_model_call(character_client, source_id, model_spy):
    """Preflight creates neither a Job nor a reservation, so it must stay free."""
    character_client.post("/v1/jobs/preflight", json=CHARACTER_PAYLOAD)
    assert model_spy.calls == []
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd kenkui-server-v2 && uv run pytest tests/test_character_casting.py -v`
Expected: FAIL — capabilities expose `mode`, not `modes`

- [ ] **Step 3: Write the implementation**

```python
# config.py
class CastingCapabilities(BaseModel):
    """Casting modes supported by this server."""

    modes: list[Literal["single", "characters"]] = Field(
        default_factory=lambda: ["single"]
    )
```

```python
# jobs/models.py
@dataclass(frozen=True, slots=True)
class CharacterCasting:
    """Character-voice casting intent for one job."""

    narrator_voice_id: str
    unknown_voice_id: str
    cast: tuple[tuple[str, str], ...]
    method: str
    model_id: str

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "narrator_voice_id",
            _required(self.narrator_voice_id, "invalid_voice_id"),
        )
        object.__setattr__(self, "model_id", _required(self.model_id, "invalid_model_id"))


Casting = SingleVoiceCasting | CharacterCasting
```

`JobSpec.casting` becomes `Casting`. `repositories.py` persists a tagged shape — `{"kind": "single", ...}` or `{"kind": "characters", ...}` — and reconstructs on read. Existing rows have no `kind`, so treat a missing key as `"single"`.

`jobs/pipeline.py` branches:

```python
def pipeline_from_job(spec: JobSpec, source: str | os.PathLike[str]) -> kk.Pipeline:
    """Reconstruct a fresh public Kenkui pipeline from one immutable job spec."""
    pipeline = kk.book(source).select_chapters(*spec.chapters)
    if spec.tts.normalize_text:
        pipeline = pipeline.normalize_text()
    if isinstance(spec.casting, SingleVoiceCasting):
        return pipeline.assign_voice(spec.casting.voice_id).tts()
    return (
        pipeline.infer_characters(model=spec.casting.model_id)
        .attribute_quotes(model=spec.casting.model_id)
        .assign_voices(
            narrator=spec.casting.narrator_voice_id,
            unknown=spec.casting.unknown_voice_id,
            cast=dict(spec.casting.cast),
            method=spec.casting.method,
        )
        .tts()
    )
```

Add `model_allowlist: list[str]` to server config. `_preflight` rejects a `model_id` outside it with `model_not_allowed`, and continues to build its own minimal pipeline so it stays free of model calls. Forward `CastResolved` onto the existing SSE event stream.

- [ ] **Step 4: Regenerate the OpenAPI document**

Run: `cd kenkui-server-v2 && uv run python -m kenkui_server.openapi > openapi/v1.json`
Expected: `CastingCapabilities.modes` replaces `mode`; `CastingRequest` gains the optional character fields.

- [ ] **Step 5: Run tests and commit**

```bash
cd kenkui-server-v2
uv run ruff check . && uv run mypy && uv run pytest
git add src tests openapi
git commit -m "feat: add capability-gated character casting"
```

---

## Task 11: Web casting UI

**Files:**
- Modify: `kenkui-web-v2/src/api/generated/v1.ts` (regenerated), `src/components/casting.tsx`, `src/pages/new-job.tsx:29,53`, `src/pages/job.tsx`
- Test: `kenkui-web-v2/tests/casting.test.tsx` (new)

**Interfaces:**
- Consumes: the Task 10 OpenAPI contract.

Per spec section 13, v1 offers narrator, unknown, and method — **not** per-character overrides, because the roster does not exist until attribution runs and preflight is contractually free. The `CastResolved` event gives the browser a read-only cast view during the job instead.

- [ ] **Step 1: Regenerate types from the server contract**

Run: `cd kenkui-web-v2 && npm run generate:api`
Expected: `VoiceResponse` unchanged; `CastingCapabilities` now has `modes`.

- [ ] **Step 2: Write the failing tests**

```tsx
// kenkui-web-v2/tests/casting.test.tsx
import { render, screen } from "@testing-library/react";
import { Casting } from "../src/components/casting";

const VOICES = [
  { id: "eponine", name: "Eponine", language: "english" },
  { id: "charles", name: "Charles", language: "english" },
];

test("single-mode server offers no character controls", () => {
  render(<Casting modes={["single"]} voices={VOICES} value={{}} onChange={() => {}} />);
  expect(screen.queryByLabelText("Casting method")).toBeNull();
});

test("character-capable server offers narrator, unknown, and method", () => {
  render(
    <Casting modes={["single", "characters"]} voices={VOICES} value={{}} onChange={() => {}} />,
  );
  expect(screen.getByLabelText("Narrator")).toBeInTheDocument();
  expect(screen.getByLabelText("Unknown speaker")).toBeInTheDocument();
  expect(screen.getByLabelText("Casting method")).toBeInTheDocument();
});

test("unknown speaker defaults to following the narrator", () => {
  render(
    <Casting modes={["single", "characters"]} voices={VOICES} value={{ narratorVoiceId: "eponine" }} onChange={() => {}} />,
  );
  expect(screen.getByLabelText("Unknown speaker")).toHaveValue("eponine");
});

test("no per-character override controls exist in v1", () => {
  render(
    <Casting modes={["single", "characters"]} voices={VOICES} value={{}} onChange={() => {}} />,
  );
  expect(screen.queryByTestId("character-override")).toBeNull();
});
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `cd kenkui-web-v2 && npm test -- casting`
Expected: FAIL — `Casting` takes a `mode` string, not `modes`

- [ ] **Step 4: Implement the component and wire the payload**

Replace `casting.tsx` with a real form. `new-job.tsx:29` builds a `casting` payload of either `{ voiceId }` or `{ narratorVoiceId, unknownVoiceId, method, modelId }` depending on the selected mode. `job.tsx` renders the `CastResolved` SSE event as a read-only cast table.

- [ ] **Step 5: Run unit and E2E tests**

Run: `cd kenkui-web-v2 && npm test && npx playwright test`
Expected: PASS, including the existing capability-shell E2E against a real local server.

- [ ] **Step 6: Commit**

```bash
cd kenkui-web-v2
git add src tests
git commit -m "feat: add capability-gated casting controls"
```

---

## Task 12: Documentation

**Files:**
- Modify: `docs/models-and-voices.md`, `docs/usage.md`, `docs/architecture.md`, `docs/pocket-tts-adapter.md`, `README.md`
- Test: `uv run mkdocs build --strict`

**Interfaces:**
- Consumes: the finished public API from Tasks 4, 5, 6, 9.

Spec section 16. Docs land last because they describe the shipped surface, not the planned one.

- [ ] **Step 1: Correct the existing misleading line**

`docs/usage.md:163` currently reads *"There are no bulk verbs. Multi-voice work composes over `list_voices()`"*. That sentence is about **bulk provisioning**, not casting, and now that casting exists it reads as a capability claim about something else entirely. Rewrite it to say bulk *provisioning* composes over `list_voices()`, and add a separate casting section.

- [ ] **Step 2: Document casting in `usage.md`**

Cover `assign_voices`, the `random` and `gendered` methods, the reserved narrator and unknown roles, `resolve()` as an optional early-payment verb, and `list_castings` / `remove_casting` / `remove_attribution` — including that `remove_attribution` cascades and forces a new model pass while `remove_casting` is free.

- [ ] **Step 3: Document the trait in `models-and-voices.md`**

State that `perceived_gender` is sourced from VCTK speaker metadata where available and `None` otherwise, that `None` is never guessed from a display name, and that a `None` voice does not join a gendered pool. Note that nine of the twenty-one English voices are currently `None`, so gendered pools are shallower than the raw count suggests.

- [ ] **Step 4: Document the boundary in `architecture.md`**

Attribution as a resolved input beside `resolved_voice`; the unchanged worker network posture (audit hook and offline env vars stay worker-only, all model traffic is in the parent); and span-then-chunk segmentation, including why single-voice output is byte-identical.

- [ ] **Step 5: Document the engine in `pocket-tts-adapter.md`**

The engine/voice split and one model holding N conditioning states — ~225 MB of weights against ~6.5 MB per embedding, which is what makes a cast cheap.

- [ ] **Step 6: Add a multi-voice example to `README.md`**

Beside the existing single-voice example, showing `load_voice` for two voices and `assign_voices`.

- [ ] **Step 7: Build the docs and commit**

```bash
uv run mkdocs build --strict
git add docs README.md
git commit -m "docs: document multi-voice casting"
```

---

## Verification

Before declaring the feature complete:

```bash
cd kenkui           && uv run ruff check . && uv run mypy && uv run pytest
cd ../kenkui-server-v2 && uv run ruff check . && uv run mypy && uv run pytest
cd ../kenkui-web-v2    && npm test && npx playwright test
```

Then the two checks no automated suite covers:

1. **Real render, two voices, real model.** Run the opt-in suite with provider credentials and real voices loaded. Listen to the output. Confirm both voices are audible, distinct, and attached to the right speakers — attribution correctness has no automated baseline, so ears are the only instrument.
2. **Load average under a real multi-voice render.** Multi-voice puts more state in each worker. Watch per-worker `%CPU`; roughly 100% each is correct, well above means the thread cap is not taking effect.
