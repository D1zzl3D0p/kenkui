"""Contract tests for WP4 pure semantic execution planning."""

from __future__ import annotations

import hashlib
import inspect
import pickle
from dataclasses import FrozenInstanceError, replace
from pathlib import Path

import pytest

import kenkui as kk
from kenkui._domain import planning
from kenkui._domain.grid import build_grid
from kenkui._domain.grid_packing import FallbackCut
from kenkui._domain.planning import (
    NORMALIZATION_SCHEMA_VERSION,
    PARSER_SCHEMA_VERSION,
    PLANNING_SCHEMA_VERSION,
    RENDER_SCHEMA_VERSION,
    CoverIntent,
    ExecutionPlan,
    compile_execution_plan,
)
from legacy_grid_folds_oracle import legacy_chunks

SOURCE_HASH = "1" * 64
MODEL_REVISION = "pocket-tts/model@0123456789abcdef"
EXPECTED_CHUNK_COUNT = 2


def _pipeline(*, title: str | None = None) -> kk.Pipeline:
    pipeline = kk.epub("ignored-location.epub").assign_voice("fixture").tts()
    return pipeline if title is None else pipeline.metadata(title=title)


def _inspection(
    *, text: str = "Exact speech.", title: str = "Chapter One"
) -> kk.BookInspection:
    chapter = kk.ChapterInspection("ch-v1-one", 3, title, len(text), text)
    return kk.BookInspection(
        kk.BookMetadata("Source Title", "Source Author", cover_available=True),
        (chapter,),
    )


def _voice(**changes: object) -> kk.Voice:
    values: dict[str, object] = {
        "id": "fixture",
        "name": "Fixture Voice",
        "enabled": True,
        "provenance": "Project-owned recording by Test Speaker",
        "license_id": "CC0-1.0",
        "commercial_use_allowed": True,
        "language": "en-US",
        "content_fingerprint": "2" * 64,
        "compatible_model_revisions": (MODEL_REVISION, "pocket-tts/model@next"),
    }
    values.update(changes)
    return kk.Voice(**values)  # type: ignore[arg-type]


def _compile(
    *,
    pipeline: kk.Pipeline | None = None,
    inspection_: kk.BookInspection | None = None,
    source_hash: str = SOURCE_HASH,
    voice: kk.Voice | None = None,
    model_revision: str = MODEL_REVISION,
) -> ExecutionPlan:
    return compile_execution_plan(
        pipeline or _pipeline(),
        inspection_ or _inspection(),
        source_bytes_hash=source_hash,
        resolved_voice=voice or _voice(),
        model_revision=model_revision,
    )


def test_plan_is_deterministic_frozen_spawn_safe_and_exact() -> None:
    """Equal semantic inputs yield equal immutable and pickle-safe plans."""
    first = _compile()
    second = _compile()

    assert first == second
    assert first.semantic_fingerprint == second.semantic_fingerprint
    assert pickle.loads(pickle.dumps(first)) == first  # noqa: S301
    assert first.schema_versions.parser == PARSER_SCHEMA_VERSION
    assert first.schema_versions.normalization == NORMALIZATION_SCHEMA_VERSION
    assert first.schema_versions.planning == PLANNING_SCHEMA_VERSION
    assert first.schema_versions.render == RENDER_SCHEMA_VERSION
    assert first.source_bytes_hash == SOURCE_HASH
    assert first.model_revision == MODEL_REVISION
    assert len(first.segments) == 1
    assert first.segments[0].chapter_id == "ch-v1-one"
    assert first.segments[0].ordinal == 0
    assert first.segments[0].text == "Exact speech."
    assert first.segments[0].character_count == len(first.segments[0].text)
    expected_hash = hashlib.sha256(first.segments[0].text.encode("utf-8")).hexdigest()
    assert first.segments[0].content_hash == expected_hash
    assert first.total_speech_characters == sum(
        len(segment.text) for segment in first.segments
    )
    assert first.output.title == "Source Title"
    assert first.output.author == "Source Author"
    assert first.output.cover is CoverIntent.SOURCE
    assert first.output.source_cover_available
    assert first.output.chapters[0].title == "Chapter One"
    with pytest.raises(FrozenInstanceError):
        first.total_speech_characters = 0  # type: ignore[misc]


def test_cast_assignments_are_immutable_after_construction_and_pickle() -> None:
    """Retained caller mappings cannot change speaker voices, even after spawn."""
    original = _compile().cast
    assignments = {"alice": original.narrator.id}
    cast = replace(original, assignments=assignments)
    assignments["alice"] = "changed"
    assert cast.assignments["alice"] == original.narrator.id
    restored = pickle.loads(pickle.dumps(cast))  # noqa: S301
    assert restored == cast
    for value in (cast, restored):
        with pytest.raises(TypeError):
            value.assignments["alice"] = "changed"  # type: ignore[index]
        assert value.voice_for("alice") == original.narrator


def test_public_local_voice_metadata_supplies_a_deterministic_plan() -> None:
    """A registry voice remains a pure local input to planning."""
    plan = _compile(
        pipeline=kk.epub("ignored-location.epub").assign_voice("fixture-voice").tts(),
        voice=_voice(id="fixture-voice", compatible_model_revisions=("fixture-v1",)),
        model_revision="fixture-v1",
    )

    assert plan.cast.narrator.id == "fixture-voice"
    assert plan.model_revision == "fixture-v1"


def test_one_nonempty_ordered_segment_per_selected_spine_chapter() -> None:
    """M1 planning preserves selected order and never performs hidden chunking."""
    chapters = (
        kk.ChapterInspection("ch-b", 8, "B", 2, "B!"),
        kk.ChapterInspection("ch-a", 2, "A", 3, "A…!"),
    )
    plan = _compile(
        inspection_=kk.BookInspection(kk.BookMetadata(), chapters),
    )

    assert [segment.chapter_id for segment in plan.segments] == ["ch-b", "ch-a"]
    assert [segment.text for segment in plan.segments] == ["B!", "A…!"]
    assert len({segment.id for segment in plan.segments}) == len(chapters)
    expected_total = sum(len(chapter.text) for chapter in chapters)
    assert plan.total_speech_characters == expected_total


def test_segment_identity_uses_chapter_normalization_ordinal_and_content() -> None:
    """Segment IDs are stable semantic identities, independent of display metadata."""
    baseline = _compile()
    retitled = _compile(inspection_=_inspection(title="Renamed display chapter"))
    retext = _compile(inspection_=_inspection(text="Different speech."))

    assert baseline.segments[0].id == retitled.segments[0].id
    assert baseline.semantic_fingerprint != retitled.semantic_fingerprint
    assert baseline.segments[0].id != retext.segments[0].id
    assert NORMALIZATION_SCHEMA_VERSION in baseline.segments[0].id


@pytest.mark.parametrize(
    "text",
    [
        "x" * planning.MAX_TTS_SEGMENT_CHARACTERS,
        "x" * (planning.MAX_TTS_SEGMENT_CHARACTERS + 1),
        ("Sentence boundary. " * 200).strip(),
        "x" * (planning.MAX_TTS_SEGMENT_CHARACTERS * 3 + 17),
    ],
)
def test_chunking_is_bounded_nonempty_exact_and_deterministic(text: str) -> None:
    """Every deterministic chunk is bounded and reconstructs its chapter exactly."""
    first = _compile(inspection_=_inspection(text=text))
    second = _compile(inspection_=_inspection(text=text))

    assert first.segments == second.segments
    assert all(
        0 < len(segment.text) <= planning.MAX_TTS_SEGMENT_CHARACTERS
        for segment in first.segments
    )
    assert "".join(segment.text for segment in first.segments) == text
    assert sum(segment.character_count for segment in first.segments) == len(text)
    assert [segment.ordinal for segment in first.segments] == list(
        range(len(first.segments))
    )


def test_chunk_ids_change_after_source_change_and_include_chunk_position() -> None:
    """Chunk identity binds both source content and its stable position."""
    text = "x" * (planning.MAX_TTS_SEGMENT_CHARACTERS + 1)
    baseline = _compile(inspection_=_inspection(text=text))
    changed = _compile(inspection_=_inspection(text="y" + text[1:]))

    assert len(baseline.segments) == EXPECTED_CHUNK_COUNT
    assert baseline.segments[0].id != changed.segments[0].id
    assert baseline.segments[0].id != baseline.segments[1].id


def test_utf8_hashing_encodes_only_bounded_chunks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Hashing never materializes one encoded copy of the complete text corpus."""
    updates: list[int] = []

    class HashSpy:
        def update(self, value: bytes) -> None:
            updates.append(len(value))

        def hexdigest(self) -> str:
            return "a" * 64

    def sha256_spy() -> HashSpy:
        return HashSpy()

    monkeypatch.setattr("kenkui._domain.planning._new_sha256", sha256_spy)
    chunk_size = planning._UTF8_HASH_CHUNK_CHARACTERS  # noqa: SLF001
    text = "🙂" * (chunk_size * 2 + 1)

    plan = _compile(inspection_=_inspection(text=text))

    assert all(segment.content_hash == "a" * 64 for segment in plan.segments)
    assert all(
        len(segment.text) <= planning.MAX_TTS_SEGMENT_CHARACTERS
        for segment in plan.segments
    )
    assert "".join(segment.text for segment in plan.segments) == text
    assert max(updates) <= max(chunk_size * 4, planning.MAX_TTS_SEGMENT_CHARACTERS * 4)
    assert len(text) * 4 not in updates


@pytest.mark.parametrize(
    ("changed", "segment_identity"),
    [
        ({"source_hash": "3" * 64}, "same"),
        ({"inspection_": _inspection(text="Changed text")}, "different"),
        ({"voice": _voice(content_fingerprint="4" * 64)}, "same"),
        ({"voice": _voice(provenance="Different lawful provenance")}, "same"),
        ({"model_revision": "pocket-tts/model@next"}, "same"),
        ({"pipeline": _pipeline(title="Override title")}, "same"),
    ],
)
def test_semantic_changes_change_fingerprint(
    changed: dict[str, object], segment_identity: str
) -> None:
    """Every required semantic input participates in the plan fingerprint."""
    baseline = _compile()
    candidate = _compile(**changed)  # type: ignore[arg-type]

    assert candidate.semantic_fingerprint != baseline.semantic_fingerprint
    assert (candidate.segments[0].id == baseline.segments[0].id) == (
        segment_identity == "same"
    )


def test_selection_changes_fingerprint_and_metadata_overrides_cover_intent() -> None:
    """Selected speech and resolved output metadata are semantic plan values."""
    first = kk.ChapterInspection("ch-first", 0, "First", 3, "One")
    second = kk.ChapterInspection("ch-second", 1, "Second", 3, "Two")
    selected_first = _compile(
        inspection_=kk.BookInspection(
            kk.BookMetadata("Book", "Writer", cover_available=True), (first,)
        )
    )
    selected_second = _compile(
        pipeline=_pipeline().metadata(author="Other", cover=None),
        inspection_=kk.BookInspection(
            kk.BookMetadata("Book", "Writer", cover_available=True), (second,)
        ),
    )

    assert selected_first.semantic_fingerprint != selected_second.semantic_fingerprint
    assert selected_second.output.title == "Book"
    assert selected_second.output.author == "Other"
    assert selected_second.output.cover is CoverIntent.NONE


def test_execution_controls_and_source_location_cannot_enter_plan() -> None:
    """The compiler accepts semantic material only, never shell policy values."""
    parameters = set(inspect.signature(compile_execution_plan).parameters)
    forbidden = {
        "output",
        "output_path",
        "workers",
        "on_event",
        "callback",
        "cancel",
        "processes",
        "cache",
        "cache_policy",
    }
    assert parameters.isdisjoint(forbidden)

    relocated = replace(
        _pipeline(), source=kk.Source(Path("a/different/location.epub"), "epub")
    )
    assert _compile(pipeline=relocated) == _compile()
    assert not hasattr(_compile(), "output_path")
    assert not hasattr(_compile(), "workers")


@pytest.mark.parametrize(
    ("voice", "code"),
    [
        (None, kk.ErrorCode.VOICE_UNRESOLVED),
        (_voice(id="other"), kk.ErrorCode.VOICE_UNRESOLVED),
        (_voice(enabled=False), kk.ErrorCode.VOICE_DISABLED),
        (_voice(enabled=1), kk.ErrorCode.VOICE_PROVENANCE_REQUIRED),
        (_voice(enabled="yes"), kk.ErrorCode.VOICE_PROVENANCE_REQUIRED),
        (
            _voice(compatible_model_revisions=("other",)),
            kk.ErrorCode.VOICE_INCOMPATIBLE,
        ),
        (_voice(provenance=None), kk.ErrorCode.VOICE_PROVENANCE_REQUIRED),
        (_voice(license_id=None), kk.ErrorCode.VOICE_PROVENANCE_REQUIRED),
        (_voice(commercial_use_allowed=None), kk.ErrorCode.VOICE_PROVENANCE_REQUIRED),
        (_voice(commercial_use_allowed=1), kk.ErrorCode.VOICE_PROVENANCE_REQUIRED),
        (_voice(commercial_use_allowed="no"), kk.ErrorCode.VOICE_PROVENANCE_REQUIRED),
        (_voice(language=None), kk.ErrorCode.VOICE_PROVENANCE_REQUIRED),
        (_voice(content_fingerprint=None), kk.ErrorCode.VOICE_PROVENANCE_REQUIRED),
        (_voice(compatible_model_revisions=()), kk.ErrorCode.VOICE_PROVENANCE_REQUIRED),
    ],
)
def test_voice_resolution_failures_are_stable(
    voice: kk.Voice | None, code: kk.ErrorCode
) -> None:
    """Unresolved, disabled, incompatible, and incomplete voices fail uniformly."""
    with pytest.raises(kk.VoiceError) as caught:
        compile_execution_plan(
            _pipeline(),
            _inspection(),
            source_bytes_hash=SOURCE_HASH,
            resolved_voice=voice,
            model_revision=MODEL_REVISION,
        )
    assert caught.value.code is code


@pytest.mark.parametrize("commercial_use_allowed", [False, True])
def test_voice_plan_preserves_exact_commercial_status(
    *,
    commercial_use_allowed: bool,
) -> None:
    """Both exact boolean rights statuses are valid and represented in the plan."""
    plan = _compile(voice=_voice(commercial_use_allowed=commercial_use_allowed))

    assert plan.cast.narrator.commercial_use_allowed is commercial_use_allowed


def test_empty_or_inconsistent_speech_fails_stably() -> None:
    """Planning independently defends exact materialized speech invariants."""
    empty = kk.BookInspection(
        kk.BookMetadata(), (kk.ChapterInspection("ch-empty", 0, "Empty", 0, ""),)
    )
    inconsistent = _inspection(text="actual")
    inconsistent = replace(
        inconsistent,
        chapters=(replace(inconsistent.chapters[0], speech_characters=99),),
    )

    for inspection_ in (empty, inconsistent):
        with pytest.raises(kk.ValidationError) as caught:
            _compile(inspection_=inspection_)
        assert caught.value.code is kk.ErrorCode.EMPTY_SPEECH


def test_execution_plan_remains_internal() -> None:
    """WP4 does not widen the public facade or add Pipeline serialization APIs."""
    assert "ExecutionPlan" not in kk.__all__
    assert "VoicePlan" not in kk.__all__
    assert not hasattr(kk.Pipeline, "to_json")
    assert not hasattr(kk.Pipeline, "from_json")


def _chunks(text: str) -> list[str]:
    plan = _compile(inspection_=_inspection(text=text))
    return [segment.text for segment in plan.segments]


def _worst_separator_free_run(text: str) -> int:
    longest = run = 0
    for character in text:
        run = 0 if character in planning.POCKET_SEPARATORS else run + 1
        longest = max(longest, run)
    return longest


def test_separator_free_runs_are_split_at_line_breaks() -> None:
    """A contents page has no separator Pocket-TTS can divide, so Kenkui divides it."""
    text = "Contents\n\n" + "\n\n".join(f"Chapter {index}" for index in range(120))
    chunks = _chunks(text)

    assert "".join(chunks) == text
    assert len(chunks) > 1
    for chunk in chunks:
        assert len(chunk) <= planning.MAX_TTS_SEGMENT_CHARACTERS


def _ends_on_a_word(chunk: str) -> bool:
    """Whether a fragment ends alphanumeric, which the engine completes with a stop."""
    stripped = chunk.rstrip()
    return bool(stripped) and stripped[-1].isalnum()


def test_forced_cut_never_ends_a_fragment_on_a_word() -> None:
    """Pocket-TTS appends a full stop to any fragment ending alphanumeric.

    Every segment is its own generate_audio call, so a cut mid-sentence is
    synthesized as a completed sentence -- audible as a break three or four
    words in, which is the defect this guards. Only a break that leaves
    punctuation or a line break behind is safe to force.
    """
    text = "Ellie stared at the horizon and said nothing at all for a very long while."

    chunks = _chunks(text)

    assert "".join(chunks) == text
    assert not any(_ends_on_a_word(chunk) for chunk in chunks[:-1])


def test_comma_free_run_on_sentence_is_left_to_the_engine() -> None:
    """The run budget no longer fragments a run holding no clean boundary.

    Formerly this was split at whitespace every 48 characters to hold the
    engine's token limit. Measured with Whisper, that limit costs no words
    below 80 tokens, while the cut costs a false sentence ending on every
    fragment -- a certain defect traded for a risk that does not bite.

    Only MAX_TTS_SEGMENT_CHARACTERS still divides it, and that bound cannot be
    declined, so this text -- 1080 characters carrying no punctuation at all --
    takes exactly one unavoidable whitespace cut rather than twenty-odd.
    """
    text = "and then " * 120
    bound = planning.MAX_TTS_SEGMENT_CHARACTERS
    expected_chunks = -(-len(text) // bound)  # one cut, so two chunks
    assert len(text) > bound

    chunks = _chunks(text)

    assert "".join(chunks) == text
    assert len(chunks) == expected_chunks
    assert all(len(chunk) <= bound for chunk in chunks)


def test_comma_bearing_prose_is_not_split_by_the_run_budget() -> None:
    """Pocket-TTS sub-splits on commas, so comma-bearing prose keeps large chunks."""
    text = ("Walking east, he counted the shuttered windows, " * 12).strip()
    chunks = _chunks(text)

    assert "".join(chunks) == text
    assert len(chunks) == 1


def test_an_early_comma_does_not_strand_the_clause_after_it() -> None:
    """A short opening clause is not torn off the long run that follows it.

    This is what the budget decides. The run after the comma is well inside
    it, so nothing is cut; at 48 the same text was severed at the comma,
    because the budget was exceeded while the only clean boundary available
    lay five characters in.
    """
    text = "Yes, " + "and he walked a very long way without stopping " * 3

    chunks = _chunks(text)

    assert "".join(chunks) == text
    assert len(chunks) == 1


def test_unbroken_token_keeps_the_character_bound() -> None:
    """With no whitespace to cut on, the run budget defers to the character bound."""
    text = "x" * (planning.MAX_TTS_SEGMENT_CHARACTERS + 1)
    chunks = _chunks(text)

    assert "".join(chunks) == text
    assert len(chunks) == EXPECTED_CHUNK_COUNT


def test_line_break_is_preferred_over_a_later_space() -> None:
    """A line break is the strongest boundary, so a full-enough one wins."""
    sentence = "Sentence one. "
    fill = planning.MAX_TTS_SEGMENT_CHARACTERS * planning.MIN_BREAK_FILL
    head = sentence * (int(fill // len(sentence)) + 1)
    text = head + "\n\n" + "Sentence two. " * 43
    chunks = _chunks(text)

    assert "".join(chunks) == text
    assert chunks[0] == head + "\n\n"


def test_punctuation_adjacent_space_is_preferred_over_a_bare_space() -> None:
    """A clause boundary beats an arbitrary space inside the same window."""
    text = "Sentence. " * 90 + "word " * 30
    chunks = _chunks(text)

    assert "".join(chunks) == text
    assert chunks[0] == "Sentence. " * 90


def test_early_line_break_yields_to_a_fuller_lower_tier_boundary() -> None:
    """Boundary quality never collapses a window into a nearly empty chunk."""
    text = "Sentence one. " * 7 + "\n\n" + "Sentence two. " * 80
    chunks = _chunks(text)

    assert "".join(chunks) == text
    assert "\n" in chunks[0]
    assert len(chunks[0]) >= planning.MAX_TTS_SEGMENT_CHARACTERS // 2


def test_dashes_break_text_that_offers_no_whitespace_at_all() -> None:
    """With no space or line break, the dash family still beats a mid-word cut."""
    text = "a-" * 700
    chunks = _chunks(text)

    assert "".join(chunks) == text
    assert all(chunk.endswith("-") for chunk in chunks[:-1])


def test_token_dense_separator_free_text_is_split_at_its_hyphens() -> None:
    """Catalogue text offers no ".!?,;:" the engine can divide, so Kenkui divides it.

    The hyphen tier is clean -- each fragment ends in "-", so the engine
    appends no full stop and the cut carries no false sentence ending.
    """
    budget = planning.MAX_TTS_SEGMENT_CHARACTERS
    text = "a-" * (budget // 2 + 100)

    chunks = _chunks(text)

    assert "".join(chunks) == text
    assert len(chunks) > 1
    assert all(chunk.endswith("-") for chunk in chunks[:-1])
    for chunk in chunks:
        assert len(chunk) <= budget


def test_grid_packing_changes_boundaries_without_changing_spoken_content() -> None:
    """The hierarchy may improve cuts while preserving the exact speech stream."""
    paragraph = "Alpha sentence. " * 40
    text = f"{paragraph}\n\n{paragraph}"
    legacy = legacy_chunks("ch-v1-one", text)
    packed = tuple(_chunks(text))

    assert packed != legacy
    assert [len(chunk) for chunk in legacy] == [994, 288]
    assert [len(chunk) for chunk in packed] == [642, 640]
    assert "".join(packed) == "".join(legacy) == text


def test_whitespace_only_spans_are_absorbed_before_bounded_packing() -> None:
    """Unspeakable carries cannot recombine bounded fallback output."""
    whitespace = " " * 1200
    text = f'"A."{whitespace}"B."'
    chapter = _inspection(text=text).chapters[0]
    first_end = 4
    second_start = first_end + len(whitespace)
    spans = (
        planning.SpeakerSpan(chapter.id, 0, first_end, "a"),
        planning.SpeakerSpan(chapter.id, first_end, second_start, None),
        planning.SpeakerSpan(chapter.id, second_start, len(text), "b"),
    )
    plan = compile_execution_plan(
        _pipeline(),
        kk.BookInspection(kk.BookMetadata("Fixture"), (chapter,)),
        source_bytes_hash=SOURCE_HASH,
        resolved_voice=_voice(),
        model_revision=MODEL_REVISION,
        cast_voices=(_voice(id="a-voice"), _voice(id="b-voice")),
        assignments={"a": "a-voice", "b": "b-voice"},
        spans=spans,
    )

    assert [segment.speaker_id for segment in plan.segments] == ["a", "b"]
    assert "".join(segment.text for segment in plan.segments) == text
    assert all(
        len(segment.text) <= planning.MAX_TTS_SEGMENT_CHARACTERS
        for segment in plan.segments
    )
    origins: list[planning._SegmentSource] = []
    replayed, _silence = planning._compile_segments(  # noqa: SLF001
        (chapter,), spans, plan.cast, origins=origins
    )
    assert replayed == plan.segments
    assert [(item.canonical_start, item.canonical_end) for item in origins] == [
        (0, 208),
        (208, len(text)),
    ]
    assert origins[0].fallback_cut_after is FallbackCut.WHITESPACE

    with_gap = compile_execution_plan(
        kk.epub("book.epub")
        .silence(
            900,
            where={"chapter": chapter.id, "sentence": 1, "phrase": 2},
        )
        .assign_voice("fixture")
        .tts(),
        kk.BookInspection(kk.BookMetadata("Fixture"), (chapter,)),
        source_bytes_hash=SOURCE_HASH,
        resolved_voice=_voice(),
        model_revision=MODEL_REVISION,
        cast_voices=(_voice(id="a-voice"), _voice(id="b-voice")),
        assignments={"a": "a-voice", "b": "b-voice"},
        spans=spans,
    )
    assert with_gap.segments == plan.segments
    assert with_gap.trailing_silence_ms == (900, 0)


def test_zero_gap_on_unspeakable_span_keeps_text_and_identity() -> None:
    """An inaudible zero-gap boundary settles without changing TTS content."""
    text = '"A." "B."'
    chapter = _inspection(text=text).chapters[0]
    spans = (
        planning.SpeakerSpan(chapter.id, 0, 4, "a"),
        planning.SpeakerSpan(chapter.id, 4, 5, None),
        planning.SpeakerSpan(chapter.id, 5, len(text), "b"),
    )
    base = kk.epub("book.epub").assign_voice("fixture").tts()
    zero = (
        kk.epub("book.epub")
        .silence(
            0,
            where={
                "chapter": chapter.id,
                "paragraph": 1,
                "line": 1,
                "sentence": 1,
                "phrase": 2,
            },
        )
        .assign_voice("fixture")
        .tts()
    )

    def compile_with(pipeline: kk.Pipeline) -> ExecutionPlan:
        return compile_execution_plan(
            pipeline,
            kk.BookInspection(kk.BookMetadata("Fixture"), (chapter,)),
            source_bytes_hash=SOURCE_HASH,
            resolved_voice=_voice(),
            model_revision=MODEL_REVISION,
            cast_voices=(_voice(id="a-voice"), _voice(id="b-voice")),
            assignments={"a": "a-voice", "b": "b-voice"},
            spans=spans,
        )

    plain_plan = compile_with(base)
    zero_plan = compile_with(zero)

    assert [segment.text for segment in plain_plan.segments] == ['"A."', ' "B."']
    assert zero_plan.segments == plain_plan.segments
    assert zero_plan.trailing_silence_ms == plain_plan.trailing_silence_ms == (0, 0)


def test_planning_reports_grid_edges_and_characterized_emergency_cuts() -> None:
    """Ordinary plan boundaries are grid edges; leaf fallbacks stay explicit."""
    ordinary_text = ("Sentence boundary. " * 90).strip()
    ordinary_inspection = _inspection(text=ordinary_text)
    ordinary_plan = _compile(inspection_=ordinary_inspection)
    ordinary_origins: list[planning._SegmentSource] = []
    replayed, _silences = planning._compile_segments(  # noqa: SLF001
        ordinary_inspection.chapters,
        (),
        ordinary_plan.cast,
        origins=ordinary_origins,
    )
    edges = {
        edge
        for unit in build_grid(ordinary_inspection.chapters[0])
        for edge in (unit.start, unit.end)
    }

    assert replayed == ordinary_plan.segments
    assert all(origin.fallback_cut_after is None for origin in ordinary_origins)
    assert all(origin.canonical_end in edges for origin in ordinary_origins[:-1])

    token = "x" * (planning.MAX_TTS_SEGMENT_CHARACTERS + 1)
    token_inspection = _inspection(text=token)
    token_plan = _compile(inspection_=token_inspection)
    token_origins: list[planning._SegmentSource] = []
    replayed, _silences = planning._compile_segments(  # noqa: SLF001
        token_inspection.chapters,
        (),
        token_plan.cast,
        origins=token_origins,
    )

    assert replayed == token_plan.segments
    assert token_origins[0].fallback_cut_after is FallbackCut.HARD_TOKEN
    assert token_origins[0].canonical_end not in {
        unit.end for unit in build_grid(token_inspection.chapters[0])
    }
