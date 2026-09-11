"""Meaningful fixtures frozen before quote, structure, and chunking move."""
# ruff: noqa: RUF001 - typographic punctuation is fixture data.

from __future__ import annotations

from dataclasses import dataclass
from itertools import pairwise

import pytest

import kenkui as kk
from kenkui._domain import planning
from kenkui._domain.grid import (
    DialogueRange,
    GapReason,
    build_grid,
    build_structure_index,
    dialogue_ranges,
    split_sentences,
    unit_text,
)
from kenkui._domain.operations import Pauses
from kenkui._domain.planning import (
    MAX_TTS_SEGMENT_CHARACTERS,
    SpeakerSpan,
    _gap_ms,
)
from legacy_grid_folds_oracle import (
    HEADING_AFTER,
    HEADING_BEFORE,
    LEGACY_PLAN_OBSERVATION,
    LINE,
    PARAGRAPH,
    BreakQualityObservation,
    PlanObservation,
    QuoteObservation,
    StructureObservation,
    legacy_break_quality,
    legacy_chunks,
    legacy_quote_partition,
    legacy_structure_partition,
)

CHAPTER_ID = "ch-v1-oracle"
SOURCE_HASH = "1" * 64
MODEL_REVISION = "pocket-tts/model@oracle"


@dataclass(frozen=True, slots=True)
class PauseFixture:
    """Complete legacy pause specification for structural observations."""

    chapter_ms: int = 0
    heading_before_ms: int = 0
    heading_after_ms: int = 0
    paragraph_ms: int = 0
    line_ms: int = 0


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        (
            'He said, "Go now." Then left.',
            (
                QuoteObservation(0, 9, dialogue=False),
                QuoteObservation(9, 18, dialogue=True),
                QuoteObservation(18, 29, dialogue=False),
            ),
        ),
        (
            "Before “Go now.” After.",
            (
                QuoteObservation(0, 7, dialogue=False),
                QuoteObservation(7, 16, dialogue=True),
                QuoteObservation(16, 23, dialogue=False),
            ),
        ),
        (
            "“She said ‘go now’ twice.”",
            (QuoteObservation(0, 26, dialogue=True),),
        ),
        (
            '"a""b"',
            (
                QuoteObservation(0, 3, dialogue=True),
                QuoteObservation(3, 6, dialogue=True),
            ),
        ),
        (
            "“a”“b”",
            (
                QuoteObservation(0, 3, dialogue=True),
                QuoteObservation(3, 6, dialogue=True),
            ),
        ),
    ],
    ids=("straight", "smart", "nested", "adjacent-straight", "adjacent-smart"),
)
def test_legacy_quote_oracle_freezes_exact_ranges(
    text: str, expected: tuple[QuoteObservation, ...]
) -> None:
    """Straight, smart, and nested quote ranges retain their exact flags."""
    assert legacy_quote_partition(CHAPTER_ID, text) == expected
    assert "".join(text[item.start : item.end] for item in expected) == text
    chapter = kk.ChapterInspection(CHAPTER_ID, 0, "Fixture", len(text), text)
    assert dialogue_ranges(build_grid(chapter)) == tuple(
        DialogueRange(CHAPTER_ID, item.start, item.end)
        for item in expected
        if item.dialogue
    )


def test_legacy_sentence_oracle_guards_titles_and_initials() -> None:
    """Titles and initials do not masquerade as sentence endings."""
    assert split_sentences("Dr. Ada met J. R. Smith. Then left.") == (
        "Dr. Ada met J. R. Smith. ",
        "Then left.",
    )


def test_legacy_structure_oracle_freezes_headings_blank_blocks_and_lines() -> None:
    """Separators stay exact while heading, paragraph, and line reasons survive."""
    text = "Chapter One\n\n\n\nLine one\nLine two\n\nTail."
    observed = legacy_structure_partition(
        text,
        frozenset({"Chapter One", "Tail."}),
        PauseFixture(
            heading_before_ms=400,
            heading_after_ms=500,
            paragraph_ms=300,
            line_ms=100,
        ),
    )
    assert observed == (
        StructureObservation("Chapter One\n\n\n\n", (HEADING_AFTER, PARAGRAPH)),
        StructureObservation("Line one\n", (LINE,)),
        StructureObservation("Line two\n\n", (HEADING_BEFORE, PARAGRAPH)),
        StructureObservation("Tail.", ()),
    )
    assert "".join(item.text for item in observed) == text


def test_grid_gap_durations_match_legacy_on_the_same_canonical_edges() -> None:
    """Pure grid reasons preserve every legacy effective internal silence."""
    text = "Chapter One\n\n\n\nLine one\nLine two\n\nTail."
    pauses = Pauses(
        heading_before_ms=400,
        heading_after_ms=500,
        paragraph_ms=300,
        line_ms=100,
    )
    chapter = kk.ChapterInspection(
        CHAPTER_ID,
        0,
        "Fixture",
        len(text),
        text,
        headings=("Chapter One", "Tail."),
    )
    units = build_grid(chapter)
    index = build_structure_index(units)
    actual = {
        unit.end: _gap_ms(reasons, pauses)
        for unit, reasons in zip(units, index.gaps, strict=True)
        if unit.end < len(text) and _gap_ms(reasons, pauses)
    }

    observed = legacy_structure_partition(text, frozenset(chapter.headings), pauses)
    position = 0
    expected: dict[int, int] = {}
    durations = {
        HEADING_BEFORE: pauses.heading_before_ms,
        HEADING_AFTER: pauses.heading_after_ms,
        PARAGRAPH: pauses.paragraph_ms,
        LINE: pauses.line_ms,
    }
    for piece in observed:
        position += len(piece.text)
        duration = max((durations[reason] for reason in piece.reasons), default=0)
        if duration:
            expected[position] = duration

    assert actual == expected == {15: 500, 24: 100, 34: 400}


def test_coincident_grid_reasons_translate_to_the_maximum_duration() -> None:
    """Adjacent headings and a paragraph edge form one gap, not stacked pauses."""
    text = "Heading A\n\nHeading B\n\nBody."
    chapter = kk.ChapterInspection(
        CHAPTER_ID,
        0,
        "Fixture",
        len(text),
        text,
        headings=("Heading A", "Heading B"),
    )
    units = build_grid(chapter)
    reasons = build_structure_index(units).gaps[0]
    pauses = Pauses(
        heading_before_ms=400,
        heading_after_ms=500,
        paragraph_ms=300,
    )

    assert GapReason.HEADING_BEFORE in reasons
    assert GapReason.HEADING_AFTER in reasons
    assert GapReason.PARAGRAPH in reasons
    assert _gap_ms(reasons, pauses) == pauses.heading_after_ms


def test_legacy_grid_oracle_freezes_emphasis_and_dialogue_edges() -> None:
    """Emphasis flags and quote edges remain addressable at exact offsets."""
    text = 'He thought "Go." Now.'
    chapter = kk.ChapterInspection(
        CHAPTER_ID,
        0,
        "Fixture",
        len(text),
        text,
        emphasis=((3, 10),),
    )
    units = build_grid(chapter)
    assert [
        (
            unit.paragraph,
            unit.line,
            unit.sentence,
            unit.phrase,
            unit.start,
            unit.end,
            unit.is_dialogue,
            unit.is_emphasised,
            unit_text(unit, text),
        )
        for unit in units
    ] == [
        (1, 1, 1, 1, 0, 11, False, True, "He thought "),
        (1, 1, 1, 2, 11, 16, True, False, '"Go."'),
        (1, 1, 1, 3, 16, 17, False, False, " "),
        (1, 1, 2, 1, 17, 21, False, False, "Now."),
    ]
    assert units[0].start == 0
    assert all(left.end == right.start for left, right in pairwise(units))
    assert units[-1].end == len(text)
    assert "".join(unit_text(unit, text) for unit in units) == text


def test_legacy_chunk_oracle_freezes_separator_free_prose_and_long_tokens() -> None:
    """The old whitespace fallback and indivisible-token cut stay observable."""
    prose = "and then " * 120
    prose_chunks = legacy_chunks(CHAPTER_ID, prose)
    assert [len(chunk) for chunk in prose_chunks] == [999, 81]
    assert prose_chunks[0].endswith(" ")
    assert "".join(prose_chunks) == prose

    token = "x" * (MAX_TTS_SEGMENT_CHARACTERS + 1)
    token_chunks = legacy_chunks(CHAPTER_ID, token)
    assert [len(chunk) for chunk in token_chunks] == [1000, 1]
    assert "".join(token_chunks) == token


def _voice(voice_id: str, fingerprint: str) -> kk.Voice:
    """Build a deterministic loaded voice for planning observations."""
    return kk.Voice(
        id=voice_id,
        name=voice_id.title(),
        enabled=True,
        provenance="Project-owned recording by Test Speaker",
        license_id="CC0-1.0",
        commercial_use_allowed=True,
        language="en-US",
        content_fingerprint=fingerprint,
        compatible_model_revisions=(MODEL_REVISION,),
        state="loaded",
    )


def _observe_current_plan(
    pipeline: kk.Pipeline,
    inspection: kk.BookInspection,
    narrator: kk.Voice,
    mira: kk.Voice,
    spans: tuple[SpeakerSpan, ...],
) -> PlanObservation:
    """Expose current planner semantics for comparison with frozen d72c1e8 data."""
    plan = planning.compile_execution_plan(
        pipeline,
        inspection,
        source_bytes_hash=SOURCE_HASH,
        resolved_voice=narrator,
        model_revision=MODEL_REVISION,
        cast_voices=(mira,),
        assignments={"mira": "mira-voice"},
        spans=spans,
    )
    spoken = planning.effective_spoken_form(pipeline.operations)
    pauses = planning._one_operation(pipeline.operations, Pauses) or Pauses()  # noqa: SLF001
    origins: list[planning._SegmentSource] = []
    replayed, replayed_silences = planning._compile_segments(  # noqa: SLF001
        inspection.chapters,
        spans,
        plan.cast,
        spoken,
        pauses,
        operations=pipeline.operations,
        origins=origins,
    )
    assert replayed == plan.segments
    assert replayed_silences == plan.trailing_silence_ms

    spoken_position = 0
    spoken_ranges: list[tuple[int, int]] = []
    for segment in plan.segments:
        spoken_ranges.append((spoken_position, spoken_position + len(segment.text)))
        spoken_position += len(segment.text)
    return PlanObservation(
        spoken_text="".join(segment.text for segment in plan.segments),
        segment_texts=tuple(segment.text for segment in plan.segments),
        speakers=tuple(segment.speaker_id for segment in plan.segments),
        voices=tuple(segment.voice_id for segment in plan.segments),
        canonical_ranges=tuple(
            (origin.canonical_start, origin.canonical_end) for origin in origins
        ),
        spoken_ranges=tuple(spoken_ranges),
        effective_silences_ms=plan.trailing_silence_ms,
    )


def test_legacy_planning_observation_freezes_semantic_sequence_and_ranges() -> None:
    """Spoken expansion preserves speakers, voices, source ranges, and gaps."""
    text = 'Chapter IV\n\nHe paid 21%.\n"Wait," Mira said.\n\nTail.'
    dialogue_start = text.index('"Wait,"')
    dialogue_end = dialogue_start + len('"Wait,"')
    spans = (
        SpeakerSpan(CHAPTER_ID, 0, dialogue_start, None),
        SpeakerSpan(CHAPTER_ID, dialogue_start, dialogue_end, "mira"),
        SpeakerSpan(CHAPTER_ID, dialogue_end, len(text), None),
    )
    chapter = kk.ChapterInspection(
        CHAPTER_ID,
        0,
        "Fixture",
        len(text),
        text,
        headings=("Chapter IV",),
    )
    narrator = _voice("narrator", "2" * 64)
    mira = _voice("mira-voice", "3" * 64)
    pipeline = (
        kk.epub("oracle.epub")
        .pronounce(numbers="standard", builtin=False)
        .pauses(heading_after_ms=500, paragraph_ms=300, line_ms=100)
        .assign_voice("narrator")
        .tts()
    )

    observed = _observe_current_plan(
        pipeline,
        kk.BookInspection(kk.BookMetadata("Oracle", "Fixture"), (chapter,)),
        narrator,
        mira,
        spans,
    )

    assert observed == LEGACY_PLAN_OBSERVATION


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        (
            "Sentence boundary. " * 90,
            BreakQualityObservation(
                characters=1710,
                chunks=2,
                internal_boundaries=(988,),
                structural_or_clause_grid_edges=(("sentence", 1),),
                emergency_within_leaf=(),
            ),
        ),
        (
            "and then " * 120,
            BreakQualityObservation(
                characters=1080,
                chunks=2,
                internal_boundaries=(999,),
                structural_or_clause_grid_edges=(),
                emergency_within_leaf=(("whitespace", 1),),
            ),
        ),
        (
            "x" * 1001,
            BreakQualityObservation(
                characters=1001,
                chunks=2,
                internal_boundaries=(1000,),
                structural_or_clause_grid_edges=(),
                emergency_within_leaf=(("hard_token", 1),),
            ),
        ),
    ],
    ids=("normal-grid-edge", "emergency-whitespace", "emergency-hard-token"),
)
def test_break_quality_report_classifies_boundaries_without_a_percentage_target(
    text: str, expected: BreakQualityObservation
) -> None:
    """The metric preserves counts and characterizations, not an old ratio target."""
    assert legacy_break_quality(CHAPTER_ID, text) == expected
