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
    SpeakerSpan,
    _gap_ms,
)
from kenkui._domain.quotes import extract_spans

CHAPTER_ID = "ch-v1-oracle"
SOURCE_HASH = "1" * 64
MODEL_REVISION = "pocket-tts/model@oracle"


@dataclass(frozen=True, slots=True)
class QuoteObservation:
    """Frozen quote range retained as compact migration evidence."""

    start: int
    end: int
    dialogue: bool


@dataclass(frozen=True, slots=True)
class PlanObservation:
    """Semantic planning fields retained from the d72c1e8 oracle run."""

    spoken_text: str
    segment_texts: tuple[str, ...]
    speakers: tuple[str | None, ...]
    voices: tuple[str, ...]
    canonical_ranges: tuple[tuple[int, int], ...]
    spoken_ranges: tuple[tuple[int, int], ...]
    effective_silences_ms: tuple[int, ...]


LEGACY_PLAN_OBSERVATION = PlanObservation(
    spoken_text=(
        'Chapter Four\n\nHe paid twenty-one percent.\n"Wait," Mira said.\n\nTail.'
    ),
    segment_texts=(
        "Chapter Four\n\n",
        "He paid twenty-one percent.\n",
        '"Wait,"',
        " Mira said.\n\n",
        "Tail.",
    ),
    speakers=(None, None, "mira", None, None),
    voices=("narrator", "narrator", "mira-voice", "narrator", "narrator"),
    canonical_ranges=((0, 12), (12, 25), (25, 32), (32, 45), (45, 50)),
    spoken_ranges=((0, 14), (14, 42), (42, 49), (49, 62), (62, 67)),
    effective_silences_ms=(500, 100, 0, 300, 0),
)


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
def test_frozen_quote_ranges_match_domain_scanner(
    text: str, expected: tuple[QuoteObservation, ...]
) -> None:
    """Straight, smart, and nested quote ranges retain their exact flags."""
    assert (
        tuple(
            QuoteObservation(span.start, span.end, span.is_dialogue)
            for span in extract_spans(CHAPTER_ID, text)
        )
        == expected
    )
    assert "".join(text[item.start : item.end] for item in expected) == text
    chapter = kk.ChapterInspection(CHAPTER_ID, 0, "Fixture", len(text), text)
    assert dialogue_ranges(build_grid(chapter)) == tuple(
        DialogueRange(CHAPTER_ID, item.start, item.end)
        for item in expected
        if item.dialogue
    )


def test_frozen_sentence_ranges_guard_titles_and_initials() -> None:
    """Titles and initials do not masquerade as sentence endings."""
    assert split_sentences("Dr. Ada met J. R. Smith. Then left.") == (
        "Dr. Ada met J. R. Smith. ",
        "Then left.",
    )


def test_frozen_gap_durations_remain_on_the_same_canonical_edges() -> None:
    """Pure grid reasons preserve the pre-migration effective silences."""
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

    assert actual == {15: 500, 24: 100, 34: 400}


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


def test_frozen_grid_fixture_keeps_emphasis_and_dialogue_edges() -> None:
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


def test_frozen_plan_observation_matches_grid_planning_semantics() -> None:
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
