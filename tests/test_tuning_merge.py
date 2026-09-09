"""Tuning merges over machine attribution at plan time, never in resolution."""

from __future__ import annotations

from typing import TYPE_CHECKING

import kenkui as kk
from kenkui._domain.operations import Attributions, Silences
from kenkui._domain.paths import parse_pattern
from kenkui._domain.planning import (
    ExecutionPlan,
    compile_execution_plan,
    effective_spans,
    manual_gaps,
)
from kenkui._domain.tuning import Rule

if TYPE_CHECKING:
    from kenkui._domain.planning import SpeakerSpan
    from kenkui.inspection import ChapterInspection

MODEL_REVISION = "pocket-tts/model@0123456789abcdef"
PLAN_CHAPTER_ID = "ch-v1-one"
PLAN_TEXT = "He woke early.\n\nShe slept on."
# Offset of the paragraph break: the manual gap declared on paragraph one
# anchors to the end of its last grid leaf, which is the blank line itself.
PARAGRAPH_ONE_END = 16


def attributions(*pairs: tuple[str, dict[str, object]]) -> tuple[Attributions]:
    """Build the operation tuple directly; no test hook on the public API."""
    rules = tuple(
        Rule(where=parse_pattern(where), value=value, index=index)
        for index, (value, where) in enumerate(pairs)
    )
    return (Attributions(rules=rules),)


def silences(*pairs: tuple[int, dict[str, object]]) -> tuple[Silences]:
    """Build a silence operation tuple directly, in declaration order."""
    rules = tuple(
        Rule(where=parse_pattern(where), value=value, index=index)
        for index, (value, where) in enumerate(pairs)
    )
    return (Silences(rules=rules),)


def test_override_replaces_machine_attribution(
    chapter_ch08: ChapterInspection, machine_spans: tuple[SpeakerSpan, ...]
) -> None:
    """One narrower rule claims exactly the subtree it addresses."""
    book = attributions(
        ("jessica", {"chapter": chapter_ch08.id, "paragraph": 2, "sentence": 2})
    )
    spans = effective_spans(chapter_ch08, machine_spans, book)
    covering = [s for s in spans if s.character_id == "jessica"]
    assert len(covering) == 1


def test_spans_still_tile_the_chapter(
    chapter_ch08: ChapterInspection, machine_spans: tuple[SpeakerSpan, ...]
) -> None:
    """Merged spans must reproduce the chapter exactly, as machine spans do."""
    book = attributions(("paul", {"chapter": chapter_ch08.id}))
    spans = effective_spans(chapter_ch08, machine_spans, book)
    rebuilt = "".join(chapter_ch08.text[s.start : s.end] for s in spans)
    assert rebuilt == chapter_ch08.text


def test_adjacent_units_with_one_speaker_merge_into_one_span(
    chapter_ch08: ChapterInspection, machine_spans: tuple[SpeakerSpan, ...]
) -> None:
    """Coalescing keeps the segment count near today's, not one span per unit."""
    book = attributions(("paul", {"chapter": chapter_ch08.id}))
    spans = effective_spans(chapter_ch08, machine_spans, book)
    assert len(spans) == 1


def test_no_tuning_returns_machine_spans_unchanged(
    chapter_ch08: ChapterInspection, machine_spans: tuple[SpeakerSpan, ...]
) -> None:
    """Every already-rendered book keeps byte-identical spans."""
    assert effective_spans(chapter_ch08, machine_spans, ()) == machine_spans


def test_a_rule_matching_nothing_leaves_the_machine_layer_alone(
    chapter_ch08: ChapterInspection, machine_spans: tuple[SpeakerSpan, ...]
) -> None:
    """A rule scoped to another chapter cannot repaint this one."""
    book = attributions(("jessica", {"chapter": "ch-v1-elsewhere"}))
    spans = effective_spans(chapter_ch08, machine_spans, book)
    assert [span.character_id for span in spans] == [None]


def test_a_later_narrower_rule_wins_inside_a_wider_one(
    chapter_ch08: ChapterInspection, machine_spans: tuple[SpeakerSpan, ...]
) -> None:
    """Precedence runs through resolve_rules, so specificity beats breadth."""
    book = attributions(
        ("paul", {"chapter": chapter_ch08.id}),
        ("jessica", {"chapter": chapter_ch08.id, "paragraph": 1}),
    )
    spans = effective_spans(chapter_ch08, machine_spans, book)
    assert [span.character_id for span in spans] == ["jessica", "paul"]


def test_silence_normalizes_to_the_last_leaf(chapter_ch08: ChapterInspection) -> None:
    """A subtree and its final child denote the same physical gap."""
    paragraph = silences((900, {"chapter": chapter_ch08.id, "paragraph": 2}))
    sentence = silences(
        (900, {"chapter": chapter_ch08.id, "paragraph": 2, "sentence": -1})
    )
    assert manual_gaps(chapter_ch08, paragraph) == manual_gaps(chapter_ch08, sentence)
    assert list(manual_gaps(chapter_ch08, paragraph).values()) == [900]


def test_zero_removes_a_pause(chapter_ch08: ChapterInspection) -> None:
    """Silence replaces the derived duration, so zero is a real instruction."""
    ops = silences((0, {"chapter": chapter_ch08.id, "paragraph": 2}))
    assert 0 in manual_gaps(chapter_ch08, ops).values()


def test_no_silence_rules_declare_no_gaps(chapter_ch08: ChapterInspection) -> None:
    """The fast path costs nothing and builds no grid."""
    assert manual_gaps(chapter_ch08, ()) == {}


def test_a_later_silence_wins_on_the_same_anchor(
    chapter_ch08: ChapterInspection,
) -> None:
    """Two rules landing on one leaf resolve through the same precedence path."""
    ops = silences(
        (900, {"chapter": chapter_ch08.id, "paragraph": 2}),
        (300, {"chapter": chapter_ch08.id, "paragraph": 2}),
    )
    assert list(manual_gaps(chapter_ch08, ops).values()) == [300]


def plan(pipeline: kk.Pipeline) -> ExecutionPlan:
    """Compile a real one-chapter plan so the wiring is not stubbed."""
    voice = kk.Voice(
        id="eponine",
        name="Eponine",
        enabled=True,
        provenance="Project-owned recording by Test Speaker",
        license_id="CC0-1.0",
        commercial_use_allowed=True,
        language="en-US",
        content_fingerprint="2" * 64,
        compatible_model_revisions=(MODEL_REVISION,),
        state="loaded",
    )
    chapter = kk.ChapterInspection(
        PLAN_CHAPTER_ID, 0, "Chapter One", len(PLAN_TEXT), PLAN_TEXT
    )
    return compile_execution_plan(
        pipeline.assign_voice("eponine").tts(),
        kk.BookInspection(kk.BookMetadata("T", "A", cover_available=True), (chapter,)),
        source_bytes_hash="1" * 64,
        resolved_voice=voice,
        model_revision=MODEL_REVISION,
    )


def test_a_manual_silence_forces_a_chunk_boundary() -> None:
    """The gap has to land between two segments, so planning must cut there."""
    plain = plan(kk.epub("book.epub"))
    tuned = plan(
        kk.epub("book.epub").silence(
            900, where={"chapter": PLAN_CHAPTER_ID, "paragraph": 1}
        )
    )
    assert len(plain.segments) == 1
    assert [segment.text for segment in tuned.segments] == [
        PLAN_TEXT[:PARAGRAPH_ONE_END],
        PLAN_TEXT[PARAGRAPH_ONE_END:],
    ]
    assert tuned.trailing_silence_ms == (900, 0)


def test_a_manual_silence_replaces_the_derived_gap() -> None:
    """Zero removes a paragraph pause that the tier model would otherwise set."""
    derived = plan(kk.epub("book.epub").pauses(paragraph_ms=500))
    removed = plan(
        kk.epub("book.epub")
        .pauses(paragraph_ms=500)
        .silence(0, where={"chapter": PLAN_CHAPTER_ID, "paragraph": 1})
    )
    assert derived.trailing_silence_ms == (500, 0)
    assert removed.trailing_silence_ms == (0, 0)


def test_tuning_that_matches_nothing_leaves_the_plan_identical() -> None:
    """A rule scoped elsewhere must not move a single segment identity."""
    plain = plan(kk.epub("book.epub").pauses(paragraph_ms=500))
    scoped = plan(
        kk.epub("book.epub")
        .pauses(paragraph_ms=500)
        .attribute("paul", where={"chapter": "ch-v1-elsewhere"})
        .silence(900, where={"chapter": "ch-v1-elsewhere"})
    )
    assert plain.semantic_fingerprint == scoped.semantic_fingerprint
    assert [segment.id for segment in plain.segments] == [
        segment.id for segment in scoped.segments
    ]
