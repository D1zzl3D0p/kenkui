"""Tuning merges over machine attribution at plan time, never in resolution."""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING

import kenkui as kk
from kenkui._domain.operations import Attributions, Silences, SpokenForm
from kenkui._domain.paths import parse_pattern
from kenkui._domain.planning import (
    ExecutionPlan,
    SpeakerSpan,
    compile_execution_plan,
    effective_spans,
    manual_gaps,
)
from kenkui._domain.quotes import extract_spans
from kenkui._domain.tuning import Rule

if TYPE_CHECKING:
    from collections.abc import Mapping
    from pathlib import Path

    from kenkui.inspection import ChapterInspection

MODEL_REVISION = "pocket-tts/model@0123456789abcdef"
PLAN_CHAPTER_ID = "ch-v1-one"
PLAN_TEXT = "He woke early.\n\nShe slept on."
# Offset of the paragraph break: the manual gap declared on paragraph one
# anchors to the end of its last grid leaf, which is the blank line itself.
PARAGRAPH_ONE_END = 16

GAP_CHAPTER_ID = "ch-v1-gap"
GAP_TEXT = "He woke. He rose. He dressed.\n\nShe slept on."
# Offsets inside the three-sentence first paragraph: sentence one's last
# grid leaf ends mid-paragraph (its trailing space, before "He rose"), and
# the paragraph itself ends at the blank line, once a paragraph pause tier
# makes the paragraph break a real structural piece boundary.
SENTENCE_ONE_END = 9
GAP_PARAGRAPH_END = 31

LEAD_CHAPTER_ID = "ch-v1-lead"
# One homograph in two paragraphs: the only shape that can tell a scoped
# lexicon apart from a whole-book one.
LEAD_TEXT = "He took the lead.\n\nThe lead pipe burst."


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


def attributed_spans(chapter: ChapterInspection) -> tuple[SpeakerSpan, ...]:
    """Build machine spans the shape real attribution emits: one per quote run.

    ``extract_spans`` always places a narration run between two quoted ones,
    so no two spans built this way share a speaker across a boundary. That
    unstated property is what ``effective_spans`` leans on once any rule
    exists and it re-tiles the chapter from grid units: without it,
    ``_coalesce`` could join two spans attribution deliberately kept apart,
    and a rule matching nothing would still move the plan.
    """
    speakers = ("jessica", "paul")
    quoted = 0
    spans: list[SpeakerSpan] = []
    for span in extract_spans(chapter.id, chapter.text):
        character = None
        if span.is_dialogue:
            character = speakers[quoted % len(speakers)]
            quoted += 1
        spans.append(SpeakerSpan(chapter.id, span.start, span.end, character))
    return tuple(spans)


def test_a_rule_matching_nothing_preserves_multi_span_attribution(
    chapter_ch08: ChapterInspection,
) -> None:
    """The cache-invalidation guarantee, against realistic machine spans.

    Every attribution rule makes ``effective_spans`` rebuild the chapter from
    grid units. A rule scoped elsewhere must leave that rebuild byte-identical
    to what attribution produced, or every book in the library re-renders the
    first time anyone corrects one line of a different chapter.
    """
    machine = attributed_spans(chapter_ch08)
    assert len(machine) > 1
    assert {span.character_id for span in machine} == {None, "jessica", "paul"}
    book = attributions(("irulan", {"chapter": "ch-v1-elsewhere"}))
    assert effective_spans(chapter_ch08, machine, book) == machine


def test_a_rule_replaces_one_genuinely_attributed_character(
    chapter_ch08: ChapterInspection,
) -> None:
    """Correcting a machine-attributed quote moves that span and nothing else."""
    machine = attributed_spans(chapter_ch08)
    corrected = next(span for span in machine if span.character_id == "jessica")
    book = attributions(
        (
            "irulan",
            {"chapter": chapter_ch08.id, "paragraph": 2, "sentence": 1, "phrase": 1},
        )
    )
    spans = effective_spans(chapter_ch08, machine, book)
    assert spans == tuple(
        SpeakerSpan(
            span.chapter_id,
            span.start,
            span.end,
            "irulan" if span is corrected else span.character_id,
        )
        for span in machine
    )
    assert "".join(chapter_ch08.text[s.start : s.end] for s in spans) == (
        chapter_ch08.text
    )


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


def plan(
    pipeline: kk.Pipeline,
    *,
    chapter_id: str = PLAN_CHAPTER_ID,
    text: str = PLAN_TEXT,
    assignments: Mapping[str, str] | None = None,
    cast_voices: tuple[kk.Voice, ...] = (),
) -> ExecutionPlan:
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
    chapter = kk.ChapterInspection(chapter_id, 0, "Chapter One", len(text), text)
    return compile_execution_plan(
        pipeline.assign_voice("eponine").tts(),
        kk.BookInspection(kk.BookMetadata("T", "A", cover_available=True), (chapter,)),
        source_bytes_hash="1" * 64,
        resolved_voice=voice,
        model_revision=MODEL_REVISION,
        cast_voices=cast_voices,
        assignments=assignments,
    )


JESSICA_VOICE = kk.Voice(
    id="jessica",
    name="Jessica",
    enabled=True,
    provenance="Project-owned recording by Test Speaker",
    license_id="CC0-1.0",
    commercial_use_allowed=True,
    language="en-US",
    content_fingerprint="3" * 64,
    compatible_model_revisions=(MODEL_REVISION,),
    state="loaded",
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


def test_an_attribution_correction_changes_the_rendered_speaker_and_voice() -> None:
    """The merge must reach compiled segments, not just effective_spans alone.

    A correction is only worth anything if it actually changes what gets
    synthesized. This exercises the whole path -- .attribute() through
    compile_execution_plan -- and checks both the speaker identity on the
    segment and the voice casting resolves it to, while the untouched half
    of the chapter keeps narrating exactly as before.
    """
    plain = plan(kk.epub("book.epub"))
    corrected = plan(
        kk.epub("book.epub").attribute(
            "jessica", where={"chapter": PLAN_CHAPTER_ID, "paragraph": 2}
        ),
        assignments={"jessica": "jessica"},
        cast_voices=(JESSICA_VOICE,),
    )
    assert [(segment.speaker_id, segment.text) for segment in plain.segments] == [
        (None, PLAN_TEXT)
    ]
    assert [(segment.speaker_id, segment.text) for segment in corrected.segments] == [
        (None, PLAN_TEXT[:PARAGRAPH_ONE_END]),
        ("jessica", PLAN_TEXT[PARAGRAPH_ONE_END:]),
    ]
    assert corrected.segments[0].voice_id == plain.segments[0].voice_id
    assert corrected.segments[1].voice_id == "jessica"
    assert corrected.segments[1].voice_id != corrected.segments[0].voice_id


def test_a_manual_silence_cuts_strictly_inside_a_structural_piece() -> None:
    """A gap declared mid-paragraph must split there, not just at a piece edge.

    With a paragraph pause tier active, the whole first paragraph is one
    structural piece running from 0 to GAP_PARAGRAPH_END. The declared
    silence anchors inside that piece, at SENTENCE_ONE_END, so the only way
    the plan can show three segments split exactly there is if planning's
    mid-piece cut loop actually ran -- this is not a piece boundary that
    would already exist without it.
    """
    tuned = plan(
        kk.epub("book.epub")
        .silence(250, where={"chapter": GAP_CHAPTER_ID, "paragraph": 1, "sentence": 1})
        .pauses(paragraph_ms=500),
        chapter_id=GAP_CHAPTER_ID,
        text=GAP_TEXT,
    )
    assert [segment.text for segment in tuned.segments] == [
        GAP_TEXT[:SENTENCE_ONE_END],
        GAP_TEXT[SENTENCE_ONE_END:GAP_PARAGRAPH_END],
        GAP_TEXT[GAP_PARAGRAPH_END:],
    ]
    assert "".join(segment.text for segment in tuned.segments) == GAP_TEXT
    assert tuned.trailing_silence_ms == (250, 500, 0)


def spoken(pipeline: kk.Pipeline, text: str = LEAD_TEXT) -> str:
    """Compile a one-chapter plan and return exactly what the engine hears."""
    return "".join(
        segment.text
        for segment in plan(pipeline, chapter_id=LEAD_CHAPTER_ID, text=text).segments
    )


def test_a_whole_book_lexicon_reaches_the_segment_text() -> None:
    """The point of pronounce(): entries must change what is synthesized.

    Recording the rule is not the feature. Nothing else in the suite follows a
    lexicon from the public method all the way to a compiled segment, which is
    how it once stopped arriving without a single test failing.
    """
    plain = spoken(kk.epub("book.epub").pronounce(numbers="off", builtin=False))
    tuned = spoken(
        kk.epub("book.epub").pronounce({"lead": "leed"}, numbers="off", builtin=False)
    )
    assert plain == LEAD_TEXT
    assert tuned == "He took the leed.\n\nThe leed pipe burst."


def test_a_scoped_lexicon_speaks_only_inside_its_scope() -> None:
    """Anchoring to a region is what makes a homograph expressible at all."""
    tuned = spoken(
        kk.epub("book.epub").pronounce(
            {"lead": "led"},
            where={"chapter": LEAD_CHAPTER_ID, "paragraph": 2},
            numbers="off",
            builtin=False,
        )
    )
    assert tuned == "He took the lead.\n\nThe led pipe burst."


def test_a_scoped_lexicon_layers_over_the_whole_book_one() -> None:
    """Both tables speak in the scope; only a shared key goes to the narrower."""
    tuned = spoken(
        kk.epub("book.epub")
        .pronounce({"lead": "leed", "pipe": "pyp"}, numbers="off", builtin=False)
        .pronounce({"lead": "led"}, where={"chapter": LEAD_CHAPTER_ID, "paragraph": 2})
    )
    assert tuned == "He took the leed.\n\nThe led pyp burst."


def test_a_scoped_lexicon_is_clipped_to_each_speaker_span() -> None:
    """An attribution rule cuts the chapter into fragments the regions outlive."""
    tuned = spoken(
        kk.epub("book.epub")
        .pronounce(
            {"lead": "led"},
            where={"chapter": LEAD_CHAPTER_ID, "paragraph": 2},
            numbers="off",
            builtin=False,
        )
        .attribute("jessica", where={"chapter": LEAD_CHAPTER_ID, "paragraph": 1})
    )
    assert tuned == "He took the lead.\n\nThe led pipe burst."


def test_a_scoped_lexicon_survives_selection_clipping() -> None:
    """Previewing one paragraph must speak it the way a full render would."""
    tuned = plan(
        kk.epub("book.epub")
        .pronounce(
            {"lead": "led", "burst": "birst"},
            where={"chapter": LEAD_CHAPTER_ID, "paragraph": 2},
            numbers="off",
            builtin=False,
        )
        .select({"chapter": LEAD_CHAPTER_ID, "paragraph": 2}),
        chapter_id=LEAD_CHAPTER_ID,
        text=LEAD_TEXT,
    )
    assert "".join(segment.text for segment in tuned.segments) == "The led pipe birst."


def test_a_whole_book_lexicon_keeps_its_pre_tuning_segment_identity() -> None:
    """Routing entries through a rule must not re-render an existing library."""
    entries = (("lead", "leed"),)
    before = plan(
        replace(
            kk.epub("book.epub"),
            operations=(
                SpokenForm(numbers="off", builtin_lexicon=False, lexicon=entries),
            ),
        ),
        chapter_id=LEAD_CHAPTER_ID,
        text=LEAD_TEXT,
    )
    after = plan(
        kk.epub("book.epub").pronounce({"lead": "leed"}, numbers="off", builtin=False),
        chapter_id=LEAD_CHAPTER_ID,
        text=LEAD_TEXT,
    )
    assert before.semantic_fingerprint == after.semantic_fingerprint
    assert [segment.id for segment in before.segments] == [
        segment.id for segment in after.segments
    ]


def test_a_lexicon_scoped_elsewhere_leaves_the_plan_identical() -> None:
    """A rule matching nothing must not move a segment identity here either."""
    plain = plan(
        kk.epub("book.epub").pronounce(numbers="off", builtin=False),
        chapter_id=LEAD_CHAPTER_ID,
        text=LEAD_TEXT,
    )
    scoped = plan(
        kk.epub("book.epub")
        .pronounce(numbers="off", builtin=False)
        .pronounce({"lead": "led"}, where={"chapter": "ch-v1-elsewhere"}),
        chapter_id=LEAD_CHAPTER_ID,
        text=LEAD_TEXT,
    )
    assert plain.semantic_fingerprint == scoped.semantic_fingerprint


def test_a_later_lexicon_keeps_the_spoken_style_already_chosen() -> None:
    """The dial-in loop adds pronunciations under a house style, not over it."""
    styled = kk.epub("book.epub").pronounce(numbers="off", builtin=False, roman=False)
    tuned = styled.pronounce({"lead": "leed"})
    settings = next(op for op in tuned.operations if isinstance(op, SpokenForm))
    assert settings.numbers == "off"
    assert settings.builtin_lexicon is False
    assert settings.features == (("roman", False),)
    assert spoken(tuned) == "He took the leed.\n\nThe leed pipe burst."


def test_reloaded_pronunciations_speak_without_a_pronounce_call(
    epub_path: Path, tmp_path: Path
) -> None:
    """A sidecar's lexicon must reach the engine, not wait for a style call."""
    saved = (
        kk.book(epub_path)
        .pronounce({"lead": "leed"})
        .write_annotations(tmp_path / "book.kenkui.json")
    )
    reloaded = kk.book(epub_path).annotations(saved)
    assert not any(isinstance(op, SpokenForm) for op in reloaded.operations)
    # Numbers stay off and the built-in table stays out: only what was asked
    # for is honoured, so "1837" is still read as digits.
    assert spoken(reloaded, "The lead in 1837.") == "The leed in 1837."
