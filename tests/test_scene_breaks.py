"""Mid-chapter scene breaks: detection, placement, and pause policy.

A scene break is the ``<hr/>`` or labelled ornament between two scenes of one
chapter. It is recorded at parse time because normalization erases it: an
``<hr/>``, an empty paragraph, and an ordinary paragraph boundary all reduce to
the same two newlines, so nothing downstream could recover it.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import kenkui as kk
from helpers import make_epub, xhtml
from kenkui._domain.grid import GapReason, build_grid, build_structure_index
from kenkui._domain.planning import ExecutionPlan, compile_execution_plan

if TYPE_CHECKING:
    from pathlib import Path

MODEL_REVISION = "pocket-tts/model@0123456789abcdef"
SOURCE_HASH = "1" * 64
SCENE_MS = 900
PARAGRAPH_MS = 250

_FIRST = "<p>She shut the door behind her and did not look back.</p>"
_SECOND = "<p>Morning came slowly over the rooftops.</p>"
_THIRD = "<p>He had not slept at all.</p>"


def _book(tmp_path: Path, body: str, name: str = "book.epub") -> kk.Pipeline:
    return kk.epub(
        make_epub(tmp_path / name, chapters={"one": xhtml(body)}, spine=["one"])
    )


def _chapter(
    tmp_path: Path, body: str, name: str = "book.epub"
) -> kk.ChapterInspection:
    return _book(tmp_path, body, name).inspect().chapters[0]


def test_horizontal_rule_records_the_block_it_introduces(tmp_path: Path) -> None:
    """An <hr/> is zero-width, so the scene-opening paragraph carries it."""
    chapter = _chapter(tmp_path, f"{_FIRST}<hr/>{_SECOND}")

    opener = "Morning came slowly over the rooftops."
    start = chapter.text.index(opener)
    assert chapter.scene_ranges == ((start, start + len(opener)),)


def test_labelled_block_is_a_marker_even_when_hidden(tmp_path: Path) -> None:
    """Publisher intent survives aria-hidden, which strips the ornament itself."""
    chapter = _chapter(
        tmp_path,
        f'{_FIRST}<div aria-hidden="true" class="transition">—</div>{_SECOND}',
    )

    # The ornament is hidden, so it contributes no canonical text at all.
    assert "—" not in chapter.text
    assert len(chapter.scene_ranges) == 1


def test_epub_type_attribute_is_honoured(tmp_path: Path) -> None:
    """A namespaced epub:type token counts the same as a class token."""
    chapter = _chapter(
        tmp_path,
        f"{_FIRST}"
        '<div xmlns:epub="http://www.idpf.org/2007/ops"'
        ' epub:type="se:scene-break"></div>'
        f"{_SECOND}",
    )

    assert len(chapter.scene_ranges) == 1


def test_canonical_text_is_unchanged_by_detection(tmp_path: Path) -> None:
    """Markers must not move canonical text by a byte: billing depends on it."""
    with_marker = _chapter(tmp_path, f"{_FIRST}<hr/>{_SECOND}", "a.epub")
    without = _chapter(tmp_path, f"{_FIRST}{_SECOND}", "b.epub")

    assert with_marker.text == without.text
    assert with_marker.speech_characters == without.speech_characters
    assert without.scene_ranges == ()


def test_consecutive_markers_collapse_to_one_break(tmp_path: Path) -> None:
    """A rule plus a labelled ornament is one break, not two."""
    chapter = _chapter(
        tmp_path,
        f'{_FIRST}<hr/><p class="ornament">* * *</p>{_SECOND}',
    )

    assert len(chapter.scene_ranges) == 1


def test_trailing_marker_is_discarded(tmp_path: Path) -> None:
    """Nothing follows it, so a chapter cannot end on a scene pause."""
    chapter = _chapter(tmp_path, f"{_FIRST}{_SECOND}<hr/>")

    assert chapter.scene_ranges == ()


def test_labelled_prose_paragraph_is_not_a_marker(tmp_path: Path) -> None:
    """A scene class on the opening paragraph must not shift the break."""
    chapter = _chapter(
        tmp_path,
        f'{_FIRST}<p class="transition">Morning came slowly.</p>{_THIRD}',
    )

    assert chapter.scene_ranges == ()


def test_unlabelled_ornament_is_not_yet_a_marker(tmp_path: Path) -> None:
    """Bare ornament detection ships with suppression, so its pause lands right."""
    chapter = _chapter(tmp_path, f"{_FIRST}<p>* * *</p>{_SECOND}")

    assert chapter.scene_ranges == ()


def test_scene_reason_closes_the_leaf_before_the_opener(tmp_path: Path) -> None:
    """The flag marks the opening leaf; the gap it implies precedes that leaf."""
    chapter = _chapter(tmp_path, f"{_FIRST}<hr/>{_SECOND}{_THIRD}")
    units = build_grid(chapter)
    gaps = build_structure_index(units).gaps

    opener = next(index for index, unit in enumerate(units) if unit.is_scene_start)
    assert opener > 0
    assert GapReason.SCENE in gaps[opener - 1]
    assert GapReason.SCENE not in gaps[opener]


def test_scene_pause_is_the_maximum_not_the_sum(tmp_path: Path) -> None:
    """A scene break is also a paragraph boundary; one gap, never two added."""
    book = _book(tmp_path, f"{_FIRST}<hr/>{_SECOND}{_THIRD}")
    rows = list(
        book.pauses(paragraph_ms=PARAGRAPH_MS, scene_ms=SCENE_MS)
        .assign_voice("eponine")
        .script()
    )

    assert [row.silence_after_ms for row in rows] == [SCENE_MS, PARAGRAPH_MS, 0]


def test_scene_tier_is_off_unless_called(tmp_path: Path) -> None:
    """No tier derives a default from another: unset means the paragraph value."""
    book = _book(tmp_path, f"{_FIRST}<hr/>{_SECOND}")
    rows = list(
        book.pauses(chapter_ms=1500, paragraph_ms=PARAGRAPH_MS)
        .assign_voice("eponine")
        .script()
    )

    assert rows[0].silence_after_ms == PARAGRAPH_MS


def test_manual_silence_overrides_a_scene_pause(tmp_path: Path) -> None:
    """Zero stays an instruction: a declared silence replaces the derived one."""
    book = _book(tmp_path, f"{_FIRST}<hr/>{_SECOND}")
    rows = list(
        book.pauses(paragraph_ms=PARAGRAPH_MS, scene_ms=SCENE_MS)
        .silence(0, where={"chapter": "*", "paragraph": 1})
        .assign_voice("eponine")
        .script()
    )

    assert rows[0].silence_after_ms == 0


def test_script_row_marks_the_opening_row(tmp_path: Path) -> None:
    """Detection has to be auditable: silence_after_ms conflates every reason."""
    book = _book(tmp_path, f"{_FIRST}<hr/>{_SECOND}")
    rows = list(book.pauses(scene_ms=SCENE_MS).assign_voice("eponine").script())

    assert [row.is_scene_start for row in rows] == [False, True]
    # The flag and the silence it causes deliberately sit on different rows.
    assert rows[0].silence_after_ms == SCENE_MS
    assert rows[1].silence_after_ms == 0


def _voice() -> kk.Voice:
    return kk.Voice(
        id="fixture",
        name="Fixture Voice",
        enabled=True,
        provenance="Project-owned recording by Test Speaker",
        license_id="CC0-1.0",
        commercial_use_allowed=True,
        language="en-US",
        content_fingerprint="2" * 64,
        compatible_model_revisions=(MODEL_REVISION,),
    )


def _plan(book: kk.Pipeline) -> ExecutionPlan:
    rendered = book.assign_voice("fixture").tts()
    return compile_execution_plan(
        rendered,
        rendered.inspect(),
        source_bytes_hash=SOURCE_HASH,
        resolved_voice=_voice(),
        model_revision=MODEL_REVISION,
    )


def test_script_and_planning_agree_on_scene_silence(tmp_path: Path) -> None:
    """Script and the renderer must not drift; they share the planner's helpers."""
    book = _book(tmp_path, f"{_FIRST}<hr/>{_SECOND}{_THIRD}").pauses(
        paragraph_ms=PARAGRAPH_MS, scene_ms=SCENE_MS
    )
    rows = [row.silence_after_ms for row in book.assign_voice("fixture").script()]

    assert _plan(book).trailing_silence_ms == tuple(rows)


def test_retuning_scene_ms_preserves_segment_identity(tmp_path: Path) -> None:
    """A scene break is already a paragraph cut, so durations replan only.

    This is what lets an existing render be retuned: the cut is already there,
    so only the gap duration moves and every cached segment stays valid.
    """
    body = f"{_FIRST}<hr/>{_SECOND}{_THIRD}"

    def plan(scene_ms: int) -> ExecutionPlan:
        return _plan(
            _book(tmp_path, body).pauses(paragraph_ms=PARAGRAPH_MS, scene_ms=scene_ms)
        )

    off, on = plan(0), plan(SCENE_MS)

    assert [s.id for s in off.segments] == [s.id for s in on.segments]
    # Not vacuous: the silence really did change, the identities did not.
    assert off.trailing_silence_ms != on.trailing_silence_ms
    assert on.trailing_silence_ms[0] == SCENE_MS


def test_scene_tier_moves_a_cut_when_paragraphs_are_silent(tmp_path: Path) -> None:
    """Without a paragraph pause there is no cut yet, so enabling one adds it."""
    body = f"{_FIRST}<hr/>{_SECOND}"

    def plan(scene_ms: int) -> ExecutionPlan:
        return _plan(_book(tmp_path, body).pauses(scene_ms=scene_ms))

    off, on = plan(0), plan(SCENE_MS)

    assert len(off.segments) == 1
    assert len(on.segments) == 2  # noqa: PLR2004 - the break became a boundary
    assert [s.id for s in off.segments] != [s.id for s in on.segments]
