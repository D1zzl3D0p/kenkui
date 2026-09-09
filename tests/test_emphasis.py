"""Emphasis offsets recorded by the EPUB parser without touching canonical text."""

from __future__ import annotations

from typing import TYPE_CHECKING

from helpers import make_epub, xhtml
from kenkui._epub.parser import inspect_epub

if TYPE_CHECKING:
    from pathlib import Path

    from kenkui.inspection import ChapterInspection

EXPECTED_REPEATED_PHRASE_RANGES = 2


def only_chapter(tmp_path: Path, body: str) -> ChapterInspection:
    """Parse a single-chapter EPUB built from ``body`` and return its chapter."""
    source = make_epub(tmp_path / "e.epub", chapters={"c1": xhtml(body)}, spine=["c1"])
    return inspect_epub(source).chapters[0]


def test_emphasis_offsets_recorded_without_changing_text(tmp_path: Path) -> None:
    """Emphasis offsets address the exact emphasised slice of canonical text."""
    chapter = only_chapter(
        tmp_path, "<p>He thought <em>I must not fear</em> and stopped.</p>"
    )
    assert chapter.text == "He thought I must not fear and stopped."
    assert chapter.emphasis == ((11, 26),)
    assert chapter.text[11:26] == "I must not fear"


def test_nested_emphasis_is_flattened_to_one_range(tmp_path: Path) -> None:
    """Nested emphasis elements flatten to a single non-nested range."""
    chapter = only_chapter(tmp_path, "<p><em>Outer <i>inner</i> tail</em></p>")
    assert len(chapter.emphasis) == 1
    start, end = chapter.emphasis[0]
    assert chapter.text[start:end] == "Outer inner tail"


def test_no_emphasis_yields_empty_tuple(tmp_path: Path) -> None:
    """A chapter with no emphasis elements records no ranges."""
    assert only_chapter(tmp_path, "<p>Plain text.</p>").emphasis == ()


def test_emphasis_survives_paragraph_boundary_collapse(tmp_path: Path) -> None:
    """Emphasis offsets stay correct once inter-paragraph newlines collapse.

    Two adjacent block elements each emit a boundary, so the raw text between
    them holds more newlines than ``normalize_text`` keeps. An emphasis run
    starting a later paragraph must still land on the right canonical offset
    once that run of newlines collapses to one blank line.
    """
    chapter = only_chapter(
        tmp_path,
        "<p>First paragraph text.</p>"
        "<p><em>Second paragraph is emphasised</em> throughout its start.</p>",
    )
    assert chapter.text == (
        "First paragraph text.\n\nSecond paragraph is emphasised throughout its start."
    )
    assert len(chapter.emphasis) == 1
    start, end = chapter.emphasis[0]
    assert chapter.text[start:end] == "Second paragraph is emphasised"


def test_repeated_phrase_does_not_misattribute_emphasis(tmp_path: Path) -> None:
    """Each emphasised range points at its own word, not an earlier look-alike.

    "alpha" also appears twice as plain narration between the two emphasised
    occurrences.
    """
    chapter = only_chapter(
        tmp_path,
        "<p><em>alpha</em> alpha decoy alpha <em>alpha</em> end.</p>",
    )
    assert len(chapter.emphasis) == EXPECTED_REPEATED_PHRASE_RANGES
    first, second = chapter.emphasis
    assert chapter.text[first[0] : first[1]] == "alpha"
    assert chapter.text[second[0] : second[1]] == "alpha"
    assert first[1] <= second[0]
    assert second[0] > chapter.text.index("decoy")


def test_whitespace_only_emphasis_is_discarded(tmp_path: Path) -> None:
    """An emphasis element with only whitespace content records no range."""
    chapter = only_chapter(tmp_path, "<p>Plain <em>   </em> text.</p>")
    assert chapter.text == "Plain text."
    assert chapter.emphasis == ()
