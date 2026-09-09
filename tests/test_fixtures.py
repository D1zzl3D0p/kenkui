"""Prove the shared book fixtures in conftest.py resolve as intended."""

from __future__ import annotations

from typing import TYPE_CHECKING

import kenkui as kk
from conftest import CH08_ID, CH09_ID

if TYPE_CHECKING:
    from pathlib import Path

    from kenkui._domain.planning import SpeakerSpan
    from kenkui.inspection import ChapterInspection


def test_epub_fixture_parses(epub_path: Path) -> None:
    """Both chapters parse, in spine order, with their expected stable IDs."""
    chapters = kk.book(epub_path).inspect().chapters
    assert [chapter.id for chapter in chapters] == [CH08_ID, CH09_ID]


def test_machine_spans_tile_the_chapter(
    chapter_ch08: ChapterInspection, machine_spans: tuple[SpeakerSpan, ...]
) -> None:
    """The spans reconstruct the chapter's text exactly, with no gap or overlap."""
    rebuilt = "".join(chapter_ch08.text[s.start : s.end] for s in machine_spans)
    assert rebuilt == chapter_ch08.text


def test_resolved_book_needs_no_network(resolved_book: kk.Pipeline) -> None:
    """Resolution completes, and produces a casting inspection, with no network."""
    assert resolved_book.inspect().casting is not None
