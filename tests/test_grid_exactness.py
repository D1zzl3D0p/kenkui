"""The grid's only safety property, asserted over real books.

Boundary placement is a quality metric to tune. Partition exactness is not
negotiable: a gap silently drops audio and an overlap silently duplicates it,
which is the same reasoning as _domain/quotes.py:7-10.
"""

from __future__ import annotations

import os
from pathlib import Path as FilePath

import pytest

from kenkui._domain.grid import (
    build_grid,
    build_structure_index,
    dialogue_ranges,
    unit_text,
)
from kenkui._domain.paths import Path as GridPath
from kenkui._domain.quotes import extract_spans
from kenkui.api import book
from kenkui.errors import SourceError

LIBRARY = FilePath("/Users/dizzler/Projects/Calibre Library")


def _epubs() -> list[FilePath]:
    if not LIBRARY.is_dir():
        return []
    return sorted(LIBRARY.rglob("*.epub"))[:40]


@pytest.mark.corpus
@pytest.mark.skipif(
    not os.environ.get("KENKUI_RUN_CORPUS"), reason="set KENKUI_RUN_CORPUS=1 to run"
)
@pytest.mark.parametrize("epub", _epubs(), ids=lambda p: p.stem)
def test_grid_partitions_every_chapter_exactly(epub: FilePath) -> None:
    """Every parseable corpus chapter is partitioned without gaps or overlaps."""
    try:
        chapters = book(epub).inspect().chapters
    except SourceError as error:
        pytest.skip(f"pre-existing parse failure: {error}")
    for chapter in chapters:
        units = build_grid(chapter)
        rebuilt = "".join(unit_text(unit, chapter.text) for unit in units)
        assert rebuilt == chapter.text, f"{epub.stem} / {chapter.id}"
        assert tuple(
            (span.start, span.end) for span in dialogue_ranges(units)
        ) == tuple(
            (span.start, span.end)
            for span in extract_spans(chapter.id, chapter.text)
            if span.is_dialogue
        ), f"{epub.stem} / {chapter.id}"
        index = build_structure_index(units)
        assert index == build_structure_index(tuple(units))
        assert len(index.gaps) == len(units)
        for leaf_index, unit in enumerate(units):
            prefixes = (
                GridPath(chapter=unit.chapter_id),
                GridPath(chapter=unit.chapter_id, paragraph=unit.paragraph),
                GridPath(
                    chapter=unit.chapter_id,
                    paragraph=unit.paragraph,
                    line=unit.line,
                ),
                GridPath(
                    chapter=unit.chapter_id,
                    paragraph=unit.paragraph,
                    line=unit.line,
                    sentence=unit.sentence,
                ),
                GridPath(
                    chapter=unit.chapter_id,
                    paragraph=unit.paragraph,
                    line=unit.line,
                    sentence=unit.sentence,
                    phrase=unit.phrase,
                ),
            )
            assert all(
                index[prefix].first <= leaf_index < index[prefix].past_last
                for prefix in prefixes
            )
