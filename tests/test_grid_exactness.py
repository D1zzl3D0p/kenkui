"""The grid's only safety property, asserted over real books.

Boundary placement is a quality metric to tune. Partition exactness is not
negotiable: a gap silently drops audio and an overlap silently duplicates it,
which is the same reasoning as _characters/quotes.py:7-10.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from kenkui._domain.grid import build_grid, unit_text
from kenkui.api import book
from kenkui.errors import SourceError

LIBRARY = Path("/Users/dizzler/Projects/Calibre Library")


def _epubs() -> list[Path]:
    if not LIBRARY.is_dir():
        return []
    return sorted(LIBRARY.rglob("*.epub"))[:40]


@pytest.mark.corpus
@pytest.mark.skipif(
    not os.environ.get("KENKUI_RUN_CORPUS"), reason="set KENKUI_RUN_CORPUS=1 to run"
)
@pytest.mark.parametrize("epub", _epubs(), ids=lambda p: p.stem)
def test_grid_partitions_every_chapter_exactly(epub: Path) -> None:
    """Every parseable corpus chapter is partitioned without gaps or overlaps."""
    try:
        chapters = book(epub).inspect().chapters
    except SourceError as error:
        pytest.skip(f"pre-existing parse failure: {error}")
    for chapter in chapters:
        units = build_grid(chapter)
        rebuilt = "".join(unit_text(unit, chapter.text) for unit in units)
        assert rebuilt == chapter.text, f"{epub.stem} / {chapter.id}"
