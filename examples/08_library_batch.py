"""Render a whole library from a table of books.

Identity (title, author, series) comes from the table below, tuning from the
per-series lexicon and each book's saved sidecar, and style from one
``house_style`` function. One book failing is logged and the batch moves on;
an existing M4B is only replaced once its successor is fully built.

    python examples/08_library_batch.py "/path/to/Calibre Library"
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path
from typing import Literal, NamedTuple

import kenkui as kk

_LOGGER = logging.getLogger(__name__)

MODEL = os.environ.get("KENKUI_MODEL", "openrouter/deepseek/deepseek-v4-flash")
NARRATOR = "ivy"
UNKNOWN = "michael"
# "auto" reserves two CPUs. On CPUs with efficiency cores, setting this to the
# number of performance cores is often faster: every worker gets an equal
# share of the book, so the render waits on the slowest core.
WORKERS: int | Literal["auto"] = "auto"

LEXICONS = {
    "dune": {"Atreides": "Ah-tray-deez", "Harkonnen": "Har-koh-nen"},
    "red-rising": {"Darrow": "Dair-oh", "Sevro": "Sev-roh"},
}


class Entry(NamedTuple):
    """One book: where it lives, and what it should be called."""

    title: str
    author: str
    path: str  # relative to the library root
    series: str | None = None
    volume: int | None = None


BOOKS = [
    Entry("Dune", "Frank Herbert", "Frank Herbert/Dune/Dune.epub", "dune", 1),
    Entry(
        "Dune Messiah",
        "Frank Herbert",
        "Frank Herbert/Dune Messiah/Dune Messiah.epub",
        "dune",
        2,
    ),
    Entry(
        "Red Rising",
        "Pierce Brown",
        "Pierce Brown/Red Rising/Red Rising.epub",
        "red-rising",
        1,
    ),
    Entry("Persuasion", "Jane Austen", "Jane Austen/Persuasion/Persuasion.epub"),
]


def report(event: kk.ExecutionEvent) -> None:
    """Print stage boundaries and progress."""
    if isinstance(event, kk.StageStarted):
        print(f"\n== {event.stage}", flush=True)
    elif isinstance(event, kk.StageProgress):
        print(f"{event.stage}: {event.completed}/{event.total}", flush=True)


def house_style(pipeline: kk.Pipeline) -> kk.Pipeline:
    """Reusable taste shared by every book."""
    return (
        pipeline.pronounce(numbers="standard")
        .pauses(chapter_ms=1200, heading_after_ms=500, paragraph_ms=300)
        .infer_characters("spacy")
        .attribute_quotes(MODEL)
        .assign_voices(narrator=NARRATOR, unknown=UNKNOWN)
    )


def render(library: Path, entry: Entry) -> kk.Result:
    """Render one table entry beside its EPUB."""
    epub = library / entry.path
    cover = epub.with_name("cover.jpg")  # Calibre keeps one beside each book
    pipeline = kk.book(epub).metadata(
        title=entry.title,
        author=entry.author,
        cover=cover if cover.exists() else "source",
    )
    if entry.series:
        pipeline = pipeline.series(entry.series, book=entry.volume)
        pipeline = pipeline.pronounce(LEXICONS.get(entry.series, {}))
    pipeline = pipeline.annotations().pipe(house_style)
    return pipeline.tts().write(
        epub.with_suffix(".m4b"),
        on_event=report,
        workers=WORKERS,
        overwrite=True,
    )


def main(library: Path) -> None:
    """Render every book in the table, continuing past failures."""
    logging.basicConfig(
        level=logging.INFO, format="%(levelname)s %(name)s: %(message)s"
    )
    for voice_id in (NARRATOR, UNKNOWN, "anna", "charles", "george", "jane", "vera"):
        kk.load_voice(voice_id)
    for entry in BOOKS:
        try:
            result = render(library, entry)
        except kk.KenkuiError:
            _LOGGER.exception("book_render_failed: %s", entry.title)
        else:
            print(f"wrote {result.output}")


if __name__ == "__main__":
    match sys.argv[1:]:
        case [path]:
            main(Path(path))
        case _:
            raise SystemExit(__doc__)
