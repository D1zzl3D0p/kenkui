#!/usr/bin/env python3
"""Render every listed Calibre book with an explicit Kenkui pipeline.

Three tiers, three sources: identity (title, author, series) comes from the
driver table below; tuning (per-book pronunciation corrections and whatever
a GUI has dialed in) comes from code and each book's sidecar; style (models,
pacing, casting) is the single reusable ``house_style`` function passed
through ``.pipe()``. Change your taste in one place and every book picks it
up; change one book's tuning and no other book moves.

uv run python spikes/examples/example.py
"""

from __future__ import annotations

import logging
from pathlib import Path

import kenkui as kk

_LOGGER = logging.getLogger(__name__)

LIBRARY = Path("/Users/dizzler/Projects/Calibre Library")
NARRATOR = "ivy"
UNKNOWN = "michael"
ATTRIBUTION_MODEL = "openrouter/deepseek/deepseek-v4-flash"
# Matched to this machine's performance cores, not its core count. The
# scheduler hands each worker a contiguous static partition sized by task
# count, so on an 8P+4E CPU the efficiency-core workers get an equal share of
# the work at a fraction of the speed and the render waits on them. Measured
# over a fixed 48-segment workload: 8 workers 13.0s, 10 workers 13.9s (and the
# spread between fastest and slowest worker widens from 0.4s to 1.9s), 12
# workers 14.5s. `auto` reserves 2 of 12 and so picks 10, which is 7% slower.
WORKERS = 8

# Per-series proper-noun corrections. These still live in code -- every book
# in a series needs the same ones -- but they vary book to book (a Red Rising
# lexicon means nothing to Dune), so they are tuning (a Pronunciations rule),
# not style: notice that unlike ``house_style`` below, applying them takes
# a ``where``-shaped decision -- which book's pipeline gets which lexicon.
LEXICONS = {
    "dune": {
        "Muad'Dib": "Moo-ahd-Deeb",
        "Bene Gesserit": "Ben-eh Jess-er-it",
        "Kwisatz Haderach": "Kwih-sats Hah-der-ock",
        "Shai-Hulud": "Shy Hoo-lood",
        "Sardaukar": "Sar-doh-kar",
        "Atreides": "Ah-tray-deez",
        "Harkonnen": "Har-koh-nen",
        "Chani": "Chah-nee",
        "Ghola": "Goh-lah",
        "Leto": "Lay-toh",
    },
    "red-rising": {
        "Darrow": "Dair-oh",
        "Sevro": "Sev-roh",
        "Eo": "Ee-oh",
        "Cassius": "Cash-us",
        "Lysander": "Lie-san-der",
        "Telemanus": "Tell-a-man-us",
        "gravBoots": "grav boots",
        "pulseFist": "pulse fist",
        "clawDrill": "claw drill",
        "ArchGovernor": "Arch Governor",
    },
}

BOOKS = [
    ("Red Rising", "Pierce Brown", "Red Rising (391)", "red-rising", 1),
    ("Golden Son", "Pierce Brown", "Golden Son (445)", "red-rising", 2),
    ("Morning Star", "Pierce Brown", "Morning Star (446)", "red-rising", 3),
    ("Iron Gold", "Pierce Brown", "Iron Gold (442)", "red-rising", 4),
    ("Dark Age", "Pierce Brown", "Dark Age (444)", "red-rising", 5),
    ("Light Bringer", "Pierce Brown", "Light Bringer (443)", "red-rising", 6),
    ("Dune", "Frank Herbert", "Dune (466)", "dune", 1),
    ("Dune Messiah", "Frank Herbert", "Dune Messiah (55)", "dune", 2),
    ("Children of Dune", "Frank Herbert", "Children of Dune (54)", "dune", 3),
    ("God Emperor of Dune", "Frank Herbert", "God Emperor of Dune (56)", "dune", 4),
    (
        "The Subtle Art of Folding Space",
        "John Chu",
        "The Subtle Art of Folding Space (465)",
        None,
        None,
    ),
]


def report_progress(event: object) -> None:
    """Print stage boundaries and measured render progress."""
    if isinstance(event, kk.StageStarted):
        print(f"\n== {event.stage} ==", flush=True)
    elif isinstance(event, kk.StageProgress):
        chapter = f" ({event.chapter_id})" if event.chapter_id else ""
        print(f"{event.stage}: {event.completed}/{event.total}{chapter}", flush=True)
    elif isinstance(event, kk.StageCompleted):
        print(f"== {event.stage} complete ==", flush=True)


def house_style(pipeline: kk.Pipeline) -> kk.Pipeline:
    """Apply the studio's reusable taste: models, pacing, numbers, casting.

    Nothing here takes ``where``, and that is the point: change this
    function and every book that pipes through it changes with it, which is
    what makes it style rather than tuning.
    """
    return (
        pipeline.pronounce(numbers="standard")
        .pauses(chapter_ms=1200, heading_after_ms=500, paragraph_ms=300)
        .infer_characters("spacy")
        .attribute_quotes(ATTRIBUTION_MODEL)
        .assign_voices(narrator=NARRATOR, unknown=UNKNOWN)
    )


def explicit_run(
    title: str, author: str, epub: Path, series: str | None, volume: int | None
) -> kk.Result:
    """Render one book from the driver table, its own sidecar, and house style.

    Identity comes from this function's arguments, tuning from the series
    lexicon and this book's sidecar (if one exists), and style from
    ``house_style``.
    """
    cover = epub.parent / "cover.jpg"
    pipeline = kk.book(epub).metadata(
        title=title,
        author=author,
        cover=cover if cover.exists() else "source",
    )
    if series:
        pipeline = pipeline.series(series, book=volume)

    # Tuning: this book's corrections. The series lexicon is declared here
    # because it varies by book; the sidecar layers a GUI-driven dial-in
    # session, if one has ever been saved, beneath it.
    lexicon = LEXICONS.get(series or "", {})
    if lexicon:
        pipeline = pipeline.pronounce(lexicon)
    sidecar = epub.with_suffix(".kenkui.json")
    if sidecar.exists():
        pipeline = pipeline.annotations()

    # Style: the studio's taste, unrelated to which book this is.
    pipeline = pipeline.pipe(house_style)

    # Replaces the previous render rather than refusing to run beside it.
    # Publication is the last step, so a book that fails anywhere earlier
    # leaves its existing M4B untouched.
    return pipeline.tts().write(
        epub.with_suffix(".m4b"),
        on_event=report_progress,
        workers=WORKERS,
        overwrite=True,
    )


def magic_run(epub: Path) -> kk.Result:
    """Use the compact alternative without cover or series configuration."""
    return kk.magic_run(epub, narrator=NARRATOR, multi=True, model=ATTRIBUTION_MODEL)


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )
    for title, author, folder, series, volume in BOOKS:
        epub = LIBRARY / author / folder / f"{title} - {author}.epub"
        try:
            explicit_run(title, author, epub, series, volume)
        except kk.KenkuiError:
            _LOGGER.exception("book_render_failed: %s", title)


if __name__ == "__main__":
    main()
