#!/usr/bin/env python3
"""Render only the Calibre books whose earlier runs failed at assembly.

Same explicit Kenkui pipeline as example.py; the BOOKS list is limited to
the seven books that failed on 2026-08-28/29 (ffmpeg stderr overflow).

uv run python spikes/examples/example_failed.py
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

# Published 2026-08-30: Morning Star, Iron Gold, Dark Age, Light Bringer, Dune.
BOOKS = [
    ("Dune Messiah", "Frank Herbert", "Dune Messiah (55)", "dune", 2),
    ("Children of Dune", "Frank Herbert", "Children of Dune (54)", "dune", 3),
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


def explicit_run(
    title: str, author: str, epub: Path, series: str | None, volume: int | None
) -> kk.Result:
    """Render one book while showing each configurable pipeline stage."""
    lexicon = kk.builtin_lexicon()
    lexicon.update(LEXICONS.get(series or "", {}))
    cover = epub.parent / "cover.jpg"
    pipeline = (
        kk.book(epub)
        # .pronounce(lexicon)
        .metadata(
            title=title,
            author=author,
            cover=cover if cover.exists() else "source",
        )
        .infer_characters("spacy")
        .attribute_quotes(ATTRIBUTION_MODEL)
        .assign_voices(narrator=NARRATOR, unknown=UNKNOWN)
    )
    if series:
        pipeline = pipeline.series(series, book=volume)
    return pipeline.tts().write(epub.with_suffix(".m4b"), on_event=report_progress)


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
