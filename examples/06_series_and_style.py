"""Keep characters' voices across a series, and reuse one house style.

A pipeline's intent falls into three tiers:

* identity -- which book (source, selection, metadata, series);
* style    -- taste that travels between books (pacing, numbers, casting);
* tuning   -- corrections that belong to one book or series (lexicons,
  per-line attribution and silence).

Keeping style in one ordinary function applied with ``.pipe()`` means a
change of taste reaches every book, while each book keeps its own tuning.

    python examples/06_series_and_style.py book1.epub book2.epub book3.epub
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import kenkui as kk

MODEL = os.environ.get("KENKUI_MODEL", "openrouter/deepseek/deepseek-v4-flash")
SERIES = "dune"
NARRATOR = "ivy"
UNKNOWN = "michael"

# Tuning shared by every book in this series: a proper-noun lexicon.
# Entries match whole words case-insensitively and keep the source's casing.
LEXICON = {
    "Atreides": "Ah-tray-deez",
    "Bene Gesserit": "Ben-eh Jess-er-it",
    "Harkonnen": "Har-koh-nen",
    "Kwisatz Haderach": "Kwih-sats Hah-der-ock",
}


def house_style(pipeline: kk.Pipeline) -> kk.Pipeline:
    """Apply reusable taste: pacing, number reading, and how casting works.

    Nothing here mentions a particular book, which is what makes it style.
    """
    return (
        pipeline.pronounce(numbers="standard")
        .pauses(chapter_ms=1200, heading_after_ms=500, paragraph_ms=300)
        .infer_characters("spacy")
        .attribute_quotes(MODEL)
        .assign_voices(narrator=NARRATOR, unknown=UNKNOWN)
    )


def volume(epub: Path, number: int) -> kk.Pipeline:
    """Build one volume's pipeline from identity, tuning, and style."""
    return (
        kk.book(epub)
        # Series membership is declared, never guessed from the EPUB. A
        # character the series already cast keeps their voice; newcomers are
        # cast from the least-used voices so far.
        .series(SERIES, book=number)
        .pronounce(LEXICON)
        # Load corrections saved by 07_dial_in.py, if this book has any.
        .annotations()
        .pipe(house_style)
    )


def main(epubs: list[Path]) -> None:
    """Render each volume in order, then show what the series remembers."""
    for voice_id in (NARRATOR, UNKNOWN, "anna", "charles", "george", "jane", "vera"):
        kk.load_voice(voice_id)

    for number, epub in enumerate(epubs, start=1):
        pipeline = volume(epub, number)
        # The tiers are cheap read-only summaries: no parsing, no I/O.
        print(pipeline.identity)
        print(pipeline.style)
        print(pipeline.tuning)

        # validate() refuses a series contradiction before any model call:
        # series_voice_missing if a voice the series used is not loaded, and
        # series_narrator_changed if NARRATOR differs from earlier volumes.
        # .series(..., allow_recast=True / allow_narrator_change=True) accepts
        # the change and makes it stick for later volumes.
        pipeline.tts().write(epub.with_suffix(".m4b"), overwrite=True)

    for record in kk.list_series():
        print(f"\n{record.series_id} (narrator {record.narrator_voice_id})")
        for character in record.characters[:10]:
            print(f"  {character.display_name:<24} -> {character.voice_id}")

    # To fix one character's voice from here on, pin it on the next render:
    #   .assign_voices(narrator=NARRATOR, cast={"paul": "charles"})
    # The series adopts the pin for this and every later volume.


if __name__ == "__main__":
    if len(sys.argv) < 2:  # noqa: PLR2004 - program name plus at least one book
        raise SystemExit(__doc__)
    main([Path(arg) for arg in sys.argv[1:]])
