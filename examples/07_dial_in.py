"""Read a book's script, correct a line, hear the fix, and save it.

The dial-in loop is how you fix a specific moment without re-rendering a
book to find out whether the fix worked:

1. ``script()`` shows what will be said, by whom, and *why*.
2. ``attribute()``, ``silence()``, and scoped ``pronounce()`` correct it.
3. ``select(...).preview()`` renders only the corrected lines to a WAV.
4. ``write_annotations()`` saves the corrections beside the EPUB.

Positions are addressed with ``where=``: a path through chapter, paragraph,
line, sentence, and phrase. Each level takes an index, ``"*"``, ``-1`` for the
last, a list, or an inclusive ``"lo..hi"`` range.

This example assumes a book where the first paragraph of every chapter is an
epigraph that should be read by a particular character, as in *Dune*.

    python examples/07_dial_in.py path/to/dune.epub
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import kenkui as kk

MODEL = os.environ.get("KENKUI_MODEL", "openrouter/deepseek/deepseek-v4-flash")


def main(epub: Path) -> None:
    """Attribute every chapter's epigraph to Irulan, probe it, and save."""
    kk.load_voice("ivy")
    book = (
        kk.book(epub)
        # A book that has never been dialed in has no sidecar yet;
        # annotations() then starts from an empty baseline.
        .annotations()
        .infer_characters("spacy")
        .attribute_quotes(MODEL)
        .assign_voices(narrator="ivy")
        # Resolve once so script() shows real attribution instead of
        # "unresolved". It is stored, so this is only paid for once.
        .resolve()
    )

    # 1. Read. provenance says which layer decided each row: "default"
    #    (narration), "machine" (the attribution model), or "rule" (yours).
    where = {"chapter": "*", "paragraph": 1}
    for row in book.script().at(where):
        print(
            row.path, row.character, row.provenance, row.silence_after_ms, row.text[:60]
        )

    # 2. Correct. Rules accumulate; a narrower pattern beats a broader one.
    #    Replace "irulan" with a character ID from your roster.
    book = book.attribute("irulan", where=where).silence(900, where=where)

    # 3. Probe. Only the selected paragraphs are synthesized, and the audio
    #    is cached, so the eventual full render reuses it.
    probe = epub.with_name(f"{epub.stem}.probe.wav")
    first_chapter = book.inspect().chapters[0].id
    book.select({"chapter": first_chapter, "paragraph": 1}).preview(
        probe, overwrite=True
    )
    print(f"listen: {probe}")

    # 4. Save. Rules are anchored to the text they matched, so an edit
    #    elsewhere in the book cannot move them onto the wrong paragraph.
    sidecar = book.write_annotations()
    print(f"saved corrections to {sidecar}")


if __name__ == "__main__":
    match sys.argv[1:]:
        case [path]:
            main(Path(path))
        case _:
            raise SystemExit(__doc__)
