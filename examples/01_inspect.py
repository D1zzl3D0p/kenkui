"""Look inside an EPUB before spending anything on it.

Nothing here downloads a model, calls a provider, or renders audio, so it is
the safe first thing to run against a new book. Chapter IDs printed here are
what every later example uses to select and address parts of the book.

    python examples/01_inspect.py path/to/book.epub
"""

from __future__ import annotations

import sys
from pathlib import Path

import kenkui as kk


def main(epub: Path) -> None:
    """Print a book's metadata, chapters, and the voices available to cast."""
    # Building a pipeline records intent and opens nothing.
    book = kk.book(epub)

    # inspect() parses the EPUB (with archive and XML safety limits) and
    # returns frozen metadata plus one entry per chapter with visible text.
    inspection = book.inspect()
    print(f"{inspection.metadata.title} by {inspection.metadata.author}")
    print(f"cover in source: {inspection.metadata.cover_available}")
    for chapter in inspection.chapters:
        print(
            f"  {chapter.index:>3}  {chapter.id:<30} {chapter.speech_characters:>7}  "
            f"{chapter.title}"
        )

    # validate() is cheaper still: it checks intent, not the book. A pipeline
    # with no voice and no tts() reports exactly what it is missing.
    for issue in book.validate().issues:
        print(f"{issue.severity}: {issue.code.value} - {issue.message}")

    # The built-in catalog is metadata shipped in the wheel. "registered"
    # voices can be loaded on demand; "loaded" ones are already on disk.
    english = [v for v in kk.list_voices() if v.language == "english"]
    print(f"\n{len(english)} English voices, for example:")
    for voice in english[:8]:
        print(f"  {voice.id:<16} {voice.perceived_gender or '-':<10} {voice.state}")


if __name__ == "__main__":
    match sys.argv[1:]:
        case [path]:
            main(Path(path))
        case _:
            raise SystemExit(__doc__)
