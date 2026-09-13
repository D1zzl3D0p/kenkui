"""Correct the character list before any dialogue is attributed.

Discovery can split one person in two, miss a nickname, or guess a gender
wrong. Fixing that before attribution is cheaper than fixing a cast after,
because the reviewed roster is what attribution works from.

    python examples/05_review_characters.py path/to/book.epub
"""

from __future__ import annotations

import os
import sys
from dataclasses import replace
from pathlib import Path

import kenkui as kk

MODEL = os.environ.get("KENKUI_MODEL", "openrouter/deepseek/deepseek-v4-flash")


def review(roster: kk.CharacterRoster) -> kk.CharacterRoster:
    """Apply corrections with ordinary Python: the roster is a frozen dataclass.

    Replace this body with the edits your book needs. Renaming, adding and
    removing characters, and changing aliases or gender are all allowed.
    """
    corrected = []
    for character in roster.characters:
        if character.id == "paul":
            character = replace(  # noqa: PLW2901 - rebinding is the point here
                character,
                display_name="Paul Atreides",
                aliases=(*character.aliases, "Usul", "Muad'Dib"),
                gender="masculine",
            )
        corrected.append(character)
    # Drop anything discovery mistook for a person.
    corrected = [c for c in corrected if c.id != "shai-hulud"]
    return replace(roster, characters=tuple(corrected))


def main(epub: Path) -> None:
    """Discover characters, print them, apply corrections, then cast and render."""
    # Stop after discovery: no narrator voice, attribution, or tts() needed.
    discovered = kk.book(epub).infer_characters("spacy").resolve(until="characters")
    roster = discovered.inspect().roster
    assert roster is not None
    for character in roster.characters:
        print(
            f"{character.id:<20} {character.display_name:<24} "
            f"{character.gender or '?':<10} {', '.join(character.aliases)}"
        )

    # with_characters() does no I/O. Invalid edits (duplicate IDs, unknown
    # chapter IDs) raise ValidationError with code invalid_roster right here.
    reviewed = discovered.with_characters(review(roster))

    # Continuing reuses the reviewed roster instead of rediscovering it.
    kk.load_voice("eponine")
    cast = reviewed.attribute_quotes(MODEL).assign_voices(narrator="eponine").resolve()
    casting = cast.inspect().casting
    assert casting is not None
    print(dict(casting.assignments))
    cast.tts().write(epub.with_suffix(".m4b"), overwrite=True)


if __name__ == "__main__":
    match sys.argv[1:]:
        case [path]:
            main(Path(path))
        case _:
            raise SystemExit(__doc__)
