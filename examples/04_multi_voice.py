"""Give every character their own voice.

Three stages, each chosen independently:

1. ``infer_characters`` finds who is in the book. ``"spacy"`` runs locally
   (``pip install "kenkui[spacy]"`` and ``python -m spacy download
   en_core_web_lg``); any LiteLLM model identifier works instead.
2. ``attribute_quotes`` decides who speaks each line of dialogue, using a
   LiteLLM model. This is the step that costs money.
3. ``assign_voices`` casts characters onto loaded voices. It is free and
   deterministic: same book, same pool, same cast.

Credentials come from the provider's own environment variable, never from
pipeline arguments. For the default OpenRouter model::

    export OPENROUTER_API_KEY=...
    python examples/04_multi_voice.py path/to/book.epub
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import kenkui as kk

MODEL = os.environ.get("KENKUI_MODEL", "openrouter/deepseek/deepseek-v4-flash")
NARRATOR = "eponine"
UNKNOWN = "paul"


def load_pool() -> None:
    """Load the narrator, the unknown-speaker voice, and a pool to cast from.

    Casting only draws on loaded voices. With one loaded voice the whole book
    is spoken by it; more voices make characters more distinct.
    """
    pool = [NARRATOR, UNKNOWN, "anna", "charles", "george", "jane", "mary", "vera"]
    for voice_id in pool:
        kk.load_voice(voice_id)


def show_cast(event: kk.ExecutionEvent) -> None:
    """Print the cast the moment it is decided, before any audio is rendered."""
    if isinstance(event, kk.CastResolved):
        for character, voice in event.assignments:
            print(f"  {character:<24} -> {voice}")
    elif isinstance(event, kk.StageStarted):
        print(f"== {event.stage}")


def main(epub: Path) -> None:
    """Resolve a cast, show it, then render with it."""
    load_pool()

    pipeline = (
        kk.book(epub)
        # identity= adds two cheap reasoning calls that merge aliases ("Paul",
        # "Muad'Dib") and drop non-people. identity=None stays fully offline.
        .infer_characters("spacy")
        .attribute_quotes(MODEL)
        # unknown= voices dialogue nobody could be placed for; by default it
        # is the narrator, so unplaced lines sound like narration.
        # cast= pins a character to a voice; the solver fills in the rest.
        .assign_voices(narrator=NARRATOR, unknown=UNKNOWN, method="gendered")
    )

    # resolve() is optional -- write() does it too -- but running it first
    # lets you look at the cast before paying for synthesis. Attribution is
    # stored, so resolving again (or rendering) makes no further model calls.
    resolved = pipeline.resolve(on_event=show_cast)
    casting = resolved.inspect().casting
    assert casting is not None
    print(
        f"\n{len(casting.characters)} characters, "
        f"{len(casting.spans)} attributed lines, "
        f"{len(casting.collisions)} same-chapter voice shares"
    )
    for character in casting.characters[:10]:
        print(
            f"  {character.display_name:<24} {character.gender or '?':<10} "
            f"{character.spoken_characters} chars"
        )

    result = resolved.tts().write(epub.with_suffix(".m4b"), overwrite=True)
    print(f"\nwrote {result.output}")


if __name__ == "__main__":
    match sys.argv[1:]:
        case [path]:
            main(Path(path))
        case _:
            raise SystemExit(__doc__)
