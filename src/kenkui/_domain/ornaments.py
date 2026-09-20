"""Typographic scene ornaments, recognised from canonical text alone.

A book divides two scenes with a rule, a blank, or a short run of separator
glyphs on a line of its own -- ``* * *``, ``#``, a lone em dash. The glyph run
is decoration standing in for a structural boundary, not something to read out,
but it survives normalization as ordinary text and so reaches the engine.

Recognition lives here because two callers need the same answer: the parser,
which decides whether a labelled block is a separator or prose, and the grid,
which marks the leaves planning must not speak.
"""

from __future__ import annotations

# A closed set rather than a punctuation category. A one-word paragraph, an
# ellipsis, or a lone em dash of dialogue must never read as an ornament, and a
# category wide enough to catch every ornament would catch those too.
ORNAMENT_GLYPHS = frozenset("*#~•◆❖⁂—–·‡§❦✦✧∗_")  # noqa: RUF001
# Long enough for "*** *** ***" and "•   •   •", short enough that no sentence
# of prose punctuation reaches it.
MAX_ORNAMENT_CHARACTERS = 24


def is_ornament(text: str) -> bool:
    """Whether this text is only separator glyphs, and short enough to be one.

    Every character must match, so ``"-- he said"`` stays prose. Emptiness is
    not an ornament: whitespace already has its own handling everywhere.
    """
    stripped = text.strip()
    return (
        bool(stripped)
        and len(stripped) <= MAX_ORNAMENT_CHARACTERS
        and all(char in ORNAMENT_GLYPHS or char.isspace() for char in stripped)
    )
