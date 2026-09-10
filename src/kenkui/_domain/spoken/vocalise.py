"""Collapse hyphenated vocal gestures into words the engine can pronounce.

The synthesis model reads an orphaned letter as the letter's name, so "Ah-h"
comes out "ah-aitch". Two gestures are spelled that way in prose and neither is
vocabulary: an elongation holds a sound ("Ah-h-h", "Wel-l-l-l"), and a stammer
restarts a word ("S-s-sorry"). Both are rewritten into a single pronounceable
word; every other hyphen in the language is left exactly as written.

Matching is whole-token and single-shot, like the lexicon: one match rewrites
one token and the output is never re-examined, so a chain such as
"Um-m-m-m-ah-hm-m-m" is resolved in the one pass rather than by iterating.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Final

if TYPE_CHECKING:
    from collections.abc import Mapping

    from kenkui._domain.spoken.numbers import Rule

# Both default on: calling `pronounce` at all is the opt-in, and a caller who
# wants one gesture and not the other declines it by name.
VOCALISE_FEATURES: Final[dict[str, bool]] = {"elongation": True, "stammer": True}

# Only the true hyphens. An en or em dash between words is punctuation -- an
# interruption or an aside -- and joining across one would weld two clauses
# into a single word.
_DASH = r"[-\u2010\u2011]"
_LB = r"(?<![0-9A-Za-z])"
_RB = r"(?![0-9A-Za-z])"
_TOKEN = re.compile(rf"{_LB}[A-Za-z]+(?:{_DASH}[A-Za-z]+)+{_RB}")
_SPLIT = re.compile(_DASH)


def _echoes(word: str, part: str) -> bool:
    """Whether a part is one letter, repeated, continuing the sound before it.

    The letter has to match what it follows: this is what separates the "-h"
    of "Ah-h" from the "-o" of "will-o", which starts a syllable of its own.
    """
    return len(set(part.lower())) == 1 and word[-1:].lower() == part[0].lower()


def _is_onset(word: str, part: str, following: str | None) -> bool:
    """Whether a single letter is a stammer's false start, not an elongation.

    "S-s-sorry" and "Ah-h" have the same shape at the second part, and only
    context tells them apart. What comes after must be the word the letter
    restarts, and what comes before must be nothing but that same letter: in
    "Um-m-m-m-ah-h-h-hm-m-m" the third "h" also precedes "hm", but it is
    holding the "ahh" already built and belongs to it.

    Without this the "s" of "S-s-sorry" is absorbed into "S" and the token
    reads "Ss sorry" -- worse than leaving it alone.
    """
    return (
        len(part) == 1
        and len(set(word.lower())) == 1
        and following is not None
        and len(following) > 1
        and following.lower().startswith(part.lower())
    )


def _stammer_span(parts: list[str]) -> int:
    """How many leading parts are false starts of the word that follows them.

    Single letters only. Two-letter prefixes are ordinary word formation --
    "co-conspirator" and "re-reading" have exactly this shape and are not
    stammers -- and admitting them costs more than the gesture is worth.
    """
    count = 0
    while count < len(parts) - 1 and len(parts[count]) == 1:
        count += 1
    if count == 0:
        return 0
    word = parts[count]
    if len(word) == 1 or not word.lower().startswith(parts[count - 1].lower()):
        return 0
    return count


def _rewrite(token: str, *, elongation: bool, stammer: bool) -> str | None:
    """Return the spoken form of one hyphenated token, or None to decline.

    Declining leaves the token exactly as written, which is what every
    compound, proper noun, and reduplication ("drip-drip-drip") wants.
    """
    parts = _SPLIT.split(token)
    start = _stammer_span(parts) if stammer else 0
    words = ["".join(parts[: start + 1])]
    rest = parts[start + 1 :]
    changed = start > 0
    for index, part in enumerate(rest):
        following = rest[index + 1] if index + 1 < len(rest) else None
        if (
            elongation
            and not _is_onset(words[-1], part, following)
            and _echoes(words[-1], part)
        ):
            words[-1] += part
            changed = True
        else:
            words.append(part)
    if not changed:
        return None
    # A part that was not absorbed begins a new gesture rather than continuing
    # one, so it becomes its own word: joining gives "Ummmmahhhhmmm".
    return " ".join(words)


def vocalise_rules(overrides: Mapping[str, bool] | None = None) -> tuple[Rule, ...]:
    """Return the single rule covering whichever gestures are enabled."""
    wanted = overrides or {}
    elongation = wanted.get("elongation", VOCALISE_FEATURES["elongation"])
    stammer = wanted.get("stammer", VOCALISE_FEATURES["stammer"])
    if not (elongation or stammer):
        return ()

    def handler(match: re.Match[str]) -> str | None:
        return _rewrite(match.group(0), elongation=elongation, stammer=stammer)

    return ((_TOKEN, handler),)
