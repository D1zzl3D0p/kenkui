"""The addressable grid: a partition of chapter text that depends on nothing else.

Every other partition in this codebase takes a setting, a budget, or a model.
This one takes only the text, which is what lets an annotation anchored to it
survive a change to pause tiers, chunk budgets, or the attribution provider.

Biased to split. Over-splitting costs one extra row in a reviewer's view and
nothing in the audio, because a grid boundary becomes a segment boundary only
when an annotation attaches to it. Under-splitting traps two speakers in one
addressable unit and makes the correction inexpressible. The two failures are
not symmetric, so the guard lists below are allowed to be short.
"""

from __future__ import annotations

import re

from kenkui._characters.identity import PREFIX_TITLES

# A terminator, any closing brackets or quotes, then whitespace. The lookahead
# keeps the whitespace with the preceding part so joining stays exact.
# Includes both ASCII and typographic (curly) quotes: " ' " ' " « » « »
_SENTENCE: re.Pattern[str] = re.compile('[.!?…]+["\'\'""\\)\\]]*\\s+')
_PHRASE: re.Pattern[str] = re.compile('[,;:]["\'\'""\\)\\]]*\\s+')

# A single capital before the period is an initial ("J. R. Smith"), not a
# sentence end.
_INITIAL: re.Pattern[str] = re.compile(r"(?:^|\s)[A-Z]\.$")


def _is_abbreviation(prefix: str) -> bool:
    """Whether a candidate sentence end is really an abbreviation."""
    if _INITIAL.search(prefix):
        return True
    trailing = re.search(r"([A-Za-z]+)\.$", prefix)
    return trailing is not None and trailing.group(1).casefold() in PREFIX_TITLES


def _split(text: str, pattern: re.Pattern[str], *, guard: bool) -> tuple[str, ...]:
    if not text:
        return ()
    parts: list[str] = []
    position = 0
    for match in pattern.finditer(text):
        if guard and _is_abbreviation(text[position : match.start() + 1]):
            continue
        parts.append(text[position : match.end()])
        position = match.end()
    if position < len(text):
        parts.append(text[position:])
    return tuple(parts) if parts else (text,)


def split_sentences(text: str) -> tuple[str, ...]:
    """Split on sentence-terminal punctuation, guarding abbreviations."""
    return _split(text, _SENTENCE, guard=True)


def split_phrases(text: str) -> tuple[str, ...]:
    """Split on clause punctuation. No guard: commas do not abbreviate."""
    return _split(text, _PHRASE, guard=False)
