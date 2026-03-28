"""Shared name-phrase filter used by all NLP tiers."""
from __future__ import annotations

_MAX_NAME_WORDS = 4

_PHRASE_STARTERS = frozenset({
    "a", "an", "the", "any", "all", "some", "no", "every",
    "his", "her", "their", "its", "my", "your",
    "this", "that", "these", "those",
})

_RELATIVE_WORDS = frozenset({
    "who", "whom", "whose", "which", "that", "when", "where",
    "standing", "sitting", "named", "called", "known", "said",
    "having", "being", "wearing", "holding", "carrying",
    "watching", "looking", "walking", "speaking", "telling",
})

_COMMON_NOUNS = frozenset({
    "man", "woman", "girl", "boy", "person", "people",
    "lord", "lady", "king", "queen", "knight", "soldier",
    "sister", "sisters", "brother", "brothers",
    "guard", "servant", "merchant", "captain",
})


def _is_proper_name(text: str) -> bool:
    """Return True if *text* looks like a character name rather than a noun phrase.

    Heuristics:
    - Rejects strings longer than _MAX_NAME_WORDS words
    - Rejects strings starting with articles, quantifiers, possessives, or demonstratives
    - Rejects single-word common nouns (never standalone character names)
    - Rejects strings containing relative pronouns or common participials
    """
    words = text.split()
    if not words:
        return False
    if len(words) > _MAX_NAME_WORDS:
        return False
    first = words[0].lower()
    if first in _PHRASE_STARTERS:
        return False
    if len(words) == 1 and first in _COMMON_NOUNS:
        return False
    if any(w.lower() in _RELATIVE_WORDS for w in words):
        return False
    return True
