"""Recognising narration written in the first person.

A narrator who says `"..." I said` is unattributable twice over: the roster is
built from names, and "I" is not one, while the prompt forbids answering with a
pronoun. This finds the tags that say a chapter is narrated that way, so the
roster pass can ask who is speaking them.

Detection is deliberately the cheap half. Which character narrates is a
question for the model, which is already reading the chapter; whether the
question is worth asking is decided here, for nothing.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence

_VERBS = (
    r"said|says|asked|asks|replied|answered|told|murmured|muttered|whispered"
    r"|shouted|called|cried|repeated|added|agreed|demanded|admitted|observed"
    r"|managed|offered|insisted|protested|snapped|breathed"
)

# "I said" and "said I", the two orders English puts a first-person tag in.
# Bounded so the match cannot run past the tag into the next sentence. The
# exclusion class also covers the curly single quote pair, written below as
# \u2018/\u2019 escapes rather than literal glyphs so they read as plain
# ASCII in the source -- quotes.py treats that pair as dialogue delimiters
# too, for British prose, and without them here the scan for "I" ... VERB
# runs straight through a closing curly single quote into the next quote,
# tagging whoever speaks it as first person. The straight single quote is
# deliberately left out: quotes.py does not treat it as a delimiter either,
# since in English prose it is almost always an apostrophe, and there is no
# delimiter here for the scan to run past.
#
# U+2019 is ambiguous in a way U+2018 is not: it closes dialogue, but it is
# also what a professionally typeset EPUB uses for every apostrophe, so
# "I'd", "I've", "don't" are ordinary between a quote's end and its tag verb.
# Excluding it unconditionally, as above, is therefore too broad on its own
# -- _mask_apostrophes resolves the ambiguity below before matching.
_QUOTE_CLOSE = '.!?"\u201c\u201d\u2018\u2019'
_FIRST_PERSON_TAG = re.compile(
    rf"^[^{_QUOTE_CLOSE}]{{0,14}}?\bI\b[^{_QUOTE_CLOSE}]{{0,14}}?\b(?:{_VERBS})\b"
    rf"|^\s*,?\s*(?:{_VERBS})\s+I\b",
    re.IGNORECASE,
)

DEFAULT_WINDOW = 46
DEFAULT_MINIMUM = 3


def _is_word_character(text: str, index: int) -> bool:
    """Whether the character at ``index`` is alphanumeric or an underscore.

    Duplicates quotes.py's own helper of the same name rather than importing
    it: that name is private there, and the two modules stay independently
    testable without narration.py reaching into quotes.py's internals for a
    four-line rule. Keep the two in sync if either changes.
    """
    return 0 <= index < len(text) and (text[index].isalnum() or text[index] == "_")


def _mask_apostrophes(text: str, start: int, end: int) -> str:
    """Return ``text[start:end]`` with word-flanked U+2019 read as apostrophes.

    Mirrors quotes.py's own rule for telling a closing curly single quote
    apart from an apostrophe: a U+2019 with a word character on both sides,
    as in "don't" or "I'd", is punctuation inside a word, not dialogue
    closing. Substituting the ASCII apostrophe for those keeps them out of
    `_QUOTE_CLOSE`, so a contraction between a quote's end and its tag verb
    no longer blocks the first-person scan the way an actual closing quote
    should.
    """
    chunk = text[start:end]
    return "".join(
        "'"
        if char == "\u2019"
        and _is_word_character(text, start + index - 1)
        and _is_word_character(text, start + index + 1)
        else char
        for index, char in enumerate(chunk)
    )


def first_person_tags(
    text: str, quote_ends: Sequence[int], window: int = DEFAULT_WINDOW
) -> int:
    """Count quotes in this chapter tagged with a first-person speech verb."""
    return sum(
        1
        for end in quote_ends
        if _FIRST_PERSON_TAG.match(_mask_apostrophes(text, end, end + window))
    )


def is_first_person(
    text: str, quote_ends: Sequence[int], minimum: int = DEFAULT_MINIMUM
) -> bool:
    """Whether this chapter is narrated in the first person.

    Several tags are required rather than one. A third-person book quoting a
    character who says "I told him" produces the occasional false match, and
    one match must not turn a book first-person.
    """
    return first_person_tags(text, quote_ends) >= minimum
