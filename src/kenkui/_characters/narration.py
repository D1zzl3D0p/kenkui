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
_QUOTE_CLOSE = '.!?"\u201c\u201d\u2018\u2019'
_FIRST_PERSON_TAG = re.compile(
    rf"^[^{_QUOTE_CLOSE}]{{0,14}}?\bI\b[^{_QUOTE_CLOSE}]{{0,14}}?\b(?:{_VERBS})\b"
    rf"|^\s*,?\s*(?:{_VERBS})\s+I\b",
    re.IGNORECASE,
)

DEFAULT_WINDOW = 46
DEFAULT_MINIMUM = 3


def first_person_tags(
    text: str, quote_ends: Sequence[int], window: int = DEFAULT_WINDOW
) -> int:
    """Count quotes in this chapter tagged with a first-person speech verb."""
    return sum(
        1 for end in quote_ends if _FIRST_PERSON_TAG.match(text[end : end + window])
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
