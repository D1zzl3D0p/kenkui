"""Find where dialogue is. Deterministic, and no model involved.

Extraction answers only *where* speech sits in a chapter; *who* said it needs
a model and happens in attribution. Keeping the two apart means this half is
exactly testable and never depends on a provider being reachable.

The output partitions the chapter: every character of the source belongs to
exactly one span, in order. Segmentation reconstructs chapter text by
concatenating span chunks, so a gap here would silently drop audio and an
overlap would silently duplicate it.

Known limitation: the single quote is not treated as a delimiter, because in
English prose it is far more often an apostrophe than a quotation mark, and
mistaking possessives for speech would turn most narration into dialogue. The
cost is that single-quoted British dialogue goes undetected and is narrated.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

# Balanced pairs only. A straight quote is its own partner; curly quotes open
# and close distinctly, which is what lets nested quotes inside a curly pair be
# left alone. Curly singles carry British dialogue and are safe to include
# because the curly apostrophe is the same character and is distinguished by
# the word-boundary rule below; the straight single quote has no such tell and
# is therefore not a delimiter at all.
_PAIRS: tuple[tuple[str, str], ...] = (
    ('"', '"'),
    ("\u201c", "\u201d"),
    ("\u2018", "\u2019"),
)
_OPENERS = dict(_PAIRS)

# Verbs that label a term rather than introduce speech. A quoted run right
# after one of these is a scare quote: it is written but never said, and
# voicing it as dialogue puts an aside in a character's mouth.
_LABEL_BEFORE = re.compile(
    r"\b(?:call|calls|called|calling|known as|so-called|titled|dubbed"
    r"|nicknamed|termed|label(?:s|ed|led|ling)?)\s+(?:an?\s+|the\s+)?$",
    re.IGNORECASE,
)
# How far back to look for a label verb. Long enough for "what they would have
# called a", short enough not to reach the previous sentence.
_LABEL_WINDOW = 60
# Quoted acronyms are glosses, not speech.
_ACRONYM = re.compile(r"^[A-Z]{2,6}$")


@dataclass(frozen=True, slots=True)
class TextSpan:
    """One contiguous run of a chapter, marked as dialogue or narration."""

    chapter_id: str
    start: int
    end: int
    is_dialogue: bool


def _is_word_character(text: str, index: int) -> bool:
    return 0 <= index < len(text) and (text[index].isalnum() or text[index] == "_")


def _is_scare_quote(text: str, open_index: int, content: str) -> bool:
    """Return whether a quoted run is a label rather than something spoken."""
    preceding = text[max(0, open_index - _LABEL_WINDOW) : open_index]
    return bool(_LABEL_BEFORE.search(preceding)) or bool(_ACRONYM.match(content))


def _closing_index(text: str, closer: str, start: int) -> int:
    """Find the closer, skipping a curly apostrophe used inside a word."""
    index = text.find(closer, start)
    while index != -1:
        # don't / Darcy's: a closer with a word character on both sides is an
        # apostrophe, so keep looking for the real one.
        if closer == "\u2019" and _is_word_character(text, index - 1) and (
            _is_word_character(text, index + 1)
        ):
            index = text.find(closer, index + 1)
            continue
        return index
    return -1


def _quote_ranges(text: str) -> list[tuple[int, int]]:
    """Return half-open ranges covering each spoken quotation, in order."""
    ranges: list[tuple[int, int]] = []
    index = 0
    while index < len(text):
        closer = _OPENERS.get(text[index])
        if closer is None:
            index += 1
            continue
        # A delimiter with letters on both sides is an apostrophe or an inch
        # mark, not an opener. Treating one as an opener would turn most prose
        # into dialogue.
        if _is_word_character(text, index - 1) and _is_word_character(text, index + 1):
            index += 1
            continue
        end = _closing_index(text, closer, index + 1)
        if end == -1:
            # Unbalanced. Leaving it as narration keeps a stray delimiter from
            # swallowing the remainder of the chapter.
            index += 1
            continue
        if _is_scare_quote(text, index, text[index + 1 : end]):
            # Left as narration, marks and all. Because this module partitions
            # rather than rewrites, the quote characters survive untouched and
            # no separate stripping pass is needed.
            index = end + 1
            continue
        ranges.append((index, end + 1))
        index = end + 1
    return ranges


def extract_spans(chapter_id: str, text: str) -> tuple[TextSpan, ...]:
    """Partition one chapter into alternating narration and dialogue spans."""
    if not text:
        return ()
    spans: list[TextSpan] = []
    cursor = 0
    for start, end in _quote_ranges(text):
        if start > cursor:
            spans.append(TextSpan(chapter_id, cursor, start, is_dialogue=False))
        spans.append(TextSpan(chapter_id, start, end, is_dialogue=True))
        cursor = end
    if cursor < len(text):
        spans.append(TextSpan(chapter_id, cursor, len(text), is_dialogue=False))
    return tuple(spans)
