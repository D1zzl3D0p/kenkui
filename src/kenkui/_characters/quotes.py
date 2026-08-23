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

from dataclasses import dataclass

# Balanced pairs only. A straight quote is its own partner; curly quotes open
# and close distinctly, which is what lets nested straight quotes inside a
# curly pair be left alone.
_PAIRS: tuple[tuple[str, str], ...] = (
    ('"', '"'),
    ("“", "”"),
)
_OPENERS = dict(_PAIRS)


@dataclass(frozen=True, slots=True)
class TextSpan:
    """One contiguous run of a chapter, marked as dialogue or narration."""

    chapter_id: str
    start: int
    end: int
    is_dialogue: bool


def _is_word_character(text: str, index: int) -> bool:
    return 0 <= index < len(text) and (text[index].isalnum() or text[index] == "_")


def _quote_ranges(text: str) -> list[tuple[int, int]]:
    """Return half-open ranges covering each balanced quotation, in order."""
    ranges: list[tuple[int, int]] = []
    index = 0
    while index < len(text):
        closer = _OPENERS.get(text[index])
        if closer is None:
            index += 1
            continue
        # An apostrophe between word characters is a possessive or a
        # contraction. Treating one as an opener would turn most prose into
        # dialogue, so a delimiter with letters on both sides is skipped.
        if _is_word_character(text, index - 1) and _is_word_character(text, index + 1):
            index += 1
            continue
        end = text.find(closer, index + 1)
        if end == -1:
            # Unbalanced. Leaving it as narration keeps a stray delimiter from
            # swallowing the remainder of the chapter.
            index += 1
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
