"""Stage 1: Regex-based dialogue quote and italic span extraction.

The LLM's job is *not* to find quotes — it is to classify who said them.
This module finds every quoted span and italicised inner-monologue span
deterministically so the LLM works from a fixed, numbered list rather than
free-form text.  ``strip_scare_quotes`` is a pre-processing step that removes
quote marks from scare quotes and acronyms before extraction runs.

Supported quote styles
----------------------
* Straight double quotes:     "..."
* Curly / typographic quotes: "..."  (U+201C / U+201D)
* Nested variants are handled by the DOTALL flag on a non-greedy match.

Italic spans
------------
* Readers wrap <em>/<i> content with STX (\\x02) / ETX (\\x03) markers.
* _ITALIC_RE detects these spans and produces Quote objects with kind="italic".
* The markers are stripped from Quote.text so downstream TTS receives clean text.
"""

from __future__ import annotations

import re

from .models import Quote

# Match dialogue wrapped in straight/curly double quotes or curly single quotes.
# Two alternates joined by |:
#   1. Double-quote path: " or " ... " or "
#   2. Curly-single-quote path: ' ... ' with apostrophe-awareness
#      (?<!\w) / (?!\w) prevent word-internal apostrophes (it's, they'd) from matching.
#      \u2019(?=\w) inside the group allows an apostrophe mid-word.
_QUOTE_RE = re.compile(
    r'["\u201c](.+?)["\u201d]'
    r'|'
    r'(?<!\w)\u2018((?:[^\u2019]|\u2019(?=\w))+)\u2019(?!\w)',
    re.DOTALL,
)

# Match italic spans inserted by the EPUB/MOBI readers using STX/ETX markers.
_ITALIC_RE = re.compile(r'\x02(.+?)\x03', re.DOTALL)

# Matches proper-noun emphasis: every word starts with a capital letter,
# optional leading article (The/A/An). Covers single-word ("Archimedes")
# and multi-word ("The Pax", "Morning Star") ship/place/title names.
# Spans with lowercase function words, sentence punctuation, or numbers
# are intentionally NOT matched — those pass through as potential inner monologue.
_DECORATIVE_ITALIC_RE = re.compile(
    r'^(?:(?:The|A|An)\s+)?[A-Z][a-zA-Z\'\-]+(?:\s+[A-Z][a-zA-Z\'\-]+)*$'
)


def _is_decorative_italic(span: str) -> bool:
    """Return True if *span* is a proper-noun emphasis rather than inner monologue.

    Matches spans where every word starts with a capital letter (optional leading
    article). Does NOT match spans with lowercase words, sentence punctuation, or
    numbers — those pass through as potential inner monologue.
    """
    return bool(_DECORATIVE_ITALIC_RE.match(span.strip()))

# Explicit labeling verbs that immediately precede a scare-quoted term.
# Includes both bare ("call") and inflected ("called") forms so patterns like
# "what you'd call a X" are handled alongside "what they called X".
# An optional article/determiner (a, an, the) may follow the verb.
_LABEL_BEFORE_RE = re.compile(
    r'\b(?:call|called|known as|so-called|titled|dubbed|nicknamed|termed|label(?:ed|led)?|labelled)\s+(?:a\s+|an\s+|the\s+)?$',
    re.IGNORECASE,
)

# All-caps acronyms (UNESCO, NATO, DNA, etc.) — 2–6 uppercase letters.
_ACRONYM_RE = re.compile(r'^[A-Z]{2,6}$')

# Opening and closing quote characters handled by this module.
_OPEN_QUOTES = ('"', '\u201c')
_CLOSE_QUOTES = ('"', '\u201d')

# Map each opening quote character to its preferred closing counterpart.
_CLOSE_FOR_OPEN = {'"': '"', '\u201c': '\u201d'}


def strip_scare_quotes(paragraphs: list[str]) -> list[str]:
    """Remove quote marks from scare quotes while preserving the quoted content.

    A scare quote is identified by either:
    - An explicit labeling verb (called, known as, so-called, …) immediately
      before the opening quote mark, OR
    - The quoted content being an all-caps acronym (UNESCO, NATO, DNA, etc.)

    Italic spans (STX/ETX markers) are not touched. Outer dialogue quote marks
    whose preceding context does not match a label verb are also left intact.

    The function scans every opening-quote character in the paragraph so that
    scare quotes nested inside a dialogue span are also detected correctly.

    Returns a list of the same length as *paragraphs* with scare-quote marks
    removed (content preserved).
    """
    result: list[str] = []
    for para in paragraphs:
        # positions (open_pos, close_pos) of quote-mark pairs to strip,
        # collected in document order and applied right-to-left.
        to_strip: list[tuple[int, int]] = []
        claimed_close_positions: set[int] = set()

        for i, ch in enumerate(para):
            if ch not in _OPEN_QUOTES:
                continue
            # Determine matching close character: prefer the typographic pair,
            # but also accept the straight-quote version as a fallback.
            preferred_close = _CLOSE_FOR_OPEN[ch]
            # Find the nearest closing quote after i: first try the preferred
            # close character, then fall back to any recognised close quote.
            close_pos = -1
            for j in range(i + 1, len(para)):
                if para[j] == preferred_close:
                    close_pos = j
                    break
            if close_pos == -1:
                for j in range(i + 1, len(para)):
                    if para[j] in _CLOSE_QUOTES:
                        close_pos = j
                        break
            if close_pos == -1:
                continue
            # C1: skip if this close position was already claimed by a prior span.
            if close_pos in claimed_close_positions:
                continue

            content = para[i + 1 : close_pos]

            # Check label-verb heuristic on the 60 chars preceding the open mark.
            preceding = para[max(0, i - 60) : i]
            is_label = bool(_LABEL_BEFORE_RE.search(preceding))
            is_acronym = bool(_ACRONYM_RE.match(content))

            if is_label or is_acronym:
                to_strip.append((i, close_pos))
                claimed_close_positions.add(close_pos)

        # Apply right-to-left so earlier offsets stay valid.
        for open_pos, close_pos in reversed(to_strip):
            # Remove close mark first (higher index), then open mark.
            para = para[:close_pos] + para[close_pos + 1:]
            para = para[:open_pos] + para[open_pos + 1:]

        result.append(para)
    return result


def extract_quotes(paragraphs: list[str]) -> list[Quote]:
    """Return all dialogue quotes and italic spans found across *paragraphs*.

    Each ``Quote`` carries:
    - ``id``          — stable integer used as the attribution key
    - ``text``        — full quoted text (incl. marks) for dialogue;
                        plain content (markers stripped) for italic spans
    - ``para_index``  — which paragraph (0-based) the quote lives in
    - ``char_offset`` — byte offset within the *joined* chapter text
                        (paragraphs joined by ``"\\n\\n"``)
    - ``kind``        — ``"dialogue"`` or ``"italic"``

    The joined-text offset lets downstream code map quotes into overlapping
    chunks without re-running the regex.
    """
    quotes: list[Quote] = []
    qid = 0
    global_offset = 0

    for para_idx, para in enumerate(paragraphs):
        # Collect all matches from both patterns, tagged by kind.
        matches: list[tuple[int, re.Match, str]] = []
        for m in _QUOTE_RE.finditer(para):
            matches.append((m.start(), m, "dialogue"))
        for m in _ITALIC_RE.finditer(para):
            matches.append((m.start(), m, "italic"))
        # Process in document order so IDs are assigned left-to-right.
        matches.sort(key=lambda t: t[0])

        last_end = 0
        for _, m, kind in matches:
            if m.start() < last_end:
                continue
            last_end = m.end()
            text = m.group(0) if kind == "dialogue" else m.group(1)
            if kind == "italic" and _is_decorative_italic(text):
                continue
            quotes.append(
                Quote(
                    id=qid,
                    text=text,
                    para_index=para_idx,
                    char_offset=global_offset + m.start(),
                    kind=kind,
                )
            )
            qid += 1

        # +2 for the "\n\n" separator used when joining paragraphs
        global_offset += len(para) + 2

    return quotes
