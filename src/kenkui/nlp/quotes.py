"""Stage 1: Regex-based dialogue quote and italic span extraction.

The LLM's job is *not* to find quotes — it is to classify who said them.
This module finds every quoted span and italicised inner-monologue span
deterministically so the LLM works from a fixed, numbered list rather than
free-form text.

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

# Match content wrapped in straight or curly double quotes.
# Non-greedy so nested/adjacent quotes don't collapse into one span.
_QUOTE_RE = re.compile(r'["\u201c](.+?)["\u201d]', re.DOTALL)

# Match italic spans inserted by the EPUB/MOBI readers using STX/ETX markers.
_ITALIC_RE = re.compile(r'\x02(.+?)\x03', re.DOTALL)


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
