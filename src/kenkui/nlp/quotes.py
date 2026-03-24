"""Stage 1: Regex-based dialogue quote extraction.

The LLM's job is *not* to find quotes — it is to classify who said them.
This module finds every quoted span deterministically so the LLM works from
a fixed, numbered list rather than free-form text.

Supported quote styles
----------------------
* Straight double quotes:     "..."
* Curly / typographic quotes: "..."  (U+201C / U+201D)
* Nested variants are handled by the DOTALL flag on a non-greedy match.
"""

from __future__ import annotations

import re

from .models import Quote

# Match content wrapped in straight or curly double quotes.
# Non-greedy so nested/adjacent quotes don't collapse into one span.
_QUOTE_RE = re.compile(r'["\u201c](.+?)["\u201d]', re.DOTALL)


def extract_quotes(paragraphs: list[str]) -> list[Quote]:
    """Return all dialogue quotes found across *paragraphs*.

    Each ``Quote`` carries:
    - ``id``          — stable integer used as the attribution key
    - ``text``        — full matched text including surrounding marks
    - ``para_index``  — which paragraph (0-based) the quote lives in
    - ``char_offset`` — byte offset within the *joined* chapter text
                        (paragraphs joined by ``"\\n\\n"``)

    The joined-text offset lets downstream code map quotes into overlapping
    chunks without re-running the regex.
    """
    quotes: list[Quote] = []
    qid = 0
    global_offset = 0

    for para_idx, para in enumerate(paragraphs):
        for m in _QUOTE_RE.finditer(para):
            quotes.append(
                Quote(
                    id=qid,
                    text=m.group(0),
                    para_index=para_idx,
                    char_offset=global_offset + m.start(),
                )
            )
            qid += 1
        # +2 for the "\n\n" separator used when joining paragraphs
        global_offset += len(para) + 2

    return quotes
