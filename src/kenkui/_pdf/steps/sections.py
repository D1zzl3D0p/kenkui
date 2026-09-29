"""Conservative omission of explicitly structured numbered endnote sections."""

from __future__ import annotations

import re
from dataclasses import replace
from typing import TYPE_CHECKING

from kenkui._pdf.models import PdfEdit

if TYPE_CHECKING:
    from kenkui._pdf.models import PdfDocument

_HEADING = re.compile(r"(?:end\s*notes|foot\s*notes|notes)", re.IGNORECASE)
_ENTRY = re.compile(r"^(\d{1,3})[.)]\s+\S")
_MIN_ENTRIES = 2


def remove_note_sections(document: PdfDocument) -> PdfDocument:
    """Omit complete numbered entries under Notes, stopping at the next heading.

    Missing numbers, unnumbered continuations, or ordinary prose make the entire
    candidate section ambiguous; leave it intact. Reference removal across
    chapter boundaries is intentionally not inferred from baseline digits.
    """
    removed: set[str] = set()
    blocks = document.narration
    for index, block in enumerate(blocks):
        if block.role != "heading" or _HEADING.fullmatch(block.text.strip()) is None:
            continue
        end = next(
            (i for i in range(index + 1, len(blocks)) if blocks[i].role == "heading"),
            len(blocks),
        )
        entries = blocks[index + 1 : end]
        matches = [_ENTRY.match(entry.text) for entry in entries]
        if (
            len(entries) >= _MIN_ENTRIES
            and all(entry.role in ("text", "note") for entry in entries)
            and [int(match[1]) if match else None for match in matches]
            == list(range(1, len(entries) + 1))
            and all(entry.text.rstrip().endswith((".", "!", "?")) for entry in entries)
        ):
            removed.update(item.id for item in blocks[index:end])
    edits = tuple(
        PdfEdit(
            "remove_note_sections",
            (block.id,),
            block.text,
            "",
            "explicit_consecutive_numbered_notes",
        )
        for block in blocks
        if block.id in removed
    )
    return replace(
        document,
        narration=tuple(b for b in blocks if b.id not in removed),
        edits=(*document.edits, *edits),
    )
