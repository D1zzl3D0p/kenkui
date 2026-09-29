"""Join native prose lines with compatible type, margins and continuation evidence."""

from __future__ import annotations

import re
from dataclasses import replace
from typing import TYPE_CHECKING

from kenkui._pdf.models import PdfBlock, PdfEdit

if TYPE_CHECKING:
    from kenkui._pdf.models import PdfDocument


_MONOSPACE = re.compile(r"Courier|Mono|Typewriter|(?:SF|CM)TT", re.IGNORECASE)
_MIN_BODY_LINES = 3
_TYPE_TOLERANCE = 0.5
_MIN_LINE_CHARACTERS = 20


def reconstruct_paragraphs(document: PdfDocument) -> PdfDocument:
    """Unwrap body lines while protecting indentation and structured barriers."""
    pages = {p.number: p for p in document.pages}
    lines = {line.id: line for page in document.pages for line in page.lines}
    out: list[PdfBlock] = []
    edits = list(document.edits)
    for block in document.narration:
        previous = out[-1] if out else None
        left = (
            lines.get(previous.sources[-1]) if previous and previous.sources else None
        )
        right = lines.get(block.sources[0]) if block.sources else None
        join = False
        if (
            previous
            and left
            and right
            and previous.role == block.role == "text"
            and previous.origin == block.origin == "native"
            and left.font == right.font
            and abs(left.size - right.size) < _TYPE_TOLERANCE
            and left.size > 0
            and not _MONOSPACE.search(left.font)
        ):
            body = [
                line
                for line in pages[right.page].lines
                if line.font == right.font
                and abs(line.size - right.size) < _TYPE_TOLERANCE
                and len(line.text) > _MIN_LINE_CHARACTERS
            ]
            if len(body) >= _MIN_BODY_LINES:
                margin = min(line.box[0] for line in body)
                left_body = [
                    line
                    for line in pages[left.page].lines
                    if line.font == left.font
                    and abs(line.size - left.size) < _TYPE_TOLERANCE
                    and len(line.text) > _MIN_LINE_CHARACTERS
                ]
                width = max((line.box[2] for line in left_body), default=left.box[2])
                unindented = abs(right.box[0] - margin) < right.size * 0.45
                full = left.box[2] >= width - left.size * 1.5
                same = (
                    left.page == right.page
                    and 0 < right.box[1] - left.box[1] < right.size * 1.8
                )
                cross = (
                    left.page + 1 == right.page
                    and left.box[1] > pages[left.page].height * 0.55
                    and right.box[1] < pages[right.page].height * 0.25
                    and re.search(r"[A-Za-z,\u00ad-]$", previous.text.rstrip())
                    is not None
                )
                quoted = re.match(r'^[\u201c\u2018"\'][A-Z]', block.text) is not None
                join = unindented and full and (same or cross) and not quoted
        if join and previous:
            text = previous.text.rstrip() + " " + block.text.lstrip()
            out[-1] = replace(
                previous, text=text, sources=previous.sources + block.sources
            )
            edits.append(
                PdfEdit(
                    "reconstruct_paragraphs",
                    (previous.id, block.id),
                    previous.text + "\n\n" + block.text,
                    text,
                    "native_continuation",
                )
            )
        else:
            out.append(block)
    return replace(document, narration=tuple(out), edits=tuple(edits))
