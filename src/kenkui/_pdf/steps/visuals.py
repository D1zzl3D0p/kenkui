"""Omit classified visual material while protecting uncorroborated captions."""

from __future__ import annotations

import re
from dataclasses import replace
from typing import TYPE_CHECKING

from kenkui._pdf.models import PdfEdit

if TYPE_CHECKING:
    from kenkui._pdf.models import PdfBlock, PdfDocument

_VISUAL = frozenset({"picture", "table", "formula"})
_CAPTION = re.compile(r"^(?:fig(?:ure)?\.?|table|plate)\s+\d+[.:\s]", re.IGNORECASE)
_MAX_CAPTION_CHARACTERS = 300


def _attached(caption: PdfBlock, visual: PdfBlock) -> bool:
    if caption.page != visual.page or caption.box is None or visual.box is None:
        return False
    left, top, right, bottom = caption.box
    other_left, other_top, other_right, other_bottom = visual.box
    gap = min(abs(top - other_bottom), abs(other_top - bottom))
    return (
        max(left, other_left) < min(right, other_right)
        and gap <= max(1, bottom - top) * 2
    )


def omit_visual_material(document: PdfDocument) -> PdfDocument:
    """Omit layout-classified visuals and short explicitly attached captions."""
    # Original layout records keep the same evidence on repeat runs.
    visuals = [
        b for page in document.pages for b in page.layout_blocks if b.role in _VISUAL
    ]
    visuals.extend(
        b for b in document.narration if b.origin == "layout" and b.role in _VISUAL
    )
    removed = {
        b.id
        for b in document.narration
        if b.origin == "layout"
        and (
            b.role in _VISUAL
            or (
                b.role == "caption"
                and len(b.text) <= _MAX_CAPTION_CHARACTERS
                and _CAPTION.match(b.text) is not None
                and any(_attached(b, visual) for visual in visuals)
            )
        )
    }
    edits = tuple(
        PdfEdit(
            "omit_visual_material",
            (b.id,),
            b.text,
            "",
            "classified_visual_or_attached_caption",
        )
        for b in document.narration
        if b.id in removed
    )
    return replace(
        document,
        narration=tuple(b for b in document.narration if b.id not in removed),
        edits=(*document.edits, *edits),
    )
