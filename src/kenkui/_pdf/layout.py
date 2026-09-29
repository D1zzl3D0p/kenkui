"""Combine layout structure with native evidence without losing native prose."""

from __future__ import annotations

import re
from dataclasses import replace
from typing import TYPE_CHECKING

from kenkui.errors import ErrorCode, SourceError

from .models import PdfIssue

if TYPE_CHECKING:
    from .models import PdfBlock, PdfDocument, PdfLine, PdfPage


def _inside(line: PdfLine, block: PdfBlock) -> bool:
    if block.box is None:
        return False
    left, top, right, bottom = block.box
    x = (line.box[0] + line.box[2]) / 2
    y = (line.box[1] + line.box[3]) / 2
    return left <= x <= right and top <= y <= bottom


def _native_projection(
    page: PdfPage,
    blocks: tuple[PdfBlock, ...],
) -> tuple[PdfBlock, ...] | None:
    """Accept layout order only when it accounts for every native line once."""
    projected = []
    covered: set[str] = set()
    for block in blocks:
        lines = tuple(line for line in page.lines if _inside(line, block))
        if not lines:
            # Empty visual records retain classification and location.
            if block.role in ("picture", "table", "formula") and not block.text.strip():
                projected.append(block)
                continue
            if block.text.strip():
                return None
            continue
        if any(line.id in covered for line in lines):
            return None
        text = " ".join(line.text.strip() for line in lines)
        if block.role not in ("picture", "table", "formula") and (
            re.sub(r"\s+", "", text) != re.sub(r"\s+", "", block.text)
        ):
            return None
        projected.append(
            replace(block, text=text, sources=tuple(line.id for line in lines))
        )
        covered.update(line.id for line in lines)
    if covered != {line.id for line in page.lines}:
        return None
    return tuple(projected)


def merge_layout(
    native: PdfDocument,
    blocks: tuple[PdfBlock, ...],
    ocr_pages: frozenset[int],
) -> PdfDocument:
    """Keep original native/layout archives and a separately checked projection."""
    pages = []
    narration: list[PdfBlock] = []
    issues = list(native.issues)
    inventory = {page.number for page in native.pages}
    if (
        any(block.page not in inventory for block in blocks)
        or not ocr_pages <= inventory
    ):
        raise SourceError(ErrorCode.INVALID_PDF_OUTPUT)
    by_page: dict[int, list[PdfBlock]] = {}
    for block in blocks:
        by_page.setdefault(block.page, []).append(block)
    projection: tuple[PdfBlock, ...] | None
    for page in native.pages:
        layout = tuple(by_page.get(page.number, ()))
        if page.number in ocr_pages:
            prose = [
                b
                for b in layout
                if b.role in ("text", "heading", "note", "verse", "code")
                and b.text.strip()
            ]
            if not prose:
                raise SourceError(ErrorCode.PDF_EXTRACTION_INCOMPLETE)
            projection = layout
        else:
            projection = _native_projection(page, layout)
            if projection is None:
                projection = page.blocks
                issues.append(
                    PdfIssue(
                        "layout_native_mismatch",
                        "Layout did not account for native text exactly; "
                        "original page retained.",
                        pages=(page.number,),
                    )
                )
            if page.disposition == "unresolved":
                raise SourceError(ErrorCode.PDF_EXTRACTION_INCOMPLETE)
        narration.extend(projection)
        pages.append(
            replace(
                page,
                layout_blocks=layout,
                disposition="text" if projection else page.disposition,
            )
        )
    return replace(
        native,
        pages=tuple(pages),
        narration=tuple(narration),
        issues=tuple(issues),
        capabilities=(*native.capabilities, "layout_blocks"),
    )
