"""Adapt prepared PDF narration to the existing normalized book representation."""

from __future__ import annotations

from typing import TYPE_CHECKING

from kenkui._domain.text import normalize_text
from kenkui.errors import ErrorCode, SourceError
from kenkui.inspection import BookInspection, BookMetadata, ChapterInspection

if TYPE_CHECKING:
    from .models import PdfDocument


def inspect_pdf_document(document: PdfDocument) -> BookInspection:
    """Create one logical chapter until heading-based segmentation is supported."""
    if any(p.disposition == "unresolved" for p in document.pages) or any(
        issue.severity == "error" for issue in document.issues
    ):
        raise SourceError(ErrorCode.PDF_EXTRACTION_INCOMPLETE)
    text = normalize_text("\n\n".join(b.text for b in document.narration))
    if not text:
        raise SourceError(ErrorCode.EMPTY_SPEECH)
    return BookInspection(
        BookMetadata(document.title, document.author, cover_available=False),
        (ChapterInspection("pdf-body", 0, document.title or "Text", len(text), text),),
    )
