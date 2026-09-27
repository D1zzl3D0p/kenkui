"""Optional native PDF extraction without page rasterization or model inference."""

from __future__ import annotations

import statistics
from collections import Counter
from typing import TYPE_CHECKING, Any

from kenkui.errors import ErrorCode, KenkuiError, SourceError

from .models import PdfBlock, PdfCharacter, PdfDocument, PdfLine, PdfPage

if TYPE_CHECKING:
    from pathlib import Path

    from kenkui.cancellation import CancellationToken

    from .options import PdfOptions


_SCAN_IMAGE_COVERAGE = 0.5
_MIN_SCAN_TEXT_CHARACTERS = 200


def _line(raw: dict[str, Any], page: int, index: int) -> PdfLine:
    characters = tuple(
        PdfCharacter(
            str(c["text"]),
            (float(c["x0"]), float(c["top"]), float(c["x1"]), float(c["bottom"])),
            float(c["size"]),
            str(c["fontname"]),
        )
        for c in raw["chars"]
    )
    return PdfLine(
        f"{page}:{index}",
        page,
        str(raw["text"]),
        (float(raw["x0"]), float(raw["top"]), float(raw["x1"]), float(raw["bottom"])),
        statistics.median(c.size for c in characters) if characters else 0,
        Counter(c.font for c in characters).most_common(1)[0][0] if characters else "",
        characters,
    )


def extract_native(  # noqa: C901 - bounded extraction with explicit failure mapping
    path: Path,
    source_hash: str,
    options: PdfOptions,
    cancel: CancellationToken | None,
) -> PdfDocument:
    """Read all native evidence, recording unreadable pages instead of omitting them."""
    try:
        import pdfplumber  # noqa: PLC0415 - optional backend boundary
        from pdfminer.pdfdocument import PDFPasswordIncorrect  # noqa: PLC0415
        from pdfminer.pdfexceptions import PDFException  # noqa: PLC0415
    except ImportError:
        raise SourceError(ErrorCode.PDF_PACKAGE_MISSING) from None
    try:
        with path.open("rb") as handle:
            if b"%PDF-" not in handle.read(1024):
                raise SourceError(ErrorCode.MALFORMED_PDF)
        with pdfplumber.open(path) as pdf:
            if pdf.doc.encryption is not None:
                raise SourceError(ErrorCode.PDF_ENCRYPTED)
            if len(pdf.pages) > options.max_pages:
                raise SourceError(ErrorCode.PDF_LIMIT)
            pages: list[PdfPage] = []
            total = 0
            for number, page in enumerate(pdf.pages, 1):
                if cancel is not None:
                    cancel.raise_if_cancelled()
                raw = page.extract_text_lines(x_tolerance_ratio=0.15)
                lines = tuple(_line(line, number, i) for i, line in enumerate(raw))
                total += sum(len(line.text) for line in lines)
                if total > options.max_characters:
                    raise SourceError(ErrorCode.PDF_LIMIT)
                blocks = tuple(
                    PdfBlock(
                        line.id, number, line.text, sources=(line.id,), box=line.box
                    )
                    for line in lines
                )
                coverage = max(
                    (
                        max(0, min(page.width, im["x1"]) - max(0, im["x0"]))
                        * max(0, min(page.height, im["bottom"]) - max(0, im["top"]))
                        / max(1, page.width * page.height)
                        for im in page.images
                    ),
                    default=0,
                )
                graphics = bool(page.images or page.curves or page.rects or page.lines)
                sparse_scan = (
                    coverage >= _SCAN_IMAGE_COVERAGE
                    and sum(len(line.text) for line in lines)
                    < _MIN_SCAN_TEXT_CHARACTERS
                )
                pages.append(
                    PdfPage(
                        number,
                        float(page.width),
                        float(page.height),
                        blocks,
                        lines,
                        float(coverage),
                        "unresolved"
                        if sparse_scan
                        else "text"
                        if blocks
                        else "unresolved"
                        if graphics
                        else "blank",
                    )
                )
                page.close()
            metadata = pdf.metadata or {}
            return PdfDocument(
                source_hash,
                tuple(pages),
                tuple(b for p in pages for b in p.blocks),
                language=options.language,
                capabilities=("native_lines", "character_geometry"),
                title=str(metadata["Title"]) if metadata.get("Title") else None,
                author=str(metadata["Author"]) if metadata.get("Author") else None,
            )
    except KenkuiError:
        raise
    except PDFPasswordIncorrect:
        raise SourceError(ErrorCode.PDF_ENCRYPTED) from None
    except (OSError, ValueError, TypeError, KeyError, ArithmeticError, PDFException):
        raise SourceError(ErrorCode.MALFORMED_PDF) from None
