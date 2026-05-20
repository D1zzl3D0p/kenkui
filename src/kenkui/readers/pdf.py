"""
PDF ebook reader implementation.

Provides EbookReader interface for PDF files using pymupdf (fitz).
"""

from __future__ import annotations

import logging
from pathlib import Path

import fitz

from ..chapter_classifier import ChapterClassifier
from ..models import Chapter
from . import EbookMetadata, EbookReader, Registry, TocEntry
from ._pdf_extract import PdfTextExtractor

logger = logging.getLogger(__name__)


@Registry.register
class PdfReader(EbookReader):
    """PDF ebook reader using pymupdf (fitz)."""

    SUPPORTED_EXTENSIONS = {".pdf"}

    def __init__(self, filepath: Path, verbose: bool = False) -> None:
        super().__init__(filepath, verbose)
        try:
            self._doc = fitz.open(str(filepath))
        except Exception as e:
            raise ValueError(f"Could not open PDF {filepath.name}: {e}") from e

        if self._doc.is_encrypted:
            raise ValueError(f"PDF is encrypted/password-protected: {filepath.name}")

        self._extractor = PdfTextExtractor(self._doc, verbose=verbose)

    def get_metadata(self) -> EbookMetadata:
        meta = self._doc.metadata or {}
        title = (meta.get("title") or "").strip() or self.filepath.stem
        author = (meta.get("author") or "").strip() or None
        return EbookMetadata(title=title, author=author)

    def get_toc(self) -> list[TocEntry]:
        return self._extractor.extract_toc()

    def get_chapters(self, min_text_len: int = 50) -> list[Chapter]:
        toc = self.get_toc()
        chapters: list[Chapter] = []

        for toc_index, entry in enumerate(toc):
            start, end = _parse_page_href(entry.href)
            paragraphs = self._extractor.extract_text_for_pages(start, end)
            paragraphs = [p for p in paragraphs if len(p) >= min_text_len]

            if not paragraphs:
                continue

            word_count = sum(len(p.split()) for p in paragraphs)
            tags = ChapterClassifier.classify(entry.title, word_count=word_count)

            chapters.append(
                Chapter(
                    index=len(chapters),
                    title=entry.title,
                    paragraphs=paragraphs,
                    tags=tags,
                    toc_index=toc_index,
                )
            )

        if not chapters:
            raise ValueError(
                f"{self.filepath.name}: no extractable text "
                "(possibly scanned — OCR not supported)"
            )

        return chapters

    def get_cover(self) -> tuple[bytes | None, str | None]:
        return self._extractor.render_cover_page()


def _parse_page_href(href: str) -> tuple[int, int]:
    """Parse 'page:<start>-<end>' → (start, end) as ints."""
    _, page_range = href.split(":", 1)
    start_s, end_s = page_range.split("-", 1)
    return int(start_s), int(end_s)
