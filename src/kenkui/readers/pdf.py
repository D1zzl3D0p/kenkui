"""
PDF ebook reader implementation.

Provides EbookReader interface for PDF files using pymupdf (fitz).
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path

import fitz

from ..chapter_classifier import ChapterClassifier
from ..models import Chapter
from . import EbookMetadata, EbookReader, Registry, TocEntry
from ._pdf_extract import PdfTextExtractor

logger = logging.getLogger(__name__)

_CODE_MARKER_RE = re.compile(r"(?i)^\s*(def|class|import|from|return|if\s+__name__|for\s+|while\s+|try:|except\s+)")
_NOTE_MARKER_RE = re.compile(r"(?i)^\s*(note|notes|footnote|endnote)\s*[:.\-]")
_ASIDE_MARKER_RE = re.compile(r"(?i)^\s*(aside|sidebar|callout|box)\s*[:.\-]")


@dataclass(frozen=True)
class PdfTranscriptSection:
    title: str
    start_page: int
    end_page: int
    raw_paragraphs: list[str]
    filtered_paragraphs: list[str]


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
        self._pdf_cleanup_options: dict[str, bool] = {}
        self._transcript_sections: list[PdfTranscriptSection] = []

    def configure_pdf_extraction(self, options: dict[str, bool | float] | None = None) -> None:
        self._pdf_cleanup_options = dict(options or {})
        self._extractor.configure(self._pdf_cleanup_options)

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
        sections: list[PdfTranscriptSection] = []

        for toc_index, entry in enumerate(toc):
            start, end = _parse_page_href(entry.href)
            raw_paragraphs = self._extractor.extract_text_for_pages(start, end)
            cleaned_paragraphs = self._apply_pdf_cleanup(raw_paragraphs)
            paragraphs = [p for p in cleaned_paragraphs if len(p) >= min_text_len]
            sections.append(
                PdfTranscriptSection(
                    title=entry.title,
                    start_page=start,
                    end_page=end,
                    raw_paragraphs=raw_paragraphs,
                    filtered_paragraphs=paragraphs,
                )
            )

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

        self._transcript_sections = sections

        if not chapters:
            raise ValueError(
                f"{self.filepath.name}: no extractable text "
                "(possibly scanned — OCR not supported)"
            )

        return chapters

    def get_transcript_sections(self) -> list[PdfTranscriptSection]:
        return list(self._transcript_sections)

    def get_cover(self) -> tuple[bytes | None, str | None]:
        return self._extractor.render_cover_page()

    def _apply_pdf_cleanup(self, paragraphs: list[str]) -> list[str]:
        if not paragraphs:
            return paragraphs

        drop_code = self._pdf_cleanup_options.get("drop_code_blocks", False)
        drop_notes = self._pdf_cleanup_options.get("drop_notes", False)
        drop_asides = self._pdf_cleanup_options.get("drop_asides", False)
        cleaned: list[str] = []

        for paragraph in paragraphs:
            text = paragraph.strip()
            if not text:
                continue
            if drop_code and self._looks_like_code_block(text):
                continue
            if drop_notes and self._looks_like_note_block(text):
                continue
            if drop_asides and self._looks_like_aside_block(text):
                continue
            cleaned.append(paragraph)
        return cleaned

    @staticmethod
    def _looks_like_code_block(text: str) -> bool:
        lines = [ln.rstrip() for ln in text.splitlines() if ln.strip()]
        if not lines:
            return False
        if len(lines) >= 2:
            indented = sum(1 for ln in lines if ln.startswith((" ", "\t")))
            if indented >= max(1, len(lines) // 2):
                return True
        if _CODE_MARKER_RE.search(text):
            return True
        code_tokens = ("{", "}", ";", "=>", "==", "!=", ":=", "::")
        return len(text) < 500 and sum(token in text for token in code_tokens) >= 2

    @staticmethod
    def _looks_like_note_block(text: str) -> bool:
        return bool(_NOTE_MARKER_RE.search(text))

    @staticmethod
    def _looks_like_aside_block(text: str) -> bool:
        return bool(_ASIDE_MARKER_RE.search(text))


def _parse_page_href(href: str) -> tuple[int, int]:
    """Parse 'page:<start>-<end>' → (start, end) as ints."""
    _, page_range = href.split(":", 1)
    start_s, end_s = page_range.split("-", 1)
    return int(start_s), int(end_s)
