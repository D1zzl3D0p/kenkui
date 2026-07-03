"""
PDF ebook reader implementation.

When docling is installed (pip install kenkui[pdf-enhanced]), PDFs are
converted to EPUB first via PdfToEpubConverter, and all subsequent parsing
is delegated to EpubReader.  When docling is not installed the existing
pymupdf (fitz) extraction path is used as a fallback.
"""

from __future__ import annotations

import logging
import re
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from ..chapter_classifier import ChapterClassifier
from ..models import Chapter
from . import EbookMetadata, EbookReader, Registry, TocEntry

if TYPE_CHECKING:
    from .epub import EpubReader as _EpubReader

logger = logging.getLogger(__name__)

try:
    import docling  # noqa: F401
    _DOCLING_AVAILABLE = True
except ImportError:
    _DOCLING_AVAILABLE = False

# ---------------------------------------------------------------------------
# Fallback path — keep existing types for pymupdf route
# ---------------------------------------------------------------------------

_CODE_MARKER_RE = re.compile(
    r"(?i)^\s*(def|class|import|from|return|if\s+__name__|for\s+|while\s+|try:|except\s+)"
)
_NOTE_MARKER_RE = re.compile(r"(?i)^\s*(note|notes|footnote|endnote)\s*[:.\-]")
_ASIDE_MARKER_RE = re.compile(r"(?i)^\s*(aside|sidebar|callout|box)\s*[:.\-]")


@dataclass(frozen=True)
class PdfTranscriptSection:
    title: str
    start_page: int
    end_page: int
    raw_paragraphs: list[str]
    filtered_paragraphs: list[str]


# ---------------------------------------------------------------------------
# PdfReader
# ---------------------------------------------------------------------------

@Registry.register
class PdfReader(EbookReader):
    """PDF ebook reader.

    Uses docling for layout-aware extraction when installed; falls back to
    pymupdf (fitz) otherwise.
    """

    SUPPORTED_EXTENSIONS = {".pdf"}

    def __init__(self, filepath: Path, verbose: bool = False) -> None:
        super().__init__(filepath, verbose)

        if _DOCLING_AVAILABLE:
            # Docling path: lazy conversion triggered on first get_* call
            self._force_ocr: bool = False
            self._epub_reader = None
        else:
            logger.warning(
                "docling not installed — using pymupdf fallback for %s. "
                "For better results: pip install kenkui[pdf-enhanced]",
                filepath.name,
            )
            self._init_pymupdf(filepath, verbose)

    # ── Docling path helpers ──────────────────────────────────────────────

    def _init_docling_reader(self) -> None:
        # Import EpubReader here rather than at module top to avoid a circular import
        # (readers/__init__.py imports both epub and pdf modules).
        # Import PdfToEpubConverter lazily since _pdf_to_epub imports docling at call time.
        from ._pdf_to_epub import PdfToEpubConverter
        from .epub import EpubReader

        epub_path = PdfToEpubConverter().convert(self.filepath, force_ocr=self._force_ocr)
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            self._epub_reader = EpubReader(epub_path, verbose=self.verbose)

    def _ensure_epub_reader(self) -> _EpubReader:
        if self._epub_reader is None:
            self._init_docling_reader()
        return self._epub_reader

    # ── pymupdf fallback helpers ──────────────────────────────────────────

    def _init_pymupdf(self, filepath: Path, verbose: bool) -> None:
        import fitz

        from ._pdf_extract import PdfTextExtractor

        try:
            self._doc = fitz.open(str(filepath))
        except Exception as e:
            raise ValueError(f"Could not open PDF {filepath.name}: {e}") from e

        if self._doc.is_encrypted:
            raise ValueError(f"PDF is encrypted/password-protected: {filepath.name}")

        self._extractor = PdfTextExtractor(self._doc, verbose=verbose)
        self._pdf_cleanup_options: dict[str, bool] = {}
        self._transcript_sections: list[PdfTranscriptSection] = []

    # ── Public interface ──────────────────────────────────────────────────

    def configure_pdf_extraction(self, options: dict[str, bool | float] | None = None) -> None:
        opts = dict(options or {})
        if _DOCLING_AVAILABLE:
            self._force_ocr = bool(opts.get("force_ocr", False))
            ignored = {k: v for k, v in opts.items() if k != "force_ocr"}
            if ignored:
                logger.debug(
                    "configure_pdf_extraction: ignoring options on docling path (handled at conversion): %s",
                    list(ignored.keys()),
                )
        else:
            self._pdf_cleanup_options = opts
            self._extractor.configure(opts)

    def get_metadata(self) -> EbookMetadata:
        if _DOCLING_AVAILABLE:
            return self._ensure_epub_reader().get_metadata()
        meta = self._doc.metadata or {}
        title = (meta.get("title") or "").strip() or self.filepath.stem
        author = (meta.get("author") or "").strip() or None
        return EbookMetadata(title=title, author=author)

    def get_toc(self) -> list[TocEntry]:
        if _DOCLING_AVAILABLE:
            return self._ensure_epub_reader().get_toc()
        return self._extractor.extract_toc()

    def get_chapters(self, min_text_len: int = 50) -> list[Chapter]:
        if _DOCLING_AVAILABLE:
            # Docling already filters low-quality content at conversion time;
            # pass min_text_len=1 so the EpubReader does not silently drop
            # short-but-valid paragraphs from the structured EPUB output.
            return self._ensure_epub_reader().get_chapters(min_text_len=1)
        return self._get_chapters_pymupdf(min_text_len)

    def get_cover(self) -> tuple[bytes | None, str | None]:
        if _DOCLING_AVAILABLE:
            return self._ensure_epub_reader().get_cover()
        return self._extractor.render_cover_page()

    def get_transcript_sections(self) -> list[PdfTranscriptSection]:
        """Available only on the pymupdf fallback path; empty on docling path.

        On the docling path, inspect the generated .epub file alongside the
        source PDF for an equivalent view of the extracted content.
        """
        if _DOCLING_AVAILABLE:
            logger.debug(
                "get_transcript_sections: not available on docling path — "
                "inspect the generated .epub alongside the source PDF instead"
            )
            return []
        return list(self._transcript_sections)

    # ── pymupdf extraction (fallback) ─────────────────────────────────────

    def _get_chapters_pymupdf(self, min_text_len: int) -> list[Chapter]:
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
                "(possibly scanned — install kenkui[pdf-enhanced] for OCR support)"
            )
        return chapters

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
            if drop_code and _looks_like_code_block(text):
                continue
            if drop_notes and _looks_like_note_block(text):
                continue
            if drop_asides and _looks_like_aside_block(text):
                continue
            cleaned.append(paragraph)
        return cleaned


# ---------------------------------------------------------------------------
# Module-level helpers (fallback path)
# ---------------------------------------------------------------------------

def _parse_page_href(href: str) -> tuple[int, int]:
    _, page_range = href.split(":", 1)
    start_s, end_s = page_range.split("-", 1)
    return int(start_s), int(end_s)


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


def _looks_like_note_block(text: str) -> bool:
    return bool(_NOTE_MARKER_RE.search(text))


def _looks_like_aside_block(text: str) -> bool:
    return bool(_ASIDE_MARKER_RE.search(text))
