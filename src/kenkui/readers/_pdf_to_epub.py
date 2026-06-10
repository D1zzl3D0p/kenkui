"""
PDF-to-EPUB conversion using docling.

Converts PDF files to structured EPUB as a preprocessing step so that
the existing EpubReader can handle all subsequent content extraction.
The generated EPUB is written alongside the source PDF for inspection.
"""

from __future__ import annotations

import logging
import uuid
from pathlib import Path

import fitz
from ebooklib import epub
from ebooklib.epub import EpubHtml, EpubReader

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# ebooklib compatibility shim
#
# ebooklib.EpubReader._load_manifest never sets item.title from the XHTML
# <title> element; titles are lost on every read_epub() call.  We patch the
# method once at import time so that callers (including tests) can rely on
# item.title being populated from the <head><title>…</title></head> value.
# The patch is purely additive: it only fills in a title that is missing.
# ---------------------------------------------------------------------------

def _patch_ebooklib_title_roundtrip() -> None:
    """Ensure EpubHtml.title is populated from XHTML <title> after read_epub."""
    from ebooklib.utils import parse_html_string

    _orig = EpubReader._load_manifest

    def _patched(self):  # type: ignore[override]
        _orig(self)
        for item in self.book.items:
            if isinstance(item, EpubHtml) and not item.title and item.content:
                try:
                    html_tree = parse_html_string(item.content)
                    title_elem = html_tree.find(".//title")
                    if title_elem is not None and title_elem.text:
                        item.title = title_elem.text.strip()
                except Exception:
                    pass

    EpubReader._load_manifest = _patched  # type: ignore[method-assign]


_patch_ebooklib_title_roundtrip()

# Skip labels — content we never want narrated
_SKIP_LABEL_NAMES = {
    "table",
    "picture",
    "figure",
    "footnote",
    "code",
    "caption",
    "page_header",
    "page_footer",
}

# Text labels — content we narrate
_TEXT_LABEL_NAMES = {
    "text",
    "paragraph",
    "list_item",
}


def _require_docling():
    """Lazy-import docling components, raising ImportError if not installed."""
    try:
        from docling.document_converter import DocumentConverter, PdfFormatOption
        from docling.datamodel.pipeline_options import PdfPipelineOptions
        from docling.datamodel.base_models import InputFormat
        from docling_core.types.doc import DocItemLabel
    except ImportError as exc:
        raise ImportError(
            "docling is required for improved PDF processing. "
            "Install it with: pip install kenkui[pdf-enhanced]"
        ) from exc
    return DocumentConverter, PdfFormatOption, PdfPipelineOptions, InputFormat, DocItemLabel


class PdfToEpubConverter:
    """Converts a PDF to a structured EPUB using docling layout analysis."""

    def convert(self, pdf_path: Path, force_ocr: bool = False) -> Path:
        """Convert *pdf_path* to EPUB; return the EPUB path.

        The EPUB is written to the same directory as the PDF with a .epub
        extension.  Subsequent calls are cached by PDF mtime.
        """
        epub_path = pdf_path.with_suffix(".epub")
        if _is_cached(pdf_path, epub_path):
            logger.info("PDF→EPUB cache hit: %s", epub_path.name)
            return epub_path

        logger.info("Converting PDF to EPUB via docling: %s", pdf_path.name)
        title, author = _read_pdf_metadata(pdf_path)
        chapters = _extract_chapters(pdf_path, force_ocr, title)
        _write_epub(epub_path, title, author, chapters)
        logger.info(
            "PDF→EPUB written (%d chapter(s)): %s", len(chapters), epub_path.name
        )
        return epub_path


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _is_cached(pdf_path: Path, epub_path: Path) -> bool:
    return (
        epub_path.exists()
        and epub_path.stat().st_mtime > pdf_path.stat().st_mtime
    )


def _read_pdf_metadata(pdf_path: Path) -> tuple[str, str | None]:
    """Read title and author from PDF metadata using pymupdf."""
    try:
        doc = fitz.open(str(pdf_path))
        meta = doc.metadata or {}
        doc.close()
        title = (meta.get("title") or "").strip() or pdf_path.stem
        author = (meta.get("author") or "").strip() or None
        return title, author
    except Exception:
        return pdf_path.stem, None


def _extract_chapters(
    pdf_path: Path, force_ocr: bool, fallback_title: str
) -> list[tuple[str, list[str]]]:
    """Run docling and return [(chapter_title, [paragraphs])]."""
    DocumentConverter, PdfFormatOption, PdfPipelineOptions, InputFormat, DocItemLabel = _require_docling()

    # Build label sets from the real DocItemLabel enum values
    skip_labels = {
        getattr(DocItemLabel, name.upper(), name)
        for name in _SKIP_LABEL_NAMES
        if hasattr(DocItemLabel, name.upper())
    }
    # Also add by string value for forward-compat with label variants
    skip_label_values = _SKIP_LABEL_NAMES

    text_labels = {
        getattr(DocItemLabel, name.upper(), name)
        for name in _TEXT_LABEL_NAMES
        if hasattr(DocItemLabel, name.upper())
    }
    text_label_values = _TEXT_LABEL_NAMES

    section_header_label = getattr(DocItemLabel, "SECTION_HEADER", "section_header")

    pipeline_options = PdfPipelineOptions(do_ocr=force_ocr)
    converter = DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
        }
    )
    result = converter.convert(str(pdf_path))
    doc = result.document

    chapters: list[tuple[str, list[str]]] = []
    current_title = fallback_title
    current_paragraphs: list[str] = []

    for item, _level in doc.iterate_items():
        label = item.label
        label_value = label.value if hasattr(label, "value") else str(label)

        if label == section_header_label or label_value == "section_header":
            if current_paragraphs:
                chapters.append((current_title, current_paragraphs))
                current_paragraphs = []
            current_title = (getattr(item, "text", "") or "").strip() or fallback_title
            continue

        if label in skip_labels or label_value in skip_label_values:
            continue

        if label in text_labels or label_value in text_label_values:
            text = (getattr(item, "text", "") or "").strip()
            if text:
                current_paragraphs.append(text)

    # Flush final chapter
    if current_paragraphs:
        chapters.append((current_title, current_paragraphs))

    if not chapters:
        chapters = [(fallback_title, [])]

    return chapters


def _write_epub(
    epub_path: Path,
    title: str,
    author: str | None,
    chapters: list[tuple[str, list[str]]],
) -> None:
    book = epub.EpubBook()
    book.set_identifier(str(uuid.uuid4()))
    book.set_title(title)
    book.set_language("en")
    if author:
        book.add_author(author)

    epub_chapters: list[epub.EpubHtml] = []
    for i, (ch_title, paragraphs) in enumerate(chapters):
        ch = epub.EpubHtml(
            title=ch_title,
            file_name=f"chapter_{i:04d}.xhtml",
            lang="en",
        )
        safe_title = _html_escape(ch_title)
        body_html = "".join(f"<p>{_html_escape(p)}</p>" for p in paragraphs)
        ch.content = f"<h1>{safe_title}</h1>{body_html}"
        book.add_item(ch)
        epub_chapters.append(ch)

    book.toc = tuple(
        epub.Link(ch.file_name, ch.title, f"ch{i}")
        for i, ch in enumerate(epub_chapters)
    )
    book.add_item(epub.EpubNcx())
    book.add_item(epub.EpubNav())
    book.spine = ["nav", *epub_chapters]
    epub.write_epub(str(epub_path), book)


def _html_escape(text: str) -> str:
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
