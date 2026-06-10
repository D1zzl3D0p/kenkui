"""Tests for PdfToEpubConverter."""

from __future__ import annotations

import time
from pathlib import Path
from unittest.mock import MagicMock, call, patch

import pytest

fitz = pytest.importorskip("fitz", reason="pymupdf not installed")


def _make_pdf(path: Path) -> Path:
    """Write a minimal real PDF so mtime comparisons work."""
    doc = fitz.open()
    page = doc.new_page(width=595, height=842)
    page.insert_text((50, 100), "Hello world", fontsize=12)
    doc.save(str(path))
    doc.close()
    return path


def _mock_docling_label():
    """Return a namespace of DocItemLabel-like string constants."""
    class _L:
        SECTION_HEADER = "section_header"
        TEXT = "text"
        PARAGRAPH = "paragraph"
        LIST_ITEM = "list_item"
        TABLE = "table"
        PICTURE = "picture"
        FOOTNOTE = "footnote"
        CODE = "code"
        CAPTION = "caption"
        PAGE_HEADER = "page_header"
        PAGE_FOOTER = "page_footer"
    return _L()


def _make_item(label, text: str):
    item = MagicMock()
    item.label = label
    item.text = text
    return item


def _mock_require_docling(doc_items: list[tuple]):
    """
    Return a patcher for _require_docling that produces a DocumentConverter
    whose .convert().document.iterate_items() yields doc_items.

    doc_items: list of (item, level) pairs already built with _make_item().
    """
    Label = _mock_docling_label()

    MockPipelineOptions = MagicMock()
    MockFormatOption = MagicMock()
    MockInputFormat = MagicMock()
    MockInputFormat.PDF = "PDF"

    mock_doc = MagicMock()
    mock_doc.iterate_items.return_value = iter(doc_items)

    mock_result = MagicMock()
    mock_result.document = mock_doc

    MockConverter = MagicMock()
    MockConverter.return_value.convert.return_value = mock_result

    return (
        MockConverter,
        MockFormatOption,
        MockPipelineOptions,
        MockInputFormat,
        Label,
    )


# ---------------------------------------------------------------------------
# Cache tests
# ---------------------------------------------------------------------------

class TestCaching:

    def test_cache_hit_skips_conversion(self, tmp_path):
        """If the EPUB is newer than the PDF, no conversion should run."""
        from kenkui.readers._pdf_to_epub import PdfToEpubConverter

        pdf = _make_pdf(tmp_path / "book.pdf")
        epub = tmp_path / "book.epub"
        epub.write_bytes(b"fake epub")
        # Make epub newer than pdf
        future_mtime = pdf.stat().st_mtime + 10
        import os
        os.utime(epub, (future_mtime, future_mtime))

        with patch(
            "kenkui.readers._pdf_to_epub._require_docling",
            side_effect=AssertionError("docling should NOT be called on cache hit"),
        ):
            result = PdfToEpubConverter().convert(pdf)

        assert result == epub

    def test_cache_miss_when_epub_older_than_pdf(self, tmp_path):
        """If the EPUB is older than the PDF, it should be re-converted."""
        from kenkui.readers._pdf_to_epub import PdfToEpubConverter

        pdf = _make_pdf(tmp_path / "book.pdf")
        epub = tmp_path / "book.epub"
        epub.write_bytes(b"stale epub")
        # Make epub *older* than pdf
        import os
        old_mtime = pdf.stat().st_mtime - 10
        os.utime(epub, (old_mtime, old_mtime))

        Label = _mock_docling_label()
        items = [
            (_make_item(Label.TEXT, "Hello world."), 0),
        ]
        require_rv = _mock_require_docling(items)

        with patch("kenkui.readers._pdf_to_epub._require_docling", return_value=require_rv):
            result = PdfToEpubConverter().convert(pdf)

        assert result == epub
        # The stale content should have been overwritten
        assert epub.read_bytes() != b"stale epub"

    def test_cache_miss_when_no_epub(self, tmp_path):
        """If no EPUB exists, it must be created."""
        from kenkui.readers._pdf_to_epub import PdfToEpubConverter

        pdf = _make_pdf(tmp_path / "book.pdf")
        epub = tmp_path / "book.epub"
        assert not epub.exists()

        Label = _mock_docling_label()
        items = [(_make_item(Label.TEXT, "Content here."), 0)]
        require_rv = _mock_require_docling(items)

        with patch("kenkui.readers._pdf_to_epub._require_docling", return_value=require_rv):
            result = PdfToEpubConverter().convert(pdf)

        assert result == epub
        assert epub.exists()
        assert epub.stat().st_size > 0


# ---------------------------------------------------------------------------
# Chapter extraction tests
# ---------------------------------------------------------------------------

class TestChapterExtraction:

    def test_section_headers_become_chapter_boundaries(self, tmp_path):
        """SectionHeaderItems create new chapters; TextItems become paragraphs."""
        from kenkui.readers._pdf_to_epub import PdfToEpubConverter
        from ebooklib import epub as epub_lib

        pdf = _make_pdf(tmp_path / "book.pdf")
        Label = _mock_docling_label()
        items = [
            (_make_item(Label.SECTION_HEADER, "Chapter One"), 0),
            (_make_item(Label.TEXT, "First paragraph."), 1),
            (_make_item(Label.TEXT, "Second paragraph."), 1),
            (_make_item(Label.SECTION_HEADER, "Chapter Two"), 0),
            (_make_item(Label.TEXT, "Third paragraph."), 1),
        ]
        require_rv = _mock_require_docling(items)

        with patch("kenkui.readers._pdf_to_epub._require_docling", return_value=require_rv):
            epub_path = PdfToEpubConverter().convert(pdf)

        import warnings
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            book = epub_lib.read_epub(str(epub_path))

        # Two chapters in the spine (plus nav)
        content_items = [i for i in book.get_items() if i.file_name.startswith("chapter_")]
        assert len(content_items) == 2
        titles = [i.title for i in content_items]
        assert "Chapter One" in titles
        assert "Chapter Two" in titles

    def test_no_section_headers_yields_single_chapter(self, tmp_path):
        """When there are no headers, all text goes into one chapter."""
        from kenkui.readers._pdf_to_epub import PdfToEpubConverter
        from ebooklib import epub as epub_lib

        pdf = _make_pdf(tmp_path / "book.pdf")
        Label = _mock_docling_label()
        items = [
            (_make_item(Label.TEXT, "Para one."), 0),
            (_make_item(Label.TEXT, "Para two."), 0),
        ]
        require_rv = _mock_require_docling(items)

        with patch("kenkui.readers._pdf_to_epub._require_docling", return_value=require_rv):
            epub_path = PdfToEpubConverter().convert(pdf)

        import warnings
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            book = epub_lib.read_epub(str(epub_path))

        content_items = [i for i in book.get_items() if i.file_name.startswith("chapter_")]
        assert len(content_items) == 1

    def test_tables_footnotes_and_code_are_skipped(self, tmp_path):
        """TABLE, FOOTNOTE, CODE items must not appear in chapter text."""
        from kenkui.readers._pdf_to_epub import PdfToEpubConverter
        from ebooklib import epub as epub_lib

        pdf = _make_pdf(tmp_path / "book.pdf")
        Label = _mock_docling_label()
        items = [
            (_make_item(Label.SECTION_HEADER, "Ch1"), 0),
            (_make_item(Label.TEXT, "Good paragraph."), 1),
            (_make_item(Label.TABLE, "col1 | col2"), 1),
            (_make_item(Label.FOOTNOTE, "Footnote text."), 1),
            (_make_item(Label.CODE, "def foo(): pass"), 1),
        ]
        require_rv = _mock_require_docling(items)

        with patch("kenkui.readers._pdf_to_epub._require_docling", return_value=require_rv):
            epub_path = PdfToEpubConverter().convert(pdf)

        import warnings
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            book = epub_lib.read_epub(str(epub_path))

        content = b"".join(
            i.get_body_content() or b""
            for i in book.get_items()
            if i.file_name.startswith("chapter_")
        ).decode()
        assert "Good paragraph." in content
        assert "col1" not in content
        assert "Footnote text." not in content
        assert "def foo" not in content

    def test_list_items_are_included_as_paragraphs(self, tmp_path):
        """LIST_ITEM label is treated as regular paragraph content."""
        from kenkui.readers._pdf_to_epub import PdfToEpubConverter
        from ebooklib import epub as epub_lib

        pdf = _make_pdf(tmp_path / "book.pdf")
        Label = _mock_docling_label()
        items = [
            (_make_item(Label.SECTION_HEADER, "Ch1"), 0),
            (_make_item(Label.LIST_ITEM, "First bullet."), 1),
            (_make_item(Label.LIST_ITEM, "Second bullet."), 1),
        ]
        require_rv = _mock_require_docling(items)

        with patch("kenkui.readers._pdf_to_epub._require_docling", return_value=require_rv):
            epub_path = PdfToEpubConverter().convert(pdf)

        import warnings
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            book = epub_lib.read_epub(str(epub_path))

        content = b"".join(
            i.get_body_content() or b""
            for i in book.get_items()
            if i.file_name.startswith("chapter_")
        ).decode()
        assert "First bullet." in content
        assert "Second bullet." in content


# ---------------------------------------------------------------------------
# force_ocr test
# ---------------------------------------------------------------------------

class TestForceOcr:

    def test_force_ocr_sets_do_ocr_true(self, tmp_path):
        """When force_ocr=True, PdfPipelineOptions must be called with do_ocr=True."""
        from kenkui.readers._pdf_to_epub import PdfToEpubConverter

        pdf = _make_pdf(tmp_path / "book.pdf")
        Label = _mock_docling_label()
        items = [(_make_item(Label.TEXT, "Text."), 0)]
        (
            MockConverter,
            MockFormatOption,
            MockPipelineOptions,
            MockInputFormat,
            Label2,
        ) = _mock_require_docling(items)

        with patch(
            "kenkui.readers._pdf_to_epub._require_docling",
            return_value=(MockConverter, MockFormatOption, MockPipelineOptions, MockInputFormat, Label2),
        ):
            PdfToEpubConverter().convert(pdf, force_ocr=True)

        MockPipelineOptions.assert_called_once_with(do_ocr=True)

    def test_force_ocr_false_by_default(self, tmp_path):
        """Default force_ocr=False means PdfPipelineOptions called with do_ocr=False."""
        from kenkui.readers._pdf_to_epub import PdfToEpubConverter

        pdf = _make_pdf(tmp_path / "book.pdf")
        Label = _mock_docling_label()
        items = [(_make_item(Label.TEXT, "Text."), 0)]
        (
            MockConverter,
            MockFormatOption,
            MockPipelineOptions,
            MockInputFormat,
            Label2,
        ) = _mock_require_docling(items)

        with patch(
            "kenkui.readers._pdf_to_epub._require_docling",
            return_value=(MockConverter, MockFormatOption, MockPipelineOptions, MockInputFormat, Label2),
        ):
            PdfToEpubConverter().convert(pdf)

        MockPipelineOptions.assert_called_once_with(do_ocr=False)


# ---------------------------------------------------------------------------
# PdfReader integration: docling path
# ---------------------------------------------------------------------------

class TestPdfReaderDoclingPath:
    """Tests for PdfReader when _DOCLING_AVAILABLE is True."""

    def test_get_chapters_delegates_to_epub_reader(self, tmp_path):
        """PdfReader.get_chapters() calls EpubReader.get_chapters() on docling path."""
        import warnings
        from kenkui.readers.pdf import PdfReader

        pdf = _make_pdf(tmp_path / "book.pdf")
        Label = _mock_docling_label()
        items = [
            (_make_item(Label.SECTION_HEADER, "Ch1"), 0),
            (_make_item(Label.TEXT, "Real paragraph here."), 1),
        ]
        require_rv = _mock_require_docling(items)

        with patch("kenkui.readers._pdf_to_epub._require_docling", return_value=require_rv), \
             patch("kenkui.readers.pdf._DOCLING_AVAILABLE", True):
            reader = PdfReader(pdf)
            chapters = reader.get_chapters()

        assert len(chapters) == 1
        assert chapters[0].title == "Ch1"
        assert any("Real paragraph here." in p for p in chapters[0].paragraphs)

    def test_configure_pdf_extraction_stores_force_ocr(self, tmp_path):
        """configure_pdf_extraction({'force_ocr': True}) is honoured on docling path."""
        from kenkui.readers.pdf import PdfReader

        pdf = _make_pdf(tmp_path / "book.pdf")
        Label = _mock_docling_label()
        items = [(_make_item(Label.TEXT, "Content."), 0)]
        (MockConverter, MockFmt, MockPipelineOpts, MockInput, L) = _mock_require_docling(items)

        with patch(
            "kenkui.readers._pdf_to_epub._require_docling",
            return_value=(MockConverter, MockFmt, MockPipelineOpts, MockInput, L),
        ), patch("kenkui.readers.pdf._DOCLING_AVAILABLE", True):
            reader = PdfReader(pdf)
            reader.configure_pdf_extraction({"force_ocr": True})
            reader.get_chapters()

        MockPipelineOpts.assert_called_once_with(do_ocr=True)

    def test_get_transcript_sections_returns_empty_on_docling_path(self, tmp_path):
        """Transcript sections are not available on the docling path (use the .epub)."""
        from kenkui.readers.pdf import PdfReader

        pdf = _make_pdf(tmp_path / "book.pdf")
        Label = _mock_docling_label()
        items = [(_make_item(Label.TEXT, "Content."), 0)]
        require_rv = _mock_require_docling(items)

        with patch("kenkui.readers._pdf_to_epub._require_docling", return_value=require_rv), \
             patch("kenkui.readers.pdf._DOCLING_AVAILABLE", True):
            reader = PdfReader(pdf)
            reader.get_chapters()
            assert reader.get_transcript_sections() == []

    def test_fallback_path_used_when_docling_unavailable(self, tmp_path):
        """When _DOCLING_AVAILABLE is False, PdfReader uses the pymupdf path."""
        import fitz as _fitz
        from kenkui.readers.pdf import PdfReader

        # Build a real PDF with fitz so pymupdf path can open it
        doc = _fitz.open()
        doc.set_metadata({"title": "Fallback Book", "author": "Bob"})
        page = doc.new_page(width=595, height=842)
        page.insert_text((50, 100), "Hello from pymupdf path.", fontsize=12)
        pdf = tmp_path / "fallback.pdf"
        doc.save(str(pdf))
        doc.close()

        with patch("kenkui.readers.pdf._DOCLING_AVAILABLE", False):
            reader = PdfReader(pdf)
            meta = reader.get_metadata()
        assert meta.title == "Fallback Book"
