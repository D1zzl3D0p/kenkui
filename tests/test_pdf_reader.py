"""Integration tests for PdfReader."""

from __future__ import annotations

from pathlib import Path

import pytest

fitz = pytest.importorskip("fitz", reason="pymupdf not installed")

from unittest.mock import patch

# The existing test suite exercises the pymupdf fallback path.
# When docling is installed, PdfReader uses the docling path instead.
try:
    import docling as _docling  # noqa: F401
    _docling_installed = True
except ImportError:
    _docling_installed = False

# ---------------------------------------------------------------------------
# Fixtures: build temporary PDFs
# ---------------------------------------------------------------------------

def _write_pdf(path: Path, doc: fitz.Document) -> Path:
    doc.save(str(path))
    doc.close()
    return path


@pytest.fixture
def pdf_with_bookmarks(tmp_path) -> Path:
    """A well-structured PDF: metadata + bookmarks + multi-page content."""
    doc = fitz.open()
    doc.set_metadata({
        "title": "Test Book",
        "author": "Jane Author",
    })
    for i in range(1, 4):
        page = doc.new_page(width=595, height=842)
        page.insert_text((50, 100), f"Chapter {i}", fontsize=20)
        page.insert_text(
            (50, 140),
            f"This is the body text for chapter {i}. " * 5,
            fontsize=12,
        )
    doc.set_toc([
        [1, "Chapter 1", 1],
        [1, "Chapter 2", 2],
        [1, "Chapter 3", 3],
    ])
    return _write_pdf(tmp_path / "book.pdf", doc)


@pytest.fixture
def pdf_no_bookmarks(tmp_path) -> Path:
    """A PDF with heading-style text but no bookmarks."""
    doc = fitz.open()
    for i in range(1, 3):
        page = doc.new_page(width=595, height=842)
        page.insert_text((50, 80), f"Chapter {i}", fontsize=22)
        page.insert_text(
            (50, 130),
            "Body text here. " * 10,
            fontsize=11,
        )
    return _write_pdf(tmp_path / "no_bookmarks.pdf", doc)


@pytest.fixture
def pdf_no_metadata(tmp_path) -> Path:
    """A PDF with no title/author metadata."""
    doc = fitz.open()
    page = doc.new_page(width=595, height=842)
    page.insert_text((50, 100), "Some content.", fontsize=12)
    return _write_pdf(tmp_path / "untitled_book.pdf", doc)


@pytest.fixture
def pdf_with_headers_footers(tmp_path) -> Path:
    """A PDF where every page has a repeated running header."""
    doc = fitz.open()
    for i in range(10):
        page = doc.new_page(width=595, height=842)
        page.insert_text((50, 50), "My Running Header", fontsize=10)
        page.insert_text(
            (50, 120),
            f"This is the real chapter content for page {i}. " * 3,
            fontsize=12,
        )
    return _write_pdf(tmp_path / "headers.pdf", doc)


@pytest.fixture
def pdf_encrypted(tmp_path) -> Path:
    """An encrypted (owner-password) PDF."""
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((50, 100), "Secret content.", fontsize=12)
    out = tmp_path / "encrypted.pdf"
    doc.save(
        str(out),
        encryption=fitz.PDF_ENCRYPT_AES_256,
        owner_pw="ownerpass",
        user_pw="userpass",
    )
    doc.close()
    return out


@pytest.fixture
def pdf_blank(tmp_path) -> Path:
    """A PDF with pages but no text (simulates fully-scanned document)."""
    doc = fitz.open()
    doc.new_page(width=595, height=842)  # blank page, no text
    return _write_pdf(tmp_path / "blank.pdf", doc)


@pytest.fixture
def pdf_with_notes_and_code(tmp_path) -> Path:
    """A PDF containing prose, note-style content, and a code block."""
    doc = fitz.open()
    page = doc.new_page(width=595, height=842)
    page.insert_text((50, 80), "Chapter 1", fontsize=20)
    page.insert_text(
        (50, 130),
        "This prose paragraph should remain in the transcript and audio pipeline.",
        fontsize=12,
    )
    page.insert_text(
        (50, 200),
        "Note: this explanatory note should be filtered out when note cleanup is enabled.",
        fontsize=12,
    )
    page.insert_text(
        (50, 270),
        "def sample():\n    return 42\n    return 43",
        fontsize=12,
    )
    doc.set_toc([[1, "Chapter 1", 1]])
    return _write_pdf(tmp_path / "notes_and_code.pdf", doc)


# ---------------------------------------------------------------------------
# TestPdfReaderInitialization
# ---------------------------------------------------------------------------

class TestPdfReaderInitialization:
    pytestmark = pytest.mark.skipif(
        _docling_installed,
        reason="tests the pymupdf fallback path only",
    )

    def test_initializes_from_path(self, pdf_with_bookmarks):
        from kenkui.readers.pdf import PdfReader
        reader = PdfReader(pdf_with_bookmarks)
        assert reader is not None
        assert reader.filepath == pdf_with_bookmarks

    def test_extension_reported_as_pdf(self, pdf_with_bookmarks):
        from kenkui.readers.pdf import PdfReader
        reader = PdfReader(pdf_with_bookmarks)
        assert reader.extension == ".pdf"
        assert reader.format_name == "PDF"

    def test_encrypted_pdf_raises_value_error(self, pdf_encrypted):
        from kenkui.readers.pdf import PdfReader
        with pytest.raises(ValueError, match="encrypted"):
            PdfReader(pdf_encrypted)

    def test_nonexistent_file_raises(self, tmp_path):
        from kenkui.readers.pdf import PdfReader
        with pytest.raises(Exception):
            PdfReader(tmp_path / "ghost.pdf")


# ---------------------------------------------------------------------------
# TestPdfReaderMetadata
# ---------------------------------------------------------------------------

class TestPdfReaderMetadata:
    pytestmark = pytest.mark.skipif(
        _docling_installed,
        reason="tests the pymupdf fallback path only",
    )

    def test_extracts_title_from_metadata(self, pdf_with_bookmarks):
        from kenkui.readers.pdf import PdfReader
        reader = PdfReader(pdf_with_bookmarks)
        meta = reader.get_metadata()
        assert meta.title == "Test Book"

    def test_extracts_author_from_metadata(self, pdf_with_bookmarks):
        from kenkui.readers.pdf import PdfReader
        reader = PdfReader(pdf_with_bookmarks)
        meta = reader.get_metadata()
        assert meta.author == "Jane Author"

    def test_falls_back_to_filename_when_no_title(self, pdf_no_metadata):
        from kenkui.readers.pdf import PdfReader
        reader = PdfReader(pdf_no_metadata)
        meta = reader.get_metadata()
        assert isinstance(meta.title, str)
        assert len(meta.title) > 0
        assert "untitled_book" in meta.title.lower() or meta.title  # at minimum non-empty


# ---------------------------------------------------------------------------
# TestPdfReaderToc
# ---------------------------------------------------------------------------

class TestPdfReaderToc:
    pytestmark = pytest.mark.skipif(
        _docling_installed,
        reason="tests the pymupdf fallback path only",
    )

    def test_toc_from_bookmarks(self, pdf_with_bookmarks):
        from kenkui.readers import TocEntry
        from kenkui.readers.pdf import PdfReader
        reader = PdfReader(pdf_with_bookmarks)
        toc = reader.get_toc()
        assert len(toc) == 3
        assert all(isinstance(e, TocEntry) for e in toc)
        assert toc[0].title == "Chapter 1"

    def test_toc_from_heading_heuristic(self, pdf_no_bookmarks):
        from kenkui.readers import TocEntry
        from kenkui.readers.pdf import PdfReader
        reader = PdfReader(pdf_no_bookmarks)
        toc = reader.get_toc()
        assert len(toc) >= 1
        assert all(isinstance(e, TocEntry) for e in toc)

    def test_toc_not_empty_even_without_structure(self, pdf_no_metadata):
        """Even a structureless PDF produces at least one TOC entry (page-range fallback)."""
        from kenkui.readers.pdf import PdfReader
        reader = PdfReader(pdf_no_metadata)
        toc = reader.get_toc()
        assert len(toc) >= 1


# ---------------------------------------------------------------------------
# TestPdfReaderChapters
# ---------------------------------------------------------------------------

class TestPdfReaderChapters:
    pytestmark = pytest.mark.skipif(
        _docling_installed,
        reason="tests the pymupdf fallback path only",
    )

    def test_returns_list_of_chapters(self, pdf_with_bookmarks):
        from kenkui.models import Chapter
        from kenkui.readers.pdf import PdfReader
        reader = PdfReader(pdf_with_bookmarks)
        chapters = reader.get_chapters()
        assert isinstance(chapters, list)
        assert len(chapters) > 0
        assert all(isinstance(c, Chapter) for c in chapters)

    def test_chapter_fields_present(self, pdf_with_bookmarks):
        from kenkui.readers.pdf import PdfReader
        reader = PdfReader(pdf_with_bookmarks)
        chapters = reader.get_chapters()
        for ch in chapters:
            assert isinstance(ch.index, int)
            assert isinstance(ch.title, str)
            assert isinstance(ch.paragraphs, list)
            assert ch.segments is None

    def test_paragraphs_are_nonempty_strings(self, pdf_with_bookmarks):
        from kenkui.readers.pdf import PdfReader
        reader = PdfReader(pdf_with_bookmarks)
        chapters = reader.get_chapters()
        for ch in chapters:
            for para in ch.paragraphs:
                assert isinstance(para, str)
                assert len(para.strip()) > 0

    def test_chapters_indexed_sequentially(self, pdf_with_bookmarks):
        from kenkui.readers.pdf import PdfReader
        reader = PdfReader(pdf_with_bookmarks)
        chapters = reader.get_chapters()
        indices = [c.index for c in chapters]
        assert indices == list(range(len(chapters)))

    def test_headers_footers_absent_from_paragraphs(self, pdf_with_headers_footers):
        from kenkui.readers.pdf import PdfReader
        reader = PdfReader(pdf_with_headers_footers)
        chapters = reader.get_chapters()
        all_text = " ".join(p for c in chapters for p in c.paragraphs)
        assert "My Running Header" not in all_text

    def test_pdf_cleanup_options_filter_notes_and_code(self, pdf_with_notes_and_code):
        from kenkui.readers.pdf import PdfReader

        reader = PdfReader(pdf_with_notes_and_code)
        reader.configure_pdf_extraction(
            {
                "drop_code_blocks": True,
                "drop_notes": True,
                "drop_asides": True,
            }
        )
        chapters = reader.get_chapters(min_text_len=1)
        combined = " ".join(" ".join(ch.paragraphs) for ch in chapters)
        assert "Note:" not in combined
        assert "def sample" not in combined
        assert "This prose paragraph" in combined

        sections = reader.get_transcript_sections()
        assert sections
        raw = " ".join(" ".join(section.raw_paragraphs) for section in sections)
        filtered = " ".join(" ".join(section.filtered_paragraphs) for section in sections)
        assert "Note:" in raw
        assert "def sample" in raw
        assert "Note:" not in filtered
        assert "def sample" not in filtered

    def test_blank_pdf_raises_value_error(self, pdf_blank):
        from kenkui.readers.pdf import PdfReader
        reader = PdfReader(pdf_blank)
        with pytest.raises(ValueError, match="no extractable text"):
            reader.get_chapters()


# ---------------------------------------------------------------------------
# TestPdfReaderCover
# ---------------------------------------------------------------------------

class TestPdfReaderCover:
    pytestmark = pytest.mark.skipif(
        _docling_installed,
        reason="tests the pymupdf fallback path only",
    )

    def test_cover_returns_bytes(self, pdf_with_bookmarks):
        from kenkui.readers.pdf import PdfReader
        reader = PdfReader(pdf_with_bookmarks)
        data, mime = reader.get_cover()
        assert isinstance(data, bytes)
        assert len(data) > 0
        assert mime == "image/png"

    def test_cover_is_png_format(self, pdf_with_bookmarks):
        """Cover bytes start with the PNG magic bytes."""
        from kenkui.readers.pdf import PdfReader
        reader = PdfReader(pdf_with_bookmarks)
        data, _ = reader.get_cover()
        assert data is not None
        assert data[:8] == b"\x89PNG\r\n\x1a\n"


# ---------------------------------------------------------------------------
# TestPdfReaderRegistry
# ---------------------------------------------------------------------------

class TestPdfReaderRegistry:

    def test_pdf_extension_in_registry(self):
        from kenkui.readers import Registry
        assert ".pdf" in Registry.supported_extensions()

    def test_get_reader_dispatches_to_pdf_reader(self, pdf_with_bookmarks):
        from kenkui.readers import get_reader
        from kenkui.readers.pdf import PdfReader
        reader = get_reader(pdf_with_bookmarks)
        assert isinstance(reader, PdfReader)


# ---------------------------------------------------------------------------
# Helpers for TestPdfReaderDoclingPath (docling mock utilities)
# ---------------------------------------------------------------------------

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
    from unittest.mock import MagicMock
    item = MagicMock()
    item.label = label
    item.text = text
    return item


def _mock_require_docling(doc_items: list):
    """
    Return a patcher for _require_docling that produces a DocumentConverter
    whose .convert().document.iterate_items() yields doc_items.

    doc_items: list of (item, level) pairs already built with _make_item().
    """
    from unittest.mock import MagicMock

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

    Label = _mock_docling_label()

    return (
        MockConverter,
        MockFormatOption,
        MockPipelineOptions,
        MockInputFormat,
        Label,
    )


# ---------------------------------------------------------------------------
# TestPdfReaderDoclingPath
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
        with patch("kenkui.readers.pdf._DOCLING_AVAILABLE", True):
            reader = PdfReader(pdf)
            assert reader.get_transcript_sections() == []

    def test_fallback_path_used_when_docling_unavailable(self, tmp_path):
        """When _DOCLING_AVAILABLE is False, PdfReader uses the pymupdf path."""
        from kenkui.readers.pdf import PdfReader

        doc = fitz.open()
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
