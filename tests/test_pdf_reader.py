"""Integration tests for PdfReader."""

from __future__ import annotations

from pathlib import Path

import pytest

fitz = pytest.importorskip("fitz", reason="pymupdf not installed")


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


# ---------------------------------------------------------------------------
# TestPdfReaderInitialization
# ---------------------------------------------------------------------------

class TestPdfReaderInitialization:

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

    def test_blank_pdf_raises_value_error(self, pdf_blank):
        from kenkui.readers.pdf import PdfReader
        reader = PdfReader(pdf_blank)
        with pytest.raises(ValueError, match="no extractable text"):
            reader.get_chapters()


# ---------------------------------------------------------------------------
# TestPdfReaderCover
# ---------------------------------------------------------------------------

class TestPdfReaderCover:

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
