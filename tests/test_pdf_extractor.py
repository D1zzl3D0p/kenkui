"""Unit tests for PdfTextExtractor internals."""

from __future__ import annotations

import pytest

fitz = pytest.importorskip("fitz", reason="pymupdf not installed")


# ---------------------------------------------------------------------------
# Helpers: build minimal in-memory PDFs
# ---------------------------------------------------------------------------

def _make_doc_with_text(pages: list[str]) -> "fitz.Document":
    """Return a fitz.Document with one page per string in *pages*."""
    doc = fitz.open()
    for text in pages:
        page = doc.new_page(width=595, height=842)
        page.insert_text((50, 100), text, fontsize=12)
    return doc


def _make_doc_with_bookmarks(chapters: list[tuple[str, int]]) -> "fitz.Document":
    """Return a multi-page doc with PDF bookmarks.

    *chapters* is a list of (title, 1-indexed page number).
    """
    doc = fitz.open()
    for title, _ in chapters:
        page = doc.new_page(width=595, height=842)
        page.insert_text((50, 100), title, fontsize=12)
        page.insert_text((50, 130), "Body text for this chapter.", fontsize=12)
    toc = [[1, title, page_no] for title, page_no in chapters]
    doc.set_toc(toc)
    return doc


def _make_doc_with_headings(
    chapters: list[tuple[str, str]],
    heading_fontsize: float = 20.0,
    body_fontsize: float = 12.0,
) -> "fitz.Document":
    """Return a doc without bookmarks but with visually distinct headings.

    *chapters* is a list of (heading_text, body_text).
    """
    doc = fitz.open()
    for heading, body in chapters:
        page = doc.new_page(width=595, height=842)
        page.insert_text((50, 80), heading, fontsize=heading_fontsize)
        page.insert_text((50, 120), body, fontsize=body_fontsize)
    return doc


# ---------------------------------------------------------------------------
# Import target — deferred so we can write tests before the module exists
# ---------------------------------------------------------------------------

def _get_extractor_class():
    from kenkui.readers._pdf_extract import PdfTextExtractor
    return PdfTextExtractor


# ---------------------------------------------------------------------------
# _clean_text
# ---------------------------------------------------------------------------

class TestCleanText:
    """Tests for PdfTextExtractor._clean_text."""

    @pytest.fixture
    def extractor(self):
        PdfTextExtractor = _get_extractor_class()
        doc = _make_doc_with_text(["Hello world."])
        return PdfTextExtractor(doc)

    def test_dehyphenates_line_wrapped_words(self, extractor):
        result = extractor._clean_text("incon-\nvenient")
        assert "inconvenient" in result
        assert "-\n" not in result

    def test_normalizes_fi_ligature(self, extractor):
        result = extractor._clean_text("ﬁrst")
        assert result.startswith("fi")

    def test_normalizes_fl_ligature(self, extractor):
        result = extractor._clean_text("ﬂoor")
        assert result.startswith("fl")

    def test_strips_lone_page_number_line(self, extractor):
        text = "Some paragraph.\n\n42\n\nNext paragraph."
        result = extractor._clean_text(text)
        assert "\n42\n" not in result

    def test_preserves_normal_numbers_in_text(self, extractor):
        result = extractor._clean_text("There were 42 soldiers.")
        assert "42" in result

    def test_handles_empty_string(self, extractor):
        assert extractor._clean_text("") == ""


# ---------------------------------------------------------------------------
# _detect_headers_footers
# ---------------------------------------------------------------------------

class TestDetectHeadersFooters:
    """Tests for PdfTextExtractor._detect_headers_footers."""

    def test_detects_repeated_footer_text(self):
        PdfTextExtractor = _get_extractor_class()
        # 10 pages, each ending with the same footer line
        pages = [f"Chapter content page {i}.\n\nMy Book Title" for i in range(10)]
        doc = _make_doc_with_text(pages)
        extractor = PdfTextExtractor(doc)
        noise = extractor._detect_headers_footers()
        assert any("My Book Title" in n for n in noise)

    def test_does_not_flag_unique_content(self):
        PdfTextExtractor = _get_extractor_class()
        pages = [f"Unique content on page {i} with different words." for i in range(10)]
        doc = _make_doc_with_text(pages)
        extractor = PdfTextExtractor(doc)
        noise = extractor._detect_headers_footers()
        # No line should appear as noise when all text is unique
        assert len(noise) == 0

    def test_threshold_respected(self):
        """With threshold=1.0 (100%), even text on 80% of pages should not be flagged."""
        import os
        PdfTextExtractor = _get_extractor_class()
        pages = (
            ["Repeated header\nContent page." for _ in range(8)]
            + ["Only content here." for _ in range(2)]
        )
        doc = _make_doc_with_text(pages)
        env_patch = {"KENKUI_PDF_FOOTER_THRESHOLD": "1.0"}
        orig = {k: os.environ.get(k) for k in env_patch}
        try:
            os.environ.update(env_patch)
            extractor = PdfTextExtractor(doc)
            noise = extractor._detect_headers_footers()
            assert len(noise) == 0
        finally:
            for k, v in orig.items():
                if v is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = v


# ---------------------------------------------------------------------------
# extract_toc — bookmarks path
# ---------------------------------------------------------------------------

class TestExtractTocFromBookmarks:
    """extract_toc returns entries derived from PDF bookmarks."""

    def test_returns_toc_entries_from_bookmarks(self):
        from kenkui.readers import TocEntry
        PdfTextExtractor = _get_extractor_class()
        doc = _make_doc_with_bookmarks([("Prologue", 1), ("Chapter 1", 2), ("Chapter 2", 3)])
        extractor = PdfTextExtractor(doc)
        toc = extractor.extract_toc()
        assert len(toc) == 3
        assert all(isinstance(e, TocEntry) for e in toc)
        assert toc[0].title == "Prologue"
        assert toc[1].title == "Chapter 1"

    def test_href_encodes_page_range(self):
        """href field carries 'page:<start>-<end>' for range extraction."""
        PdfTextExtractor = _get_extractor_class()
        doc = _make_doc_with_bookmarks([("Ch 1", 1), ("Ch 2", 2)])
        extractor = PdfTextExtractor(doc)
        toc = extractor.extract_toc()
        assert toc[0].href.startswith("page:")


# ---------------------------------------------------------------------------
# extract_toc — heading heuristic path
# ---------------------------------------------------------------------------

class TestExtractTocHeadingHeuristic:
    """extract_toc falls back to font-size heuristic when no bookmarks."""

    def test_detects_chapters_by_large_font(self):
        from kenkui.readers import TocEntry
        PdfTextExtractor = _get_extractor_class()
        chapters = [
            ("Chapter One", "The story began on a dark night."),
            ("Chapter Two", "Morning arrived with unexpected news."),
        ]
        doc = _make_doc_with_headings(chapters, heading_fontsize=22.0, body_fontsize=11.0)
        extractor = PdfTextExtractor(doc)
        toc = extractor.extract_toc()
        assert len(toc) >= 2
        assert all(isinstance(e, TocEntry) for e in toc)

    def test_heading_ratio_env_var_respected(self):
        """With a very high ratio, no text qualifies as a heading."""
        import os
        PdfTextExtractor = _get_extractor_class()
        chapters = [("Chapter One", "Body text here.")]
        doc = _make_doc_with_headings(chapters, heading_fontsize=14.0, body_fontsize=12.0)
        env_patch = {"KENKUI_PDF_HEADING_SIZE_RATIO": "5.0"}
        orig = {k: os.environ.get(k) for k in env_patch}
        try:
            os.environ.update(env_patch)
            extractor = PdfTextExtractor(doc)
            toc = extractor.extract_toc()
            # With ratio=5x, 14pt vs 12pt body won't qualify; expect page-range fallback
            for entry in toc:
                assert entry.href.startswith("page:")
        finally:
            for k, v in orig.items():
                if v is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = v


# ---------------------------------------------------------------------------
# extract_toc — page-range fallback
# ---------------------------------------------------------------------------

class TestExtractTocPageRangeFallback:
    """extract_toc falls back to page chunks when no bookmarks and no headings."""

    def test_page_chunk_fallback_produces_entries(self):
        from kenkui.readers import TocEntry
        PdfTextExtractor = _get_extractor_class()
        # Uniform 12pt text, no bookmark, no heading contrast
        pages = [f"Body text on page {i}." for i in range(5)]
        doc = _make_doc_with_text(pages)
        extractor = PdfTextExtractor(doc)
        toc = extractor.extract_toc()
        assert len(toc) >= 1
        assert all(isinstance(e, TocEntry) for e in toc)

    def test_page_chunk_size_env_var(self):
        """KENKUI_PDF_PAGE_CHUNK_SIZE controls chunk size."""
        import os
        PdfTextExtractor = _get_extractor_class()
        pages = [f"Body text on page {i}." for i in range(6)]
        doc = _make_doc_with_text(pages)
        env_patch = {"KENKUI_PDF_PAGE_CHUNK_SIZE": "2"}
        orig = {k: os.environ.get(k) for k in env_patch}
        try:
            os.environ.update(env_patch)
            extractor = PdfTextExtractor(doc)
            toc = extractor.extract_toc()
            assert len(toc) == 3  # 6 pages / 2 per chunk = 3 entries
        finally:
            for k, v in orig.items():
                if v is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = v


# ---------------------------------------------------------------------------
# extract_text_for_pages
# ---------------------------------------------------------------------------

class TestExtractTextForPages:
    """extract_text_for_pages returns clean paragraphs for a page range."""

    def test_returns_list_of_strings(self):
        PdfTextExtractor = _get_extractor_class()
        doc = _make_doc_with_text(["First page text.", "Second page text."])
        extractor = PdfTextExtractor(doc)
        paras = extractor.extract_text_for_pages(0, 1)
        assert isinstance(paras, list)
        assert all(isinstance(p, str) for p in paras)

    def test_returns_nonempty_paragraphs_for_text_pages(self):
        PdfTextExtractor = _get_extractor_class()
        doc = _make_doc_with_text(["This is a meaningful paragraph with real content."])
        extractor = PdfTextExtractor(doc)
        paras = extractor.extract_text_for_pages(0, 0)
        assert len(paras) > 0

    def test_strips_detected_header_footer(self):
        PdfTextExtractor = _get_extractor_class()
        pages = [f"Real content page {i}.\n\nRunning Header" for i in range(10)]
        doc = _make_doc_with_text(pages)
        extractor = PdfTextExtractor(doc)
        paras = extractor.extract_text_for_pages(0, 9)
        combined = " ".join(paras)
        assert "Running Header" not in combined


# ---------------------------------------------------------------------------
# OCR seam
# ---------------------------------------------------------------------------

class TestOcrSeam:
    """OCR backend is called when a page has no text layer."""

    def test_ocr_backend_called_for_empty_page(self):
        PdfTextExtractor = _get_extractor_class()
        # Create a page with no text
        doc = fitz.open()
        doc.new_page(width=595, height=842)  # blank page

        called = []

        def fake_ocr(page):
            called.append(page.number)
            return "OCR extracted text from page."

        extractor = PdfTextExtractor(doc, ocr_backend=fake_ocr)
        paras = extractor.extract_text_for_pages(0, 0)
        assert len(called) == 1
        assert any("OCR extracted" in p for p in paras)

    def test_no_ocr_backend_skips_empty_page(self):
        PdfTextExtractor = _get_extractor_class()
        doc = fitz.open()
        doc.new_page(width=595, height=842)  # blank page
        extractor = PdfTextExtractor(doc, ocr_backend=None)
        paras = extractor.extract_text_for_pages(0, 0)
        assert paras == []


# ---------------------------------------------------------------------------
# render_cover_page
# ---------------------------------------------------------------------------

class TestRenderCoverPage:
    """render_cover_page returns PNG bytes from page 0."""

    def test_returns_bytes_and_mime_type(self):
        PdfTextExtractor = _get_extractor_class()
        doc = _make_doc_with_text(["Cover page content."])
        extractor = PdfTextExtractor(doc)
        data, mime = extractor.render_cover_page()
        assert isinstance(data, bytes)
        assert len(data) > 0
        assert mime == "image/png"

    def test_returns_none_for_empty_doc(self):
        PdfTextExtractor = _get_extractor_class()
        doc = fitz.open()  # zero pages
        extractor = PdfTextExtractor(doc)
        data, mime = extractor.render_cover_page()
        assert data is None
        assert mime is None
