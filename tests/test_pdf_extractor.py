"""Unit tests for PdfTextExtractor internals."""

from __future__ import annotations

import pytest

fitz = pytest.importorskip("fitz", reason="pymupdf not installed")


# ---------------------------------------------------------------------------
# Helpers: build minimal in-memory PDFs
# ---------------------------------------------------------------------------

def _make_doc_with_text(pages: list[str]) -> fitz.Document:
    """Return a fitz.Document with one page per string in *pages*."""
    doc = fitz.open()
    for text in pages:
        page = doc.new_page(width=595, height=842)
        page.insert_text((50, 100), text, fontsize=12)
    return doc


def _make_doc_with_bookmarks(chapters: list[tuple[str, int]]) -> fitz.Document:
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
) -> fitz.Document:
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

    def test_same_page_entries_give_chapter_empty_range(self):
        """Parent chapter must not duplicate content with its first child."""
        PdfTextExtractor = _get_extractor_class()
        doc = _make_doc_with_bookmarks([
            ("Chapter 1", 1),
            ("1.1 Introduction", 1),
            ("1.2 Details", 3),
        ])
        extractor = PdfTextExtractor(doc)
        toc = extractor.extract_toc()

        assert len(toc) == 3

        def _parse(href: str) -> tuple[int, int]:
            _, page_range = href.split(":", 1)
            start_s, end_s = page_range.split("-", 1)
            return int(start_s), int(end_s)

        ch1_start, ch1_end = _parse(toc[0].href)
        assert ch1_end < ch1_start, "Chapter 1 should have an empty range (end < start)"

        intro_start, intro_end = _parse(toc[1].href)
        assert intro_start == 0
        assert intro_end == 1


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

    def test_verbose_mode_does_not_log_paragraph_text(self, caplog):
        PdfTextExtractor = _get_extractor_class()
        doc = _make_doc_with_text(["First paragraph.\n\nSecond paragraph."])
        extractor = PdfTextExtractor(doc, verbose=True)

        with caplog.at_level("DEBUG", logger="kenkui.readers._pdf_extract"):
            paras = extractor.extract_text_for_pages(0, 0)

        assert len(paras) >= 1
        assert not any("First paragraph" in record.message for record in caplog.records)
        assert not any("Second paragraph" in record.message for record in caplog.records)

    def test_prefers_pymupdf_layout_table_detector_when_available(self, monkeypatch, request):
        import sys
        import types

        from kenkui.readers import _pdf_extract as pdf_extract_mod

        # Clear cache before the test and register a finalizer to clear it after,
        # so the fake detector does not leak into subsequent tests.
        pdf_extract_mod._get_pymupdf_layout_table_detector.cache_clear()
        request.addfinalizer(pdf_extract_mod._get_pymupdf_layout_table_detector.cache_clear)

        PdfTextExtractor = _get_extractor_class()
        doc = _make_doc_with_text(["Table block should be excluded.", "Body text remains."])
        extractor = PdfTextExtractor(doc)

        fake_module = types.SimpleNamespace(
            find_tables=lambda page: types.SimpleNamespace(
                tables=[
                    types.SimpleNamespace(bbox=(0.0, 0.0, 100.0, 100.0)),
                ]
            )
        )
        monkeypatch.setitem(sys.modules, "pymupdf_layout", fake_module)

        pdf_extract_mod._get_pymupdf_layout_table_detector.cache_clear()
        boxes = extractor._get_table_bboxes(doc[0])

        assert boxes == [(0.0, 0.0, 100.0, 100.0)]


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


# ---------------------------------------------------------------------------
# PdfTextExtractor.configure()
# ---------------------------------------------------------------------------

class TestPdfTextExtractorConfigure:
    """Tests for PdfTextExtractor.configure()."""

    def test_configure_sets_strip_margin_notes(self):
        PdfTextExtractor = _get_extractor_class()
        doc = _make_doc_with_text(["Text."])
        extractor = PdfTextExtractor(doc)
        extractor.configure({"drop_margin_notes": False})
        assert extractor._strip_margin_notes is False

    def test_configure_sets_zone_ratios(self):
        PdfTextExtractor = _get_extractor_class()
        doc = _make_doc_with_text(["Text."])
        extractor = PdfTextExtractor(doc)
        extractor.configure({"header_zone_ratio": 0.12, "footer_zone_ratio": 0.07})
        assert abs(extractor._header_zone_ratio - 0.12) < 1e-9
        assert abs(extractor._footer_zone_ratio - 0.07) < 1e-9

    def test_configure_ignores_unknown_keys(self):
        PdfTextExtractor = _get_extractor_class()
        doc = _make_doc_with_text(["Text."])
        extractor = PdfTextExtractor(doc)
        extractor.configure({"unknown_key": True, "drop_margin_notes": False})
        assert extractor._strip_margin_notes is False

    def test_strip_margin_notes_default_is_true(self, monkeypatch):
        monkeypatch.delenv("KENKUI_PDF_STRIP_MARGIN_NOTES", raising=False)
        PdfTextExtractor = _get_extractor_class()
        doc = _make_doc_with_text(["Text."])
        extractor = PdfTextExtractor(doc)
        assert extractor._strip_margin_notes is True

    def test_env_var_can_disable_strip_margin_notes(self, monkeypatch):
        monkeypatch.setenv("KENKUI_PDF_STRIP_MARGIN_NOTES", "false")
        PdfTextExtractor = _get_extractor_class()
        doc = _make_doc_with_text(["Text."])
        extractor = PdfTextExtractor(doc)
        assert extractor._strip_margin_notes is False

    def test_zone_defaults_are_zero(self, monkeypatch):
        monkeypatch.delenv("KENKUI_PDF_HEADER_ZONE", raising=False)
        monkeypatch.delenv("KENKUI_PDF_FOOTER_ZONE", raising=False)
        PdfTextExtractor = _get_extractor_class()
        doc = _make_doc_with_text(["Text."])
        extractor = PdfTextExtractor(doc)
        assert extractor._header_zone_ratio == 0.0
        assert extractor._footer_zone_ratio == 0.0


# ---------------------------------------------------------------------------
# Span-level extraction / margin note filtering
# ---------------------------------------------------------------------------

class TestStripMarginNotes:
    """Span-level extraction correctly identifies and removes margin-column text."""

    def test_body_column_x_range_from_dominant_line_starts(self, monkeypatch):
        monkeypatch.delenv("KENKUI_PDF_STRIP_MARGIN_NOTES", raising=False)
        PdfTextExtractor = _get_extractor_class()
        doc = fitz.open()
        page = doc.new_page(width=540, height=666)
        for i in range(20):
            page.insert_text((70, 100 + i * 20), f"Body line {i}.", fontsize=10)
        page.insert_text((420, 200), "Margin note line one.", fontsize=9)
        page.insert_text((420, 220), "Margin note line two.", fontsize=9)

        extractor = PdfTextExtractor(doc)
        x0, x1 = extractor._get_body_x_range(page)

        assert abs(x0 - 70) <= 15, f"body x0 should be ~70, got {x0}"
        assert x1 < 420, f"body x1 should be less than margin at 420, got {x1}"

    def test_span_level_excludes_right_margin_blocks(self, monkeypatch):
        monkeypatch.delenv("KENKUI_PDF_STRIP_MARGIN_NOTES", raising=False)
        PdfTextExtractor = _get_extractor_class()
        doc = fitz.open()
        page = doc.new_page(width=540, height=666)
        for i in range(20):
            page.insert_text((70, 100 + i * 20), f"Body text line {i}.", fontsize=10)
        page.insert_text((420, 200), "Right margin note content.", fontsize=9)

        extractor = PdfTextExtractor(doc)
        paras = extractor.extract_text_for_pages(0, 0)
        combined = " ".join(paras)

        assert "Body text line" in combined
        assert "Right margin note content" not in combined

    def test_span_level_excludes_left_margin_blocks(self, monkeypatch):
        monkeypatch.delenv("KENKUI_PDF_STRIP_MARGIN_NOTES", raising=False)
        PdfTextExtractor = _get_extractor_class()
        doc = fitz.open()
        page = doc.new_page(width=540, height=666)
        for i in range(20):
            page.insert_text((200, 100 + i * 20), f"Body text line {i}.", fontsize=10)
        page.insert_text((30, 200), "Left margin note content.", fontsize=9)

        extractor = PdfTextExtractor(doc)
        paras = extractor.extract_text_for_pages(0, 0)
        combined = " ".join(paras)

        assert "Body text line" in combined
        assert "Left margin note content" not in combined

    def test_configure_drop_margin_notes_false_includes_margin(self, monkeypatch):
        monkeypatch.delenv("KENKUI_PDF_STRIP_MARGIN_NOTES", raising=False)
        PdfTextExtractor = _get_extractor_class()
        doc = fitz.open()
        page = doc.new_page(width=540, height=666)
        for i in range(20):
            page.insert_text((70, 100 + i * 20), f"Body text line {i}.", fontsize=10)
        page.insert_text((420, 200), "Margin note content.", fontsize=9)

        extractor = PdfTextExtractor(doc)
        extractor.configure({"drop_margin_notes": False})
        paras = extractor.extract_text_for_pages(0, 0)
        combined = " ".join(paras)

        assert "Body text line" in combined
        assert "Margin note content" in combined


# ---------------------------------------------------------------------------
# Header / footer zone filtering
# ---------------------------------------------------------------------------

class TestZoneFiltering:
    """header_zone_ratio and footer_zone_ratio skip blocks in the top/bottom zones."""

    def _make_page_with_zones(self):
        doc = fitz.open()
        page = doc.new_page(width=540, height=666)
        ph = 666.0
        page.insert_text((50, ph * 0.05), "Header zone text.", fontsize=10)
        page.insert_text((50, ph * 0.50), "Body text in middle.", fontsize=10)
        page.insert_text((50, ph * 0.96), "Footer zone text.", fontsize=10)
        return doc

    def test_header_zone_skips_top_block(self, monkeypatch):
        monkeypatch.delenv("KENKUI_PDF_STRIP_MARGIN_NOTES", raising=False)
        PdfTextExtractor = _get_extractor_class()
        doc = self._make_page_with_zones()
        extractor = PdfTextExtractor(doc)
        extractor.configure({"drop_margin_notes": False, "header_zone_ratio": 0.10})
        paras = extractor.extract_text_for_pages(0, 0)
        combined = " ".join(paras)
        assert "Body text in middle" in combined
        assert "Header zone text" not in combined

    def test_footer_zone_skips_bottom_block(self, monkeypatch):
        monkeypatch.delenv("KENKUI_PDF_STRIP_MARGIN_NOTES", raising=False)
        PdfTextExtractor = _get_extractor_class()
        doc = self._make_page_with_zones()
        extractor = PdfTextExtractor(doc)
        extractor.configure({"drop_margin_notes": False, "footer_zone_ratio": 0.10})
        paras = extractor.extract_text_for_pages(0, 0)
        combined = " ".join(paras)
        assert "Body text in middle" in combined
        assert "Footer zone text" not in combined

    def test_zero_ratio_disables_zone_filtering(self, monkeypatch):
        monkeypatch.delenv("KENKUI_PDF_STRIP_MARGIN_NOTES", raising=False)
        PdfTextExtractor = _get_extractor_class()
        doc = self._make_page_with_zones()
        extractor = PdfTextExtractor(doc)
        extractor.configure({
            "drop_margin_notes": False,
            "header_zone_ratio": 0.0,
            "footer_zone_ratio": 0.0,
        })
        paras = extractor.extract_text_for_pages(0, 0)
        combined = " ".join(paras)
        assert "Header zone text" in combined
        assert "Body text in middle" in combined
        assert "Footer zone text" in combined

    def test_both_zones_active_via_configure(self, monkeypatch):
        monkeypatch.delenv("KENKUI_PDF_STRIP_MARGIN_NOTES", raising=False)
        PdfTextExtractor = _get_extractor_class()
        doc = self._make_page_with_zones()
        extractor = PdfTextExtractor(doc)
        extractor.configure({
            "drop_margin_notes": False,
            "header_zone_ratio": 0.10,
            "footer_zone_ratio": 0.10,
        })
        paras = extractor.extract_text_for_pages(0, 0)
        combined = " ".join(paras)
        assert "Body text in middle" in combined
        assert "Header zone text" not in combined
        assert "Footer zone text" not in combined
