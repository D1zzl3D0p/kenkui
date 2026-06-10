# PDF → EPUB via docling Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the fragile pymupdf-based PDF extraction pipeline with docling, which produces a human-inspectable EPUB as an intermediate artifact and delegates to the existing EpubReader.

**Architecture:** `PdfReader` lazily calls `PdfToEpubConverter.convert()` (new) when docling is installed; the converter runs docling on the PDF, synthesizes a valid EPUB via ebooklib (already a dependency), writes it alongside the source PDF for inspection, then delegates all `EbookReader` interface calls to the existing `EpubReader`. When docling is not installed, the existing pymupdf path is preserved as a fallback.

**Tech Stack:** `docling>=2.0` (optional extra), `pymupdf` (fallback/metadata), `ebooklib` (existing), `unittest.mock` (tests)

---

## File Map

| Action | Path |
|--------|------|
| Create | `src/kenkui/readers/_pdf_to_epub.py` |
| Rewrite | `src/kenkui/readers/pdf.py` |
| Modify | `src/kenkui/models/config.py` (add `pdf_force_ocr` field) |
| Modify | `src/kenkui/models/audio.py` (add `pdf_force_ocr` field) |
| Modify | `src/kenkui/parsing.py` (pass `force_ocr` in `_configure_reader`) |
| Modify | `pyproject.toml` (add `pdf-enhanced` optional dep) |
| Create | `tests/test_pdf_to_epub.py` |
| Modify | `tests/test_pdf_reader.py` (skip-when-docling markers) |

---

## Task 1: Add `pdf-enhanced` optional dependency

**Files:**
- Modify: `pyproject.toml:46-48`

- [ ] **Step 1: Add the optional extra**

  In `pyproject.toml`, replace:
  ```toml
  [project.optional-dependencies]
  dev = ["pytest>=7.0.0", "pytest-cov>=4.0.0"]
  ```
  With:
  ```toml
  [project.optional-dependencies]
  dev = ["pytest>=7.0.0", "pytest-cov>=4.0.0"]
  pdf-enhanced = ["docling>=2.0"]
  ```

- [ ] **Step 2: Commit**

  ```bash
  git add pyproject.toml
  git commit -m "chore: add pdf-enhanced optional dependency for docling"
  ```

---

## Task 2: Create `_pdf_to_epub.py` (TDD)

**Files:**
- Create: `src/kenkui/readers/_pdf_to_epub.py`
- Create: `tests/test_pdf_to_epub.py`

- [ ] **Step 1: Write the failing tests**

  Create `tests/test_pdf_to_epub.py`:

  ```python
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
  ```

- [ ] **Step 2: Run tests to confirm they fail**

  ```bash
  pytest tests/test_pdf_to_epub.py -v
  ```
  Expected: `ModuleNotFoundError: No module named 'kenkui.readers._pdf_to_epub'`

- [ ] **Step 3: Implement `_pdf_to_epub.py`**

  Create `src/kenkui/readers/_pdf_to_epub.py`:

  ```python
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

  logger = logging.getLogger(__name__)

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
  ```

- [ ] **Step 4: Run tests to confirm they pass**

  ```bash
  pytest tests/test_pdf_to_epub.py -v
  ```
  Expected: all tests PASS. If any `DocItemLabel` attribute lookup fails, the string-value fallback paths in `_extract_chapters` handle it.

- [ ] **Step 5: Commit**

  ```bash
  git add src/kenkui/readers/_pdf_to_epub.py tests/test_pdf_to_epub.py
  git commit -m "feat: add PdfToEpubConverter using docling for layout-aware PDF extraction"
  ```

---

## Task 3: Rewrite `pdf.py` to support docling path (TDD)

**Files:**
- Rewrite: `src/kenkui/readers/pdf.py`
- Modify: `tests/test_pdf_reader.py`

- [ ] **Step 1: Write failing tests for the docling delegation path**

  Add a new test class to `tests/test_pdf_to_epub.py` (append to existing file):

  ```python
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
  ```

- [ ] **Step 2: Run the new tests to confirm they fail**

  ```bash
  pytest tests/test_pdf_to_epub.py::TestPdfReaderDoclingPath -v
  ```
  Expected: FAIL — `_DOCLING_AVAILABLE` doesn't exist yet in `pdf.py`.

- [ ] **Step 3: Rewrite `src/kenkui/readers/pdf.py`**

  Replace the entire file with:

  ```python
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

  from ..chapter_classifier import ChapterClassifier
  from ..models import Chapter
  from . import EbookMetadata, EbookReader, Registry, TocEntry

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

      def _init_docling_reader(self):
          from .epub import EpubReader
          from ._pdf_to_epub import PdfToEpubConverter

          epub_path = PdfToEpubConverter().convert(self.filepath, force_ocr=self._force_ocr)
          with warnings.catch_warnings(record=True):
              warnings.simplefilter("always")
              self._epub_reader = EpubReader(epub_path, verbose=self.verbose)

      def _ensure_epub_reader(self):
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
              return self._ensure_epub_reader().get_chapters(min_text_len)
          return self._get_chapters_pymupdf(min_text_len)

      def get_cover(self) -> tuple[bytes | None, str | None]:
          if _DOCLING_AVAILABLE:
              return self._ensure_epub_reader().get_cover()
          return self._extractor.render_cover_page()

      def get_transcript_sections(self) -> list[PdfTranscriptSection]:
          """Available only on the pymupdf fallback path; empty on docling path."""
          if _DOCLING_AVAILABLE:
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
  ```

- [ ] **Step 4: Run tests**

  ```bash
  pytest tests/test_pdf_to_epub.py -v
  ```
  Expected: all tests PASS.

- [ ] **Step 5: Update `tests/test_pdf_reader.py` to mark pymupdf-only tests**

  The existing tests in `test_pdf_reader.py` all exercise the pymupdf fallback path. Add skip markers so they're skipped when docling is installed.

  At the top of `tests/test_pdf_reader.py`, after the existing imports, add:

  ```python
  from unittest.mock import patch

  # The existing test suite exercises the pymupdf fallback path.
  # When docling is installed, PdfReader uses the docling path instead.
  try:
      import docling as _docling  # noqa: F401
      _docling_installed = True
  except ImportError:
      _docling_installed = False

  pytestmark = pytest.mark.skipif(
      _docling_installed,
      reason="existing tests target pymupdf fallback; skip when docling is installed",
  )
  ```

  Note: `pytestmark` at module level applies the skipif to every test in the file.

- [ ] **Step 6: Run all PDF tests**

  ```bash
  pytest tests/test_pdf_reader.py tests/test_pdf_to_epub.py -v
  ```
  Expected: `test_pdf_reader.py` tests either all pass (no docling) or all skip (docling installed). `test_pdf_to_epub.py` tests all pass.

- [ ] **Step 7: Commit**

  ```bash
  git add src/kenkui/readers/pdf.py tests/test_pdf_reader.py
  git commit -m "feat: refactor PdfReader to delegate to EpubReader via docling conversion"
  ```

---

## Task 4: Add `pdf_force_ocr` to config models

**Files:**
- Modify: `src/kenkui/models/config.py:73-78`
- Modify: `src/kenkui/models/audio.py:47-52`

- [ ] **Step 1: Add field to `AppConfig` in `config.py`**

  After line 78 (`pdf_footer_zone_ratio: float = 0.0`), add:

  ```python
  pdf_force_ocr: bool = False
  ```

  The block should now read:
  ```python
  pdf_drop_code_blocks: bool = False
  pdf_drop_notes: bool = False
  pdf_drop_asides: bool = False
  pdf_drop_margin_notes: bool = True
  pdf_header_zone_ratio: float = 0.0
  pdf_footer_zone_ratio: float = 0.0
  pdf_force_ocr: bool = False
  ```

- [ ] **Step 2: Add field to `ProcessingConfig` in `audio.py`**

  After line 52 (`pdf_footer_zone_ratio: float = 0.0`), add:

  ```python
  pdf_force_ocr: bool = False
  ```

  The block should now read:
  ```python
  pdf_drop_code_blocks: bool = False
  pdf_drop_notes: bool = False
  pdf_drop_asides: bool = False
  pdf_drop_margin_notes: bool = True
  pdf_header_zone_ratio: float = 0.0
  pdf_footer_zone_ratio: float = 0.0
  pdf_force_ocr: bool = False
  ```

- [ ] **Step 3: Run existing tests to confirm no regression**

  ```bash
  pytest tests/ -v --ignore=tests/test_pdf_to_epub.py -x -q
  ```
  Expected: PASS (adding a field with a default does not break anything).

- [ ] **Step 4: Commit**

  ```bash
  git add src/kenkui/models/config.py src/kenkui/models/audio.py
  git commit -m "feat: add pdf_force_ocr config field for docling OCR control"
  ```

---

## Task 5: Wire `pdf_force_ocr` in `parsing.py`

**Files:**
- Modify: `src/kenkui/parsing.py:982-991`

- [ ] **Step 1: Update `_configure_reader`**

  Replace the existing `_configure_reader` method (lines 982-991) with:

  ```python
  def _configure_reader(self, reader) -> None:
      pdf_options: dict[str, bool | float] = {
          "drop_code_blocks": bool(getattr(self.cfg, "pdf_drop_code_blocks", False)),
          "drop_notes": bool(getattr(self.cfg, "pdf_drop_notes", False)),
          "drop_asides": bool(getattr(self.cfg, "pdf_drop_asides", False)),
          "drop_margin_notes": bool(getattr(self.cfg, "pdf_drop_margin_notes", True)),
          "header_zone_ratio": float(getattr(self.cfg, "pdf_header_zone_ratio", 0.0)),
          "footer_zone_ratio": float(getattr(self.cfg, "pdf_footer_zone_ratio", 0.0)),
          "force_ocr": bool(getattr(self.cfg, "pdf_force_ocr", False)),
      }
      reader.configure_pdf_extraction(pdf_options)
  ```

- [ ] **Step 2: Run the full test suite**

  ```bash
  pytest tests/ -v -q
  ```
  Expected: all tests PASS.

- [ ] **Step 3: Commit**

  ```bash
  git add src/kenkui/parsing.py
  git commit -m "feat: pass pdf_force_ocr to configure_pdf_extraction for docling OCR control"
  ```

---

## Verification Checklist

After all tasks are complete:

- [ ] Install docling: `pip install 'kenkui[pdf-enhanced]'`
- [ ] Run full test suite: `pytest tests/ -v` — confirm `test_pdf_reader.py` tests skip (docling path active), `test_pdf_to_epub.py` all pass
- [ ] Take a text-based PDF that previously had layout issues, run: `kenkui convert book.pdf --dry-run` (or equivalent CLI). Open the generated `book.epub` in Books/Calibre — verify chapter structure is correct with no header/footer bleed
- [ ] For OCR test: if you have a scanned PDF, add `pdf_force_ocr = true` to config and re-run. Inspect the EPUB — OCR text should be legible
- [ ] Confirm cache: run the conversion twice. Second run log should show `PDF→EPUB cache hit`
- [ ] Uninstall docling (`pip uninstall docling`) and re-run `pytest tests/test_pdf_reader.py -v` — pymupdf tests should run and pass again
