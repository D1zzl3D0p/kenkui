# PDF → EPUB via docling: Design Spec

**Date:** 2026-06-09  
**Status:** Approved

## Context

PDF processing in kenkui has required a growing series of per-PDF workarounds — zone ratios, margin note filtering, heading size tuning, code block detection — because raw `pymupdf` text extraction is fragile across the wide variety of PDFs encountered in practice (published ebooks, academic/multi-column layouts, and scanned books all behave differently). The feedback loop is slow: problems only become visible in the final M4B or by manually reading raw transcript files, and a full audio render takes ~1 hour per book.

This design replaces the bespoke PDF extraction pipeline with a structured document conversion step using docling, producing a human-inspectable EPUB artifact before any audio is generated.

## Goals

1. Handle text PDFs, complex academic layouts, and scanned (OCR-required) PDFs uniformly
2. Produce a human-inspectable EPUB artifact as a fast intermediate checkpoint
3. Delegate all subsequent parsing to the existing, battle-tested `EpubReader`
4. Require no new user-facing configuration beyond `pdf_force_ocr`

## Architecture

```
PDF → docling DocumentConverter → DoclingDocument → ebooklib → EPUB (on disk)
                                                                     ↓
                                                                EpubReader (existing, unchanged)
                                                                     ↓
                                                               Chapter[] → AudioBuilder
```

The EPUB artifact is written alongside the source PDF (`MyBook.pdf` → `MyBook.epub`). Users can open it in any EPUB viewer (Books, Calibre) to verify chapter structure and content quality before committing to a full audio run — collapsing the feedback loop from ~1 hour down to seconds.

If the EPUB already exists and is newer than the PDF, conversion is skipped (mtime-based cache hit).

## Components

### `src/kenkui/readers/_pdf_to_epub.py` (new)

`PdfToEpubConverter` with one public method:

```python
def convert(pdf_path: Path, force_ocr: bool = False) -> Path
```

Internal flow:
1. Check cache: if `{stem}.epub` exists and `mtime > pdf mtime`, return cached path immediately
2. Configure docling `PdfPipelineOptions` (`do_ocr` based on `force_ocr`; docling auto-detects scanned PDFs otherwise)
3. Run `DocumentConverter().convert(pdf_path)`
4. Iterate `doc.iterate_items()`:
   - `SectionHeaderItem` → chapter boundary (new chapter)
   - `TextItem` / `ParagraphItem` → paragraph content for current chapter
   - `TableItem`, `FigureItem`, `FootnoteItem` → skip entirely
5. If no section headers found, treat entire document as one chapter using the book title
6. Assemble chapters into `ebooklib.EpubBook`, write `{stem}.epub` next to source PDF
7. Return EPUB path

### `src/kenkui/readers/pdf.py` (modified)

`PdfReader` becomes an adapter with a soft docling import at module level:

```python
try:
    from docling.document_converter import DocumentConverter  # noqa: F401
    _DOCLING_AVAILABLE = True
except ImportError:
    _DOCLING_AVAILABLE = False
```

`__init__` behavior:
- If docling available: call `PdfToEpubConverter.convert()` → store `EpubReader` instance, delegate all interface methods to it
- If not available: warn once and fall back to existing `PdfTextExtractor` path (no silent failures)

`configure_pdf_extraction()` on the docling path: only `force_ocr` is forwarded; all layout options (`header_zone_ratio`, `margin_notes`, etc.) become no-ops (docling handles layout correctly at source).

### `pyproject.toml` (modified)

```toml
[project.optional-dependencies]
dev = ["pytest>=7.0.0", "pytest-cov>=4.0.0"]
pdf-enhanced = ["docling>=2.0"]
```

Users who want the improved pipeline install with `pip install kenkui[pdf-enhanced]`.

### Config models (modified)

`src/kenkui/models/config.py` and `src/kenkui/models/audio.py`: add one new field:

```python
pdf_force_ocr: bool = False
```

All existing PDF layout config fields (`pdf_drop_code_blocks`, `pdf_drop_notes`, `pdf_drop_asides`, `pdf_header_zone_ratio`, `pdf_footer_zone_ratio`, `pdf_drop_margin_notes`) are preserved in the model schema for TOML compatibility but become no-ops on the docling path.

### `src/kenkui/parsing.py` (modified, minimal)

`_configure_reader()` passes `force_ocr=bool(getattr(self.cfg, "pdf_force_ocr", False))` to `configure_pdf_extraction()`.

## Configuration Surface

| Option | Type | Default | Purpose |
|--------|------|---------|---------|
| `pdf_force_ocr` | bool | `False` | Force OCR even when a text layer exists — for PDFs where the text layer is corrupt or garbage |

EPUB artifact location is always `{pdf_stem}.epub` next to the source PDF. No configuration needed.

## What docling Handles (No Config Needed)

Replacing the following per-PDF workarounds that currently require manual tuning:

| Problem | Old approach | docling approach |
|---------|-------------|-----------------|
| Multi-column layout | `pdf_drop_margin_notes`, zone ratios | Layout detection at source |
| Headers/footers bleeding in | `pdf_header_zone_ratio`, `pdf_footer_zone_ratio` | Page structure analysis |
| Tables narrated as prose | Partial block exclusion | `TableItem` skipped entirely |
| Footnotes/asides narrated | `pdf_drop_notes`, `pdf_drop_asides` | `FootnoteItem` skipped entirely |
| Code blocks narrated | `pdf_drop_code_blocks` | `CodeItem` skipped entirely |
| Scanned/image PDFs | Broken (no OCR) | Auto-detected, Tesseract OCR |
| Chapter detection failures | Heading size heuristic, page chunks | `SectionHeaderItem` hierarchy |

## Testing

`tests/test_pdf_to_epub.py` (new):
- Cache hit: EPUB newer than PDF → no re-conversion, returns cached path
- Cache miss: stale/missing EPUB → calls docling, writes EPUB
- Chapter extraction: mock docling output with known section headers → verify EPUB chapter count and titles
- No section headers: single-chapter EPUB with book title
- `force_ocr` forwarded to `PdfPipelineOptions.do_ocr`

`tests/test_pdf_reader.py` (updated):
- Docling-path tests: `pytest.importorskip("docling")`
- Pymupdf fallback tests: `@pytest.mark.skipif(_DOCLING_AVAILABLE, reason="docling installed")`

## Verification Checklist

1. `pip install kenkui[pdf-enhanced]` installs cleanly
2. `pytest tests/test_pdf_to_epub.py` — all pass
3. `pytest tests/test_pdf_reader.py` — both paths pass
4. Open generated `.epub` in Books/Calibre — chapter structure correct, no header/footer bleed
5. Scanned PDF with `pdf_force_ocr=True` — OCR text is readable
6. Complex multi-column PDF — columns not interleaved
7. Full kenkui run on a previously-problematic PDF — clean audio output
8. Second run — logs "cache hit", skips conversion
