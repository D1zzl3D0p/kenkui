"""
PDF text extraction and layout analysis.

Internal module — used only by PdfReader. Not registered with the reader Registry.
"""

from __future__ import annotations

import logging
import os
import re
from collections import Counter
from collections.abc import Callable
from functools import lru_cache

import fitz

from . import TocEntry

logger = logging.getLogger(__name__)

# Unicode ligature normalization map
_LIGATURES: dict[str, str] = {
    "ﬀ": "ff",
    "ﬁ": "fi",
    "ﬂ": "fl",
    "ﬃ": "ffi",
    "ﬄ": "ffl",
    "ﬅ": "ft",
    "ﬆ": "st",
}


class PdfTextExtractor:
    """Handles all layout analysis and text extraction for a fitz.Document.

    Env-var configuration (read at init time):
        KENKUI_PDF_PAGE_CHUNK_SIZE    Pages per chapter in page-range fallback (default 20)
        KENKUI_PDF_HEADING_SIZE_RATIO Font size multiplier to classify text as a heading (default 1.2)
        KENKUI_PDF_FOOTER_THRESHOLD   Fraction of pages a line must appear on to be noise (default 0.80)
    """

    def __init__(
        self,
        doc: fitz.Document,
        ocr_backend: Callable[[fitz.Page], str] | None = None,
        verbose: bool = False,
    ) -> None:
        self._doc = doc
        self._ocr_backend = ocr_backend
        self._verbose = verbose

        self._page_chunk_size = int(os.environ.get("KENKUI_PDF_PAGE_CHUNK_SIZE", "20"))
        self._heading_size_ratio = float(os.environ.get("KENKUI_PDF_HEADING_SIZE_RATIO", "1.2"))
        self._footer_threshold = float(os.environ.get("KENKUI_PDF_FOOTER_THRESHOLD", "0.80"))

        self._strip_margin_notes: bool = (
            os.environ.get("KENKUI_PDF_STRIP_MARGIN_NOTES", "true").lower() == "true"
        )
        self._header_zone_ratio: float = float(os.environ.get("KENKUI_PDF_HEADER_ZONE", "0.0"))
        self._footer_zone_ratio: float = float(os.environ.get("KENKUI_PDF_FOOTER_ZONE", "0.0"))

        self._noise_lines: set[str] | None = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def extract_toc(self) -> list[TocEntry]:
        """Return TOC entries via bookmarks → heading heuristic → page-range fallback."""
        toc = self._toc_from_bookmarks()
        if toc:
            return toc
        toc = self._heading_scan()
        if toc:
            return toc
        return self._page_range_fallback()

    def extract_text_for_pages(self, start: int, end: int) -> list[str]:
        """Extract clean paragraphs from pages [start, end] (inclusive, 0-indexed).

        Uses block-level extraction so each PDF text block becomes a paragraph
        candidate — more reliable than splitting on double-newlines since PDFs
        rarely emit those in raw text output.
        """
        noise = self._detect_headers_footers()
        paragraphs: list[str] = []

        for page_num in range(start, min(end + 1, len(self._doc))):
            page = self._doc[page_num]
            # sort=True preserves reading order across multi-column layouts
            blocks = page.get_text("blocks", sort=True)
            text_blocks = [b for b in blocks if b[6] == 0]  # type 0 = text, 1 = image

            if not text_blocks:
                if self._ocr_backend is not None:
                    raw = self._ocr_backend(page)
                    parts = [p.strip() for p in re.split(r"\n\s*\n", raw) if p.strip()]
                    paragraphs.extend(parts)
                else:
                    if self._verbose:
                        logger.warning("Page %d has no text layer; skipping", page_num)
                continue

            # Collect table bounding boxes to skip table blocks
            table_bboxes = self._get_table_bboxes(page)

            for block in text_blocks:
                if _bbox_overlaps_any(block[:4], table_bboxes):
                    continue

                # Strip noise lines within the block
                lines = [ln for ln in block[4].splitlines() if ln.strip() not in noise]
                text = self._clean_text("\n".join(lines)).strip()
                if text:
                    paragraphs.append(text)
                elif self._verbose:
                    logger.debug(
                        "PDF page %d skipped empty block at %s",
                        page_num + 1,
                        tuple(round(v, 2) for v in block[:4]),
                    )

            if self._verbose and paragraphs:
                logger.debug("PDF page %d yielded %d paragraph(s)", page_num + 1, len(paragraphs))

        return paragraphs

    def render_cover_page(self) -> tuple[bytes | None, str | None]:
        """Render page 0 as PNG at 150 DPI. Returns (None, None) for zero-page docs."""
        if len(self._doc) == 0:
            return None, None
        page = self._doc[0]
        mat = fitz.Matrix(150 / 72, 150 / 72)
        pix = page.get_pixmap(matrix=mat)
        return pix.tobytes("png"), "image/png"

    def configure(self, options: dict[str, bool | float]) -> None:
        """Apply per-job overrides on top of env-var defaults."""
        if "drop_margin_notes" in options:
            self._strip_margin_notes = bool(options["drop_margin_notes"])
        if "header_zone_ratio" in options:
            self._header_zone_ratio = float(options["header_zone_ratio"])
        if "footer_zone_ratio" in options:
            self._footer_zone_ratio = float(options["footer_zone_ratio"])

    # ------------------------------------------------------------------
    # Text cleaning
    # ------------------------------------------------------------------

    def _clean_text(self, text: str) -> str:
        """Normalize ligatures, dehyphenate line breaks, strip lone page numbers."""
        if not text:
            return text

        for src, dst in _LIGATURES.items():
            text = text.replace(src, dst)

        # Rejoin words split across lines with a hyphen: "incon-\nvenient" → "inconvenient"
        text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)

        # Remove lines that are nothing but a page number
        lines = [ln for ln in text.split("\n") if not re.fullmatch(r"\s*\d+\s*", ln)]
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Header / footer detection
    # ------------------------------------------------------------------

    def _detect_headers_footers(self) -> set[str]:
        """Return stripped line strings appearing on >= threshold fraction of pages."""
        if self._noise_lines is not None:
            return self._noise_lines

        n_pages = len(self._doc)
        if n_pages == 0:
            self._noise_lines = set()
            return self._noise_lines

        line_counts: Counter[str] = Counter()
        for page in self._doc:
            seen = {ln.strip() for ln in page.get_text().splitlines() if ln.strip()}
            line_counts.update(seen)

        min_occurrences = self._footer_threshold * n_pages
        self._noise_lines = {
            # Also require count > 1 so single-page docs are never noise-filtered
            line for line, count in line_counts.items()
            if count >= min_occurrences and count > 1
        }
        return self._noise_lines

    # ------------------------------------------------------------------
    # TOC extraction strategies
    # ------------------------------------------------------------------

    def _toc_from_bookmarks(self) -> list[TocEntry]:
        raw = self._doc.get_toc()
        if not raw:
            return []

        n_pages = len(self._doc)
        entries: list[TocEntry] = []
        for i, item in enumerate(raw):
            level, title = item[0], item[1]
            start_page = max(0, min(item[2] - 1, n_pages - 1))  # 1-indexed → 0-indexed

            if i + 1 < len(raw):
                end_page = raw[i + 1][2] - 2
            else:
                end_page = n_pages - 1

            entries.append(
                TocEntry(title=title, href=f"page:{start_page}-{end_page}", level=level - 1)
            )

        return entries

    def _heading_scan(self) -> list[TocEntry]:
        """Detect chapter boundaries by font-size contrast. Returns [] if inconclusive."""
        spans_by_page: list[list[tuple[float, str]]] = []
        for page in self._doc:
            page_spans: list[tuple[float, str]] = []
            for block in page.get_text("dict").get("blocks", []):
                if block.get("type") != 0:
                    continue
                for line in block.get("lines", []):
                    for span in line.get("spans", []):
                        txt = span.get("text", "").strip()
                        size = float(span.get("size", 0))
                        if txt and size > 0:
                            page_spans.append((size, txt))
            spans_by_page.append(page_spans)

        all_sizes = [size for ps in spans_by_page for size, _ in ps]
        if not all_sizes:
            return []

        size_counts = Counter(round(s, 1) for s in all_sizes)
        max_count = max(size_counts.values())
        # Tie-break toward smaller size to prefer body text over headings
        body_size = min(s for s, c in size_counts.items() if c == max_count)
        heading_threshold = body_size * self._heading_size_ratio

        found: list[tuple[int, str]] = []
        for page_num, page_spans in enumerate(spans_by_page):
            for size, txt in page_spans:
                if size >= heading_threshold and len(txt) > 2:
                    found.append((page_num, txt))
                    break  # one heading per page

        if not found:
            return []

        n_pages = len(self._doc)
        entries: list[TocEntry] = []
        for i, (page_num, title) in enumerate(found):
            end_page = found[i + 1][0] - 1 if i + 1 < len(found) else n_pages - 1
            entries.append(
                TocEntry(title=title, href=f"page:{page_num}-{end_page}", level=0)
            )

        return entries

    def _page_range_fallback(self) -> list[TocEntry]:
        """Split document into fixed-size page-range chunks."""
        n_pages = len(self._doc)
        if n_pages == 0:
            return []

        chunk = self._page_chunk_size
        entries: list[TocEntry] = []
        for start in range(0, n_pages, chunk):
            end = min(start + chunk - 1, n_pages - 1)
            part_num = start // chunk + 1
            entries.append(
                TocEntry(title=f"Part {part_num}", href=f"page:{start}-{end}", level=0)
            )

        return entries

    def _get_table_bboxes(self, page: fitz.Page) -> list[tuple[float, float, float, float]]:
        """Return bounding boxes of detected tables on a page."""
        layout_bboxes = self._get_layout_table_bboxes(page)
        if layout_bboxes:
            return layout_bboxes
        try:
            return [tuple(t.bbox) for t in page.find_tables().tables]  # type: ignore[misc]
        except Exception:
            return []

    def _get_layout_table_bboxes(self, page: fitz.Page) -> list[tuple[float, float, float, float]]:
        """Return table bounding boxes from pymupdf_layout when available."""
        detector = _get_pymupdf_layout_table_detector()
        if detector is None:
            return []
        try:
            result = detector(page)
        except Exception:
            return []
        return _coerce_table_bboxes(result)


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------

def _bbox_overlaps_any(
    bbox: tuple[float, float, float, float],
    others: list[tuple[float, float, float, float]],
) -> bool:
    """Return True if *bbox* has non-trivial overlap with any bbox in *others*."""
    x0, y0, x1, y1 = bbox
    for ox0, oy0, ox1, oy1 in others:
        ix0 = max(x0, ox0)
        iy0 = max(y0, oy0)
        ix1 = min(x1, ox1)
        iy1 = min(y1, oy1)
        if ix1 > ix0 and iy1 > iy0:
            return True
    return False


@lru_cache(maxsize=1)
def _get_pymupdf_layout_table_detector():
    """Return a best-effort table detector from pymupdf_layout, if installed."""
    try:
        import pymupdf_layout  # type: ignore[import-not-found]
    except Exception:
        return None

    candidate_names = (
        "find_tables",
        "extract_tables",
        "get_tables",
        "analyze_page",
        "analyze",
        "page_layout",
    )

    for name in candidate_names:
        detector = getattr(pymupdf_layout, name, None)
        if callable(detector):
            return detector

    factory_names = (
        "LayoutAnalyzer",
        "PageLayoutAnalyzer",
        "PageAnalyzer",
        "DocumentLayout",
        "Layout",
        "Analyzer",
    )
    for name in factory_names:
        factory = getattr(pymupdf_layout, name, None)
        if factory is None:
            continue
        try:
            instance = factory()
        except TypeError:
            continue
        for method_name in candidate_names:
            detector = getattr(instance, method_name, None)
            if callable(detector):
                return detector

    return None


def _coerce_table_bboxes(result) -> list[tuple[float, float, float, float]]:
    """Normalize a layout result into plain bounding boxes."""
    if result is None:
        return []

    if isinstance(result, dict):
        for key in ("tables", "table_bboxes", "bboxes", "boxes", "regions"):
            if key in result:
                return _coerce_table_bboxes(result[key])
        return []

    bbox = getattr(result, "bbox", None)
    if bbox is not None:
        try:
            return [tuple(float(v) for v in bbox)]
        except Exception:
            return []

    tables = getattr(result, "tables", None)
    if tables is not None:
        return _coerce_table_bboxes(tables)

    if isinstance(result, (list, tuple, set)):
        boxes: list[tuple[float, float, float, float]] = []
        for item in result:
            boxes.extend(_coerce_table_bboxes(item))
        return boxes

    try:
        values = tuple(float(v) for v in result)
    except Exception:
        return []
    if len(values) == 4:
        return [values]
    return []
