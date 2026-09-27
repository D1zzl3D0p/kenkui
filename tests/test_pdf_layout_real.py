"""Opt-in layout/OCR acceptance with provisioned local assets and no downloads."""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

import pytest

import kenkui as kk
from kenkui.pdf_processing import PdfLayoutOptions
from pdf_helpers import make_pdf

if TYPE_CHECKING:
    from pathlib import Path

pytestmark = [
    pytest.mark.pdf_layout,
    pytest.mark.skipif(
        os.environ.get("KENKUI_RUN_PDF_LAYOUT") != "1",
        reason="opt-in local PDF models required",
    ),
]
_LINES = (
    "A quiet reader follows the original prose.",
    "The second sentence remains on the page.",
    "Another complete paragraph ends the story.",
)


def _digital(path: Path) -> Path:
    stream = b" ".join(
        f"BT /F1 14 Tf 60 {640 - i * 32} Td ({text}) Tj ET".encode()
        for i, text in enumerate(_LINES)
    )
    return make_pdf(path, ("",), streams=(stream,))


@pytest.mark.parametrize("kind", ["digital", "scanned", "mixed"])
def test_real_layout_and_ocr_keep_all_fixture_prose(tmp_path: Path, kind: str) -> None:
    """Digital, scanned, and mixed sources complete through the public worker path."""
    pdfium = pytest.importorskip("pypdfium2")
    pytest.importorskip("docling")
    assets = os.environ.get("KENKUI_PDF_MODELS")
    if not assets:
        pytest.skip("KENKUI_PDF_MODELS must name pre-provisioned local assets")
    native = _digital(tmp_path / "digital.pdf")
    source = native
    if kind != "digital":
        scan = tmp_path / "scanned.pdf"
        with pdfium.PdfDocument(native) as document:
            page = document[0]
            bitmap = page.render(scale=2)
            bitmap.to_pil().convert("RGB").save(scan, "PDF", resolution=144)
            bitmap.close()
            page.close()
        source = scan
        if kind == "mixed":
            source = tmp_path / "mixed.pdf"
            with pdfium.PdfDocument.new() as merged:
                with pdfium.PdfDocument(native) as first:
                    merged.import_pages(first)
                with pdfium.PdfDocument(scan) as second:
                    merged.import_pages(second)
                merged.save(source)
    prepared = (
        kk.pdf(source)
        .pdf_processing(
            layout=PdfLayoutOptions(
                artifacts_path=assets,
                timeout_seconds=240,
                max_memory_mb=3000,
                batch_pages=1,
            )
        )
        .prepare()
    )
    text = prepared.inspect().chapters[0].text
    for line in _LINES:
        assert line in text
    expected_pages = 2 if kind == "mixed" else 1
    assert len(prepared.pdf_report().pages) == expected_pages
    assert all(page.disposition == "text" for page in prepared.pdf_report().pages)
    assert all(page.layout_blocks for page in prepared.pdf_report().pages)
