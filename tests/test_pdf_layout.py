"""Layout adaptation preserves native evidence and accounts for every page."""

from __future__ import annotations

from dataclasses import replace

import pytest

from kenkui._pdf.layout import merge_layout
from kenkui.errors import SourceError
from kenkui.pdf_processing import PdfBlock
from test_pdf_steps import _document, _page


def test_layout_boundaries_preserve_native_text_and_native_archive() -> None:
    """A model supplies structure, but cannot silently rewrite native prose."""
    page = _page(
        1,
        (
            ("The original words remain", 60, 150, 500, 12),
            ("in their original spelling.", 60, 166, 500, 12),
        ),
    )
    raw = _document(page)
    layout = PdfBlock(
        "layout:1",
        1,
        "The original words remain in their original spelling.",
        box=(55, 145, 505, 185),
        origin="layout",
    )
    result = merge_layout(raw, (layout,), frozenset())
    assert result.narration[0].sources == ("1:0", "1:1")
    assert result.pages[0].lines == page.lines
    assert result.pages[0].blocks == page.blocks
    assert result.pages[0].layout_blocks == (layout,)
    assert result.narration[0].origin == "layout"


def test_omitted_or_changed_native_prose_falls_back_without_loss() -> None:
    """Layout omissions and substitutions do not become silent lost narration."""
    page = _page(1, (("Preserve this original sentence.", 60, 150, 500, 12),))
    layout = PdfBlock(
        "layout:1", 1, "Changed sentence.", box=(55, 145, 505, 185), origin="layout"
    )
    raw = _document(page)
    result = merge_layout(raw, (layout,), frozenset())
    assert result.narration == raw.narration
    assert result.issues[0].code == "layout_native_mismatch"
    assert merge_layout(raw, (), frozenset()).narration == raw.narration


def test_ocr_page_requires_real_text_instead_of_a_picture_placeholder() -> None:
    """A classified full-page image does not prove that scanned prose was read."""
    page = replace(_page(1, ()), image_coverage=1, disposition="unresolved")
    picture = PdfBlock(
        "layout:1", 1, "", role="picture", box=(0, 0, 600, 800), origin="layout"
    )
    with pytest.raises(SourceError):
        merge_layout(_document(page), (picture,), frozenset({1}))
    text = replace(picture, text="Recovered scanned prose.", role="text")
    result = merge_layout(_document(page), (text,), frozenset({1}))
    assert result.narration[0].text == text.text
    assert result.pages[0].disposition == "text"
    assert result.pages[0].lines == ()


def test_layout_does_not_merge_distinct_paragraphs() -> None:
    """Model block boundaries survive the optional native unwrapping step."""
    from kenkui.pdf_processing import reconstruct_paragraphs  # noqa: PLC0415

    page = _page(
        1,
        tuple(
            (text, 60, 150 + i * 16, 500, 12)
            for i, text in enumerate(
                (
                    "A complete paragraph with meaningful words.",
                    "Another complete paragraph with meaningful words.",
                    "A third complete paragraph with meaningful words.",
                )
            )
        ),
    )
    blocks = tuple(
        replace(b, id="layout:" + b.id, origin="layout") for b in page.blocks
    )
    document = merge_layout(_document(page), blocks, frozenset())
    assert reconstruct_paragraphs(document) == document


def test_docling_export_preserves_page_ranges_and_all_text() -> None:
    """Local batch page numbers and character spans map back to source pages."""
    from kenkui._pdf.docling import export_blocks  # noqa: PLC0415

    payload = {
        "pages": {"1": {"size": {"height": 800}}, "2": {"size": {"height": 800}}},
        "body": {"children": [{"$ref": "#/texts/0"}]},
        "texts": [
            {
                "self_ref": "#/texts/0",
                "label": "text",
                "text": "First. Second.",
                "prov": [
                    {
                        "page_no": 1,
                        "charspan": [0, 7],
                        "bbox": {
                            "l": 60,
                            "r": 500,
                            "t": 650,
                            "b": 600,
                            "coord_origin": "BOTTOMLEFT",
                        },
                    },
                    {
                        "page_no": 2,
                        "charspan": [7, 14],
                        "bbox": {
                            "l": 60,
                            "r": 500,
                            "t": 650,
                            "b": 600,
                            "coord_origin": "BOTTOMLEFT",
                        },
                    },
                ],
            }
        ],
    }
    blocks = export_blocks(payload, (5, 6))
    assert [(b.page, b.text) for b in blocks] == [(5, "First. "), (6, "Second.")]
    assert blocks[0].box == (60, 150, 500, 200)
    payload["texts"][0]["prov"][1]["charspan"] = [8, 14]  # type: ignore[index]
    with pytest.raises(SourceError):
        export_blocks(payload, (5, 6))


def test_docling_list_spans_use_original_numbered_text() -> None:
    """Docling spans can include a list marker removed from normalized text."""
    from kenkui._pdf.docling import export_blocks  # noqa: PLC0415

    original = "1. A complete numbered note."
    payload = {
        "pages": {"1": {"size": {"height": 800}}},
        "body": {"children": [{"$ref": "#/texts/0"}]},
        "texts": [
            {
                "self_ref": "#/texts/0",
                "label": "list_item",
                "text": "A complete numbered note.",
                "orig": original,
                "marker": "1.",
                "prov": [
                    {
                        "page_no": 1,
                        "charspan": [0, len(original)],
                        "bbox": {"l": 60, "r": 500, "t": 650, "b": 600},
                    }
                ],
            }
        ],
    }
    assert export_blocks(payload, (1,))[0].text == original
