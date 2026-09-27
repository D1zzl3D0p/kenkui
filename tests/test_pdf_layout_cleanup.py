"""Classified layout enables broader omissions without trusting labels alone."""

from dataclasses import replace

from kenkui.pdf_processing import (
    PdfBlock,
    omit_visual_material,
    remove_note_sections,
    remove_notes,
)
from test_pdf_notes import _notes
from test_pdf_steps import _document, _page


def test_layout_paragraphs_allow_verified_native_notes_and_references() -> None:
    """Multi-line layout blocks retain the same native corroboration requirements."""
    native = _notes()
    page = native.pages[0]
    blocks = (
        PdfBlock(
            "body",
            1,
            " ".join(b.text for b in page.blocks[:3]),
            sources=tuple(b.id for b in page.blocks[:3]),
            origin="layout",
        ),
        PdfBlock(
            "note",
            1,
            " ".join(b.text for b in page.blocks[3:]),
            role="note",
            sources=tuple(b.id for b in page.blocks[3:]),
            origin="layout",
        ),
    )
    document = replace(native, narration=blocks)
    result = remove_notes(document)
    assert len(result.narration) == 1
    assert "claim2" not in result.narration[0].text
    assert "A claim remains" in result.narration[0].text
    assert result.pages == native.pages
    assert remove_notes(result) == result


def test_explicit_numbered_endnote_section_stops_at_next_heading() -> None:
    """Remove only complete, consecutive note entries under an explicit heading."""
    document = _document(_page(1, ()))
    blocks = (
        PdfBlock("prose", 1, "The chapter prose."),
        PdfBlock("heading", 1, "Notes", role="heading", origin="layout"),
        PdfBlock("one", 1, "1. The first citation.", origin="layout"),
        PdfBlock("two", 1, "2. The second citation.", origin="layout"),
        PdfBlock("next", 1, "The next chapter", role="heading", origin="layout"),
        PdfBlock("more", 1, "More prose remains.", origin="layout"),
    )
    document = replace(document, narration=blocks)
    result = remove_note_sections(document)
    assert [b.id for b in result.narration] == ["prose", "next", "more"]
    assert remove_note_sections(result) == result
    uncertain = replace(
        document,
        narration=(
            *blocks[:3],
            replace(blocks[3], text="Ordinary prose."),
            *blocks[4:],
        ),
    )
    assert remove_note_sections(uncertain) == uncertain


def test_visual_omissions_do_not_drop_uncorroborated_captions() -> None:
    """A prose paragraph labelled caption alone must still be read."""
    document = _document(_page(1, ()))
    blocks = (
        PdfBlock(
            "picture", 1, "", role="picture", box=(50, 50, 500, 200), origin="layout"
        ),
        PdfBlock(
            "caption",
            1,
            "Figure 1. A landscape.",
            role="caption",
            box=(50, 205, 500, 220),
            origin="layout",
        ),
        PdfBlock(
            "wrong",
            1,
            "An exercise or quoted paragraph.",
            role="caption",
            box=(50, 400, 500, 420),
            origin="layout",
        ),
        PdfBlock("table", 1, "A table.", role="table", origin="layout"),
    )
    result = omit_visual_material(replace(document, narration=blocks))
    assert [b.id for b in result.narration] == ["wrong"]
    assert omit_visual_material(result) == result


def test_layout_furniture_requires_repetition_edges_and_consistent_folios() -> None:
    """OCR headers can be removed without erasing isolated or central labels."""
    from kenkui.pdf_processing import remove_furniture  # noqa: PLC0415

    pages = []
    narration: list[PdfBlock] = []
    for i in range(1, 6):
        blocks = (
            PdfBlock(
                f"header:{i}",
                i,
                "Running title",
                role="header",
                box=(60, 20, 300, 30),
                origin="layout",
            ),
            PdfBlock(
                f"folio:{i}",
                i,
                str(i + 10),
                role="footer",
                box=(60, 775, 100, 785),
                origin="layout",
            ),
            PdfBlock(
                f"body:{i}",
                i,
                "A central paragraph.",
                role="header",
                box=(60, 350, 500, 370),
                origin="layout",
            ),
        )
        pages.append(replace(_page(i, ()), layout_blocks=blocks))
        narration.extend(blocks)
    document = replace(_document(*pages), narration=tuple(narration))
    result = remove_furniture(document)
    assert all(block.id.startswith("body:") for block in result.narration)
    assert len(result.narration) == len(pages)
    assert result.pages == document.pages
    assert remove_furniture(result) == result
    short = replace(
        document, pages=document.pages[:2], narration=document.narration[:6]
    )
    assert remove_furniture(short) == short
