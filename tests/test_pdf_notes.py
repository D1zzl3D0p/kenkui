"""Footnote removal needs both note geometry and an attached prose reference."""
# ruff: noqa: PLR2004

from dataclasses import replace

import pytest

from kenkui.pdf_processing import PdfCharacter, PdfDocument, remove_notes
from test_pdf_steps import _document, _page


def _notes(
    *,
    raised: bool = True,
    reference: str = "claim2",
    note: str = "2 This is the explanatory note.",
    note_size: float = 8,
) -> PdfDocument:
    page = _page(
        1,
        (
            (
                f"A {reference} remains supported by the surrounding prose.",
                60,
                150,
                500,
                12,
            ),
            ("Another body line supplies reliable size evidence.", 60, 166, 500, 12),
            ("The main prose stays intact on this page.", 60, 182, 500, 12),
            (note, 60, 650, 500, note_size),
            ("The explanatory note continues here.", 60, 661, 500, note_size),
        ),
    )
    line = page.lines[0]
    characters = tuple(
        PdfCharacter(
            char,
            (
                60 + i * 6,
                148 if char.isdigit() and raised else 150,
                66 + i * 6,
                156 if char.isdigit() and raised else 162,
            ),
            8 if char.isdigit() and raised else 12,
            "Body",
        )
        for i, char in enumerate(line.text)
    )
    page = replace(page, lines=(replace(line, characters=characters), *page.lines[1:]))
    return _document(page)


def test_verified_note_and_reference_removed_together() -> None:
    """Omit the small footer note and continuation, retaining the full main prose."""
    document = _notes()
    result = remove_notes(document)
    assert [b.text for b in result.narration] == [
        "A claim remains supported by the surrounding prose.",
        "Another body line supplies reliable size evidence.",
        "The main prose stays intact on this page.",
    ]
    assert result.pages == document.pages
    assert len(result.edits) == 3
    assert remove_notes(result) == result


@pytest.mark.parametrize("reference", ["x2", "102"])
def test_exponents_are_not_note_references(reference: str) -> None:
    """Numbers and single-letter variables do not authorize note removal."""
    document = _notes(reference=reference)
    assert remove_notes(document) == document


@pytest.mark.parametrize(
    "changes",
    [
        {"raised": False},
        {"note": "3 A different note number."},
        {"note": '"A line of dialogue in small type."'},
        {"note_size": 12},
    ],
)
def test_uncorroborated_small_type_or_digits_are_retained(
    changes: dict[str, object],
) -> None:
    """Typography, a number or a classifier label alone is insufficient."""
    document = _notes(**changes)  # type: ignore[arg-type]
    assert remove_notes(document) == document


def test_missing_or_edited_note_does_not_remove_reference() -> None:
    """A partial custom projection cannot authorize an orphaned reference edit."""
    document = _notes()
    missing = replace(document, narration=document.narration[:3])
    assert remove_notes(missing) == missing
    edited = replace(
        document,
        narration=(
            *document.narration[:3],
            replace(document.narration[3], text="Custom replacement"),
            document.narration[4],
        ),
    )
    assert remove_notes(edited) == edited


def test_note_removal_stops_at_normal_prose() -> None:
    """A succeeding body paragraph is never swallowed as a note continuation."""
    document = _notes()
    page = document.pages[0]
    body = replace(page.lines[4], text="The next prose paragraph.", size=12)
    block = replace(page.blocks[4], text=body.text)
    page = replace(
        page, lines=(*page.lines[:4], body), blocks=(*page.blocks[:4], block)
    )
    result = remove_notes(_document(page))
    assert result.narration[-1].text == body.text
    assert len(result.narration) == 4


def test_inserted_extraction_spaces_protect_single_letter_exponents() -> None:
    """Missing space glyphs cannot turn 'A x2' into a multi-letter prose word."""
    document = _notes(reference="x2")
    page = document.pages[0]
    line = page.lines[0]
    line = replace(line, characters=tuple(c for c in line.characters if c.text != " "))
    page = replace(page, lines=(line, *page.lines[1:]))
    document = _document(page)
    assert remove_notes(document) == document


def test_math_fonts_and_duplicate_note_numbers_are_ambiguous() -> None:
    """Conflicting or mathematical evidence cannot authorize deleting prose."""
    document = _notes()
    page = document.pages[0]
    line = page.lines[0]
    line = replace(
        line, characters=tuple(replace(c, font="Math") for c in line.characters)
    )
    mathematical = _document(replace(page, lines=(line, *page.lines[1:])))
    assert remove_notes(mathematical) == mathematical
    duplicate = replace(page.lines[4], text="2 Another possible numbered note.")
    block = replace(page.blocks[4], text=duplicate.text)
    ambiguous = _document(
        replace(
            page,
            lines=(*page.lines[:4], duplicate),
            blocks=(*page.blocks[:4], block),
        )
    )
    assert remove_notes(ambiguous) == ambiguous


def test_unfinished_note_at_page_end_is_retained() -> None:
    """Do not omit the first part of a note that may continue onto another page."""
    document = _notes()
    page = document.pages[0]
    line = replace(page.lines[-1], text="This note continues onto")
    block = replace(page.blocks[-1], text=line.text)
    page = replace(
        page, lines=(*page.lines[:-1], line), blocks=(*page.blocks[:-1], block)
    )
    document = _document(page)
    assert remove_notes(document) == document
