"""Word repair uses source line boundaries and evidence from the same document."""

from dataclasses import replace

import pytest

from kenkui.pdf_processing import PdfBlock, PdfDocument, repair_line_break_words
from test_pdf_steps import _document, _page


def _wrapped(left: str, right: str, evidence: str) -> PdfDocument:
    page = _page(
        1,
        (
            (left, 60, 150, 500, 12),
            (right, 60, 166, 500, 12),
            (evidence, 60, 220, 500, 12),
        ),
    )
    return replace(
        _document(page),
        narration=(
            PdfBlock("joined", 1, left + " " + right, sources=("1:0", "1:1")),
            page.blocks[2],
        ),
    )


@pytest.mark.parametrize(
    ("left", "right", "evidence", "expected"),
    [
        (
            "An invest-",
            "ment was made.",
            "The investment remained.",
            "An investment was made.",
        ),
        ("A long-", "term view.", "It was a long-term plan.", "A long-term view."),
        (
            "An invest\u00ad",
            "ment was made.",
            "No supporting occurrence.",
            "An investment was made.",
        ),
    ],
)
def test_supported_words_are_joined_without_changing_letters(
    left: str,
    right: str,
    evidence: str,
    expected: str,
) -> None:
    """Only the boundary separator changes; spelling and source evidence survive."""
    document = _wrapped(left, right, evidence)
    result = repair_line_break_words(document)
    assert result.narration[0].text == expected
    assert result.pages == document.pages
    assert "".join(c for c in result.narration[0].text if c.isalpha()) == (
        "".join(c for c in document.narration[0].text if c.isalpha())
    )
    assert repair_line_break_words(result) == result
    assert len(result.edits) == 1


@pytest.mark.parametrize(
    "evidence",
    [
        "No attested complete spelling.",
        "Both reentry and re-entry occur.",
        "Only preentry is present.",
    ],
)
def test_unknown_or_conflicting_spelling_is_preserved(evidence: str) -> None:
    """No lexical guess, substring match, or OCR correction is allowed."""
    document = _wrapped("A re-", "entry here.", evidence)
    result = repair_line_break_words(document)
    assert result.narration == document.narration
    assert result.issues[0].code == "ambiguous_line_break_word"
    assert repair_line_break_words(result) == result


def test_unrelated_midline_spelling_and_custom_text_are_untouched() -> None:
    """The same character sequence elsewhere must not be globally replaced."""
    document = _wrapped("An invest-", "ment and invest- ment.", "An investment.")
    result = repair_line_break_words(document)
    assert result.narration[0].text == "An investment and invest- ment."
    custom = replace(
        document,
        narration=(replace(document.narration[0], text="Edited invest- ment."),),
    )
    assert repair_line_break_words(custom) == custom


def test_code_and_paragraph_boundaries_are_not_word_repair_evidence() -> None:
    """A source line break is insufficient without an established prose continuation."""
    document = _wrapped("An invest-", "ment here.", "An investment.")
    code = replace(document, narration=(replace(document.narration[0], role="code"),))
    assert repair_line_break_words(code) == code
    unmerged = _document(*document.pages)
    assert repair_line_break_words(unmerged) == unmerged
