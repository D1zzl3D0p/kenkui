"""Tests for grid splitter functions."""

import pytest

from kenkui._domain.grid import split_phrases, split_sentences


def test_sentences_split_on_terminal_punctuation() -> None:
    """Sentences should split on terminal punctuation."""
    assert split_sentences("One. Two! Three?") == ("One. ", "Two! ", "Three?")


def test_sentence_split_is_exact() -> None:
    """Sentence splits should be exact: joining parts recreates the text."""
    text = "One.  Two!\tThree?"
    assert "".join(split_sentences(text)) == text


def test_abbreviation_does_not_split() -> None:
    """Abbreviations like 'Dr.' should not trigger sentence splits."""
    assert split_sentences("Dr. Yueh smiled.") == ("Dr. Yueh smiled.",)


def test_single_initial_does_not_split() -> None:
    """Single initials like 'J. R.' should not trigger sentence splits."""
    assert split_sentences("J. R. Smith left.") == ("J. R. Smith left.",)


def test_closing_quote_stays_with_its_sentence() -> None:
    """Closing quotes should stay attached to their sentence."""
    assert split_sentences('"Go." He left.') == ('"Go." ', "He left.")


def test_closing_curly_quote_stays_with_its_sentence() -> None:
    """Closing curly quotes should stay attached to their sentence."""
    text = "\u201cGo.\u201d He left."
    assert ord(text[0]) == ord("\u201c")
    assert ord(text[4]) == ord("\u201d")
    assert split_sentences(text) == ("\u201cGo.\u201d ", "He left.")


def test_closing_curly_quote_exactness() -> None:
    """Curly quotes must preserve exact reconstruction invariant."""
    text = "\u201cGo.\u201d He left."
    assert "".join(split_sentences(text)) == text


def test_ellipsis_splits_once() -> None:
    """Ellipsis should split only once."""
    assert split_sentences("Wait... Then go.") == ("Wait... ", "Then go.")


def test_phrases_split_on_clause_punctuation() -> None:
    """Phrases should split on clause punctuation: commas, semicolons, colons."""
    assert split_phrases("Yes, he said; then left.") == (
        "Yes, ",
        "he said; ",
        "then left.",
    )


def test_phrase_split_is_exact() -> None:
    """Phrase splits should be exact: joining parts recreates the text."""
    text = "A, b; c: d"
    assert "".join(split_phrases(text)) == text


def test_phrase_split_with_curly_quotes() -> None:
    """Phrases should split correctly with curly punctuation."""
    text = "\u201cYes,\u201d \u201cNo;\u201d done."
    assert ord(text[0]) == ord("\u201c")
    assert ord(text[5]) == ord("\u201d")
    assert ord(text[7]) == ord("\u201c")
    assert ord(text[11]) == ord("\u201d")
    assert split_phrases(text) == (
        "\u201cYes,\u201d ",
        "\u201cNo;\u201d ",
        "done.",
    )


def test_phrase_split_curly_exactness() -> None:
    """Phrase splits with curly quotes must preserve exactness invariant."""
    text = "\u201cYes,\u201d \u201cNo;\u201d done."
    assert "".join(split_phrases(text)) == text


def test_empty_text_yields_no_parts() -> None:
    """Empty text should yield empty tuple."""
    assert split_sentences("") == ()
    assert split_phrases("") == ()


@pytest.mark.parametrize(
    "text",
    [
        "No punctuation here",
        "Mr. and Mrs. Smith.",
        "He said 'no.' She left.",
        "A.B.C. Corp. filed.",
        "...",
        "  ",
        "\u201cGo.\u201d He left.",
    ],
)
def test_splitters_are_exact_over_awkward_input(text: str) -> None:
    """Splitters must be exact over various edge cases."""
    assert "".join(split_sentences(text)) == text
    assert "".join(split_phrases(text)) == text
