"""First-person narration is found by its dialogue tags, or not at all."""
# ruff: noqa: RUF001 - the typographic quotes are the data under test;
# writing them as escapes would make every case unreadable.

from __future__ import annotations

from kenkui._characters.narration import first_person_tags, is_first_person
from kenkui._characters.quotes import extract_spans

FIRST = '"I will not," I said. "You know that." She turned away.'
THIRD = '"I will not," Anne said. "You know that." She turned away.'


def _ends(text: str) -> list[int]:
    return [s.end for s in extract_spans("ch", text) if s.is_dialogue]


def test_first_person_tag_is_counted() -> None:
    """`"..." I said` is the plain first-person tag this module exists for."""
    assert first_person_tags(FIRST, _ends(FIRST)) == 1


def test_third_person_tag_is_not_counted() -> None:
    """A named speaker must not be mistaken for the narrator."""
    assert first_person_tags(THIRD, _ends(THIRD)) == 0


def test_inverted_tag_is_counted() -> None:
    """`said I` is the other order English puts a first-person tag in."""
    text = '"I will not," said I. "You know that."'
    assert first_person_tags(text, _ends(text)) == 1


def test_a_book_needs_several_tags_to_qualify() -> None:
    """One match is noise; the minimum is what turns a book first-person."""
    assert is_first_person(FIRST, _ends(FIRST)) is False
    doubled = FIRST * 3
    assert is_first_person(doubled, _ends(doubled)) is True


def test_curly_single_interrupted_tag_is_not_counted() -> None:
    """The scan must not run through a closing curly single quote.

    quotes.py treats the curly single pair as a dialogue delimiter for
    British prose, so the tag here is "she said" -- the "I said" that
    follows belongs to the next, separate quote.
    """
    text = "‘Well,’ she said, ‘I said nothing.’"
    assert first_person_tags(text, _ends(text)) == 0


def test_curly_single_back_to_back_quotes_are_not_counted() -> None:
    """A closing quote immediately followed by another must not be bridged."""
    text = "‘Stop it.’ ‘I told you already,’ he shouted at her sister."
    assert first_person_tags(text, _ends(text)) == 0


def test_curly_single_first_person_tag_is_still_counted() -> None:
    """The fix for the bridging bug must not over-correct into a miss."""
    text = "‘I will not,’ I said."
    assert first_person_tags(text, _ends(text)) == 1
