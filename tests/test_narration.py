"""First-person narration is found by its dialogue tags, or not at all."""

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
