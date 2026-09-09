"""Labeled path parsing, matching, and display tests."""

import pytest

from kenkui._domain.paths import Path, contains, parse_path, render_path
from kenkui.errors import ValidationError

SENTENCE_NUMBER = 2


def test_parse_full_path() -> None:
    """Parse a complete sparse path into its labeled components."""
    parsed = parse_path({"chapter": "ch08", "paragraph": 3, "sentence": 2})
    assert parsed == Path(
        chapter="ch08", paragraph=3, line=None, sentence=2, phrase=None
    )


def test_empty_mapping_is_the_whole_book() -> None:
    """An empty mapping addresses the whole book."""
    assert parse_path({}).chapter is None


def test_holes_are_allowed_and_mean_any_of_that_level() -> None:
    """Omitted levels remain unconstrained independent wildcards."""
    parsed = parse_path({"chapter": "ch08", "sentence": SENTENCE_NUMBER})
    assert parsed.paragraph is None
    assert parsed.sentence == SENTENCE_NUMBER


def test_direct_path_construction_allows_holes() -> None:
    """Direct construction permits omitted intermediate coordinates."""
    assert Path("ch08", None, None, SENTENCE_NUMBER, None).sentence == SENTENCE_NUMBER


def test_unknown_level_is_refused() -> None:
    """Unknown labels are rejected during parsing."""
    with pytest.raises(ValidationError):
        parse_path({"chapter": "ch08", "stanza": 1})


def test_zero_and_negative_indices_are_refused_in_paths() -> None:
    """Zero coordinates are rejected during parsing."""
    with pytest.raises(ValidationError):
        parse_path({"chapter": "ch08", "paragraph": 0})


@pytest.mark.parametrize("value", [True, False, "1", 1.5, 0, -1])
def test_invalid_coordinate_types_are_refused(value: object) -> None:
    """Invalid coordinate types and signs are rejected directly."""
    with pytest.raises(ValidationError):
        Path("ch08", value, None, None, None)  # type: ignore[arg-type]


def test_non_string_chapter_is_refused() -> None:
    """Chapter identifiers must be strings when provided."""
    with pytest.raises(ValidationError):
        Path(8, None, None, None, None)  # type: ignore[arg-type]


def test_subtree_containment() -> None:
    """A more specific path is contained by its broader subtree path."""
    paragraph = Path("ch08", 3, None, None, None)
    sentence = Path("ch08", 3, 1, 2, None)
    assert contains(paragraph, sentence)
    assert not contains(sentence, paragraph)


def test_containment_is_reflexive() -> None:
    """Every path contains itself."""
    path = Path("ch08", 3, None, None, None)
    assert contains(path, path)


def test_book_contains_everything() -> None:
    """The whole-book path contains every concrete path."""
    assert contains(Path(None, None, None, None, None), Path("ch08", 3, 1, 2, 1))


def test_render_elides_degenerate_middle_levels() -> None:
    """Display rendering elides line one and labels remaining levels."""
    assert render_path(Path("ch08", 3, 1, 2, None)) == "ch08  ¶3  s2"
    assert render_path(Path("ch08", 3, None, None, None)) == "ch08  ¶3"
    assert render_path(Path(None, None, None, None, None)) == "whole book"
