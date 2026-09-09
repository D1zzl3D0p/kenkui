import pytest

from kenkui._domain.paths import Path, contains, parse_path, render_path
from kenkui.errors import ValidationError


def test_parse_full_path() -> None:
    parsed = parse_path({"chapter": "ch08", "paragraph": 3, "sentence": 2})
    assert parsed == Path(
        chapter="ch08", paragraph=3, line=None, sentence=2, phrase=None
    )


def test_empty_mapping_is_the_whole_book() -> None:
    assert parse_path({}).chapter is None


def test_holes_are_allowed_and_mean_any_of_that_level() -> None:
    parsed = parse_path({"chapter": "ch08", "sentence": 2})
    assert parsed.paragraph is None
    assert parsed.sentence == 2


def test_unknown_level_is_refused() -> None:
    with pytest.raises(ValidationError):
        parse_path({"chapter": "ch08", "stanza": 1})


def test_zero_and_negative_indices_are_refused_in_paths() -> None:
    with pytest.raises(ValidationError):
        parse_path({"chapter": "ch08", "paragraph": 0})


def test_subtree_containment() -> None:
    paragraph = Path("ch08", 3, None, None, None)
    sentence = Path("ch08", 3, 1, 2, None)
    assert contains(paragraph, sentence)
    assert not contains(sentence, paragraph)


def test_containment_is_reflexive() -> None:
    path = Path("ch08", 3, None, None, None)
    assert contains(path, path)


def test_book_contains_everything() -> None:
    assert contains(Path(None, None, None, None, None), Path("ch08", 3, 1, 2, 1))


def test_render_elides_degenerate_middle_levels() -> None:
    assert render_path(Path("ch08", 3, 1, 2, None)) == "ch08  ¶3  s2"
    assert render_path(Path("ch08", 3, None, None, None)) == "ch08  ¶3"
    assert render_path(Path(None, None, None, None, None)) == "whole book"
