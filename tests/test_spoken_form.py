"""Composition and rank order of the combined spoken-form matcher."""

from __future__ import annotations

from kenkui._domain.spoken import SPOKEN_FORM_VERSION, spoken_identity, to_spoken


def test_numbers_and_lexicon_apply_in_one_pass() -> None:
    """Both rule families act on the same text without a second sweep."""
    result = to_spoken(
        "100,000 cellos", numbers="conservative", lexicon=(), builtin=True
    )
    assert result == "one hundred thousand chellos"


def test_caller_lexicon_outranks_numbers() -> None:
    """A caller entry matching at a position beats a number rule there."""
    result = to_spoken(
        "Room 101",
        numbers="conservative",
        lexicon=(("101", "one oh one"),),
        builtin=False,
    )
    assert result == "Room one oh one"


def test_everything_off_is_the_identity_function() -> None:
    """With no rules the text is returned unchanged, character for character."""
    source = "100,000 cellos in 1984."
    assert to_spoken(source, numbers="off", lexicon=(), builtin=False) == source


def test_text_without_matches_is_unchanged() -> None:
    """Ordinary prose survives the pass byte for byte."""
    source = "The quick brown fox jumps over the lazy dog."
    assert to_spoken(source, numbers="conservative", lexicon=(), builtin=True) == source


def test_identity_changes_with_each_input() -> None:
    """Every knob that can change output also changes the identity payload."""
    base = spoken_identity(numbers="conservative", lexicon=(), builtin=True)
    assert base["spoken_form_schema"] == SPOKEN_FORM_VERSION
    assert base != spoken_identity(numbers="standard", lexicon=(), builtin=True)
    assert base != spoken_identity(numbers="conservative", lexicon=(), builtin=False)
    assert base != spoken_identity(
        numbers="conservative", lexicon=(("a", "b"),), builtin=True
    )


def test_identity_is_stable_across_calls() -> None:
    """The payload is a pure function of its inputs."""
    first = spoken_identity(numbers="standard", lexicon=(("a", "b"),), builtin=True)
    second = spoken_identity(numbers="standard", lexicon=(("a", "b"),), builtin=True)
    assert first == second
