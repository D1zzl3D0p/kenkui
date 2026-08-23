"""Whole-word pronunciation matching, capitalization shape, and validation."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from kenkui._domain.spoken.lexicon import (
    MAX_LEXICON_ENTRIES,
    builtin_entries,
    lexicon_rules,
    validate_entries,
)
from kenkui.errors import ErrorCode, ValidationError

if TYPE_CHECKING:
    from kenkui._domain.spoken.numbers import Rule


def apply_rules(rules: tuple[Rule, ...], text: str) -> str:
    """Drive ordered rules across text the way the Task 5 matcher will."""
    out: list[str] = []
    position = 0
    while position < len(text):
        for pattern, handler in rules:
            match = pattern.match(text, position)
            if match is None:
                continue
            replacement = handler(match)
            if replacement is None:
                continue
            out.append(replacement)
            position = match.end()
            break
        else:
            out.append(text[position])
            position += 1
    return "".join(out)


def caller(**entries: str) -> tuple[Rule, ...]:
    """Build rules from caller entries only, with the built-in list disabled."""
    return lexicon_rules(validate_entries(entries), builtin=False)


def test_matches_whole_words_only() -> None:
    """A phrase inside a longer word is not a match."""
    rules = caller(cat="kat")
    assert apply_rules(rules, "the cat sat") == "the kat sat"
    assert apply_rules(rules, "concatenate") == "concatenate"


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        ("cello", "chello"),
        ("Cello", "Chello"),
        ("CELLO", "CHELLO"),
    ],
)
def test_replacement_inherits_capitalization_shape(source: str, expected: str) -> None:
    """The replacement takes the source's shape, not the entry's."""
    assert apply_rules(caller(cello="chello"), source) == expected


def test_longest_phrase_wins() -> None:
    """A multi-word entry beats a shorter entry that starts at the same place."""
    rules = lexicon_rules(
        validate_entries({"coup": "koo", "coup de grace": "coo de grahss"}),
        builtin=False,
    )
    assert apply_rules(rules, "a coup de grace") == "a coo de grahss"


def test_builtin_matches_diacritic_insensitively() -> None:
    """Built-in entries catch both the accented and unaccented spelling."""
    rules = lexicon_rules((), builtin=True)
    accented = "coup de grâce"
    assert apply_rules(rules, accented) == apply_rules(rules, "coup de grace")
    assert apply_rules(rules, "coup de grace") != "coup de grace"


def test_caller_entry_overrides_builtin() -> None:
    """A caller entry wins over a built-in for the same phrase."""
    rules = lexicon_rules(validate_entries({"cello": "SELLO"}), builtin=True)
    assert apply_rules(rules, "cello") == "SELLO"


def test_output_is_never_rescanned() -> None:
    """A replacement containing another entry's key does not cascade."""
    rules = lexicon_rules(
        validate_entries({"alpha": "beta", "beta": "gamma"}), builtin=False
    )
    assert apply_rules(rules, "alpha") == "beta"


def test_builtin_entries_are_sorted_and_nonempty() -> None:
    """The shipped data file loads and is deterministically ordered."""
    entries = builtin_entries()
    assert entries
    assert list(entries) == sorted(entries)


@pytest.mark.parametrize(
    "entries",
    [
        {"": "x"},
        {"x": ""},
        {"   ": "x"},
        {"x": "   "},
        {"a" * 200: "x"},
        {"x": "a" * 200},
    ],
)
def test_validate_rejects_malformed_entries(entries: dict[str, str]) -> None:
    """Empty, blank, or over-long entries are refused."""
    with pytest.raises(ValidationError) as error:
        validate_entries(entries)
    assert error.value.code is ErrorCode.INVALID_PRONUNCIATION


def test_validate_rejects_too_many_entries() -> None:
    """The entry count is bounded because a server accepts this untrusted."""
    entries = {f"word{index}": "x" for index in range(MAX_LEXICON_ENTRIES + 1)}
    with pytest.raises(ValidationError) as error:
        validate_entries(entries)
    assert error.value.code is ErrorCode.INVALID_PRONUNCIATION


def test_validate_returns_sorted_pairs() -> None:
    """Order is canonical so identity does not depend on caller dict order."""
    assert validate_entries({"b": "2", "a": "1"}) == (("a", "1"), ("b", "2"))
