"""Elongation and stammer collapse, grounded in real book text.

Every token asserted here was taken from a parsed epub: the elongation cases
from Dune and Foundation, the stammer cases from The Eye of the World, and the
untouched cases from the same books' ordinary hyphenated vocabulary.
"""

from __future__ import annotations

import pytest

import kenkui as kk
from kenkui._domain.spoken import spoken_identity, to_spoken
from kenkui.errors import ValidationError


def speak(text: str, **features: bool) -> str:
    """Run the spoken-form pass with only the vocalise family able to match."""
    return to_spoken(text, numbers="off", lexicon=(), builtin=False, features=features)


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        ("Ah-h", "Ahh"),
        ("Ah-h-h", "Ahhh"),
        ("Ah-h-h-h-h-h", "Ahhhhhh"),
        ("hm-m-m", "hmmm"),
        ("Sh-h-h-h", "Shhhh"),
        ("No-o-o", "Nooo"),
        ("Wel-l-l-l", "Wellll"),
        ("Stilgar-r-r-r", "Stilgarrrr"),
        ("Aiee-e-e", "Aieeee"),
    ],
)
def test_elongation_absorbs_the_repeated_letter(source: str, expected: str) -> None:
    """A dash-run echoing the stem's last letter is spoken as one long word."""
    assert speak(source) == expected


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        ("Ah-h-h-um-m-m", "Ahhh ummm"),
        ("Um-m-m-m-ah", "Ummmm ah"),
        ("Um-m-m-m-ah-h-h-hm-m-m", "Ummmm ahhh hmmm"),
        ("um-m-m-ah-h", "ummm ahh"),
        ("Ho-ho-ho-o-o-o", "Ho ho hoooo"),
    ],
)
def test_a_new_interjection_starts_a_new_word(source: str, expected: str) -> None:
    """Joining across an interjection boundary would give "Ummmmahhhhmmm"."""
    assert speak(source) == expected


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        ("S-s-sorry", "Sssorry"),
        ("M-m-mat", "Mmmat"),
        ("f-father", "ffather"),
        ("M-my", "Mmy"),
        ("t-the", "tthe"),
    ],
)
def test_stammer_collapses_onto_the_word_it_restarts(
    source: str, expected: str
) -> None:
    """Leading single letters are false starts, not a word of their own."""
    assert speak(source) == expected


@pytest.mark.parametrize(
    "source",
    [
        # Proper nouns and ordinary compounds, from Dune.
        "al-Gaib",
        "na-Baron",
        "Bi-lal",
        "blue-within-blue",
        "so-called",
        "great-great-grandmother",
        "G-force",
        "matter-of-fact",
        # Reduplication: collapsing it would give "dripdripdrip".
        "Go-go-go",
        "drip-drip-drip",
        "thwok-thwok",
        "Soo-Soo",
        # Prefixed compounds whose prefix is not a single letter.
        "co-conspirator",
        "re-reading",
        "de-emphasis",
    ],
)
def test_ordinary_hyphenation_is_left_alone(source: str) -> None:
    """The rule must not reach past the two vocal families into vocabulary."""
    assert speak(source) == source


def test_stammer_off_leaves_the_token_whole() -> None:
    """Elongation must not eat a stammer's onset when stammer is declined.

    Without the guard the leading "s" absorbs into "S" and this reads
    "Ss sorry", which is worse than the untouched text.
    """
    assert speak("S-s-sorry", stammer=False) == "S-s-sorry"


def test_elongation_can_be_declined() -> None:
    """Each family is switchable on its own, as number features are."""
    assert speak("Ah-h-h", elongation=False) == "Ah-h-h"


def test_both_families_are_on_by_default() -> None:
    """Calling pronounce at all is the opt-in; neither needs naming."""
    assert speak("Ah-h-h and S-s-sorry") == "Ahhh and Sssorry"


def test_surrounding_text_is_untouched() -> None:
    """Only the token is rewritten; spacing and punctuation survive."""
    assert speak('"Ah-h-h," he said.') == '"Ahhh," he said.'


def test_identity_records_the_vocalise_state() -> None:
    """Output that can differ must be reachable from the cache key.

    Recorded unconditionally rather than only when overridden: both families
    are on by default, so a render cached before they existed would otherwise
    keep serving audio that says "ah-aitch".
    """
    base = spoken_identity(numbers="off", lexicon=(), builtin=False)
    assert base["vocalise"] == "elongation=1,stammer=1"
    declined = spoken_identity(
        numbers="off", lexicon=(), builtin=False, features={"elongation": False}
    )
    assert declined["vocalise"] == "elongation=0,stammer=1"


def test_vocalise_features_do_not_leak_into_the_number_identity() -> None:
    """Each family owns its own names, so neither can shadow the other."""
    identity = spoken_identity(
        numbers="off", lexicon=(), builtin=False, features={"stammer": False}
    )
    assert "number_features" not in identity


def test_pronounce_accepts_the_vocalise_features() -> None:
    """The pipeline surface must know the names, or they cannot be declined."""
    pipeline = kk.epub("book.epub").pronounce(
        numbers="off", elongation=False, stammer=False
    )
    assert pipeline.operations


def test_pronounce_still_rejects_an_unknown_feature() -> None:
    """Widening the set must not turn a caller's typo into a silent no-op."""
    with pytest.raises(ValidationError):
        kk.epub("book.epub").pronounce(stammerr=True)


def test_declining_both_families_restores_the_identity_function() -> None:
    """With every family off the pass must not touch the text at all.

    Guards the short-circuit: the vocalise rule is built unconditionally
    otherwise, so "everything off" would stop meaning byte-for-byte identical.
    """
    source = 'He said "Ah-h-h" and "S-s-sorry" in 1984.'
    assert (
        to_spoken(
            source,
            numbers="off",
            lexicon=(),
            builtin=False,
            features={"elongation": False, "stammer": False},
        )
        == source
    )
