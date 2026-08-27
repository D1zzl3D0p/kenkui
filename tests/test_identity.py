"""One person or two, decided from the names alone."""
# ruff: noqa: D103, FBT001, PLR2004

from __future__ import annotations

import pytest

from kenkui._characters.identity import (
    detect_titles,
    group_full_names,
    resolve_short_forms,
    same_person,
)
from kenkui._characters.infer import merge_rosters
from kenkui._domain.casting import CharacterProfile


@pytest.mark.parametrize(
    ("first", "second", "expected"),
    [
        # Two differing prefix honorifics are two people.
        ("Mr Elliot", "Miss Elliot", False),
        ("Mr Geary", "Mrs Geary", False),
        ("Admiral Brand", "Admiral Croft", False),
        # One honorific against none is one person.
        ("Dr. Pelorat", "Janov Pelorat", True),
        ("Captain Han Pritcher", "Han Pritcher", True),
        # A trailing title attaches to a single person.
        ("Moiraine Sedai", "Moiraine Aes Sedai", True),
        # Anything else differing is a surname.
        ("Charles Hayter", "Charles Musgrove", False),
        ("Balwen Ironhand", "Balwen Mayel", False),
    ],
)
def test_same_person(first: str, second: str, expected: bool) -> None:
    assert same_person(first, second) is expected


def test_detect_titles_learns_invented_honorifics() -> None:
    names = [
        "Brightlord Dalinar",
        "Brightlord Sadeas",
        "Brightlord Roshone",
        "Dalinar Kholin",
    ]
    assert "brightlord" in detect_titles(names)


def test_invented_honorific_does_not_split_one_person() -> None:
    names = [
        "Brightlord Dalinar",
        "Brightlord Sadeas",
        "Brightlord Roshone",
        "Dalinar Kholin",
    ]
    titles = detect_titles(names)
    assert same_person("Brightlord Dalinar", "Dalinar Kholin", titles) is True


def test_group_full_names_folds_one_person() -> None:
    entity = group_full_names(["Moiraine Sedai", "Moiraine Aes Sedai"])
    assert len(set(entity.values())) == 1


def test_short_form_with_one_host_is_assigned() -> None:
    entity = group_full_names(["Tam al'Thor"])
    resolved = resolve_short_forms(["Tam"], entity)
    assert resolved.assigned["Tam"] == "Tam al'Thor"
    assert resolved.ambiguous == {}


def test_short_form_with_two_hosts_is_dropped() -> None:
    entity = group_full_names(["Charles Hayter", "Charles Musgrove"])
    resolved = resolve_short_forms(["Charles"], entity)
    assert "Charles" not in resolved.assigned
    assert sorted(resolved.ambiguous["Charles"]) == [
        "Charles Hayter",
        "Charles Musgrove",
    ]


def test_short_form_with_no_host_stands_alone() -> None:
    resolved = resolve_short_forms(["Egwene"], {})
    assert resolved.assigned["Egwene"] == "Egwene"


def _profile(character_id: str, display: str) -> CharacterProfile:
    return CharacterProfile(
        id=character_id,
        display_name=display,
        gender=None,
        spoken_characters=0,
        chapter_ids=(),
    )


def test_merge_rosters_folds_one_person_under_two_names() -> None:
    merged = merge_rosters(
        (
            (_profile("moiraine-sedai", "Moiraine Sedai"),),
            (_profile("moiraine-aes-sedai", "Moiraine Aes Sedai"),),
        )
    )
    assert len(merged) == 1


def test_merge_rosters_keeps_two_people_apart() -> None:
    merged = merge_rosters(
        (
            (_profile("mr-elliot", "Mr Elliot"),),
            (_profile("miss-elliot", "Miss Elliot"),),
        )
    )
    assert len(merged) == 2


def test_merge_rosters_drops_an_ambiguous_short_form() -> None:
    merged = merge_rosters(
        (
            (_profile("charles-hayter", "Charles Hayter"),),
            (_profile("charles-musgrove", "Charles Musgrove"),),
            (_profile("charles", "Charles"),),
        )
    )
    assert sorted(character.id for character in merged) == [
        "charles-hayter",
        "charles-musgrove",
    ]


def test_merge_rosters_attaches_an_unambiguous_short_form() -> None:
    merged = merge_rosters(
        (
            (_profile("tam-althor", "Tam al'Thor"),),
            (_profile("tam", "Tam"),),
        )
    )
    assert len(merged) == 1
    assert merged[0].id == "tam-althor"


def test_merge_rosters_keeps_two_ids_with_an_identical_display_name_apart() -> None:
    """Two distinct ids sharing one bare name are not folded into each other.

    `id` is the model's stable identifier; a shared surface name across two
    distinct ids is not evidence they are one person - it is the model
    telling us, via the ids, that they are not.
    """
    merged = merge_rosters(
        (
            (_profile("charles-hayter", "Charles"),),
            (_profile("charles-musgrove", "Charles"),),
        )
    )
    assert sorted(character.id for character in merged) == [
        "charles-hayter",
        "charles-musgrove",
    ]


def test_one_person_under_two_ids_folds_despite_a_shared_display_name() -> None:
    """The production split: Corwi and Lizbyet Corwi cast as two women.

    A model asked for an id "stable across the whole book" does not reliably
    give one, so the same person arrives under two ids that happen to share a
    surface name. Treating that as two people gave her two voices.
    """
    merged = merge_rosters(
        (
            (_profile("corwi", "Corwi"),),
            (_profile("lizbyet-corwi", "Corwi"),),
            (_profile("lizbyet-corwi", "Lizbyet Corwi"),),
        )
    )
    assert len(merged) == 1
    assert merged[0].id == "lizbyet-corwi"


def test_two_people_sharing_a_bare_name_still_stand_apart() -> None:
    """The guard this rule exists for: ids that do not nest are two people."""
    merged = merge_rosters(
        (
            (_profile("charles-hayter", "Charles"),),
            (_profile("charles-musgrove", "Charles"),),
        )
    )
    assert {character.id for character in merged} == {
        "charles-hayter",
        "charles-musgrove",
    }


def test_a_rank_prefixed_id_folds_into_the_bare_one() -> None:
    """Dhatt and Senior Detective Dhatt are one man with one voice."""
    merged = merge_rosters(
        (
            (_profile("dhatt", "Dhatt"),),
            (_profile("senior-detective-dhatt", "Dhatt"),),
        )
    )
    assert len(merged) == 1


def test_merged_names_are_kept_as_aliases() -> None:
    """Folding Lizbyet Corwi into Corwi must not lose the other surface form.

    Volume 1 records the fullest name it saw and volume 3 uses a shorter one.
    Keeping only the head's display name throws away the string that would
    have matched them.
    """
    merged = merge_rosters(
        (
            (_profile("corwi", "Corwi"),),
            (_profile("lizbyet-corwi", "Corwi"),),
            (_profile("lizbyet-corwi", "Lizbyet Corwi"),),
        )
    )
    assert len(merged) == 1
    assert set(merged[0].aliases) == {"Corwi", "Lizbyet Corwi"}


def test_a_lone_character_aliases_to_its_own_name() -> None:
    """Every character has at least one surface form: the one it was given."""
    merged = merge_rosters(((_profile("dhatt", "Dhatt"),),))
    assert merged[0].aliases == ("Dhatt",)
