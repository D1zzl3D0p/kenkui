"""One person across volumes, decided from the names alone."""

from __future__ import annotations

from kenkui._characters import store
from kenkui._characters.series import match_roster, merged_series
from kenkui._domain.casting import CharacterProfile


def _known(
    canonical_id: str, display: str, aliases: tuple[str, ...]
) -> store.SeriesCharacter:
    return store.SeriesCharacter(
        canonical_id=canonical_id,
        display_name=display,
        gender="masculine",
        voice_id="alf",
        spoken_characters=500,
        aliases=aliases,
    )


def _profile(character_id: str, display: str, *aliases: str) -> CharacterProfile:
    return CharacterProfile(
        id=character_id,
        display_name=display,
        gender=None,
        spoken_characters=100,
        chapter_ids=("ch1",),
        aliases=tuple(sorted({display, *aliases})),
    )


def test_an_exact_alias_matches() -> None:
    """Volume 3 says "Kaladin"; volume 1 recorded it as an alias."""
    record = store.SeriesRecord(
        "stormlight", "eponine",
        (_known("kaladin-stormblessed", "Kaladin Stormblessed",
                ("Kaladin", "Kaladin Stormblessed")),),
    )
    matched = match_roster(record, (_profile("kaladin", "Kaladin"),))
    assert matched == {"kaladin": "kaladin-stormblessed"}


def test_a_nested_name_matches_without_an_exact_alias() -> None:
    """Identity resolution, not just string equality.

    The series recorded only the full name; this volume uses the short one
    and it was never stored as an alias, so an exact hit cannot find it.
    """
    record = store.SeriesRecord(
        "stormlight", "eponine",
        (_known("dalinar-kholin", "Dalinar Kholin", ("Dalinar Kholin",)),),
    )
    matched = match_roster(record, (_profile("dalinar", "Dalinar"),))
    assert matched == {"dalinar": "dalinar-kholin"}


def test_two_people_sharing_a_surname_do_not_match() -> None:
    """Over-merging gives two people one voice, which is the worse failure."""
    record = store.SeriesRecord(
        "s", "eponine",
        (_known("charles-hayter", "Charles Hayter", ("Charles Hayter",)),),
    )
    matched = match_roster(record, (_profile("charles-musgrove", "Charles Musgrove"),))
    assert matched == {}


def test_an_ambiguous_short_form_matches_nobody() -> None:
    """Two hosts could claim it, so it names neither."""
    record = store.SeriesRecord(
        "s", "eponine",
        (
            _known("charles-hayter", "Charles Hayter", ("Charles Hayter",)),
            _known("charles-musgrove", "Charles Musgrove", ("Charles Musgrove",)),
        ),
    )
    assert match_roster(record, (_profile("charles", "Charles"),)) == {}


def test_no_series_yet_matches_nothing() -> None:
    """The first volume has nothing to match against."""
    assert match_roster(None, (_profile("kaladin", "Kaladin"),)) == {}


def test_merging_accumulates_speech_and_aliases() -> None:
    """A returning character's totals grow; their voice does not change."""
    record = store.SeriesRecord(
        "s", "eponine", (_known("kaladin-stormblessed", "Kaladin Stormblessed",
                                ("Kaladin Stormblessed",)),),
    )
    updated = merged_series(
        record,
        (_profile("kaladin", "Kaladin"),),
        {"kaladin": "alf"},
        "eponine",
        "s",
    )
    kaladin = next(
        c for c in updated.characters if c.canonical_id == "kaladin-stormblessed"
    )
    assert kaladin.voice_id == "alf"
    assert kaladin.spoken_characters == 600  # noqa: PLR2004 - 500 known + 100 this volume
    assert "Kaladin" in kaladin.aliases


def test_merging_adds_a_newcomer() -> None:
    """A character the series has not met joins it with the voice just solved."""
    updated = merged_series(
        None, (_profile("shallan", "Shallan Davar"),), {"shallan": "aoife"},
        "eponine", "s",
    )
    assert updated.series_id == "s"
    assert updated.narrator_voice_id == "eponine"
    assert [c.canonical_id for c in updated.characters] == ["shallan"]
    assert updated.characters[0].voice_id == "aoife"


def test_two_local_characters_claiming_one_series_person_match_neither() -> None:
    """A maid named "Elizabeth" and the protagonist "Lizzy" are not one person.

    Both surface forms the series accumulated for Elizabeth Bennet also reach
    two different characters in this volume. Attaching either would hand one
    of them her voice and discard the one the in-volume solver already gave
    them, so a canonical id claimed by more than one local character is
    refused for all of its claimants -- the same one-person-two-voices bias
    `identity` states, applied across the series boundary too.
    """
    record = store.SeriesRecord(
        "s", "eponine",
        (_known("elizabeth-bennet", "Elizabeth Bennet", ("Elizabeth", "Lizzy")),),
    )
    characters = (_profile("elizabeth-maid", "Elizabeth"), _profile("lizzy", "Lizzy"))
    assert match_roster(record, characters) == {}

    updated = merged_series(
        record, characters, {"elizabeth-maid": "v1", "lizzy": "v2"}, "eponine", "s",
    )
    assert len(updated.characters) == 3  # noqa: PLR2004 - two locals plus the untouched original
    by_id = {c.canonical_id: c.voice_id for c in updated.characters}
    assert by_id["elizabeth-maid"] == "v1"
    assert by_id["lizzy"] == "v2"
    assert by_id["elizabeth-bennet"] == "alf"


def test_unmatched_newcomer_does_not_inherit_an_unrelated_slug_collision() -> None:
    """Two different volumes' characters can slug to the same raw id.

    Volume 1's "Town Guard" and a later volume's "Castle Guard" are different
    people who never matched by name. merged_series must not fall back to the
    raw character id when that id is already claimed by someone else in the
    series -- doing so would hand the second guard the first guard's voice
    and running total with no name comparison ever happening.
    """
    record = merged_series(
        None, (_profile("guard", "Town Guard"),), {"guard": "v1"}, "n", "s",
    )
    updated = merged_series(
        record, (_profile("guard", "Castle Guard"),), {"guard": "v2"}, "n", "s",
    )
    assert len(updated.characters) == 2  # noqa: PLR2004 - two distinct guards
    canonical_ids = {c.canonical_id for c in updated.characters}
    assert len(canonical_ids) == 2  # noqa: PLR2004 - two distinct guards
    by_display = {c.display_name: c for c in updated.characters}
    assert by_display["Town Guard"].voice_id == "v1"
    assert by_display["Town Guard"].spoken_characters == 100  # noqa: PLR2004
    assert by_display["Castle Guard"].voice_id == "v2"
    assert by_display["Castle Guard"].spoken_characters == 100  # noqa: PLR2004


def test_merging_the_same_volume_twice_replaces_its_contribution() -> None:
    """A re-render must not double the count for the same volume.

    ``resolve()``/``write()`` call this on every run, so re-rendering one
    volume is ordinary, not hypothetical -- without ``book_digest`` a second
    pass over the same speech would add it again rather than replace it.
    """
    first = merged_series(
        None, (_profile("kaladin", "Kaladin"),), {"kaladin": "alf"}, "eponine", "s",
        book_digest="volume-1",
    )
    again = merged_series(
        first, (_profile("kaladin", "Kaladin"),), {"kaladin": "alf"}, "eponine", "s",
        book_digest="volume-1",
    )
    kaladin = next(c for c in again.characters if c.canonical_id == "kaladin")
    assert kaladin.spoken_characters == 100  # noqa: PLR2004 - unchanged by the redo


def test_merging_a_different_volume_still_accumulates() -> None:
    """Naming volumes must not turn off the series' whole point: accumulation."""
    first = merged_series(
        None, (_profile("kaladin", "Kaladin"),), {"kaladin": "alf"}, "eponine", "s",
        book_digest="volume-1",
    )
    second = merged_series(
        first, (_profile("kaladin", "Kaladin"),), {"kaladin": "alf"}, "eponine", "s",
        book_digest="volume-2",
    )
    kaladin = next(c for c in second.characters if c.canonical_id == "kaladin")
    assert kaladin.spoken_characters == 200  # noqa: PLR2004 - 100 each, two volumes


def test_an_unnamed_volume_still_accumulates_as_before() -> None:
    """Omitting ``book_digest`` keeps the original, purely additive contract."""
    first = merged_series(
        None, (_profile("kaladin", "Kaladin"),), {"kaladin": "alf"}, "eponine", "s",
    )
    second = merged_series(
        first, (_profile("kaladin", "Kaladin"),), {"kaladin": "alf"}, "eponine", "s",
    )
    kaladin = next(c for c in second.characters if c.canonical_id == "kaladin")
    assert kaladin.spoken_characters == 200  # noqa: PLR2004 - additive, as always


def test_re_merging_an_unchanged_volume_does_not_mint_a_new_canonical() -> None:
    """A withheld character must not become a new person on every render.

    ``match_roster`` withholds a canonical two local characters both reach,
    which is correct. What follows must not be: an unmatched character fell
    through to a fresh mint, found its own slug taken by the mint the
    previous render made, and took ``elizabeth-2``, then ``elizabeth-3``.
    Each mint carried an empty ledger, so the ``book_digest`` idempotency
    never got the chance to replace anything, and the series grew without
    bound while re-casting the volume every time.
    """
    record = store.SeriesRecord(
        "s", "eponine",
        (_known("elizabeth-bennet", "Elizabeth Bennet", ("Elizabeth", "Lizzy")),),
    )
    characters = (_profile("elizabeth", "Elizabeth"), _profile("lizzy", "Lizzy"))
    for _ in range(4):
        record = merged_series(
            record, characters, {"elizabeth": "v1", "lizzy": "v2"}, "eponine", "s",
            book_digest="volume-1",
        )
        assert {c.canonical_id for c in record.characters} == {
            "elizabeth-bennet",
            "elizabeth",
            "lizzy",
        }
    elizabeth = next(c for c in record.characters if c.canonical_id == "elizabeth")
    assert elizabeth.spoken_characters == 100  # noqa: PLR2004 - replaced, never piled up


def test_a_bare_series_alias_does_not_capture_a_longer_new_name() -> None:
    """Volume 3 introduces Dalinar's son while Dalinar himself is absent.

    ``merge_rosters`` routinely records bare surnames as aliases, and
    ``same_person`` treats nesting as identity in both directions, so the
    series' bare "Kholin" hosted the newcomer "Adolin Kholin" -- pinning the
    son to his father's voice and folding him into that record for good.
    Inside one book this direction is impossible: ``resolve_short_forms``
    attaches shorts to fulls and never the reverse. The series must match
    the same way round.
    """
    record = store.SeriesRecord(
        "s", "eponine",
        (
            _known(
                "dalinar-kholin",
                "Dalinar Kholin",
                ("Dalinar", "Dalinar Kholin", "Kholin"),
            ),
        ),
    )
    assert match_roster(record, (_profile("adolin-kholin", "Adolin Kholin"),)) == {}


def test_a_bare_series_forename_does_not_capture_a_longer_new_name() -> None:
    """The forename variant of the same defect: "John" hosting "John Watson"."""
    record = store.SeriesRecord(
        "s", "eponine", (_known("john", "John", ("John",)),)
    )
    assert match_roster(record, (_profile("john-watson", "John Watson"),)) == {}


def test_an_alias_two_series_characters_share_matches_nobody() -> None:
    """One surface form on two canonicals is ambiguity, which refuses.

    The store can now record that, so ``match_roster`` has to mean it: an
    alias reaching two people names neither, exactly as two full names
    sharing a short form do.
    """
    record = store.SeriesRecord(
        "s", "eponine",
        (
            _known("elizabeth-bennet", "Elizabeth Bennet", ("Elizabeth",)),
            _known("elizabeth-gardiner", "Elizabeth Gardiner", ("Elizabeth",)),
        ),
    )
    assert match_roster(record, (_profile("elizabeth", "Elizabeth"),)) == {}
