"""Deciding when a character in this volume is one the series already knows.

Pure functions over names and records: no model, no I/O. The rules are the
ones `identity` already applies within a book, which matters more here --
a series has more names competing for the same short forms than any single
volume does, so over-merging is likelier and costs more.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from kenkui._characters.identity import detect_titles, same_person
from kenkui._characters.store import SeriesCharacter, SeriesRecord

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from kenkui._domain.casting import CharacterProfile


def match_roster(
    record: SeriesRecord | None, characters: Sequence[CharacterProfile]
) -> dict[str, str]:
    """Map this volume's character ids onto the series' canonical ids.

    An exact alias decides on its own: the series recorded that surface form
    against exactly one person. Otherwise every name this character is known
    by is compared to every name the series knows, and a match counts only
    when it lands on exactly one series character -- two candidates means the
    name belongs to neither, which is `resolve_short_forms`' rule and the
    reason a series does not quietly merge its two Charleses.
    """
    if record is None:
        return {}
    by_alias: dict[str, set[str]] = {}
    for known in record.characters:
        for alias in (*known.aliases, known.display_name):
            by_alias.setdefault(alias, set()).add(known.canonical_id)
    titles = detect_titles(sorted(by_alias))

    matched: dict[str, str] = {}
    for character in characters:
        names = (*character.aliases, character.display_name)
        hosts = {
            canonical
            for name in names
            for alias, owners in by_alias.items()
            for canonical in owners
            if alias == name or same_person(alias, name, titles)
        }
        if len(hosts) == 1:
            matched[character.id] = next(iter(hosts))
    return matched


def merged_series(
    record: SeriesRecord | None,
    characters: Sequence[CharacterProfile],
    assignments: Mapping[str, str],
    narrator_voice_id: str,
    series_id: str,
) -> SeriesRecord:
    """Return the series as it stands after this volume.

    A returning character keeps the voice the series gave them and gains this
    volume's speech and surface forms. A newcomer joins with whatever voice
    the solver just chose.
    """
    matched = match_roster(record, characters)
    known = {c.canonical_id: c for c in (record.characters if record else ())}
    for character in characters:
        voice_id = assignments.get(character.id)
        if voice_id is None:
            continue
        canonical = matched.get(character.id, character.id)
        existing = known.get(canonical)
        aliases = {*character.aliases, character.display_name}
        if existing is None:
            known[canonical] = SeriesCharacter(
                canonical_id=canonical,
                display_name=character.display_name,
                gender=character.gender,
                voice_id=voice_id,
                spoken_characters=character.spoken_characters,
                aliases=tuple(sorted(aliases)),
            )
            continue
        known[canonical] = SeriesCharacter(
            canonical_id=canonical,
            display_name=existing.display_name,
            # The series keeps the first gender it was sure of: a later
            # volume answering None must not un-gender a cast character.
            gender=existing.gender if existing.gender is not None else character.gender,
            voice_id=existing.voice_id,
            spoken_characters=existing.spoken_characters
            + character.spoken_characters,
            aliases=tuple(sorted({*existing.aliases, *aliases})),
        )
    return SeriesRecord(
        series_id=series_id,
        narrator_voice_id=(
            record.narrator_voice_id if record is not None else narrator_voice_id
        ),
        characters=tuple(
            sorted(known.values(), key=lambda c: (-c.spoken_characters, c.canonical_id))
        ),
    )
