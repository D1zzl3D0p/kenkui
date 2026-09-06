"""Pure validation and normalization of caller-reviewed character rosters."""

from __future__ import annotations

from dataclasses import replace

from kenkui._characters.infer import PRONOUNS, UNKNOWN, slugify
from kenkui._characters.models import CharacterRoster
from kenkui.errors import ErrorCode, ValidationError


def validate_roster(
    roster: CharacterRoster, chapter_ids: frozenset[str]
) -> CharacterRoster:
    """Return a canonical roster, refusing ambiguous IDs and dangling references."""
    ids = [character.id for character in roster.characters]
    if len(ids) != len(set(ids)) or (
        roster.narrator_id is not None and roster.narrator_id not in ids
    ):
        raise ValidationError(ErrorCode.INVALID_ROSTER)
    for character in roster.characters:
        if (
            not character.id
            or slugify(character.id) != character.id
            or character.id in PRONOUNS
            or character.id == UNKNOWN
            or (character.id == "narrator" and roster.narrator_id != character.id)
            or not character.display_name.strip()
            or character.gender not in (None, "masculine", "feminine")
            or not set(character.chapter_ids) <= chapter_ids
            or any(not alias.strip() for alias in character.aliases)
        ):
            raise ValidationError(ErrorCode.INVALID_ROSTER)
    return CharacterRoster(
        characters=tuple(
            replace(
                character,
                display_name=character.display_name.strip(),
                aliases=tuple(sorted({alias.strip() for alias in character.aliases})),
                chapter_ids=tuple(sorted(set(character.chapter_ids))),
                spoken_characters=0,
            )
            for character in sorted(
                roster.characters, key=lambda character: character.id
            )
        ),
        narrator_id=roster.narrator_id,
    )
