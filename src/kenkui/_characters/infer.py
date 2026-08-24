"""Normalise a model's character roster into deterministic values.

The model returns free text. Everything downstream keys off character ids, so
they are slugified here and never trusted as returned.
"""

from __future__ import annotations

import re
import unicodedata

from kenkui._domain.casting import CharacterProfile

# Never a speaker. A model asked for a name will happily answer "he", and a
# pronoun as a character id would collapse every male speaker into one voice.
PRONOUNS = frozenset(
    {
        "i",
        "me",
        "my",
        "mine",
        "myself",
        "you",
        "your",
        "yours",
        "yourself",
        "he",
        "him",
        "his",
        "himself",
        "she",
        "her",
        "hers",
        "herself",
        "it",
        "its",
        "itself",
        "we",
        "us",
        "our",
        "ours",
        "ourselves",
        "they",
        "them",
        "their",
        "theirs",
        "themselves",
        "who",
        "whom",
        "whose",
        "someone",
        "somebody",
        "everyone",
        "everybody",
        "no-one",
        "nobody",
        "anyone",
        "anybody",
    }
)

# The value the prompt reserves for "could not tell".
UNKNOWN = "unknown"

_GENDERS = frozenset({"feminine", "masculine"})
_SLUG_STRIP = re.compile(r"[^a-z0-9]+")


def slugify(value: str) -> str:
    """Return a stable lowercase-hyphen id, or empty when nothing survives."""
    folded = unicodedata.normalize("NFKD", value).encode("ascii", "ignore").decode()
    return _SLUG_STRIP.sub("-", folded.lower()).strip("-")


def normalise_roster(payload: object) -> tuple[CharacterProfile, ...]:
    """Convert one model response into an ordered, deduplicated roster.

    Sorted by id so two runs returning the same characters in a different
    order produce the same roster, which the plan fingerprint requires.
    Spoken volume is filled in later, once spans are attributed.
    """
    if not isinstance(payload, list):
        return ()
    seen: dict[str, CharacterProfile] = {}
    for item in payload:
        if not isinstance(item, dict):
            continue
        name = item.get("name")
        if not isinstance(name, str) or not name.strip():
            continue
        raw_id = item.get("id")
        character_id = slugify(raw_id if isinstance(raw_id, str) else name) or slugify(
            name
        )
        # A pronoun id would merge unrelated speakers under one voice, and a
        # pronoun is never what was asked for.
        if not character_id or character_id in PRONOUNS or name.lower() in PRONOUNS:
            continue
        gender = item.get("gender")
        seen.setdefault(
            character_id,
            CharacterProfile(
                id=character_id,
                display_name=name.strip(),
                gender=gender if gender in _GENDERS else None,
                spoken_characters=0,
                chapter_ids=(),
            ),
        )
    return tuple(sorted(seen.values(), key=lambda character: character.id))


def merge_rosters(
    rosters: tuple[tuple[CharacterProfile, ...], ...],
) -> tuple[CharacterProfile, ...]:
    """Combine per-chapter rosters, folding the names that are one person.

    A character named in ten chapters must be one entry, or casting would
    assign them ten voices. Exact-id matching is not enough for that: a model
    asked about one chapter answers "Tam" and about another "Tam al'Thor", and
    the two are one man. `identity` decides which pairs fold, and refuses the
    short forms that two people could claim.
    """
    from kenkui._characters.identity import (  # noqa: PLC0415 - avoids a cycle
        group_full_names,
        resolve_short_forms,
    )

    by_display: dict[str, CharacterProfile] = {}
    for roster in rosters:
        for character in roster:
            existing = by_display.get(character.display_name)
            if existing is None:
                by_display[character.display_name] = character
            elif existing.gender is None and character.gender is not None:
                by_display[character.display_name] = existing.__class__(
                    id=existing.id,
                    display_name=existing.display_name,
                    gender=character.gender,
                    spoken_characters=existing.spoken_characters,
                    chapter_ids=existing.chapter_ids,
                )

    names = sorted(by_display)
    entity = group_full_names([n for n in names if len(n.split()) > 1])
    resolved = resolve_short_forms([n for n in names if len(n.split()) == 1], entity)
    canonical = {**entity, **resolved.assigned}

    merged: dict[str, CharacterProfile] = {}
    for name, character in by_display.items():
        target = canonical.get(name)
        if target is None:
            # Ambiguous short form: two people could claim it, so it names
            # neither. See identity.resolve_short_forms.
            continue
        head = by_display[target]
        gender = head.gender or character.gender
        merged[head.id] = head.__class__(
            id=head.id,
            display_name=head.display_name,
            gender=gender,
            spoken_characters=head.spoken_characters,
            chapter_ids=head.chapter_ids,
        )
    return tuple(sorted(merged.values(), key=lambda character: character.id))
