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

    Folding is layered on top of id matching, never instead of it. `id` is
    the model's stable book-wide identifier; `name` is only how the text
    names someone in one chapter, per `ROSTER_PROMPT`. So two distinct ids
    are grouped by id first, exactly as before identity-aware folding
    existed. Only then are the *distinct* ids folded into each other, by
    comparing their display names through `identity`. And two distinct ids
    that happen to display the exact same bare name are never folded into
    each other on that evidence alone: the model already told us, by giving
    them different ids, that they are not the same person, and a name
    collision is not grounds to override that.
    """
    from kenkui._characters.identity import (  # noqa: PLC0415 - identity is
        # only needed once a book has more than one roster to fold, so the
        # import is kept off the path of callers that never merge rosters.
        group_full_names,
        resolve_short_forms,
    )

    by_id: dict[str, CharacterProfile] = {}
    for roster in rosters:
        for character in roster:
            existing = by_id.get(character.id)
            if existing is None:
                by_id[character.id] = character
            elif existing.gender is None and character.gender is not None:
                by_id[character.id] = existing.__class__(
                    id=existing.id,
                    display_name=existing.display_name,
                    gender=character.gender,
                    spoken_characters=existing.spoken_characters,
                    chapter_ids=existing.chapter_ids,
                )

    ids_by_name: dict[str, list[str]] = {}
    for character_id, character in by_id.items():
        ids_by_name.setdefault(character.display_name, []).append(character_id)
    # Two distinct ids sharing one display name is the model's own signal
    # that they are two people wearing the same surface name in different
    # chapters (e.g. two characters each only ever called "Charles"). That
    # name is therefore excluded from identity resolution entirely: it must
    # neither fold those ids into each other nor act as a host that some
    # other name resolves to.
    contested = {name for name, ids in ids_by_name.items() if len(ids) > 1}

    usable_names = sorted(name for name in ids_by_name if name not in contested)
    entity = group_full_names([n for n in usable_names if len(n.split()) > 1])
    resolved = resolve_short_forms(
        [n for n in usable_names if len(n.split()) == 1], entity
    )
    canonical: dict[str, str] = {
        **entity,
        **resolved.assigned,
        # No identity signal either way: each contested id stands alone.
        **{name: name for name in contested},
    }

    name_to_id = {
        name: ids[0] for name, ids in ids_by_name.items() if name not in contested
    }

    merged: dict[str, CharacterProfile] = {}
    for character_id, character in by_id.items():
        target_name = canonical.get(character.display_name)
        if target_name is None:
            # Ambiguous short form: two people could claim it, so it names
            # neither. See identity.resolve_short_forms.
            continue
        head_id = name_to_id.get(target_name, character_id)
        head = by_id[head_id]
        existing = merged.get(head_id, head)
        gender = existing.gender if existing.gender is not None else character.gender
        merged[head_id] = head.__class__(
            id=head.id,
            display_name=head.display_name,
            gender=gender,
            spoken_characters=head.spoken_characters,
            chapter_ids=head.chapter_ids,
        )
    return tuple(sorted(merged.values(), key=lambda character: character.id))
