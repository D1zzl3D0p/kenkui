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
        same_person,
    )

    def _one_person(ids: list[str]) -> bool:
        """Whether every id sharing a display name denotes one person.

        Ids are the model's own answer to "who is this", so they carry the
        evidence the display name lacks. Read as names, `corwi` nests inside
        `lizbyet-corwi` and the two are one woman; `charles-hayter` and
        `charles-musgrove` share only a forename and stay two men.
        """
        names = [character_id.replace("-", " ") for character_id in ids]
        return all(same_person(names[0], other) for other in names[1:])

    def _fullest(ids: list[str]) -> str:
        """Pick the id carrying the most name, deterministically."""
        return max(sorted(ids), key=lambda value: len(value.split("-")))

    by_id: dict[str, CharacterProfile] = {}
    # by_id keeps one display name per id -- the first seen, gender aside --
    # but the same id can surface under different names in different
    # chapters ("Corwi" in one, "Lizbyet Corwi" in another). Every name is
    # worth keeping, so it is tracked here rather than left to fall out of
    # the dedup above.
    id_names: dict[str, set[str]] = {}
    for roster in rosters:
        for character in roster:
            id_names.setdefault(character.id, set()).add(character.display_name)
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
    # Two distinct ids sharing one display name is usually the model's own
    # signal that they are two people wearing the same surface name in
    # different chapters (e.g. two characters each only ever called
    # "Charles"). Such a name is excluded from identity resolution entirely:
    # it must neither fold those ids into each other nor act as a host that
    # some other name resolves to.
    #
    # But the ids themselves can say otherwise. A model asked for an id
    # "stable across the whole book" does not reliably give one, so one
    # person arrives under two ids that happen to share a surface name.
    # When those ids nest, they are that person twice, not two people, and
    # treating them as contested gives her two voices.
    contested = {
        name
        for name, ids in ids_by_name.items()
        if len(ids) > 1 and not _one_person(ids)
    }

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

    # The fullest id wins the name: "lizbyet-corwi" carries more of who she
    # is than "corwi", and the head's id is what the cast is keyed on.
    name_to_id = {
        name: _fullest(ids)
        for name, ids in ids_by_name.items()
        if name not in contested
    }

    merged: dict[str, CharacterProfile] = {}
    aliases: dict[str, set[str]] = {}
    for character_id, character in by_id.items():
        target_name = canonical.get(character.display_name)
        if target_name is None:
            # Ambiguous short form: two people could claim it, so it names
            # neither. See identity.resolve_short_forms.
            continue
        head_id = name_to_id.get(target_name, character_id)
        head = by_id[head_id]
        # Every display name folded under this head is a surface form series
        # matching will later need, not just the one the head kept.
        seen_names = aliases.setdefault(head_id, set())
        seen_names.update(id_names[character_id])
        seen_names.update(id_names[head_id])
        existing = merged.get(head_id, head)
        gender = existing.gender if existing.gender is not None else character.gender
        merged[head_id] = head.__class__(
            id=head.id,
            display_name=head.display_name,
            gender=gender,
            spoken_characters=head.spoken_characters,
            chapter_ids=head.chapter_ids,
            aliases=tuple(sorted(aliases[head_id])),
        )
    return tuple(sorted(merged.values(), key=lambda character: character.id))
