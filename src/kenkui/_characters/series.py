"""Deciding when a character in this volume is one the series already knows.

Pure functions over names and records: no model, no I/O. The rules are the
ones `identity` already applies within a book, which matters more here --
a series has more names competing for the same short forms than any single
volume does, so over-merging is likelier and costs more.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from kenkui._characters.identity import (
    SUFFIX_TITLES,
    detect_titles,
    same_person,
)
from kenkui._characters.infer import ROLE_PREFIX
from kenkui._characters.store import SeriesCharacter, SeriesRecord

if TYPE_CHECKING:
    from collections.abc import Container, Mapping, Sequence

    from kenkui._domain.casting import CharacterProfile


def match_roster(
    record: SeriesRecord | None, characters: Sequence[CharacterProfile]
) -> dict[str, str]:
    """Map this volume's character ids onto the series' canonical ids.

    Every name this character is known by is compared to every name the
    series knows, and a match counts only when it lands on exactly one
    series character -- two candidates means the name belongs to neither,
    which is `resolve_short_forms`' rule and the reason a series does not
    quietly merge its two Charleses. An exact surface form is no exception:
    two series characters can legitimately share one, and a shared alias is
    exactly the ambiguity that refuses.

    `_may_host` decides each individual comparison, and is directional --
    see there for why a symmetric one hands a son his father's voice.
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
            if _may_host(alias, name, titles)
            for canonical in owners
        }
        if len(hosts) == 1:
            matched[character.id] = next(iter(hosts))

    # Two characters in *this* volume can each independently reach the same
    # series person -- a maid named "Elizabeth" and the protagonist "Lizzy"
    # both land on the series' one Elizabeth Bennet. Attaching either would
    # discard one of this volume's own solved voices to wear hers, which is
    # the over-merge this module exists to refuse. A canonical claimed by
    # more than one local character is withheld from all of its claimants;
    # each keeps its own, separately-solved voice instead.
    claimants: dict[str, list[str]] = {}
    for character_id, canonical in matched.items():
        claimants.setdefault(canonical, []).append(character_id)
    return {
        character_id: canonical
        for character_id, canonical in matched.items()
        if len(claimants[canonical]) == 1
    }


def _residue(name: str, titles: frozenset[str]) -> frozenset[str]:
    """Return the tokens that identify a person, titles set aside.

    The same reduction ``same_person`` compares by, exposed so the *shape*
    of a name can be judged as well as its content: "Kholin" carries one
    identifying token, "Adolin Kholin" two.
    """
    tokens = {token.lower().strip(".") for token in name.split()}
    return frozenset(tokens - titles - SUFFIX_TITLES)


def _may_host(alias: str, name: str, titles: frozenset[str]) -> bool:
    """Whether one name the series knows may stand for one name in this volume.

    ``same_person`` is symmetric and reads nesting as identity, which is
    right when both sides are full names and wrong when the series' side is
    bare. ``merge_rosters`` records bare surnames as a matter of course --
    volume one's Dalinar contributes "Kholin" alongside "Dalinar Kholin" --
    so a symmetric rule lets that lone token host volume three's "Adolin
    Kholin" and pins a son to his father's voice, permanently.

    Inside one book that direction cannot happen: ``resolve_short_forms``
    attaches a short form TO a full name and never the reverse. This
    mirrors it. A bare series name hosts only a bare incoming one; an
    incoming name carrying more than one identifying token must find a
    series name that carries more than one too, where the surnames
    discriminate -- ``same_person("Dalinar Kholin", "Adolin Kholin")`` is
    already False.

    The cost is a series that only ever saw a bare name failing to claim a
    later full one. That is an under-merge -- one person, two voices -- and
    it is the failure ``identity`` is built to prefer.
    """
    if alias == name:
        return True
    if not same_person(alias, name, titles):
        return False
    return len(_residue(alias, titles)) > 1 or len(_residue(name, titles)) <= 1


def series_members(
    characters: Sequence[CharacterProfile],
) -> tuple[CharacterProfile, ...]:
    """Return the characters a series should remember, dropping minted roles.

    A ``role:<slug>@<chapter_id>`` id is minted for a speaker no roster
    listed, and is scoped to its chapter on purpose: chapter 40's officer is
    not chapter 12's. Carrying one into the series inverts that decision
    across volumes -- volume one's officer would be merged with volume
    four's -- for parts that are one scene long by construction.

    It is also why an ordinary first volume grew on every re-render: every
    volume mints the same slugs, so the roles collided with themselves and
    minted a fresh canonical each time. They are cast normally inside their
    own book; they are simply not people a series knows.
    """
    return tuple(
        character
        for character in characters
        if not character.id.startswith(ROLE_PREFIX)
    )


def _claimed_slots(base: str, known: Mapping[str, SeriesCharacter]) -> list[str]:
    """Return the ids the series already holds under ``base``, in mint order."""
    if base not in known:
        return []
    slots = [base]
    suffix = 2
    while f"{base}-{suffix}" in known:
        slots.append(f"{base}-{suffix}")
        suffix += 1
    return slots


def _mint_canonical(
    base: str,
    known: Mapping[str, SeriesCharacter],
    book_digest: str | None = None,
    claimed: Container[str] = frozenset(),
) -> str:
    """Return the id this series holds for a character no name matched.

    Character ids are slugged from display names alone (see ``infer.py``),
    so two unrelated people in different volumes can land on the same raw
    id even though ``match_roster`` found no name link between them --
    two "Guard"s, say. Handing the newcomer the raw id anyway would fold
    them into whoever already holds it with no name comparison involved;
    minting a fresh, deterministic id keeps them the two people they are.

    A re-render is not a newcomer, and must not be mistaken for one.
    ``match_roster`` withholds a canonical that two of this volume's
    characters both reach, so the same character can be unmatched on every
    render of the same volume; minting each time gave ``guard``, then
    ``guard-2``, then ``guard-3``, without bound -- and each fresh mint
    carried an empty ledger, so ``_accumulate`` never got the chance to
    replace anything and the ``book_digest`` idempotency was defeated by
    the very path meant to be safe. An id already carrying a contribution
    from THIS volume is that same character seen again, so it is reused.

    That ledger test only holds across calls. Within one call it says the
    slot belongs to *some* character of this volume, which is no longer
    evidence about *this* one, because this call has been writing the same
    digest into every slot it touches: a "Guard" minting ``guard-2``
    earlier in this very roster leaves its digest there for a later
    "Warden" whose raw id happens to be ``guard-2`` to find, and the two
    become one person on one voice, permanently and silently. ``claimed``
    is what this call has already handed out, and is excluded from the
    scan -- a slot spoken for by another character of this same volume is
    by construction not this character seen again. Keying the ledger on
    the source character id instead would tell the two apart directly, but
    it rewrites a stored schema (and its migration) to answer a question
    that is only ever asked about one in-flight call.
    """
    slots = _claimed_slots(base, known)
    if book_digest is not None:
        for canonical in slots:
            if canonical in claimed:
                continue
            if book_digest in dict(known[canonical].contributions):
                return canonical
    return base if not slots else f"{base}-{len(slots) + 1}"


def _accumulate(
    existing: SeriesCharacter, spoken_characters: int, book_digest: str | None
) -> tuple[int, tuple[tuple[str, int], ...]]:
    """Fold in one character's speech, replacing rather than repeating a volume.

    Untracked contributions (``book_digest`` was never supplied, on this call
    or any earlier one that reached this canonical id) still just add: there
    is nothing to compare the new amount against. A named volume that has
    contributed before has its old amount removed before the new one is
    added, so re-merging it lands on the new total rather than the sum of
    both.
    """
    if book_digest is None:
        return existing.spoken_characters + spoken_characters, existing.contributions
    ledger = dict(existing.contributions)
    total = existing.spoken_characters - ledger.get(book_digest, 0) + spoken_characters
    ledger[book_digest] = spoken_characters
    return total, tuple(sorted(ledger.items()))


def merged_series(  # noqa: PLR0913 - one call site, every input explicit.
    record: SeriesRecord | None,
    characters: Sequence[CharacterProfile],
    assignments: Mapping[str, str],
    narrator_voice_id: str,
    series_id: str,
    *,
    book_digest: str | None = None,
) -> SeriesRecord:
    """Return the series as it stands after this volume.

    A returning character keeps the voice ``assignments`` actually gave them
    this render and gains this volume's speech and surface forms. A newcomer
    joins with whatever voice the solver just chose.

    Recording ``assignments`` rather than re-asserting the series' own prior
    voice_id is deliberate, not redundant: ordinarily they agree, because the
    caller pins a returning character to the series' voice before solving.
    They can legitimately disagree two ways -- a dropped pin the solver had
    to recast because the pool could no longer honour it, or this render's
    caller overriding the series pin with an explicit ``cast=`` -- and both
    are cases where what was actually spoken must become the series' new
    voice for that person. Keeping the stale value instead would mean the
    series never converges: the next render pins the same unavailable voice,
    drops it again, and may recast to something else again.

    The narrator is adopted from ``narrator_voice_id`` the same way, on every
    call, not only the first: a caller who rendered under
    ``allow_narrator_change`` used a different narrator for this volume, and
    the series must record what was actually narrated, not what an earlier
    volume happened to use.

    ``book_digest`` identifies the volume whose speech is being folded in --
    the same content hash attribution is keyed by, so an edited copy of a
    volume is a different one. Without it, speech accumulates blindly: two
    calls over the same volume's assignments add its total twice. Naming the
    volume lets a repeat call replace what it contributed last time rather
    than pile on top of it, which is what a re-render is -- the ordinary way
    to render a book, since ``resolve()`` and ``write()`` both call this on
    every run, not just the first. Omitting it keeps the old, purely additive
    behaviour exactly, for callers with no volume identity to offer.
    """
    matched = match_roster(record, characters)
    known = {c.canonical_id: c for c in (record.characters if record else ())}
    # Every canonical this call has spoken for. Two characters of one volume
    # are two people, so a slot claimed here is closed to the rest of them --
    # see `_mint_canonical`.
    #
    # Seeded with every matched canonical rather than filled as the loop
    # goes, because `matched` is already fully known and the loop is not.
    # A character that mints early would otherwise be free to take a slot a
    # later-iterated character matches to by name, which is the same
    # over-merge one step over: ids sort before names, so the minting
    # character genuinely can run first.
    claimed: set[str] = set(matched.values())
    for character in characters:
        voice_id = assignments.get(character.id)
        if voice_id is None:
            continue
        # A canonical from match_roster is a name-checked return; falling
        # back to the raw id is only safe once it is confirmed free, or
        # confirmed to be this same volume's own earlier mint, since an
        # unmatched character's id can coincide with someone else's.
        canonical = matched.get(character.id)
        if canonical is None:
            canonical = _mint_canonical(character.id, known, book_digest, claimed)
        claimed.add(canonical)
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
                contributions=((book_digest, character.spoken_characters),)
                if book_digest is not None
                else (),
            )
            continue
        total, contributions = _accumulate(
            existing, character.spoken_characters, book_digest
        )
        known[canonical] = SeriesCharacter(
            canonical_id=canonical,
            display_name=existing.display_name,
            # The series keeps the first gender it was sure of: a later
            # volume answering None must not un-gender a cast character.
            gender=existing.gender if existing.gender is not None else character.gender,
            # `voice_id`, not `existing.voice_id`: see the docstring above.
            voice_id=voice_id,
            spoken_characters=total,
            aliases=tuple(sorted({*existing.aliases, *aliases})),
            contributions=contributions,
        )
    return SeriesRecord(
        series_id=series_id,
        narrator_voice_id=narrator_voice_id,
        characters=tuple(
            sorted(known.values(), key=lambda c: (-c.spoken_characters, c.canonical_id))
        ),
    )
