"""Deciding when two names are one character, and when they are two.

Pure functions over names: no model, no I/O. `merge_rosters` composes them.

Two failure directions with different costs. Over-merging puts two people in
one voice, which is the failure attribution is organised against. Under-merging
gives one person two voices, audible but locally consistent within any stretch
where a single name is used. Under-merging is therefore the default here, and
every rule below either prevents an over-merge or refuses to guess.
"""

from __future__ import annotations

from collections import Counter
from typing import TYPE_CHECKING, NamedTuple

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

# Honorifics that PRECEDE a name separate individuals: "Mr Elliot" and "Miss
# Elliot" are two Elliots, "Mr Geary" and "Mrs Geary" a husband and wife who
# both speak. Honorifics that FOLLOW attach to one person: "Moiraine Sedai"
# and "Moiraine Aes Sedai" are one Moiraine.
PREFIX_TITLES: frozenset[str] = frozenset(
    {
        "mr",
        "mrs",
        "miss",
        "ms",
        "master",
        "mistress",
        "lord",
        "lady",
        "sir",
        "dame",
        "dr",
        "doctor",
        "captain",
        "admiral",
        "colonel",
        "major",
        "general",
        "inspector",
        "sergeant",
        "king",
        "queen",
        "prince",
        "princess",
        "goodman",
        "goodwife",
        "mother",
        "father",
        "elder",
        "mayor",
    }
)
SUFFIX_TITLES: frozenset[str] = frozenset(
    {
        "sedai",
        "aes",
        "gaidin",
        "jr",
        "sr",
        "ii",
        "iii",
    }
)


class ShortForms(NamedTuple):
    """Where each bare name went, and which were too ambiguous to place."""

    assigned: dict[str, str]
    ambiguous: dict[str, list[str]]


def _tokens(name: str) -> set[str]:
    return {token.lower().strip(".") for token in name.split()}


def same_person(
    first: str, second: str, titles: frozenset[str] = PREFIX_TITLES
) -> bool:
    """Report whether two full names denote one person.

    Two differing prefix titles separate people. One title against none does
    not: "Brightlord Dalinar" and "Dalinar Kholin" are one Dalinar. With titles
    set aside, names denote one person when what remains is equal or nested;
    two different residues are two surnames, hence two people.
    """
    tokens_a, tokens_b = _tokens(first), _tokens(second)
    if tokens_a == tokens_b:
        return True
    prefix_a, prefix_b = tokens_a & titles, tokens_b & titles
    if prefix_a and prefix_b and prefix_a != prefix_b:
        return False
    rest_a = tokens_a - titles - SUFFIX_TITLES
    rest_b = tokens_b - titles - SUFFIX_TITLES
    return rest_a == rest_b or rest_a < rest_b or rest_b < rest_a


def detect_titles(names: Sequence[str], threshold: int = 3) -> frozenset[str]:
    """Find a book's own honorifics: leading tokens shared by many names.

    No fixed list holds every invented honorific, and a missed one splits a
    character in two. A token opening `threshold` or more distinct names is
    doing a title's job whatever the book calls it.
    """
    leading: Counter[str] = Counter()
    for name in names:
        parts = name.split()
        if len(parts) > 1:
            leading[parts[0].lower().strip(".")] += 1
    return PREFIX_TITLES | {
        token for token, count in leading.items() if count >= threshold
    }


def group_full_names(names: Sequence[str]) -> dict[str, str]:
    """Map each multi-token name to the entity that owns it."""
    titles = detect_titles(names)
    entity: dict[str, str] = {}
    for name in names:
        entity[name] = next(
            (
                entity[other]
                for other in names
                if other in entity and same_person(other, name, titles)
            ),
            name,
        )
    return entity


def resolve_short_forms(shorts: Sequence[str], entity: Mapping[str, str]) -> ShortForms:
    """Attach each bare name to its entity, or refuse when two could claim it.

    A refused short form is dropped rather than kept. Keeping it as its own
    entry would split one person into two voices; assigning it would merge two
    people into one. Dropping it leaves the model to name a full form from the
    passage, which it is better placed to do than any rule here.
    """
    assigned: dict[str, str] = {}
    ambiguous: dict[str, list[str]] = {}
    for short in shorts:
        token = short.lower().strip(".")
        hosts = {entity[full] for full in entity if token in _tokens(full)}
        if len(hosts) == 1:
            assigned[short] = next(iter(hosts))
        elif hosts:
            ambiguous[short] = sorted(hosts)
        else:
            assigned[short] = short
    return ShortForms(assigned, ambiguous)
