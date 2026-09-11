"""A roster entry before it becomes a character, and who may reshape the list."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping, Sequence


@dataclass(frozen=True, slots=True)
class RosterEntry:
    """One candidate character and every surface form folded into it."""

    display_name: str
    aliases: frozenset[str]
    mentions: int
    evidence: int


def most_mentioned(names: Iterable[str], mentions: Mapping[str, int]) -> str:
    """Return the most-used name, breaking ties alphabetically."""
    return max(sorted(names), key=lambda name: mentions.get(name, 0))


class IdentityResolver(Protocol):
    """Decide which entries are one person and which are not people."""

    def resolve(
        self, entries: Sequence[RosterEntry], text: str, mentions: Mapping[str, int]
    ) -> tuple[RosterEntry, ...] | None:
        """Return reshaped entries, or None to use the offline fallback."""
        ...
