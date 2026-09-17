"""Gender evidence returned alongside quote attribution, independent of role IDs."""

from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import replace
from typing import TYPE_CHECKING

from kenkui.observability import get_logger, log_event

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

    from kenkui._domain.casting import CharacterProfile

_LOGGER = get_logger(__name__)


def merge(evidence: Iterable[tuple[str, str]]) -> tuple[tuple[str, str], ...]:
    """Keep only unanimous evidence for each identity across aliases and chapters."""
    votes: dict[str, set[str]] = defaultdict(set)
    for character_id, gender in evidence:
        votes[character_id].add(gender)
    result: list[tuple[str, str]] = []
    for character_id, genders in sorted(votes.items()):
        if len(genders) == 1:
            result.append((character_id, next(iter(genders))))
        else:
            log_event(
                _LOGGER,
                "attribution_gender_ambiguous",
                level=logging.WARNING,
                context={"boundary": "characters", "character": character_id},
            )
    return tuple(result)


def apply(
    characters: Sequence[CharacterProfile], evidence: Sequence[tuple[str, str]]
) -> tuple[CharacterProfile, ...]:
    """Apply contextual evidence before dialogue checks and reviewed overrides."""
    genders = dict(evidence)
    return tuple(
        replace(character, gender=genders.get(character.id, character.gender))
        for character in characters
    )
