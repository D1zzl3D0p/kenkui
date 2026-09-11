"""Use two reasoning runs to merge identities and exclude non-people."""

from __future__ import annotations

import re
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from itertools import combinations
from typing import TYPE_CHECKING, Any

from kenkui._characters.entries import RosterEntry, most_mentioned
from kenkui._characters.llm import DEFAULT_BACKOFF_BASE, complete_json, reasoning_client
from kenkui._characters.prompts import IDENTITY_PROMPT
from kenkui.errors import ModelError
from kenkui.observability import get_logger, log_event

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from kenkui._characters.llm import Client
    from kenkui.cancellation import CancellationToken

_LOGGER = get_logger(__name__)
IDENTITY_REASONING = "high"
_RUNS = 2
_RADIUS = 110
_SCHEMA: Mapping[str, type] = {"same_person": list}


@dataclass(frozen=True, slots=True)
class Verdict:
    """One run's answer, as zero-based entry indices."""

    groups: tuple[tuple[int, ...], ...]
    excluded: frozenset[int]


@dataclass(frozen=True, slots=True)
class Decision:
    """The pairs and exclusions on which both runs agree."""

    pairs: frozenset[frozenset[int]]
    excluded: frozenset[int]


def excerpts(name: str, text: str) -> list[str]:
    """Return three windows around the name at its quartiles."""
    pattern = re.escape(name).replace("'", "['\u2019\u2018\u02bc]")
    hits = [match.start() for match in re.finditer(rf"(?<!\w){pattern}(?!\w)", text)]
    if not hits:
        return []
    count = len(hits)
    picks = sorted({hits[count // 4], hits[count // 2], hits[(3 * count) // 4]})
    return [
        " ".join(text[max(0, index - _RADIUS) : index + len(name) + _RADIUS].split())
        for index in picks
    ]


def build_prompt(
    entries: Sequence[RosterEntry], text: str, mentions: Mapping[str, int]
) -> str:
    """Build the numbered cast list in the order given."""
    lines = []
    for number, entry in enumerate(entries, start=1):
        head = most_mentioned(entry.aliases, mentions)
        others = sorted(entry.aliases - {head})
        also = f'; also "{", ".join(others)}"' if others else ""
        quoted = " | ".join(f'"{window}"' for window in excerpts(head, text))
        lines.append(f"{number}. {head} ({entry.mentions} mentions{also}): {quoted}")
    return IDENTITY_PROMPT + "\n".join(lines) + "\n"


def _number(value: object, count: int) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int | str):
        return None
    candidate = str(value).strip()
    if not candidate.isdigit() or not 1 <= int(candidate) <= count:
        return None
    return int(candidate) - 1


def parse(payload: Mapping[str, Any], count: int) -> Verdict:
    """Keep valid groups of two or more and valid exclusions."""
    groups = []
    for raw in payload.get("same_person") or []:
        if not isinstance(raw, list):
            continue
        members = sorted(
            {
                index
                for index in (_number(item, count) for item in raw)
                if index is not None
            }
        )
        if len(members) >= 2:  # noqa: PLR2004
            groups.append(tuple(members))
    excluded = frozenset(
        index
        for index in (
            _number(item, count) for item in payload.get("not_individuals") or []
        )
        if index is not None
    )
    return Verdict(tuple(groups), excluded)


def _pairs(verdict: Verdict) -> frozenset[frozenset[int]]:
    return frozenset(
        frozenset(pair) for group in verdict.groups for pair in combinations(group, 2)
    )


def agree(first: Verdict, second: Verdict) -> Decision:
    """Keep only pairs and exclusions returned by both runs."""
    excluded = first.excluded & second.excluded
    pairs = frozenset(
        pair for pair in _pairs(first) & _pairs(second) if not pair & excluded
    )
    return Decision(pairs, excluded)


def apply(
    entries: Sequence[RosterEntry], decision: Decision
) -> tuple[RosterEntry, ...]:
    """Apply agreed merges and exclusions, naming a group by its biggest entry."""
    parent = list(range(len(entries)))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    for pair in sorted(decision.pairs, key=sorted):
        first, second = sorted(pair)
        parent[find(first)] = find(second)
    groups: dict[int, list[int]] = defaultdict(list)
    for index in range(len(entries)):
        if index not in decision.excluded:
            groups[find(index)].append(index)
    merged = []
    for members in groups.values():
        biggest = max(
            sorted(members, key=lambda index: entries[index].display_name),
            key=lambda index: entries[index].mentions,
        )
        merged.append(
            RosterEntry(
                display_name=entries[biggest].display_name,
                aliases=frozenset().union(
                    *(entries[index].aliases for index in members)
                ),
                mentions=sum(entries[index].mentions for index in members),
                evidence=sum(entries[index].evidence for index in members),
            )
        )
    return tuple(
        sorted(merged, key=lambda entry: (-entry.evidence, entry.display_name))
    )


class IdentityPass:
    """Run identity reasoning twice and apply only the intersection."""

    def __init__(
        self,
        model_id: str,
        *,
        client: Client | None = None,
        cancel: CancellationToken | None = None,
        backoff_base: float = DEFAULT_BACKOFF_BASE,
    ) -> None:
        self._model_id = model_id
        self._client = client
        self._cancel = cancel
        self._backoff_base = backoff_base

    def resolve(
        self, entries: Sequence[RosterEntry], text: str, mentions: Mapping[str, int]
    ) -> tuple[RosterEntry, ...] | None:
        """Return reshaped entries, or None when either run fails."""
        ordered = sorted(
            entries, key=lambda entry: (-entry.mentions, entry.display_name)
        )
        decision = self._decide(build_prompt(ordered, text, mentions), len(ordered))
        return None if decision is None else apply(ordered, decision)

    def _decide(self, prompt: str, count: int) -> Decision | None:
        caller = self._client or reasoning_client(IDENTITY_REASONING)
        with ThreadPoolExecutor(max_workers=_RUNS) as pool:
            futures = [
                pool.submit(
                    complete_json,
                    self._model_id,
                    prompt,
                    _SCHEMA,
                    client=caller,
                    cancel=self._cancel,
                    backoff_base=self._backoff_base,
                )
                for _ in range(_RUNS)
            ]
            try:
                payloads = [future.result() for future in futures]
            except ModelError:
                log_event(
                    _LOGGER,
                    "identity_pass_failed",
                    context={"boundary": "characters", "model": self._model_id},
                )
                return None
        return agree(parse(payloads[0], count), parse(payloads[1], count))
