"""The identity reasoning pass."""
# ruff: noqa: D101, D102, D103, D107, PLR2004

from __future__ import annotations

import json

from kenkui._characters.entries import RosterEntry
from kenkui._characters.identity_pass import (
    Decision,
    IdentityPass,
    Verdict,
    agree,
    apply,
    build_prompt,
    excerpts,
    parse,
)
from kenkui._characters.prompts import IDENTITY_PROMPT

TEXT = " ".join(f"Paul spoke to Usul number {index}." for index in range(8))
ENTRIES = (
    RosterEntry("Paul", frozenset({"Paul", "Paul Atreides"}), 100, 90),
    RosterEntry("Usul", frozenset({"Usul"}), 40, 30),
    RosterEntry("Arrakis", frozenset({"Arrakis"}), 20, 5),
)
MENTIONS = {"Paul": 90, "Paul Atreides": 10, "Usul": 40, "Arrakis": 20}


class ScriptedClient:
    def __init__(self, *answers: str) -> None:
        self.answers = list(answers)
        self.prompts: list[str] = []

    def complete(self, model: str, prompt: str) -> str:
        assert model
        self.prompts.append(prompt)
        return self.answers.pop(0)


def test_excerpts_are_three_windows_at_the_quartiles() -> None:
    found = excerpts("Paul", TEXT)
    assert len(found) == 3
    assert all("Paul" in window for window in found)


def test_prompt_numbers_entries_with_other_names_and_excerpts() -> None:
    prompt = build_prompt(ENTRIES, TEXT, MENTIONS)
    assert prompt.startswith(IDENTITY_PROMPT)
    assert '1. Paul (100 mentions; also "Paul Atreides"): "' in prompt
    assert "2. Usul (40 mentions):" in prompt


def test_parse_discards_what_it_cannot_use() -> None:
    verdict = parse(
        {
            "same_person": [[1, 2], [3, 3], [9, 1], "x"],
            "not_individuals": [3, 0, True, "2"],
        },
        3,
    )
    assert verdict.groups == ((0, 1),)
    assert verdict.excluded == frozenset({1, 2})


def test_agreement_keeps_only_what_both_runs_say() -> None:
    first = Verdict(((0, 1),), frozenset({2}))
    second = Verdict(((0, 1), (1, 2)), frozenset())
    decision = agree(first, second)
    assert decision.pairs == frozenset({frozenset({0, 1})})
    assert decision.excluded == frozenset()


def test_a_pair_touching_an_excluded_entry_is_dropped() -> None:
    both = Verdict(((0, 2),), frozenset({2}))
    decision = agree(both, both)
    assert decision.pairs == frozenset()
    assert decision.excluded == frozenset({2})


def test_apply_merges_and_excludes() -> None:
    merged = apply(ENTRIES, Decision(frozenset({frozenset({0, 1})}), frozenset({2})))
    assert len(merged) == 1
    assert merged[0].display_name == "Paul"
    assert merged[0].aliases == frozenset({"Paul", "Paul Atreides", "Usul"})
    assert merged[0].mentions == 140


def test_resolve_applies_what_two_runs_agree_on() -> None:
    answer = json.dumps({"same_person": [[1, 2]], "not_individuals": [3]})
    client = ScriptedClient(answer, answer)
    result = IdentityPass("m", client=client, backoff_base=0).resolve(
        ENTRIES, TEXT, MENTIONS
    )
    assert result is not None
    assert {entry.display_name for entry in result} == {"Paul"}
    assert len(client.prompts) == 2


class BrokenClient:
    def complete(self, model: str, prompt: str) -> str:
        del model, prompt
        return "not json"


def test_resolve_returns_none_when_the_runs_fail() -> None:
    assert (
        IdentityPass("m", client=BrokenClient(), backoff_base=0).resolve(
            ENTRIES, TEXT, MENTIONS
        )
        is None
    )


def test_a_decision_is_reused_without_calling_the_model() -> None:
    answer = json.dumps({"same_person": [[1, 2]], "not_individuals": [3]})
    first = ScriptedClient(answer, answer)
    IdentityPass("cached-model", client=first, backoff_base=0).resolve(
        ENTRIES, TEXT, MENTIONS
    )
    second = ScriptedClient()
    result = IdentityPass("cached-model", client=second, backoff_base=0).resolve(
        ENTRIES, TEXT, MENTIONS
    )
    assert result is not None
    assert {entry.display_name for entry in result} == {"Paul"}
    assert second.prompts == []


def test_a_different_model_is_a_different_decision() -> None:
    from kenkui._characters import store  # noqa: PLC0415
    from kenkui._characters.prompts import PROMPT_VERSION  # noqa: PLC0415

    assert store.identity_key("p", "a", "high", PROMPT_VERSION) != store.identity_key(
        "p", "b", "high", PROMPT_VERSION
    )
