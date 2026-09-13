"""A failed chapter is retried next time, and only that chapter is paid for again."""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING

import pytest

import kenkui as kk
from kenkui._characters import resolve_attribution, store

if TYPE_CHECKING:
    from collections.abc import Iterator

BOOK = "b" * 64
_ROSTER = "List the speaking characters"
_TEXTS = {
    "ch1": 'The inspector waited. "You are late," he said. Nobody moved.',
    "ch2": 'Rain fell on the barricade. "Hold the line," he said. They held.',
    "ch3": 'Morning came slowly. "It is over," he said. The street was empty.',
}


@pytest.fixture(autouse=True)
def _no_retry_backoff(monkeypatch: pytest.MonkeyPatch) -> None:
    """Retries still happen; only their waiting is skipped."""
    monkeypatch.setattr("kenkui._characters.llm.time.sleep", lambda _seconds: None)


class FlakyClient:
    """Answers every prompt, except attribution for chapters it is told to fail."""

    def __init__(
        self,
        failing: frozenset[str] = frozenset(),
        failing_rosters: frozenset[str] = frozenset(),
    ) -> None:
        """Fail attribution or roster calls for the named chapters."""
        self.failing = failing
        self.failing_rosters = failing_rosters
        self.calls: list[str] = []

    def complete(self, model: str, prompt: str) -> str:
        """Return a scripted reply, or raise like a provider outage."""
        assert model
        self.calls.append(prompt)
        if _ROSTER in prompt:
            if any(_TEXTS[chapter] in prompt for chapter in self.failing_rosters):
                message = "provider unavailable"
                raise ConnectionError(message)
            return json.dumps(
                {"characters": [{"id": "javert", "name": "Javert", "gender": None}]}
            )
        if any(_TEXTS[chapter] in prompt for chapter in self.failing):
            message = "provider unavailable"
            raise ConnectionError(message)
        return json.dumps({"attributions": [{"quote_id": 0, "speaker": "javert"}]})

    def attribution_calls(self) -> list[str]:
        """Return the attribution prompts, excluding roster discovery."""
        return [prompt for prompt in self.calls if _ROSTER not in prompt]


def _inspection() -> kk.BookInspection:
    chapters = tuple(
        kk.ChapterInspection(chapter_id, index, chapter_id, len(text), text)
        for index, (chapter_id, text) in enumerate(_TEXTS.items())
    )
    return kk.BookInspection(kk.BookMetadata("T", "A", cover_available=False), chapters)


def _speakers(record: store.AttributionRecord) -> dict[str, set[str | None]]:
    by_chapter: dict[str, set[str | None]] = {}
    for span in record.spans:
        if span.character_id is not None or span.chapter_id not in by_chapter:
            by_chapter.setdefault(span.chapter_id, set())
        if span.character_id is not None:
            by_chapter[span.chapter_id].add(span.character_id)
    return by_chapter


@pytest.fixture
def warnings(caplog: pytest.LogCaptureFixture) -> Iterator[pytest.LogCaptureFixture]:
    """Capture kenkui warnings."""
    with caplog.at_level(logging.WARNING, logger="kenkui"):
        yield caplog


def test_a_book_with_a_failed_chapter_is_rendered_but_not_stored(
    warnings: pytest.LogCaptureFixture,
) -> None:
    """Storing it would narrate that chapter's dialogue on every later render."""
    record = resolve_attribution(
        _inspection(), BOOK, "fake/model", client=FlakyClient(frozenset({"ch2"}))
    )

    assert _speakers(record) == {"ch1": {"javert"}, "ch2": set(), "ch3": {"javert"}}
    assert store.read_attribution(record.attribution_id) is None
    assert any(r.getMessage() == "attribution_incomplete" for r in warnings.records)


def test_the_next_run_pays_only_for_the_chapter_that_failed() -> None:
    """The chapters that succeeded are not bought twice."""
    first = FlakyClient(frozenset({"ch2"}))
    resolve_attribution(_inspection(), BOOK, "fake/model", client=first)

    healed = FlakyClient()
    record = resolve_attribution(_inspection(), BOOK, "fake/model", client=healed)

    assert len(healed.attribution_calls()) == 1
    assert _TEXTS["ch2"] in healed.attribution_calls()[0]
    assert _speakers(record) == {
        "ch1": {"javert"},
        "ch2": {"javert"},
        "ch3": {"javert"},
    }
    assert store.read_attribution(record.attribution_id) == record


def test_a_complete_book_is_stored_and_reused_without_model_calls() -> None:
    """A finished book is one stored record, as before."""
    resolve_attribution(_inspection(), BOOK, "fake/model", client=FlakyClient())
    again = FlakyClient()
    resolve_attribution(_inspection(), BOOK, "fake/model", client=again)
    assert again.calls == []


def test_a_response_is_reused_only_for_the_same_model() -> None:
    """Another model is different material, never a cache hit."""
    resolve_attribution(_inspection(), BOOK, "fake/model", client=FlakyClient())
    other = FlakyClient()
    resolve_attribution(_inspection(), BOOK, "fake/other-model", client=other)
    assert len(other.attribution_calls()) == len(_TEXTS)


def test_a_failed_roster_chapter_is_not_stored_and_is_retried_alone() -> None:
    """A roster missing a chapter's characters is as unfinished as a lost quote."""
    first = FlakyClient(failing_rosters=frozenset({"ch3"}))
    record = resolve_attribution(_inspection(), BOOK, "fake/model", client=first)
    assert store.read_attribution(record.attribution_id) is None

    healed = FlakyClient()
    resolve_attribution(_inspection(), BOOK, "fake/model", client=healed)
    rosters = [prompt for prompt in healed.calls if _ROSTER in prompt]
    assert len(rosters) == 1
    assert _TEXTS["ch3"] in rosters[0]
    assert store.read_attribution(record.attribution_id) is not None
