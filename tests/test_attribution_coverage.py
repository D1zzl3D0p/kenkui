"""Incomplete quote lists are repaired before they can enter either cache."""
# ruff: noqa: D103 - test names describe the behavior being verified

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import pytest

import kenkui as kk
from kenkui._characters import attribution as module
from kenkui._characters import resolve_attribution, store
from kenkui._characters.attribution import attribute_chapter
from kenkui._characters.llm import DEFAULT_ATTEMPTS
from kenkui._characters.models import CharacterRoster
from kenkui._domain.casting import CharacterProfile
from kenkui.errors import ErrorCode, ModelError

if TYPE_CHECKING:
    from collections.abc import Iterator

COUNT = 409
SHORT_COUNT = 73
MODEL = "fake/coverage"
PROFILE = CharacterProfile("reader", "Reader", None, 0, ())


def _reply(count: int) -> str:
    return json.dumps(
        {
            "attributions": [
                {"quote_id": index, "speaker": "unknown"} for index in range(count)
            ]
        }
    )


def _chapter(count: int = COUNT) -> kk.ChapterInspection:
    text = "\n\n".join(f'"Line {index}," the reader said.' for index in range(count))
    return kk.ChapterInspection("chapter", 0, "Chapter", len(text), text)


class Replies:
    """Return supplied model answers while recording every actual request."""

    def __init__(self, *replies: str) -> None:
        """Supply one response per expected request."""
        self.replies: Iterator[str] = iter(replies)
        self.calls: list[str] = []

    def complete(self, model: str, prompt: str) -> str:
        """Fail the test if retrying makes more requests than supplied."""
        assert model == MODEL
        self.calls.append(prompt)
        return next(self.replies)


@pytest.fixture(autouse=True)
def no_wait(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep retries real but skip their clock delay."""
    monkeypatch.setattr("kenkui._characters.llm.time.sleep", lambda _: None)


def test_short_response_is_retried_unchanged_and_only_complete_reply_cached() -> None:
    client = Replies(_reply(SHORT_COUNT), _reply(COUNT))
    chapter = _chapter()
    _, coverage, _ = attribute_chapter(chapter, (PROFILE,), MODEL, client=client)
    assert coverage.unknown == COUNT
    assert coverage.dropped == 0
    assert client.calls == [client.calls[0]] * 2
    key = store.response_key(MODEL, client.calls[0], {"attributions": list})
    assert store.read_response(key) == json.loads(_reply(COUNT))
    attribute_chapter(chapter, (PROFILE,), MODEL, client=Replies())


def test_old_short_response_cache_is_revalidated_and_replaced() -> None:
    chapter = _chapter()
    seed = Replies(_reply(COUNT))
    attribute_chapter(chapter, (PROFILE,), MODEL, client=seed)
    key = store.response_key(MODEL, seed.calls[0], {"attributions": list})
    store.write_response(key, MODEL, json.loads(_reply(SHORT_COUNT)))
    healed = Replies(_reply(COUNT))
    _, coverage, _ = attribute_chapter(chapter, (PROFILE,), MODEL, client=healed)
    assert len(healed.calls) == 1
    assert coverage.dropped == 0
    assert store.read_response(key) == json.loads(_reply(COUNT))


def test_exhausted_incomplete_replies_raise_before_saving_book() -> None:
    chapter = _chapter()
    inspection = kk.BookInspection(
        kk.BookMetadata("T", "A", cover_available=False), (chapter,)
    )
    client = Replies(*([_reply(SHORT_COUNT)] * DEFAULT_ATTEMPTS))
    with pytest.raises(ModelError) as caught:
        resolve_attribution(
            inspection,
            "a" * 64,
            MODEL,
            client=client,
            roster=CharacterRoster((PROFILE,)),
        )
    assert caught.value.code is ErrorCode.MODEL_RESPONSE_INVALID
    assert len(client.calls) == DEFAULT_ATTEMPTS
    key = store.response_key(MODEL, client.calls[0], {"attributions": list})
    assert store.read_response(key) is None
    # A later successful attempt must pay for a fresh response, not hit an aggregate.
    healed = Replies(_reply(COUNT))
    resolve_attribution(
        inspection,
        "a" * 64,
        MODEL,
        client=healed,
        roster=CharacterRoster((PROFILE,)),
    )
    assert len(healed.calls) == 1


@pytest.mark.parametrize(
    "entries",
    [
        [],
        [{"quote_id": 0, "speaker": "unknown"}] * 2,
        [
            {"quote_id": False, "speaker": "unknown"},
            {"quote_id": 1, "speaker": "unknown"},
        ],
        [
            {"quote_id": "0", "speaker": "unknown"},
            {"quote_id": 1, "speaker": "unknown"},
        ],
        [{"quote_id": 0, "speaker": "unknown"}, {"quote_id": 2, "speaker": "unknown"}],
        [{"quote_id": 0, "speaker": "unknown"}, {"quote_id": 1}],
        [{"quote_id": 0, "speaker": "unknown"}, {"quote_id": 1, "speaker": " "}],
        [{"quote_id": 0, "speaker": "unknown"}, None],
    ],
)
def test_invalid_id_or_answer_cannot_satisfy_coverage(entries: list[object]) -> None:
    client = Replies(json.dumps({"attributions": entries}), _reply(2))
    _, coverage, _ = attribute_chapter(_chapter(2), (PROFILE,), MODEL, client=client)
    assert coverage.unknown == coverage.quotes
    assert client.calls == [client.calls[0]] * 2


def test_old_aggregate_does_not_bypass_coverage_check(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import kenkui._characters as characters  # noqa: PLC0415 - patch module policy

    chapter = _chapter()
    inspection = kk.BookInspection(
        kk.BookMetadata("T", "A", cover_available=False), (chapter,)
    )
    roster = CharacterRoster((PROFILE,))
    with monkeypatch.context() as old:
        old.setattr(characters, "PARAMS", {"temperature": 0.0})
        old.setattr(module, "_validate_coverage", lambda *_args, **_kwargs: None)
        previous = resolve_attribution(
            inspection,
            "b" * 64,
            MODEL,
            client=Replies(_reply(SHORT_COUNT)),
            roster=roster,
        )
    assert store.read_attribution(previous.attribution_id) is not None
    healed = Replies(_reply(COUNT))
    current = resolve_attribution(
        inspection, "b" * 64, MODEL, client=healed, roster=roster
    )
    assert current.attribution_id != previous.attribution_id
    assert len(healed.calls) == 1
