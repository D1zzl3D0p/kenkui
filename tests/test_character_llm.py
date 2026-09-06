"""The model boundary: bounded retry, strict validation, no leaked book text."""
# ruff: noqa: D102, D103, D107 - test names carry the intent; the ones
# with a non-obvious rationale are documented individually.

from __future__ import annotations

import sys
from types import SimpleNamespace

import pytest

from kenkui._characters.llm import _LiteLLMClient, complete_json
from kenkui.cancellation import CancellationToken
from kenkui.errors import CancelledError, ErrorCode, ModelError


class FakeClient:
    """Deterministic stand-in. Records prompts so retry behaviour is visible."""

    def __init__(self, *responses: str | BaseException) -> None:
        self.responses = list(responses)
        self.calls: list[str] = []

    def complete(self, model: str, prompt: str) -> str:
        assert model
        self.calls.append(prompt)
        if not self.responses:
            message = "exhausted"
            raise AssertionError(message)
        reply = self.responses.pop(0)
        # BaseException, not Exception: KeyboardInterrupt is one of the
        # cases under test and does not inherit from Exception.
        if isinstance(reply, BaseException):
            raise reply
        return reply


def test_litellm_client_bounds_provider_request(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An unresponsive provider must not hold the batch forever."""

    def completion(**kwargs: object) -> SimpleNamespace:
        if kwargs.get("timeout") != _REQUEST_TIMEOUT_SECONDS:
            raise TimeoutError
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="answer"))]
        )

    monkeypatch.setitem(sys.modules, "litellm", SimpleNamespace(completion=completion))

    assert _LiteLLMClient().complete("openrouter/test", "prompt") == "answer"


def test_litellm_client_disables_reasoning(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Reasoning tokens dominate attribution latency; they must be off."""

    def completion(**kwargs: object) -> SimpleNamespace:
        if kwargs.get("reasoning_effort") != "none":
            raise TimeoutError
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="answer"))]
        )

    monkeypatch.setitem(sys.modules, "litellm", SimpleNamespace(completion=completion))

    assert _LiteLLMClient().complete("openrouter/test", "prompt") == "answer"


SCHEMA = {"items": list}
_RETRIED_ONCE = 2
_REQUEST_TIMEOUT_SECONDS = 600.0


def test_valid_json_is_returned() -> None:
    client = FakeClient('{"items": [1, 2]}')
    assert complete_json("fake/model", "p", SCHEMA, client=client) == {"items": [1, 2]}


def test_prose_around_the_json_is_tolerated() -> None:
    """Models wrap JSON in explanation despite being told not to."""
    client = FakeClient('Sure!\n```json\n{"items": []}\n```\nHope that helps.')
    assert complete_json("fake/model", "p", SCHEMA, client=client) == {"items": []}


def test_transient_failure_is_retried_then_succeeds() -> None:
    client = FakeClient(TimeoutError("network"), '{"items": []}')
    assert complete_json("fake/model", "p", SCHEMA, client=client) == {"items": []}
    assert len(client.calls) == _RETRIED_ONCE


def test_transport_failure_gives_up_and_reports_the_call() -> None:
    client = FakeClient(*[TimeoutError("network")] * 5)
    with pytest.raises(ModelError) as caught:
        complete_json("fake/model", "p", SCHEMA, client=client, backoff_base=0.0)
    assert caught.value.code is ErrorCode.MODEL_CALL_FAILED


def test_unparseable_output_is_reported_as_invalid() -> None:
    client = FakeClient("not json", "still not json", "nope")
    with pytest.raises(ModelError) as caught:
        complete_json("fake/model", "p", SCHEMA, client=client, backoff_base=0.0)
    assert caught.value.code is ErrorCode.MODEL_RESPONSE_INVALID


def test_schema_violation_is_rejected_not_coerced() -> None:
    """A string where a list belongs must fail, not silently become one."""
    client = FakeClient('{"items": "one"}')
    with pytest.raises(ModelError) as caught:
        complete_json("fake/model", "p", SCHEMA, client=client, backoff_base=0.0)
    assert caught.value.code is ErrorCode.MODEL_RESPONSE_INVALID


def test_missing_key_is_rejected() -> None:
    client = FakeClient('{"other": []}')
    with pytest.raises(ModelError):
        complete_json("fake/model", "p", SCHEMA, client=client, backoff_base=0.0)


def test_a_programmer_error_is_never_retried() -> None:
    """Retrying a bug wastes three provider calls to fail the same way."""
    client = FakeClient(TypeError("bug"))
    with pytest.raises(TypeError):
        complete_json("fake/model", "p", SCHEMA, client=client, backoff_base=0.0)
    assert len(client.calls) == 1


def test_cancellation_is_never_retried() -> None:
    """Ctrl-C during a long attribution pass must not be swallowed."""
    client = FakeClient(KeyboardInterrupt())
    with pytest.raises(KeyboardInterrupt):
        complete_json("fake/model", "p", SCHEMA, client=client, backoff_base=0.0)
    assert len(client.calls) == 1


def test_failures_do_not_leak_book_text(caplog: pytest.LogCaptureFixture) -> None:
    """Prompts carry the book; an error message must not republish it."""
    manuscript = "The unpublished manuscript sentence."
    client = FakeClient("not json", "not json", "not json")
    with pytest.raises(ModelError) as caught:
        complete_json("fake/model", manuscript, SCHEMA, client=client, backoff_base=0.0)
    assert manuscript not in str(caught.value)
    assert manuscript not in caplog.text


@pytest.mark.parametrize("response", ['{"items": []}', "invalid", None])
def test_cancelled_provider_call_does_not_retry(response: str | None) -> None:
    """A returned response or provider error must not conceal cancellation."""
    token = CancellationToken()

    class CancellingClient:
        """Cancel after entering the provider boundary."""

        calls = 0

        def complete(self, model: str, prompt: str) -> str:
            del model, prompt
            self.calls += 1
            token.cancel()
            if response is None:
                raise ConnectionError
            return response

    client = CancellingClient()
    with pytest.raises(CancelledError):
        complete_json("fake/model", "p", SCHEMA, client=client, cancel=token)
    assert client.calls == 1


def test_cancellation_interrupts_retry_backoff(monkeypatch: pytest.MonkeyPatch) -> None:
    """A retry delay checks cancellation without waiting out the whole delay."""
    token = CancellationToken()
    sleeps: list[float] = []

    def cancel_during_sleep(delay: float) -> None:
        sleeps.append(delay)
        token.cancel()

    monkeypatch.setattr("kenkui._characters.llm.time.sleep", cancel_during_sleep)
    client = FakeClient("invalid")
    with pytest.raises(CancelledError):
        complete_json("fake/model", "p", SCHEMA, client=client, cancel=token)
    assert len(sleeps) == 1
    assert sleeps == [pytest.approx(0.1)]
