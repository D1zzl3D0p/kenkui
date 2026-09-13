"""The single model-calling boundary.

One narrow function: send a prompt, get validated JSON back, or raise. No
provider framework and no proxy — LiteLLM is called directly with a model
identifier carried in pipeline intent, and credentials are resolved from the
environment at execution.

Runs in the parent process only. The render workers deny sockets outright, so
nothing here can reach them.
"""

from __future__ import annotations

import json
import re
import time
from typing import TYPE_CHECKING, Any, Protocol

from kenkui.errors import CancelledError, ErrorCode, ModelError
from kenkui.observability import LogContext, get_logger, log_event

if TYPE_CHECKING:
    from collections.abc import Mapping

    from kenkui.cancellation import CancellationToken

_LOGGER = get_logger(__name__)

DEFAULT_ATTEMPTS = 3
DEFAULT_BACKOFF_BASE = 2.0
_REQUEST_TIMEOUT_SECONDS = 600.0

# Retrying a bug burns provider calls to fail the same way, and retrying an
# interrupt swallows a Ctrl-C during what may be a very long attribution pass.
_NEVER_RETRY = (
    KeyboardInterrupt,
    SystemExit,
    TypeError,
    AttributeError,
    NameError,
    CancelledError,
)

# Models wrap JSON in prose and fences despite being told not to.
_FENCE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL)


class Client(Protocol):
    """Minimal completion boundary, so tests never need a provider."""

    def complete(self, model: str, prompt: str) -> str:
        """Return the model's raw text response."""
        ...


class _LiteLLMClient:
    """Direct LiteLLM call at fixed parameters."""

    def __init__(self, reasoning_effort: str = "none") -> None:
        self._reasoning_effort = reasoning_effort

    def complete(self, model: str, prompt: str) -> str:
        """Return one completion, imported lazily to keep import cheap."""
        import litellm  # noqa: PLC0415 - heavy, and unused unless casting runs

        extra: dict[str, Any] = {}
        if model.startswith("openrouter/"):
            # LiteLLM's price table lags OpenRouter's catalogue, so the charge
            # OpenRouter itself reports is the only reliable number, and it is
            # returned only when asked for.
            extra["extra_body"] = {"usage": {"include": True}}
        started = time.monotonic()
        response = litellm.completion(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            timeout=_REQUEST_TIMEOUT_SECONDS,
            # Reasoning tokens dominated attribution latency; these calls are
            # extraction, not deliberation.
            reasoning_effort=self._reasoning_effort,
            **extra,
        )
        _log_usage(model, response, int((time.monotonic() - started) * 1000))
        return str(response.choices[0].message.content)


def _log_usage(model: str, response: object, elapsed_ms: int) -> None:
    """Log what one call consumed, as the provider reported it.

    Kenkui does not price anything; this records the provider's own figures
    so an operator can see what a book cost. A charge that cannot be
    established is left out rather than logged as zero, which would quietly
    understate it. Nothing from the prompt or response content is logged.
    """
    usage = getattr(response, "usage", None)
    context: dict[str, LogContext] = {
        "boundary": "characters",
        "model": model,
        "elapsed_ms": elapsed_ms,
    }
    for name, field in (
        ("input_tokens", "prompt_tokens"),
        ("output_tokens", "completion_tokens"),
    ):
        count = getattr(usage, field, None)
        if isinstance(count, int) and not isinstance(count, bool):
            context[name] = count
    cost = _reported_cost(response, usage)
    if cost is not None:
        context["cost_usd"] = f"{cost:.6g}"
    log_event(_LOGGER, "model_call_completed", context=context)


def _reported_cost(response: object, usage: object) -> float | None:
    """Return the provider's charge, then LiteLLM's, else None."""
    hidden = getattr(response, "_hidden_params", None)
    for candidate in (
        getattr(usage, "cost", None),
        hidden.get("response_cost") if isinstance(hidden, dict) else None,
    ):
        if (
            isinstance(candidate, (int, float))
            and not isinstance(candidate, bool)
            and candidate > 0
        ):
            return float(candidate)
    return None


def reasoning_client(effort: str) -> Client:
    """Return a provider client with an explicit reasoning effort."""
    return _LiteLLMClient(reasoning_effort=effort)


def _extract_json(raw: str) -> object:
    """Parse the first JSON object in a response, fenced or bare."""
    fenced = _FENCE.search(raw)
    candidate = fenced.group(1) if fenced else raw
    try:
        return json.loads(candidate)
    except ValueError:
        pass
    start = candidate.find("{")
    end = candidate.rfind("}")
    if start == -1 or end <= start:
        raise ValueError(candidate[:0])
    return json.loads(candidate[start : end + 1])


def _validated(payload: object, schema: Mapping[str, type]) -> dict[str, Any]:
    """Check exact key presence and type. No coercion: wrong shape is wrong.

    Raises ValueError rather than TypeError even for type mismatches: every
    failure here means the same thing to the caller, that the model answered
    unusably, and splitting them would only widen the except clause.
    """
    if not isinstance(payload, dict):
        message = "response is not an object"
        raise ValueError(message)  # noqa: TRY004 - see docstring
    for key, kind in schema.items():
        if key not in payload:
            message = f"response is missing {key!r}"
            raise ValueError(message)
        if not isinstance(payload[key], kind):
            message = f"response {key!r} is not {kind.__name__}"
            raise ValueError(message)  # noqa: TRY004 - see docstring
    return payload


def complete_json(  # noqa: PLR0913 - the tuning surface of one entry point.
    model: str,
    prompt: str,
    schema: Mapping[str, type],
    *,
    client: Client | None = None,
    attempts: int = DEFAULT_ATTEMPTS,
    backoff_base: float = DEFAULT_BACKOFF_BASE,
    cancel: CancellationToken | None = None,
) -> dict[str, Any]:
    """Return one validated JSON response, retrying transient failures.

    Output errors are retried too, not only transport ones: at temperature 0 a
    repeat is usually identical, but providers are not perfectly deterministic
    and the attempt is cheap next to abandoning a chapter.

    Nothing from the prompt or the response reaches an exception message or a
    log line. The prompt carries the book, and a failure is not a reason to
    republish it.
    """
    caller = client or _LiteLLMClient()
    invalid: Exception | None = None
    transport: Exception | None = None
    for attempt in range(1, attempts + 1):
        _check_cancel(cancel)
        try:
            raw = caller.complete(model, prompt)
        except _NEVER_RETRY:
            raise
        except Exception as error:  # noqa: BLE001 - provider errors are opaque
            transport = error
            log_event(
                _LOGGER,
                "model_call_failed",
                context={
                    "boundary": "characters",
                    "attempt": attempt,
                    "error": type(error).__name__,
                },
            )
        else:
            try:
                validated = _validated(_extract_json(raw), schema)
            except ValueError as error:
                invalid = error
                log_event(
                    _LOGGER,
                    "model_response_invalid",
                    context={"boundary": "characters", "attempt": attempt},
                )
            else:
                if cancel is not None:
                    cancel.raise_if_cancelled()
                return validated
        _check_cancel(cancel)
        if attempt < attempts and backoff_base > 0:
            _wait_for_retry(backoff_base ** (attempt - 1), cancel)
    # A response that arrived but never validated is the more useful diagnosis:
    # it says the model is reachable and answering the wrong shape.
    if invalid is not None:
        raise ModelError(ErrorCode.MODEL_RESPONSE_INVALID) from invalid
    raise ModelError(ErrorCode.MODEL_CALL_FAILED) from transport


def _wait_for_retry(delay: float, cancel: CancellationToken | None) -> None:
    """Wait between attempts without trapping cancellation in a long backoff."""
    if cancel is None:
        time.sleep(delay)
        return
    deadline = time.monotonic() + delay
    while True:
        cancel.raise_if_cancelled()
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return
        time.sleep(min(remaining, 0.1))


def _check_cancel(cancel: CancellationToken | None) -> None:
    """Check cooperative cancellation at each provider and retry boundary."""
    if cancel is not None:
        cancel.raise_if_cancelled()
