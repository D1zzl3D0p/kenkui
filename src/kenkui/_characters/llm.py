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

from kenkui.errors import ErrorCode, ModelError
from kenkui.observability import get_logger, log_event

if TYPE_CHECKING:
    from collections.abc import Mapping

_LOGGER = get_logger(__name__)

DEFAULT_ATTEMPTS = 3
DEFAULT_BACKOFF_BASE = 2.0

# Retrying a bug burns provider calls to fail the same way, and retrying an
# interrupt swallows a Ctrl-C during what may be a very long attribution pass.
_NEVER_RETRY = (KeyboardInterrupt, SystemExit, TypeError, AttributeError, NameError)

# Models wrap JSON in prose and fences despite being told not to.
_FENCE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL)


class Client(Protocol):
    """Minimal completion boundary, so tests never need a provider."""

    def complete(self, model: str, prompt: str) -> str:
        """Return the model's raw text response."""
        ...


class _LiteLLMClient:
    """Direct LiteLLM call at fixed parameters."""

    def complete(self, model: str, prompt: str) -> str:
        """Return one completion, imported lazily to keep import cheap."""
        import litellm  # noqa: PLC0415 - heavy, and unused unless casting runs

        response = litellm.completion(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
        )
        return str(response.choices[0].message.content)


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
                return _validated(_extract_json(raw), schema)
            except ValueError as error:
                invalid = error
                log_event(
                    _LOGGER,
                    "model_response_invalid",
                    context={"boundary": "characters", "attempt": attempt},
                )
        if attempt < attempts and backoff_base > 0:
            time.sleep(backoff_base ** (attempt - 1))
    # A response that arrived but never validated is the more useful diagnosis:
    # it says the model is reachable and answering the wrong shape.
    if invalid is not None:
        raise ModelError(ErrorCode.MODEL_RESPONSE_INVALID) from invalid
    raise ModelError(ErrorCode.MODEL_CALL_FAILED) from transport
