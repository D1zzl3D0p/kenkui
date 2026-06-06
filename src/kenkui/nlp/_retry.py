"""Exponential-backoff retry decorator for NLP pipeline steps."""
from __future__ import annotations

import functools
import json
import logging
import time
from collections.abc import Callable
from typing import TypeVar

from kenkui.errors import KenkuiDependencyError

_logger = logging.getLogger(__name__)

_RETRYABLE = (OSError, TimeoutError, ConnectionError, UnicodeDecodeError, json.JSONDecodeError)
_NOT_RETRYABLE = (KeyboardInterrupt, SystemExit, ValueError, TypeError, AttributeError)

T = TypeVar("T")


def _simplify_provider_error(exc: Exception) -> str | None:
    """Return a short user-facing message for known provider failures."""
    message = str(exc)
    if "llama-server binary not found" in message:
        return (
            "Ollama cannot start because its llama-server binary is missing. "
            "Reinstall or upgrade Ollama, then restart the Ollama service."
        )
    return None


def with_retry(
    fn: Callable[..., T],
    max_attempts: int = 3,
    backoff_base: float = 2.0,
) -> Callable[..., T]:
    """Wrap *fn* with exponential-backoff retry on transient errors.

    Retries on network, IO, timeout, and connection errors.
    Never retries on programmer errors (ValueError, TypeError) or signals.
    Backoff: attempt 1 → wait 1s, attempt 2 → wait 2s, attempt 3 → fail.
    (wait = backoff_base ** (attempt_index), zero-indexed)
    """
    @functools.wraps(fn)
    def _wrapper(*args: object, **kwargs: object) -> T:
        last_exc: BaseException | None = None
        for attempt in range(1, max_attempts + 1):
            try:
                return fn(*args, **kwargs)
            except _NOT_RETRYABLE:
                raise
            except Exception as exc:  # noqa: BLE001
                last_exc = exc
                simplified = _simplify_provider_error(exc)
                if simplified is not None:
                    _logger.warning("%s", simplified)
                    _logger.debug(
                        "Full non-retryable NLP provider error: %s: %s",
                        type(exc).__name__,
                        exc,
                        exc_info=True,
                    )
                    raise KenkuiDependencyError(simplified) from exc
                if attempt == max_attempts:
                    break
                wait = backoff_base ** (attempt - 1)
                _logger.warning(
                    "Attempt %d/%d failed (%s); retrying in %.1fs",
                    attempt,
                    max_attempts,
                    _simplify_provider_error(exc) or f"{type(exc).__name__}: {exc}",
                    wait,
                )
                _logger.debug(
                    "Full retryable NLP provider error on attempt %d/%d: %s: %s",
                    attempt,
                    max_attempts,
                    type(exc).__name__,
                    exc,
                    exc_info=True,
                )
                time.sleep(wait)
        raise last_exc  # type: ignore[misc]

    return _wrapper
