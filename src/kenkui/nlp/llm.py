"""Thin wrapper around the Ollama Python client.

All LLM calls in the pipeline go through ``LLMClient.generate()``, which
sends the prompt, requests structured JSON output matching the supplied
Pydantic schema, validates the response, and retries on transient failures.

Key options
-----------
num_predict : 32768  (env: KENKUI_NLP_OLLAMA_NUM_PREDICT)
    Maximum response tokens.  Large character rosters (~35 k chars / ~9 k
    tokens) and long chapters with many quotes (~27 k chars / ~7 k tokens)
    exceeded the old 8 192-token ceiling, producing truncated JSON.
    ``generate()`` detects EOF truncation and attempts partial recovery
    before falling back to the caller's own error handling.  Retries are
    skipped for truncation errors since the same prompt always produces
    the same cut-off point.
num_ctx : 65536  (env: KENKUI_NLP_OLLAMA_NUM_CTX)
    Total context window (prompt + output). Must exceed prompt_tokens +
    num_predict.  The roster prompt is ~5 500 tokens; the attribution
    prompt for a long chapter can be 8 000+ tokens.  65 536 gives both
    paths adequate headroom.
temperature : 0
    Deterministic sampling improves JSON schema compliance — the model
    spends less probability mass on syntactic variants and more on content.
"""

from __future__ import annotations

import logging
import os
from typing import TypeVar

from pydantic import BaseModel, ValidationError

_logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)

_num_ctx = int(os.environ.get("KENKUI_NLP_OLLAMA_NUM_CTX", "65536"))
_num_predict = int(os.environ.get("KENKUI_NLP_OLLAMA_NUM_PREDICT", "32768"))
_LLM_OPTIONS = {
    "num_predict": _num_predict,
    "num_ctx": _num_ctx,
    "temperature": 0,
}
_MAX_RETRIES = 2


def _is_eof_truncation(exc: ValidationError) -> bool:
    """Return True if the ValidationError is caused by a truncated (EOF) JSON response."""
    return any(
        e.get("type") == "json_invalid" and "EOF" in str(e.get("msg", ""))
        for e in exc.errors()
    )


def _try_recover_truncated_json(raw: str, schema: type[T]) -> T | None:
    """Attempt to parse a schema instance from truncated JSON.

    When the LLM hits its token limit mid-array the JSON is cut off somewhere
    inside an incomplete object.  This function finds the last complete JSON
    object (closing ``}``) before the cut-off, discards the partial entry, and
    closes the array/document so Pydantic can validate what arrived intact.

    Returns the recovered instance, or None if recovery fails.
    """
    if not raw or not raw.strip().startswith("{"):
        return None

    last_brace = raw.rfind("}")
    if last_brace <= 0:
        return None

    truncated = raw[:last_brace + 1]
    for closing in ("\n  ]\n}", "\n]\n}", "]}"):
        try:
            return schema.model_validate_json(truncated + closing)
        except Exception:
            continue

    return None


class LLMClient:
    """Stateless wrapper around ``ollama.chat`` with Pydantic schema enforcement."""

    def __init__(self, model: str) -> None:
        self.model = model

    def generate(self, prompt: str, schema: type[T]) -> T:
        """Send *prompt* to the model and return a validated *schema* instance.

        Ollama's ``format`` parameter is set to the JSON Schema derived from
        *schema*, which instructs the model to produce conforming output.
        The response is validated by Pydantic.  Up to ``_MAX_RETRIES`` retries
        are attempted on validation failures before the exception propagates.

        EOF truncation (token-limit cut-off) is handled specially: retrying
        the same prompt will always produce the same cut-off, so instead we
        attempt partial JSON recovery and skip further retries.
        """
        import ollama  # lazy — avoids import error when ollama not installed

        last_exc: Exception | None = None

        _logger.debug(
            "LLM send: model=%s schema=%s prompt=%d words / %d chars (num_ctx=%d)",
            self.model, schema.__name__, len(prompt.split()), len(prompt), _num_ctx,
        )

        for attempt in range(_MAX_RETRIES + 1):
            try:
                response = ollama.chat(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    format=schema.model_json_schema(),
                    options=_LLM_OPTIONS,
                    think=False,
                )
                raw = response.message.content
                if not raw:
                    raise ConnectionError(
                        f"Ollama model '{self.model}' returned empty content "
                        "(possible context overflow or grammar constraint failure)"
                    )
                prompt_tokens = getattr(response, "prompt_eval_count", None)
                if prompt_tokens is not None and prompt_tokens >= _num_ctx * 0.9:
                    _logger.warning(
                        "LLM prompt used %d / %d context tokens (%.0f%%) for model '%s' "
                        "— responses may be truncated",
                        prompt_tokens, _num_ctx, prompt_tokens / _num_ctx * 100, self.model,
                    )
                _logger.debug(
                    "LLM attempt %d: %d chars received (prompt_tokens=%s)",
                    attempt + 1, len(raw), prompt_tokens,
                )
                try:
                    return schema.model_validate_json(raw)
                except ValidationError as val_exc:
                    if _is_eof_truncation(val_exc):
                        eval_count = getattr(response, "eval_count", None)
                        recovered = _try_recover_truncated_json(raw, schema)
                        if recovered is not None:
                            _logger.warning(
                                "LLM response truncated at %d chars "
                                "(eval_count=%s, limit=%d); partially recovered "
                                "(skipping retries — same prompt produces same cut-off)",
                                len(raw), eval_count, _num_predict,
                            )
                            return recovered
                        # Truncation but unrecoverable — retrying won't help
                        _logger.warning(
                            "LLM response truncated at %d chars "
                            "(eval_count=%s, limit=%d) and could not be recovered",
                            len(raw), eval_count, _num_predict,
                        )
                        last_exc = val_exc
                        break
                    raise
            except Exception as exc:
                last_exc = exc
                if attempt < _MAX_RETRIES:
                    _logger.debug(
                        "LLM attempt %d failed (%s), retrying…", attempt + 1, exc
                    )

        raise last_exc  # type: ignore[misc]
