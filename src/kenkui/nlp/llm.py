"""Thin wrapper around the Ollama Python client.

All LLM calls in the pipeline go through ``LLMClient.generate()``, which
sends the prompt, requests structured JSON output matching the supplied
Pydantic schema, validates the response, and retries on transient failures.

Key options
-----------
num_predict : 8192
    Prevents Ollama from truncating long JSON responses mid-stream, which
    was the root cause of "EOF while parsing" Pydantic validation errors.
temperature : 0
    Deterministic sampling improves JSON schema compliance — the model
    spends less probability mass on syntactic variants and more on content.
"""

from __future__ import annotations

import logging
from typing import TypeVar

from pydantic import BaseModel

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)

_LLM_OPTIONS = {"num_predict": 8192, "temperature": 0}
_MAX_RETRIES = 2


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
        """
        import ollama  # lazy — avoids import error when ollama not installed

        last_exc: Exception | None = None

        for attempt in range(_MAX_RETRIES + 1):
            try:
                response = ollama.chat(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    format=schema.model_json_schema(),
                    options=_LLM_OPTIONS,
                )
                raw = response.message.content
                logger.debug(
                    "LLM attempt %d: %d chars received", attempt + 1, len(raw)
                )
                return schema.model_validate_json(raw)
            except Exception as exc:
                last_exc = exc
                if attempt < _MAX_RETRIES:
                    logger.debug(
                        "LLM attempt %d failed (%s), retrying…", attempt + 1, exc
                    )

        raise last_exc  # type: ignore[misc]
