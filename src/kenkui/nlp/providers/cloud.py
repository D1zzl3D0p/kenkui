"""CloudProvider — two-pass large-context NLP via LiteLLM + instructor.

Pass 1 (build_roster):  Send the full book text (chunked at chapter
    boundaries if needed) to the LLM and extract a rich CharacterRoster.
Pass 2 (attribute_chapter): Send each chapter + the full roster and get
    back speaker attributions with slug-keyed speakers.
"""

from __future__ import annotations

import os
from collections.abc import Callable

import instructor
import litellm

from kenkui.models import AppConfig, Chapter
from kenkui.nlp.models import (
    AttributionResult,
    CharacterRecord,
    CharacterRoster,
    slugify,
)

# ---------------------------------------------------------------------------
# Token / budget utilities
# ---------------------------------------------------------------------------

_CHARS_PER_TOKEN = 4          # rough heuristic (conservative)
_SYSTEM_PROMPT_TOKENS = 2_000  # headroom for system/instruction prompt
_MIN_OUTPUT_RESERVE = 0.20     # always reserve at least 20% of context for output
_TOKENS_PER_CHARACTER = 600    # estimated tokens per CharacterRecord in JSON output


def estimate_tokens(text: str) -> int:
    """Estimate token count for *text* using a 4-chars-per-token heuristic."""
    return max(1, len(text) // _CHARS_PER_TOKEN)


def compute_output_budget(context_limit: int, estimated_chars: int) -> int:
    """Compute how many tokens to reserve for LLM output.

    Takes the larger of: 20% of context_limit, or tokens needed for
    *estimated_chars* CharacterRecords.
    """
    floor = int(context_limit * _MIN_OUTPUT_RESERVE)
    character_estimate = estimated_chars * _TOKENS_PER_CHARACTER
    return max(floor, character_estimate)


def needs_chunking(
    input_tokens: int,
    output_budget: int,
    system_tokens: int,
    context_limit: int,
) -> bool:
    """Return True if the combined token count exceeds the model's context limit."""
    return (input_tokens + output_budget + system_tokens) > context_limit


# ---------------------------------------------------------------------------
# Model context limit registry
# ---------------------------------------------------------------------------

_CONTEXT_LIMITS: dict[str, int] = {
    "claude-opus-4-6": 200_000,
    "claude-sonnet-4-6": 200_000,
    "claude-haiku-4-5-20251001": 200_000,
    "claude-3-5-sonnet-20241022": 200_000,
    "gpt-4o": 128_000,
    "gpt-4o-mini": 128_000,
    "gemini/gemini-2.0-flash": 1_000_000,
    "gemini/gemini-1.5-pro": 2_000_000,
}
_DEFAULT_CONTEXT_LIMIT = 128_000


def _context_limit_for(model: str) -> int:
    return _CONTEXT_LIMITS.get(model, _DEFAULT_CONTEXT_LIMIT)


# ---------------------------------------------------------------------------
# CloudProvider
# ---------------------------------------------------------------------------


class CloudProvider:
    """NLP provider using LiteLLM + instructor for cloud model access."""

    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self._client = instructor.from_litellm(litellm.completion)
        self._inject_credentials()

    def _inject_credentials(self) -> None:
        """Load credentials from credentials.toml and set env vars for LiteLLM."""
        from kenkui.config import inject_provider_env_vars, load_provider_credentials
        inject_provider_env_vars(load_provider_credentials())

    def _resolved_model(self) -> str:
        """Return the LiteLLM model string to use.

        Falls back to provider default from credentials.toml when nlp_model is empty.
        """
        if self.config.nlp_model:
            return self.config.nlp_model
        from kenkui.config import load_provider_credentials
        creds = load_provider_credentials()
        provider_creds = creds.get(self.config.nlp_provider)
        if provider_creds and provider_creds.default_model:
            return provider_creds.default_model
        # Last resort defaults
        defaults = {
            "anthropic": "claude-sonnet-4-6",
            "openai": "gpt-4o",
            "google": "gemini/gemini-2.0-flash",
        }
        return defaults.get(self.config.nlp_provider, "gpt-4o")

    def build_roster(
        self,
        chapters: list[Chapter],
        series_roster: CharacterRoster | None = None,
        progress_callback: Callable[[str], None] | None = None,
    ) -> CharacterRoster:
        """Extract character roster — whole-book pass, chunked if needed."""
        raise NotImplementedError("Implemented in Task 9")

    def attribute_chapter(
        self,
        chapter: Chapter,
        roster: CharacterRoster,
        progress_callback: Callable[[str], None] | None = None,
    ) -> AttributionResult:
        """Attribute quotes in *chapter* to speakers — single chapter pass."""
        raise NotImplementedError("Implemented in Task 11")
