"""NLPConfig — configuration for the kenkui NLP pipeline.

All fields are readable from KENKUI_NLP_* environment variables.
KENKUI_NLP_EXTRACTION_TOOL=booknlp  → NLPConfig().extraction_tool == ExtractionTool.BOOKNLP
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from .models import AttributionExecutionMode, AttributionTool, ExtractionTool, NlpExecutionMode

if TYPE_CHECKING:
    from .models import AppConfig

_logger = logging.getLogger(__name__)


class NLPConfig(BaseSettings):
    """Configuration for NLP extraction and attribution steps.

    Sourced from KENKUI_NLP_* env vars first, then defaults.
    """

    model_config = SettingsConfigDict(env_prefix="KENKUI_NLP_")

    # Step 1: character extraction + coreference resolution
    extraction_tool: ExtractionTool = ExtractionTool.OLLAMA
    extraction_mode: NlpExecutionMode = NlpExecutionMode.LOCAL
    extraction_model: str = "llama3.2"
    discovery_method: str = "auto"

    # Step 2: quote attribution
    attribution_tool: AttributionTool = AttributionTool.OLLAMA
    attribution_mode: AttributionExecutionMode = AttributionExecutionMode.LOCAL
    attribution_model: str = "llama3.2"

    # Tool endpoints
    ollama_url: str = "http://localhost:11434"
    ollama_num_ctx: int = 16384

    # Retry policy
    retry_max_attempts: int = 3
    retry_backoff_base: float = 2.0

    # Cloud attribution concurrency
    openrouter_attribution_concurrency: int = 4

    @field_validator("openrouter_attribution_concurrency", mode="before")
    @classmethod
    def _clamp_openrouter_attribution_concurrency(cls, v: Any) -> int:
        try:
            value = int(v)
        except (TypeError, ValueError):
            return 4
        return min(32, max(1, value))

    @classmethod
    def from_app_config(cls, config: AppConfig) -> NLPConfig:
        """Build NLPConfig from an AppConfig for backwards compatibility."""
        _tool_map: dict[str, ExtractionTool] = {
            "ollama": ExtractionTool.OLLAMA,
            "booknlp": ExtractionTool.BOOKNLP,
            "litellm": ExtractionTool.LITELLM,
            "openrouter": ExtractionTool.OPENROUTER,
            "anthropic": ExtractionTool.ANTHROPIC,
            "openai": ExtractionTool.OPENAI,
            "google": ExtractionTool.GOOGLE,
        }
        _attr_map: dict[str, AttributionTool] = {
            "ollama": AttributionTool.OLLAMA,
            "booknlp": AttributionTool.BOOKNLP,
            "litellm": AttributionTool.LITELLM,
            "openrouter": AttributionTool.OPENROUTER,
            "anthropic": AttributionTool.ANTHROPIC,
            "openai": AttributionTool.OPENAI,
            "google": AttributionTool.GOOGLE,
        }
        attr_provider = config.nlp_attribution_provider or config.nlp_provider
        extraction_tool = _tool_map.get(config.nlp_provider)
        if extraction_tool is None:
            _logger.warning("Unknown nlp_provider %r; defaulting to OLLAMA", config.nlp_provider)
            extraction_tool = ExtractionTool.OLLAMA
        attribution_tool = _attr_map.get(attr_provider)
        if attribution_tool is None:
            _logger.warning("Unknown nlp_attribution_provider %r; defaulting to OLLAMA", attr_provider)
            attribution_tool = AttributionTool.OLLAMA

        # Coerce enum fields — fall back to defaults when the AppConfig carries
        # a non-string value (e.g. a MagicMock in tests or an already-enum value).
        raw_extraction_mode = config.nlp_execution_mode
        try:
            extraction_mode = NlpExecutionMode(raw_extraction_mode)
        except (ValueError, TypeError):
            extraction_mode = NlpExecutionMode.LOCAL

        raw_attribution_mode = config.attribution_execution_mode
        try:
            attribution_mode = AttributionExecutionMode(raw_attribution_mode)
        except (ValueError, TypeError):
            attribution_mode = AttributionExecutionMode.LOCAL

        raw_ollama_url = config.ollama_url
        ollama_url = raw_ollama_url if isinstance(raw_ollama_url, str) else "http://localhost:11434"

        raw_extraction_model = config.nlp_roster_model or config.nlp_model
        extraction_model = raw_extraction_model if isinstance(raw_extraction_model, str) else "llama3.2"

        raw_attribution_model = config.nlp_attribution_model or config.nlp_model
        attribution_model = raw_attribution_model if isinstance(raw_attribution_model, str) else "llama3.2"

        discovery_method = getattr(config, "nlp_discovery_method", "auto") or "auto"
        openrouter_concurrency = getattr(config, "nlp_openrouter_attribution_concurrency", 4)

        return cls(
            extraction_tool=extraction_tool,
            extraction_mode=extraction_mode,
            extraction_model=extraction_model,
            discovery_method=discovery_method,
            attribution_tool=attribution_tool,
            attribution_mode=attribution_mode,
            attribution_model=attribution_model,
            ollama_url=ollama_url,
            openrouter_attribution_concurrency=openrouter_concurrency,
        )
