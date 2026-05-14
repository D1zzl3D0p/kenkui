"""NLPConfig — configuration for the kenkui NLP pipeline.

All fields are readable from KENKUI_NLP_* environment variables.
KENKUI_NLP_EXTRACTION_TOOL=booknlp  → NLPConfig().extraction_tool == ExtractionTool.BOOKNLP
"""
from __future__ import annotations

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from .models import AttributionExecutionMode, AttributionTool, ExtractionTool, NlpExecutionMode


class NLPConfig(BaseSettings):
    """Configuration for NLP extraction and attribution steps.

    Sourced from KENKUI_NLP_* env vars first, then defaults.
    """

    model_config = SettingsConfigDict(env_prefix="KENKUI_NLP_")

    # Step 1: character extraction + coreference resolution
    extraction_tool: ExtractionTool = ExtractionTool.OLLAMA
    extraction_mode: NlpExecutionMode = NlpExecutionMode.LOCAL
    extraction_model: str = "llama3.2"

    # Step 2: quote attribution
    attribution_tool: AttributionTool = AttributionTool.OLLAMA
    attribution_mode: AttributionExecutionMode = AttributionExecutionMode.LOCAL
    attribution_model: str = "llama3.2"

    # Tool endpoints
    ollama_url: str = "http://localhost:11434"
    litellm_api_base: str | None = None
    litellm_api_key: str | None = Field(None, alias="LITELLM_API_KEY")

    # Retry policy
    retry_max_attempts: int = 3
    retry_backoff_base: float = 2.0

    @classmethod
    def from_app_config(cls, config: "AppConfig") -> "NLPConfig":  # type: ignore[name-defined]
        """Build NLPConfig from an AppConfig for backwards compatibility."""
        from .models import AppConfig  # noqa: F401 — local import avoids circular

        _tool_map: dict[str, ExtractionTool] = {
            "ollama": ExtractionTool.OLLAMA,
            "booknlp": ExtractionTool.BOOKNLP,
        }
        _attr_map: dict[str, AttributionTool] = {
            "ollama": AttributionTool.OLLAMA,
            "booknlp": AttributionTool.BOOKNLP,
        }
        attr_provider = config.nlp_attribution_provider or config.nlp_provider
        return cls(
            extraction_tool=_tool_map.get(config.nlp_provider, ExtractionTool.OLLAMA),
            extraction_mode=config.nlp_execution_mode,
            extraction_model=config.nlp_roster_model or config.nlp_model,
            attribution_tool=_attr_map.get(attr_provider, AttributionTool.OLLAMA),
            attribution_mode=config.attribution_execution_mode,
            attribution_model=config.nlp_attribution_model or config.nlp_model,
            ollama_url=config.ollama_url,
        )
