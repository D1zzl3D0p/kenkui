"""Provider factory for the NLP pipeline.

get_extraction_provider(config) -> ExtractionProvider
get_attribution_provider(config) -> AttributionProvider
register_nlp_extension(extraction, attribution) -> None
"""
from __future__ import annotations

import logging
from collections.abc import Callable
from typing import TYPE_CHECKING

from kenkui.models import (
    AttributionExecutionMode,
    AttributionTool,
    ExtractionTool,
    NlpExecutionMode,
)
from kenkui.nlp.providers._base import AttributionProvider, ExtractionProvider
from kenkui.nlp.providers.local import LocalAttributionProvider, LocalExtractionProvider

if TYPE_CHECKING:
    from kenkui.nlp_config import NLPConfig

_logger = logging.getLogger(__name__)

_ExtractionExt: Callable[[NLPConfig], ExtractionProvider | None] | None = None
_AttributionExt: Callable[[NLPConfig], AttributionProvider | None] | None = None

_LITELLM_EXTRACTION_TOOLS = {
    ExtractionTool.LITELLM,
    ExtractionTool.OPENROUTER,
    ExtractionTool.ANTHROPIC,
    ExtractionTool.OPENAI,
    ExtractionTool.GOOGLE,
}
_LITELLM_ATTRIBUTION_TOOLS = {
    AttributionTool.LITELLM,
    AttributionTool.OPENROUTER,
    AttributionTool.ANTHROPIC,
    AttributionTool.OPENAI,
    AttributionTool.GOOGLE,
}


def register_nlp_extension(
    extraction: Callable[[NLPConfig], ExtractionProvider | None],
    attribution: Callable[[NLPConfig], AttributionProvider | None],
) -> None:
    """Register server-side factories for non-LOCAL execution modes.

    Called once at process startup by the HTTP runtime. Each callable receives
    an NLPConfig and returns a provider or None (None = delegate to kenkui LOCAL logic).
    """
    global _ExtractionExt, _AttributionExt
    _ExtractionExt, _AttributionExt = extraction, attribution


def _make_extraction_adapter(config: NLPConfig) -> ExtractionProvider:
    if config.extraction_tool in _LITELLM_EXTRACTION_TOOLS:
        from kenkui.nlp.providers.litellm import LiteLLMExtractionAdapter
        return LiteLLMExtractionAdapter(config)
    match config.extraction_tool:
        case ExtractionTool.BOOKNLP:
            from kenkui.nlp.providers.booknlp import BookNLPExtractionAdapter
            return BookNLPExtractionAdapter(config)
        case ExtractionTool.OLLAMA:
            from kenkui.nlp.providers.ollama import OllamaExtractionAdapter
            return OllamaExtractionAdapter(config)
        case _:
            raise ValueError(f"Unknown extraction tool: {config.extraction_tool!r}")


def _make_attribution_adapter(config: NLPConfig) -> AttributionProvider:
    if config.attribution_tool in _LITELLM_ATTRIBUTION_TOOLS:
        from kenkui.nlp.providers.litellm import LiteLLMAttributionAdapter
        return LiteLLMAttributionAdapter(config)
    match config.attribution_tool:
        case AttributionTool.BOOKNLP:
            from kenkui.nlp.providers.booknlp import BookNLPAttributionAdapter
            return BookNLPAttributionAdapter(config)
        case AttributionTool.OLLAMA:
            from kenkui.nlp.providers.ollama import OllamaAttributionAdapter
            return OllamaAttributionAdapter(config)
        case _:
            raise ValueError(f"Unknown attribution tool: {config.attribution_tool!r}")


def get_extraction_provider(config: NLPConfig) -> ExtractionProvider:
    """Return an ExtractionProvider for the given NLPConfig."""
    if config.extraction_mode != NlpExecutionMode.LOCAL:
        if _ExtractionExt is not None:
            result = _ExtractionExt(config)
            if result is not None:
                return result
        else:
            raise NotImplementedError(
                f"NLP extraction mode {config.extraction_mode!r} requires a registered "
                "kenkui NLP extension. Call register_nlp_extension() at startup."
            )
    adapter = _make_extraction_adapter(config)
    return LocalExtractionProvider(adapter)


def get_attribution_provider(config: NLPConfig) -> AttributionProvider:
    """Return an AttributionProvider for the given NLPConfig."""
    if config.attribution_mode != AttributionExecutionMode.LOCAL:
        if _AttributionExt is not None:
            result = _AttributionExt(config)
            if result is not None:
                return result
        else:
            raise NotImplementedError(
                f"NLP attribution mode {config.attribution_mode!r} requires a registered "
                "kenkui NLP extension. Call register_nlp_extension() at startup."
            )
    adapter = _make_attribution_adapter(config)
    return LocalAttributionProvider(adapter)


__all__ = [
    "get_extraction_provider",
    "get_attribution_provider",
    "register_nlp_extension",
]
