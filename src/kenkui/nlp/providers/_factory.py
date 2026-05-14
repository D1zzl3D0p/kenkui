"""Provider factory for the NLP pipeline.

get_extraction_provider(config) -> ExtractionProvider
get_attribution_provider(config) -> AttributionProvider
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from kenkui.models import AttributionExecutionMode, AttributionTool, ExtractionTool, NlpExecutionMode
from kenkui.nlp.providers._base import AttributionProvider, ExtractionProvider
from kenkui.nlp.providers.local import LocalAttributionProvider, LocalExtractionProvider
from kenkui.nlp.providers.modal import ModalAttributionProvider, ModalExtractionProvider

if TYPE_CHECKING:
    from kenkui.nlp_config import NLPConfig

_logger = logging.getLogger(__name__)


def _make_extraction_adapter(config: "NLPConfig") -> ExtractionProvider:
    match config.extraction_tool:
        case ExtractionTool.BOOKNLP:
            from kenkui.nlp.providers.booknlp import BookNLPExtractionAdapter
            return BookNLPExtractionAdapter(config)
        case ExtractionTool.OLLAMA:
            from kenkui.nlp.providers.ollama import OllamaExtractionAdapter
            return OllamaExtractionAdapter(config)
        case ExtractionTool.LITELLM:
            from kenkui.nlp.providers.litellm import LiteLLMExtractionAdapter
            return LiteLLMExtractionAdapter(config)
        case _:
            raise ValueError(f"Unknown extraction tool: {config.extraction_tool!r}")


def _make_attribution_adapter(config: "NLPConfig") -> AttributionProvider:
    match config.attribution_tool:
        case AttributionTool.BOOKNLP:
            from kenkui.nlp.providers.booknlp import BookNLPAttributionAdapter
            return BookNLPAttributionAdapter(config)
        case AttributionTool.OLLAMA:
            from kenkui.nlp.providers.ollama import OllamaAttributionAdapter
            return OllamaAttributionAdapter(config)
        case AttributionTool.LITELLM:
            from kenkui.nlp.providers.litellm import LiteLLMAttributionAdapter
            return LiteLLMAttributionAdapter(config)
        case _:
            raise ValueError(f"Unknown attribution tool: {config.attribution_tool!r}")


def get_extraction_provider(config: "NLPConfig") -> ExtractionProvider:
    """Return an ExtractionProvider for the given NLPConfig."""
    adapter = _make_extraction_adapter(config)
    match config.extraction_mode:
        case NlpExecutionMode.LOCAL:
            return LocalExtractionProvider(adapter)
        case NlpExecutionMode.MODAL:
            return ModalExtractionProvider(adapter)
        case _:
            raise ValueError(f"Unknown extraction mode: {config.extraction_mode!r}")


def get_attribution_provider(config: "NLPConfig") -> AttributionProvider:
    """Return an AttributionProvider for the given NLPConfig."""
    adapter = _make_attribution_adapter(config)
    match config.attribution_mode:
        case AttributionExecutionMode.LOCAL:
            return LocalAttributionProvider(adapter)
        case AttributionExecutionMode.MODAL:
            return ModalAttributionProvider(adapter)
        case _:
            raise ValueError(f"Unknown attribution mode: {config.attribution_mode!r}")


__all__ = [
    "get_extraction_provider",
    "get_attribution_provider",
]
