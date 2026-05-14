"""Tests for the NLP provider factory (Phase 8).

Covers get_extraction_provider() and get_nlp_attribution_provider() from
kenkui.nlp.providers._factory, exercised via the public __init__ exports.
"""
from __future__ import annotations

import pytest

from kenkui.models import AttributionExecutionMode, AttributionTool, ExtractionTool, NlpExecutionMode
from kenkui.nlp_config import NLPConfig
from kenkui.nlp.providers import get_extraction_provider, get_nlp_attribution_provider
from kenkui.nlp.providers._factory import get_extraction_provider as factory_extraction
from kenkui.nlp.providers._factory import get_attribution_provider as factory_attribution
from kenkui.nlp.providers.local import LocalExtractionProvider, LocalAttributionProvider
from kenkui.nlp.providers.modal import ModalExtractionProvider, ModalAttributionProvider
from kenkui.nlp.providers.ollama import OllamaExtractionAdapter, OllamaAttributionAdapter
from kenkui.nlp.providers.booknlp import BookNLPExtractionAdapter, BookNLPAttributionAdapter
from kenkui.nlp.providers.litellm import LiteLLMExtractionAdapter, LiteLLMAttributionAdapter


# ---------------------------------------------------------------------------
# Extraction provider tests
# ---------------------------------------------------------------------------


def test_get_extraction_provider_ollama_local():
    config = NLPConfig(extraction_tool=ExtractionTool.OLLAMA, extraction_mode=NlpExecutionMode.LOCAL)
    provider = get_extraction_provider(config)
    assert isinstance(provider, LocalExtractionProvider)
    assert isinstance(provider._adapter, OllamaExtractionAdapter)


def test_get_extraction_provider_booknlp_local():
    config = NLPConfig(extraction_tool=ExtractionTool.BOOKNLP, extraction_mode=NlpExecutionMode.LOCAL)
    provider = get_extraction_provider(config)
    assert isinstance(provider, LocalExtractionProvider)
    assert isinstance(provider._adapter, BookNLPExtractionAdapter)


def test_get_extraction_provider_litellm_local():
    config = NLPConfig(extraction_tool=ExtractionTool.LITELLM, extraction_mode=NlpExecutionMode.LOCAL)
    provider = get_extraction_provider(config)
    assert isinstance(provider, LocalExtractionProvider)
    assert isinstance(provider._adapter, LiteLLMExtractionAdapter)


def test_get_extraction_provider_modal_mode():
    config = NLPConfig(extraction_tool=ExtractionTool.OLLAMA, extraction_mode=NlpExecutionMode.MODAL)
    provider = get_extraction_provider(config)
    assert isinstance(provider, ModalExtractionProvider)
    assert isinstance(provider._adapter, OllamaExtractionAdapter)


# ---------------------------------------------------------------------------
# Attribution provider tests
# ---------------------------------------------------------------------------


def test_get_attribution_provider_ollama_local():
    config = NLPConfig(
        attribution_tool=AttributionTool.OLLAMA,
        attribution_mode=AttributionExecutionMode.LOCAL,
    )
    provider = get_nlp_attribution_provider(config)
    assert isinstance(provider, LocalAttributionProvider)
    assert isinstance(provider._adapter, OllamaAttributionAdapter)


def test_get_attribution_provider_booknlp_local():
    config = NLPConfig(
        attribution_tool=AttributionTool.BOOKNLP,
        attribution_mode=AttributionExecutionMode.LOCAL,
    )
    provider = get_nlp_attribution_provider(config)
    assert isinstance(provider, LocalAttributionProvider)
    assert isinstance(provider._adapter, BookNLPAttributionAdapter)


def test_get_attribution_provider_litellm_local():
    config = NLPConfig(
        attribution_tool=AttributionTool.LITELLM,
        attribution_mode=AttributionExecutionMode.LOCAL,
    )
    provider = get_nlp_attribution_provider(config)
    assert isinstance(provider, LocalAttributionProvider)
    assert isinstance(provider._adapter, LiteLLMAttributionAdapter)


def test_get_attribution_provider_modal_mode():
    config = NLPConfig(
        attribution_tool=AttributionTool.OLLAMA,
        attribution_mode=AttributionExecutionMode.MODAL,
    )
    provider = get_nlp_attribution_provider(config)
    assert isinstance(provider, ModalAttributionProvider)
    assert isinstance(provider._adapter, OllamaAttributionAdapter)


# ---------------------------------------------------------------------------
# Error cases
# ---------------------------------------------------------------------------


def test_unknown_extraction_tool_raises():
    config = NLPConfig(extraction_tool=ExtractionTool.OLLAMA, extraction_mode=NlpExecutionMode.LOCAL)
    object.__setattr__(config, "extraction_tool", "not_a_real_tool")
    with pytest.raises(ValueError, match="Unknown extraction tool"):
        factory_extraction(config)


def test_unknown_attribution_tool_raises():
    config = NLPConfig(
        attribution_tool=AttributionTool.OLLAMA,
        attribution_mode=AttributionExecutionMode.LOCAL,
    )
    object.__setattr__(config, "attribution_tool", "not_a_real_tool")
    with pytest.raises(ValueError, match="Unknown attribution tool"):
        factory_attribution(config)
