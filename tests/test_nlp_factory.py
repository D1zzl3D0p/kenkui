"""Tests for the NLP provider factory.

Covers get_extraction_provider() and get_nlp_attribution_provider() from
kenkui.nlp.providers._factory, exercised via the public __init__ exports.
"""
from __future__ import annotations

import pytest

from kenkui.models import AttributionExecutionMode, AttributionTool, ExtractionTool, NlpExecutionMode
from kenkui.nlp_config import NLPConfig
from kenkui.nlp.providers import get_extraction_provider, get_nlp_attribution_provider
from kenkui.nlp.providers._factory import (
    get_extraction_provider as factory_extraction,
    get_attribution_provider as factory_attribution,
    register_nlp_extension,
)
from kenkui.nlp.providers.local import LocalExtractionProvider, LocalAttributionProvider
from kenkui.nlp.providers.ollama import OllamaExtractionAdapter, OllamaAttributionAdapter
from kenkui.nlp.providers.booknlp import BookNLPExtractionAdapter, BookNLPAttributionAdapter
import kenkui.nlp.providers._factory as _factory_module


@pytest.fixture(autouse=True)
def reset_extension_hook():
    """Reset the global extension hook before and after each test."""
    _factory_module._ExtractionExt = None
    _factory_module._AttributionExt = None
    yield
    _factory_module._ExtractionExt = None
    _factory_module._AttributionExt = None


# ---------------------------------------------------------------------------
# LOCAL mode — kenkui handles these directly
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


# ---------------------------------------------------------------------------
# Extension hook — non-LOCAL modes
# ---------------------------------------------------------------------------

def test_non_local_extraction_mode_without_hook_raises():
    config = NLPConfig(extraction_tool=ExtractionTool.OLLAMA, extraction_mode=NlpExecutionMode.MODAL)
    with pytest.raises(NotImplementedError, match="kenkui-server"):
        factory_extraction(config)


def test_non_local_attribution_mode_without_hook_raises():
    config = NLPConfig(
        attribution_tool=AttributionTool.OLLAMA,
        attribution_mode=AttributionExecutionMode.MODAL,
    )
    with pytest.raises(NotImplementedError, match="kenkui-server"):
        factory_attribution(config)


def test_extension_hook_called_for_non_local_extraction():
    sentinel = object()
    register_nlp_extension(lambda cfg: sentinel, lambda cfg: None)
    config = NLPConfig(extraction_tool=ExtractionTool.OLLAMA, extraction_mode=NlpExecutionMode.MODAL)
    assert factory_extraction(config) is sentinel


def test_extension_hook_called_for_non_local_attribution():
    sentinel = object()
    register_nlp_extension(lambda cfg: None, lambda cfg: sentinel)
    config = NLPConfig(
        attribution_tool=AttributionTool.OLLAMA,
        attribution_mode=AttributionExecutionMode.MODAL,
    )
    assert factory_attribution(config) is sentinel


def test_extension_hook_returning_none_falls_through_to_local():
    """Hook returning None signals kenkui to use its own LOCAL logic."""
    register_nlp_extension(lambda cfg: None, lambda cfg: None)
    config = NLPConfig(extraction_tool=ExtractionTool.OLLAMA, extraction_mode=NlpExecutionMode.MODAL)
    provider = factory_extraction(config)
    assert isinstance(provider, LocalExtractionProvider)
    assert isinstance(provider._adapter, OllamaExtractionAdapter)


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
