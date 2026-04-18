import pytest
from unittest.mock import MagicMock, patch
from kenkui.models import AppConfig
from kenkui.nlp.providers import get_provider
from kenkui.nlp.providers.ollama import OllamaProvider
from kenkui.nlp.providers.cloud import CloudProvider


def test_get_provider_ollama():
    config = AppConfig(nlp_provider="ollama")
    provider = get_provider(config)
    assert isinstance(provider, OllamaProvider)


def test_get_provider_anthropic():
    config = AppConfig(nlp_provider="anthropic")
    provider = get_provider(config)
    assert isinstance(provider, CloudProvider)


def test_get_provider_openai():
    config = AppConfig(nlp_provider="openai")
    provider = get_provider(config)
    assert isinstance(provider, CloudProvider)


def test_get_provider_google():
    config = AppConfig(nlp_provider="google")
    provider = get_provider(config)
    assert isinstance(provider, CloudProvider)


def test_get_provider_default_is_ollama():
    config = AppConfig()
    provider = get_provider(config)
    assert isinstance(provider, OllamaProvider)
