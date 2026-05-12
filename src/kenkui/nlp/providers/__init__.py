"""NLP provider factory.

Usage::

    from kenkui.nlp.providers import get_provider

    provider = get_provider(app_config)
    roster   = provider.build_roster(chapters)
    result   = provider.attribute_chapter(chapter, roster)
"""

from __future__ import annotations

from kenkui.models import AppConfig
from kenkui.nlp.providers.base import NLPProvider


def get_provider(config: AppConfig) -> NLPProvider:
    """Return the appropriate NLP provider for *config.nlp_provider*.

    Supported local providers: ``"ollama"``, ``"spacy"``, ``"booknlp"``.
    Cloud providers (anthropic, openai, google) require kenkui-server.
    """
    from kenkui.nlp.providers.ollama import OllamaProvider
    return OllamaProvider(config)


def get_discovery_provider(config: AppConfig) -> NLPProvider:
    """Return the provider for character discovery (Stage 2)."""
    return get_provider(config)


def get_attribution_provider(config: AppConfig) -> NLPProvider:
    """Return the provider for quote attribution (Stage 4).

    Uses ``config.nlp_attribution_provider`` / ``config.nlp_attribution_model``
    when non-empty, falling back to ``config.nlp_provider`` / ``config.nlp_model``.
    """
    attr_provider = getattr(config, "nlp_attribution_provider", "") or config.nlp_provider
    attr_model = getattr(config, "nlp_attribution_model", "") or config.nlp_model
    if attr_provider != config.nlp_provider or attr_model != config.nlp_model:
        config = config.model_copy(update={"nlp_provider": attr_provider, "nlp_model": attr_model})
    return get_provider(config)


__all__ = ["get_provider", "get_discovery_provider", "get_attribution_provider", "NLPProvider"]
