"""NLP provider factory.

Legacy API (still works):
    from kenkui.nlp.providers import get_provider
    provider = get_provider(app_config)

New API (after Phase 4+):
    from kenkui.nlp.providers import get_extraction_provider, get_attribution_provider
    extraction = get_extraction_provider(nlp_config)
    attribution = get_attribution_provider(nlp_config)
"""
from __future__ import annotations

from kenkui.nlp.providers._base import AttributionProvider, ExtractionProvider

# Re-export NLPProvider from legacy base module for backwards compatibility.
from kenkui.nlp.providers.base import NLPProvider


def get_provider(config):  # type: ignore[no-untyped-def]
    """Legacy: return an OllamaProvider for config. Kept for backwards compat."""
    from kenkui.nlp.providers.ollama import OllamaProvider
    return OllamaProvider(config)


def get_discovery_provider(config):  # type: ignore[no-untyped-def]
    return get_provider(config)


def get_attribution_provider(config):  # type: ignore[no-untyped-def]
    """Legacy: return provider for attribution step."""
    attr_provider = getattr(config, "nlp_attribution_provider", "") or config.nlp_provider
    attr_model = getattr(config, "nlp_attribution_model", "") or config.nlp_model
    if attr_provider != config.nlp_provider or attr_model != config.nlp_model:
        config = config.model_copy(update={"nlp_provider": attr_provider, "nlp_model": attr_model})
    return get_provider(config)


__all__ = [
    "get_provider",
    "get_discovery_provider",
    "get_attribution_provider",
    "ExtractionProvider",
    "AttributionProvider",
    "NLPProvider",
]
