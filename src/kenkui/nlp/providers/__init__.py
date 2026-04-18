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

    ``"ollama"`` returns ``OllamaProvider``.
    Any other value (``"anthropic"``, ``"openai"``, ``"google"``, custom
    LiteLLM prefix) returns ``CloudProvider``.
    """
    if config.nlp_provider == "ollama":
        from kenkui.nlp.providers.ollama import OllamaProvider
        return OllamaProvider(config)
    from kenkui.nlp.providers.cloud import CloudProvider
    return CloudProvider(config)


__all__ = ["get_provider", "NLPProvider"]
