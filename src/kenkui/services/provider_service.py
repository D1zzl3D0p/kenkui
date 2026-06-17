"""Provider discovery and credential smoke-test helpers."""

from __future__ import annotations

import logging
import os
from collections.abc import Iterable
from typing import Any

from kenkui.config import (
    _KENKUI_PROVIDER_ENV_VARS,
    _PROVIDER_ENV_VARS,
    ProviderCredentials,
    load_provider_credentials,
)
from kenkui.models import OkResponse
from kenkui.models.api import ProviderModelListResponse
from kenkui.nlp.setup import list_recommended_models
from kenkui.services.openrouter_service import search_openrouter_models

logger = logging.getLogger(__name__)

_CREDENTIAL_PROVIDER_NAMES = ("anthropic", "openai", "google", "openrouter")
_MODEL_PROVIDER_NAMES = ("ollama",) + _CREDENTIAL_PROVIDER_NAMES
_OPENROUTER_ROUTER_VENDORS = {"openrouter", "switchpoint"}
_PROVIDER_MODEL_PREFIXES: dict[str, tuple[str, ...]] = {
    "anthropic": ("anthropic/",),
    "google": ("gemini/", "models/", "google/"),
    "ollama": ("ollama/",),
    "openai": ("openai/",),
    "openrouter": ("openrouter/",),
}


def _normalize_provider(provider: str) -> str:
    return provider.lower().strip()


def _unique_preserve_order(values: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        candidate = value.strip()
        if not candidate or candidate in seen:
            continue
        seen.add(candidate)
        result.append(candidate)
    return result


def _normalize_model_id(provider: str, model: str) -> str:
    candidate = str(model).strip()
    for prefix in _PROVIDER_MODEL_PREFIXES.get(provider, ()):
        if candidate.startswith(prefix):
            return candidate.removeprefix(prefix)
    return candidate


def _is_openrouter_router_model(model_id: str) -> bool:
    head = model_id.split("/", 1)[0].strip().lower()
    return head in _OPENROUTER_ROUTER_VENDORS or model_id.lower() in {"auto", "router"}


def _is_chat_model(info: dict[str, Any] | None) -> bool:
    if not isinstance(info, dict):
        return False
    if info.get("mode") != "chat":
        return False
    return info.get("supports_response_schema") is not False


def _catalog_models_from_litellm(provider: str, source_models: Iterable[str]) -> list[str]:
    import litellm

    models: list[str] = []
    for raw_model in source_models:
        raw = str(raw_model).strip()
        if not raw:
            continue
        candidate = _normalize_model_id(provider, raw)
        if not candidate:
            continue
        if provider == "openrouter" and _is_openrouter_router_model(candidate):
            continue
        info = litellm.get_model_info(raw) or litellm.get_model_info(candidate)
        if not _is_chat_model(info):
            continue
        models.append(candidate)
    return _unique_preserve_order(sorted(models, key=str.lower))


def _ollama_models() -> list[str]:
    models = [
        item["name"]
        for item in list_recommended_models()
        if isinstance(item, dict) and isinstance(item.get("name"), str) and item["name"].strip()
    ]
    return _unique_preserve_order(models)


def _openrouter_models() -> list[str]:
    try:
        models = [item.id for item in search_openrouter_models(require_strict_json_schema=True)]
    except Exception as exc:
        logger.warning("OpenRouter model lookup failed: %s", exc)
        models = []
    if models:
        normalized = (
            _normalize_model_id("openrouter", model)
            for model in models
            if not _is_openrouter_router_model(_normalize_model_id("openrouter", model))
        )
        return _unique_preserve_order(normalized)

    import litellm

    return _catalog_models_from_litellm("openrouter", getattr(litellm, "openrouter_models", set()))


def _provider_model_ids(provider: str) -> list[str]:
    provider = _normalize_provider(provider)
    if provider not in _MODEL_PROVIDER_NAMES:
        raise KeyError(provider)
    if provider == "ollama":
        return _ollama_models()
    if provider == "openrouter":
        return _openrouter_models()

    import litellm

    source_name = {
        "anthropic": "anthropic_models",
        "google": "gemini_models",
        "openai": "open_ai_chat_completion_models",
    }[provider]
    source_models = getattr(litellm, source_name, set())
    return _catalog_models_from_litellm(provider, source_models)


def list_provider_models(provider: str) -> ProviderModelListResponse:
    provider = _normalize_provider(provider)
    return ProviderModelListResponse(provider=provider, models=_provider_model_ids(provider))


def _provider_api_key(provider: str, credentials: dict[str, ProviderCredentials]) -> str:
    env_key = os.environ.get(_KENKUI_PROVIDER_ENV_VARS.get(provider, ""), "")
    if env_key:
        return env_key
    std_key = os.environ.get(_PROVIDER_ENV_VARS.get(provider, ""), "")
    if std_key:
        return std_key
    stored = credentials.get(provider)
    return stored.api_key if stored is not None else ""


def _provider_default_model(provider: str, credentials: dict[str, ProviderCredentials]) -> str:
    stored = credentials.get(provider)
    return stored.default_model.strip() if stored is not None else ""


def _runtime_model(provider: str, model: str) -> str:
    prefixes = {
        "anthropic": ("anthropic/",),
        "google": ("gemini/", "google/", "vertex_ai/"),
        "ollama": ("ollama/",),
        "openai": ("openai/",),
        "openrouter": ("openrouter/",),
    }.get(provider, ())
    if model.startswith(prefixes):
        return model
    if provider == "google":
        return f"gemini/{model}"
    if provider in {"anthropic", "ollama", "openai", "openrouter"}:
        return f"{provider}/{model}"
    return model


def _run_provider_smoke_test(provider: str, model: str, api_key: str) -> None:
    import litellm

    try:
        litellm.completion(
            model=_runtime_model(provider, model),
            messages=[{"role": "user", "content": "Reply with OK."}],
            api_key=api_key,
            timeout=20,
            temperature=0,
            max_tokens=16,
        )
    except Exception as exc:
        raise RuntimeError(f"{provider} credential test failed: {exc}") from exc


def validate_provider_credentials(
    provider: str,
    *,
    credentials: dict[str, ProviderCredentials] | None = None,
) -> OkResponse:
    provider = _normalize_provider(provider)
    if provider not in _CREDENTIAL_PROVIDER_NAMES:
        raise KeyError(provider)
    credentials = credentials if credentials is not None else load_provider_credentials()
    api_key = _provider_api_key(provider, credentials)
    if not api_key:
        raise ValueError(f"No credentials configured for provider: {provider}")

    model = _provider_default_model(provider, credentials)
    if not model:
        available_models = _provider_model_ids(provider)
        if available_models:
            model = available_models[0]
    if not model:
        raise ValueError(f"No model available to test provider: {provider}")

    _run_provider_smoke_test(provider, model, api_key)
    return OkResponse(status="ok", message=f"Validated {provider} credentials.")


__all__ = [
    "list_provider_models",
    "validate_provider_credentials",
]
