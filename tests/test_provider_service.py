from __future__ import annotations

from types import SimpleNamespace

from kenkui.config import ProviderCredentials
from kenkui.services.provider_service import (
    list_provider_models,
    validate_provider_credentials,
)


def test_list_provider_models_returns_ollama_recommendations(monkeypatch):
    monkeypatch.setattr(
        "kenkui.services.provider_service.list_recommended_models",
        lambda: [
            {"name": "llama3.2"},
            {"name": "gemma2:2b"},
            {"name": "llama3.2"},
        ],
    )

    result = list_provider_models("ollama")

    assert result.provider == "ollama"
    assert result.models == ["llama3.2", "gemma2:2b"]


def test_list_provider_models_normalizes_openrouter_models(monkeypatch):
    monkeypatch.setattr(
        "kenkui.services.provider_service.search_openrouter_models",
        lambda require_strict_json_schema=True: [
            SimpleNamespace(id="openai/gpt-4.1-mini"),
            SimpleNamespace(id="openrouter/auto"),
            SimpleNamespace(id="anthropic/claude-sonnet-4-5"),
        ],
    )

    result = list_provider_models("openrouter")

    assert result.provider == "openrouter"
    assert result.models == ["openai/gpt-4.1-mini", "anthropic/claude-sonnet-4-5"]


def test_test_provider_credentials_uses_saved_default_model(monkeypatch):
    captured: dict[str, str] = {}

    monkeypatch.setattr(
        "litellm.completion",
        lambda **kwargs: captured.update(model=kwargs["model"]) or None,
    )
    monkeypatch.setattr(
        "kenkui.services.provider_service._provider_default_model",
        lambda provider, credentials: "gpt-4o",
    )
    monkeypatch.setattr(
        "kenkui.services.provider_service._provider_api_key",
        lambda provider, credentials: "sk-secret",
    )

    result = validate_provider_credentials(
        "openai",
        credentials={
            "openai": ProviderCredentials(
                api_key="sk-secret",
                default_model="gpt-4o",
            )
        },
    )

    assert result.status == "ok"
    assert captured == {
        "model": "openai/gpt-4o",
    }


def test_test_provider_credentials_prefixes_anthropic_model(monkeypatch):
    captured: dict[str, str] = {}

    monkeypatch.setattr(
        "litellm.completion",
        lambda **kwargs: captured.update(model=kwargs["model"]) or None,
    )

    result = validate_provider_credentials(
        "anthropic",
        credentials={
            "anthropic": ProviderCredentials(
                api_key="sk-secret",
                default_model="claude-sonnet-4.6",
            )
        },
    )

    assert result.status == "ok"
    assert captured == {
        "model": "anthropic/claude-sonnet-4.6",
    }


def test_test_provider_credentials_falls_back_to_first_available_model(monkeypatch):
    captured: dict[str, str] = {}

    monkeypatch.setattr(
        "kenkui.services.provider_service._provider_model_ids",
        lambda provider: ["gpt-4o-mini"],
    )
    monkeypatch.setattr(
        "kenkui.services.provider_service._run_provider_smoke_test",
        lambda provider, model, api_key: captured.update(
            provider=provider,
            model=model,
            api_key=api_key,
        ),
    )

    result = validate_provider_credentials(
        "openai",
        credentials={
            "openai": ProviderCredentials(
                api_key="sk-secret",
                default_model="",
            )
        },
    )

    assert result.status == "ok"
    assert captured == {
        "provider": "openai",
        "model": "gpt-4o-mini",
        "api_key": "sk-secret",
    }
