from __future__ import annotations

import json
import urllib.parse

from kenkui.services.openrouter_service import search_openrouter_models


class _FakeResponse:
    def __init__(self, payload: dict):
        self._payload = payload

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def read(self) -> bytes:
        return json.dumps(self._payload).encode("utf-8")


def test_search_openrouter_models_filters_for_strict_json_schema(monkeypatch):
    calls = []

    def fake_urlopen(url, timeout):
        calls.append((url, timeout))
        return _FakeResponse(
            {
                "data": [
                    {
                        "id": "openai/gpt-4.1-mini",
                        "canonical_slug": "openai/gpt-4.1-mini",
                        "name": "GPT-4.1 Mini",
                        "description": "OpenAI model",
                        "context_length": 1047576,
                        "supported_parameters": [
                            "response_format",
                            "structured_outputs",
                            "temperature",
                        ],
                    },
                    {
                        "id": "example/plain-json",
                        "name": "Plain JSON",
                        "description": "Supports basic response format only",
                        "context_length": 8192,
                        "supported_parameters": ["response_format"],
                    },
                ]
            }
        )

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)

    models = search_openrouter_models(require_strict_json_schema=True, timeout=3)

    assert [model.id for model in models] == ["openai/gpt-4.1-mini"]
    assert models[0].supports_strict_json_schema is True
    assert models[0].context_length == 1047576

    url, timeout = calls[0]
    assert timeout == 3
    query = urllib.parse.parse_qs(urllib.parse.urlparse(url).query)
    assert query["output_modalities"] == ["text"]
    assert query["supported_parameters"] == ["structured_outputs"]


def test_search_openrouter_models_can_leave_strict_json_schema_filter_off(monkeypatch):
    calls = []

    def fake_urlopen(url, timeout):
        calls.append(url)
        return _FakeResponse(
            {
                "data": [
                    {
                        "id": "openai/gpt-4.1-mini",
                        "name": "GPT-4.1 Mini",
                        "supported_parameters": ["structured_outputs"],
                    },
                    {
                        "id": "example/plain-json",
                        "name": "Plain JSON",
                        "supported_parameters": ["response_format"],
                    },
                ]
            }
        )

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)

    models = search_openrouter_models(require_strict_json_schema=False)

    assert [model.id for model in models] == ["openai/gpt-4.1-mini", "example/plain-json"]
    assert [model.supports_strict_json_schema for model in models] == [True, False]

    query = urllib.parse.parse_qs(urllib.parse.urlparse(calls[0]).query)
    assert "supported_parameters" not in query


def test_search_openrouter_models_applies_text_search_and_limit(monkeypatch):
    def fake_urlopen(url, timeout):
        return _FakeResponse(
            {
                "data": [
                    {
                        "id": "anthropic/claude-sonnet-4.5",
                        "name": "Claude Sonnet 4.5",
                        "description": "Anthropic model",
                        "supported_parameters": ["structured_outputs"],
                    },
                    {
                        "id": "openai/gpt-4.1-mini",
                        "name": "GPT-4.1 Mini",
                        "description": "OpenAI model",
                        "supported_parameters": ["structured_outputs"],
                    },
                    {
                        "id": "openai/gpt-5-nano",
                        "name": "GPT-5 Nano",
                        "description": "OpenAI model",
                        "supported_parameters": ["structured_outputs"],
                    },
                ]
            }
        )

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)

    models = search_openrouter_models("gpt", require_strict_json_schema=True, limit=1)

    assert [model.id for model in models] == ["openai/gpt-4.1-mini"]
