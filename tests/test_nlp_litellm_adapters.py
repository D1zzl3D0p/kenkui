from __future__ import annotations

import json
import logging
import sys
from types import SimpleNamespace

from kenkui.models import AttributionTool, Chapter
from kenkui.nlp.models import (
    AttributionResultWire,
    CanonicalMergeResult,
    CharacterRecord,
    CharacterRoster,
    CharacterRosterWire,
    EpithetResolutionResult,
    NameNormalizationResult,
)
from kenkui.nlp.providers.litellm import (
    LiteLLMAttributionAdapter,
    LiteLLMClient,
    _litellm_model,
)
from kenkui.nlp_config import NLPConfig


def _assert_strict_object_schema(schema: dict):
    assert schema["additionalProperties"] is False
    for def_schema in schema.get("$defs", {}).values():
        if def_schema.get("type") == "object":
            assert def_schema["additionalProperties"] is False


def test_openrouter_runtime_model_prefixes_for_litellm_but_preserves_model_id():
    assert _litellm_model("openrouter", "openai/gpt-4.1-mini") == "openrouter/openai/gpt-4.1-mini"
    client = LiteLLMClient("openrouter", "openai/gpt-4.1-mini")
    assert client.model == "openai/gpt-4.1-mini"
    assert client.runtime_model == "openrouter/openai/gpt-4.1-mini"


def test_litellm_attribution_uses_completion_without_ollama(monkeypatch):
    calls = []

    def fake_completion(**kwargs):
        calls.append(kwargs)
        return {
            "choices": [
                {
                    "message": {
                        "content": json.dumps({"a": [{"q": 0, "s": "jane"}]}),
                    }
                }
            ]
        }

    monkeypatch.setitem(sys.modules, "litellm", SimpleNamespace(completion=fake_completion))

    config = NLPConfig(
        attribution_tool=AttributionTool.OPENROUTER,
        attribution_model="openai/gpt-4.1-mini",
    )
    adapter = LiteLLMAttributionAdapter(config)
    chapter = Chapter(index=0, title="Ch 1", paragraphs=['"Hello," Jane said.'])
    roster = CharacterRoster(
        characters=[
            CharacterRecord(
                slug="jane",
                canonical_name="Jane",
                aliases=["Jane"],
                gender="she/her",
            )
        ]
    )

    result = adapter.attribute_chapter(chapter, roster)

    assert result.attributions[0].speaker == "jane"
    assert calls[0]["model"] == "openrouter/openai/gpt-4.1-mini"
    assert calls[0]["messages"][0]["role"] == "user"
    assert calls[0]["response_format"]["type"] == "json_schema"


def test_litellm_wire_schemas_forbid_additional_properties():
    for schema in (
        AttributionResultWire,
        CharacterRosterWire,
        CanonicalMergeResult,
        EpithetResolutionResult,
        NameNormalizationResult,
    ):
        _assert_strict_object_schema(schema.model_json_schema())


def test_litellm_client_sends_strict_object_schema(monkeypatch):
    calls = []

    def fake_completion(**kwargs):
        calls.append(kwargs)
        return {
            "choices": [
                {
                    "message": {
                        "content": json.dumps({"a": [{"q": 0, "s": "jane"}]}),
                    }
                }
            ]
        }

    monkeypatch.setitem(sys.modules, "litellm", SimpleNamespace(completion=fake_completion))

    client = LiteLLMClient("openrouter", "openai/gpt-4.1-mini")
    client.generate("prompt text", AttributionResultWire)

    response_format = calls[0]["response_format"]
    assert response_format["type"] == "json_schema"
    assert response_format["json_schema"]["strict"] is True
    _assert_strict_object_schema(response_format["json_schema"]["schema"])


def test_litellm_attribution_warns_when_dialogue_quote_count_is_low(monkeypatch, caplog):
    def fake_completion(**kwargs):
        return {
            "choices": [
                {
                    "message": {
                        "content": json.dumps({"a": [{"q": 0, "s": "jane"}]}),
                    }
                }
            ]
        }

    monkeypatch.setitem(sys.modules, "litellm", SimpleNamespace(completion=fake_completion))

    config = NLPConfig(
        attribution_tool=AttributionTool.OPENROUTER,
        attribution_model="openai/gpt-4.1-mini",
    )
    adapter = LiteLLMAttributionAdapter(config)
    chapter = Chapter(index=3, title="Ch 4", paragraphs=['"Hello," Jane said. "Goodbye," Jane said.'])
    roster = CharacterRoster(
        characters=[
            CharacterRecord(
                slug="jane",
                canonical_name="Jane",
                aliases=["Jane"],
                gender="she/her",
            )
        ]
    )

    with caplog.at_level(logging.WARNING, logger="kenkui.nlp.providers.litellm"):
        adapter.attribute_chapter(chapter, roster)

    assert any("returned 1/2 dialogue quotes" in r.message for r in caplog.records)
