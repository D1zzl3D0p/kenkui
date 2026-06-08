from __future__ import annotations

import json
import logging
import re
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
    _openrouter_extra_body,
    _remote_attribution_call_budget,
    _remote_context_tokens,
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


def test_litellm_client_extracts_json_from_provider_wrapped_response(monkeypatch):
    def fake_completion(**kwargs):
        return {
            "choices": [
                {
                    "message": {
                        "content": (
                            "Provider List: https://docs.litellm.ai/docs/providers\n"
                            "```json\n"
                            '{"a": [{"q": 0, "s": "jane"}]}\n'
                            "```"
                        ),
                    }
                }
            ]
        }

    monkeypatch.setitem(sys.modules, "litellm", SimpleNamespace(completion=fake_completion))

    client = LiteLLMClient("openrouter", "openai/gpt-4.1-mini")
    result = client.generate("prompt text", AttributionResultWire)

    assert result.a[0].s == "jane"


def test_litellm_client_recovers_truncated_attribution_json(monkeypatch):
    def fake_completion(**kwargs):
        return {
            "choices": [
                {
                    "message": {
                        "content": '{"a": [{"q": 0, "s": "jane"},\n',
                    }
                }
            ]
        }

    monkeypatch.setitem(sys.modules, "litellm", SimpleNamespace(completion=fake_completion))

    client = LiteLLMClient("openrouter", "openai/gpt-4.1-mini")
    result = client.generate("prompt text", AttributionResultWire)

    assert [(item.q, item.s) for item in result.a] == [(0, "jane")]


def test_litellm_client_retries_unrecoverable_malformed_json_without_warning(monkeypatch, caplog):
    calls = []

    def fake_completion(**kwargs):
        calls.append(kwargs)
        content = '{"a":' if len(calls) == 1 else json.dumps({"a": [{"q": 0, "s": "jane"}]})
        return {"choices": [{"message": {"content": content}}]}

    monkeypatch.setitem(sys.modules, "litellm", SimpleNamespace(completion=fake_completion))

    client = LiteLLMClient("openrouter", "openai/gpt-4.1-mini")
    with caplog.at_level(logging.WARNING, logger="kenkui.nlp.providers.litellm"):
        result = client.generate("prompt text", AttributionResultWire)

    assert result.a[0].s == "jane"
    assert len(calls) == 2
    assert not any("validation failed response_chars" in r.message for r in caplog.records)


def test_litellm_client_strips_provider_list_from_warning_logs(monkeypatch, caplog):
    def fake_completion(**kwargs):
        raise RuntimeError(
            "No provider available\n"
            "Provider List: https://docs.litellm.ai/docs/providers"
        )

    monkeypatch.setitem(sys.modules, "litellm", SimpleNamespace(completion=fake_completion))

    client = LiteLLMClient("openrouter", "openai/gpt-4.1-mini")
    with caplog.at_level(logging.WARNING, logger="kenkui.nlp.providers.litellm"):
        try:
            client.generate("prompt text", AttributionResultWire)
        except RuntimeError:
            pass

    assert caplog.records
    assert not any("Provider List:" in r.message for r in caplog.records)
    assert not any("docs.litellm.ai/docs/providers" in r.message for r in caplog.records)


def test_openrouter_requires_providers_that_support_requested_parameters(monkeypatch):
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

    assert calls[0]["extra_body"]["provider"]["require_parameters"] is True
    assert calls[0]["response_format"]["type"] == "json_schema"
    assert calls[0]["response_format"]["json_schema"]["strict"] is True


def test_openrouter_parameter_requirement_is_not_sent_to_other_litellm_providers(monkeypatch):
    assert _openrouter_extra_body("openai") == {}
    assert _openrouter_extra_body("anthropic") == {}


def test_phi4_uses_context_hint_and_bounded_attribution_tokens(monkeypatch):
    monkeypatch.delenv("KENKUI_NLP_REMOTE_CONTEXT_TOKENS", raising=False)
    monkeypatch.delenv("KENKUI_NLP_REMOTE_OUTPUT_TOKENS", raising=False)

    assert _remote_context_tokens("openrouter/microsoft/phi-4") == 16384

    prompt = "x" * (8193 * 4)
    prompt_tokens, context_tokens, max_tokens = _remote_attribution_call_budget(
        prompt,
        quote_count=4,
        model="openrouter/microsoft/phi-4",
    )

    assert prompt_tokens == 8193
    assert context_tokens == 16384
    assert prompt_tokens + max_tokens < context_tokens
    assert max_tokens < 8192


def test_litellm_attribution_chunks_before_oversized_full_chapter(monkeypatch):
    calls = []

    def fake_completion(**kwargs):
        calls.append(kwargs)
        ids = [int(m) for m in re.findall(r"\[QUOTE:(\d+)", kwargs["messages"][0]["content"])]
        return {
            "choices": [
                {
                    "message": {
                        "content": json.dumps({"a": [{"q": qid, "s": "jane"} for qid in ids]}),
                    }
                }
            ]
        }

    monkeypatch.delenv("KENKUI_NLP_REMOTE_CONTEXT_TOKENS", raising=False)
    monkeypatch.setitem(sys.modules, "litellm", SimpleNamespace(completion=fake_completion))
    monkeypatch.setattr(
        "kenkui.nlp.annotator._build_attribution_static_block",
        lambda roster: "x" * 70000,
    )

    config = NLPConfig(
        attribution_tool=AttributionTool.OPENROUTER,
        attribution_model="microsoft/phi-4",
    )
    adapter = LiteLLMAttributionAdapter(config)
    paragraphs = [
        '"Hello," Jane said. ' + "word " * 720,
        '"Goodbye," Jane said.',
    ]
    chapter = Chapter(index=3, title="Ch 4", paragraphs=paragraphs)
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

    assert len(calls) >= 2
    assert all(call["model"] == "openrouter/microsoft/phi-4" for call in calls)
    assert all(call["max_tokens"] < 8192 for call in calls)
    assert {item.quote_id for item in result.attributions} == {0, 1}


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


def test_litellm_attribution_retries_missing_chunk_quotes(monkeypatch, caplog):
    calls = []

    def fake_completion(**kwargs):
        calls.append(kwargs)
        if len(calls) == 1:
            raise RuntimeError("context overflow")
        if len(calls) == 2:
            return {"choices": [{"message": {"content": json.dumps({"a": []})}}]}
        ids = [int(m) for m in re.findall(r"\[QUOTE:(\d+)", kwargs["messages"][0]["content"])]
        return {
            "choices": [
                {
                    "message": {
                        "content": json.dumps({"a": [{"q": ids[0], "s": "jane"}]}),
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
    chapter = Chapter(
        index=3,
        title="Ch 4",
        paragraphs=[
            '"One," Jane said. "Two," Jane said. "Three," Jane said. "Four," Jane said.',
        ],
    )
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
        result = adapter.attribute_chapter(chapter, roster)

    speakers = {item.quote_id: item.speaker for item in result.attributions}
    assert speakers == {0: "jane", 1: "jane", 2: "jane", 3: "jane"}
    assert len(calls) == 6
    assert any("retrying 4 missing quote" in r.message for r in caplog.records)
