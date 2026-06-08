from __future__ import annotations

import json
import logging
import re
import sys
from types import SimpleNamespace

from kenkui.models import AttributionTool, Chapter
from kenkui.nlp.models import (
    AttributionResultConfidenceWire,
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
    assert "default" not in schema
    assert schema["additionalProperties"] is False
    assert sorted(schema["required"]) == sorted(schema["properties"])
    for property_schema in schema["properties"].values():
        assert "default" not in property_schema
    for def_schema in schema.get("$defs", {}).values():
        if def_schema.get("type") == "object":
            assert def_schema["additionalProperties"] is False
            assert sorted(def_schema["required"]) == sorted(def_schema["properties"])
            for property_schema in def_schema["properties"].values():
                assert "default" not in property_schema


def _assert_forbids_additional_properties(schema: dict):
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
        AttributionResultConfidenceWire,
        CharacterRosterWire,
        CanonicalMergeResult,
        EpithetResolutionResult,
        NameNormalizationResult,
    ):
        _assert_forbids_additional_properties(schema.model_json_schema())


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


def test_litellm_client_does_not_rewrite_non_openrouter_schema_or_routing(monkeypatch):
    calls = []

    def fake_completion(**kwargs):
        calls.append(kwargs)
        return {
            "choices": [
                {
                    "message": {
                        "content": json.dumps({"characters": []}),
                    }
                }
            ]
        }

    monkeypatch.setitem(sys.modules, "litellm", SimpleNamespace(completion=fake_completion))

    client = LiteLLMClient("openai", "gpt-4.1-mini")
    client.generate("prompt text", CharacterRosterWire)

    schema = calls[0]["response_format"]["json_schema"]["schema"]
    assert calls[0]["temperature"] == 0
    assert "extra_body" not in calls[0]
    assert "reasoning" not in calls[0]
    character_schema = schema.get("$defs", {}).get("CharacterRecordWire", {})
    assert "aliases" in character_schema.get("properties", {})
    assert "aliases" not in character_schema.get("required", [])


def test_litellm_client_prefers_async_completion(monkeypatch):
    calls = []

    def fake_completion(**kwargs):
        raise AssertionError("sync completion should not be used when acompletion exists")

    async def fake_acompletion(**kwargs):
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

    monkeypatch.setitem(
        sys.modules,
        "litellm",
        SimpleNamespace(completion=fake_completion, acompletion=fake_acompletion),
    )

    client = LiteLLMClient("openrouter", "openai/gpt-4.1-mini")
    result = client.generate("prompt text", AttributionResultWire)

    assert result.a[0].s == "jane"
    assert calls[0]["model"] == "openrouter/openai/gpt-4.1-mini"


def test_litellm_client_extracts_content_from_object_response(monkeypatch):
    def fake_completion(**kwargs):
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(
                        content=json.dumps({"a": [{"q": 0, "s": "jane"}]})
                    )
                )
            ]
        )

    monkeypatch.setitem(sys.modules, "litellm", SimpleNamespace(completion=fake_completion))

    client = LiteLLMClient("openrouter", "openai/gpt-4.1-mini")
    result = client.generate("prompt text", AttributionResultWire)

    assert result.a[0].s == "jane"


def test_litellm_client_rejects_non_content_response_fields(monkeypatch):
    calls = []

    def fake_completion(**kwargs):
        calls.append(kwargs)
        return {
            "choices": [
                {
                    "message": {
                        "parsed": {"a": [{"q": 0, "s": "jane"}]},
                    },
                    "finish_reason": "stop",
                }
            ]
        }

    monkeypatch.setitem(sys.modules, "litellm", SimpleNamespace(completion=fake_completion))

    client = LiteLLMClient("openrouter", "openai/gpt-4.1-mini")
    try:
        client.generate("prompt text", AttributionResultWire)
    except ConnectionError:
        pass
    else:
        raise AssertionError("non-content response fields must not be parsed as JSON")

    assert len(calls) == 3


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
    assert "temperature" not in calls[0]


def test_openrouter_parameter_routing_failure_stays_strict(monkeypatch, caplog):
    calls = []

    def fake_completion(**kwargs):
        calls.append(kwargs)
        raise RuntimeError(
            "NotFoundError: OpenrouterException - "
            '{"error":{"message":"No endpoints found that can handle the requested '
            'parameters.","code":404}}'
        )

    monkeypatch.setitem(sys.modules, "litellm", SimpleNamespace(completion=fake_completion))

    client = LiteLLMClient("openrouter", "openai/gpt-5-nano")
    with caplog.at_level(logging.WARNING, logger="kenkui.nlp.providers.litellm"):
        try:
            client.generate("prompt text", AttributionResultWire)
        except RuntimeError:
            pass
        else:
            raise AssertionError("strict OpenRouter routing failure should be surfaced")

    assert len(calls) == 1
    assert calls[0]["extra_body"]["provider"]["require_parameters"] is True
    assert calls[0]["response_format"]["type"] == "json_schema"
    assert "temperature" not in calls[0]
    assert any("strict OpenRouter parameter routing failed" in r.message for r in caplog.records)
    assert not any("retrying without provider.require_parameters" in r.message for r in caplog.records)


def test_openrouter_parameter_requirement_is_not_sent_to_other_litellm_providers(monkeypatch):
    assert _openrouter_extra_body("openai") == {}
    assert _openrouter_extra_body("anthropic") == {}


def test_openrouter_gpt5_uses_reasoning_controls_and_larger_attribution_budget(monkeypatch):
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

    monkeypatch.setenv("KENKUI_NLP_OPENROUTER_REASONING_EFFORT", "low")
    monkeypatch.setitem(sys.modules, "litellm", SimpleNamespace(completion=fake_completion))

    client = LiteLLMClient("openrouter", "openai/gpt-5-nano")
    client.generate("prompt text", AttributionResultWire, max_tokens=2048)

    assert calls[0]["extra_body"]["reasoning"] == {"effort": "low", "exclude": True}
    assert calls[0]["max_tokens"] == 2048

    prompt_tokens, context_tokens, max_tokens = _remote_attribution_call_budget(
        "short prompt",
        quote_count=1,
        model="openrouter/openai/gpt-5-nano",
    )
    assert prompt_tokens < context_tokens
    assert max_tokens == 2048


def test_empty_content_logs_provider_metadata(monkeypatch, caplog):
    calls = []

    def fake_completion(**kwargs):
        calls.append(kwargs)
        return {
            "choices": [
                {
                    "message": {
                        "content": "",
                        "refusal": "schema not supported",
                    },
                    "finish_reason": "stop",
                    "native_finish_reason": "length",
                }
            ],
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 1,
                "total_tokens": 11,
            },
        }

    monkeypatch.setitem(sys.modules, "litellm", SimpleNamespace(completion=fake_completion))

    client = LiteLLMClient("openrouter", "openai/gpt-4.1-mini")
    with caplog.at_level(logging.WARNING, logger="kenkui.nlp.providers.litellm"):
        try:
            client.generate("prompt text", AttributionResultWire)
        except ConnectionError:
            pass

    assert len(calls) == 3
    assert any("finish_reason='stop'" in r.message for r in caplog.records)
    assert any("native_finish_reason='length'" in r.message for r in caplog.records)
    assert any("refusal='schema not supported'" in r.message for r in caplog.records)
    assert any("'total_tokens': 11" in r.message for r in caplog.records)
    assert any("message_keys=['content', 'refusal']" in r.message for r in caplog.records)


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
    assert len(calls) == 3
    assert any("retrying 4 missing quote" in r.message for r in caplog.records)


def test_litellm_attribution_repairs_missing_full_chapter_quote_ids(monkeypatch):
    calls = []

    def fake_completion(**kwargs):
        calls.append(kwargs)
        ids = [int(m) for m in re.findall(r"\[QUOTE:(\d+)", kwargs["messages"][0]["content"])]
        payload = {"a": [{"q": 0, "s": "jane"}]} if len(calls) == 1 else {
            "a": [{"q": qid, "s": "jane"} for qid in ids]
        }
        return {"choices": [{"message": {"content": json.dumps(payload)}}]}

    monkeypatch.setitem(sys.modules, "litellm", SimpleNamespace(completion=fake_completion))

    adapter = LiteLLMAttributionAdapter(NLPConfig(
        attribution_tool=AttributionTool.OPENROUTER,
        attribution_model="openai/gpt-4.1-mini",
    ))
    chapter = Chapter(index=0, title="Ch 1", paragraphs=['"One," Jane said. "Two," Jane said.'])
    roster = CharacterRoster(characters=[
        CharacterRecord(slug="jane", canonical_name="Jane", aliases=["Jane"], gender="she/her")
    ])

    result = adapter.attribute_chapter(chapter, roster)

    assert {item.quote_id for item in result.attributions} == {0, 1}
    assert {item.speaker for item in result.attributions} == {"jane"}
    assert len(calls) == 2


def test_missing_quote_repair_prompt_includes_adjacent_paragraphs(monkeypatch):
    calls = []

    def fake_completion(**kwargs):
        calls.append(kwargs["messages"][0]["content"])
        if len(calls) == 1:
            return {"choices": [{"message": {"content": json.dumps({"a": []})}}]}
        ids = [int(m) for m in re.findall(r"\[QUOTE:(\d+)", calls[-1])]
        return {
            "choices": [{
                "message": {"content": json.dumps({"a": [{"q": qid, "s": "jane"} for qid in ids]})}
            }]
        }

    monkeypatch.setitem(sys.modules, "litellm", SimpleNamespace(completion=fake_completion))

    adapter = LiteLLMAttributionAdapter(NLPConfig(
        attribution_tool=AttributionTool.OPENROUTER,
        attribution_model="openai/gpt-4.1-mini",
    ))
    chapter = Chapter(index=0, title="Ch 1", paragraphs=[
        "Before context with Jane.",
        '"Middle," Jane said.',
        "After context with Jane.",
    ])
    roster = CharacterRoster(characters=[
        CharacterRecord(slug="jane", canonical_name="Jane", aliases=["Jane"], gender="she/her")
    ])

    adapter.attribute_chapter(chapter, roster)

    assert "Before context" in calls[1]
    assert "After context" in calls[1]
    assert "[QUOTE:0" in calls[1]


def test_missing_quote_repair_batches_multiple_quotes(monkeypatch):
    calls = []

    def fake_completion(**kwargs):
        content = kwargs["messages"][0]["content"]
        calls.append(content)
        if len(calls) == 1:
            return {"choices": [{"message": {"content": json.dumps({"a": []})}}]}
        ids = [int(m) for m in re.findall(r"\[QUOTE:(\d+)", content)]
        return {
            "choices": [{
                "message": {"content": json.dumps({"a": [{"q": qid, "s": "jane"} for qid in ids]})}
            }]
        }

    monkeypatch.setitem(sys.modules, "litellm", SimpleNamespace(completion=fake_completion))

    adapter = LiteLLMAttributionAdapter(NLPConfig(
        attribution_tool=AttributionTool.OPENROUTER,
        attribution_model="openai/gpt-4.1-mini",
    ))
    chapter = Chapter(index=0, title="Ch 1", paragraphs=[
        '"One," Jane said. "Two," Jane said. "Three," Jane said.',
    ])
    roster = CharacterRoster(characters=[
        CharacterRecord(slug="jane", canonical_name="Jane", aliases=["Jane"], gender="she/her")
    ])

    result = adapter.attribute_chapter(chapter, roster)

    assert len(calls) == 2
    assert len(re.findall(r"\[QUOTE:\d+", calls[1])) == 3
    assert {item.speaker for item in result.attributions} == {"jane"}


def test_hint_conflict_is_corrected_without_resolver(monkeypatch):
    calls = []

    def fake_completion(**kwargs):
        calls.append(kwargs)
        return {"choices": [{"message": {"content": json.dumps({"a": [{"q": 0, "s": "bob"}]})}}]}

    monkeypatch.setitem(sys.modules, "litellm", SimpleNamespace(completion=fake_completion))

    adapter = LiteLLMAttributionAdapter(NLPConfig(
        attribution_tool=AttributionTool.OPENROUTER,
        attribution_model="openai/gpt-4.1-mini",
    ))
    chapter = Chapter(index=0, title="Ch 1", paragraphs=['"Hello," Jane said.'])
    roster = CharacterRoster(characters=[
        CharacterRecord(slug="jane", canonical_name="Jane", aliases=["Jane"], gender="she/her"),
        CharacterRecord(slug="bob", canonical_name="Bob", aliases=["Bob"], gender="he/him"),
    ])

    result = adapter.attribute_chapter(chapter, roster)

    assert result.attributions[0].speaker == "jane"
    assert len(calls) == 1


def test_guess_conflict_is_flagged_but_overridable_by_resolver(monkeypatch):
    calls = []

    def fake_completion(**kwargs):
        calls.append(kwargs)
        speaker = "jane" if kwargs["model"] == "openrouter/reviewer" else "bob"
        return {"choices": [{"message": {"content": json.dumps({"a": [{"q": 0, "s": speaker}]})}}]}

    monkeypatch.setitem(sys.modules, "litellm", SimpleNamespace(completion=fake_completion))

    adapter = LiteLLMAttributionAdapter(NLPConfig(
        attribution_tool=AttributionTool.OPENROUTER,
        attribution_model="openai/gpt-4.1-mini",
        review_model="reviewer",
    ))
    chapter = Chapter(index=0, title="Ch 1", paragraphs=['"Hello," Jane smiled.'])
    roster = CharacterRoster(characters=[
        CharacterRecord(slug="jane", canonical_name="Jane", aliases=["Jane"], gender="she/her"),
        CharacterRecord(slug="bob", canonical_name="Bob", aliases=["Bob"], gender="he/him"),
    ])

    result = adapter.attribute_chapter(chapter, roster)

    assert result.attributions[0].speaker == "jane"
    assert [call["model"] for call in calls] == ["openrouter/openai/gpt-4.1-mini", "openrouter/reviewer"]


def test_pronoun_conflict_rejects_incompatible_roster_speaker(monkeypatch):
    def fake_completion(**kwargs):
        return {"choices": [{"message": {"content": json.dumps({"a": [{"q": 0, "s": "jane"}]})}}]}

    monkeypatch.setitem(sys.modules, "litellm", SimpleNamespace(completion=fake_completion))

    adapter = LiteLLMAttributionAdapter(NLPConfig(
        attribution_tool=AttributionTool.OPENROUTER,
        attribution_model="openai/gpt-4.1-mini",
    ))
    chapter = Chapter(index=0, title="Ch 1", paragraphs=['"Hello," he said.'])
    roster = CharacterRoster(characters=[
        CharacterRecord(slug="jane", canonical_name="Jane", aliases=["Jane"], gender="she/her")
    ])

    result = adapter.attribute_chapter(chapter, roster)

    assert result.attributions[0].speaker == "Unknown"


def test_optional_attribution_confidence_schema_preserves_confidence(monkeypatch):
    calls = []

    def fake_completion(**kwargs):
        calls.append(kwargs)
        return {"choices": [{"message": {"content": json.dumps({"a": [{"q": 0, "s": "jane", "c": 5}]})}}]}

    monkeypatch.setitem(sys.modules, "litellm", SimpleNamespace(completion=fake_completion))

    adapter = LiteLLMAttributionAdapter(NLPConfig(
        attribution_tool=AttributionTool.OPENROUTER,
        attribution_model="openai/gpt-4.1-mini",
        attribution_review_confidence=True,
    ))
    chapter = Chapter(index=0, title="Ch 1", paragraphs=['"Hello," Jane said.'])
    roster = CharacterRoster(characters=[
        CharacterRecord(slug="jane", canonical_name="Jane", aliases=["Jane"], gender="she/her")
    ])

    result = adapter.attribute_chapter(chapter, roster)

    assert result.attributions[0].confidence == 5
    schema = calls[0]["response_format"]["json_schema"]["schema"]
    item_ref = schema["properties"]["a"]["items"]["$ref"].split("/")[-1]
    assert "c" in schema["$defs"][item_ref]["properties"]


def test_quote_count_cap_splits_calls_when_token_budget_fits(monkeypatch):
    calls = []

    def fake_completion(**kwargs):
        calls.append(kwargs)
        ids = [int(m) for m in re.findall(r"\[QUOTE:(\d+)", kwargs["messages"][0]["content"])]
        return {
            "choices": [{
                "message": {"content": json.dumps({"a": [{"q": qid, "s": "jane"} for qid in ids]})}
            }]
        }

    monkeypatch.setitem(sys.modules, "litellm", SimpleNamespace(completion=fake_completion))

    adapter = LiteLLMAttributionAdapter(NLPConfig(
        attribution_tool=AttributionTool.OPENROUTER,
        attribution_model="openai/gpt-4.1-mini",
        attribution_max_quotes_per_call=1,
    ))
    chapter = Chapter(index=0, title="Ch 1", paragraphs=[
        '"One," Jane said. "Two," Jane said. "Three," Jane said.',
    ])
    roster = CharacterRoster(characters=[
        CharacterRecord(slug="jane", canonical_name="Jane", aliases=["Jane"], gender="she/her")
    ])

    result = adapter.attribute_chapter(chapter, roster)

    assert len(calls) == 3
    assert all(len(re.findall(r"\[QUOTE:\d+", call["messages"][0]["content"])) == 1 for call in calls)
    assert {item.quote_id for item in result.attributions} == {0, 1, 2}


def test_review_model_resolves_only_suspicious_quotes(monkeypatch):
    calls = []

    def fake_completion(**kwargs):
        calls.append(kwargs)
        if kwargs["model"] == "openrouter/reviewer":
            return {"choices": [{"message": {"content": json.dumps({"a": [{"q": 0, "s": "jane"}]})}}]}
        return {
            "choices": [{
                "message": {"content": json.dumps({"a": [{"q": 0, "s": "bob"}, {"q": 1, "s": "bob"}]})}
            }]
        }

    monkeypatch.setitem(sys.modules, "litellm", SimpleNamespace(completion=fake_completion))

    adapter = LiteLLMAttributionAdapter(NLPConfig(
        attribution_tool=AttributionTool.OPENROUTER,
        attribution_model="openai/gpt-4.1-mini",
        review_model="reviewer",
    ))
    chapter = Chapter(index=0, title="Ch 1", paragraphs=[
        '"Hello," Jane said.',
        '"Right," Bob said.',
    ])
    roster = CharacterRoster(characters=[
        CharacterRecord(slug="jane", canonical_name="Jane", aliases=["Jane"], gender="she/her"),
        CharacterRecord(slug="bob", canonical_name="Bob", aliases=["Bob"], gender="he/him"),
    ])

    result = adapter.attribute_chapter(chapter, roster)

    speakers = {item.quote_id: item.speaker for item in result.attributions}
    assert speakers == {0: "jane", 1: "bob"}
    assert len(calls) == 2
    assert "[QUOTE:0" in calls[1]["messages"][0]["content"]
    assert "[QUOTE:1" not in calls[1]["messages"][0]["content"]


def test_no_resolver_call_when_review_model_is_empty(monkeypatch):
    calls = []

    def fake_completion(**kwargs):
        calls.append(kwargs)
        return {"choices": [{"message": {"content": json.dumps({"a": [{"q": 0, "s": "bob"}]})}}]}

    monkeypatch.setitem(sys.modules, "litellm", SimpleNamespace(completion=fake_completion))

    adapter = LiteLLMAttributionAdapter(NLPConfig(
        attribution_tool=AttributionTool.OPENROUTER,
        attribution_model="openai/gpt-4.1-mini",
    ))
    chapter = Chapter(index=0, title="Ch 1", paragraphs=['"Hello," Jane said.'])
    roster = CharacterRoster(characters=[
        CharacterRecord(slug="jane", canonical_name="Jane", aliases=["Jane"], gender="she/her"),
        CharacterRecord(slug="bob", canonical_name="Bob", aliases=["Bob"], gender="he/him"),
    ])

    adapter.attribute_chapter(chapter, roster)

    assert len(calls) == 1
