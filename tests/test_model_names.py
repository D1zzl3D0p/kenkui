from __future__ import annotations

import kenkui
from kenkui.model_names import normalize_model_for_provider


def test_public_api_reexports_normalize():
    assert kenkui.normalize_model_for_provider is normalize_model_for_provider
    assert "normalize_model_for_provider" in kenkui.__all__


def test_normalize_openrouter_anthropic_canonical_slug():
    assert (
        normalize_model_for_provider("openrouter", "anthropic/claude-4.5-sonnet-20250929")
        == "anthropic/claude-sonnet-4.5"
    )


def test_normalize_openrouter_litellm_canonical_slug():
    assert (
        normalize_model_for_provider(
            "litellm", "openrouter/anthropic/claude-4.5-sonnet-20250929"
        )
        == "openrouter/anthropic/claude-sonnet-4.5"
    )


def test_normalize_litellm_direct_anthropic_snapshot_id():
    assert (
        normalize_model_for_provider("litellm", "anthropic/claude-4.5-sonnet-20250929")
        == "anthropic/claude-sonnet-4-5-20250929"
    )


def test_normalize_anthropic_first_party_snapshot_id():
    assert (
        normalize_model_for_provider("anthropic", "claude-4.5-sonnet-20250929")
        == "claude-sonnet-4-5-20250929"
    )


def test_normalize_passthrough_and_trim():
    assert normalize_model_for_provider("openai", "  gpt-4o  ") == "gpt-4o"
    assert normalize_model_for_provider("anthropic", "") == ""
    assert normalize_model_for_provider("unknown", "some-model") == "some-model"
