import pytest
from unittest.mock import MagicMock, patch
from kenkui.nlp.providers.cloud import (
    estimate_tokens,
    compute_output_budget,
    needs_chunking,
    CloudProvider,
)
from kenkui.models import AppConfig


def test_estimate_tokens_approx():
    # ~4 chars per token heuristic
    text = "a" * 4000
    assert 900 <= estimate_tokens(text) <= 1100


def test_compute_output_budget_reserves_20_percent():
    budget = compute_output_budget(context_limit=100_000, estimated_chars=0)
    assert budget == 20_000  # 20% of 100_000


def test_compute_output_budget_scales_with_characters():
    # More characters → larger budget (more characters in output)
    budget_few = compute_output_budget(context_limit=500_000, estimated_chars=10)
    budget_many = compute_output_budget(context_limit=500_000, estimated_chars=200)
    assert budget_many > budget_few


def test_needs_chunking_false_when_fits():
    # 10k tokens of text + 5k output budget + 2k system << 200k context
    assert not needs_chunking(
        input_tokens=10_000,
        output_budget=5_000,
        system_tokens=2_000,
        context_limit=200_000,
    )


def test_needs_chunking_true_when_over():
    assert needs_chunking(
        input_tokens=900_000,
        output_budget=100_000,
        system_tokens=5_000,
        context_limit=1_000_000,
    )


def test_cloud_provider_init():
    config = AppConfig(nlp_provider="anthropic", nlp_model="claude-sonnet-4-6")
    provider = CloudProvider(config)
    assert provider.config.nlp_provider == "anthropic"
