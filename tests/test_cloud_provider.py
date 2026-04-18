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


from kenkui.nlp.models import CharacterRecord, CharacterRoster, TitleRecord
from unittest.mock import patch


def _make_chapters(texts: list[str]) -> list:
    chapters = []
    for i, text in enumerate(texts):
        ch = MagicMock()
        ch.index = i
        ch.paragraphs = [text]
        chapters.append(ch)
    return chapters


def test_build_roster_single_pass(monkeypatch):
    """When book fits in context, build_roster makes one LLM call."""
    config = AppConfig(nlp_provider="anthropic", nlp_model="claude-sonnet-4-6")

    mock_roster = CharacterRoster(characters=[
        CharacterRecord(
            slug="frodo_baggins",
            canonical_name="Frodo Baggins",
            aliases=["Mr. Baggins"],
            gender="he/him",
            role="protagonist",
            description="A hobbit from the Shire.",
            chapters=[0],
            first_appearance=("fellowship", 0),
            last_appearance=("fellowship", 0),
        )
    ])

    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = mock_roster

    with patch("kenkui.nlp.providers.cloud.instructor") as mock_instructor:
        mock_instructor.from_litellm.return_value = mock_client
        with patch("kenkui.nlp.providers.cloud.litellm"):
            provider = CloudProvider(config)
            provider._client = mock_client

            ch = MagicMock()
            ch.index = 0
            ch.paragraphs = ["In a hole in the ground there lived a hobbit."]

            roster = provider.build_roster([ch])

    assert len(roster.characters) == 1
    assert roster.characters[0].slug == "frodo_baggins"
    mock_client.chat.completions.create.assert_called_once()


def test_build_roster_injects_series_context(monkeypatch):
    """Series roster characters appear in the prompt as 'known characters'."""
    config = AppConfig(nlp_provider="anthropic", nlp_model="claude-sonnet-4-6")
    series_roster = CharacterRoster(characters=[
        CharacterRecord(slug="gandalf", canonical_name="Gandalf", aliases=["Mithrandir"])
    ])

    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = CharacterRoster(characters=[])

    with patch("kenkui.config.load_provider_credentials", return_value={}):
        with patch("kenkui.config.inject_provider_env_vars"):
            provider = CloudProvider.__new__(CloudProvider)
            provider.config = config
            provider._client = mock_client

            ch = MagicMock()
            ch.index = 0
            ch.paragraphs = ["Gandalf the Grey appeared."]

            provider.build_roster([ch], series_roster=series_roster)

    call_kwargs = mock_client.chat.completions.create.call_args[1]
    prompt_text = str(call_kwargs.get("messages", ""))
    assert "Gandalf" in prompt_text
    assert "Mithrandir" in prompt_text
