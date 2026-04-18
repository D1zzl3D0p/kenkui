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


from kenkui.nlp.providers.cloud import merge_rosters, split_chapters_into_segments


def test_merge_rosters_exact_slug_match():
    """Same slug in two rosters → merged into one record with unioned data."""
    r1 = CharacterRoster(characters=[
        CharacterRecord(
            slug="elizabeth_bennet", canonical_name="Elizabeth Bennet",
            aliases=["Lizzy"], chapters=[0, 1],
            first_appearance=("book", 0), last_appearance=("book", 1),
            mention_count=50,
        )
    ])
    r2 = CharacterRoster(characters=[
        CharacterRecord(
            slug="elizabeth_bennet", canonical_name="Elizabeth Bennet",
            aliases=["Miss Bennet"], chapters=[2, 3],
            description="Witty heroine",
            first_appearance=("book", 2), last_appearance=("book", 3),
            mention_count=100,
        )
    ])
    merged = merge_rosters([r1, r2], book_slug="book")
    assert len(merged.characters) == 1
    c = merged.characters[0]
    assert set(c.aliases) >= {"Lizzy", "Miss Bennet"}
    assert set(c.chapters) == {0, 1, 2, 3}
    assert c.first_appearance == ("book", 0)
    assert c.last_appearance == ("book", 3)
    assert c.description == "Witty heroine"   # prefer non-empty
    assert c.mention_count == 150             # summed


def test_merge_rosters_alias_intersection():
    """Character with alias matching another's canonical → merged."""
    r1 = CharacterRoster(characters=[
        CharacterRecord(slug="mr_darcy", canonical_name="Mr. Darcy", aliases=["Darcy"],
                        chapters=[0], first_appearance=("book", 0), last_appearance=("book", 0))
    ])
    r2 = CharacterRoster(characters=[
        CharacterRecord(slug="darcy", canonical_name="Darcy", aliases=[],
                        chapters=[1], first_appearance=("book", 1), last_appearance=("book", 1))
    ])
    merged = merge_rosters([r1, r2], book_slug="book")
    assert len(merged.characters) == 1
    assert merged.characters[0].slug == "mr_darcy"   # keep richer canonical


def test_merge_rosters_distinct_characters():
    """Unrelated characters stay separate."""
    r1 = CharacterRoster(characters=[
        CharacterRecord(slug="frodo", canonical_name="Frodo Baggins", chapters=[0],
                        first_appearance=("book", 0), last_appearance=("book", 0))
    ])
    r2 = CharacterRoster(characters=[
        CharacterRecord(slug="sam_gamgee", canonical_name="Sam Gamgee", chapters=[0],
                        first_appearance=("book", 0), last_appearance=("book", 0))
    ])
    merged = merge_rosters([r1, r2], book_slug="book")
    assert len(merged.characters) == 2


def test_split_chapters_into_segments_overlap():
    """Segments have 2-chapter overlap and cover all chapters."""
    chapters = [MagicMock(index=i, paragraphs=[f"text {i}"]) for i in range(10)]
    # Force very small segment size so chunking triggers
    segments = split_chapters_into_segments(chapters, max_tokens_per_segment=5)
    # All chapters appear in at least one segment
    all_indices = {ch.index for seg in segments for ch in seg}
    assert all_indices == set(range(10))
    # Overlap: adjacent segments share ≥ 1 chapter (except possibly first)
    if len(segments) > 1:
        for i in range(len(segments) - 1):
            seg_indices = {ch.index for ch in segments[i]}
            next_indices = {ch.index for ch in segments[i + 1]}
            assert seg_indices & next_indices  # non-empty overlap
