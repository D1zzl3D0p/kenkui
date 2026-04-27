import pytest
from unittest.mock import MagicMock, patch
from kenkui.nlp.providers.cloud import (
    estimate_tokens,
    compute_roster_output_budget,
    compute_attribution_output_budget,
    needs_chunking,
    CloudProvider,
    _build_roster_prompt,
)
from kenkui.models import AppConfig


def test_estimate_tokens_approx():
    # ~4 chars per token heuristic
    text = "a" * 4000
    assert 900 <= estimate_tokens(text) <= 1100


def test_compute_roster_output_budget_minimum():
    # With 0 characters, should return minimum floor
    budget = compute_roster_output_budget(estimated_chars=0, compact=True)
    assert budget >= 2048


def test_compute_roster_output_budget_scales_with_characters():
    # More characters → larger budget
    budget_few = compute_roster_output_budget(estimated_chars=10, compact=True)
    budget_many = compute_roster_output_budget(estimated_chars=200, compact=True)
    assert budget_many > budget_few


def test_compute_attribution_output_budget_minimum():
    # With 0 quotes, should return minimum floor
    budget = compute_attribution_output_budget(num_quotes=0)
    assert budget >= 512


def test_compute_attribution_output_budget_scales_with_quotes():
    budget_few = compute_attribution_output_budget(num_quotes=10)
    budget_many = compute_attribution_output_budget(num_quotes=100)
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


def test_build_roster_prompt_escapes_literal_json_example():
    prompt = _build_roster_prompt("Mr. Darcy arrived.", series_roster=None)
    assert '[{"title": "Mr."}, {"title": "Queen of Andor"}]' in prompt


def test_build_roster_prompt_omits_description_field_by_default():
    prompt = _build_roster_prompt("Mr. Darcy arrived.", series_roster=None)
    assert "- description:" not in prompt


def test_build_roster_prompt_can_include_description_field():
    prompt = _build_roster_prompt(
        "Mr. Darcy arrived.",
        series_roster=None,
        include_descriptions=True,
        descriptions_protagonists_only=True,
    )
    assert "- description:" in prompt


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


from kenkui.nlp.providers.cloud import _call_with_rate_limit_retry
from instructor.exceptions import IncompleteOutputException


def test_call_with_rate_limit_retry_doubles_max_tokens_on_incomplete():
    """IncompleteOutputException triggers a retry with doubled max_tokens."""
    calls = []

    def fake_fn(**kwargs):
        calls.append(kwargs.get("max_tokens"))
        if len(calls) == 1:
            raise IncompleteOutputException(last_completion=None)
        return "ok"

    result = _call_with_rate_limit_retry(fake_fn, max_tokens=1000)
    assert result == "ok"
    assert calls[0] == 1000
    assert calls[1] == 2000


def test_call_with_rate_limit_retry_raises_after_max_incomplete_retries():
    """After exhausting retries, IncompleteOutputException is re-raised."""
    def always_incomplete(**kwargs):
        raise IncompleteOutputException(last_completion=None)

    with pytest.raises(IncompleteOutputException):
        _call_with_rate_limit_retry(always_incomplete, max_tokens=512)


def test_call_with_rate_limit_retry_retries_on_incomplete_message_text():
    """Retry should also trigger when providers surface truncation via message text."""
    calls = []

    def fake_fn(**kwargs):
        calls.append(kwargs.get("max_tokens"))
        if len(calls) == 1:
            raise RuntimeError("The output is incomplete due to a max_tokens length limit.")
        return "ok"

    result = _call_with_rate_limit_retry(fake_fn, max_tokens=1024)
    assert result == "ok"
    assert calls == [1024, 2048]


from kenkui.nlp.models import AttributionItem, AttributionResult, AttributionItemWire, AttributionResultWire
from kenkui.nlp.providers.cloud import _build_attribution_static_block, _build_attribution_dynamic_block


def test_attribute_chapter_returns_slug_speakers():
    """attribute_chapter returns AttributionResult with slug-keyed speakers."""
    config = AppConfig(nlp_provider="anthropic", nlp_model="claude-sonnet-4-6")

    roster = CharacterRoster(characters=[
        CharacterRecord(slug="frodo_baggins", canonical_name="Frodo Baggins"),
        CharacterRecord(slug="gandalf", canonical_name="Gandalf"),
    ])

    mock_result = AttributionResultWire(attributions=[
        AttributionItemWire(quote_id=0, speaker="frodo_baggins", confidence=5),
    ])

    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = mock_result

    with patch("kenkui.config.load_provider_credentials", return_value={}):
        with patch("kenkui.config.inject_provider_env_vars"):
            provider = CloudProvider.__new__(CloudProvider)
            provider.config = config
            provider._client = mock_client

            chapter = MagicMock()
            chapter.index = 0
            chapter.paragraphs = ['"I will take the Ring," said Frodo.']

            result = provider.attribute_chapter(chapter, roster)

    assert len(result.attributions) == 1
    speakers = {a.quote_id: a.speaker for a in result.attributions}
    assert speakers[0] == "frodo_baggins"


def test_build_attribution_static_block_has_pronouns():
    """Static block includes PRONOUNS column and character gender values."""
    roster = CharacterRoster(characters=[
        CharacterRecord(
            slug="darrow_of_lykos",
            canonical_name="Darrow of Lykos",
            aliases=["The Reaper", "Darrow"],
            gender="he/him",
        ),
        CharacterRecord(
            slug="lysander_au_lune",
            canonical_name="Lysander au Lune",
            aliases=["Lysander"],
            gender="he/him",
        ),
    ])
    block = _build_attribution_static_block(roster)
    assert "PRONOUNS" in block
    assert "he/him" in block
    assert "darrow_of_lykos" in block


def test_alias_to_slug_first_writer_wins():
    """When two characters share an alias, first-writer-wins and no crash occurs."""
    config = AppConfig(nlp_provider="anthropic", nlp_model="claude-sonnet-4-6")

    roster = CharacterRoster(characters=[
        CharacterRecord(
            slug="darrow_of_lykos",
            canonical_name="Darrow of Lykos",
            aliases=["the reaper"],
        ),
        CharacterRecord(
            slug="ares",
            canonical_name="Ares",
            aliases=["the reaper"],  # same alias — collision
        ),
    ])

    mock_result = AttributionResultWire(attributions=[])

    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = mock_result

    captured_alias_to_slug: dict = {}

    def fake_annotate(paragraphs, quotes, alias_to_slug, slug_to_pronoun):
        captured_alias_to_slug.update(alias_to_slug)
        return ""  # empty annotated text → 0 quotes → early return below

    with patch("kenkui.config.load_provider_credentials", return_value={}):
        with patch("kenkui.config.inject_provider_env_vars"):
            with patch("kenkui.nlp.annotator.annotate_chapter", fake_annotate):
                provider = CloudProvider.__new__(CloudProvider)
                provider.config = config
                provider._client = mock_client

                chapter = MagicMock()
                chapter.index = 0
                chapter.paragraphs = ['"Come," said the Reaper.']

                result = provider.attribute_chapter(chapter, roster)

    # The collision key "the reaper" must map to exactly one slug (first writer wins)
    reaper_slug = captured_alias_to_slug.get("the reaper")
    assert reaper_slug == "darrow_of_lykos", (
        f"Expected first-writer slug 'darrow_of_lykos', got {reaper_slug!r}"
    )
    # The second character's unambiguous canonical name is still present
    assert "ares" in captured_alias_to_slug


def test_attribute_chapter_uses_annotated_format():
    """Dynamic block uses ANNOTATED CHAPTER: header (not old CHAPTER TEXT: header)."""
    from kenkui.nlp.quotes import extract_quotes, strip_scare_quotes
    from kenkui.nlp.annotator import annotate_chapter

    paragraphs = ['"I will take the Ring," said Frodo.']
    clean = strip_scare_quotes(paragraphs)
    quotes = extract_quotes(clean)
    alias_to_slug = {"frodo": "frodo_baggins", "frodo baggins": "frodo_baggins"}
    slug_to_pronoun = {"frodo_baggins": "he/him"}
    annotated = annotate_chapter(clean, quotes, alias_to_slug, slug_to_pronoun)
    dynamic_block = _build_attribution_dynamic_block(annotated)
    assert dynamic_block.startswith("ANNOTATED CHAPTER:")
    assert "CHAPTER TEXT:" not in dynamic_block
