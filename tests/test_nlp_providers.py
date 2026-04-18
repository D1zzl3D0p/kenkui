import pytest
from unittest.mock import MagicMock, patch
from kenkui.models import AppConfig
from kenkui.nlp.providers import get_provider
from kenkui.nlp.providers.ollama import OllamaProvider
from kenkui.nlp.providers.cloud import CloudProvider
from kenkui.nlp.models import CharacterRecord as NLPCharacterRecord, AttributionItem, AttributionResult
from kenkui.models import FastScanResult, CharacterInfo


def test_get_provider_ollama():
    config = AppConfig(nlp_provider="ollama")
    provider = get_provider(config)
    assert isinstance(provider, OllamaProvider)


def test_get_provider_anthropic():
    config = AppConfig(nlp_provider="anthropic")
    provider = get_provider(config)
    assert isinstance(provider, CloudProvider)


def test_get_provider_openai():
    config = AppConfig(nlp_provider="openai")
    provider = get_provider(config)
    assert isinstance(provider, CloudProvider)


def test_get_provider_google():
    config = AppConfig(nlp_provider="google")
    provider = get_provider(config)
    assert isinstance(provider, CloudProvider)


def test_get_provider_default_is_ollama():
    config = AppConfig()
    provider = get_provider(config)
    assert isinstance(provider, OllamaProvider)


def test_ollama_provider_build_roster_returns_character_roster():
    """OllamaProvider.build_roster wraps run_fast_scan and converts to CharacterRecord."""
    config = AppConfig(nlp_provider="ollama", nlp_model="llama3.2")
    provider = get_provider(config)

    mock_fast_scan_result = FastScanResult(
        roster=MagicMock(characters=[
            MagicMock(
                canonical="Elizabeth Bennet",
                aliases=["Lizzy", "Miss Bennet"],
                gender="she/her",
            )
        ]),
        characters=[
            CharacterInfo(
                character_id="Elizabeth Bennet",
                display_name="Elizabeth Bennet",
                mention_count=150,
                quote_count=80,
                gender_pronoun="she",
            )
        ],
        book_hash="abc123",
    )

    mock_chapter = MagicMock()
    mock_chapter.paragraphs = ["It is a truth universally acknowledged."]

    with patch("kenkui.nlp.providers.ollama.run_fast_scan", return_value=mock_fast_scan_result):
        roster = provider.build_roster([mock_chapter])

    assert len(roster.characters) == 1
    c = roster.characters[0]
    assert c.slug == "elizabeth_bennet"
    assert c.canonical_name == "Elizabeth Bennet"
    assert "Lizzy" in c.aliases


def test_ollama_provider_attribute_chapter_converts_speakers_to_slugs():
    """OllamaProvider.attribute_chapter converts canonical-name speakers to slugs."""
    config = AppConfig(nlp_provider="ollama", nlp_model="llama3.2")
    provider = get_provider(config)

    roster = MagicMock()
    roster.characters = [
        NLPCharacterRecord(slug="elizabeth_bennet", canonical_name="Elizabeth Bennet"),
        NLPCharacterRecord(slug="mr_darcy", canonical_name="Mr. Darcy"),
    ]

    # Simulate old pipeline returning canonical names as speakers
    old_attribution = AttributionResult(attributions=[
        AttributionItem(quote_id=1, speaker="Elizabeth Bennet", emotion="neutral", confidence=5),
        AttributionItem(quote_id=2, speaker="Mr. Darcy", emotion="neutral", confidence=4),
        AttributionItem(quote_id=3, speaker="NARRATOR", emotion="neutral", confidence=5),
        AttributionItem(quote_id=4, speaker="Unknown", emotion="neutral", confidence=1),
    ])

    mock_chapter = MagicMock()

    with patch("kenkui.nlp.providers.ollama._run_attribution_for_chapter", return_value=old_attribution):
        result = provider.attribute_chapter(mock_chapter, roster)

    speakers = {a.quote_id: a.speaker for a in result.attributions}
    assert speakers[1] == "elizabeth_bennet"
    assert speakers[2] == "mr_darcy"
    assert speakers[3] == "NARRATOR"   # preserved
    assert speakers[4] == "Unknown"    # preserved
