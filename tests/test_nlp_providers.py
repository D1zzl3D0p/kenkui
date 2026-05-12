import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch
from kenkui.models import AppConfig
from kenkui.nlp.providers import get_provider
from kenkui.nlp.providers.ollama import OllamaProvider
from kenkui.nlp.models import CharacterRecord as NLPCharacterRecord, AttributionItem, AttributionResult, AttributionItemWire, AttributionResultWire
from kenkui.models import FastScanResult, CharacterInfo


def test_get_provider_ollama():
    config = AppConfig(nlp_provider="ollama")
    provider = get_provider(config)
    assert isinstance(provider, OllamaProvider)


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
        roster = provider.build_roster([mock_chapter], book_path=Path("/tmp/test_book.epub"))

    assert len(roster.characters) == 1
    c = roster.characters[0]
    assert c.slug == "elizabeth_bennet"
    assert c.canonical_name == "Elizabeth Bennet"
    assert "Lizzy" in c.aliases


def test_ollama_provider_attribute_chapter_uses_slug_keyed_pipeline():
    """OllamaProvider.attribute_chapter uses the annotation-based pipeline returning slugs directly."""
    config = AppConfig(nlp_provider="ollama", nlp_model="llama3.2")
    provider = get_provider(config)

    roster = MagicMock()
    roster.characters = [
        NLPCharacterRecord(slug="elizabeth_bennet", canonical_name="Elizabeth Bennet"),
        NLPCharacterRecord(slug="mr_darcy", canonical_name="Mr. Darcy"),
    ]

    wire_result = AttributionResultWire(attributions=[
        AttributionItemWire(quote_id=1, speaker="elizabeth_bennet", confidence=5),
        AttributionItemWire(quote_id=2, speaker="mr_darcy", confidence=4),
        AttributionItemWire(quote_id=3, speaker="NARRATOR", confidence=5),
        AttributionItemWire(quote_id=4, speaker="Unknown", confidence=1),
    ])

    mock_chapter = MagicMock()
    mock_chapter.paragraphs = ['"It is a truth universally acknowledged," said Elizabeth.']

    with patch("kenkui.nlp.llm.LLMClient.generate", return_value=wire_result):
        with patch("kenkui.nlp.annotator.annotate_chapter", return_value="[QUOTE:1] text"):
            result = provider.attribute_chapter(mock_chapter, roster)

    speakers = {a.quote_id: a.speaker for a in result.attributions}
    assert speakers[1] == "elizabeth_bennet"
    assert speakers[2] == "mr_darcy"
    assert speakers[3] == "NARRATOR"
    assert speakers[4] == "Unknown"
