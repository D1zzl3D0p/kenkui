"""Tests for OllamaExtractionAdapter and OllamaAttributionAdapter."""
from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from kenkui.models import Chapter, FastScanResult, CharacterInfo
from kenkui.nlp.models import AttributionItem, AttributionResult, AttributionItemWire, AttributionResultWire, CharacterRecord, CharacterRoster
from kenkui.nlp.providers.ollama import OllamaAttributionAdapter, OllamaExtractionAdapter
from kenkui.nlp_config import NLPConfig


def _make_config(**kwargs):
    defaults = {
        "extraction_model": "llama3.2",
        "attribution_model": "llama3.2",
        "ollama_url": "http://localhost:11434",
    }
    defaults.update(kwargs)
    return NLPConfig(**defaults)


def _make_roster(*names):
    chars = [
        CharacterRecord(slug=n.lower().replace(" ", "_"), canonical_name=n)
        for n in names
    ]
    return CharacterRoster(characters=chars)


def _make_chapter(paragraphs=None):
    ch = MagicMock(spec=Chapter)
    ch.paragraphs = paragraphs or ["Hello world."]
    return ch


def _make_fast_scan_result(canonical_name: str, slug: str, aliases: list[str] | None = None):
    """Build a minimal FastScanResult with one character."""
    char_record = MagicMock()
    char_record.canonical_name = canonical_name
    char_record.slug = slug
    char_record.aliases = aliases or []
    char_record.gender = ""

    mock_roster = MagicMock()
    mock_roster.characters = [char_record]

    char_info = CharacterInfo(
        character_id=canonical_name,
        display_name=canonical_name,
        mention_count=10,
        quote_count=5,
    )
    return FastScanResult(roster=mock_roster, characters=[char_info], book_hash="abc123")


BOOK_PATH = Path("/tmp/test_book.epub")


class TestOllamaExtractionAdapter:
    def test_calls_run_fast_scan_with_extraction_model(self):
        config = _make_config(extraction_model="mistral")
        adapter = OllamaExtractionAdapter(config)
        fast_result = _make_fast_scan_result("Alice", "alice")

        with patch("kenkui.nlp.providers.ollama.run_fast_scan", return_value=fast_result) as mock_scan:
            result = adapter.build_roster([_make_chapter()], book_path=BOOK_PATH)

        assert result is not None
        call_args = mock_scan.call_args
        # extraction_model ("mistral") must appear as positional arg
        assert "mistral" in call_args.args

    def test_requires_book_path(self):
        config = _make_config()
        adapter = OllamaExtractionAdapter(config)
        with pytest.raises(ValueError, match="book_path"):
            adapter.build_roster([_make_chapter()])

    def test_passes_progress_callback(self):
        config = _make_config()
        adapter = OllamaExtractionAdapter(config)
        cb = MagicMock()
        fast_result = _make_fast_scan_result("Alice", "alice")

        with patch("kenkui.nlp.providers.ollama.run_fast_scan", return_value=fast_result) as mock_scan:
            adapter.build_roster([], book_path=BOOK_PATH, progress_callback=cb)

        _, kwargs = mock_scan.call_args
        assert kwargs.get("progress_callback") is cb

    def test_converts_fast_scan_result_to_character_roster(self):
        config = _make_config()
        adapter = OllamaExtractionAdapter(config)
        fast_result = _make_fast_scan_result("Elizabeth Bennet", "elizabeth_bennet", aliases=["Lizzy"])

        with patch("kenkui.nlp.providers.ollama.run_fast_scan", return_value=fast_result):
            result = adapter.build_roster([_make_chapter()], book_path=BOOK_PATH)

        assert len(result.characters) == 1
        c = result.characters[0]
        assert c.canonical_name == "Elizabeth Bennet"
        assert c.slug == "elizabeth_bennet"
        assert "Lizzy" in c.aliases

    def test_populates_mention_count_from_char_info(self):
        config = _make_config()
        adapter = OllamaExtractionAdapter(config)
        fast_result = _make_fast_scan_result("Alice", "alice")
        fast_result.characters[0] = CharacterInfo(
            character_id="Alice",
            display_name="Alice",
            mention_count=42,
            quote_count=7,
        )

        with patch("kenkui.nlp.providers.ollama.run_fast_scan", return_value=fast_result):
            result = adapter.build_roster([_make_chapter()], book_path=BOOK_PATH)

        assert result.characters[0].mention_count == 42

    def test_returns_empty_roster_when_no_characters(self):
        config = _make_config()
        adapter = OllamaExtractionAdapter(config)

        mock_roster = MagicMock()
        mock_roster.characters = []
        fast_result = FastScanResult(roster=mock_roster, characters=[], book_hash="xyz")

        with patch("kenkui.nlp.providers.ollama.run_fast_scan", return_value=fast_result):
            result = adapter.build_roster([], book_path=BOOK_PATH)

        assert result.characters == []


class TestOllamaAttributionAdapter:
    """Tests for OllamaAttributionAdapter.attribute_chapter.

    Patch targets use the original module paths because attribute_chapter
    uses local imports (from kenkui.nlp.llm import LLMClient, etc.).
    """

    def _wire_result(self, items: list[AttributionItemWire]) -> AttributionResultWire:
        return AttributionResultWire(attributions=items)

    def test_returns_attribution_result_for_chapter_with_quotes(self):
        config = _make_config()
        adapter = OllamaAttributionAdapter(config)
        roster = _make_roster("Alice")
        chapter = _make_chapter(['"Hello," said Alice.'])

        wire = self._wire_result([AttributionItemWire(quote_id=0, speaker="alice", confidence=5)])

        with patch("kenkui.nlp.quotes.strip_scare_quotes", return_value=['"Hello," said Alice.']), \
             patch("kenkui.nlp.quotes.extract_quotes", return_value=[MagicMock(id=0)]), \
             patch("kenkui.nlp.annotator._build_alias_to_slug", return_value={}), \
             patch("kenkui.nlp.annotator.annotate_chapter", return_value="[QUOTE:0] text"), \
             patch("kenkui.nlp.annotator._build_attribution_static_block", return_value="static"), \
             patch("kenkui.nlp.annotator._build_attribution_dynamic_block", return_value="dynamic"), \
             patch("kenkui.nlp.llm.LLMClient.generate", return_value=wire):
            result = adapter.attribute_chapter(chapter, roster)

        assert isinstance(result, AttributionResult)

    def test_returns_empty_result_when_no_quotes(self):
        config = _make_config()
        adapter = OllamaAttributionAdapter(config)
        roster = _make_roster("Alice")
        chapter = _make_chapter(["No dialogue here."])

        with patch("kenkui.nlp.quotes.strip_scare_quotes", return_value=["No dialogue here."]), \
             patch("kenkui.nlp.quotes.extract_quotes", return_value=[]):
            result = adapter.attribute_chapter(chapter, roster)

        assert isinstance(result, AttributionResult)
        assert result.attributions == []

    def test_calls_llm_with_attribution_model(self):
        config = _make_config(attribution_model="mixtral")
        adapter = OllamaAttributionAdapter(config)
        roster = _make_roster("Bob")
        chapter = _make_chapter(['"Hi," Bob said.'])

        wire = self._wire_result([AttributionItemWire(quote_id=0, speaker="bob", confidence=4)])

        captured_model = []

        def fake_init(self_inner, model):
            captured_model.append(model)

        with patch("kenkui.nlp.quotes.strip_scare_quotes", return_value=['"Hi," Bob said.']), \
             patch("kenkui.nlp.quotes.extract_quotes", return_value=[MagicMock(id=0)]), \
             patch("kenkui.nlp.annotator._build_alias_to_slug", return_value={}), \
             patch("kenkui.nlp.annotator.annotate_chapter", return_value="annotated"), \
             patch("kenkui.nlp.annotator._build_attribution_static_block", return_value=""), \
             patch("kenkui.nlp.annotator._build_attribution_dynamic_block", return_value=""), \
             patch("kenkui.nlp.llm.LLMClient.__init__", fake_init), \
             patch("kenkui.nlp.llm.LLMClient.generate", return_value=wire):
            adapter.attribute_chapter(chapter, roster)

        assert "mixtral" in captured_model

    def test_calls_progress_callback(self):
        config = _make_config()
        adapter = OllamaAttributionAdapter(config)
        roster = _make_roster()
        chapter = _make_chapter(["No dialogue."])
        cb = MagicMock()

        with patch("kenkui.nlp.quotes.strip_scare_quotes", return_value=["No dialogue."]), \
             patch("kenkui.nlp.quotes.extract_quotes", return_value=[]):
            adapter.attribute_chapter(chapter, roster, progress_callback=cb)

        cb.assert_called_once()

    def test_attribution_result_has_expected_quote_ids(self):
        config = _make_config()
        adapter = OllamaAttributionAdapter(config)
        roster = _make_roster("Alice", "Bob")
        chapter = _make_chapter(['"Hello," Alice said. "Bye," said Bob.'])

        wire = self._wire_result([
            AttributionItemWire(quote_id=0, speaker="alice", confidence=5),
            AttributionItemWire(quote_id=1, speaker="bob", confidence=4),
        ])

        with patch("kenkui.nlp.quotes.strip_scare_quotes", return_value=['"Hello," Alice said. "Bye," said Bob.']), \
             patch("kenkui.nlp.quotes.extract_quotes", return_value=[MagicMock(id=0), MagicMock(id=1)]), \
             patch("kenkui.nlp.annotator._build_alias_to_slug", return_value={}), \
             patch("kenkui.nlp.annotator.annotate_chapter", return_value="annotated"), \
             patch("kenkui.nlp.annotator._build_attribution_static_block", return_value=""), \
             patch("kenkui.nlp.annotator._build_attribution_dynamic_block", return_value=""), \
             patch("kenkui.nlp.llm.LLMClient.generate", return_value=wire):
            result = adapter.attribute_chapter(chapter, roster)

        quote_ids = {a.quote_id for a in result.attributions}
        assert 0 in quote_ids
        assert 1 in quote_ids


class TestOllamaProviderLegacyWrapper:
    """Verify OllamaProvider delegates correctly to the adapters."""

    def test_build_roster_delegates_to_extraction_adapter(self):
        from kenkui.models import AppConfig
        from kenkui.nlp.providers.ollama import OllamaProvider

        config = AppConfig(nlp_provider="ollama", nlp_model="llama3.2")
        provider = OllamaProvider(config)

        fast_result = _make_fast_scan_result("Alice", "alice")
        with patch("kenkui.nlp.providers.ollama.run_fast_scan", return_value=fast_result):
            result = provider.build_roster([_make_chapter()], book_path=BOOK_PATH)

        assert len(result.characters) == 1
        assert result.characters[0].canonical_name == "Alice"

    def test_attribute_chapter_delegates_to_attribution_adapter(self):
        from kenkui.models import AppConfig
        from kenkui.nlp.providers.ollama import OllamaProvider

        config = AppConfig(nlp_provider="ollama", nlp_model="llama3.2")
        provider = OllamaProvider(config)
        roster = _make_roster("Alice")
        chapter = _make_chapter(["No quotes."])

        with patch("kenkui.nlp.quotes.strip_scare_quotes", return_value=["No quotes."]), \
             patch("kenkui.nlp.quotes.extract_quotes", return_value=[]):
            result = provider.attribute_chapter(chapter, roster)

        assert isinstance(result, AttributionResult)
        assert result.attributions == []
