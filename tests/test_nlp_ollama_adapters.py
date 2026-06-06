"""Tests for OllamaExtractionAdapter and OllamaAttributionAdapter."""
from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from kenkui.models import Chapter, CharacterInfo, FastScanResult
from kenkui.nlp.models import (
    AttributionItemWire,
    AttributionResult,
    AttributionResultWire,
    CharacterRecord,
    CharacterRoster,
)
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
        return AttributionResultWire(a=items)

    def test_returns_attribution_result_for_chapter_with_quotes(self):
        config = _make_config()
        adapter = OllamaAttributionAdapter(config)
        roster = _make_roster("Alice")
        chapter = _make_chapter(['"Hello," said Alice.'])

        wire = self._wire_result([AttributionItemWire(q=0, s="alice")])

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

        wire = self._wire_result([AttributionItemWire(q=0, s="bob")])

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
            AttributionItemWire(q=0, s="alice"),
            AttributionItemWire(q=1, s="bob"),
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


# ---------------------------------------------------------------------------
# OllamaAttributionAdapter — quote-count gap diagnostic logging
#
# Regression test for:
#   WARNING: Chapter N: quote id=X missing from LLM attribution
#
# The adapter must warn immediately (before segment building) when the LLM
# returns fewer attributions than quotes sent, so operators can trace the
# gap to the attribution step rather than hunting through segment warnings.
# ---------------------------------------------------------------------------


class TestOllamaAttributionAdapterGapLogging:
    """When LLM returns fewer quotes than requested, a WARNING must fire at the adapter."""

    def _wire_result(self, items):
        return AttributionResultWire(a=items)

    def test_warns_when_llm_returns_fewer_quotes_than_requested(self, caplog):
        """Regression: gap between quotes sent and quotes returned was previously silent.

        If 3 quotes are sent and only 2 come back, the missing one is filled by
        _attribution_to_segments — but operators couldn't see where the loss happened.
        """
        import logging
        config = _make_config()
        adapter = OllamaAttributionAdapter(config)

        chapter = _make_chapter(['"First," she said. "Second," he said. "Third," it said.'])
        roster = _make_roster("Alice", "Bob")

        # LLM only returns attribution for 2 of 3 quotes
        wire = self._wire_result([
            AttributionItemWire(q=0, s="alice"),
            AttributionItemWire(q=1, s="bob"),
            # quote_id=2 intentionally missing
        ])

        with patch("kenkui.nlp.quotes.strip_scare_quotes", return_value=['"First," she said. "Second," he said. "Third," it said.']), \
             patch("kenkui.nlp.quotes.extract_quotes", return_value=[
                 MagicMock(id=0), MagicMock(id=1), MagicMock(id=2)
             ]), \
             patch("kenkui.nlp.annotator._build_alias_to_slug", return_value={}), \
             patch("kenkui.nlp.annotator.annotate_chapter", return_value="annotated"), \
             patch("kenkui.nlp.annotator._build_attribution_static_block", return_value=""), \
             patch("kenkui.nlp.annotator._build_attribution_dynamic_block", return_value=""), \
             patch("kenkui.nlp.llm.LLMClient.generate", return_value=wire), \
             caplog.at_level(logging.WARNING, logger="kenkui.nlp.providers.ollama"):
            adapter.attribute_chapter(chapter, roster)

        warning_messages = [
            r.message for r in caplog.records if r.levelno >= logging.WARNING
        ]
        assert warning_messages, (
            "A WARNING must fire when the LLM returns fewer quotes than were sent. "
            f"Sent 3, got 2. No warning recorded. Records: {[r.message for r in caplog.records]}"
        )
        assert any("quot" in m.lower() or "2" in m or "3" in m for m in warning_messages), (
            f"WARNING must reference quote counts. Got: {warning_messages}"
        )

    def test_no_warning_when_all_quotes_returned(self, caplog):
        """When LLM returns attribution for every quote, no WARNING fires."""
        import logging
        config = _make_config()
        adapter = OllamaAttributionAdapter(config)

        chapter = _make_chapter(['"Hello," Alice said.'])
        roster = _make_roster("Alice")

        wire = self._wire_result([
            AttributionItemWire(q=0, s="alice"),
        ])

        with patch("kenkui.nlp.quotes.strip_scare_quotes", return_value=['"Hello," Alice said.']), \
             patch("kenkui.nlp.quotes.extract_quotes", return_value=[MagicMock(id=0)]), \
             patch("kenkui.nlp.annotator._build_alias_to_slug", return_value={}), \
             patch("kenkui.nlp.annotator.annotate_chapter", return_value="annotated"), \
             patch("kenkui.nlp.annotator._build_attribution_static_block", return_value=""), \
             patch("kenkui.nlp.annotator._build_attribution_dynamic_block", return_value=""), \
             patch("kenkui.nlp.llm.LLMClient.generate", return_value=wire), \
             caplog.at_level(logging.WARNING, logger="kenkui.nlp.providers.ollama"):
            adapter.attribute_chapter(chapter, roster)

        gap_warnings = [
            r for r in caplog.records
            if r.levelno >= logging.WARNING and "quot" in r.message.lower()
        ]
        assert not gap_warnings, (
            f"No quote-gap WARNING expected when all quotes returned: {[r.message for r in gap_warnings]}"
        )


# ---------------------------------------------------------------------------
# Wire format slim-down + italic pre-assignment + conditional chunking
#
# Regressions for the 104625-char truncation and verbose output format.
#
# Goals:
#   1. AttributionItemWire uses short fields (q, s) — cuts output ~60%
#   2. italic quotes are pre-assigned NARRATOR — LLM only sees dialogue
#   3. Chunking is a safety valve, not the default
# ---------------------------------------------------------------------------


class TestAttributionWireSlimFormat:
    """AttributionItemWire must use short field names q/s to minimise output tokens."""

    def test_attribution_item_wire_has_q_field(self):
        """Wire item must use 'q' for quote_id — short name saves output tokens."""
        from kenkui.nlp.models import AttributionItemWire
        item = AttributionItemWire(q=42, s="alice")
        assert item.q == 42

    def test_attribution_item_wire_has_s_field(self):
        """Wire item must use 's' for speaker slug."""
        from kenkui.nlp.models import AttributionItemWire
        item = AttributionItemWire(q=0, s="mr_darcy")
        assert item.s == "mr_darcy"

    def test_attribution_result_wire_has_a_field(self):
        """Result wrapper must use 'a' for the attributions list."""
        from kenkui.nlp.models import AttributionItemWire, AttributionResultWire
        result = AttributionResultWire(a=[AttributionItemWire(q=1, s="alice")])
        assert len(result.a) == 1

    def test_attribution_wire_to_full_maps_q_to_quote_id(self):
        """attribution_wire_to_full must read 'q' as quote_id."""
        from kenkui.nlp.models import (
            AttributionItemWire,
            AttributionResultWire,
            attribution_wire_to_full,
        )
        wire = AttributionResultWire(a=[AttributionItemWire(q=7, s="bob")])
        full = attribution_wire_to_full(wire)
        assert full.attributions[0].quote_id == 7

    def test_attribution_wire_to_full_maps_s_to_speaker(self):
        """attribution_wire_to_full must read 's' as speaker."""
        from kenkui.nlp.models import (
            AttributionItemWire,
            AttributionResultWire,
            attribution_wire_to_full,
        )
        wire = AttributionResultWire(a=[AttributionItemWire(q=0, s="NARRATOR")])
        full = attribution_wire_to_full(wire)
        assert full.attributions[0].speaker == "NARRATOR"

    def test_slim_wire_has_no_confidence_field(self):
        """Wire item must NOT include confidence — extra fields waste output tokens."""
        from kenkui.nlp.models import AttributionItemWire
        item = AttributionItemWire(q=0, s="alice")
        assert not hasattr(item, "confidence"), (
            "AttributionItemWire must not include 'confidence' — it costs output tokens "
            "and is not used downstream."
        )


class TestItalicQuotePreAssignment:
    """Italic quotes must be pre-assigned NARRATOR; only dialogue quotes go to the LLM."""

    def _adapter(self):
        return OllamaAttributionAdapter(_make_config())

    def _italic_quote(self, id: int, para_index: int = 0):
        from kenkui.nlp.models import Quote
        return Quote(id=id, text="thought", para_index=para_index, char_offset=0, kind="italic")

    def _dialogue_quote(self, id: int, para_index: int = 0):
        from kenkui.nlp.models import Quote
        return Quote(id=id, text='"hello"', para_index=para_index, char_offset=0, kind="dialogue")

    def test_italic_quotes_are_not_sent_to_llm(self):
        """When all quotes are italic, no LLM call is made — all pre-assigned NARRATOR."""
        adapter = self._adapter()
        roster = _make_roster("Alice")
        chapter = _make_chapter(["Some \x02italic thought\x03 text."])

        italic = self._italic_quote(0)

        with patch("kenkui.nlp.quotes.strip_scare_quotes", return_value=["Some text."]), \
             patch("kenkui.nlp.quotes.extract_quotes", return_value=[italic]), \
             patch("kenkui.nlp.llm.LLMClient.generate") as mock_gen:
            result = adapter.attribute_chapter(chapter, roster)

        mock_gen.assert_not_called()
        assert len(result.attributions) == 1
        assert result.attributions[0].speaker == "NARRATOR"

    def test_italic_quotes_in_result_are_narrator(self):
        """Italic quotes must appear in the result as NARRATOR, not Unknown."""
        adapter = self._adapter()
        roster = _make_roster("Alice")
        chapter = _make_chapter(["text"])
        italic = self._italic_quote(0)

        with patch("kenkui.nlp.quotes.strip_scare_quotes", return_value=["text"]), \
             patch("kenkui.nlp.quotes.extract_quotes", return_value=[italic]), \
             patch("kenkui.nlp.llm.LLMClient.generate"):
            result = adapter.attribute_chapter(chapter, roster)

        assert result.attributions[0].speaker == "NARRATOR"

    def test_llm_called_only_for_dialogue_quotes(self):
        """When chapter has both italic and dialogue quotes, LLM receives only dialogue."""
        from kenkui.nlp.models import AttributionItemWire, AttributionResultWire
        adapter = self._adapter()
        roster = _make_roster("Alice")
        chapter = _make_chapter(['"Hello," said Alice. \x02thought\x03'])

        italic = self._italic_quote(0)
        dialogue = self._dialogue_quote(1)

        wire = AttributionResultWire(a=[AttributionItemWire(q=1, s="alice")])

        captured_quotes: list = []

        def capture_annotate(paragraphs, quotes, *args, **kwargs):
            captured_quotes.extend(quotes)
            return "annotated"

        with patch("kenkui.nlp.quotes.strip_scare_quotes", return_value=['"Hello," said Alice.']), \
             patch("kenkui.nlp.quotes.extract_quotes", return_value=[italic, dialogue]), \
             patch("kenkui.nlp.annotator._build_alias_to_slug", return_value={}), \
             patch("kenkui.nlp.annotator.annotate_chapter", side_effect=capture_annotate), \
             patch("kenkui.nlp.annotator._build_attribution_static_block", return_value=""), \
             patch("kenkui.nlp.annotator._build_attribution_dynamic_block", return_value=""), \
             patch("kenkui.nlp.llm.LLMClient.generate", return_value=wire):
            result = adapter.attribute_chapter(chapter, roster)

        # Only the dialogue quote should have been passed to annotate_chapter
        assert all(q.kind != "italic" for q in captured_quotes), (
            f"annotate_chapter received italic quotes: {captured_quotes}. "
            "Only dialogue quotes should be sent to the LLM."
        )
        # Both quotes must appear in the final result
        speakers = {a.quote_id: a.speaker for a in result.attributions}
        assert speakers[0] == "NARRATOR", f"Italic quote must be NARRATOR, got {speakers.get(0)}"
        assert speakers[1] == "alice", f"Dialogue quote must be attributed, got {speakers.get(1)}"
