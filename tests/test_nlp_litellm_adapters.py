"""Tests for LiteLLMExtractionAdapter and LiteLLMAttributionAdapter.

litellm and instructor are NOT required — they are mocked via sys.modules so
these tests run in environments where neither package is installed.
"""
from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, call, patch

import pytest

# ---------------------------------------------------------------------------
# Pre-inject mock modules BEFORE any import of the adapters so that the lazy
# ``import litellm`` / ``import instructor`` inside attribute_chapter resolve
# to our mocks without requiring the real packages.
# ---------------------------------------------------------------------------
_litellm_mock = MagicMock()
_instructor_mock = MagicMock()
sys.modules.setdefault("litellm", _litellm_mock)
sys.modules.setdefault("instructor", _instructor_mock)

# ---------------------------------------------------------------------------
# Project imports (after mocks are injected)
# ---------------------------------------------------------------------------
from kenkui.models import Chapter, CharacterInfo, FastScanResult  # noqa: E402
from kenkui.nlp.models import (  # noqa: E402
    AttributionItem,
    AttributionItemWire,
    AttributionResult,
    AttributionResultWire,
    CharacterRecord,
    CharacterRoster,
)
from kenkui.nlp.providers.litellm import (  # noqa: E402
    LiteLLMAttributionAdapter,
    LiteLLMExtractionAdapter,
)
from kenkui.nlp_config import NLPConfig  # noqa: E402


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

BOOK_PATH = Path("/tmp/test_litellm_book.epub")


def _make_config(**kwargs) -> NLPConfig:
    defaults = {
        "extraction_model": "gpt-4o-mini",
        "attribution_model": "gpt-4o-mini",
        "ollama_url": "http://localhost:11434",
    }
    defaults.update(kwargs)
    return NLPConfig(**defaults)


def _make_roster(*names: str) -> CharacterRoster:
    chars = [
        CharacterRecord(slug=n.lower().replace(" ", "_"), canonical_name=n)
        for n in names
    ]
    return CharacterRoster(characters=chars)


def _make_chapter(paragraphs: list[str] | None = None):
    ch = MagicMock(spec=Chapter)
    ch.paragraphs = paragraphs or ["Hello world."]
    return ch


def _make_fast_scan_result(
    canonical_name: str,
    slug: str,
    aliases: list[str] | None = None,
    mention_count: int = 10,
) -> FastScanResult:
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
        mention_count=mention_count,
        quote_count=3,
    )
    return FastScanResult(roster=mock_roster, characters=[char_info], book_hash="aabbcc")


def _make_wire_result(items: list[AttributionItemWire]) -> AttributionResultWire:
    return AttributionResultWire(attributions=items)


# ---------------------------------------------------------------------------
# LiteLLMExtractionAdapter tests
# ---------------------------------------------------------------------------


class TestLiteLLMExtractionAdapter:
    def test_extraction_adapter_delegates_to_run_fast_scan(self):
        """Adapter should call run_fast_scan and convert the result to a CharacterRoster."""
        config = _make_config()
        adapter = LiteLLMExtractionAdapter(config)
        fast_result = _make_fast_scan_result("Alice", "alice")

        with patch(
            "kenkui.nlp.providers.litellm.run_fast_scan", return_value=fast_result
        ) as mock_scan:
            result = adapter.build_roster([_make_chapter()], book_path=BOOK_PATH)

        mock_scan.assert_called_once()
        assert len(result.characters) == 1
        assert result.characters[0].canonical_name == "Alice"
        assert result.characters[0].slug == "alice"

    def test_extraction_adapter_raises_if_no_book_path(self):
        """build_roster must raise ValueError when book_path is None."""
        config = _make_config()
        adapter = LiteLLMExtractionAdapter(config)

        with pytest.raises(ValueError, match="book_path"):
            adapter.build_roster([_make_chapter()])

    def test_extraction_adapter_calls_progress_callback(self):
        """progress_callback must be forwarded to run_fast_scan."""
        config = _make_config()
        adapter = LiteLLMExtractionAdapter(config)
        cb = MagicMock()
        fast_result = _make_fast_scan_result("Bob", "bob")

        with patch(
            "kenkui.nlp.providers.litellm.run_fast_scan", return_value=fast_result
        ) as mock_scan:
            adapter.build_roster([], book_path=BOOK_PATH, progress_callback=cb)

        _, kwargs = mock_scan.call_args
        assert kwargs.get("progress_callback") is cb

    def test_extraction_uses_extraction_model_from_config(self):
        """The extraction_model from NLPConfig must be passed to run_fast_scan."""
        config = _make_config(extraction_model="claude-3-haiku")
        adapter = LiteLLMExtractionAdapter(config)
        fast_result = _make_fast_scan_result("Charlie", "charlie")

        with patch(
            "kenkui.nlp.providers.litellm.run_fast_scan", return_value=fast_result
        ) as mock_scan:
            adapter.build_roster([_make_chapter()], book_path=BOOK_PATH)

        # extraction_model is passed as a positional arg
        call_args = mock_scan.call_args
        assert "claude-3-haiku" in call_args.args

    def test_extraction_populates_mention_count(self):
        """mention_count from CharacterInfo must end up on the CharacterRecord."""
        config = _make_config()
        adapter = LiteLLMExtractionAdapter(config)
        fast_result = _make_fast_scan_result("Diana", "diana", mention_count=55)

        with patch(
            "kenkui.nlp.providers.litellm.run_fast_scan", return_value=fast_result
        ):
            result = adapter.build_roster([_make_chapter()], book_path=BOOK_PATH)

        assert result.characters[0].mention_count == 55

    def test_extraction_returns_empty_roster_when_no_characters(self):
        """An empty FastScanResult roster should produce an empty CharacterRoster."""
        config = _make_config()
        adapter = LiteLLMExtractionAdapter(config)

        mock_roster = MagicMock()
        mock_roster.characters = []
        fast_result = FastScanResult(roster=mock_roster, characters=[], book_hash="xyz")

        with patch(
            "kenkui.nlp.providers.litellm.run_fast_scan", return_value=fast_result
        ):
            result = adapter.build_roster([], book_path=BOOK_PATH)

        assert result.characters == []


# ---------------------------------------------------------------------------
# LiteLLMAttributionAdapter tests
# ---------------------------------------------------------------------------


class TestLiteLLMAttributionAdapter:
    """Tests for LiteLLMAttributionAdapter.attribute_chapter.

    litellm and instructor are pre-mocked in sys.modules; individual tests
    configure the instructor mock's return value as needed.
    """

    def _setup_instructor_mock(self, wire_result: AttributionResultWire):
        """Wire up the instructor mock so client.chat.completions.create returns wire_result."""
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = wire_result
        _instructor_mock.from_litellm.return_value = mock_client
        return mock_client

    def test_attribution_adapter_calls_litellm_completion(self):
        """LiteLLM completion and instructor.from_litellm must be called."""
        config = _make_config(attribution_model="gpt-4o")
        adapter = LiteLLMAttributionAdapter(config)
        roster = _make_roster("Alice")
        chapter = _make_chapter(['"Hello," said Alice.'])

        wire = _make_wire_result([AttributionItemWire(quote_id=0, speaker="alice", confidence=5)])
        mock_client = self._setup_instructor_mock(wire)

        with patch("kenkui.nlp.quotes.strip_scare_quotes", return_value=['"Hello," said Alice.']), \
             patch("kenkui.nlp.quotes.extract_quotes", return_value=[MagicMock(id=0)]), \
             patch("kenkui.nlp.annotator._build_alias_to_slug", return_value={}), \
             patch("kenkui.nlp.annotator.annotate_chapter", return_value="[QUOTE:0] text"), \
             patch("kenkui.nlp.annotator._build_attribution_static_block", return_value="static"), \
             patch("kenkui.nlp.annotator._build_attribution_dynamic_block", return_value="dynamic"):
            adapter.attribute_chapter(chapter, roster)

        _instructor_mock.from_litellm.assert_called()
        mock_client.chat.completions.create.assert_called_once()
        call_kwargs = mock_client.chat.completions.create.call_args.kwargs
        assert call_kwargs["model"] == "gpt-4o"

    def test_attribution_adapter_returns_attribution_result(self):
        """attribute_chapter must return a fully-typed AttributionResult."""
        config = _make_config()
        adapter = LiteLLMAttributionAdapter(config)
        roster = _make_roster("Alice")
        chapter = _make_chapter(['"Hi," said Alice.'])

        wire = _make_wire_result([AttributionItemWire(quote_id=0, speaker="alice", confidence=4)])
        self._setup_instructor_mock(wire)

        with patch("kenkui.nlp.quotes.strip_scare_quotes", return_value=['"Hi," said Alice.']), \
             patch("kenkui.nlp.quotes.extract_quotes", return_value=[MagicMock(id=0)]), \
             patch("kenkui.nlp.annotator._build_alias_to_slug", return_value={}), \
             patch("kenkui.nlp.annotator.annotate_chapter", return_value="annotated"), \
             patch("kenkui.nlp.annotator._build_attribution_static_block", return_value=""), \
             patch("kenkui.nlp.annotator._build_attribution_dynamic_block", return_value=""):
            result = adapter.attribute_chapter(chapter, roster)

        assert isinstance(result, AttributionResult)
        assert len(result.attributions) == 1
        assert result.attributions[0].speaker == "alice"

    def test_attribution_adapter_returns_empty_on_no_quotes(self):
        """When extract_quotes returns [], attribute_chapter should return AttributionResult(attributions=[])."""
        config = _make_config()
        adapter = LiteLLMAttributionAdapter(config)
        roster = _make_roster("Alice")
        chapter = _make_chapter(["No dialogue here."])

        with patch("kenkui.nlp.quotes.strip_scare_quotes", return_value=["No dialogue here."]), \
             patch("kenkui.nlp.quotes.extract_quotes", return_value=[]):
            result = adapter.attribute_chapter(chapter, roster)

        assert isinstance(result, AttributionResult)
        assert result.attributions == []

    def test_attribution_adapter_passes_api_base_when_set(self):
        """api_base from config must be forwarded to the instructor client call."""
        config = _make_config(litellm_api_base="https://my.proxy/v1")
        adapter = LiteLLMAttributionAdapter(config)
        roster = _make_roster("Bob")
        chapter = _make_chapter(['"Yo," said Bob.'])

        wire = _make_wire_result([AttributionItemWire(quote_id=0, speaker="bob", confidence=3)])
        mock_client = self._setup_instructor_mock(wire)

        with patch("kenkui.nlp.quotes.strip_scare_quotes", return_value=['"Yo," said Bob.']), \
             patch("kenkui.nlp.quotes.extract_quotes", return_value=[MagicMock(id=0)]), \
             patch("kenkui.nlp.annotator._build_alias_to_slug", return_value={}), \
             patch("kenkui.nlp.annotator.annotate_chapter", return_value="annotated"), \
             patch("kenkui.nlp.annotator._build_attribution_static_block", return_value=""), \
             patch("kenkui.nlp.annotator._build_attribution_dynamic_block", return_value=""):
            adapter.attribute_chapter(chapter, roster)

        call_kwargs = mock_client.chat.completions.create.call_args.kwargs
        assert call_kwargs.get("api_base") == "https://my.proxy/v1"

    def test_attribution_adapter_passes_api_key_when_set(self):
        """api_key from config must be forwarded to the instructor client call."""
        config = _make_config(LITELLM_API_KEY="sk-secret")
        adapter = LiteLLMAttributionAdapter(config)
        roster = _make_roster("Eve")
        chapter = _make_chapter(['"Hello," said Eve.'])

        wire = _make_wire_result([AttributionItemWire(quote_id=0, speaker="eve", confidence=5)])
        mock_client = self._setup_instructor_mock(wire)

        with patch("kenkui.nlp.quotes.strip_scare_quotes", return_value=['"Hello," said Eve.']), \
             patch("kenkui.nlp.quotes.extract_quotes", return_value=[MagicMock(id=0)]), \
             patch("kenkui.nlp.annotator._build_alias_to_slug", return_value={}), \
             patch("kenkui.nlp.annotator.annotate_chapter", return_value="annotated"), \
             patch("kenkui.nlp.annotator._build_attribution_static_block", return_value=""), \
             patch("kenkui.nlp.annotator._build_attribution_dynamic_block", return_value=""):
            adapter.attribute_chapter(chapter, roster)

        call_kwargs = mock_client.chat.completions.create.call_args.kwargs
        assert call_kwargs.get("api_key") == "sk-secret"

    def test_attribution_adapter_omits_api_base_when_none(self):
        """api_base must NOT appear in the call when litellm_api_base is None."""
        config = _make_config()  # no litellm_api_base
        adapter = LiteLLMAttributionAdapter(config)
        roster = _make_roster("Frank")
        chapter = _make_chapter(['"Sup," said Frank.'])

        wire = _make_wire_result([AttributionItemWire(quote_id=0, speaker="frank", confidence=2)])
        mock_client = self._setup_instructor_mock(wire)

        with patch("kenkui.nlp.quotes.strip_scare_quotes", return_value=['"Sup," said Frank.']), \
             patch("kenkui.nlp.quotes.extract_quotes", return_value=[MagicMock(id=0)]), \
             patch("kenkui.nlp.annotator._build_alias_to_slug", return_value={}), \
             patch("kenkui.nlp.annotator.annotate_chapter", return_value="annotated"), \
             patch("kenkui.nlp.annotator._build_attribution_static_block", return_value=""), \
             patch("kenkui.nlp.annotator._build_attribution_dynamic_block", return_value=""):
            adapter.attribute_chapter(chapter, roster)

        call_kwargs = mock_client.chat.completions.create.call_args.kwargs
        assert "api_base" not in call_kwargs

    def test_attribution_adapter_calls_progress_callback(self):
        """progress_callback must be called at least once during attribution."""
        config = _make_config()
        adapter = LiteLLMAttributionAdapter(config)
        roster = _make_roster()
        chapter = _make_chapter(["No dialogue."])
        cb = MagicMock()

        with patch("kenkui.nlp.quotes.strip_scare_quotes", return_value=["No dialogue."]), \
             patch("kenkui.nlp.quotes.extract_quotes", return_value=[]):
            adapter.attribute_chapter(chapter, roster, progress_callback=cb)

        cb.assert_called_once()

    def test_attribution_adapter_messages_contain_prompt(self):
        """The messages list sent to LiteLLM must contain the built prompt."""
        config = _make_config()
        adapter = LiteLLMAttributionAdapter(config)
        roster = _make_roster("Grace")
        chapter = _make_chapter(['"Hey," said Grace.'])

        wire = _make_wire_result([AttributionItemWire(quote_id=0, speaker="grace", confidence=5)])
        mock_client = self._setup_instructor_mock(wire)

        with patch("kenkui.nlp.quotes.strip_scare_quotes", return_value=['"Hey," said Grace.']), \
             patch("kenkui.nlp.quotes.extract_quotes", return_value=[MagicMock(id=0)]), \
             patch("kenkui.nlp.annotator._build_alias_to_slug", return_value={}), \
             patch("kenkui.nlp.annotator.annotate_chapter", return_value="annotated_text"), \
             patch("kenkui.nlp.annotator._build_attribution_static_block", return_value="STATIC"), \
             patch("kenkui.nlp.annotator._build_attribution_dynamic_block", return_value="DYNAMIC"):
            adapter.attribute_chapter(chapter, roster)

        call_kwargs = mock_client.chat.completions.create.call_args.kwargs
        messages = call_kwargs["messages"]
        assert len(messages) == 1
        assert messages[0]["role"] == "user"
        assert "STATIC" in messages[0]["content"]
        assert "DYNAMIC" in messages[0]["content"]

    def test_attribution_adapter_uses_attribution_result_wire_schema(self):
        """response_model passed to LiteLLM must be AttributionResultWire."""
        config = _make_config()
        adapter = LiteLLMAttributionAdapter(config)
        roster = _make_roster("Helen")
        chapter = _make_chapter(['"Whoa," said Helen.'])

        wire = _make_wire_result([AttributionItemWire(quote_id=0, speaker="helen", confidence=5)])
        mock_client = self._setup_instructor_mock(wire)

        with patch("kenkui.nlp.quotes.strip_scare_quotes", return_value=['"Whoa," said Helen.']), \
             patch("kenkui.nlp.quotes.extract_quotes", return_value=[MagicMock(id=0)]), \
             patch("kenkui.nlp.annotator._build_alias_to_slug", return_value={}), \
             patch("kenkui.nlp.annotator.annotate_chapter", return_value="annotated"), \
             patch("kenkui.nlp.annotator._build_attribution_static_block", return_value=""), \
             patch("kenkui.nlp.annotator._build_attribution_dynamic_block", return_value=""):
            adapter.attribute_chapter(chapter, roster)

        call_kwargs = mock_client.chat.completions.create.call_args.kwargs
        assert call_kwargs["response_model"] is AttributionResultWire
