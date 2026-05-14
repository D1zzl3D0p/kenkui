"""Tests for BookNLP extraction and attribution adapters.

All BookNLP I/O is mocked — the real ``booknlp`` package is never required.
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from kenkui.models import Chapter
from kenkui.nlp.booknlp_roster import BookNLPRosterData
from kenkui.nlp.models import AttributionItem, AttributionResult, CharacterRecord, CharacterRoster
from kenkui.nlp.providers.booknlp import (
    BookNLPAttributionAdapter,
    BookNLPExtractionAdapter,
    _name_to_slug,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_config(extraction_model: str = "small", attribution_model: str = "small") -> MagicMock:
    cfg = MagicMock()
    cfg.extraction_model = extraction_model
    cfg.attribution_model = attribution_model
    return cfg


def _make_chapter(paragraphs: list[str], title: str = "Chapter 1", index: int = 0) -> Chapter:
    return Chapter(index=index, title=title, paragraphs=paragraphs)


def _make_roster(characters: list[CharacterRecord] | None = None) -> CharacterRoster:
    return CharacterRoster(characters=characters or [])


def _make_record(
    slug: str,
    canonical_name: str,
    aliases: list[str] | None = None,
    gender: str = "",
) -> CharacterRecord:
    return CharacterRecord(
        slug=slug,
        canonical_name=canonical_name,
        aliases=aliases or [],
        gender=gender,
    )


def _make_booknlp_data(
    quotes: list[dict] | None = None,
    characters: list[dict] | None = None,
) -> dict:
    return {
        "quotes": quotes or [],
        "characters": characters or [],
    }


# ---------------------------------------------------------------------------
# ExtractionAdapter tests
# ---------------------------------------------------------------------------


class TestBookNLPExtractionAdapter:

    def test_extraction_adapter_returns_roster_from_booknlp(self):
        """build_roster returns the roster inside BookNLPRosterData."""
        record = _make_record("alice", "Alice")
        roster = _make_roster([record])
        mock_result = BookNLPRosterData(roster=roster, common_phrases=[])

        config = _make_config()
        adapter = BookNLPExtractionAdapter(config)
        chapters = [_make_chapter(["Alice said hello."])]

        with patch(
            "kenkui.nlp.providers.booknlp.build_roster_from_booknlp",
            return_value=mock_result,
        ):
            result = adapter.build_roster(chapters)

        assert len(result.characters) == 1
        assert result.characters[0].slug == "alice"

    def test_extraction_adapter_returns_empty_roster_when_booknlp_none(self):
        """Returns CharacterRoster(characters=[]) when build_roster_from_booknlp returns None."""
        config = _make_config()
        adapter = BookNLPExtractionAdapter(config)
        chapters = [_make_chapter(["Some text."])]

        with patch(
            "kenkui.nlp.providers.booknlp.build_roster_from_booknlp",
            return_value=None,
        ):
            result = adapter.build_roster(chapters)

        assert result == CharacterRoster(characters=[])

    def test_extraction_adapter_concatenates_paragraphs(self):
        """All paragraph text from all chapters is concatenated before passing to build_roster_from_booknlp."""
        config = _make_config()
        adapter = BookNLPExtractionAdapter(config)
        chapters = [
            _make_chapter(["Para A1.", "Para A2."], title="Ch1"),
            _make_chapter(["Para B1."], title="Ch2"),
        ]
        expected_text = "Para A1.\n\nPara A2.\n\nPara B1."

        captured: list[str] = []

        def _fake_build(text: str, model_size: str) -> None:
            captured.append(text)
            return None

        with patch(
            "kenkui.nlp.providers.booknlp.build_roster_from_booknlp",
            side_effect=_fake_build,
        ):
            adapter.build_roster(chapters)

        assert len(captured) == 1
        assert captured[0] == expected_text

    def test_extraction_adapter_calls_progress_callback(self):
        """progress_callback is invoked before the BookNLP call."""
        config = _make_config()
        adapter = BookNLPExtractionAdapter(config)
        chapters = [_make_chapter(["Text."])]
        callback_calls: list[str] = []

        with patch(
            "kenkui.nlp.providers.booknlp.build_roster_from_booknlp",
            return_value=None,
        ):
            adapter.build_roster(chapters, progress_callback=callback_calls.append)

        assert "Extracting characters via BookNLP" in callback_calls

    def test_extraction_uses_extraction_model_from_config(self):
        """model_size=config.extraction_model is passed to build_roster_from_booknlp."""
        config = _make_config(extraction_model="big")
        adapter = BookNLPExtractionAdapter(config)
        chapters = [_make_chapter(["Text."])]
        captured_kwargs: list[dict] = []

        def _fake_build(text: str, model_size: str) -> None:
            captured_kwargs.append({"model_size": model_size})
            return None

        with patch(
            "kenkui.nlp.providers.booknlp.build_roster_from_booknlp",
            side_effect=_fake_build,
        ):
            adapter.build_roster(chapters)

        assert captured_kwargs[0]["model_size"] == "big"


# ---------------------------------------------------------------------------
# AttributionAdapter tests
# ---------------------------------------------------------------------------


class TestBookNLPAttributionAdapter:

    def test_attribution_adapter_returns_attribution_result(self):
        """Returns AttributionResult with items for each quote."""
        config = _make_config()
        adapter = BookNLPAttributionAdapter(config)
        chapter = _make_chapter(["\"Hello,\" said Alice."])
        roster = _make_roster([_make_record("alice", "Alice")])

        booknlp_data = _make_booknlp_data(
            quotes=[{"quote_start": 0, "quote_end": 6, "char_id": 1}],
            characters=[{"id": 1, "mentions": {"proper": [{"n": "Alice"}]}}],
        )

        with patch(
            "kenkui.nlp.providers.booknlp._run_booknlp",
            return_value=booknlp_data,
        ):
            result = adapter.attribute_chapter(chapter, roster)

        assert isinstance(result, AttributionResult)
        assert len(result.attributions) == 1
        assert result.attributions[0].quote_id == 1
        assert result.attributions[0].speaker == "alice"
        assert result.attributions[0].confidence == 5

    def test_attribution_adapter_maps_char_id_to_slug(self):
        """char_id in quotes is correctly resolved to the character's slug via roster lookup."""
        config = _make_config()
        adapter = BookNLPAttributionAdapter(config)
        chapter = _make_chapter(["\"Go away,\" said Bob. \"Fine,\" said Alice."])
        roster = _make_roster([
            _make_record("alice", "Alice"),
            _make_record("bob", "Bob"),
        ])

        booknlp_data = _make_booknlp_data(
            quotes=[
                {"quote_start": 0, "quote_end": 9, "char_id": 2},
                {"quote_start": 20, "quote_end": 26, "char_id": 1},
            ],
            characters=[
                {"id": 1, "mentions": {"proper": [{"n": "Alice"}]}},
                {"id": 2, "mentions": {"proper": [{"n": "Bob"}]}},
            ],
        )

        with patch(
            "kenkui.nlp.providers.booknlp._run_booknlp",
            return_value=booknlp_data,
        ):
            result = adapter.attribute_chapter(chapter, roster)

        assert result.attributions[0].speaker == "bob"
        assert result.attributions[1].speaker == "alice"

    def test_attribution_adapter_unknown_speaker_for_neg_one(self):
        """char_id == -1 maps to 'Unknown' speaker."""
        config = _make_config()
        adapter = BookNLPAttributionAdapter(config)
        chapter = _make_chapter(["\"Who said that?\""])
        roster = _make_roster()

        booknlp_data = _make_booknlp_data(
            quotes=[{"quote_start": 0, "quote_end": 15, "char_id": -1}],
            characters=[],
        )

        with patch(
            "kenkui.nlp.providers.booknlp._run_booknlp",
            return_value=booknlp_data,
        ):
            result = adapter.attribute_chapter(chapter, roster)

        assert result.attributions[0].speaker == "Unknown"

    def test_attribution_adapter_returns_empty_on_import_error(self):
        """Returns AttributionResult(attributions=[]) when ImportError is raised."""
        config = _make_config()
        adapter = BookNLPAttributionAdapter(config)
        chapter = _make_chapter(["Some text."])
        roster = _make_roster()

        with patch(
            "kenkui.nlp.providers.booknlp._run_booknlp",
            side_effect=ImportError("booknlp not installed"),
        ):
            result = adapter.attribute_chapter(chapter, roster)

        assert result == AttributionResult(attributions=[])

    def test_attribution_adapter_returns_empty_on_exception(self):
        """Returns AttributionResult(attributions=[]) when any other exception is raised."""
        config = _make_config()
        adapter = BookNLPAttributionAdapter(config)
        chapter = _make_chapter(["Some text."])
        roster = _make_roster()

        with patch(
            "kenkui.nlp.providers.booknlp._run_booknlp",
            side_effect=RuntimeError("BookNLP failed"),
        ):
            result = adapter.attribute_chapter(chapter, roster)

        assert result == AttributionResult(attributions=[])

    def test_attribution_adapter_calls_progress_callback(self):
        """progress_callback is invoked before the BookNLP call."""
        config = _make_config()
        adapter = BookNLPAttributionAdapter(config)
        chapter = _make_chapter(["Text."])
        roster = _make_roster()
        callback_calls: list[str] = []

        booknlp_data = _make_booknlp_data()

        with patch(
            "kenkui.nlp.providers.booknlp._run_booknlp",
            return_value=booknlp_data,
        ):
            adapter.attribute_chapter(chapter, roster, progress_callback=callback_calls.append)

        assert "Attributing chapter via BookNLP" in callback_calls


# ---------------------------------------------------------------------------
# _name_to_slug tests
# ---------------------------------------------------------------------------


class TestNameToSlug:

    def test_name_to_slug_canonical_match(self):
        """Finds character by canonical_name, case-insensitively."""
        roster = _make_roster([
            _make_record("elizabeth_bennet", "Elizabeth Bennet"),
        ])
        assert _name_to_slug("elizabeth bennet", roster) == "elizabeth_bennet"
        assert _name_to_slug("ELIZABETH BENNET", roster) == "elizabeth_bennet"
        assert _name_to_slug("Elizabeth Bennet", roster) == "elizabeth_bennet"

    def test_name_to_slug_alias_match(self):
        """Finds character by alias, case-insensitively."""
        roster = _make_roster([
            _make_record("elizabeth_bennet", "Elizabeth Bennet", aliases=["Lizzy", "Eliza"]),
        ])
        assert _name_to_slug("lizzy", roster) == "elizabeth_bennet"
        assert _name_to_slug("LIZZY", roster) == "elizabeth_bennet"
        assert _name_to_slug("Eliza", roster) == "elizabeth_bennet"

    def test_name_to_slug_fallback_slugify(self):
        """Falls back to slugify(name) when no roster match is found."""
        roster = _make_roster([
            _make_record("alice", "Alice"),
        ])
        result = _name_to_slug("Mr. Darcy", roster)
        # slugify("Mr. Darcy") == "mr_darcy"
        assert result == "mr_darcy"
