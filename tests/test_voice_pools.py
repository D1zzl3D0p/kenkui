"""Tests for excluded_voices filtering in auto-assignment."""
from unittest.mock import MagicMock, patch
import pytest
from kenkui.cli.add import _auto_assign_character_voices
from kenkui.models import CharacterInfo


def _make_registry(male_names, female_names):
    def make_voice(name):
        v = MagicMock()
        v.name = name
        return v
    mock = MagicMock()
    mock.filter.side_effect = lambda gender=None, **_: (
        [make_voice(n) for n in male_names] if gender == "Male"
        else [make_voice(n) for n in female_names]
    )
    return mock


class TestExcludedVoicesFiltering:
    def test_excluded_voice_not_assigned(self):
        registry = _make_registry(
            male_names=["alba", "jean", "marius"],
            female_names=["cosette", "fantine"],
        )
        characters = [
            CharacterInfo(character_id="Rand", display_name="Rand",
                          gender_pronoun="he/him", mention_count=100),
            CharacterInfo(character_id="Mat", display_name="Mat",
                          gender_pronoun="he/him", mention_count=80),
        ]
        with patch("kenkui.voice_registry.get_registry", return_value=registry):
            result = _auto_assign_character_voices(
                characters,
                narrator_voice="cosette",
                excluded_voices=["alba"],
            )
        assert result.get("Rand") != "alba"
        assert result.get("Mat") != "alba"

    def test_fallback_when_all_excluded(self):
        """When all male voices are excluded, auto-assign falls back gracefully (no crash)."""
        registry = _make_registry(
            male_names=["alba"],
            female_names=["cosette"],
        )
        characters = [
            CharacterInfo(character_id="Hero", display_name="Hero",
                          gender_pronoun="he/him", mention_count=50),
        ]
        with patch("kenkui.voice_registry.get_registry", return_value=registry):
            result = _auto_assign_character_voices(
                characters,
                narrator_voice="cosette",
                excluded_voices=["alba"],
            )
        assert "Hero" in result  # something assigned, no crash

    def test_no_exclusions_unchanged(self):
        registry = _make_registry(
            male_names=["alba", "jean"],
            female_names=["cosette"],
        )
        characters = [
            CharacterInfo(character_id="Hero", display_name="Hero",
                          gender_pronoun="he/him", mention_count=50),
        ]
        with patch("kenkui.voice_registry.get_registry", return_value=registry):
            result = _auto_assign_character_voices(
                characters,
                narrator_voice="cosette",
                excluded_voices=[],
            )
        assert result["Hero"] in {"alba", "jean"}
