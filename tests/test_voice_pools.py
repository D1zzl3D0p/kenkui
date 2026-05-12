"""Tests for excluded_voices filtering in suggest_cast service."""
from unittest.mock import MagicMock, patch
import pytest
from kenkui.models import CharacterInfo
from kenkui.services.voice_service import suggest_cast


def _make_voices(male_names, female_names):
    voices = []
    for n in male_names:
        v = MagicMock()
        v.name = n
        v.gender = "Male"
        voices.append(v)
    for n in female_names:
        v = MagicMock()
        v.name = n
        v.gender = "Female"
        voices.append(v)
    return voices


class TestExcludedVoicesFiltering:
    def test_excluded_voice_not_assigned(self):
        characters = [
            CharacterInfo(character_id="Rand", display_name="Rand",
                          gender_pronoun="he/him", mention_count=100),
            CharacterInfo(character_id="Mat", display_name="Mat",
                          gender_pronoun="he/him", mention_count=80),
        ]
        voices = _make_voices(["alba", "jean", "marius"], ["cosette", "fantine"])
        with patch("kenkui.services.voice_service.list_voices", return_value=voices):
            result = suggest_cast(
                roster=characters,
                excluded_voices=["alba"],
                default_voice="cosette",
            )
        assert result.speaker_voices.get("Rand") != "alba"
        assert result.speaker_voices.get("Mat") != "alba"

    def test_fallback_when_all_excluded(self):
        """When all voices are excluded, suggest_cast falls back gracefully."""
        characters = [
            CharacterInfo(character_id="Hero", display_name="Hero",
                          gender_pronoun="he/him", mention_count=50),
        ]
        voices = _make_voices(["alba"], ["cosette"])
        with patch("kenkui.services.voice_service.list_voices", return_value=voices):
            result = suggest_cast(
                roster=characters,
                excluded_voices=["alba"],
                default_voice="cosette",
            )
        assert "Hero" in result.speaker_voices

    def test_no_exclusions_assigns_from_pool(self):
        """Without exclusions, assigned voice comes from the male pool."""
        characters = [
            CharacterInfo(character_id="Hero", display_name="Hero",
                          gender_pronoun="he/him", mention_count=50),
        ]
        voices = _make_voices(["alba", "jean"], ["cosette"])
        with patch("kenkui.services.voice_service.list_voices", return_value=voices):
            result = suggest_cast(
                roster=characters,
                excluded_voices=[],
                default_voice="cosette",
            )
        assert result.speaker_voices["Hero"] in {"alba", "jean"}
