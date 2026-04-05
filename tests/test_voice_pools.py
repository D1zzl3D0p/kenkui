"""Tests for excluded_voices filtering in auto-assignment."""
from unittest.mock import MagicMock, patch
import pytest
from kenkui.cli.add import _auto_assign_character_voices, _auto_assign_voices
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


def _make_client(male_names, female_names):
    """Return a mock APIClient that responds to list_voices() and suggest_cast()."""
    client = MagicMock()

    def _list_voices(gender=None, source=None, **kwargs):
        if gender == "Male":
            return {"voices": [{"name": n, "display_label": n} for n in male_names], "total": len(male_names)}
        if gender == "Female":
            return {"voices": [{"name": n, "display_label": n} for n in female_names], "total": len(female_names)}
        all_voices = [{"name": n, "display_label": n} for n in male_names + female_names]
        return {"voices": all_voices, "total": len(all_voices)}

    client.list_voices.side_effect = _list_voices

    def _suggest_cast(roster, excluded_voices, default_voice):
        excluded = set(excluded_voices or [])
        available_male = [n for n in male_names if n not in excluded and n != default_voice] or \
                         [n for n in male_names if n not in excluded] or male_names
        available_female = [n for n in female_names if n not in excluded and n != default_voice] or \
                           [n for n in female_names if n not in excluded] or female_names
        speaker_voices = {}
        m_idx = f_idx = 0
        for char in roster:
            pronoun = (char.get("pronoun") or "").lower()
            if "he" in pronoun or "him" in pronoun:
                speaker_voices[char["name"]] = available_male[m_idx % len(available_male)] if available_male else default_voice
                m_idx += 1
            elif "she" in pronoun or "her" in pronoun:
                speaker_voices[char["name"]] = available_female[f_idx % len(available_female)] if available_female else default_voice
                f_idx += 1
            else:
                speaker_voices[char["name"]] = default_voice
        speaker_voices["NARRATOR"] = default_voice
        return {"speaker_voices": speaker_voices, "warnings": []}

    client.suggest_cast.side_effect = _suggest_cast
    return client


class TestExcludedVoicesFiltering:
    def test_excluded_voice_not_assigned(self):
        """Via _auto_assign_voices (API path): excluded voice not assigned."""
        characters = [
            CharacterInfo(character_id="Rand", display_name="Rand",
                          gender_pronoun="he/him", mention_count=100),
            CharacterInfo(character_id="Mat", display_name="Mat",
                          gender_pronoun="he/him", mention_count=80),
        ]
        client = _make_client(
            male_names=["alba", "jean", "marius"],
            female_names=["cosette", "fantine"],
        )
        result, _ = _auto_assign_voices(
            client,
            characters,
            narrator_voice="cosette",
            excluded_voices=["alba"],
        )
        assert result.get("Rand") != "alba"
        assert result.get("Mat") != "alba"

    def test_fallback_when_all_excluded(self):
        """When all male voices are excluded, auto-assign falls back gracefully (no crash)."""
        characters = [
            CharacterInfo(character_id="Hero", display_name="Hero",
                          gender_pronoun="he/him", mention_count=50),
        ]
        client = _make_client(
            male_names=["alba"],
            female_names=["cosette"],
        )
        result, _ = _auto_assign_voices(
            client,
            characters,
            narrator_voice="cosette",
            excluded_voices=["alba"],
        )
        assert "Hero" in result  # something assigned, no crash

    def test_no_exclusions_unchanged(self):
        """Without exclusions, assigned voice comes from the male pool."""
        characters = [
            CharacterInfo(character_id="Hero", display_name="Hero",
                          gender_pronoun="he/him", mention_count=50),
        ]
        client = _make_client(
            male_names=["alba", "jean"],
            female_names=["cosette"],
        )
        result, _ = _auto_assign_voices(
            client,
            characters,
            narrator_voice="cosette",
            excluded_voices=[],
        )
        assert result["Hero"] in {"alba", "jean"}


class TestLegacyAutoAssignLocalFallback:
    """Verify the local _auto_assign_character_voices still works (used as fallback)."""

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
