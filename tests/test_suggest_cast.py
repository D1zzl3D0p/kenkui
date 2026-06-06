"""Tests for voice_service.suggest_cast."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

from kenkui.services.voice_service import SuggestCastResult, suggest_cast


def _make_roster(names_pronouns):
    from kenkui.models import CharacterInfo
    return [CharacterInfo(character_id=n, display_name=n, gender_pronoun=p,
                          quote_count=1, mention_count=1)
            for n, p in names_pronouns]


def _voice(name: str, gender: str, source: str = "builtin") -> MagicMock:
    """Create a VoiceInfo-like mock with name, gender, and source set."""
    m = MagicMock()
    m.name = name
    m.gender = gender
    m.source = source
    return m


def test_suggest_cast_returns_suggest_cast_result():
    voices = [_voice("alice", "Female"), _voice("bob", "Male")]
    roster = _make_roster([("Alice", "she/her"), ("Bob", "he/him")])
    with patch("kenkui.services.voice_service.list_voices", return_value=voices):
        result = suggest_cast(roster=roster, excluded_voices=[], default_voice="narrator")
    assert isinstance(result, SuggestCastResult)
    assert "Alice" in result.speaker_voices
    assert "Bob" in result.speaker_voices


def test_suggest_cast_excludes_excluded_voices():
    voices = [_voice("alice", "Female"), _voice("bob", "Male"), _voice("carol", "Female")]
    roster = _make_roster([("Alice", "she/her")])
    with patch("kenkui.services.voice_service.list_voices", return_value=voices):
        result = suggest_cast(roster=roster, excluded_voices=["alice"], default_voice="narrator")
    assert result.speaker_voices.get("Alice") != "alice"


def test_suggest_cast_empty_pool_falls_back_to_default_voice():
    voices = []
    roster = _make_roster([("Alice", "she/her")])
    with patch("kenkui.services.voice_service.list_voices", return_value=voices):
        result = suggest_cast(roster=roster, excluded_voices=[], default_voice="narrator")
    assert result.speaker_voices.get("Alice") == "narrator"
    assert len(result.warnings) > 0


def test_suggest_cast_resolves_chapter_conflicts():
    """Two chars sharing a chapter should get different voices if pool is large enough."""
    voices = [_voice("v1", "Female"), _voice("v2", "Female")]
    roster = _make_roster([("Alice", "she/her"), ("Eve", "she/her")])
    chapters = [MagicMock(paragraphs=[
        MagicMock(speaker="Alice", is_spoken=True),
        MagicMock(speaker="Eve", is_spoken=True),
    ])]
    with patch("kenkui.services.voice_service.list_voices", return_value=voices):
        result = suggest_cast(roster=roster, excluded_voices=[], default_voice="narrator",
                              chapters=chapters)
    alice_v = result.speaker_voices["Alice"]
    eve_v = result.speaker_voices["Eve"]
    assert alice_v != eve_v


def test_suggest_cast_excludes_uncompiled_voices():
    """Uncompiled (.wav) voices must never appear in the assigned pool."""
    voices = [
        _voice("wav_voice", "Female", source="uncompiled"),
        _voice("builtin_voice", "Female", source="builtin"),
    ]
    roster = _make_roster([("Alice", "she/her")])
    with patch("kenkui.services.voice_service.list_voices", return_value=voices):
        result = suggest_cast(roster=roster, excluded_voices=[], default_voice="narrator")
    assert result.speaker_voices.get("Alice") != "wav_voice"
