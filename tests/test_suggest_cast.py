"""Tests for voice_service.suggest_cast."""
from __future__ import annotations
from unittest.mock import patch, MagicMock
from kenkui.services.voice_service import suggest_cast, SuggestCastResult


def _make_roster(names_pronouns):
    from kenkui.models import CharacterInfo
    return [CharacterInfo(character_id=n, display_name=n, gender_pronoun=p,
                          quote_count=1, mention_count=1)
            for n, p in names_pronouns]


def test_suggest_cast_returns_suggest_cast_result():
    voices = [MagicMock(name="alice", gender="Female"),
              MagicMock(name="bob", gender="Male")]
    roster = _make_roster([("Alice", "she/her"), ("Bob", "he/him")])
    with patch("kenkui.services.voice_service.list_voices", return_value=voices):
        result = suggest_cast(roster=roster, excluded_voices=[], default_voice="narrator")
    assert isinstance(result, SuggestCastResult)
    assert "Alice" in result.speaker_voices
    assert "Bob" in result.speaker_voices


def test_suggest_cast_excludes_excluded_voices():
    voices = [MagicMock(name="alice", gender="Female"),
              MagicMock(name="bob", gender="Male"),
              MagicMock(name="carol", gender="Female")]
    roster = _make_roster([("Alice", "she/her")])
    with patch("kenkui.services.voice_service.list_voices", return_value=voices):
        result = suggest_cast(roster=roster, excluded_voices=["alice"], default_voice="narrator")
    assert result.speaker_voices.get("Alice") != "alice"


def test_suggest_cast_empty_pool_falls_back_to_default_voice():
    voices = []  # empty pool forces fallback to default_voice
    roster = _make_roster([("Alice", "she/her")])
    with patch("kenkui.services.voice_service.list_voices", return_value=voices):
        result = suggest_cast(roster=roster, excluded_voices=[], default_voice="narrator")
    assert result.speaker_voices.get("Alice") == "narrator"
    assert len(result.warnings) > 0


def test_suggest_cast_resolves_chapter_conflicts():
    """Two chars sharing a chapter should get different voices if pool is large enough."""
    voices = [
        MagicMock(name="v1", gender="Female"),
        MagicMock(name="v2", gender="Female"),
    ]
    roster = _make_roster([("Alice", "she/her"), ("Eve", "she/her")])
    # Provide chapters with both characters speaking
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


def test_suggest_cast_api_route():
    from unittest.mock import patch, MagicMock
    from fastapi.testclient import TestClient
    from kenkui.server.api import app
    client = TestClient(app)
    mock_result = MagicMock()
    mock_result.speaker_voices = {"Alice": "alba", "Bob": "archie"}
    mock_result.warnings = []
    with patch("kenkui.server.api.get_server"), \
         patch("kenkui.services.voice_service.suggest_cast", return_value=mock_result):
        resp = client.post("/voices/suggest-cast", json={
            "roster": [{"name": "Alice", "pronoun": "she/her"},
                       {"name": "Bob", "pronoun": "he/him"}],
            "excluded_voices": [],
            "default_voice": "narrator",
        })
    assert resp.status_code == 200
    body = resp.json()
    assert "speaker_voices" in body
    assert "warnings" in body
