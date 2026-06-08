from __future__ import annotations

from unittest.mock import MagicMock, patch

from kenkui.services.voice_service import SuggestCastResult, suggest_cast
from kenkui.voice_registry import VoiceCatalogEntry


def _make_roster(names_pronouns):
    from kenkui.models import CharacterInfo
    return [
        CharacterInfo(
            character_id=n,
            display_name=n,
            gender_pronoun=p,
            quote_count=1,
            mention_count=1,
        )
        for n, p in names_pronouns
    ]


def _voice(voice_id: str, gender: str) -> VoiceCatalogEntry:
    return VoiceCatalogEntry(
        voice_id=voice_id,
        display_name=voice_id.title(),
        origin="pocket_tts_builtin",
        asset_kind="pocket_tts_builtin",
        gender=gender,
        pool_enabled=True,
    )


def test_suggest_cast_returns_suggest_cast_result():
    catalog = MagicMock()
    catalog.pool.return_value = [_voice("alice", "Female"), _voice("bob", "Male")]
    roster = _make_roster([("Alice", "she/her"), ("Bob", "he/him")])

    with patch("kenkui.services.voice_service.get_catalog", return_value=catalog):
        result = suggest_cast(roster=roster, default_voice="narrator")

    assert isinstance(result, SuggestCastResult)
    assert result.speaker_voices["Alice"] == "alice"
    assert result.speaker_voices["Bob"] == "bob"


def test_suggest_cast_empty_pool_falls_back_to_default_voice():
    catalog = MagicMock()
    catalog.pool.return_value = []
    roster = _make_roster([("Alice", "she/her")])

    with patch("kenkui.services.voice_service.get_catalog", return_value=catalog):
        result = suggest_cast(roster=roster, default_voice="narrator")

    assert result.speaker_voices.get("Alice") == "narrator"
    assert result.warnings


def test_suggest_cast_accepts_legacy_excluded_voices_keyword():
    catalog = MagicMock()
    catalog.pool.return_value = [_voice("alice", "Female")]
    roster = _make_roster([("Alice", "she/her")])

    with patch("kenkui.services.voice_service.get_catalog", return_value=catalog):
        result = suggest_cast(
            roster=roster,
            default_voice="narrator",
            excluded_voices=["alice"],
        )

    assert result.speaker_voices["Alice"] == "alice"


def test_suggest_cast_resolves_chapter_conflicts():
    catalog = MagicMock()
    catalog.pool.return_value = [_voice("v1", "Female"), _voice("v2", "Female")]
    roster = _make_roster([("Alice", "she/her"), ("Eve", "she/her")])
    chapters = [
        MagicMock(
            paragraphs=[
                MagicMock(speaker="Alice", is_spoken=True),
                MagicMock(speaker="Eve", is_spoken=True),
            ]
        )
    ]

    with patch("kenkui.services.voice_service.get_catalog", return_value=catalog):
        result = suggest_cast(roster=roster, default_voice="narrator", chapters=chapters)

    assert result.speaker_voices["Alice"] != result.speaker_voices["Eve"]
