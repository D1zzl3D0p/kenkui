"""Tests for kenkui.services.voice_service."""

from __future__ import annotations

from dataclasses import dataclass
from unittest.mock import MagicMock, patch

import pytest

from kenkui.services.voice_service import (
    ExcludeResult,
    IncludeResult,
    VoiceInfo,
    exclude_voice,
    gender_from_pronoun,
    get_voice,
    include_voice,
    list_voices,
    sort_cast,
    synthesize_preview,
    top_gender_matched_voice,
)
from kenkui.voice_registry import VoiceMetadata


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_meta(
    name: str,
    gender: str | None = "Male",
    source: str = "builtin",
    accent: str | None = "American",
    dataset: str | None = None,
    speaker_id: str | None = None,
) -> VoiceMetadata:
    return VoiceMetadata(
        name=name,
        file_path=None,
        source=source,
        gender=gender,
        accent=accent,
        dataset=dataset,
        speaker_id=speaker_id,
    )


def _make_app_config(excluded_voices: list[str] | None = None):
    cfg = MagicMock()
    cfg.excluded_voices = excluded_voices or []
    return cfg


@dataclass
class FakeCharacter:
    gender_pronoun: str | None
    prominence: int


# ---------------------------------------------------------------------------
# list_voices
# ---------------------------------------------------------------------------


def test_list_voices_sets_excluded_flag():
    """Excluded flag is True for voices in excluded_voices config list."""
    meta_a = _make_meta("alba")
    meta_b = _make_meta("marius")

    mock_registry = MagicMock()
    mock_registry.filter.return_value = [meta_a, meta_b]

    mock_config = _make_app_config(excluded_voices=["alba"])

    with (
        patch("kenkui.services.voice_service.get_registry", return_value=mock_registry),
        patch("kenkui.services.voice_service.load_app_config", return_value=mock_config),
    ):
        result = list_voices()

    assert len(result) == 2
    alba_info = next(v for v in result if v.name == "alba")
    marius_info = next(v for v in result if v.name == "marius")
    assert alba_info.excluded is True
    assert marius_info.excluded is False


def test_list_voices_filters_by_gender():
    """list_voices passes gender filter through to registry.filter."""
    mock_registry = MagicMock()
    mock_registry.filter.return_value = []
    mock_config = _make_app_config()

    with (
        patch("kenkui.services.voice_service.get_registry", return_value=mock_registry),
        patch("kenkui.services.voice_service.load_app_config", return_value=mock_config),
    ):
        list_voices(gender="Female")

    mock_registry.filter.assert_called_once_with(
        gender="Female", accent=None, dataset=None, source=None
    )


# ---------------------------------------------------------------------------
# get_voice
# ---------------------------------------------------------------------------


def test_get_voice_found():
    """get_voice returns a VoiceInfo with correct fields and excluded flag."""
    meta = _make_meta("alba", gender="Male", accent="American")

    mock_registry = MagicMock()
    mock_registry.resolve.return_value = meta

    mock_config = _make_app_config(excluded_voices=[])

    with (
        patch("kenkui.services.voice_service.get_registry", return_value=mock_registry),
        patch("kenkui.services.voice_service.load_app_config", return_value=mock_config),
    ):
        result = get_voice("alba")

    assert result is not None
    assert isinstance(result, VoiceInfo)
    assert result.name == "alba"
    assert result.gender == "Male"
    assert result.accent == "American"
    assert result.excluded is False
    mock_registry.resolve.assert_called_once_with("alba")


def test_get_voice_not_found():
    """get_voice returns None when the registry does not recognise the name."""
    mock_registry = MagicMock()
    mock_registry.resolve.return_value = None

    with patch("kenkui.services.voice_service.get_registry", return_value=mock_registry):
        result = get_voice("nonexistent_voice")

    assert result is None


# ---------------------------------------------------------------------------
# exclude_voice
# ---------------------------------------------------------------------------


def test_exclude_voice_adds_to_list():
    """exclude_voice adds the voice name to config.excluded_voices and saves."""
    mock_config = _make_app_config(excluded_voices=[])
    mock_config.temp = 1.0
    mock_config.lsd_decode_steps = 10
    mock_config.noise_clamp = 1.0
    mock_config.eos_threshold = -4.0

    mock_registry = MagicMock()
    mock_registry.filter.return_value = []

    with (
        patch("kenkui.services.voice_service.load_app_config", return_value=mock_config),
        patch("kenkui.services.voice_service.save_app_config") as mock_save,
        patch("kenkui.services.voice_service.get_registry", return_value=mock_registry),
        patch("kenkui.services.voice_service.DEFAULT_CONFIG_PATH", "/fake/path.toml"),
    ):
        result = exclude_voice("alba")

    assert "alba" in result.excluded_voices
    mock_save.assert_called_once()


def test_exclude_voice_already_excluded_is_noop():
    """Re-excluding an already-excluded voice does not duplicate the entry."""
    mock_config = _make_app_config(excluded_voices=["alba"])

    mock_registry = MagicMock()
    mock_registry.filter.return_value = []

    with (
        patch("kenkui.services.voice_service.load_app_config", return_value=mock_config),
        patch("kenkui.services.voice_service.save_app_config"),
        patch("kenkui.services.voice_service.get_registry", return_value=mock_registry),
        patch("kenkui.services.voice_service.DEFAULT_CONFIG_PATH", "/fake/path.toml"),
    ):
        result = exclude_voice("alba")

    assert result.excluded_voices.count("alba") == 1


def test_exclude_voice_warns_on_pool_exhaustion():
    """When all Male voices are excluded, warning is set on ExcludeResult."""
    male_meta = _make_meta("alba", gender="Male")
    female_meta = _make_meta("fantine", gender="Female")

    # After exclusion, both "alba" are excluded — entire Male pool excluded.
    mock_config = _make_app_config(excluded_voices=["alba"])

    mock_registry = MagicMock()
    mock_registry.filter.side_effect = lambda gender=None, **_: (
        [male_meta] if gender == "Male" else [female_meta]
    )

    with (
        patch("kenkui.services.voice_service.load_app_config", return_value=mock_config),
        patch("kenkui.services.voice_service.save_app_config"),
        patch("kenkui.services.voice_service.get_registry", return_value=mock_registry),
        patch("kenkui.services.voice_service.DEFAULT_CONFIG_PATH", "/fake/path.toml"),
    ):
        # Excluding the one remaining male voice (config already has "alba" excluded,
        # but exclude_voice is idempotent, so the result still has only ["alba"])
        result = exclude_voice("alba")

    assert result.warning == "All Male voices are now excluded from auto-assignment"


# ---------------------------------------------------------------------------
# include_voice
# ---------------------------------------------------------------------------


def test_include_voice_removes_from_list():
    """include_voice removes the voice from excluded_voices and saves."""
    mock_config = _make_app_config(excluded_voices=["alba", "marius"])

    with (
        patch("kenkui.services.voice_service.load_app_config", return_value=mock_config),
        patch("kenkui.services.voice_service.save_app_config") as mock_save,
        patch("kenkui.services.voice_service.DEFAULT_CONFIG_PATH", "/fake/path.toml"),
    ):
        result = include_voice("alba")

    assert "alba" not in result.excluded_voices
    assert "marius" in result.excluded_voices
    mock_save.assert_called_once()


def test_include_voice_not_in_list_is_noop():
    """Including a voice not in the excluded list returns unchanged list without saving."""
    mock_config = _make_app_config(excluded_voices=["marius"])

    with (
        patch("kenkui.services.voice_service.load_app_config", return_value=mock_config),
        patch("kenkui.services.voice_service.save_app_config") as mock_save,
    ):
        result = include_voice("alba")

    assert result.excluded_voices == ["marius"]
    mock_save.assert_not_called()


# ---------------------------------------------------------------------------
# gender_from_pronoun
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "pronoun, expected",
    [
        ("she/her", "female"),
        ("she/her/hers", "female"),
        ("he/him/his", "male"),
        ("he/him", "male"),
        ("they/them", "they"),
        ("", "they"),
        (None, "they"),
        ("female", "female"),
        ("male", "male"),
        ("She/Her", "female"),   # case-insensitive
        ("HE/HIM", "male"),
    ],
)
def test_gender_from_pronoun_various_inputs(pronoun, expected):
    assert gender_from_pronoun(pronoun) == expected


# ---------------------------------------------------------------------------
# top_gender_matched_voice
# ---------------------------------------------------------------------------


def test_top_gender_matched_voice_female_dominant():
    """Female character has more prominence → returns first female voice."""
    chars = [
        FakeCharacter(gender_pronoun="she/her", prominence=100),
        FakeCharacter(gender_pronoun="he/him", prominence=50),
    ]

    female_meta = _make_meta("fantine", gender="Female")
    male_meta = _make_meta("alba", gender="Male")

    mock_registry = MagicMock()
    mock_registry.filter.side_effect = lambda gender=None, **_: (
        [male_meta] if gender == "Male" else [female_meta]
    )

    with patch("kenkui.services.voice_service.get_registry", return_value=mock_registry):
        result = top_gender_matched_voice(chars, excluded=[], default_voice="alba")

    assert result == "fantine"


def test_top_gender_matched_voice_falls_back_to_default():
    """When no voices are available (all excluded or registry empty), returns default_voice."""
    chars = [
        FakeCharacter(gender_pronoun="she/her", prominence=100),
    ]

    mock_registry = MagicMock()
    mock_registry.filter.return_value = []

    with patch("kenkui.services.voice_service.get_registry", return_value=mock_registry):
        result = top_gender_matched_voice(chars, excluded=[], default_voice="fallback")

    assert result == "fallback"


def test_top_gender_matched_voice_respects_excluded():
    """Excluded voices are not returned even when they match the dominant gender."""
    # Two male characters with equal prominence — only "marius" is not excluded.
    chars = [
        FakeCharacter(gender_pronoun="he/him", prominence=80),
        FakeCharacter(gender_pronoun="he/him", prominence=80),
    ]

    meta_excluded = _make_meta("alba", gender="Male")
    meta_available = _make_meta("marius", gender="Male")

    mock_registry = MagicMock()
    mock_registry.filter.side_effect = lambda gender=None, **_: (
        [meta_excluded, meta_available] if gender == "Male" else []
    )

    with patch("kenkui.services.voice_service.get_registry", return_value=mock_registry):
        result = top_gender_matched_voice(
            chars, excluded=["alba"], default_voice="fallback"
        )

    assert result == "marius"


# ---------------------------------------------------------------------------
# sort_cast
# ---------------------------------------------------------------------------


def test_sort_cast_narrator_last():
    """NARRATOR is pinned alphabetically after all other speakers."""
    cast = {
        "NARRATOR": "alba",
        "Alice": "fantine",
        "Bob": "marius",
    }
    result = sort_cast(cast)
    names = [k for k, _ in result]
    assert names[-1] == "NARRATOR"
    assert names[0] == "Alice"
    assert names[1] == "Bob"


def test_sort_cast_alphabetical_order():
    """Non-NARRATOR speakers are sorted case-insensitively."""
    cast = {
        "Zara": "v1",
        "alice": "v2",
        "Bob": "v3",
    }
    result = sort_cast(cast)
    names = [k for k, _ in result]
    assert names == ["alice", "Bob", "Zara"]


# ---------------------------------------------------------------------------
# synthesize_preview — skipped (requires pocket_tts model)
# ---------------------------------------------------------------------------


@pytest.mark.skip(reason="requires pocket_tts model — integration test only")
def test_synthesize_preview_creates_wav_file():
    result = synthesize_preview("alba", text="Hello world.")
    assert result.audio_path.endswith(".wav")
    assert result.duration_ms > 0
