from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from kenkui.services.voice_service import (
    VoiceInfo,
    import_custom_voice,
    list_voices,
    prepare_voice_preview,
    set_voice_pool_enabled,
    suggest_cast,
)
from kenkui.voice_registry import (
    PreviewInfo,
    VoiceCatalog,
    VoiceCatalogEntry,
    load_manifest,
    write_manifest,
)


def _entry(
    voice_id: str,
    gender: str,
    *,
    pool_enabled: bool = True,
    origin: str = "pocket_tts_builtin",
    path: Path | None = None,
) -> VoiceCatalogEntry:
    return VoiceCatalogEntry(
        voice_id=voice_id,
        display_name=voice_id.title(),
        origin=origin,  # type: ignore[arg-type]
        asset_kind="pocket_tts_builtin" if origin == "pocket_tts_builtin" else "safetensors",
        gender=gender,
        pool_enabled=pool_enabled,
        path=path,
    )


@dataclass
class FakeCharacter:
    character_id: str
    gender_pronoun: str | None
    prominence: int
    display_name: str | None = None

    def __post_init__(self) -> None:
        if self.display_name is None:
            self.display_name = self.character_id


def test_manifest_validation_rejects_duplicate_ids(tmp_path):
    path = tmp_path / "manifest.json"
    data = {
        "voices": [
            {
                "voice_id": "dup",
                "display_name": "One",
                "origin": "pocket_tts_builtin",
                "asset_kind": "pocket_tts_builtin",
                "gender": "Male",
                "pool_enabled": True,
            },
            {
                "voice_id": "dup",
                "display_name": "Two",
                "origin": "pocket_tts_builtin",
                "asset_kind": "pocket_tts_builtin",
                "gender": "Female",
                "pool_enabled": True,
            },
        ]
    }
    path.write_text(__import__("json").dumps(data), encoding="utf-8")

    with pytest.raises(ValueError, match="Duplicate voice_id"):
        load_manifest(path)


def test_manifest_validation_requires_gender(tmp_path):
    path = tmp_path / "manifest.json"
    path.write_text(
        __import__("json").dumps(
            {
                "voices": [
                    {
                        "voice_id": "bad",
                        "display_name": "Bad",
                        "origin": "pocket_tts_builtin",
                        "asset_kind": "pocket_tts_builtin",
                        "pool_enabled": True,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="invalid gender"):
        load_manifest(path)


def test_list_voices_returns_catalog_fields():
    catalog = MagicMock()
    catalog.filter.return_value = [
        _entry("custom_voice", "Female", origin="custom_compiled"),
        _entry("alba", "Male"),
    ]

    with patch("kenkui.services.voice_service.get_catalog", return_value=catalog):
        result = list_voices(gender="Male", pool_enabled=True)

    assert isinstance(result[0], VoiceInfo)
    assert result[0].voice_id == "alba"
    assert result[0].origin == "pocket_tts_builtin"
    assert result[0].pool_enabled is True
    assert [voice.voice_id for voice in result] == ["alba", "custom_voice"]
    catalog.filter.assert_called_once_with(
        gender="Male",
        accent=None,
        dataset=None,
        origin=None,
        asset_kind=None,
        pool_enabled=True,
        status=None,
    )


def test_list_voices_sorting_all_origins():
    """pocket_tts_builtin < kenkui_compiled < custom_compiled regardless of input order."""
    catalog = MagicMock()
    catalog.filter.return_value = [
        _entry("custom_voice", "Female", origin="custom_compiled"),
        _entry("downloaded_voice", "Male", origin="kenkui_compiled"),
        _entry("alba", "Male"),
    ]

    with patch("kenkui.services.voice_service.get_catalog", return_value=catalog):
        result = list_voices()

    assert [v.origin for v in result] == [
        "pocket_tts_builtin",
        "kenkui_compiled",
        "custom_compiled",
    ]
    assert [v.voice_id for v in result] == ["alba", "downloaded_voice", "custom_voice"]


def test_set_voice_pool_enabled_writes_catalog_state():
    catalog = MagicMock()
    catalog.set_pool_enabled.return_value = _entry("alba", "Male", pool_enabled=False)

    with patch("kenkui.services.voice_service.get_catalog", return_value=catalog):
        result = set_voice_pool_enabled("alba", False)

    assert result.voice_id == "alba"
    assert result.pool_enabled is False


def test_catalog_pool_override_replaces_builtin_without_duplicate(tmp_path):
    catalog = VoiceCatalog(data_dir=tmp_path)

    updated = catalog.set_pool_enabled("alba", False)

    assert updated.voice_id == "alba"
    assert updated.pool_enabled is False

    reloaded = VoiceCatalog(data_dir=tmp_path)
    alba_entries = [v for v in reloaded.voices if v.voice_id == "alba"]
    assert len(alba_entries) == 1
    assert alba_entries[0].pool_enabled is False


def test_suggest_cast_uses_pool_enabled_entries_only():
    catalog = MagicMock()
    catalog.pool.return_value = [
        _entry("enabled_male", "Male", pool_enabled=True),
        _entry("enabled_female", "Female", pool_enabled=True),
    ]
    roster = [
        FakeCharacter("Rand", "he/him", 10),
        FakeCharacter("Egwene", "she/her", 9),
    ]

    with patch("kenkui.services.voice_service.get_catalog", return_value=catalog):
        result = suggest_cast(roster=roster, default_voice="alba")

    assert result.speaker_voices == {
        "Rand": "enabled_male",
        "Egwene": "enabled_female",
    }


def test_suggest_cast_empty_pool_falls_back_to_default():
    catalog = MagicMock()
    catalog.pool.return_value = []

    with patch("kenkui.services.voice_service.get_catalog", return_value=catalog):
        result = suggest_cast(
            roster=[FakeCharacter("Alice", "she/her", 1)],
            default_voice="narrator",
        )

    assert result.speaker_voices["Alice"] == "narrator"
    assert result.warnings


def test_prepare_voice_preview_reuses_manifest_path(tmp_path):
    preview = tmp_path / "preview.wav"
    preview.write_bytes(b"wav")
    entry = VoiceCatalogEntry(
        voice_id="custom",
        display_name="Custom",
        origin="custom_compiled",
        asset_kind="safetensors",
        gender="Female",
        pool_enabled=False,
        path=tmp_path / "custom.safetensors",
        preview=PreviewInfo(path=str(preview)),
    )
    catalog = MagicMock()
    catalog.resolve.return_value = entry

    with patch("kenkui.services.voice_service.get_catalog", return_value=catalog):
        result = prepare_voice_preview("custom")

    assert result.audio_path == str(preview)


def test_prepare_voice_preview_reuses_manifest_path_only_for_matching_text(tmp_path):
    preview = tmp_path / "preview.wav"
    preview.write_bytes(b"wav")
    entry = VoiceCatalogEntry(
        voice_id="custom",
        display_name="Custom",
        origin="custom_compiled",
        asset_kind="safetensors",
        gender="Female",
        pool_enabled=False,
        path=tmp_path / "custom.safetensors",
        preview=PreviewInfo(text="Manifest words", path=str(preview)),
    )
    catalog = MagicMock()
    catalog.resolve.return_value = entry

    with (
        patch("kenkui.services.voice_service.get_catalog", return_value=catalog),
        patch("kenkui.services.voice_service._synthesize_preview") as synthesize,
        patch("kenkui.services.voice_service.preview_cache_dir", return_value=tmp_path / "cache"),
    ):
        matching = prepare_voice_preview("custom", text="Manifest words")
        different = prepare_voice_preview("custom", text="Different words")

    assert matching.audio_path == str(preview)
    assert different.audio_path != str(preview)
    synthesize.assert_called_once_with(entry, Path(different.audio_path), "Different words")


def test_prepare_voice_preview_cache_is_text_sensitive(tmp_path):
    entry = VoiceCatalogEntry(
        voice_id="alba",
        display_name="Alba",
        origin="pocket_tts_builtin",
        asset_kind="pocket_tts_builtin",
        gender="Male",
        pool_enabled=True,
    )
    catalog = MagicMock()
    catalog.resolve.return_value = entry

    with (
        patch("kenkui.services.voice_service.get_catalog", return_value=catalog),
        patch("kenkui.services.voice_service._synthesize_preview") as synthesize,
        patch("kenkui.services.voice_service.preview_cache_dir", return_value=tmp_path),
    ):
        first = prepare_voice_preview("alba", text="First text")
        second = prepare_voice_preview("alba", text="Second text")

    assert first.audio_path != second.audio_path
    assert synthesize.call_count == 2


def test_import_custom_voice_compiles_and_writes_manifest(tmp_path):
    catalog_path = tmp_path / "custom" / "custom_manifest.json"

    class FakeCatalog:
        custom_manifest_path = catalog_path

        def add_custom_voice(self, **kwargs):
            entry = VoiceCatalogEntry(
                voice_id=kwargs["voice_id"],
                display_name=kwargs["display_name"],
                origin="custom_compiled",
                asset_kind="safetensors",
                gender=kwargs["gender"],
                pool_enabled=kwargs["pool_enabled"],
                path=kwargs["compiled_path"],
                tags=tuple(kwargs["tags"] or ()),
                notes=kwargs["notes"],
            )
            write_manifest(catalog_path, [entry])
            return entry

    def compiler(_source: str, output_path: Path) -> Path:
        output_path.write_bytes(b"compiled")
        return output_path

    with patch("kenkui.services.voice_service.get_catalog", return_value=FakeCatalog()):
        result = import_custom_voice(
            source="prompt.wav",
            voice_id="my_voice",
            display_name="My Voice",
            gender="Female",
            pool_enabled=True,
            tags=["narrator"],
            notes="local test",
            compiler=compiler,
        )

    assert result.voice.voice_id == "my_voice"
    assert result.voice.origin == "custom_compiled"
    assert catalog_path.exists()


def test_add_custom_voice_copies_preview_out_of_temp_dir(tmp_path, monkeypatch):
    cache_home = tmp_path / "cache"
    monkeypatch.setenv("XDG_CACHE_HOME", str(cache_home))
    catalog = VoiceCatalog(data_dir=tmp_path / "data")
    compiled = tmp_path / "compiled.safetensors"
    compiled.write_bytes(b"compiled")
    preview = tmp_path / "preview.wav"
    preview.write_bytes(b"preview")

    entry = catalog.add_custom_voice(
        voice_id="local_preview",
        display_name="Local Preview",
        gender="Female",
        compiled_path=compiled,
        preview_path=preview,
    )

    assert entry.preview.path is not None
    assert Path(entry.preview.path).exists()
    assert Path(entry.preview.path).is_relative_to(cache_home)
