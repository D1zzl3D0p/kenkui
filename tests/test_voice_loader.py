from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

from kenkui.voice_loader import load_voice_conditioning_source
from kenkui.voice_registry import VoiceCatalog, VoiceCatalogEntry


def test_builtin_voice_conditioning_uses_predefined_prompt_url():
    catalog = MagicMock()
    catalog.resolve.return_value = VoiceCatalogEntry(
        voice_id="alba",
        display_name="Alba",
        origin="pocket_tts_builtin",
        asset_kind="pocket_tts_builtin",
        gender="Male",
        pool_enabled=True,
    )

    with patch("kenkui.voice_registry.get_catalog", return_value=catalog):
        source = load_voice_conditioning_source("alba")

    assert source.startswith("hf://")


def test_compiled_voice_conditioning_returns_local_path(tmp_path):
    asset = tmp_path / "voice.safetensors"
    asset.write_bytes(b"compiled")
    catalog = MagicMock()
    catalog.resolve.return_value = VoiceCatalogEntry(
        voice_id="custom",
        display_name="Custom",
        origin="custom_compiled",
        asset_kind="safetensors",
        gender="Female",
        pool_enabled=False,
        path=Path(asset),
    )

    with patch("kenkui.voice_registry.get_catalog", return_value=catalog):
        source = load_voice_conditioning_source("custom")

    assert source == str(asset)


def test_catalog_falls_back_to_bundled_compiled_voices(tmp_path):
    catalog = VoiceCatalog(data_dir=tmp_path)

    entry = catalog.resolve("alasdair-m-vctk-p246-scottish")

    assert entry is not None
    assert entry.origin == "kenkui_compiled"
    assert entry.path is not None
    assert entry.path.exists()
