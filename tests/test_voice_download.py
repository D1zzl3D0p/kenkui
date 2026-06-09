import json
from unittest.mock import MagicMock, patch


def _write_voice_pack_manifest(root, voice_id="voice"):
    compiled = root / "compiled"
    compiled.mkdir(parents=True, exist_ok=True)
    asset = compiled / f"{voice_id}.safetensors"
    asset.write_bytes(b"compiled")
    (root / "manifest.json").write_text(
        json.dumps(
            {
                "voices": [
                    {
                        "voice_id": voice_id,
                        "display_name": voice_id.title(),
                        "origin": "kenkui_compiled",
                        "asset_kind": "safetensors",
                        "gender": "Female",
                        "pool_enabled": True,
                        "path": f"compiled/{voice_id}.safetensors",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )


def test_voices_are_present_false_when_dir_missing(tmp_path):
    from kenkui import voice_download as dl

    with patch("kenkui.voice_download.voice_data_dir", return_value=tmp_path / "missing"):
        assert dl.voices_are_present() is False


def test_voices_are_present_false_when_empty(tmp_path):
    from kenkui import voice_download as dl

    tmp_path.mkdir(exist_ok=True)
    with patch("kenkui.voice_download.voice_data_dir", return_value=tmp_path):
        assert dl.voices_are_present() is False


def test_voices_are_present_false_with_assets_but_no_manifest(tmp_path):
    from kenkui import voice_download as dl

    compiled = tmp_path / "compiled"
    compiled.mkdir()
    (compiled / "voice.safetensors").touch()
    with patch("kenkui.voice_download.voice_data_dir", return_value=tmp_path):
        assert dl.voices_are_present() is False


def test_voices_are_present_true_with_manifest_and_asset(tmp_path):
    from kenkui import voice_download as dl

    _write_voice_pack_manifest(tmp_path)
    with patch("kenkui.voice_download.voice_data_dir", return_value=tmp_path):
        assert dl.voices_are_present() is True


def test_download_voices_calls_snapshot_download(tmp_path):
    from kenkui import voice_download as dl

    catalog = MagicMock()
    with (
        patch("kenkui.voice_download.voice_data_dir", return_value=tmp_path),
        patch("huggingface_hub.snapshot_download", side_effect=lambda **_: _write_voice_pack_manifest(tmp_path)) as mock_snap,
        patch("kenkui.voice_download.get_catalog", return_value=catalog),
    ):
        dl.download_voices()

    mock_snap.assert_called_once()
    assert mock_snap.call_args.kwargs["repo_id"] == dl.HF_VOICES_REPO
    assert mock_snap.call_args.kwargs["revision"] == dl.HF_VOICES_REVISION
    catalog.invalidate.assert_called_once()


def test_download_voices_verifies_manifest_hashes(tmp_path):
    from kenkui import voice_download as dl

    compiled = tmp_path / "compiled"
    compiled.mkdir()
    asset = compiled / "voice.safetensors"
    asset.write_bytes(b"compiled")
    (tmp_path / "manifest.json").write_text(
        json.dumps(
            {
                "voices": [
                    {
                        "voice_id": "bad_hash",
                        "display_name": "Bad Hash",
                        "origin": "kenkui_compiled",
                        "asset_kind": "safetensors",
                        "gender": "Female",
                        "pool_enabled": True,
                        "path": "compiled/voice.safetensors",
                        "sha256": "0" * 64,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    with (
        patch("kenkui.voice_download.voice_data_dir", return_value=tmp_path),
        patch("huggingface_hub.snapshot_download"),
    ):
        try:
            dl.download_voices()
        except ValueError as exc:
            assert "hash" in str(exc)
        else:
            raise AssertionError("expected hash verification failure")


def test_download_voices_requires_manifest_after_snapshot(tmp_path):
    from kenkui import voice_download as dl

    with (
        patch("kenkui.voice_download.voice_data_dir", return_value=tmp_path),
        patch("huggingface_hub.snapshot_download"),
    ):
        try:
            dl.download_voices()
        except RuntimeError as exc:
            assert "manifest" in str(exc)
        else:
            raise AssertionError("expected missing manifest failure")


def test_force_download_preserves_custom_voices(tmp_path):
    from kenkui import voice_download as dl

    custom = tmp_path / "custom"
    custom.mkdir(parents=True)
    custom_asset = custom / "my_voice.safetensors"
    custom_asset.write_bytes(b"custom")
    custom_manifest = custom / "custom_manifest.json"
    custom_manifest.write_text('{"voices": []}\n', encoding="utf-8")
    _write_voice_pack_manifest(tmp_path, voice_id="old_voice")
    catalog = MagicMock()

    with (
        patch("kenkui.voice_download.voice_data_dir", return_value=tmp_path),
        patch("huggingface_hub.snapshot_download", side_effect=lambda **_: _write_voice_pack_manifest(tmp_path, voice_id="new_voice")),
        patch("kenkui.voice_download.get_catalog", return_value=catalog),
    ):
        dl.download_voices(force=True)

    assert custom_asset.exists()
    assert custom_manifest.exists()
    assert not (tmp_path / "compiled" / "old_voice.safetensors").exists()
    assert (tmp_path / "compiled" / "new_voice.safetensors").exists()
    catalog.invalidate.assert_called_once()


def test_fetch_uncompiled_voices_is_removed():
    from kenkui.voice_download import fetch_uncompiled_voices

    try:
        fetch_uncompiled_voices()
    except RuntimeError as exc:
        assert "Uncompiled voice sources" in str(exc)
    else:
        raise AssertionError("expected RuntimeError")
