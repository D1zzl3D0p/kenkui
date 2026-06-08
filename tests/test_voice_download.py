from unittest.mock import MagicMock, patch


def test_voices_are_present_false_when_dir_missing(tmp_path):
    from kenkui import voice_download as dl

    with patch("kenkui.voice_download.compiled_voices_dir", return_value=tmp_path / "missing"):
        assert dl.voices_are_present() is False


def test_voices_are_present_false_when_empty(tmp_path):
    from kenkui import voice_download as dl

    compiled = tmp_path / "compiled"
    compiled.mkdir()
    with patch("kenkui.voice_download.compiled_voices_dir", return_value=compiled):
        assert dl.voices_are_present() is False


def test_voices_are_present_true_with_safetensors(tmp_path):
    from kenkui import voice_download as dl

    compiled = tmp_path / "compiled"
    compiled.mkdir()
    (compiled / "voice.safetensors").touch()
    with patch("kenkui.voice_download.compiled_voices_dir", return_value=compiled):
        assert dl.voices_are_present() is True


def test_download_voices_calls_snapshot_download(tmp_path):
    from kenkui import voice_download as dl

    catalog = MagicMock()
    with (
        patch("kenkui.voice_download.voice_data_dir", return_value=tmp_path),
        patch("huggingface_hub.snapshot_download") as mock_snap,
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
        __import__("json").dumps(
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


def test_fetch_uncompiled_voices_is_removed():
    from kenkui.voice_download import fetch_uncompiled_voices

    try:
        fetch_uncompiled_voices()
    except RuntimeError as exc:
        assert "Uncompiled voice sources" in str(exc)
    else:
        raise AssertionError("expected RuntimeError")
