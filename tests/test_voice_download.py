"""Tests for src/kenkui/voice_download.py"""
import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch


def test_voices_are_present_false_when_dir_missing(tmp_path):
    from kenkui.voice_download import voices_are_present, _VOICES_LOCAL_DIR
    with patch("kenkui.voice_download._VOICES_LOCAL_DIR", tmp_path / "nonexistent"):
        assert voices_are_present() is False


def test_voices_are_present_false_when_empty(tmp_path):
    from kenkui.voice_download import voices_are_present
    compiled = tmp_path / "compiled"
    compiled.mkdir()
    with patch("kenkui.voice_download._VOICES_LOCAL_DIR", tmp_path):
        assert voices_are_present() is False


def test_voices_are_present_true_with_safetensors(tmp_path):
    from kenkui.voice_download import voices_are_present
    compiled = tmp_path / "compiled"
    compiled.mkdir()
    (compiled / "TestVoice-M-VCTK-P001-American.safetensors").touch()
    with patch("kenkui.voice_download._VOICES_LOCAL_DIR", tmp_path):
        assert voices_are_present() is True


def test_download_voices_calls_snapshot_download(tmp_path):
    from kenkui import voice_download as dl
    mock_invalidate = MagicMock()
    with (
        patch("kenkui.voice_download._VOICES_LOCAL_DIR", tmp_path),
        patch("huggingface_hub.snapshot_download") as mock_snap,
        patch("kenkui.voice_download.get_registry") as mock_reg,
    ):
        mock_reg.return_value.invalidate = mock_invalidate
        dl.download_voices()
    mock_snap.assert_called_once()
    call_kwargs = mock_snap.call_args
    assert call_kwargs.kwargs.get("repo_id") == dl.HF_VOICES_REPO or call_kwargs.args[0] == dl.HF_VOICES_REPO


def test_download_voices_calls_invalidate(tmp_path):
    from kenkui import voice_download as dl
    mock_registry = MagicMock()
    with (
        patch("kenkui.voice_download._VOICES_LOCAL_DIR", tmp_path),
        patch("huggingface_hub.snapshot_download"),
        patch("kenkui.voice_download.get_registry", return_value=mock_registry),
    ):
        dl.download_voices()
    mock_registry.invalidate.assert_called_once()
