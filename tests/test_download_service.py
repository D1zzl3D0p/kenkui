"""Tests for services/download_service.py and voices/download.py progress callbacks."""
from __future__ import annotations

from unittest.mock import MagicMock, patch, call


# ---------------------------------------------------------------------------
# download_service tests
# ---------------------------------------------------------------------------


def test_download_compiled_success():
    with patch("kenkui.services.download_service.download_voices") as mock_dv:
        from kenkui.services.download_service import download_compiled
        result = download_compiled()

    mock_dv.assert_called_once()
    assert result.success is True
    assert "kenkui" in result.path
    assert result.message == "Download complete"


def test_download_compiled_failure(tmp_path):
    with (
        patch("kenkui.voices.download._VOICES_LOCAL_DIR", tmp_path),
        patch("kenkui.services.download_service.download_voices", side_effect=RuntimeError("HF error")),
    ):
        from kenkui.services.download_service import download_compiled
        result = download_compiled()

    assert result.success is False
    assert "HF error" in result.message


def test_download_compiled_passes_force(tmp_path):
    with (
        patch("kenkui.voices.download._VOICES_LOCAL_DIR", tmp_path),
        patch("kenkui.services.download_service.download_voices") as mock_dv,
    ):
        from kenkui.services.download_service import download_compiled
        download_compiled(force=True)

    mock_dv.assert_called_once_with(force=True, progress_callback=None)


def test_download_compiled_passes_callback(tmp_path):
    cb = MagicMock()
    with (
        patch("kenkui.voices.download._VOICES_LOCAL_DIR", tmp_path),
        patch("kenkui.services.download_service.download_voices") as mock_dv,
    ):
        from kenkui.services.download_service import download_compiled
        download_compiled(progress_callback=cb)

    mock_dv.assert_called_once_with(force=False, progress_callback=cb)


def test_fetch_uncompiled_success(tmp_path):
    with (
        patch("kenkui.voices.download._VOICES_LOCAL_DIR", tmp_path),
        patch("kenkui.services.download_service.fetch_uncompiled_voices") as mock_fuv,
    ):
        from kenkui.services.download_service import fetch_uncompiled
        result = fetch_uncompiled()

    mock_fuv.assert_called_once()
    assert result.success is True
    assert result.message == "Fetch complete"


def test_fetch_uncompiled_failure(tmp_path):
    with (
        patch("kenkui.voices.download._VOICES_LOCAL_DIR", tmp_path),
        patch(
            "kenkui.services.download_service.fetch_uncompiled_voices",
            side_effect=RuntimeError("network error"),
        ),
    ):
        from kenkui.services.download_service import fetch_uncompiled
        result = fetch_uncompiled()

    assert result.success is False
    assert "network error" in result.message


def test_fetch_uncompiled_passes_repo_and_patterns(tmp_path):
    with (
        patch("kenkui.voices.download._VOICES_LOCAL_DIR", tmp_path),
        patch("kenkui.services.download_service.fetch_uncompiled_voices") as mock_fuv,
    ):
        from kenkui.services.download_service import fetch_uncompiled
        fetch_uncompiled(repo_id="org/repo", patterns=["*.wav"])

    mock_fuv.assert_called_once_with(repo_id="org/repo", patterns=["*.wav"], progress_callback=None)


# ---------------------------------------------------------------------------
# voices/download.py — progress callback integration tests
# ---------------------------------------------------------------------------


def test_download_voices_callback_called(tmp_path):
    """download_voices() should call progress_callback at 0, 50, and 100."""
    calls: list[tuple[int, str]] = []

    def cb(percent: int, message: str) -> None:
        calls.append((percent, message))

    mock_registry = MagicMock()
    with (
        patch("kenkui.voices.download._VOICES_LOCAL_DIR", tmp_path),
        patch("huggingface_hub.snapshot_download"),
        patch("kenkui.voices.download.get_registry", return_value=mock_registry),
    ):
        from kenkui.voices import download as dl
        dl.download_voices(progress_callback=cb)

    percents = [p for p, _ in calls]
    assert percents == [0, 50, 100], f"Expected [0, 50, 100], got {percents}"
    # Verify ordering is strictly ascending
    assert calls[0][0] == 0
    assert calls[1][0] == 50
    assert calls[2][0] == 100


def test_download_voices_no_callback_is_silent(tmp_path):
    """download_voices() with no callback should not raise."""
    mock_registry = MagicMock()
    with (
        patch("kenkui.voices.download._VOICES_LOCAL_DIR", tmp_path),
        patch("huggingface_hub.snapshot_download"),
        patch("kenkui.voices.download.get_registry", return_value=mock_registry),
    ):
        from kenkui.voices import download as dl
        dl.download_voices()  # should not raise


def test_fetch_uncompiled_voices_callback_called(tmp_path):
    """fetch_uncompiled_voices() should call progress_callback at 0, 50, and 100."""
    calls: list[tuple[int, str]] = []

    def cb(percent: int, message: str) -> None:
        calls.append((percent, message))

    mock_registry = MagicMock()
    with (
        patch("kenkui.voices.download._VOICES_LOCAL_DIR", tmp_path),
        patch("huggingface_hub.snapshot_download"),
        patch("kenkui.voices.download.get_registry", return_value=mock_registry),
    ):
        from kenkui.voices import download as dl
        dl.fetch_uncompiled_voices(repo_id="org/repo", patterns=["*.wav"], progress_callback=cb)

    percents = [p for p, _ in calls]
    assert percents == [0, 50, 100], f"Expected [0, 50, 100], got {percents}"


def test_fetch_uncompiled_voices_uses_default_repo(tmp_path):
    """fetch_uncompiled_voices() without repo_id uses HF_VOICES_REPO."""
    mock_registry = MagicMock()
    with (
        patch("kenkui.voices.download._VOICES_LOCAL_DIR", tmp_path),
        patch("huggingface_hub.snapshot_download") as mock_snap,
        patch("kenkui.voices.download.get_registry", return_value=mock_registry),
    ):
        from kenkui.voices import download as dl
        dl.fetch_uncompiled_voices()

    call_kwargs = mock_snap.call_args
    repo_arg = call_kwargs.kwargs.get("repo_id") or (call_kwargs.args[0] if call_kwargs.args else None)
    assert repo_arg == dl.HF_VOICES_REPO


def test_fetch_uncompiled_voices_passes_patterns(tmp_path):
    """fetch_uncompiled_voices() forwards allow_patterns when provided."""
    mock_registry = MagicMock()
    with (
        patch("kenkui.voices.download._VOICES_LOCAL_DIR", tmp_path),
        patch("huggingface_hub.snapshot_download") as mock_snap,
        patch("kenkui.voices.download.get_registry", return_value=mock_registry),
    ):
        from kenkui.voices import download as dl
        dl.fetch_uncompiled_voices(patterns=["*.wav", "*.pt"])

    call_kwargs = mock_snap.call_args
    assert call_kwargs.kwargs.get("allow_patterns") == ["*.wav", "*.pt"]
