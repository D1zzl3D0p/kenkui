"""Tests for services/download_service.py and voice_download.py."""
from __future__ import annotations

from unittest.mock import MagicMock, patch


def test_download_compiled_success(tmp_path):
    with (
        patch("kenkui.services.download_service.download_voices") as mock_dv,
        patch("kenkui.services.download_service.voices_local_dir", return_value=tmp_path),
    ):
        from kenkui.services.download_service import download_compiled

        result = download_compiled()

    mock_dv.assert_called_once_with(force=False, progress_callback=None)
    assert result.success is True
    assert result.path == str(tmp_path)
    assert result.message == "Download complete"


def test_download_compiled_failure(tmp_path):
    with (
        patch("kenkui.services.download_service.voices_local_dir", return_value=tmp_path),
        patch("kenkui.services.download_service.download_voices", side_effect=RuntimeError("HF error")),
    ):
        from kenkui.services.download_service import download_compiled

        result = download_compiled()

    assert result.success is False
    assert result.path == str(tmp_path)
    assert "HF error" in result.message


def test_download_compiled_passes_force_and_callback():
    cb = MagicMock()
    with patch("kenkui.services.download_service.download_voices") as mock_dv:
        from kenkui.services.download_service import download_compiled

        download_compiled(force=True, progress_callback=cb)

    mock_dv.assert_called_once_with(force=True, progress_callback=cb)


def test_fetch_uncompiled_reports_removed_runtime_path(tmp_path):
    with (
        patch("kenkui.services.download_service.voices_local_dir", return_value=tmp_path),
        patch(
            "kenkui.services.download_service.fetch_uncompiled_voices",
            side_effect=RuntimeError("Uncompiled voice sources are import inputs only"),
        ),
    ):
        from kenkui.services.download_service import fetch_uncompiled

        result = fetch_uncompiled(repo_id="org/repo", patterns=["*.wav"])

    assert result.success is False
    assert result.path == str(tmp_path)
    assert "Uncompiled voice sources" in result.message


def test_download_voices_callback_called(tmp_path):
    calls: list[tuple[int, str]] = []

    def cb(percent: int, message: str) -> None:
        calls.append((percent, message))

    catalog = MagicMock()
    with (
        patch("kenkui.voice_download.voice_data_dir", return_value=tmp_path),
        patch("huggingface_hub.snapshot_download"),
        patch("kenkui.voice_download.get_catalog", return_value=catalog),
    ):
        from kenkui import voice_download as dl

        dl.download_voices(progress_callback=cb)

    assert [p for p, _ in calls] == [0, 10, 90, 100]
    catalog.invalidate.assert_called_once()


def test_download_voices_no_callback_is_silent(tmp_path):
    catalog = MagicMock()
    with (
        patch("kenkui.voice_download.voice_data_dir", return_value=tmp_path),
        patch("huggingface_hub.snapshot_download"),
        patch("kenkui.voice_download.get_catalog", return_value=catalog),
    ):
        from kenkui import voice_download as dl

        dl.download_voices()


def test_download_voices_uses_pinned_revision(tmp_path):
    catalog = MagicMock()
    with (
        patch("kenkui.voice_download.voice_data_dir", return_value=tmp_path),
        patch("huggingface_hub.snapshot_download") as mock_snap,
        patch("kenkui.voice_download.get_catalog", return_value=catalog),
    ):
        from kenkui import voice_download as dl

        dl.download_voices()

    assert mock_snap.call_args.kwargs["repo_id"] == dl.HF_VOICES_REPO
    assert mock_snap.call_args.kwargs["revision"] == dl.HF_VOICES_REVISION
    assert "uncompiled/**" in mock_snap.call_args.kwargs["ignore_patterns"]


def test_fetch_uncompiled_voices_is_removed():
    from kenkui.voice_download import fetch_uncompiled_voices

    try:
        fetch_uncompiled_voices()
    except RuntimeError as exc:
        assert "Uncompiled voice sources" in str(exc)
    else:
        raise AssertionError("expected RuntimeError")
