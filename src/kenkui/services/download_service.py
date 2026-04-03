"""Download service wrapping voices/download.py."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from kenkui.voices.download import download_voices, fetch_uncompiled_voices


@dataclass
class DownloadResult:
    success: bool
    path: str
    message: str


def download_compiled(
    force: bool = False,
    progress_callback: Callable[[int, str], None] | None = None,
) -> DownloadResult:
    """Download compiled voices from HuggingFace."""
    from kenkui.voices.download import _VOICES_LOCAL_DIR
    try:
        download_voices(force=force, progress_callback=progress_callback)
        return DownloadResult(success=True, path=str(_VOICES_LOCAL_DIR), message="Download complete")
    except Exception as e:
        return DownloadResult(success=False, path=str(_VOICES_LOCAL_DIR), message=str(e))


def fetch_uncompiled(
    repo_id: str | None = None,
    patterns: list[str] | None = None,
    progress_callback: Callable[[int, str], None] | None = None,
) -> DownloadResult:
    """Fetch uncompiled voice sources from HuggingFace."""
    from kenkui.voices.download import _VOICES_LOCAL_DIR
    try:
        fetch_uncompiled_voices(repo_id=repo_id, patterns=patterns, progress_callback=progress_callback)
        return DownloadResult(success=True, path=str(_VOICES_LOCAL_DIR), message="Fetch complete")
    except Exception as e:
        return DownloadResult(success=False, path=str(_VOICES_LOCAL_DIR), message=str(e))


__all__ = [
    "DownloadResult",
    "download_compiled",
    "fetch_uncompiled",
]
