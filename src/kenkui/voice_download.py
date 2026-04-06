"""First-run voice download from HuggingFace Hub."""
from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from .voice_registry import get_registry

HF_VOICES_REPO = "D1zzl3D0p/kenkui-voices"
HF_REPO_TYPE = "dataset"

# XDG-compliant user data directory for downloaded voices
_VOICES_LOCAL_DIR = Path.home() / ".local" / "share" / "kenkui" / "voices"


def compiled_voices_dir() -> Path:
    return _VOICES_LOCAL_DIR / "compiled"


def voices_are_present() -> bool:
    """Return True if compiled voice files exist in the user data directory."""
    d = compiled_voices_dir()
    return d.exists() and any(d.glob("*.safetensors"))


def download_voices(
    *,
    force: bool = False,
    progress_callback: Callable[[int, str], None] | None = None,
) -> None:
    """Download compiled + uncompiled voices from HuggingFace to XDG data dir.

    Uses huggingface_hub.snapshot_download which handles resume, progress,
    and local caching automatically.  Pass ``force=True`` to wipe the local
    cache and re-download everything from scratch.

    ``progress_callback`` receives coarse-grained (percent, message) tuples:
    (0, "Starting download"), (10, "Downloading from HuggingFace"),
    (90, "Updating voice registry"), (100, "Download complete").
    Pass ``None`` (default) for silent operation.
    """
    import shutil
    from huggingface_hub import snapshot_download

    if progress_callback is not None:
        progress_callback(0, "Starting download")

    if force and _VOICES_LOCAL_DIR.exists():
        shutil.rmtree(_VOICES_LOCAL_DIR)

    _VOICES_LOCAL_DIR.mkdir(parents=True, exist_ok=True)

    if progress_callback is not None:
        progress_callback(10, "Downloading from HuggingFace")

    snapshot_download(
        repo_id=HF_VOICES_REPO,
        repo_type=HF_REPO_TYPE,
        local_dir=str(_VOICES_LOCAL_DIR),
        ignore_patterns=["*.md", "*.gitattributes", ".gitattributes"],
    )

    if progress_callback is not None:
        progress_callback(90, "Updating voice registry")

    get_registry().invalidate()

    if progress_callback is not None:
        progress_callback(100, "Download complete")


def fetch_uncompiled_voices(
    *,
    repo_id: str | None = None,
    patterns: list[str] | None = None,
    progress_callback: Callable[[int, str], None] | None = None,
) -> None:
    """Fetch uncompiled voice sources from a HuggingFace dataset repo.

    ``repo_id`` defaults to ``HF_VOICES_REPO`` when not provided.
    ``patterns`` is forwarded to ``snapshot_download`` as ``allow_patterns``.

    ``progress_callback`` receives coarse-grained (percent, message) tuples:
    (0, "Starting download"), (10, "Downloading from HuggingFace"),
    (90, "Updating voice registry"), (100, "Download complete").
    Pass ``None`` (default) for silent operation.
    """
    from huggingface_hub import snapshot_download

    if progress_callback is not None:
        progress_callback(0, "Starting download")

    _VOICES_LOCAL_DIR.mkdir(parents=True, exist_ok=True)

    if progress_callback is not None:
        progress_callback(10, "Downloading from HuggingFace")

    kwargs: dict[str, object] = {
        "repo_id": repo_id or HF_VOICES_REPO,
        "repo_type": HF_REPO_TYPE,
        "local_dir": str(_VOICES_LOCAL_DIR),
        "ignore_patterns": ["*.md", "*.gitattributes", ".gitattributes"],
    }
    if patterns:
        kwargs["allow_patterns"] = patterns
    snapshot_download(**kwargs)

    if progress_callback is not None:
        progress_callback(90, "Updating voice registry")

    get_registry().invalidate()

    if progress_callback is not None:
        progress_callback(100, "Download complete")
