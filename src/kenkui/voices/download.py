"""First-run voice download from HuggingFace Hub."""
from __future__ import annotations

from pathlib import Path

from ..voice_registry import get_registry

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


def download_voices(*, force: bool = False) -> None:
    """Download compiled + uncompiled voices from HuggingFace to XDG data dir.

    Uses huggingface_hub.snapshot_download which handles resume, progress,
    and local caching automatically.
    """
    from huggingface_hub import snapshot_download

    _VOICES_LOCAL_DIR.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=HF_VOICES_REPO,
        repo_type=HF_REPO_TYPE,
        local_dir=str(_VOICES_LOCAL_DIR),
        ignore_patterns=["*.md", "*.gitattributes", ".gitattributes"],
    )
    get_registry().invalidate()
