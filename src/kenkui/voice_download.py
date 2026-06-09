"""First-run voice download from HuggingFace Hub."""
from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from .voice_registry import (
    DEFAULT_VOICE_PACK_REPO,
    DEFAULT_VOICE_PACK_REVISION,
    get_catalog,
    validate_manifest,
    verify_manifest_assets,
    voice_data_dir,
    voice_pack_manifest_is_current,
)

HF_VOICES_REPO = DEFAULT_VOICE_PACK_REPO
HF_REPO_TYPE = "dataset"
HF_VOICES_REVISION = DEFAULT_VOICE_PACK_REVISION


def voices_local_dir():
    return voice_data_dir()


def voices_are_present() -> bool:
    """Return True when the managed compiled voice pack is cataloged and usable."""
    root = voice_data_dir()
    manifest = _voice_pack_manifest(root)
    if manifest is None:
        return False
    if not voice_pack_manifest_is_current(manifest):
        return False
    try:
        entries = validate_manifest(manifest)
    except Exception:
        return False
    return any(
        entry.origin == "kenkui_compiled"
        and entry.status == "available"
        and entry.path is not None
        and entry.path.exists()
        for entry in entries
    )


def _voice_pack_manifest(root: Path) -> Path | None:
    for name in ("manifest.json", "voice_manifest.json", "voices/manifest.json"):
        path = root / name
        if path.exists():
            return path
    return None


def _remove_managed_voice_pack(root: Path) -> None:
    import shutil

    for name in ("manifest.json", "voice_manifest.json"):
        path = root / name
        if path.exists():
            path.unlink()
    for name in ("compiled", "previews", "voices"):
        path = root / name
        if path.exists():
            if path.is_dir():
                shutil.rmtree(path)
            else:
                path.unlink()


def download_voices(
    *,
    force: bool = False,
    progress_callback: Callable[[int, str], None] | None = None,
) -> None:
    """Download the pinned compiled voice-pack manifest/assets to XDG data.

    Uses huggingface_hub.snapshot_download which handles resume, progress,
    and local caching automatically.  Pass ``force=True`` to wipe the local
    cache and re-download everything from scratch.

    ``progress_callback`` receives coarse-grained (percent, message) tuples:
    (0, "Starting download"), (10, "Downloading from HuggingFace"),
    (90, "Updating voice registry"), (100, "Download complete").
    Pass ``None`` (default) for silent operation.
    """
    from huggingface_hub import snapshot_download

    if progress_callback is not None:
        progress_callback(0, "Starting download")

    local_dir = voice_data_dir()
    if force and local_dir.exists():
        _remove_managed_voice_pack(local_dir)

    local_dir.mkdir(parents=True, exist_ok=True)

    if progress_callback is not None:
        progress_callback(10, "Downloading from HuggingFace")

    snapshot_download(
        repo_id=HF_VOICES_REPO,
        repo_type=HF_REPO_TYPE,
        revision=HF_VOICES_REVISION,
        local_dir=str(local_dir),
        allow_patterns=["manifest.json", "voice_manifest.json", "voices/**", "compiled/**", "previews/**"],
        ignore_patterns=["*.md", "*.gitattributes", ".gitattributes", "sources/**", "uncompiled/**"],
    )

    manifest = _voice_pack_manifest(local_dir)
    if manifest is None:
        raise RuntimeError("Downloaded voice pack did not contain a supported manifest")
    if not voice_pack_manifest_is_current(manifest):
        raise RuntimeError("Downloaded voice pack is stale and must be rebuilt")
    verify_manifest_assets(validate_manifest(manifest))

    if progress_callback is not None:
        progress_callback(90, "Updating voice registry")

    get_catalog().invalidate()

    if progress_callback is not None:
        progress_callback(100, "Download complete")


def fetch_uncompiled_voices(
    *,
    repo_id: str | None = None,
    patterns: list[str] | None = None,
    progress_callback: Callable[[int, str], None] | None = None,
) -> None:
    """Raw voice sources are no longer installed for runtime use."""
    del repo_id, patterns, progress_callback
    raise RuntimeError("Uncompiled voice sources are import inputs only; install compiled voice packs instead")
