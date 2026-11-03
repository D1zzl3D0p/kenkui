"""Shared local model metadata for production-manifest tests."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from pathlib import Path


def production_manifest_engine(tmp_path: Path, config_name: str) -> dict[str, Any]:
    """Build the common engine fixture; voice provenance stays scenario-local."""
    root = tmp_path / "model"
    root.mkdir(mode=0o700, exist_ok=True)
    config = root / config_name
    config.write_text("model: local\n", encoding="utf-8")
    return {
        "language": "english",
        "model_root": str(root),
        "config_path": str(config),
        "model_revision": "revision-1",
        "package_version": "2.1.0",
        "files": [
            {
                "relative_path": config_name,
                "size": config.stat().st_size,
                "sha256": "a" * 64,
            }
        ],
        "sample_rate_hz": 24000,
        "device": "cpu",
        "timeout_seconds": 30.0,
        "cloning_capable": False,
    }
