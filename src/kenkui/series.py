"""Series voice consistency — manifest I/O and character matching.

A series manifest lives at:
    ~/.config/kenkui/series/{slug}.toml

It accumulates character→voice assignments across books, so the same
character always gets the same voice in subsequent installments.
"""
from __future__ import annotations

import re
import tomllib

import tomli_w
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .models import CharacterInfo, FastScanResult

# Test seam: override the series directory in tests via monkeypatch.
_series_dir_override: Path | None = None


def series_dir() -> Path:
    """Return (and create if needed) the series manifest directory."""
    if _series_dir_override is not None:
        _series_dir_override.mkdir(parents=True, exist_ok=True)
        return _series_dir_override
    from .config import CONFIG_DIR
    d = CONFIG_DIR / "series"
    d.mkdir(parents=True, exist_ok=True)
    return d


def slugify(name: str) -> str:
    """Convert a series name to a filesystem-safe slug."""
    s = name.lower().strip()
    s = re.sub(r"[^\w\s-]", "", s)
    s = re.sub(r"[\s_]+", "-", s)
    s = re.sub(r"-+", "-", s)
    return s.strip("-")


@dataclass
class SeriesCharacter:
    canonical: str
    aliases: list[str] = field(default_factory=list)
    voice: str = ""
    gender: str = ""


@dataclass
class SeriesManifest:
    name: str
    slug: str
    updated_at: str
    characters: list[SeriesCharacter] = field(default_factory=list)


def load_series(slug: str) -> SeriesManifest | None:
    """Load a series manifest by slug. Returns None if not found."""
    path = series_dir() / f"{slug}.toml"
    if not path.exists():
        return None
    try:
        data = tomllib.loads(path.read_text(encoding="utf-8"))
        return SeriesManifest(
            name=data["name"],
            slug=slug,
            updated_at=data.get("updated_at", ""),
            characters=[
                SeriesCharacter(
                    canonical=c["canonical"],
                    aliases=c.get("aliases", []),
                    voice=c.get("voice", ""),
                    gender=c.get("gender", ""),
                )
                for c in data.get("characters", [])
            ],
        )
    except Exception:
        return None


def save_series(manifest: SeriesManifest) -> None:
    """Persist a SeriesManifest to disk atomically."""
    manifest.updated_at = datetime.now(timezone.utc).isoformat()
    data: dict = {
        "name": manifest.name,
        "updated_at": manifest.updated_at,
        "characters": [
            {
                "canonical": c.canonical,
                "aliases": c.aliases,
                "voice": c.voice,
                "gender": c.gender,
            }
            for c in manifest.characters
        ],
    }
    path = series_dir() / f"{manifest.slug}.toml"
    tmp = path.with_suffix(".tmp")
    tmp.write_bytes(tomli_w.dumps(data).encode("utf-8"))
    tmp.replace(path)


def list_series() -> list[SeriesManifest]:
    """Return all series manifests sorted by mtime descending (most recent first)."""
    d = series_dir()
    results = []
    for f in sorted(d.glob("*.toml"), key=lambda p: p.stat().st_mtime, reverse=True):
        m = load_series(f.stem)
        if m is not None:
            results.append(m)
    return results
