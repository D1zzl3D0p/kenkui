"""Cache layer for NLP pipeline results.

Cache key: {book_hash}-{step}-{tool}-{model_slug}.json
  step  = "extraction" | "attribution"
  tool  = "booknlp"   | "ollama"      | "litellm"
  model = model name slugified (non-alphanumeric → underscore)

LOCAL and MODAL runs share the same cache (identical output for same model).
Reads are sorted by `created_at` descending (most recent first).
Writes are atomic: write to .tmp then os.replace().
"""
from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_logger = logging.getLogger(__name__)

_STEP_VALUES = frozenset({"extraction", "attribution"})


@dataclass
class CacheMeta:
    """Metadata for a single NLP cache entry."""
    path: Path
    step: str          # "extraction" | "attribution"
    tool: str          # "booknlp" | "ollama" | "litellm"
    model: str
    created_at: datetime
    description: str
    book_hash: str


def _model_slug(model: str) -> str:
    """Normalize a model name for use in a filename."""
    return re.sub(r"[^a-z0-9]+", "_", model.lower()).strip("_") or "default"


def _cache_filename(book_hash: str, step: str, tool: str, model: str) -> str:
    return f"{book_hash}-{step}-{tool}-{_model_slug(model)}.json"


def _get_cache_dir(cache_dir: Path | None = None) -> Path:
    if cache_dir is not None:
        return cache_dir
    # Lazy import to avoid circular dependency at module load time
    from kenkui.nlp import _get_config_dir  # noqa: PLC0415
    return _get_config_dir() / "nlp_cache"


def list_caches(
    book_path: Path,
    step: str | None = None,
    cache_dir: Path | None = None,
) -> list[CacheMeta]:
    """Return all cache entries for *book_path*, newest first.

    Args:
        book_path: Path to the ebook file (used to compute book_hash).
        step:      Filter to "extraction" or "attribution". None = both.
        cache_dir: Override cache directory (used in tests).
    """
    from kenkui.nlp import book_hash as _book_hash
    bh = _book_hash(book_path)
    cache_root = _get_cache_dir(cache_dir)
    if not cache_root.exists():
        return []

    metas: list[CacheMeta] = []
    pattern = re.compile(
        rf"^{re.escape(bh)}-(?P<step>[a-z]+)-(?P<tool>[a-z]+)-(?P<model>[a-z0-9_]+)\.json$"
    )
    for entry in cache_root.iterdir():
        m = pattern.match(entry.name)
        if not m or (step and m.group("step") != step):
            continue
        try:
            data = json.loads(entry.read_text(encoding="utf-8"))
            created_str = data.get("created_at", "")
            created_at = datetime.fromisoformat(created_str) if created_str else datetime.fromtimestamp(entry.stat().st_mtime, tz=timezone.utc)
        except Exception:  # noqa: BLE001
            continue
        metas.append(CacheMeta(
            path=entry,
            step=m.group("step"),
            tool=m.group("tool"),
            model=data.get("model", m.group("model")),
            created_at=created_at,
            description=data.get("description", ""),
            book_hash=bh,
        ))

    metas.sort(key=lambda x: x.created_at, reverse=True)
    return metas


def get_cache(
    book_path: Path,
    step: str,
    tool: str,
    model: str,
    cache_dir: Path | None = None,
) -> dict[str, Any] | None:
    """Return cached data dict or None if not found / unreadable."""
    from kenkui.nlp import book_hash as _book_hash
    bh = _book_hash(book_path)
    path = _get_cache_dir(cache_dir) / _cache_filename(bh, step, tool, model)
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        _logger.warning("Failed to read cache %s: %s", path, exc)
        return None


def put_cache(
    data: dict[str, Any],
    book_path: Path,
    step: str,
    tool: str,
    model: str,
    description: str = "",
    cache_dir: Path | None = None,
) -> Path:
    """Write *data* to cache atomically. Returns the cache path.

    Adds `created_at`, `description`, `step`, `tool`, `model` to the envelope.
    """
    from kenkui.nlp import book_hash as _book_hash
    bh = _book_hash(book_path)
    cache_root = _get_cache_dir(cache_dir)
    cache_root.mkdir(parents=True, exist_ok=True)

    envelope: dict[str, Any] = {
        "created_at": datetime.now(tz=timezone.utc).isoformat(),
        "description": description,
        "step": step,
        "tool": tool,
        "model": model,
        "book_hash": bh,
        **data,
    }
    dest = cache_root / _cache_filename(bh, step, tool, model)
    tmp = dest.with_suffix(".tmp")
    tmp.write_text(json.dumps(envelope, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp, dest)
    _logger.debug("Cached %s/%s result to %s", step, tool, dest.name)
    return dest


def delete_cache(meta: CacheMeta) -> None:
    """Delete a cache entry by its CacheMeta."""
    meta.path.unlink(missing_ok=True)
    _logger.debug("Deleted cache %s", meta.path.name)
