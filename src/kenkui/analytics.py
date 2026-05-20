"""Analytics recording for kenkui pipeline stages.

Appends StageRecord objects as newline-delimited JSON (JSONL) to:
    $XDG_STATE_HOME/kenkui/analytics.jsonl   (default)
    $KENKUI_ANALYTICS_FILE                   (env var override)

One record per pipeline stage per invocation. Write failures are always
swallowed — analytics must never interrupt a conversion job.

Stages recorded:
    "nlp_extraction"   — character discovery (fast_scan)
    "nlp_attribution"  — speaker attribution (full_analysis)
    "tts_synthesis"    — TTS rendering across all chapters
    "stitching"        — WAV concatenation → M4B
    "normalization"    — loudness normalization (if enabled)
    "cover_embedding"  — cover art embed (if enabled)
"""
from __future__ import annotations

import json
import logging
import os
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

_logger = logging.getLogger(__name__)


def _analytics_path() -> Path:
    env = os.environ.get("KENKUI_ANALYTICS_FILE", "")
    if env:
        return Path(env)
    from .config import STATE_DIR
    return STATE_DIR / "analytics.jsonl"


@dataclass
class StageRecord:
    # ── Required fields ─────────────────────────────────────────────────────
    stage: str              # see module docstring for valid values
    started_at: str         # ISO 8601 UTC (datetime.now(timezone.utc).isoformat())
    duration_seconds: float
    success: bool

    # ── Auto-generated ───────────────────────────────────────────────────────
    record_id: str = field(default_factory=lambda: str(uuid.uuid4()))

    # ── Book context ─────────────────────────────────────────────────────────
    book_hash: str = ""
    book_title: str = ""
    book_char_count: int = 0
    chapter_count: int = 0

    # ── Provider / model ─────────────────────────────────────────────────────
    provider: str = ""      # "litellm" | "ollama" | "modal" | "booknlp" | "kokoro" | …
    model: str = ""

    # ── TTS-specific ─────────────────────────────────────────────────────────
    chars_per_second: float = 0.0
    narration_mode: str = ""    # "single" | "multi_voice" | "chapter_voice"

    # ── LLM API usage (populated from LiteLLM response when available) ───────
    llm_prompt_tokens: int = 0
    llm_completion_tokens: int = 0
    llm_api_calls: int = 0

    # ── Cache ────────────────────────────────────────────────────────────────
    cache_hit: bool = False


def append_record(record: StageRecord, path: Path | None = None) -> None:
    """Append one StageRecord to the analytics JSONL file. Never raises."""
    target = path or _analytics_path()
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        with open(target, "a", encoding="utf-8") as f:
            f.write(json.dumps(asdict(record), ensure_ascii=False) + "\n")
    except Exception as exc:
        _logger.debug("Analytics write failed (non-fatal): %s", exc)


def load_records(path: Path | None = None) -> list[StageRecord]:
    """Load all StageRecord objects from the analytics JSONL file."""
    target = path or _analytics_path()
    if not target.exists():
        return []
    records: list[StageRecord] = []
    with open(target, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(StageRecord(**json.loads(line)))
            except Exception:
                continue
    return records


def analytics_path() -> Path:
    """Return the path to the analytics JSONL file."""
    return _analytics_path()


def now_utc() -> str:
    """Return the current UTC time as an ISO 8601 string."""
    return datetime.now(timezone.utc).isoformat()
