from __future__ import annotations

import re
from enum import Enum

_SLUG_RE = re.compile(r"^[a-z0-9_]+$")
_PRESERVED_SPEAKER_KEYS = frozenset({"NARRATOR", "Unknown", "SCENE_BREAK"})


def _migrate_speaker_voices_keys(d: dict[str, str]) -> dict[str, str]:
    """Convert legacy canonical-name keys to slug form."""
    result: dict[str, str] = {}
    for key, voice in d.items():
        if key in _PRESERVED_SPEAKER_KEYS or _SLUG_RE.match(key):
            result[key] = voice
        else:
            s = key.lower()
            s = re.sub(r"['\u2018\u2019]", "", s)
            s = re.sub(r"[^a-z0-9]+", "_", s)
            s = s.strip("_")
            result[s] = voice
    return result


def _normalize_bitrate(value: str | None, default: str = "96k") -> str:
    """Ensure a bitrate string always has a unit suffix."""
    if not value:
        return default
    v = str(value).strip().lower()
    if not v:
        return default
    if re.match(r"^\d+[kmg]$", v):
        return v
    if re.match(r"^\d+$", v):
        return v if int(v) >= 1000 else f"{v}k"
    return default


class JobStatus(Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    PAUSED = "paused"


class TTSExecutionMode(Enum):
    LOCAL = "local"
    MODAL = "modal"


class NlpExecutionMode(Enum):
    LOCAL = "local"
    MODAL = "modal"


class AttributionExecutionMode(Enum):
    LOCAL = "local"
    MODAL = "modal"


class ExtractionTool(Enum):
    BOOKNLP = "booknlp"
    OLLAMA = "ollama"
    LITELLM = "litellm"
    OPENROUTER = "openrouter"
    ANTHROPIC = "anthropic"
    OPENAI = "openai"
    GOOGLE = "google"


class AttributionTool(Enum):
    BOOKNLP = "booknlp"
    OLLAMA = "ollama"
    LITELLM = "litellm"
    OPENROUTER = "openrouter"
    ANTHROPIC = "anthropic"
    OPENAI = "openai"
    GOOGLE = "google"


class CostStatus(Enum):
    NONE = "none"
    ESTIMATED = "estimated"
    FINAL = "final"


class NarrationMode(Enum):
    """Whether a job uses a single narrator voice or per-character multi-voice."""

    SINGLE = "single"
    MULTI = "multi"


class ChapterPreset(Enum):
    NONE = "none"
    CONTENT_ONLY = "content-only"
    CHAPTERS_ONLY = "chapters-only"
    WITH_PARTS = "with-parts"
    MANUAL = "manual"
    CUSTOM = "custom"
