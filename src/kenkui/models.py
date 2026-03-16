import multiprocessing
import re
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .chapter_classifier import ChapterTags
    from .chapter_filter import FilterOperation


def _normalize_bitrate(value: str | None, default: str = "96k") -> str:
    """Ensure a bitrate string always has a unit suffix.

    Accepts: '96k', '96K', '128000', '64', '96k ', ''
    Returns: '96k', '96k', '128000', '64k', '96k', default

    The key fix: bare integers like '64' become '64k' so ffmpeg interprets
    them as 64 kilobits/second rather than 64 bits/second.
    """
    if not value:
        return default
    v = str(value).strip().lower()
    if not v:
        return default
    # Already has a valid suffix (k, m, g) or is a large number (e.g. '128000')
    if re.match(r"^\d+[kmg]$", v):
        return v
    # Pure integer: values >= 1000 are already in bps, leave them alone;
    # values < 1000 are kbps entered without the unit — add 'k'.
    if re.match(r"^\d+$", v):
        return v if int(v) >= 1000 else f"{v}k"
    # Unrecognised — return default
    return default


class JobStatus(Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class ChapterPreset(Enum):
    NONE = "none"
    CONTENT_ONLY = "content-only"
    CHAPTERS_ONLY = "chapters-only"
    WITH_PARTS = "with-parts"
    MANUAL = "manual"
    CUSTOM = "custom"  # User manually modified selection


@dataclass
class BookInfo:
    """Lightweight book info for display in selection UI."""

    path: Path
    title: str
    author: str | None = None
    chapter_count: int = 0
    format: str = ""

    @property
    def display_name(self) -> str:
        return self.title or self.path.stem

    @property
    def display_author(self) -> str:
        return self.author or "Unknown"


@dataclass
class ChapterSelection:
    preset: ChapterPreset = ChapterPreset.CONTENT_ONLY
    included: list[int] = field(default_factory=list)
    excluded: list[int] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "preset": self.preset.value,
            "included": self.included,
            "excluded": self.excluded,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ChapterSelection":
        return cls(
            preset=ChapterPreset(data.get("preset", "content-only")),
            included=data.get("included", []),
            excluded=data.get("excluded", []),
        )


@dataclass
class JobConfig:
    ebook_path: Path
    voice: str = "alba"
    chapter_selection: ChapterSelection = field(default_factory=ChapterSelection)
    output_path: Path | None = None
    name: str = ""

    def __post_init__(self):
        if not self.name:
            self.name = self.ebook_path.stem

    def to_dict(self) -> dict[str, Any]:
        return {
            "ebook_path": str(self.ebook_path),
            "voice": self.voice,
            "chapter_selection": self.chapter_selection.to_dict(),
            "output_path": str(self.output_path) if self.output_path else None,
            "name": self.name,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "JobConfig":
        return cls(
            ebook_path=Path(data["ebook_path"]),
            voice=data.get("voice", "alba"),
            chapter_selection=ChapterSelection.from_dict(
                data.get("chapter_selection", {})
            ),
            output_path=Path(data["output_path"]) if data.get("output_path") else None,
            name=data.get("name", ""),
        )


@dataclass
class AppConfig:
    name: str = "default"  # Config name for saving/loading
    workers: int = max(2, multiprocessing.cpu_count() - 2)
    verbose: bool = False
    log_path: Path | None = None
    keep_temp: bool = False
    m4b_bitrate: str = "96k"
    pause_line_ms: int = 400
    pause_chapter_ms: int = 2000
    temp: float = 0.7  # Sampling temperature (lower = stable, higher = expressive)
    lsd_decode_steps: int = 1  # LSD decode steps (higher = better quality, slower)
    noise_clamp: float | None = (
        None  # Noise clamp (None = off; ~3.0 reduces audio glitches)
    )
    # --- Job defaults (used by CLI / headless mode) ---
    default_voice: str = "alba"  # Voice used when no per-job override
    default_chapter_preset: str = "content-only"  # Chapter filter preset for CLI
    default_output_dir: Path | None = None  # Output directory for CLI runs

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "workers": self.workers,
            "verbose": self.verbose,
            "log_path": str(self.log_path) if self.log_path else None,
            "keep_temp": self.keep_temp,
            "m4b_bitrate": self.m4b_bitrate,
            "pause_line_ms": self.pause_line_ms,
            "pause_chapter_ms": self.pause_chapter_ms,
            "temp": self.temp,
            "lsd_decode_steps": self.lsd_decode_steps,
            "noise_clamp": self.noise_clamp,
            "default_voice": self.default_voice,
            "default_chapter_preset": self.default_chapter_preset,
            "default_output_dir": str(self.default_output_dir)
            if self.default_output_dir
            else None,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "AppConfig":
        return cls(
            name=data.get("name", "default"),
            workers=data.get("workers", max(2, multiprocessing.cpu_count() - 2)),
            verbose=data.get("verbose", False),
            log_path=Path(data["log_path"]) if data.get("log_path") else None,
            keep_temp=data.get("keep_temp", False),
            m4b_bitrate=_normalize_bitrate(data.get("m4b_bitrate"), default="96k"),
            pause_line_ms=data.get("pause_line_ms", 400),
            pause_chapter_ms=data.get("pause_chapter_ms", 2000),
            temp=data.get("temp", 0.7),
            lsd_decode_steps=data.get("lsd_decode_steps", 1),
            noise_clamp=data.get("noise_clamp"),
            default_voice=data.get("default_voice", "alba"),
            default_chapter_preset=data.get("default_chapter_preset", "content-only"),
            default_output_dir=Path(data["default_output_dir"])
            if data.get("default_output_dir")
            else None,
        )


@dataclass
class QueueItem:
    id: str
    job: JobConfig
    status: JobStatus = JobStatus.PENDING
    progress: float = 0.0
    current_chapter: str = ""
    eta_seconds: int = 0
    error_message: str = ""
    output_path: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "job": self.job.to_dict(),
            "status": self.status.value,
            "progress": self.progress,
            "current_chapter": self.current_chapter,
            "eta_seconds": self.eta_seconds,
            "error_message": self.error_message,
            "output_path": self.output_path,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "QueueItem":
        return cls(
            id=data["id"],
            job=JobConfig.from_dict(data["job"]),
            status=JobStatus(data.get("status", "pending")),
            progress=data.get("progress", 0.0),
            current_chapter=data.get("current_chapter", ""),
            eta_seconds=data.get("eta_seconds", 0),
            error_message=data.get("error_message", ""),
            output_path=data.get("output_path", ""),
        )


@dataclass
class Segment:
    """A unit of attributed speech within a chapter.

    Populated by BookNLP during multi-voice tagging.
    When ``Chapter.segments`` is ``None`` the existing single-voice
    paragraph rendering path is used unchanged.
    """

    text: str
    speaker: str = "narrator"  # voice name or character identifier
    index: int = 0  # original position in the chapter


def _default_chapter_tags() -> "ChapterTags":
    from .chapter_classifier import ChapterTags

    return ChapterTags(is_chapter=True)


@dataclass
class Chapter:
    index: int
    title: str
    paragraphs: list[str]
    tags: "ChapterTags" = field(default_factory=_default_chapter_tags)
    toc_index: int = 0
    segments: list[Segment] | None = None  # Populated by BookNLP for multi-voice


@dataclass
class AudioResult:
    chapter_index: int
    title: str
    file_path: Path
    duration_ms: int


@dataclass
class ProcessingConfig:
    voice: str
    ebook_path: Path
    output_path: Path
    pause_line_ms: int
    pause_chapter_ms: int
    workers: int
    m4b_bitrate: str
    keep_temp: bool
    debug_html: bool
    chapter_filters: list["FilterOperation"]
    preview: bool = False
    verbose: bool = False
    tts_model: str = "kyutai/pocket-tts"
    tts_provider: str = "huggingface"
    model_name: str = "pocket-tts"
    elevenlabs_key: str = ""
    elevenlabs_turbo: bool = False
    temp: float = 0.7
    lsd_decode_steps: int = 1
    noise_clamp: float | None = None

    @property
    def epub_path(self) -> Path:
        return self.ebook_path

    @epub_path.setter
    def epub_path(self, value: Path):
        self.ebook_path = value
