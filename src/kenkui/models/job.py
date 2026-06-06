from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

from ..utils import ApostropheMode
from .book import ChapterSelection
from .common import (
    AttributionExecutionMode,
    NarrationMode,
    NlpExecutionMode,
    TTSExecutionMode,
    _migrate_speaker_voices_keys,
)


@dataclass
class JobConfig:
    ebook_path: Path
    voice: str = "alba"
    chapter_selection: ChapterSelection = field(default_factory=ChapterSelection)
    output_path: Path | None = None
    name: str = ""
    tts_execution_mode: TTSExecutionMode = TTSExecutionMode.LOCAL
    modal_endpoint: str | None = None
    modal_environment: str | None = None
    # --- Multi-voice fields ---
    narration_mode: NarrationMode = NarrationMode.SINGLE
    speaker_voices: dict[str, str] = field(default_factory=dict)  # char_id → voice
    annotated_chapters_path: Path | None = None  # NLP cache JSON path
    roster_cache_path: Path | None = None  # Path to Stage 2 roster cache (set by wizard fast scan)
    # Per-chapter voice override (chapter-voice mode): str(chapter_index) → voice_name
    chapter_voices: dict[str, str] = field(default_factory=dict)
    # Series slug for cross-book voice consistency (deferred cast assignment)
    series_slug: str | None = None
    # Per-job NLP provider/model (None = inherit from AppConfig)
    job_nlp_provider: str | None = None
    job_nlp_model: str | None = None
    # Per-job quality overrides (None = inherit from AppConfig)
    job_temp: float | None = None
    job_lsd_decode_steps: int | None = None
    job_noise_clamp: float | None = None
    job_eos_threshold: float | None = None
    job_m4b_bitrate: str | None = None
    job_pause_line_ms: int | None = None
    job_pause_chapter_ms: int | None = None
    job_speak_chapter_titles: bool | None = None
    job_pause_before_chapter_title_ms: int | None = None
    job_pause_after_chapter_title_ms: int | None = None
    job_frames_after_eos: int | None = None
    job_apostrophe_mode: ApostropheMode | None = None
    job_post_processing_enabled: bool | None = None
    job_nlp_execution_mode: NlpExecutionMode | None = None
    job_attribution_execution_mode: AttributionExecutionMode | None = None
    # Per-job character discovery / attribution overrides
    job_character_discovery_method: str | None = None  # "booknlp" | "llm" | None (auto)
    job_attribution_provider: str | None = None
    job_attribution_model: str | None = None

    def __post_init__(self):
        if not self.name:
            self.name = self.ebook_path.stem

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "ebook_path": str(self.ebook_path),
            "voice": self.voice,
            "chapter_selection": self.chapter_selection.to_dict(),
            "output_path": str(self.output_path) if self.output_path else None,
            "name": self.name,
            "tts_execution_mode": self.tts_execution_mode.value,
            "narration_mode": self.narration_mode.value,
            "speaker_voices": self.speaker_voices,
            "annotated_chapters_path": str(self.annotated_chapters_path)
            if self.annotated_chapters_path
            else None,
            "roster_cache_path": str(self.roster_cache_path) if self.roster_cache_path else None,
            "chapter_voices": self.chapter_voices,
            "series_slug": self.series_slug,
        }
        for key in ("modal_endpoint", "modal_environment"):
            val = getattr(self, key)
            if val is not None:
                d[key] = val
        # Only include per-job overrides when explicitly set (non-None)
        for key in (
            "job_nlp_provider",
            "job_nlp_model",
            "job_temp",
            "job_lsd_decode_steps",
            "job_noise_clamp",
            "job_eos_threshold",
            "job_m4b_bitrate",
            "job_pause_line_ms",
            "job_pause_chapter_ms",
            "job_speak_chapter_titles",
            "job_pause_before_chapter_title_ms",
            "job_pause_after_chapter_title_ms",
            "job_frames_after_eos",
            "job_apostrophe_mode",
            "job_post_processing_enabled",
            "job_nlp_execution_mode",
            "job_attribution_execution_mode",
            "job_character_discovery_method",
            "job_attribution_provider",
            "job_attribution_model",
        ):
            val = getattr(self, key)
            if val is not None:
                d[key] = val.value if isinstance(val, Enum) else val
        return d

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> JobConfig:
        return cls(
            ebook_path=Path(data["ebook_path"]),
            voice=data.get("voice", "alba"),
            chapter_selection=ChapterSelection.from_dict(data.get("chapter_selection", {})),
            output_path=Path(data["output_path"]) if data.get("output_path") else None,
            name=data.get("name", ""),
            tts_execution_mode=TTSExecutionMode(data.get("tts_execution_mode", "local")),
            modal_endpoint=data.get("modal_endpoint"),
            modal_environment=data.get("modal_environment"),
            narration_mode=NarrationMode(data.get("narration_mode", "single")),
            speaker_voices=_migrate_speaker_voices_keys(data.get("speaker_voices") or {}),
            annotated_chapters_path=Path(data["annotated_chapters_path"])
            if data.get("annotated_chapters_path")
            else None,
            roster_cache_path=Path(data["roster_cache_path"])
            if data.get("roster_cache_path")
            else None,
            chapter_voices=data.get("chapter_voices") or {},
            series_slug=data.get("series_slug"),
            job_nlp_provider=data.get("job_nlp_provider"),
            job_nlp_model=data.get("job_nlp_model"),
            job_temp=data.get("job_temp"),
            job_lsd_decode_steps=data.get("job_lsd_decode_steps"),
            job_noise_clamp=data.get("job_noise_clamp"),
            job_eos_threshold=data.get("job_eos_threshold"),
            job_m4b_bitrate=data.get("job_m4b_bitrate"),
            job_pause_line_ms=data.get("job_pause_line_ms"),
            job_pause_chapter_ms=data.get("job_pause_chapter_ms"),
            job_speak_chapter_titles=data.get("job_speak_chapter_titles"),
            job_pause_before_chapter_title_ms=data.get("job_pause_before_chapter_title_ms"),
            job_pause_after_chapter_title_ms=data.get("job_pause_after_chapter_title_ms"),
            job_frames_after_eos=data.get("job_frames_after_eos"),
            job_apostrophe_mode=ApostropheMode(data["job_apostrophe_mode"])
            if data.get("job_apostrophe_mode")
            else None,
            job_post_processing_enabled=data.get("job_post_processing_enabled"),
            job_nlp_execution_mode=NlpExecutionMode(data["job_nlp_execution_mode"])
            if data.get("job_nlp_execution_mode")
            else None,
            job_attribution_execution_mode=AttributionExecutionMode(data["job_attribution_execution_mode"])
            if data.get("job_attribution_execution_mode")
            else None,
            job_character_discovery_method=data.get("job_character_discovery_method"),
            job_attribution_provider=data.get("job_attribution_provider"),
            job_attribution_model=data.get("job_attribution_model"),
        )


