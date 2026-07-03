from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from ..utils import ApostropheMode
from .config import PostProcessingConfig

if TYPE_CHECKING:
    from ..chapter_filter import FilterOperation
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
    chapter_filters: list[FilterOperation]
    speak_chapter_titles: bool = True
    pause_before_chapter_title_ms: int = 2000
    pause_after_chapter_title_ms: int = 3000
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
    eos_threshold: float = -4.0  # EOS detection threshold; higher (→0) = later cut-off
    frames_after_eos: int | None = None  # Frames after EoS (None = auto from text length)
    tts_max_tokens_per_chunk: int = 50
    pdf_drop_code_blocks: bool = False
    pdf_drop_notes: bool = False
    pdf_drop_asides: bool = False
    pdf_drop_margin_notes: bool = True
    pdf_header_zone_ratio: float = 0.0
    pdf_footer_zone_ratio: float = 0.0
    pdf_force_ocr: bool = False
    # --- Multi-voice fields ---
    speaker_voices: dict[str, str] = field(default_factory=dict)
    annotated_chapters_path: Path | None = None
    roster_cache_path: Path | None = None
    # Per-chapter voice override (chapter-voice mode): str(chapter_index) → voice_name
    chapter_voices: dict[str, str] = field(default_factory=dict)
    # Audio post-processing effects chain
    post_processing: PostProcessingConfig = field(default_factory=PostProcessingConfig)
    # Chapter indices to include when loading from annotated cache.
    # An empty list means "include all chapters in the cache file".
    # Set by job_service.build_processing_config() / Processor._build_config() from
    # JobConfig.chapter_selection.included so that multi-voice jobs respect
    # the user's chapter selection without needing a runtime-injected attribute.
    _included_indices: list[int] = field(default_factory=list)
    apostrophe_mode: ApostropheMode = ApostropheMode.EXPAND_CONTRACTIONS

    @property
    def epub_path(self) -> Path:
        return self.ebook_path

    @epub_path.setter
    def epub_path(self, value: Path):
        self.ebook_path = value
