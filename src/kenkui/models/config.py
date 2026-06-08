from __future__ import annotations

import multiprocessing
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from ..utils import ApostropheMode
from .common import AttributionExecutionMode, NlpExecutionMode, _normalize_bitrate


class PostProcessingConfig(BaseModel):
    """Broadcast-quality audio effects chain applied per-chapter WAV."""

    enabled: bool = True
    noise_reduce: bool = True
    noise_reduce_prop_decrease: float = 0.8
    highpass_hz: int = 80
    lowshelf_hz: int = 250
    lowshelf_db: float = -3.0
    presence_hz: int = 3500
    presence_db: float = 2.0
    deesser: bool = True
    deesser_hz: int = 6500
    deesser_db: float = -4.0
    compressor_threshold_db: float = -18.0
    compressor_ratio: float = 3.0
    compressor_attack_ms: float = 5.0
    compressor_release_ms: float = 50.0
    limiter_threshold_db: float = -1.0
    autogain: bool = True
    autogain_target_lufs: float = -23.0
    normalize: bool = False
    normalize_target_db: float = -3.0
    normalize_lufs: float | None = None

    def to_dict(self) -> dict:
        return self.model_dump(mode="json", exclude_none=True)

    @classmethod
    def from_dict(cls, data: dict) -> PostProcessingConfig:
        return cls.model_validate(data)


class AppConfig(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="KENKUI_",
        env_nested_delimiter="__",
        arbitrary_types_allowed=True,
        populate_by_name=True,
        extra="ignore",
    )

    name: str = "default"
    workers: int = Field(default_factory=lambda: max(2, multiprocessing.cpu_count() - 2))
    verbose: bool = False
    log_path: Path | None = None
    keep_temp: bool = False
    m4b_bitrate: str = "96k"
    pause_line_ms: int = 800
    pause_chapter_ms: int = 2000
    pause_scene_break_ms: int = 4000
    speak_chapter_titles: bool = True
    pause_before_chapter_title_ms: int = 2000
    pause_after_chapter_title_ms: int = 3000
    temp: float = 0.7
    lsd_decode_steps: int = 1
    noise_clamp: float | None = None
    eos_threshold: float = -4.0
    frames_after_eos: int | None = None
    default_voice: str = "alba"
    default_chapter_preset: str = "content-only"
    default_output_dir: Path | None = None
    nlp_provider: str = "ollama"
    nlp_model: str = "llama3.2"
    nlp_roster_model: str = ""
    nlp_confidence_threshold: int = 0
    nlp_review_model: str = ""
    nlp_attribution_max_quotes_per_call: int = 0
    nlp_attribution_review_confidence: bool = False
    nlp_omit_position_echo: bool = True
    nlp_omit_emotion: bool = True
    nlp_compact_roster: bool = True
    nlp_include_character_descriptions: bool = False
    nlp_descriptions_protagonists_only: bool = True
    credits_enabled: bool = True
    credits_acknowledgements: str = ""
    credits_license: str = ""
    apostrophe_mode: ApostropheMode = ApostropheMode.EXPAND_CONTRACTIONS
    post_processing: PostProcessingConfig = Field(default_factory=PostProcessingConfig)
    server_host: str = "127.0.0.1"
    server_port: int = 45365
    ollama_url: str = "http://localhost:11434"
    nlp_execution_mode: NlpExecutionMode = NlpExecutionMode.LOCAL
    attribution_execution_mode: AttributionExecutionMode = AttributionExecutionMode.LOCAL
    nlp_discovery_method: str = "auto"   # "booknlp" | "llm" | "auto"
    nlp_attribution_provider: str = ""   # falls back to nlp_provider when empty
    nlp_attribution_model: str = ""      # falls back to nlp_model when empty
    nlp_openrouter_attribution_concurrency: int = 4
    cors_origins: list[str] = Field(
        default_factory=lambda: ["tauri://localhost", "http://tauri.localhost"]
    )

    @field_validator("m4b_bitrate", mode="before")
    @classmethod
    def _normalize_m4b_bitrate(cls, v: Any) -> str:
        return _normalize_bitrate(str(v) if v is not None else None, default="96k")

    @field_validator("nlp_openrouter_attribution_concurrency", mode="before")
    @classmethod
    def _clamp_openrouter_attribution_concurrency(cls, v: Any) -> int:
        try:
            value = int(v)
        except (TypeError, ValueError):
            return 4
        return min(32, max(1, value))

    @field_validator("nlp_attribution_max_quotes_per_call", mode="before")
    @classmethod
    def _clamp_nlp_attribution_max_quotes_per_call(cls, v: Any) -> int:
        try:
            value = int(v)
        except (TypeError, ValueError):
            return 0
        return max(0, value)

    def to_dict(self) -> dict[str, Any]:
        return self.model_dump(mode="json", exclude_none=True)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> AppConfig:
        """Create AppConfig from a dict only, bypassing env-var settings sources."""
        class _InitOnly(cls):  # type: ignore[valid-type]
            @classmethod
            def settings_customise_sources(
                cls,
                settings_cls,
                init_settings,
                env_settings,
                dotenv_settings,
                file_secret_settings,
            ):
                return (init_settings,)

        return _InitOnly(**data)  # type: ignore[return-value]
