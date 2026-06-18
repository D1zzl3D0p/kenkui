from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from .book import ChapterSelection


class OkResponse(BaseModel):
    status: str = "ok"
    message: str = ""


class HealthResponse(BaseModel):
    status: str
    version: str
    server_version: str
    api_version: str
    capabilities: list[str]


class JobCreateRequest(BaseModel):
    ebook_path: str
    voice: str = "alba"
    chapter_selection: ChapterSelection | None = None
    output_path: str | None = None
    name: str | None = None
    tts_execution_mode: str = "local"
    modal_endpoint: str | None = None
    modal_environment: str | None = None
    narration_mode: str = "single"
    speaker_voices: dict[str, str] = Field(default_factory=dict)
    annotated_chapters_path: str | None = None
    chapter_voices: dict[str, str] = Field(default_factory=dict)
    roster_cache_path: str | None = None
    series_slug: str | None = None
    job_nlp_provider: str | None = None
    job_nlp_model: str | None = None
    job_temp: float | None = None
    job_lsd_decode_steps: int | None = None
    job_noise_clamp: float | None = None
    job_eos_threshold: float | None = None
    job_post_processing_enabled: bool | None = None
    job_m4b_bitrate: str | None = None
    job_pause_line_ms: int | None = None
    job_pause_chapter_ms: int | None = None
    job_speak_chapter_titles: bool | None = None
    job_pause_before_chapter_title_ms: int | None = None
    job_pause_after_chapter_title_ms: int | None = None
    job_frames_after_eos: int | None = None
    job_apostrophe_mode: str | None = None
    job_nlp_execution_mode: str | None = None
    job_attribution_execution_mode: str | None = None
    job_character_discovery_method: str | None = None
    job_attribution_provider: str | None = None
    job_attribution_model: str | None = None


class JobResponse(BaseModel):
    id: str
    job: dict[str, Any]
    status: str
    progress: float
    current_chapter: str
    eta_seconds: int
    error_message: str
    output_path: str = ""
    started_at: float = 0.0
    completed_at: float = 0.0
    execution_provider: str = ""
    remote_job_id: str = ""
    estimated_cost_usd: float | None = None
    actual_cost_usd: float | None = None
    cost_status: str = "none"
    artifact_uri: str = ""
    artifact_source: str = ""
    provider_status: str = ""


class QueueResponse(BaseModel):
    items: list[JobResponse]
    current_item: JobResponse | None = None
    pending_count: int
    completed_count: int
    failed_count: int


class ConfigResponse(BaseModel):
    config: dict[str, Any]


class StatusResponse(BaseModel):
    status: str
    is_running: bool
    current_job: str | None = None


class BookParseRequest(BaseModel):
    ebook_path: str


class ChapterSummaryModel(BaseModel):
    index: int
    title: str
    word_count: int
    paragraph_count: int
    toc_index: int
    tags: dict[str, Any]


class BookParseResponse(BaseModel):
    book_hash: str
    metadata: dict[str, Any]
    chapters: list[ChapterSummaryModel]
    total_chapters: int
    total_word_count: int


class ChapterFilterRequest(BaseModel):
    book_hash: str
    chapter_selection: ChapterSelection


class ChapterFilterResponse(BaseModel):
    included_indices: list[int]
    chapter_count: int
    estimated_word_count: int
    chapters: list[ChapterSummaryModel]


class BookScanRequest(BaseModel):
    ebook_path: str
    nlp_model: str | None = None
    nlp_provider: str | None = None


class BookAnalyzeRequest(BaseModel):
    ebook_path: str
    nlp_model: str | None = None
    nlp_provider: str | None = None
    discovery_method: str | None = None
    attribution_provider: str | None = None
    attribution_model: str | None = None
    use_cache: bool = True


class ProviderCredentialStatus(BaseModel):
    provider: str
    configured: bool
    default_model: str = ""
    masked_key_hint: str = ""


class ProviderCredentialListResponse(BaseModel):
    providers: list[ProviderCredentialStatus]


class ProviderCredentialUpdateRequest(BaseModel):
    api_key: str | None = None
    default_model: str | None = None


class ProviderModelListResponse(BaseModel):
    provider: str
    models: list[str]


class VoiceResponse(BaseModel):
    name: str
    source: str
    gender: str | None = None
    accent: str | None = None
    dataset: str | None = None
    speaker_id: str | None = None
    description: str
    display_label: str
    excluded: bool


class VoiceListResponse(BaseModel):
    voices: list[VoiceResponse]
    total: int


class AuditionRequest(BaseModel):
    voice_name: str
    text: str | None = None


class DownloadRequest(BaseModel):
    force: bool = False


class FetchRequest(BaseModel):
    repo_id: str | None = None
    patterns: list[str] | None = None


class CharacterInfoModel(BaseModel):
    name: str
    pronoun: str | None = None
    quote_count: int = 0
    mention_count: int = 0


class SuggestCastRequest(BaseModel):
    roster: list[CharacterInfoModel]
    excluded_voices: list[str] = Field(default_factory=list)
    default_voice: str = "narrator"


class SuggestCastResponse(BaseModel):
    speaker_voices: dict[str, str]
    warnings: list[str]


class NarratorRecommendationRequest(BaseModel):
    roster: list[CharacterInfoModel]
    excluded_voices: list[str] = Field(default_factory=list)
    default_voice: str = "narrator"


class NarratorRecommendationResponse(BaseModel):
    voice_name: str


class SimpleCastRequest(BaseModel):
    roster: list[CharacterInfoModel]
    narrator_voice: str
    male_voice: str
    female_voice: str


class SimpleCastResponse(BaseModel):
    speaker_voices: dict[str, str]


class VoicePoolResponse(BaseModel):
    voice_id: str
    pool_enabled: bool


class TaskResponse(BaseModel):
    task_id: str
    type: str
    status: str
    progress: int
    message: str
    result: dict[str, Any] | None = None
    error: str | None = None


class SeriesCharacterModel(BaseModel):
    canonical: str
    aliases: list[str] = Field(default_factory=list)
    voice: str = ""
    gender: str = ""


class SeriesModel(BaseModel):
    slug: str
    name: str
    updated_at: str = ""
    characters: list[SeriesCharacterModel] = Field(default_factory=list)


class SeriesListResponse(BaseModel):
    series: list[SeriesModel]
    total: int


class RosterCandidateModel(BaseModel):
    hash: str
    title: str
    path: str
    speaker_voices: dict[str, str] = Field(default_factory=dict)
    roster_path: str


class RosterCandidateListResponse(BaseModel):
    candidates: list[RosterCandidateModel]
    total: int


class CreateEmptySeriesRequest(BaseModel):
    name: str


class CreateSeriesFromCandidateRequest(BaseModel):
    name: str
    roster_path: str


class SeriesMatchRequest(BaseModel):
    fast_result: dict[str, Any]


class SeriesMatchResponse(BaseModel):
    inherited_voices: dict[str, str]
    pinned: list[str]


class HFTokenRequest(BaseModel):
    token: str


class HFAuthResponse(BaseModel):
    authenticated: bool
    username: str | None = None
    has_pocket_tts_access: bool
    error: str | None = None


class MultivoiceStatusResponse(BaseModel):
    spacy_ok: bool
    spacy_model: str | None
    ollama_ok: bool
    ollama_url: str | None
    message: str


class CastEntry(BaseModel):
    character_id: str
    display_name: str
    voice_name: str
    quote_count: int
    mention_count: int
    gender_pronoun: str | None = None


class CastResponse(BaseModel):
    job_id: str
    book_name: str
    narration_mode: str
    cast: list[CastEntry]
