"""Canonical in-process application API for kenkui clients and HTTP adapters."""

from __future__ import annotations

import dataclasses
import json
import logging
import threading
import time
import tomllib
import uuid
from pathlib import Path
from typing import Any

import tomli_w

from kenkui.config import CONFIG_DIR
from kenkui.models import (
    AppConfig,
    AttributionExecutionMode,
    ChapterSelection,
    CharacterInfo,
    CostStatus,
    JobConfig,
    JobStatus,
    NarrationMode,
    NlpExecutionMode,
    NLPResult,
    QueueItem,
    TTSExecutionMode,
)
from kenkui.models.api import (
    BookParseResponse,
    CastEntry,
    CastResponse,
    ChapterFilterResponse,
    ChapterSummaryModel,
    ConfigResponse,
    HealthResponse,
    HFAuthResponse,
    JobCreateRequest,
    JobResponse,
    MultivoiceStatusResponse,
    NarratorRecommendationResponse,
    OkResponse,
    QueueResponse,
    RosterCandidateListResponse,
    RosterCandidateModel,
    SeriesCharacterModel,
    SeriesListResponse,
    SeriesMatchResponse,
    SeriesModel,
    SimpleCastResponse,
    StatusResponse,
    SuggestCastResponse,
    TaskResponse,
    VoiceListResponse,
    VoicePoolResponse,
    VoiceResponse,
)
from kenkui.progress import ProgressEvent
from kenkui.services.execution_service import get_tts_execution_provider
from kenkui.services.job_service import build_processing_config
from kenkui.services.task_service import Task, TaskRegistry, TaskRunner, TaskType
from kenkui.utils import ApostropheMode

logger = logging.getLogger(__name__)

API_VERSION = "v1"
SERVICE_VERSION = "0.1.0"
QUEUE_FILE = CONFIG_DIR / "queue.toml"
LEGACY_QUEUE_FILE = CONFIG_DIR / "queue.yaml"


def _strip_none(obj: object) -> object:
    if isinstance(obj, dict):
        return {k: _strip_none(v) for k, v in obj.items() if v is not None}
    if isinstance(obj, list):
        return [_strip_none(v) for v in obj if v is not None]
    return obj


def _model_dict(value: Any) -> dict[str, Any] | None:
    if value is None:
        return None
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    if dataclasses.is_dataclass(value):
        return dataclasses.asdict(value)
    if hasattr(value, "__dict__"):
        return dict(value.__dict__)
    return {"value": str(value)}


def _progress_percent(event: ProgressEvent) -> float:
    if event.total_units:
        return max(0.0, min(100.0, (event.completed_units / event.total_units) * 100.0))
    if event.status == "completed":
        return 100.0
    return 0.0


def _chapter_summary(chapter: Any) -> ChapterSummaryModel:
    tags = chapter.tags
    if dataclasses.is_dataclass(tags):
        tags_dict = dataclasses.asdict(tags)
    elif isinstance(tags, dict):
        tags_dict = tags
    else:
        tags_dict = dict(getattr(tags, "__dict__", {}))
    return ChapterSummaryModel(
        index=chapter.index,
        title=chapter.title,
        word_count=chapter.word_count,
        paragraph_count=chapter.paragraph_count,
        toc_index=chapter.toc_index,
        tags=tags_dict,
    )


def job_create_request_to_config(request: JobCreateRequest) -> JobConfig:
    """Convert an API job creation payload into the stable JobConfig model."""
    return JobConfig(
        ebook_path=Path(request.ebook_path),
        voice=request.voice,
        chapter_selection=request.chapter_selection or ChapterSelection(),
        output_path=Path(request.output_path) if request.output_path else None,
        name=request.name or "",
        tts_execution_mode=TTSExecutionMode(request.tts_execution_mode),
        modal_endpoint=request.modal_endpoint,
        modal_environment=request.modal_environment,
        narration_mode=NarrationMode(request.narration_mode),
        speaker_voices=request.speaker_voices or {},
        annotated_chapters_path=Path(request.annotated_chapters_path)
        if request.annotated_chapters_path
        else None,
        chapter_voices=request.chapter_voices or {},
        roster_cache_path=Path(request.roster_cache_path) if request.roster_cache_path else None,
        series_slug=request.series_slug,
        job_nlp_provider=request.job_nlp_provider,
        job_nlp_model=request.job_nlp_model,
        job_temp=request.job_temp,
        job_lsd_decode_steps=request.job_lsd_decode_steps,
        job_noise_clamp=request.job_noise_clamp,
        job_eos_threshold=request.job_eos_threshold,
        job_tts_max_tokens_per_chunk=None,
        job_m4b_bitrate=request.job_m4b_bitrate,
        job_pause_line_ms=request.job_pause_line_ms,
        job_pause_chapter_ms=request.job_pause_chapter_ms,
        job_speak_chapter_titles=request.job_speak_chapter_titles,
        job_pause_before_chapter_title_ms=request.job_pause_before_chapter_title_ms,
        job_pause_after_chapter_title_ms=request.job_pause_after_chapter_title_ms,
        job_frames_after_eos=request.job_frames_after_eos,
        job_apostrophe_mode=ApostropheMode(request.job_apostrophe_mode)
        if request.job_apostrophe_mode
        else None,
        job_post_processing_enabled=request.job_post_processing_enabled,
        job_nlp_execution_mode=NlpExecutionMode(request.job_nlp_execution_mode)
        if request.job_nlp_execution_mode
        else None,
        job_attribution_execution_mode=AttributionExecutionMode(
            request.job_attribution_execution_mode
        )
        if request.job_attribution_execution_mode
        else None,
        job_character_discovery_method=request.job_character_discovery_method,
        job_attribution_provider=request.job_attribution_provider,
        job_attribution_model=request.job_attribution_model,
    )


class KenkuiService:
    """In-process API boundary shared by local clients and HTTP adapters."""

    def __init__(
        self,
        *,
        queue_file: Path = QUEUE_FILE,
        app_config: AppConfig | None = None,
        task_workers: int = 4,
    ) -> None:
        self.queue_file = queue_file
        self.legacy_queue_file = queue_file.with_suffix(".yaml")
        self._items: list[QueueItem] = []
        self._current_id: str | None = None
        self._app_config = app_config or AppConfig()
        self._lock = threading.RLock()
        self._processing_thread: threading.Thread | None = None
        self._running = False
        self._pause_requested = False
        self.task_registry = TaskRegistry()
        self.task_runner = TaskRunner(self.task_registry, max_workers=task_workers)
        from kenkui.services.book_cache import BookCache

        self.book_cache = BookCache()
        self._load()

    def _load(self) -> None:
        if not self.queue_file.exists() and self.legacy_queue_file.exists():
            self._migrate_yaml_to_toml()
        if self.queue_file.exists():
            try:
                data = tomllib.loads(self.queue_file.read_text(encoding="utf-8"))
                self._items = [QueueItem.from_dict(d) for d in data.get("items", [])]
                if "app_config" in data:
                    self._app_config = AppConfig.from_dict(data.get("app_config", {}))
            except Exception as exc:
                logger.warning("Could not load queue file %s: %s", self.queue_file, exc)
        self._reset_stale_processing()

    def _migrate_yaml_to_toml(self) -> None:
        try:
            import yaml

            data = yaml.safe_load(self.legacy_queue_file.read_text())
            if data:
                self.queue_file.parent.mkdir(parents=True, exist_ok=True)
                self.queue_file.write_bytes(tomli_w.dumps(_strip_none(data)).encode("utf-8"))
            self.legacy_queue_file.unlink(missing_ok=True)
        except Exception as exc:
            logger.warning("Could not migrate legacy queue yaml: %s", exc)

    def _save(self) -> None:
        raw = {
            "items": [item.to_dict() for item in self._items],
            "app_config": self._app_config.to_dict(),
        }
        self.queue_file.parent.mkdir(parents=True, exist_ok=True)
        self.queue_file.write_bytes(tomli_w.dumps(_strip_none(raw)).encode("utf-8"))

    @property
    def app_config(self) -> AppConfig:
        return self._app_config

    @app_config.setter
    def app_config(self, config: AppConfig) -> None:
        with self._lock:
            self._app_config = config
            self._save()

    @property
    def is_running(self) -> bool:
        return self._running

    @property
    def current_item(self) -> QueueItem | None:
        with self._lock:
            return next((i for i in self._items if i.status == JobStatus.PROCESSING), None)

    @property
    def pending_items(self) -> list[QueueItem]:
        with self._lock:
            return [i for i in self._items if i.status == JobStatus.PENDING]

    @property
    def completed_items(self) -> list[QueueItem]:
        with self._lock:
            return [i for i in self._items if i.status == JobStatus.COMPLETED]

    @property
    def failed_items(self) -> list[QueueItem]:
        with self._lock:
            return [i for i in self._items if i.status == JobStatus.FAILED]

    @property
    def all_items(self) -> list[QueueItem]:
        with self._lock:
            return list(self._items)

    def health(self) -> HealthResponse:
        return HealthResponse(
            status="healthy",
            version=SERVICE_VERSION,
            server_version=SERVICE_VERSION,
            api_version=API_VERSION,
            capabilities=["local-queue", "single-voice", "multi-voice", "voices", "book-parse"],
        )

    def status(self) -> StatusResponse:
        current = self.current_item
        return StatusResponse(
            status="running" if self.is_running else "idle",
            is_running=self.is_running,
            current_job=current.id if current else None,
        )

    def get_config(self) -> ConfigResponse:
        return ConfigResponse(config=self._app_config.to_dict())

    def update_config(self, config_data: dict[str, Any]) -> OkResponse:
        self.app_config = AppConfig.from_dict(config_data)
        return OkResponse()

    def add_job(self, job: JobConfig) -> QueueItem:
        with self._lock:
            item = QueueItem(
                id=str(uuid.uuid4())[:8],
                job=job,
                status=JobStatus.PENDING,
                execution_provider=job.tts_execution_mode.value,
            )
            self._items.append(item)
            self._save()
            return item

    def add_job_from_request(self, request: JobCreateRequest) -> JobResponse:
        return self.job_response(self.add_job(job_create_request_to_config(request)))

    def get_job_item(self, job_id: str) -> QueueItem | None:
        with self._lock:
            return next((i for i in self._items if i.id == job_id), None)

    def get_job(self, job_id: str) -> JobResponse | None:
        item = self.get_job_item(job_id)
        return self.job_response(item) if item else None

    def remove_job(self, job_id: str) -> bool:
        with self._lock:
            for i, item in enumerate(self._items):
                if item.id == job_id:
                    if item.status == JobStatus.PROCESSING:
                        return False
                    self._items.pop(i)
                    self._save()
                    return True
        return False

    def clear_all_jobs(self) -> OkResponse:
        with self._lock:
            self._items = []
            self._save()
        return OkResponse()

    def queue(self) -> QueueResponse:
        current = self.current_item
        return QueueResponse(
            items=[self.job_response(i) for i in self.all_items],
            current_item=self.job_response(current) if current else None,
            pending_count=len(self.pending_items),
            completed_count=len(self.completed_items),
            failed_count=len(self.failed_items),
        )

    def job_response(self, item: QueueItem) -> JobResponse:
        return JobResponse(
            id=item.id,
            job=item.job.to_dict(),
            status=item.status.value,
            progress=item.progress,
            current_chapter=item.current_chapter,
            eta_seconds=item.eta_seconds,
            error_message=item.error_message,
            output_path=item.output_path,
            started_at=item.started_at,
            completed_at=item.completed_at,
            execution_provider=item.execution_provider,
            remote_job_id=item.remote_job_id,
            estimated_cost_usd=item.estimated_cost_usd,
            actual_cost_usd=item.actual_cost_usd,
            cost_status=item.cost_status.value,
            artifact_uri=item.artifact_uri,
            artifact_source=item.artifact_source,
            provider_status=item.provider_status,
        )

    def _reset_stale_processing(self) -> None:
        changed = False
        with self._lock:
            for item in self._items:
                if item.status == JobStatus.PROCESSING:
                    item.status = JobStatus.PENDING
                    item.progress = 0.0
                    item.current_chapter = ""
                    item.error_message = ""
                    changed = True
            if changed:
                self._save()

    def _next_pending(self) -> QueueItem | None:
        with self._lock:
            return next((i for i in self._items if i.status == JobStatus.PENDING), None)

    def start_job(self, job_id: str) -> bool:
        item = self.get_job_item(job_id)
        if item is None or self.is_running:
            return False
        with self._lock:
            item.status = JobStatus.PROCESSING
            item.started_at = time.time()
            self._current_id = item.id
            self._save()
        self.start_processing()
        return True

    def start_processing(self) -> bool:
        if self._running:
            return False
        self._running = True
        self._processing_thread = threading.Thread(target=self._process_loop, daemon=True)
        self._processing_thread.start()
        return True

    def stop_processing(self) -> None:
        current = self.current_item
        if current is not None:
            try:
                get_tts_execution_provider(current).cancel(current)
            except Exception as exc:
                logger.warning("Could not cancel provider job %s: %s", current.id, exc)
        self._running = False
        if self._processing_thread:
            self._processing_thread.join(timeout=5)

    def pause_job(self, job_id: str) -> bool:
        with self._lock:
            item = self.get_job_item(job_id)
            if item is None or item.status != JobStatus.PROCESSING:
                return False
            self._pause_requested = True
            return True

    def resume_job(self, job_id: str) -> bool:
        with self._lock:
            item = self.get_job_item(job_id)
            if item is None or item.status != JobStatus.PAUSED:
                return False
            item.status = JobStatus.PENDING
            self._pause_requested = False
            self._save()
        self.start_processing()
        return True

    def _process_loop(self) -> None:
        try:
            while self._running:
                item = self._next_pending()
                if item is None:
                    break
                with self._lock:
                    item.status = JobStatus.PROCESSING
                    item.started_at = time.time()
                    self._current_id = item.id
                    self._save()
                self._process_job(item)
        finally:
            self._running = False
            self._current_id = None

    def _process_job(self, item: QueueItem) -> None:
        try:
            cfg = build_processing_config(item.job, self._app_config)
            provider = get_tts_execution_provider(item)
            self.update_job_metadata(
                item.id,
                execution_provider=item.job.tts_execution_mode.value,
                provider_status="starting",
            )
            outcome = provider.execute(
                item=item,
                cfg=cfg,
                app_config=self._app_config,
                progress_callback=lambda event: self.update_progress_from_event(item.id, event),
                metadata_callback=lambda **fields: self.update_job_metadata(item.id, **fields),
                pause_check=lambda: self._pause_requested,
            )
            if outcome.paused:
                with self._lock:
                    item.status = JobStatus.PAUSED
                    self._pause_requested = False
                    self._save()
                return
            self.update_job_metadata(
                item.id,
                remote_job_id=outcome.remote_job_id,
                estimated_cost_usd=outcome.estimated_cost_usd,
                actual_cost_usd=outcome.actual_cost_usd,
                cost_status="final"
                if outcome.actual_cost_usd is not None
                else ("estimated" if outcome.estimated_cost_usd is not None else "none"),
                artifact_uri=outcome.artifact_uri,
                artifact_source=outcome.artifact_source,
                provider_status=outcome.provider_status or ("completed" if outcome.success else "failed"),
            )
            if outcome.success:
                output_path = outcome.output_path or str(cfg.output_path / f"{item.job.name}.m4b")
                self.complete_job(item.id, output_path)
            else:
                self.fail_job(item.id, outcome.error_message or "Conversion failed")
        except Exception as exc:
            logger.exception("Job %s failed: %s", item.id, exc)
            self.fail_job(item.id, str(exc))

    def update_progress_from_event(self, job_id: str, event: ProgressEvent) -> None:
        title = event.message
        if event.active_chapters:
            title = event.active_chapters[0].title or title
        self.update_progress(job_id, _progress_percent(event), title, 0)

    def update_progress(self, job_id: str, progress: float, current_chapter: str, eta_seconds: int) -> None:
        with self._lock:
            item = self.get_job_item(job_id)
            if item is not None:
                item.progress = progress
                item.current_chapter = current_chapter
                item.eta_seconds = eta_seconds
                self._save()

    def update_job_metadata(self, job_id: str, **fields) -> None:
        with self._lock:
            item = self.get_job_item(job_id)
            if item is None:
                return
            for key, value in fields.items():
                if value is None:
                    continue
                if key == "cost_status" and not isinstance(value, CostStatus):
                    value = CostStatus(str(value))
                setattr(item, key, value)
            self._save()

    def complete_job(self, job_id: str, output_path: str = "") -> None:
        with self._lock:
            item = self.get_job_item(job_id)
            if item is not None:
                item.status = JobStatus.COMPLETED
                item.progress = 100.0
                item.current_chapter = ""
                item.output_path = output_path
                item.completed_at = time.time()
                self._save()

    def fail_job(self, job_id: str, error: str) -> None:
        with self._lock:
            item = self.get_job_item(job_id)
            if item is not None:
                item.status = JobStatus.FAILED
                item.error_message = error
                self._save()

    def parse_book(self, ebook_path: str) -> BookParseResponse:
        from kenkui.services.book_service import parse_book

        result = parse_book(ebook_path, self.book_cache)
        return BookParseResponse(
            book_hash=result.book_hash,
            metadata=result.metadata,
            chapters=[_chapter_summary(c) for c in result.chapters],
            total_chapters=result.total_chapters,
            total_word_count=result.total_word_count,
        )

    def filter_chapters(
        self,
        book_hash: str,
        chapter_selection: ChapterSelection,
    ) -> ChapterFilterResponse:
        from kenkui.services.book_service import filter_chapters

        result = filter_chapters(book_hash, chapter_selection, self.book_cache)
        return ChapterFilterResponse(
            included_indices=result.included_indices,
            chapter_count=result.chapter_count,
            estimated_word_count=result.estimated_word_count,
            chapters=[_chapter_summary(c) for c in result.chapters],
        )

    def scan_book(self, ebook_path: str, nlp_model: str | None = None, nlp_provider: str | None = None) -> TaskResponse:
        from kenkui.services.nlp_service import fast_scan

        task = self.task_runner.submit(
            TaskType.FAST_SCAN,
            fast_scan,
            ebook_path=ebook_path,
            nlp_model=nlp_model,
            nlp_provider=nlp_provider,
        )
        return self.task_response(task)

    def task_response(self, task: Task) -> TaskResponse:
        return TaskResponse(
            task_id=task.task_id,
            type=task.type.value,
            status=task.status.value,
            progress=task.progress,
            message=task.message,
            result=_model_dict(task.result),
            error=task.error,
        )

    def get_task(self, task_id: str) -> TaskResponse | None:
        task = self.task_registry.get(task_id)
        return self.task_response(task) if task else None

    def list_voices(
        self,
        *,
        gender: str | None = None,
        accent: str | None = None,
        dataset: str | None = None,
        source: str | None = None,
    ) -> VoiceListResponse:
        from kenkui.services.voice_service import list_voices

        voices = list_voices(gender=gender, accent=accent, dataset=dataset, origin=source)
        return VoiceListResponse(
            voices=[
                VoiceResponse(
                    name=v.voice_id,
                    source=v.origin,
                    gender=v.gender,
                    accent=v.accent,
                    dataset=v.dataset,
                    speaker_id=v.speaker_id,
                    description=v.description,
                    display_label=v.display_label,
                    excluded=not v.pool_enabled,
                )
                for v in voices
            ],
            total=len(voices),
        )

    def get_voice(self, name: str) -> VoiceResponse | None:
        from kenkui.services.voice_service import get_voice

        v = get_voice(name)
        if v is None:
            return None
        return VoiceResponse(
            name=v.voice_id,
            source=v.origin,
            gender=v.gender,
            accent=v.accent,
            dataset=v.dataset,
            speaker_id=v.speaker_id,
            description=v.description,
            display_label=v.display_label,
            excluded=not v.pool_enabled,
        )

    def set_voice_pool_enabled(self, name: str, enabled: bool) -> VoicePoolResponse:
        from kenkui.services.voice_service import set_voice_pool_enabled

        result = set_voice_pool_enabled(name, enabled)
        return VoicePoolResponse(voice_id=result.voice_id, pool_enabled=result.pool_enabled)

    def suggest_cast(
        self,
        roster: list[Any],
        *,
        excluded_voices: list[str] | None = None,
        default_voice: str = "narrator",
    ) -> SuggestCastResponse:
        from kenkui.services.voice_service import suggest_cast

        result = suggest_cast(
            roster=self._character_models(roster),
            excluded_voices=excluded_voices or [],
            default_voice=default_voice,
        )
        return SuggestCastResponse(speaker_voices=result.speaker_voices, warnings=result.warnings)

    def recommend_narrator(
        self,
        roster: list[Any],
        *,
        excluded_voices: list[str] | None = None,
        default_voice: str = "narrator",
    ) -> NarratorRecommendationResponse:
        from kenkui.services.voice_service import top_gender_matched_voice

        voice_name = top_gender_matched_voice(
            self._character_models(roster),
            excluded=excluded_voices or [],
            default_voice=default_voice,
        )
        return NarratorRecommendationResponse(voice_name=voice_name)

    def assign_simple_cast(
        self,
        roster: list[Any],
        *,
        narrator_voice: str,
        male_voice: str,
        female_voice: str,
    ) -> SimpleCastResponse:
        from kenkui.services.voice_service import assign_simple_cast

        return SimpleCastResponse(
            speaker_voices=assign_simple_cast(
                roster=self._character_models(roster),
                narrator_voice=narrator_voice,
                male_voice=male_voice,
                female_voice=female_voice,
            )
        )

    def _character_models(self, roster: list[Any]) -> list[CharacterInfo]:
        result: list[CharacterInfo] = []
        for c in roster:
            if isinstance(c, CharacterInfo):
                result.append(c)
            else:
                result.append(
                    CharacterInfo(
                        character_id=getattr(c, "name", ""),
                        display_name=getattr(c, "name", ""),
                        gender_pronoun=getattr(c, "pronoun", "") or "",
                        quote_count=getattr(c, "quote_count", 0),
                        mention_count=getattr(c, "mention_count", 0),
                    )
                )
        return result

    def list_series(self) -> SeriesListResponse:
        from kenkui.services import series_service

        result = series_service.list_series()
        return SeriesListResponse(series=[self._series_model(e) for e in result.series], total=result.total)

    def list_series_roster_candidates(self) -> RosterCandidateListResponse:
        from kenkui.services import series_service

        result = series_service.list_roster_candidates()
        return RosterCandidateListResponse(
            candidates=[
                RosterCandidateModel(
                    hash=c.hash,
                    title=c.title,
                    path=c.path,
                    speaker_voices=c.speaker_voices,
                    roster_path=c.roster_path,
                )
                for c in result.candidates
            ],
            total=result.total,
        )

    def create_empty_series(self, name: str) -> SeriesModel:
        from kenkui.services import series_service

        return self._series_model(series_service.create_empty_series(name))

    def create_series_from_candidate(self, roster_path: str, name: str) -> SeriesModel:
        from kenkui.services import series_service

        return self._series_model(series_service.build_series_from_candidate(roster_path, name))

    def get_series(self, slug: str) -> SeriesModel:
        from kenkui.services import series_service

        return self._series_model(series_service.load_series(slug))

    def match_series(self, slug: str, fast_result: dict[str, Any]) -> SeriesMatchResponse:
        from kenkui.services import series_service

        result = series_service.match_series_characters(slug, fast_result)
        return SeriesMatchResponse(inherited_voices=result.inherited_voices, pinned=result.pinned)

    def delete_series(self, slug: str) -> bool:
        from kenkui.services import series_service

        return series_service.delete_series(slug)

    def _series_model(self, entry: Any) -> SeriesModel:
        return SeriesModel(
            slug=entry.slug,
            name=entry.name,
            updated_at=getattr(entry, "updated_at", ""),
            characters=[
                SeriesCharacterModel(
                    canonical=c.canonical,
                    aliases=c.aliases,
                    voice=c.voice,
                    gender=c.gender,
                )
                for c in getattr(entry, "characters", [])
            ],
        )

    def get_hf_auth(self) -> HFAuthResponse:
        from kenkui.services.auth_service import get_hf_status

        status = get_hf_status()
        return HFAuthResponse(
            authenticated=status.authenticated,
            username=status.username,
            has_pocket_tts_access=status.has_pocket_tts_access,
        )

    def login_hf(self, token: str) -> HFAuthResponse:
        from kenkui.services.auth_service import get_hf_status, login

        result = login(token)
        if not result.authenticated:
            return HFAuthResponse(
                authenticated=False,
                username=None,
                has_pocket_tts_access=False,
                error=result.error,
            )
        status = get_hf_status()
        return HFAuthResponse(
            authenticated=status.authenticated,
            username=status.username,
            has_pocket_tts_access=status.has_pocket_tts_access,
            error=None,
        )

    def multivoice_status(self) -> MultivoiceStatusResponse:
        spacy_ok = False
        spacy_model = None
        try:
            import spacy.util

            spacy_ok = spacy.util.is_package("en_core_web_sm")
            spacy_model = "en_core_web_sm" if spacy_ok else None
        except Exception:
            pass

        ollama_ok = False
        ollama_url = None
        try:
            import httpx

            url = getattr(self._app_config, "ollama_url", "http://localhost:11434")
            resp = httpx.get(f"{url}/api/tags", timeout=2.0)
            if resp.status_code == 200:
                ollama_ok = True
                ollama_url = url
        except Exception:
            pass

        if spacy_ok and ollama_ok:
            message = "Multi-voice ready"
        elif not spacy_ok and not ollama_ok:
            message = "spaCy and Ollama not available"
        elif not spacy_ok:
            message = "spaCy not available"
        else:
            message = "Ollama not available"
        return MultivoiceStatusResponse(
            spacy_ok=spacy_ok,
            spacy_model=spacy_model,
            ollama_ok=ollama_ok,
            ollama_url=ollama_url,
            message=message,
        )

    def get_cast(self, job_id: str) -> CastResponse | None:
        item = self.get_job_item(job_id)
        if item is None:
            return None
        job = item.job
        if job.annotated_chapters_path is None or not job.annotated_chapters_path.exists():
            return None
        data = json.loads(job.annotated_chapters_path.read_text(encoding="utf-8"))
        nlp_result = NLPResult.from_dict(data)
        entries = []
        roster = getattr(nlp_result, "roster", getattr(nlp_result, "characters", []))
        for char_id, voice_name in (job.speaker_voices or {}).items():
            char_info = next(
                (
                    c
                    for c in roster
                    if getattr(c, "id", None) == char_id
                    or getattr(c, "character_id", None) == char_id
                ),
                None,
            )
            entries.append(
                CastEntry(
                    character_id=char_id,
                    display_name=getattr(char_info, "display_name", char_id),
                    voice_name=voice_name,
                    quote_count=getattr(char_info, "quote_count", 0),
                    mention_count=getattr(char_info, "mention_count", 0),
                    gender_pronoun=getattr(char_info, "gender_pronoun", None),
                )
            )
        return CastResponse(
            job_id=job_id,
            book_name=job.name,
            narration_mode=job.narration_mode.value,
            cast=entries,
        )

    def shutdown(self) -> None:
        self.stop_processing()
        self.task_runner.shutdown(wait=False)


_service: KenkuiService | None = None


def get_service() -> KenkuiService:
    global _service
    if _service is None:
        _service = KenkuiService()
    return _service


def reset_service() -> None:
    global _service
    if _service is not None:
        _service.shutdown()
    _service = None


__all__ = [
    "API_VERSION",
    "KenkuiService",
    "SERVICE_VERSION",
    "get_service",
    "job_create_request_to_config",
    "reset_service",
]

