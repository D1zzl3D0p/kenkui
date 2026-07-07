"""Canonical in-process application API for kenkui clients and HTTP adapters."""

from __future__ import annotations

import dataclasses
import importlib.metadata
import json
import logging
import os
import uuid
from pathlib import Path
from typing import Any

from kenkui.config import (
    CONFIG_DIR,
    ProviderCredentials,
    inject_provider_env_vars,
    load_provider_credentials,
    save_provider_credentials,
)
from kenkui.models import (
    AppConfig,
    AttributionExecutionMode,
    ChapterSelection,
    CharacterInfo,
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
    ProviderCredentialListResponse,
    ProviderCredentialStatus,
    ProviderModelListResponse,
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
from kenkui.services.execution_service import (
    get_tts_execution_provider,
)
from kenkui.services.job_executor import JobExecutor, progress_update_from_args
from kenkui.services.provider_service import list_provider_models as _list_provider_models
from kenkui.services.provider_service import (
    validate_provider_credentials as _validate_provider_credentials,
)
from kenkui.services.queue_manager import QueueManager
from kenkui.services.task_coordinator import TaskCoordinator
from kenkui.services.task_service import Task, TaskRegistry, TaskRunner, TaskType
from kenkui.utils import ApostropheMode

logger = logging.getLogger(__name__)

API_VERSION = "v1"
try:
    SERVICE_VERSION = importlib.metadata.version("kenkui")
except importlib.metadata.PackageNotFoundError:
    SERVICE_VERSION = "0.0.0+unknown"
QUEUE_FILE = CONFIG_DIR / "queue.toml"
LEGACY_QUEUE_FILE = CONFIG_DIR / "queue.yaml"
PROVIDER_NAMES = ("anthropic", "openai", "google", "openrouter")
MODEL_PROVIDER_NAMES = PROVIDER_NAMES + ("ollama",)


def _strip_none(obj: object) -> object:
    if isinstance(obj, dict):
        return {k: _strip_none(v) for k, v in obj.items() if v is not None}
    if isinstance(obj, list):
        return [_strip_none(v) for v in obj if v is not None]
    return obj


def _model_dict(value: Any) -> dict[str, Any] | None:
    if value is None:
        return None
    if isinstance(value, dict):
        # Task functions commonly return API-ready dictionaries. Preserve that
        # shape so clients receive `result.characters` instead of a stringified
        # wrapper like `{value: "{'characters': ...}"}`.
        return _strip_none(value)  # type: ignore[return-value]
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    if dataclasses.is_dataclass(value):
        return dataclasses.asdict(value)
    if hasattr(value, "__dict__"):
        return dict(value.__dict__)
    return {"value": str(value)}


def _merge_dict(base: dict[str, Any], patch: dict[str, Any]) -> dict[str, Any]:
    result = dict(base)
    for key, value in patch.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _merge_dict(result[key], value)
        else:
            result[key] = value
    return result


def _masked_key_hint(api_key: str) -> str:
    if not api_key:
        return ""
    if len(api_key) <= 8:
        return f"{api_key[:2]}...{api_key[-2:]}"
    return f"{api_key[:4]}...{api_key[-4:]}"


# Re-exported for backwards compatibility; the canonical implementation now
# lives in kenkui.services.job_executor.
_progress_update_from_args = progress_update_from_args


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
        job_tts_max_tokens_per_chunk=request.job_tts_max_tokens_per_chunk,
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
        job_number_normalization=request.job_number_normalization,
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
        # Register runtimes against the injected/default config BEFORE loading
        # the queue file.  QueueManager.load() may overwrite app_config from a
        # persisted queue.toml (e.g. one with modal_enabled=True), which would
        # cause RuntimeRegistrationError at construction for previously-valid
        # states.  The original KenkuiService semantics were: register first,
        # then load — preserve that ordering here.
        from kenkui.services.runtime_service import register_configured_runtimes
        register_configured_runtimes(app_config or AppConfig())
        self._queue = QueueManager(
            queue_file=queue_file,
            app_config=app_config or AppConfig(),
        )
        # Shared reentrant lock: the queue and the job-orchestration flags below
        # were historically mutated atomically under one RLock. JobExecutor and
        # this facade therefore share the QueueManager lock rather than adding a
        # second lock (which would create a lock-ordering hazard).
        self._lock = self._queue.lock
        self._jobs = JobExecutor(
            queue=self._queue,
            resolve_provider=lambda item: get_tts_execution_provider(item),
        )
        self._tasks = TaskCoordinator(max_workers=task_workers)
        from kenkui.services.book_cache import BookCache

        self.book_cache = BookCache()

    @property
    def task_registry(self) -> TaskRegistry:
        return self._tasks.registry

    @property
    def task_runner(self) -> TaskRunner:
        return self._tasks.runner

    @property
    def queue_file(self) -> Path:
        return self._queue.queue_file

    def _save(self) -> None:
        self._queue.save()

    def _reset_stale_processing(self) -> None:
        self._queue.reset_stale_processing()

    @property
    def _app_config(self) -> AppConfig:
        return self._queue.app_config

    @property
    def app_config(self) -> AppConfig:
        return self._queue.app_config

    @app_config.setter
    def app_config(self, config: AppConfig) -> None:
        self._queue.app_config = config

    @property
    def is_running(self) -> bool:
        return self._jobs.running

    @property
    def _running(self) -> bool:
        return self._jobs.running

    @_running.setter
    def _running(self, value: bool) -> None:
        self._jobs.running = value

    def _process_job(self, item: QueueItem) -> None:
        self._jobs.process_job(item)

    def _process_loop(self) -> None:
        self._jobs.process_loop()

    @property
    def current_item(self) -> QueueItem | None:
        return self._queue.current_item

    def _processing_item(self) -> QueueItem | None:
        return self._queue.processing_item()

    @property
    def pending_items(self) -> list[QueueItem]:
        return self._queue.pending_items

    @property
    def completed_items(self) -> list[QueueItem]:
        return self._queue.completed_items

    @property
    def failed_items(self) -> list[QueueItem]:
        return self._queue.failed_items

    @property
    def all_items(self) -> list[QueueItem]:
        return self._queue.all_items

    def health(self) -> HealthResponse:
        return HealthResponse(
            status="healthy",
            version=SERVICE_VERSION,
            server_version=SERVICE_VERSION,
            api_version=API_VERSION,
            capabilities=[
                "local-queue",
                "single-voice",
                "multi-voice",
                "voices",
                "book-parse",
                "provider-models",
                "provider-credentials",
            ],
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

    def patch_config(self, config_patch: dict[str, Any]) -> ConfigResponse:
        merged = _merge_dict(self._app_config.to_dict(), config_patch)
        self.app_config = AppConfig.from_dict(merged)
        return self.get_config()

    def _provider_env_key(self, provider: str) -> str:
        from kenkui.config import _KENKUI_PROVIDER_ENV_VARS, _PROVIDER_ENV_VARS

        return os.environ.get(_KENKUI_PROVIDER_ENV_VARS.get(provider, ""), "") or os.environ.get(
            _PROVIDER_ENV_VARS.get(provider, ""),
            "",
        )

    def _provider_credential_status(
        self,
        provider: str,
        credentials: dict[str, ProviderCredentials] | None = None,
    ) -> ProviderCredentialStatus:
        credentials = credentials if credentials is not None else load_provider_credentials()
        stored = credentials.get(provider)
        api_key = self._provider_env_key(provider) or (stored.api_key if stored else "")
        return ProviderCredentialStatus(
            provider=provider,
            configured=bool(api_key),
            default_model=stored.default_model if stored else "",
            masked_key_hint=_masked_key_hint(api_key),
        )

    def list_provider_credentials(self) -> ProviderCredentialListResponse:
        credentials = load_provider_credentials()
        return ProviderCredentialListResponse(
            providers=[
                self._provider_credential_status(provider, credentials)
                for provider in PROVIDER_NAMES
            ]
        )

    def update_provider_credentials(
        self,
        provider: str,
        *,
        api_key: str | None = None,
        default_model: str | None = None,
    ) -> ProviderCredentialStatus:
        provider = provider.lower().strip()
        if provider not in PROVIDER_NAMES:
            raise KeyError(provider)
        credentials = load_provider_credentials()
        current = credentials.get(provider, ProviderCredentials(api_key="", default_model=""))
        next_api_key = current.api_key
        if api_key is not None and api_key.strip():
            next_api_key = api_key.strip()
        next_default_model = current.default_model if default_model is None else default_model.strip()
        credentials[provider] = ProviderCredentials(
            api_key=next_api_key,
            default_model=next_default_model,
        )
        save_provider_credentials(credentials)
        inject_provider_env_vars(credentials)
        return self._provider_credential_status(provider, credentials)

    def delete_provider_credentials(self, provider: str) -> OkResponse:
        provider = provider.lower().strip()
        if provider not in PROVIDER_NAMES:
            raise KeyError(provider)
        credentials = load_provider_credentials()
        removed = credentials.pop(provider, None)
        save_provider_credentials(credentials)
        if removed is not None:
            from kenkui.config import _KENKUI_PROVIDER_ENV_VARS, _PROVIDER_ENV_VARS

            standard_var = _PROVIDER_ENV_VARS.get(provider, "")
            kenkui_var = _KENKUI_PROVIDER_ENV_VARS.get(provider, "")
            if (
                standard_var
                and not os.environ.get(kenkui_var, "")
                and os.environ.get(standard_var, "") == removed.api_key
            ):
                os.environ.pop(standard_var, None)
        return OkResponse()

    def list_provider_models(self, provider: str) -> ProviderModelListResponse:
        provider = provider.lower().strip()
        if provider not in MODEL_PROVIDER_NAMES:
            raise KeyError(provider)
        return _list_provider_models(provider)

    def test_provider_credentials(self, provider: str) -> OkResponse:
        provider = provider.lower().strip()
        if provider not in PROVIDER_NAMES:
            raise KeyError(provider)
        credentials = load_provider_credentials()
        return _validate_provider_credentials(provider, credentials=credentials)

    def add_job(self, job: JobConfig) -> QueueItem:
        item = QueueItem(
            id=str(uuid.uuid4())[:8],
            job=job,
            status=JobStatus.PENDING,
            execution_provider=job.tts_execution_mode.value,
        )
        return self._queue.add(item)

    def add_job_from_request(self, request: JobCreateRequest) -> JobResponse:
        return self.job_response(self.add_job(job_create_request_to_config(request)))

    def get_job_item(self, job_id: str) -> QueueItem | None:
        return self._queue.get(job_id)

    def get_job(self, job_id: str) -> JobResponse | None:
        item = self.get_job_item(job_id)
        return self.job_response(item) if item else None

    def remove_job(self, job_id: str) -> bool:
        return self._queue.remove(job_id)

    def cancel_job(self, job_id: str) -> bool:
        return self._jobs.cancel_job(job_id)

    def clear_all_jobs(self) -> OkResponse:
        self._queue.clear_all()
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

    def start_job(self, job_id: str) -> bool:
        return self._jobs.start_job(job_id)

    def start_processing(self) -> bool:
        return self._jobs.start_processing()

    def stop_processing(self) -> None:
        self._jobs.stop_processing()

    def pause_job(self, job_id: str) -> bool:
        return self._jobs.pause_job(job_id)

    def resume_job(self, job_id: str) -> bool:
        return self._jobs.resume_job(job_id)

    def retry_job(self, job_id: str) -> bool:
        return self._jobs.retry_job(job_id)

    def update_progress_from_event(self, job_id: str, event: ProgressEvent) -> None:
        progress, current_chapter, eta_seconds = _progress_update_from_args(event)
        self.update_progress(job_id, progress, current_chapter, eta_seconds)

    def update_progress(self, job_id: str, progress: float, current_chapter: str, eta_seconds: int) -> None:
        self._queue.update_progress(job_id, progress, current_chapter, eta_seconds)

    def update_job_metadata(self, job_id: str, **fields) -> None:
        self._queue.update_job_metadata(job_id, **fields)

    def complete_job(self, job_id: str, output_path: str = "") -> None:
        self._queue.complete(job_id, output_path)

    def fail_job(self, job_id: str, error: str) -> None:
        self._queue.fail(job_id, error)

    def parse_book(self, ebook_path: str) -> BookParseResponse:
        from kenkui.services.book_service import parse_book

        logger.info("Parsing ebook ebook_path=%s", ebook_path)
        result = parse_book(ebook_path, self.book_cache)
        logger.info(
            "Parsed ebook ebook_path=%s book_hash=%s chapters=%d words=%d",
            ebook_path,
            result.book_hash,
            result.total_chapters,
            result.total_word_count,
        )
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

        logger.info(
            "Filtering chapters book_hash=%s preset=%s included=%d excluded=%d",
            book_hash,
            chapter_selection.preset.value,
            len(chapter_selection.included or []),
            len(chapter_selection.excluded or []),
        )
        result = filter_chapters(book_hash, chapter_selection, self.book_cache)
        logger.info(
            "Filtered chapters book_hash=%s included=%d estimated_words=%d",
            book_hash,
            len(result.included_indices),
            result.estimated_word_count,
        )
        return ChapterFilterResponse(
            included_indices=result.included_indices,
            chapter_count=result.chapter_count,
            estimated_word_count=result.estimated_word_count,
            chapters=[_chapter_summary(c) for c in result.chapters],
        )

    def scan_book(self, ebook_path: str, nlp_model: str | None = None, nlp_provider: str | None = None) -> TaskResponse:
        from kenkui.services.nlp_service import fast_scan

        task = self._tasks.submit(
            TaskType.FAST_SCAN,
            fast_scan,
            ebook_path=ebook_path,
            nlp_model=nlp_model,
            nlp_provider=nlp_provider,
        )
        logger.info(
            "Submitted scan task task_id=%s ebook_path=%s provider=%s model=%s",
            task.task_id,
            ebook_path,
            nlp_provider or self._app_config.nlp_provider,
            nlp_model or self._app_config.nlp_model,
        )
        return self.task_response(task)

    def analysis_cache_candidates(self, ebook_path: str) -> dict[str, Any]:
        """List reusable NLP cache files for a book with user-facing parameters.

        The GUI uses this as a preflight so cache reuse is an explicit user
        decision rather than an invisible shortcut around local discovery and
        attribution work.
        """
        import kenkui.nlp as nlp_module
        from kenkui.nlp._cache import list_caches as _list_step_caches

        ebook = Path(ebook_path)
        if not ebook.exists():
            raise FileNotFoundError(f"Ebook not found: {ebook_path}")
        bh = nlp_module.book_hash(ebook)
        cache_dir = nlp_module._get_config_dir() / "nlp_cache"
        candidates: list[dict[str, Any]] = []
        seen: set[str] = set()

        def _counts(data: dict[str, Any]) -> tuple[int, int, int]:
            raw_characters = data.get("characters")
            raw_chapters = data.get("chapters")
            characters = raw_characters if isinstance(raw_characters, list) else []
            chapters = raw_chapters if isinstance(raw_chapters, list) else []
            quote_count = sum(
                int(item.get("quote_count") or 0)
                for item in characters
                if isinstance(item, dict)
            )
            return len(characters), len(chapters), quote_count

        def _add(
            *,
            path: Path,
            step: str,
            provider: str,
            model: str,
            method: str = "",
            created_at: str = "",
            description: str = "",
            data: dict[str, Any] | None = None,
        ) -> None:
            if path.name in seen:
                return
            seen.add(path.name)
            character_count, chapter_count, quote_count = _counts(data or {})
            candidates.append({
                "cache_id": path.name,
                "step": step,
                "provider": provider,
                "model": model,
                "method": method,
                "created_at": created_at,
                "description": description,
                "path": str(path),
                "character_count": character_count,
                "chapter_count": chapter_count,
                "quote_count": quote_count,
            })

        for meta in _list_step_caches(ebook):
            data = {}
            try:
                data = json.loads(meta.path.read_text(encoding="utf-8"))
            except Exception:
                data = {}
            _add(
                path=meta.path,
                step=meta.step,
                provider=meta.tool,
                model=meta.model,
                method="",
                created_at=meta.created_at.isoformat(),
                description=meta.description or f"{meta.step.title()} cache · {meta.tool} · {meta.model}",
                data=data,
            )

        for meta in nlp_module.list_cached_rosters(ebook):
            data = getattr(meta, "_data", {})
            _add(
                path=meta.path,
                step="roster",
                provider=meta.provider,
                model=meta.model,
                method=meta.method,
                created_at=meta.created_at,
                description=meta.description or f"Roster cache · {meta.method} · {meta.provider} · {meta.model}",
                data=data,
            )

        if cache_dir.exists():
            prefix = f"{bh}-"
            for path in cache_dir.glob(f"{bh}-*.json"):
                name = path.name
                if name in seen or "-roster-" in name or "-extraction-" in name or "-attribution-" in name:
                    continue
                try:
                    data = json.loads(path.read_text(encoding="utf-8"))
                except Exception:
                    continue
                provider = str(data.get("provider") or "")
                model = str(data.get("model") or "")
                if not provider:
                    stem = path.stem[len(prefix):]
                    provider = stem.split("-", 1)[0] if stem else ""
                created_at = str(data.get("created_at") or "")
                _add(
                    path=path,
                    step="attribution",
                    provider=provider,
                    model=model,
                    created_at=created_at,
                    description=f"Full attribution cache · {provider or 'unknown'} · {model or 'unknown model'}",
                    data=data,
                )

        candidates.sort(key=lambda item: str(item.get("created_at") or ""), reverse=True)
        return {"book_hash": bh, "candidates": candidates}

    def _run_full_analysis(
        self,
        *,
        ebook_path: str,
        nlp_model: str | None = None,
        nlp_provider: str | None = None,
        discovery_method: str | None = None,
        attribution_provider: str | None = None,
        attribution_model: str | None = None,
        use_cache: bool = True,
        progress_callback=None,
    ) -> dict[str, Any]:
        from kenkui.nlp import attribution_cache_path, get_cached_result, list_cached_rosters
        from kenkui.services.nlp_service import full_analysis

        ebook = Path(ebook_path)
        request_attribution_provider = attribution_provider if attribution_provider is not None else nlp_provider
        request_attribution_model = attribution_model if attribution_model is not None else nlp_model
        effective_extraction_provider = nlp_provider or self._app_config.nlp_provider
        effective_extraction_model = nlp_model or self._app_config.nlp_model
        effective_attribution_provider = (
            request_attribution_provider
            or self._app_config.nlp_attribution_provider
            or effective_extraction_provider
        )
        effective_attribution_model = (
            request_attribution_model
            or self._app_config.nlp_attribution_model
            or effective_extraction_model
        )
        cache_hit = use_cache and (
            get_cached_result(
                ebook,
                provider=effective_attribution_provider,
                model=effective_attribution_model,
            )
            is not None
        )

        def _extraction_progress(percent: int, message: str) -> None:
            if progress_callback is not None:
                progress_callback(int(percent * 0.45), f"Discovery: {message}")

        def _attribution_progress(percent: int, message: str) -> None:
            if progress_callback is not None:
                progress_callback(45 + int(percent * 0.55), f"Attribution: {message}")

        result = full_analysis(
            ebook_path=ebook_path,
            nlp_model=nlp_model,
            nlp_provider=nlp_provider,
            discovery_method=discovery_method,
            attribution_provider=request_attribution_provider,
            attribution_model=request_attribution_model,
            extraction_progress_callback=_extraction_progress,
            attribution_progress_callback=_attribution_progress,
            use_cache=use_cache,
        )
        annotated_path = attribution_cache_path(
            ebook,
            provider=effective_attribution_provider,
            model=effective_attribution_model,
        )
        roster_cache_path = None
        for meta in list_cached_rosters(ebook):
            if (
                meta.provider == effective_extraction_provider
                and meta.model == effective_extraction_model
                and (discovery_method is None or meta.method == discovery_method)
            ):
                roster_cache_path = str(meta.path)
                break
        return {
            "characters": [character.to_dict() for character in result.characters],
            "book_hash": result.book_hash,
            "annotated_chapters_path": str(annotated_path),
            "roster_cache_path": roster_cache_path,
            "nlp_provider": effective_extraction_provider,
            "nlp_model": effective_extraction_model,
            "attribution_provider": effective_attribution_provider,
            "attribution_model": effective_attribution_model,
            "cache_status": "hit" if cache_hit else "miss",
        }

    def analyze_book(
        self,
        ebook_path: str,
        *,
        nlp_model: str | None = None,
        nlp_provider: str | None = None,
        discovery_method: str | None = None,
        attribution_provider: str | None = None,
        attribution_model: str | None = None,
        use_cache: bool = True,
    ) -> TaskResponse:
        task = self._tasks.submit(
            TaskType.FULL_ANALYSIS,
            self._run_full_analysis,
            ebook_path=ebook_path,
            nlp_model=nlp_model,
            nlp_provider=nlp_provider,
            discovery_method=discovery_method,
            attribution_provider=attribution_provider,
            attribution_model=attribution_model,
            use_cache=use_cache,
        )
        logger.info(
            "Submitted analysis task task_id=%s ebook_path=%s provider=%s model=%s",
            task.task_id,
            ebook_path,
            nlp_provider or self._app_config.nlp_provider,
            nlp_model or self._app_config.nlp_model,
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
        task = self._tasks.get(task_id)
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
        nlp_mode = getattr(self._app_config, "nlp_execution_mode", NlpExecutionMode.LOCAL)
        if nlp_mode == NlpExecutionMode.MODAL:
            spacy_ok = True
            spacy_model = "modal"
        else:
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

        if nlp_mode == NlpExecutionMode.MODAL:
            message = "Multi-voice ready via Modal NLP runtime"
        elif spacy_ok and ollama_ok:
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
        self._tasks.shutdown(wait=False)


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
