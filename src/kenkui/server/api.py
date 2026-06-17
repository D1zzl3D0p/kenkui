"""FastAPI adapter for the canonical kenkui in-process service API."""

from __future__ import annotations

import logging
import time
import uuid
from contextlib import asynccontextmanager

from kenkui.models import (
    AuditionRequest,
    BookAnalyzeRequest,
    BookParseRequest,
    BookParseResponse,
    BookScanRequest,
    CastResponse,
    ChapterFilterRequest,
    ChapterFilterResponse,
    ConfigResponse,
    CreateEmptySeriesRequest,
    CreateSeriesFromCandidateRequest,
    DownloadRequest,
    FetchRequest,
    HealthResponse,
    HFAuthResponse,
    HFTokenRequest,
    JobCreateRequest,
    JobResponse,
    JobStatus,
    MultivoiceStatusResponse,
    NarratorRecommendationRequest,
    NarratorRecommendationResponse,
    OkResponse,
    ProviderCredentialListResponse,
    ProviderCredentialStatus,
    ProviderCredentialUpdateRequest,
    ProviderModelListResponse,
    QueueResponse,
    RosterCandidateListResponse,
    SeriesListResponse,
    SeriesMatchRequest,
    SeriesMatchResponse,
    SeriesModel,
    SimpleCastRequest,
    SimpleCastResponse,
    StatusResponse,
    SuggestCastRequest,
    SuggestCastResponse,
    TaskResponse,
    VoiceListResponse,
    VoicePoolResponse,
    VoiceResponse,
)
from kenkui.services.application_service import get_service

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app):
    service = get_service()
    logger.info(
        "Server startup queue_file=%s queue_items=%d cors_origins=%d",
        service.queue_file,
        len(service.all_items),
        len(service.app_config.cors_origins),
    )
    yield
    logger.info("Server shutdown")
    get_service().shutdown()


def create_app():
    try:
        from fastapi import FastAPI, HTTPException
        from fastapi.middleware.cors import CORSMiddleware
    except ImportError as exc:
        raise RuntimeError("Install kenkui[server] to use the FastAPI adapter.") from exc

    app = FastAPI(title="kenkui API", version="0.1.0", lifespan=lifespan)
    service = get_service()
    app.add_middleware(
        CORSMiddleware,
        allow_origins=service.app_config.cors_origins,
        allow_methods=["*"],
        allow_headers=["*"],
        allow_credentials=False,
    )

    @app.middleware("http")
    async def log_requests(request, call_next):
        request_id = request.headers.get("x-request-id") or uuid.uuid4().hex[:12]
        started = time.perf_counter()
        response = None
        try:
            response = await call_next(request)
            return response
        except Exception:
            logger.exception(
                "HTTP request failed request_id=%s method=%s path=%s client=%s",
                request_id,
                request.method,
                request.url.path,
                request.client.host if request.client else "",
            )
            raise
        finally:
            duration_ms = (time.perf_counter() - started) * 1000.0
            status = response.status_code if response is not None else 500
            logger.info(
                "HTTP request request_id=%s method=%s path=%s status=%s duration_ms=%.1f client=%s",
                request_id,
                request.method,
                request.url.path,
                status,
                duration_ms,
                request.client.host if request.client else "",
            )
            if response is not None:
                response.headers["X-Request-Id"] = request_id

    @app.get("/health", response_model=HealthResponse)
    @app.get("/v1/health", response_model=HealthResponse)
    def health_check():
        return get_service().health()

    @app.get("/queue", response_model=QueueResponse)
    @app.get("/v1/queue", response_model=QueueResponse)
    def get_queue():
        return get_service().queue()

    @app.post("/queue", response_model=JobResponse)
    @app.post("/v1/queue", response_model=JobResponse)
    def add_job(request: JobCreateRequest):
        return get_service().add_job_from_request(request)

    @app.get("/queue/{job_id}", response_model=JobResponse)
    @app.get("/v1/queue/{job_id}", response_model=JobResponse)
    def get_job(job_id: str):
        response = get_service().get_job(job_id)
        if response is None:
            raise HTTPException(status_code=404, detail="Job not found")
        return response

    @app.delete("/queue/{job_id}", response_model=OkResponse)
    @app.delete("/v1/queue/{job_id}", response_model=OkResponse)
    def remove_job(job_id: str):
        service = get_service()
        item = service.get_job_item(job_id)
        if item is None:
            raise HTTPException(status_code=404, detail="Job not found")
        if item.status in {JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.CANCELLED}:
            if not service.remove_job(job_id):
                raise HTTPException(status_code=400, detail="Cannot remove job")
        else:
            if not service.cancel_job(job_id):
                raise HTTPException(status_code=400, detail="Cannot cancel job")
        return OkResponse()

    @app.post("/queue/{job_id}/start")
    @app.post("/v1/queue/{job_id}/start")
    def start_job(job_id: str):
        if not get_service().start_job(job_id):
            raise HTTPException(status_code=400, detail="Cannot start job")
        return {"status": "started", "job_id": job_id}

    @app.post("/queue/{job_id}/stop")
    @app.post("/v1/queue/{job_id}/stop")
    def stop_job(job_id: str):
        if not get_service().cancel_job(job_id):
            raise HTTPException(status_code=400, detail="Cannot cancel job")
        return {"status": "cancelled", "job_id": job_id}

    @app.post("/queue/{job_id}/cancel", response_model=OkResponse)
    @app.post("/v1/queue/{job_id}/cancel", response_model=OkResponse)
    def cancel_job(job_id: str):
        if not get_service().cancel_job(job_id):
            raise HTTPException(status_code=400, detail="Cannot cancel job")
        return OkResponse()

    @app.post("/queue/{job_id}/pause", response_model=OkResponse)
    @app.post("/v1/queue/{job_id}/pause", response_model=OkResponse)
    def pause_job(job_id: str):
        if not get_service().pause_job(job_id):
            raise HTTPException(status_code=400, detail="Job is not currently processing")
        return OkResponse()

    @app.post("/queue/{job_id}/resume", response_model=OkResponse)
    @app.post("/v1/queue/{job_id}/resume", response_model=OkResponse)
    def resume_job(job_id: str):
        if not get_service().resume_job(job_id):
            raise HTTPException(status_code=400, detail="Job is not paused")
        return OkResponse()

    @app.post("/queue/start")
    @app.post("/v1/queue/start")
    def start_processing():
        if not get_service().start_processing():
            raise HTTPException(status_code=400, detail="Processing already in progress")
        return {"status": "started"}

    @app.post("/queue/stop")
    @app.post("/v1/queue/stop")
    def stop_processing():
        get_service().stop_processing()
        return {"status": "stopped"}

    @app.delete("/queue", response_model=OkResponse)
    @app.delete("/v1/queue", response_model=OkResponse)
    def clear_queue():
        return get_service().clear_all_jobs()

    @app.get("/status", response_model=StatusResponse)
    @app.get("/v1/status", response_model=StatusResponse)
    def get_status():
        return get_service().status()

    @app.get("/config", response_model=ConfigResponse)
    @app.get("/v1/config", response_model=ConfigResponse)
    def get_config():
        return get_service().get_config()

    @app.put("/config", response_model=OkResponse)
    @app.put("/v1/config", response_model=OkResponse)
    def update_config(config_data: dict):
        try:
            return get_service().update_config(config_data)
        except Exception as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    @app.patch("/config", response_model=ConfigResponse)
    @app.patch("/v1/config", response_model=ConfigResponse)
    def patch_config(config_data: dict):
        try:
            return get_service().patch_config(config_data)
        except Exception as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    @app.get("/provider-credentials", response_model=ProviderCredentialListResponse)
    @app.get("/v1/provider-credentials", response_model=ProviderCredentialListResponse)
    def list_provider_credentials():
        return get_service().list_provider_credentials()

    @app.get("/provider-models/{provider}", response_model=ProviderModelListResponse)
    @app.get("/v1/provider-models/{provider}", response_model=ProviderModelListResponse)
    def list_provider_models(provider: str):
        try:
            return get_service().list_provider_models(provider)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=f"Provider not found: {provider}") from exc

    @app.put("/provider-credentials/{provider}", response_model=ProviderCredentialStatus)
    @app.put("/v1/provider-credentials/{provider}", response_model=ProviderCredentialStatus)
    def update_provider_credentials(provider: str, request: ProviderCredentialUpdateRequest):
        try:
            return get_service().update_provider_credentials(
                provider,
                api_key=request.api_key,
                default_model=request.default_model,
            )
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=f"Provider not found: {provider}") from exc

    @app.delete("/provider-credentials/{provider}", response_model=OkResponse)
    @app.delete("/v1/provider-credentials/{provider}", response_model=OkResponse)
    def delete_provider_credentials(provider: str):
        try:
            return get_service().delete_provider_credentials(provider)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=f"Provider not found: {provider}") from exc

    @app.post("/provider-credentials/{provider}/test", response_model=OkResponse)
    @app.post("/v1/provider-credentials/{provider}/test", response_model=OkResponse)
    def test_provider_credentials(provider: str):
        try:
            return get_service().test_provider_credentials(provider)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=f"Provider not found: {provider}") from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except Exception as exc:
            raise HTTPException(status_code=502, detail=str(exc)) from exc

    @app.post("/books/parse", response_model=BookParseResponse)
    @app.post("/v1/books/parse", response_model=BookParseResponse)
    def parse_book(request: BookParseRequest):
        try:
            return get_service().parse_book(request.ebook_path)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except Exception as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    @app.post("/books/chapters/filter", response_model=ChapterFilterResponse)
    @app.post("/v1/books/chapters/filter", response_model=ChapterFilterResponse)
    def filter_chapters(request: ChapterFilterRequest):
        try:
            return get_service().filter_chapters(request.book_hash, request.chapter_selection)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=f"Book not found in cache: {exc}") from exc

    @app.post("/books/scan", response_model=TaskResponse, status_code=202)
    @app.post("/v1/books/scan", response_model=TaskResponse, status_code=202)
    def scan_book(request: BookScanRequest):
        return get_service().scan_book(
            request.ebook_path,
            nlp_model=request.nlp_model,
            nlp_provider=request.nlp_provider,
        )

    @app.post("/books/analyze", response_model=TaskResponse, status_code=202)
    @app.post("/v1/books/analyze", response_model=TaskResponse, status_code=202)
    def analyze_book(request: BookAnalyzeRequest):
        return get_service().analyze_book(
            request.ebook_path,
            nlp_model=request.nlp_model,
            nlp_provider=request.nlp_provider,
            discovery_method=request.discovery_method,
            attribution_provider=request.attribution_provider,
            attribution_model=request.attribution_model,
        )

    @app.get("/voices", response_model=VoiceListResponse)
    @app.get("/v1/voices", response_model=VoiceListResponse)
    def list_voices(
        gender: str | None = None,
        accent: str | None = None,
        dataset: str | None = None,
        source: str | None = None,
    ):
        return get_service().list_voices(
            gender=gender,
            accent=accent,
            dataset=dataset,
            source=source,
        )

    @app.post("/voices/suggest-cast", response_model=SuggestCastResponse)
    @app.post("/v1/voices/suggest-cast", response_model=SuggestCastResponse)
    def voices_suggest_cast(req: SuggestCastRequest):
        return get_service().suggest_cast(
            req.roster,
            excluded_voices=req.excluded_voices,
            default_voice=req.default_voice,
        )

    @app.post("/voices/recommend-narrator", response_model=NarratorRecommendationResponse)
    @app.post("/v1/voices/recommend-narrator", response_model=NarratorRecommendationResponse)
    def recommend_narrator(req: NarratorRecommendationRequest):
        return get_service().recommend_narrator(
            req.roster,
            excluded_voices=req.excluded_voices,
            default_voice=req.default_voice,
        )

    @app.post("/voices/assign-simple", response_model=SimpleCastResponse)
    @app.post("/v1/voices/assign-simple", response_model=SimpleCastResponse)
    def assign_simple_cast(req: SimpleCastRequest):
        return get_service().assign_simple_cast(
            req.roster,
            narrator_voice=req.narrator_voice,
            male_voice=req.male_voice,
            female_voice=req.female_voice,
        )

    @app.post("/voices/audition", response_model=TaskResponse, status_code=202)
    @app.post("/v1/voices/audition", response_model=TaskResponse, status_code=202)
    def audition_voice(request: AuditionRequest):
        from kenkui.services.task_service import TaskType
        from kenkui.services.voice_service import prepare_voice_preview

        def _prepare(*, voice_id: str, text: str | None = None, progress_callback=None):
            del progress_callback
            return prepare_voice_preview(voice_id, text=text)

        service = get_service()
        return service.task_response(
            service.task_runner.submit(
                TaskType.AUDITION,
                _prepare,
                voice_id=request.voice_name,
                text=request.text,
            )
        )

    @app.get("/voices/audition/{task_id}.wav")
    @app.get("/v1/voices/audition/{task_id}.wav")
    def get_audition_audio(task_id: str):
        from fastapi.responses import FileResponse

        service = get_service()
        task = service.task_registry.get(task_id)
        if task is None:
            raise HTTPException(status_code=404, detail="Task not found")
        if task.status.value != "completed":
            raise HTTPException(status_code=409, detail=f"Task not yet completed: {task.status.value}")
        if task.result is None or not hasattr(task.result, "audio_path"):
            raise HTTPException(status_code=500, detail="No audio result available")
        return FileResponse(task.result.audio_path, media_type="audio/wav")

    @app.get("/voices/{name}", response_model=VoiceResponse)
    @app.get("/v1/voices/{name}", response_model=VoiceResponse)
    def get_voice(name: str):
        voice = get_service().get_voice(name)
        if voice is None:
            raise HTTPException(status_code=404, detail=f"Voice not found: {name}")
        return voice

    @app.post("/voices/{name}/exclude", response_model=VoicePoolResponse)
    @app.post("/v1/voices/{name}/exclude", response_model=VoicePoolResponse)
    def exclude_voice(name: str):
        return get_service().set_voice_pool_enabled(name, False)

    @app.delete("/voices/{name}/exclude", response_model=VoicePoolResponse)
    @app.delete("/v1/voices/{name}/exclude", response_model=VoicePoolResponse)
    def include_voice(name: str):
        return get_service().set_voice_pool_enabled(name, True)

    @app.post("/voices/download/compiled", response_model=TaskResponse, status_code=202)
    @app.post("/v1/voices/download/compiled", response_model=TaskResponse, status_code=202)
    def download_compiled_voices(request: DownloadRequest):
        from kenkui.services.download_service import download_compiled
        from kenkui.services.task_service import TaskType

        service = get_service()
        return service.task_response(
            service.task_runner.submit(TaskType.VOICE_DOWNLOAD, download_compiled, force=request.force)
        )

    @app.post("/voices/download/uncompiled", response_model=TaskResponse, status_code=202)
    @app.post("/v1/voices/download/uncompiled", response_model=TaskResponse, status_code=202)
    def download_uncompiled_voices(request: FetchRequest):
        from kenkui.services.download_service import fetch_uncompiled
        from kenkui.services.task_service import TaskType

        service = get_service()
        return service.task_response(
            service.task_runner.submit(
                TaskType.VOICE_FETCH,
                fetch_uncompiled,
                repo_id=request.repo_id,
                patterns=request.patterns,
            )
        )

    @app.get("/tasks/{task_id}", response_model=TaskResponse)
    @app.get("/v1/tasks/{task_id}", response_model=TaskResponse)
    def get_task(task_id: str):
        task = get_service().get_task(task_id)
        if task is None:
            raise HTTPException(status_code=404, detail="Task not found")
        return task

    @app.get("/series", response_model=SeriesListResponse)
    @app.get("/v1/series", response_model=SeriesListResponse)
    def list_series():
        return get_service().list_series()

    @app.get("/series/roster-candidates", response_model=RosterCandidateListResponse)
    @app.get("/v1/series/roster-candidates", response_model=RosterCandidateListResponse)
    def list_series_roster_candidates():
        return get_service().list_series_roster_candidates()

    @app.post("/series/empty", response_model=SeriesModel)
    @app.post("/v1/series/empty", response_model=SeriesModel)
    def create_empty_series(req: CreateEmptySeriesRequest):
        return get_service().create_empty_series(req.name)

    @app.post("/series/from-candidate", response_model=SeriesModel)
    @app.post("/v1/series/from-candidate", response_model=SeriesModel)
    def create_series_from_candidate(req: CreateSeriesFromCandidateRequest):
        return get_service().create_series_from_candidate(req.roster_path, req.name)

    @app.get("/series/{slug}", response_model=SeriesModel)
    @app.get("/v1/series/{slug}", response_model=SeriesModel)
    def get_series(slug: str):
        try:
            return get_service().get_series(slug)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=f"Series not found: {slug}") from exc

    @app.post("/series/{slug}/match", response_model=SeriesMatchResponse)
    @app.post("/v1/series/{slug}/match", response_model=SeriesMatchResponse)
    def match_series(slug: str, req: SeriesMatchRequest):
        try:
            return get_service().match_series(slug, req.fast_result)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=f"Series not found: {slug}") from exc

    @app.delete("/series/{slug}", response_model=OkResponse)
    @app.delete("/v1/series/{slug}", response_model=OkResponse)
    def delete_series(slug: str):
        if not get_service().delete_series(slug):
            raise HTTPException(status_code=404, detail=f"Series not found: {slug}")
        return OkResponse()

    @app.get("/auth/huggingface", response_model=HFAuthResponse)
    @app.get("/v1/auth/huggingface", response_model=HFAuthResponse)
    def get_hf_auth():
        return get_service().get_hf_auth()

    @app.post("/auth/huggingface", response_model=HFAuthResponse)
    @app.post("/v1/auth/huggingface", response_model=HFAuthResponse)
    def login_hf(request: HFTokenRequest):
        return get_service().login_hf(request.token)

    @app.get("/status/multivoice", response_model=MultivoiceStatusResponse)
    @app.get("/v1/status/multivoice", response_model=MultivoiceStatusResponse)
    def multivoice_status():
        return get_service().multivoice_status()

    @app.get("/queue/{job_id}/cast", response_model=CastResponse)
    @app.get("/v1/queue/{job_id}/cast", response_model=CastResponse)
    def get_cast(job_id: str):
        response = get_service().get_cast(job_id)
        if response is None:
            raise HTTPException(status_code=404, detail="No cast available for this job")
        return response

    return app

app = None
