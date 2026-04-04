"""FastAPI routes for the kenkui server."""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from ..models import AppConfig, ChapterSelection, JobConfig, JobStatus, NarrationMode
from .worker import get_server

app = FastAPI(title="kenkui Worker Server", version="0.8.0")


class JobCreateRequest(BaseModel):
    ebook_path: str
    voice: str = "alba"
    chapter_selection: ChapterSelection | None = None
    output_path: str | None = None
    name: str | None = None
    # Multi-voice fields
    narration_mode: str = "single"
    speaker_voices: dict[str, str] = {}
    annotated_chapters_path: str | None = None
    # Chapter-voice mode
    chapter_voices: dict[str, str] = {}
    # Multi-voice roster cache
    roster_cache_path: str | None = None
    # Per-job quality overrides (None = inherit from AppConfig)
    job_temp: float | None = None
    job_lsd_decode_steps: int | None = None
    job_noise_clamp: float | None = None
    job_m4b_bitrate: str | None = None
    job_pause_line_ms: int | None = None
    job_pause_chapter_ms: int | None = None
    job_frames_after_eos: int | None = None


class JobResponse(BaseModel):
    id: str
    job: dict
    status: str
    progress: float
    current_chapter: str
    eta_seconds: int
    error_message: str
    output_path: str = ""


class QueueResponse(BaseModel):
    items: list[JobResponse]
    current_item: JobResponse | None = None
    pending_count: int
    completed_count: int
    failed_count: int


class ConfigResponse(BaseModel):
    config: dict


class StatusResponse(BaseModel):
    status: str
    is_running: bool
    current_job: str | None = None


# --- Books ---
class BookParseRequest(BaseModel):
    ebook_path: str


class ChapterSummaryModel(BaseModel):
    index: int
    title: str
    word_count: int
    paragraph_count: int
    toc_index: int
    tags: dict  # ChapterTags serialized


class BookParseResponse(BaseModel):
    book_hash: str
    metadata: dict
    chapters: list[ChapterSummaryModel]
    total_chapters: int
    total_word_count: int


class ChapterFilterRequest(BaseModel):
    book_hash: str
    chapter_selection: ChapterSelection   # reuse existing import


class ChapterFilterResponse(BaseModel):
    included_indices: list[int]
    chapter_count: int
    estimated_word_count: int
    chapters: list[ChapterSummaryModel]


class BookScanRequest(BaseModel):
    ebook_path: str
    nlp_model: str | None = None


# --- Voices ---
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


# --- Suggest Cast ---
class CharacterInfoModel(BaseModel):
    name: str
    pronoun: str | None = None
    quote_count: int = 0
    mention_count: int = 0


class SuggestCastRequest(BaseModel):
    roster: list[CharacterInfoModel]
    excluded_voices: list[str] = []
    default_voice: str = "narrator"


class SuggestCastResponse(BaseModel):
    speaker_voices: dict[str, str]
    warnings: list[str]


# --- Tasks ---
class TaskResponse(BaseModel):
    task_id: str
    type: str
    status: str
    progress: int
    message: str
    result: dict | None = None
    error: str | None = None


# --- Series ---
class SeriesCharacterModel(BaseModel):
    canonical: str
    aliases: list[str] = []
    voice: str = ""
    gender: str = ""


class SeriesModel(BaseModel):
    slug: str
    name: str
    updated_at: str = ""
    characters: list[SeriesCharacterModel] = []


class SeriesListResponse(BaseModel):
    series: list[SeriesModel]
    total: int


# --- Auth ---
class HFTokenRequest(BaseModel):
    token: str


class HFAuthResponse(BaseModel):
    authenticated: bool
    username: str | None = None
    has_pocket_tts_access: bool
    error: str | None = None


# --- Queue Cast ---
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


def _job_to_response(item) -> JobResponse:
    """Convert a QueueItem to a JobResponse."""
    return JobResponse(
        id=item.id,
        job=item.job.to_dict(),
        status=item.status.value,
        progress=item.progress,
        current_chapter=item.current_chapter,
        eta_seconds=item.eta_seconds,
        error_message=item.error_message,
        output_path=item.output_path,
    )


def _task_to_response(task) -> "TaskResponse":
    """Convert a Task to a TaskResponse."""
    result_dict = None
    if task.result is not None:
        if hasattr(task.result, "model_dump"):
            result_dict = task.result.model_dump()
        elif hasattr(task.result, "__dict__"):
            result_dict = task.result.__dict__
        else:
            result_dict = {"value": str(task.result)}
    return TaskResponse(
        task_id=task.task_id,
        type=task.type.value,
        status=task.status.value,
        progress=task.progress,
        message=task.message,
        result=result_dict,
        error=task.error,
    )


@app.get("/health")
def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "version": "0.8.0"}


@app.get("/queue", response_model=QueueResponse)
def get_queue():
    """Get all jobs in the queue."""
    server = get_server()

    current = server.current_item
    pending = server.pending_items
    completed = server.completed_items
    failed = server.failed_items

    all_items = server.all_items

    return QueueResponse(
        items=[_job_to_response(i) for i in all_items],
        current_item=_job_to_response(current) if current else None,
        pending_count=len(pending),
        completed_count=len(completed),
        failed_count=len(failed),
    )


@app.post("/queue", response_model=JobResponse)
def add_job(request: JobCreateRequest):
    """Add a new job to the queue."""
    from pathlib import Path

    job = JobConfig(
        ebook_path=Path(request.ebook_path),
        voice=request.voice,
        chapter_selection=request.chapter_selection or ChapterSelection(),
        output_path=Path(request.output_path) if request.output_path else None,
        name=request.name or "",
        narration_mode=NarrationMode(request.narration_mode),
        speaker_voices=request.speaker_voices or {},
        annotated_chapters_path=Path(request.annotated_chapters_path)
        if request.annotated_chapters_path
        else None,
        chapter_voices=request.chapter_voices or {},
        roster_cache_path=Path(request.roster_cache_path)
        if request.roster_cache_path
        else None,
        job_temp=request.job_temp,
        job_lsd_decode_steps=request.job_lsd_decode_steps,
        job_noise_clamp=request.job_noise_clamp,
        job_m4b_bitrate=request.job_m4b_bitrate,
        job_pause_line_ms=request.job_pause_line_ms,
        job_pause_chapter_ms=request.job_pause_chapter_ms,
        job_frames_after_eos=request.job_frames_after_eos,
    )

    server = get_server()
    item = server.add_job(job)
    return _job_to_response(item)


@app.get("/queue/{job_id}", response_model=JobResponse)
def get_job(job_id: str):
    """Get a specific job by ID."""
    server = get_server()
    item = server.get_job(job_id)

    if not item:
        raise HTTPException(status_code=404, detail="Job not found")

    return _job_to_response(item)


@app.delete("/queue/{job_id}")
def remove_job(job_id: str):
    """Remove a job from the queue."""
    server = get_server()
    success = server.remove_job(job_id)

    if not success:
        raise HTTPException(
            status_code=400, detail="Cannot remove job - it may be currently processing"
        )

    return {"status": "removed", "job_id": job_id}


@app.post("/queue/{job_id}/start")
def start_job(job_id: str):
    """Start processing a specific job."""
    server = get_server()
    item = server.get_job(job_id)

    if not item:
        raise HTTPException(status_code=404, detail="Job not found")

    if server.is_running:
        raise HTTPException(status_code=400, detail="Processing already in progress")

    item.status = JobStatus.PROCESSING
    server._save()

    server.start_processing()

    return {"status": "started", "job_id": job_id}


@app.post("/queue/{job_id}/stop")
def stop_job(job_id: str):
    """Stop a processing job."""
    server = get_server()
    item = server.get_job(job_id)

    if not item:
        raise HTTPException(status_code=404, detail="Job not found")

    server.stop_processing()
    item.status = JobStatus.CANCELLED
    server._save()

    return {"status": "stopped", "job_id": job_id}


@app.post("/queue/start")
def start_processing():
    """Start processing the next job in the queue."""
    server = get_server()

    if server.is_running:
        raise HTTPException(status_code=400, detail="Processing already in progress")

    server.start_processing()

    return {"status": "started"}


@app.post("/queue/stop")
def stop_processing():
    """Stop the current processing job."""
    server = get_server()
    server.stop_processing()

    return {"status": "stopped"}


@app.get("/status", response_model=StatusResponse)
def get_status():
    """Get server status."""
    server = get_server()
    current = server.current_item

    return StatusResponse(
        status="running" if server.is_running else "idle",
        is_running=server.is_running,
        current_job=current.id if current else None,
    )


@app.get("/config", response_model=ConfigResponse)
def get_config():
    """Get the current app configuration."""
    server = get_server()
    return ConfigResponse(config=server.app_config.to_dict())


@app.put("/config")
def update_config(config_data: dict):
    """Update the app configuration."""
    server = get_server()
    config = AppConfig.from_dict(config_data)
    server.app_config = config

    return {"status": "updated", "config": config.to_dict()}


@app.delete("/queue")
def clear_queue():
    """Clear all jobs from the queue."""
    server = get_server()
    server.clear_all_jobs()

    return {"status": "cleared"}


@app.post("/books/parse", response_model=BookParseResponse)
def parse_book(request: BookParseRequest):
    """Parse an ebook and cache chapter metadata."""
    from ..services.book_service import parse_book as _parse_book
    server = get_server()
    try:
        result = _parse_book(request.ebook_path, server.book_cache)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    return BookParseResponse(
        book_hash=result.book_hash,
        metadata=result.metadata,
        chapters=[ChapterSummaryModel(**{
            "index": c.index, "title": c.title, "word_count": c.word_count,
            "paragraph_count": c.paragraph_count, "toc_index": c.toc_index,
            "tags": {"is_toc": c.tags.is_toc, "is_foreword": c.tags.is_foreword,
                     "is_content": c.tags.is_content, "is_appendix": c.tags.is_appendix,
                     "is_short": c.tags.is_short},
        }) for c in result.chapters],
        total_chapters=result.total_chapters,
        total_word_count=result.total_word_count,
    )


@app.post("/books/chapters/filter", response_model=ChapterFilterResponse)
def filter_chapters(request: ChapterFilterRequest):
    """Filter chapters for a previously-parsed book."""
    from ..services.book_service import filter_chapters as _filter_chapters
    server = get_server()
    try:
        result = _filter_chapters(request.book_hash, request.chapter_selection, server.book_cache)
    except KeyError as e:
        raise HTTPException(status_code=404, detail=f"Book not found in cache: {e}")
    return ChapterFilterResponse(
        included_indices=result.included_indices,
        chapter_count=result.chapter_count,
        estimated_word_count=result.estimated_word_count,
        chapters=[ChapterSummaryModel(**{
            "index": c.index, "title": c.title, "word_count": c.word_count,
            "paragraph_count": c.paragraph_count, "toc_index": c.toc_index,
            "tags": {"is_toc": c.tags.is_toc, "is_foreword": c.tags.is_foreword,
                     "is_content": c.tags.is_content, "is_appendix": c.tags.is_appendix,
                     "is_short": c.tags.is_short},
        }) for c in result.chapters],
    )


@app.post("/books/scan", response_model=TaskResponse, status_code=202)
def scan_book(request: BookScanRequest):
    """Start an async NLP fast scan. Returns a task_id to poll."""
    from ..services.nlp_service import fast_scan
    from ..server.tasks import TaskType
    server = get_server()
    task = server.task_runner.submit(
        TaskType.FAST_SCAN, fast_scan,
        ebook_path=request.ebook_path,
        nlp_model=request.nlp_model,
    )
    return _task_to_response(task)


@app.get("/voices", response_model=VoiceListResponse)
def list_voices(gender: str | None = None, accent: str | None = None,
                dataset: str | None = None, source: str | None = None):
    """List available voices with optional filters."""
    from ..services.voice_service import list_voices as _list_voices
    voices = _list_voices(gender=gender, accent=accent, dataset=dataset, source=source)
    return VoiceListResponse(
        voices=[VoiceResponse(**v.__dict__) for v in voices],
        total=len(voices),
    )


@app.post("/voices/suggest-cast", response_model=SuggestCastResponse)
def voices_suggest_cast(req: SuggestCastRequest):
    """Suggest a voice cast for a roster of characters."""
    from kenkui.services.voice_service import suggest_cast
    from kenkui.models import CharacterInfo
    roster = [CharacterInfo(character_id=c.name, display_name=c.name,
                            gender_pronoun=c.pronoun or "",
                            quote_count=c.quote_count, mention_count=c.mention_count)
              for c in req.roster]
    result = suggest_cast(
        roster=roster,
        excluded_voices=req.excluded_voices,
        default_voice=req.default_voice,
    )
    return SuggestCastResponse(speaker_voices=result.speaker_voices, warnings=result.warnings)


@app.get("/voices/{name}", response_model=VoiceResponse)
def get_voice(name: str):
    """Get details for a specific voice."""
    from ..services.voice_service import get_voice as _get_voice
    voice = _get_voice(name)
    if voice is None:
        raise HTTPException(status_code=404, detail=f"Voice not found: {name}")
    return VoiceResponse(**voice.__dict__)


@app.post("/voices/{name}/exclude")
def exclude_voice(name: str):
    """Exclude a voice from the random pool."""
    from ..services.voice_service import exclude_voice as _exclude_voice
    result = _exclude_voice(name)
    return {"excluded_voices": result.excluded_voices, "warning": result.warning}


@app.delete("/voices/{name}/exclude")
def include_voice(name: str):
    """Re-include a voice in the random pool."""
    from ..services.voice_service import include_voice as _include_voice
    result = _include_voice(name)
    return {"excluded_voices": result.excluded_voices}


@app.post("/voices/audition", response_model=TaskResponse, status_code=202)
def audition_voice(request: AuditionRequest):
    """Start async voice preview synthesis. Returns a task_id to poll."""
    from ..services.voice_service import synthesize_preview
    from ..server.tasks import TaskType
    server = get_server()
    task = server.task_runner.submit(
        TaskType.AUDITION, synthesize_preview,
        voice_name=request.voice_name,
        text=request.text,
    )
    return _task_to_response(task)


# NOTE: /voices/audition routes must be registered before /voices/{name} to prevent
# the {name} wildcard from shadowing the literal 'audition' path segment.
@app.get("/voices/audition/{task_id}.wav")
def get_audition_audio(task_id: str):
    """Download the audio file for a completed audition task."""
    from fastapi.responses import FileResponse
    server = get_server()
    task = server.task_registry.get(task_id)
    if task is None:
        raise HTTPException(status_code=404, detail="Task not found")
    if task.status.value != "completed":
        raise HTTPException(status_code=409, detail=f"Task not yet completed: {task.status.value}")
    if task.result is None or not hasattr(task.result, "audio_path"):
        raise HTTPException(status_code=500, detail="No audio result available")
    return FileResponse(task.result.audio_path, media_type="audio/wav")


@app.post("/voices/download/compiled", response_model=TaskResponse, status_code=202)
def download_compiled_voices(request: DownloadRequest):
    """Start async download of compiled voices from HuggingFace."""
    from ..services.download_service import download_compiled
    from ..server.tasks import TaskType
    server = get_server()
    task = server.task_runner.submit(
        TaskType.VOICE_DOWNLOAD, download_compiled,
        force=request.force,
    )
    return _task_to_response(task)


@app.post("/voices/download/uncompiled", response_model=TaskResponse, status_code=202)
def download_uncompiled_voices(request: FetchRequest):
    """Start async fetch of uncompiled voice sources from HuggingFace."""
    from ..services.download_service import fetch_uncompiled
    from ..server.tasks import TaskType
    server = get_server()
    task = server.task_runner.submit(
        TaskType.VOICE_FETCH, fetch_uncompiled,
        repo_id=request.repo_id,
        patterns=request.patterns,
    )
    return _task_to_response(task)


@app.get("/tasks/{task_id}", response_model=TaskResponse)
def get_task(task_id: str):
    """Poll an async task for status, progress, and result."""
    server = get_server()
    task = server.task_registry.get(task_id)
    if task is None:
        raise HTTPException(status_code=404, detail="Task not found")
    return _task_to_response(task)


@app.get("/series", response_model=SeriesListResponse)
def list_series_route():
    """List all series manifests."""
    from ..services.series_service import list_series as _list_series
    result = _list_series()
    return SeriesListResponse(
        series=[
            SeriesModel(
                slug=e.slug,
                name=e.name,
                updated_at=e.updated_at,
                characters=[
                    SeriesCharacterModel(
                        canonical=c.canonical,
                        aliases=c.aliases,
                        voice=c.voice,
                        gender=c.gender,
                    )
                    for c in e.characters
                ],
            )
            for e in result.series
        ],
        total=result.total,
    )


@app.get("/series/{slug}", response_model=SeriesModel)
def get_series_route(slug: str):
    """Get a single series manifest by slug."""
    from ..services.series_service import load_series as _load_series
    try:
        entry = _load_series(slug)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Series not found: {slug}")
    return SeriesModel(
        slug=entry.slug,
        name=entry.name,
        updated_at=entry.updated_at,
        characters=[
            SeriesCharacterModel(
                canonical=c.canonical,
                aliases=c.aliases,
                voice=c.voice,
                gender=c.gender,
            )
            for c in entry.characters
        ],
    )


@app.delete("/series/{slug}")
def delete_series_route(slug: str):
    """Delete a series manifest by slug."""
    from ..services.series_service import delete_series as _delete_series
    found = _delete_series(slug)
    if not found:
        raise HTTPException(status_code=404, detail=f"Series not found: {slug}")
    return {"status": "deleted", "slug": slug}


@app.get("/auth/huggingface", response_model=HFAuthResponse)
def get_hf_auth():
    """Check HuggingFace authentication status."""
    from ..services.auth_service import get_hf_status
    status = get_hf_status()
    return HFAuthResponse(
        authenticated=status.authenticated,
        username=status.username,
        has_pocket_tts_access=status.has_pocket_tts_access,
    )


@app.post("/auth/huggingface", response_model=HFAuthResponse)
def login_hf(request: HFTokenRequest):
    """Log in to HuggingFace with a token."""
    from ..services.auth_service import get_hf_status, login
    result = login(request.token)
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


@app.get("/queue/{job_id}/cast", response_model=CastResponse)
def get_cast(job_id: str):
    """Get the voice cast for a job (requires NLP analysis to have been run)."""
    import json
    server = get_server()
    item = server.get_job(job_id)
    if item is None:
        raise HTTPException(status_code=404, detail="Job not found")
    job = item.job
    if job.annotated_chapters_path is None or not job.annotated_chapters_path.exists():
        raise HTTPException(status_code=404, detail="No NLP analysis available for this job")
    try:
        from ..models import NLPResult
        data = json.loads(job.annotated_chapters_path.read_text(encoding="utf-8"))
        nlp_result = NLPResult.from_dict(data)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Could not read NLP analysis: {e}")

    cast_entries = []
    for char_id, voice_name in (job.speaker_voices or {}).items():
        char_info = next((c for c in nlp_result.roster if c.id == char_id), None)
        cast_entries.append(CastEntry(
            character_id=char_id,
            display_name=char_info.display_name if char_info else char_id,
            voice_name=voice_name,
            quote_count=char_info.quote_count if char_info else 0,
            mention_count=char_info.mention_count if char_info else 0,
            gender_pronoun=char_info.gender_pronoun if char_info else None,
        ))

    return CastResponse(
        job_id=job_id,
        book_name=job.name,
        narration_mode=job.narration_mode.value,
        cast=cast_entries,
    )


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    return app
