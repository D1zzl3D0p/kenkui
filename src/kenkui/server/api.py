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


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    return app
