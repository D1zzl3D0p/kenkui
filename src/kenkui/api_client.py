"""HTTP API client for connecting to kenkui worker server."""

import time
from dataclasses import dataclass

import httpx

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 45365


@dataclass
class JobInfo:
    """Job information from the server."""

    id: str
    job: dict
    status: str
    progress: float
    current_chapter: str
    eta_seconds: int
    error_message: str
    output_path: str = ""


@dataclass
class QueueInfo:
    """Queue information from the server."""

    items: list[JobInfo]
    current_item: JobInfo | None
    pending_count: int
    completed_count: int
    failed_count: int


@dataclass
class ServerStatus:
    """Server status information."""

    status: str
    is_running: bool
    current_job: str | None


class APIClient:
    """HTTP client for the kenkui worker server API."""

    def __init__(self, host: str = DEFAULT_HOST, port: int = DEFAULT_PORT):
        self._host = host
        self._port = port
        self.base_url = f"http://{host}:{port}"
        self._client = httpx.Client(timeout=30.0)

    def close(self):
        """Close the HTTP client."""
        self._client.close()

    def _request(self, method: str, path: str, **kwargs):
        """Make an HTTP request to the server."""
        url = f"{self.base_url}{path}"
        response = self._client.request(method, url, **kwargs)
        response.raise_for_status()
        return response.json()

    def health_check(self) -> dict:
        """Check if the server is healthy."""
        return self._request("GET", "/health")

    def get_queue(self) -> QueueInfo:
        """Get all jobs in the queue."""
        data = self._request("GET", "/queue")

        items = [JobInfo(**item) for item in data["items"]]
        current = JobInfo(**data["current_item"]) if data.get("current_item") else None

        return QueueInfo(
            items=items,
            current_item=current,
            pending_count=data["pending_count"],
            completed_count=data["completed_count"],
            failed_count=data["failed_count"],
        )

    def add_job(
        self,
        ebook_path: str,
        voice: str = "alba",
        chapter_selection: dict | None = None,
        output_path: str | None = None,
        name: str | None = None,
        narration_mode: str = "single",
        speaker_voices: dict | None = None,
        annotated_chapters_path: str | None = None,
        chapter_voices: dict | None = None,
        roster_cache_path: str | None = None,
        job_temp: float | None = None,
        job_lsd_decode_steps: int | None = None,
        job_noise_clamp: float | None = None,
        job_m4b_bitrate: str | None = None,
        job_pause_line_ms: int | None = None,
        job_pause_chapter_ms: int | None = None,
        job_frames_after_eos: int | None = None,
    ) -> JobInfo:
        """Add a new job to the queue."""
        payload: dict = {
            "ebook_path": ebook_path,
            "voice": voice,
            "narration_mode": narration_mode,
        }
        if chapter_selection:
            payload["chapter_selection"] = chapter_selection
        if output_path:
            payload["output_path"] = output_path
        if name:
            payload["name"] = name
        if speaker_voices:
            payload["speaker_voices"] = speaker_voices
        if annotated_chapters_path:
            payload["annotated_chapters_path"] = annotated_chapters_path
        if chapter_voices:
            payload["chapter_voices"] = chapter_voices
        if roster_cache_path:
            payload["roster_cache_path"] = roster_cache_path
        # Per-job quality overrides — only include when explicitly set
        for key, val in {
            "job_temp": job_temp,
            "job_lsd_decode_steps": job_lsd_decode_steps,
            "job_noise_clamp": job_noise_clamp,
            "job_m4b_bitrate": job_m4b_bitrate,
            "job_pause_line_ms": job_pause_line_ms,
            "job_pause_chapter_ms": job_pause_chapter_ms,
            "job_frames_after_eos": job_frames_after_eos,
        }.items():
            if val is not None:
                payload[key] = val

        data = self._request("POST", "/queue", json=payload)
        return JobInfo(**data)

    def get_job(self, job_id: str) -> JobInfo:
        """Get a specific job by ID."""
        data = self._request("GET", f"/queue/{job_id}")
        return JobInfo(**data)

    def remove_job(self, job_id: str) -> bool:
        """Remove a job from the queue."""
        try:
            self._request("DELETE", f"/queue/{job_id}")
            return True
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 400:
                return False
            raise

    def start_job(self, job_id: str) -> dict:
        """Start processing a specific job."""
        return self._request("POST", f"/queue/{job_id}/start")

    def stop_job(self, job_id: str) -> dict:
        """Stop a processing job."""
        return self._request("POST", f"/queue/{job_id}/stop")

    def start_processing(self) -> dict:
        """Start processing the next job in the queue."""
        return self._request("POST", "/queue/start")

    def stop_processing(self) -> dict:
        """Stop the current processing job."""
        return self._request("POST", "/queue/stop")

    def get_status(self) -> ServerStatus:
        """Get server status."""
        data = self._request("GET", "/status")
        return ServerStatus(
            status=data["status"],
            is_running=data["is_running"],
            current_job=data.get("current_job"),
        )

    def get_config(self) -> dict:
        """Get the current app configuration."""
        data = self._request("GET", "/config")
        return data["config"]

    def update_config(self, config: dict) -> dict:
        """Update the app configuration."""
        return self._request("PUT", "/config", json=config)

    def clear_queue(self) -> dict:
        """Clear all jobs from the queue."""
        return self._request("DELETE", "/queue")

    # --- Books ---

    def parse_book(self, ebook_path: str) -> dict:
        """Parse an ebook and return chapter metadata."""
        return self._request("POST", "/books/parse", json={"ebook_path": ebook_path})

    def filter_chapters(self, book_hash: str, chapter_selection: dict) -> dict:
        """Filter chapters by selection criteria."""
        return self._request(
            "POST",
            "/books/chapters/filter",
            json={"book_hash": book_hash, "chapter_selection": chapter_selection},
        )

    def scan_book(self, ebook_path: str, nlp_model: str | None = None) -> dict:
        """Start async NLP scan of an ebook. Returns a task dict."""
        body: dict = {"ebook_path": ebook_path}
        if nlp_model:
            body["nlp_model"] = nlp_model
        return self._request("POST", "/books/scan", json=body)

    # --- Voices ---

    def list_voices(
        self,
        gender: str | None = None,
        accent: str | None = None,
        dataset: str | None = None,
        source: str | None = None,
    ) -> dict:
        """List available voices with optional filters."""
        params = {
            k: v
            for k, v in {"gender": gender, "accent": accent, "dataset": dataset, "source": source}.items()
            if v is not None
        }
        return self._request("GET", "/voices", params=params)

    def get_voice(self, name: str) -> dict:
        """Get a single voice by name."""
        return self._request("GET", f"/voices/{name}")

    def exclude_voice(self, name: str) -> dict:
        """Exclude a voice from the pool."""
        return self._request("POST", f"/voices/{name}/exclude")

    def include_voice(self, name: str) -> dict:
        """Re-include a previously excluded voice."""
        return self._request("DELETE", f"/voices/{name}/exclude")

    def audition_voice(self, voice_name: str, text: str | None = None) -> dict:
        """Start async voice preview synthesis. Returns a task dict."""
        return self._request("POST", "/voices/audition", json={"voice_name": voice_name, "text": text})

    def get_audition_wav_url(self, task_id: str) -> str:
        """Return the URL to download a completed audition WAV."""
        return f"http://{self._host}:{self._port}/voices/audition/{task_id}.wav"

    def download_compiled_voices(self, force: bool = False) -> dict:
        """Start async download of compiled voices from HuggingFace."""
        return self._request("POST", "/voices/download/compiled", json={"force": force})

    def fetch_uncompiled_voices(
        self,
        repo_id: str | None = None,
        patterns: list[str] | None = None,
    ) -> dict:
        """Start async fetch of uncompiled voice sources from HuggingFace."""
        return self._request(
            "POST",
            "/voices/download/uncompiled",
            json={"repo_id": repo_id, "patterns": patterns},
        )

    def suggest_cast(
        self,
        roster: list[dict],
        excluded_voices: list[str],
        default_voice: str,
    ) -> dict:
        """Suggest a voice cast for a roster of characters."""
        return self._request(
            "POST",
            "/voices/suggest-cast",
            json={
                "roster": roster,
                "excluded_voices": excluded_voices,
                "default_voice": default_voice,
            },
        )

    # --- Tasks ---

    def get_task(self, task_id: str) -> dict:
        """Poll an async task by ID."""
        return self._request("GET", f"/tasks/{task_id}")

    def poll_task(
        self,
        task_id: str,
        *,
        interval: float = 0.5,
        timeout: float = 300.0,
        progress_callback=None,
    ) -> dict:
        """Poll a task until it completes or times out."""
        deadline = time.time() + timeout
        while time.time() < deadline:
            task = self.get_task(task_id)
            if progress_callback:
                progress_callback(task.get("progress", 0), task.get("message", ""))
            status = task.get("status")
            if status in ("completed", "failed"):
                return task
            time.sleep(interval)
        raise TimeoutError(f"Task {task_id} did not complete within {timeout}s")

    # --- Auth ---

    def get_hf_status(self) -> dict:
        """Check HuggingFace authentication status."""
        return self._request("GET", "/auth/huggingface")

    def login_hf(self, token: str) -> dict:
        """Log in to HuggingFace with a token."""
        return self._request("POST", "/auth/huggingface", json={"token": token})

    # --- Status ---

    def get_multivoice_status(self) -> dict:
        """Check multi-voice readiness (spaCy + Ollama)."""
        return self._request("GET", "/status/multivoice")

    # --- Series ---

    def list_series(self) -> dict:
        """List all series manifests."""
        return self._request("GET", "/series")

    def get_series(self, slug: str) -> dict:
        """Get a single series manifest by slug."""
        return self._request("GET", f"/series/{slug}")

    def delete_series(self, slug: str) -> dict:
        """Delete a series manifest by slug."""
        return self._request("DELETE", f"/series/{slug}")

    # --- Queue Cast ---

    def get_queue_cast(self, job_id: str) -> dict:
        """Get the voice cast for a queued job."""
        return self._request("GET", f"/queue/{job_id}/cast")

    # --- Context manager ---

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


_client: APIClient | None = None


def get_client(host: str = DEFAULT_HOST, port: int = DEFAULT_PORT) -> APIClient:
    """Get or create the global API client."""
    global _client
    if _client is None:
        _client = APIClient(host, port)
    return _client


def reset_client():
    """Reset the global API client."""
    global _client
    if _client:
        _client.close()
    _client = None
