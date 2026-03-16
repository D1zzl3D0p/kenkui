"""HTTP API client for connecting to kenkui worker server."""

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
    ) -> JobInfo:
        """Add a new job to the queue."""
        payload = {
            "ebook_path": ebook_path,
            "voice": voice,
        }
        if chapter_selection:
            payload["chapter_selection"] = chapter_selection
        if output_path:
            payload["output_path"] = output_path
        if name:
            payload["name"] = name

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
