from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .common import CostStatus, JobStatus
from .job import JobConfig


@dataclass
class QueueItem:
    id: str
    job: JobConfig
    status: JobStatus = JobStatus.PENDING
    progress: float = 0.0
    current_chapter: str = ""
    eta_seconds: int = 0
    error_message: str = ""
    output_path: str = ""
    started_at: float = 0.0  # Unix timestamp set when job enters PROCESSING
    completed_at: float = 0.0  # Unix timestamp set when job completes
    execution_provider: str = ""
    remote_job_id: str = ""
    estimated_cost_usd: float | None = None
    actual_cost_usd: float | None = None
    cost_status: CostStatus = CostStatus.NONE
    artifact_uri: str = ""
    artifact_source: str = ""
    provider_status: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "job": self.job.to_dict(),
            "status": self.status.value,
            "progress": self.progress,
            "current_chapter": self.current_chapter,
            "eta_seconds": self.eta_seconds,
            "error_message": self.error_message,
            "output_path": self.output_path,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "execution_provider": self.execution_provider,
            "remote_job_id": self.remote_job_id,
            "estimated_cost_usd": self.estimated_cost_usd,
            "actual_cost_usd": self.actual_cost_usd,
            "cost_status": self.cost_status.value,
            "artifact_uri": self.artifact_uri,
            "artifact_source": self.artifact_source,
            "provider_status": self.provider_status,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> QueueItem:
        return cls(
            id=data["id"],
            job=JobConfig.from_dict(data["job"]),
            status=JobStatus(data.get("status", "pending")),
            progress=data.get("progress", 0.0),
            current_chapter=data.get("current_chapter", ""),
            eta_seconds=data.get("eta_seconds", 0),
            error_message=data.get("error_message", ""),
            output_path=data.get("output_path", ""),
            started_at=data.get("started_at", 0.0),
            completed_at=data.get("completed_at", 0.0),
            execution_provider=data.get("execution_provider", ""),
            remote_job_id=data.get("remote_job_id", ""),
            estimated_cost_usd=data.get("estimated_cost_usd"),
            actual_cost_usd=data.get("actual_cost_usd"),
            cost_status=CostStatus(data.get("cost_status", "none")),
            artifact_uri=data.get("artifact_uri", ""),
            artifact_source=data.get("artifact_source", ""),
            provider_status=data.get("provider_status", ""),
        )


