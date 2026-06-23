"""Serializable request/response contracts for Modal runtime calls."""
from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class RemoteChapterRenderRequest(BaseModel):
    chapter: dict[str, Any]
    config: dict[str, Any]
    is_first_chapter: bool = False


class RemoteChapterRenderResult(BaseModel):
    success: bool
    chapter_index: int
    wav_artifact_uri: str = ""
    duration_ms: int = 0
    error_message: str = ""


class RemoteTTSRequest(BaseModel):
    job_id: str
    config: dict[str, Any]
    chapters: list[dict[str, Any]] = Field(default_factory=list)
    output_name: str = ""
    artifact_prefix: str = ""


class RemoteTTSResult(BaseModel):
    success: bool
    artifact_uri: str = ""
    output_filename: str = ""
    error_message: str = ""
    provider_status: str = ""
    remote_job_id: str = ""


class RemoteExtractionRequest(BaseModel):
    chapters: list[dict[str, Any]] = Field(default_factory=list)
    config: dict[str, Any]
    book_path: str | None = None
    series_roster: dict[str, Any] | None = None
    book_hash: str = ""
    provider: str = ""
    model: str = ""
    method: str = ""


class RemoteExtractionResult(BaseModel):
    roster: dict[str, Any]
    error_message: str = ""
    provider_status: str = ""


class RemoteAttributionRequest(BaseModel):
    chapter: dict[str, Any]
    roster: dict[str, Any]
    config: dict[str, Any]
    book_hash: str = ""
    provider: str = ""
    model: str = ""


class RemoteAttributionResult(BaseModel):
    result: dict[str, Any]
    error_message: str = ""
    provider_status: str = ""


__all__ = [
    "RemoteAttributionRequest",
    "RemoteAttributionResult",
    "RemoteChapterRenderRequest",
    "RemoteChapterRenderResult",
    "RemoteExtractionRequest",
    "RemoteExtractionResult",
    "RemoteTTSRequest",
    "RemoteTTSResult",
]
