"""Intentional public package interface for Kenkui."""

from ._domain.operations import MetadataIntent
from .api import (
    ExecutionStats,
    Result,
    ValidationIssue,
    ValidationResult,
    book,
    epub,
)
from .cancellation import CancellationToken
from .errors import (
    CancelledError,
    EncodingError,
    ErrorCode,
    KenkuiError,
    ModelError,
    RenderError,
    SourceError,
    ValidationError,
    VoiceError,
)
from .events import (
    Completed,
    ExecutionEvent,
    StageCompleted,
    StageProgress,
    StageStarted,
    Started,
    Warning,
)
from .inspection import BookInspection, BookMetadata, ChapterInspection
from .pipeline import Pipeline, Source
from .voices import Engine, Voice
from .voices.provision import add_voice

__version__ = "0.1.0"

__all__ = [
    "BookInspection",
    "BookMetadata",
    "CancellationToken",
    "CancelledError",
    "ChapterInspection",
    "Completed",
    "EncodingError",
    "Engine",
    "ErrorCode",
    "ExecutionEvent",
    "ExecutionStats",
    "KenkuiError",
    "MetadataIntent",
    "ModelError",
    "Pipeline",
    "RenderError",
    "Result",
    "Source",
    "SourceError",
    "StageCompleted",
    "StageProgress",
    "StageStarted",
    "Started",
    "ValidationError",
    "ValidationIssue",
    "ValidationResult",
    "Voice",
    "VoiceError",
    "Warning",
    "__version__",
    "add_voice",
    "book",
    "epub",
]
