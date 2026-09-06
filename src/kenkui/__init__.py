"""Intentional public package interface for Kenkui."""

from ._characters.models import SeriesCharacter, SeriesRecord
from ._characters.store import (
    list_castings,
    list_series,
    remove_attribution,
    remove_casting,
    remove_series,
)
from ._domain.casting import CharacterProfile, Collision
from ._domain.operations import MetadataIntent
from ._domain.planning import SpeakerSpan
from .api import (
    ExecutionStats,
    Result,
    ValidationIssue,
    ValidationResult,
    book,
    builtin_lexicon,
    epub,
    magic_run,
    read_lexicon,
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
    CastResolved,
    Completed,
    ExecutionEvent,
    StageCompleted,
    StageProgress,
    StageStarted,
    Started,
    Warning,
)
from .inspection import (
    BookInspection,
    BookMetadata,
    CastingInspection,
    ChapterInspection,
)
from .pipeline import Pipeline, Source
from .voices import Engine, Voice
from .voices.provision import (
    add_voice,
    list_voices,
    load_voice,
    remove_voice,
    unload_voice,
)

__version__ = "0.1.0"

__all__ = [
    "BookInspection",
    "BookMetadata",
    "CancellationToken",
    "CancelledError",
    "CastResolved",
    "CastingInspection",
    "ChapterInspection",
    "CharacterProfile",
    "Collision",
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
    "SeriesCharacter",
    "SeriesRecord",
    "Source",
    "SourceError",
    "SpeakerSpan",
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
    "builtin_lexicon",
    "epub",
    "list_castings",
    "list_series",
    "list_voices",
    "load_voice",
    "magic_run",
    "read_lexicon",
    "remove_attribution",
    "remove_casting",
    "remove_series",
    "remove_voice",
    "unload_voice",
]
