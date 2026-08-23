"""Stable public Kenkui errors."""

from __future__ import annotations

from dataclasses import FrozenInstanceError
from enum import StrEnum


class ErrorCode(StrEnum):
    """Machine-readable public failure codes."""

    UNSUPPORTED_FORMAT = "unsupported_format"
    SOURCE_NOT_FOUND = "source_not_found"
    SOURCE_NOT_READABLE = "source_not_readable"
    DUPLICATE_OPERATION = "duplicate_operation"
    INVALID_OPERATION_ORDER = "invalid_operation_order"
    EMPTY_SELECTION = "empty_selection"
    DUPLICATE_CHAPTER_ID = "duplicate_chapter_id"
    INVALID_VOICE = "invalid_voice"
    VOICE_REQUIRED = "voice_required"
    VOICE_UNRESOLVED = "voice_unresolved"
    VOICE_DISABLED = "voice_disabled"
    VOICE_INCOMPATIBLE = "voice_incompatible"
    VOICE_PROVENANCE_REQUIRED = "voice_provenance_required"
    VOICE_NOT_PROVISIONED = "voice_not_provisioned"
    VOICE_UNKNOWN = "voice_unknown"
    ENGINE_NOT_CLONING_CAPABLE = "engine_not_cloning_capable"
    VOICE_VARIETY_INVALID = "voice_variety_invalid"
    CASTING_METHOD_UNKNOWN = "casting_method_unknown"
    CAST_POOL_EMPTY = "cast_pool_empty"
    CHARACTER_UNKNOWN = "character_unknown"
    MODEL_CALL_FAILED = "model_call_failed"
    MODEL_RESPONSE_INVALID = "model_response_invalid"
    ATTRIBUTION_UNAVAILABLE = "attribution_unavailable"
    TTS_REQUIRED = "tts_required"
    EMPTY_SPEECH = "empty_speech"
    INVALID_SOURCE_HASH = "invalid_source_hash"
    INVALID_MODEL_REVISION = "invalid_model_revision"
    INVALID_METADATA = "invalid_metadata"
    INVALID_WORKERS = "invalid_workers"
    INVALID_OUTPUT = "invalid_output"
    OUTPUT_EXISTS = "output_exists"
    INSPECTION_UNAVAILABLE = "inspection_unavailable"
    MALFORMED_EPUB = "malformed_epub"
    UNSAFE_ARCHIVE_PATH = "unsafe_archive_path"
    ARCHIVE_LIMIT = "archive_limit"
    CHAPTER_NOT_FOUND = "chapter_not_found"
    EMPTY_CHAPTER = "empty_chapter"
    REVERSED_CHAPTER_RANGE = "reversed_chapter_range"
    RENDERER_UNAVAILABLE = "renderer_unavailable"
    POCKET_PACKAGE_MISSING = "pocket_package_missing"
    POCKET_VERSION_UNSUPPORTED = "pocket_version_unsupported"
    POCKET_MODEL_INVALID = "pocket_model_invalid"
    POCKET_VOICE_INVALID = "pocket_voice_invalid"
    POCKET_MODEL_LOAD_FAILED = "pocket_model_load_failed"
    POCKET_VOICE_LOAD_FAILED = "pocket_voice_load_failed"
    POCKET_INFERENCE_FAILED = "pocket_inference_failed"
    FFMPEG_NOT_FOUND = "ffmpeg_not_found"
    FFPROBE_NOT_FOUND = "ffprobe_not_found"
    FFMPEG_UNSUPPORTED = "ffmpeg_unsupported"
    FFPROBE_UNSUPPORTED = "ffprobe_unsupported"
    SYNTHESIS_FAILED = "synthesis_failed"
    INVALID_AUDIO = "invalid_audio"
    ASSEMBLY_FAILED = "assembly_failed"
    ENCODING_FAILED = "encoding_failed"
    COVER_FAILED = "cover_failed"
    PROBE_FAILED = "probe_failed"
    DECODE_FAILED = "decode_failed"
    INVALID_ARTIFACT = "invalid_artifact"
    PUBLICATION_FAILED = "publication_failed"
    CALLBACK_FAILED = "callback_failed"
    CANCELLED = "cancelled"


_DEFAULT_MESSAGES: dict[ErrorCode, str] = {
    ErrorCode.UNSUPPORTED_FORMAT: "The source format is not supported.",
    ErrorCode.SOURCE_NOT_FOUND: "The source does not exist.",
    ErrorCode.SOURCE_NOT_READABLE: "The source is not readable.",
    ErrorCode.DUPLICATE_OPERATION: "The operation was already requested.",
    ErrorCode.INVALID_OPERATION_ORDER: "The operation is not valid at this point.",
    ErrorCode.EMPTY_SELECTION: "At least one chapter ID is required.",
    ErrorCode.DUPLICATE_CHAPTER_ID: "Chapter IDs must be unique.",
    ErrorCode.INVALID_VOICE: "The voice ID is invalid.",
    ErrorCode.VOICE_REQUIRED: "A voice assignment is required.",
    ErrorCode.VOICE_UNRESOLVED: "The assigned voice could not be resolved.",
    ErrorCode.VOICE_DISABLED: "The assigned voice is disabled.",
    ErrorCode.VOICE_INCOMPATIBLE: "The voice is incompatible with the model revision.",
    ErrorCode.VOICE_PROVENANCE_REQUIRED: (
        "The voice is missing required content, compatibility, or provenance metadata."
    ),
    ErrorCode.VOICE_NOT_PROVISIONED: (
        "The voice is registered but not loaded. Call load_voice with its ID."
    ),
    ErrorCode.VOICE_UNKNOWN: "The voice ID is not in the catalog or the manifest.",
    ErrorCode.ENGINE_NOT_CLONING_CAPABLE: (
        "The engine lacks voice-cloning weights required for this voice."
    ),
    ErrorCode.VOICE_VARIETY_INVALID: "The voice variety or state is not recognized.",
    ErrorCode.CASTING_METHOD_UNKNOWN: "The casting method is not recognized.",
    ErrorCode.CAST_POOL_EMPTY: "No loaded voice is available to cast characters.",
    ErrorCode.CHARACTER_UNKNOWN: "A cast entry names a character not in the roster.",
    ErrorCode.MODEL_CALL_FAILED: "The language model could not be reached.",
    ErrorCode.MODEL_RESPONSE_INVALID: (
        "The language model returned an unusable response."
    ),
    ErrorCode.ATTRIBUTION_UNAVAILABLE: (
        "Character casting requires inferred characters."
    ),
    ErrorCode.TTS_REQUIRED: "Explicit TTS intent is required.",
    ErrorCode.EMPTY_SPEECH: "Selected speech must be non-empty and exactly counted.",
    ErrorCode.INVALID_SOURCE_HASH: "The source-bytes hash is invalid.",
    ErrorCode.INVALID_MODEL_REVISION: "The model revision is invalid.",
    ErrorCode.INVALID_METADATA: "The metadata intent is invalid.",
    ErrorCode.INVALID_WORKERS: "Workers must be 'auto' or a positive integer.",
    ErrorCode.INVALID_OUTPUT: "The output must be an M4B path.",
    ErrorCode.OUTPUT_EXISTS: "The output already exists.",
    ErrorCode.INSPECTION_UNAVAILABLE: "EPUB inspection is not available yet.",
    ErrorCode.MALFORMED_EPUB: "The EPUB is malformed or unsupported.",
    ErrorCode.UNSAFE_ARCHIVE_PATH: "The EPUB contains an unsafe archive path.",
    ErrorCode.ARCHIVE_LIMIT: "The EPUB exceeds a safe archive limit.",
    ErrorCode.CHAPTER_NOT_FOUND: "A selected chapter ID was not found.",
    ErrorCode.EMPTY_CHAPTER: "A spine chapter has no visible text.",
    ErrorCode.REVERSED_CHAPTER_RANGE: "The chapter range is reversed.",
    ErrorCode.RENDERER_UNAVAILABLE: "Audiobook rendering is not available yet.",
    ErrorCode.POCKET_PACKAGE_MISSING: (
        "The optional Pocket-TTS package is not installed."
    ),
    ErrorCode.POCKET_VERSION_UNSUPPORTED: (
        "The installed Pocket-TTS version is unsupported."
    ),
    ErrorCode.POCKET_MODEL_INVALID: "The local speech model is invalid or unapproved.",
    ErrorCode.POCKET_VOICE_INVALID: (
        "The local voice prompt is invalid or unauthorized."
    ),
    ErrorCode.POCKET_MODEL_LOAD_FAILED: "The local speech model could not be loaded.",
    ErrorCode.POCKET_VOICE_LOAD_FAILED: "The local voice prompt could not be loaded.",
    ErrorCode.POCKET_INFERENCE_FAILED: "Speech synthesis failed.",
    ErrorCode.FFMPEG_NOT_FOUND: "FFmpeg is required but was not found.",
    ErrorCode.FFPROBE_NOT_FOUND: "ffprobe is required but was not found.",
    ErrorCode.FFMPEG_UNSUPPORTED: "FFmpeg lacks a required M4B capability.",
    ErrorCode.FFPROBE_UNSUPPORTED: "ffprobe lacks a required media capability.",
    ErrorCode.SYNTHESIS_FAILED: "Speech synthesis failed.",
    ErrorCode.INVALID_AUDIO: "Synthesized audio is invalid or inconsistent.",
    ErrorCode.ASSEMBLY_FAILED: "Audiobook assembly failed.",
    ErrorCode.ENCODING_FAILED: "FFmpeg could not encode the audiobook.",
    ErrorCode.COVER_FAILED: "The requested source cover could not be materialized.",
    ErrorCode.PROBE_FAILED: "The encoded audiobook could not be probed.",
    ErrorCode.DECODE_FAILED: "The encoded audiobook did not fully decode.",
    ErrorCode.INVALID_ARTIFACT: "The encoded audiobook failed semantic validation.",
    ErrorCode.PUBLICATION_FAILED: "The audiobook artifact could not be published.",
    ErrorCode.CALLBACK_FAILED: "The execution event callback failed.",
    ErrorCode.CANCELLED: "Execution was cancelled.",
}


class KenkuiError(Exception):
    """Base class for sanitized public failures."""

    __slots__ = ("code", "message")

    code: ErrorCode
    message: str | None

    def __init__(self, code: ErrorCode, message: str | None = None) -> None:
        """Create an immutable public error while preserving exception mechanics."""
        Exception.__init__(self, message or _DEFAULT_MESSAGES[code])
        object.__setattr__(self, "code", code)
        object.__setattr__(self, "message", message)

    def __setattr__(self, name: str, value: object) -> None:
        """Allow Python to maintain traceback state but freeze public fields."""
        if name in {
            "__cause__",
            "__context__",
            "__suppress_context__",
            "__traceback__",
        }:
            Exception.__setattr__(self, name, value)
            return
        error_message = f"cannot assign to field {name!r}"
        raise FrozenInstanceError(error_message)

    def __str__(self) -> str:
        """Return a stable sanitized message."""
        return self.message or _DEFAULT_MESSAGES[self.code]


class ValidationError(KenkuiError):
    """Pipeline intent or argument validation failed."""


class SourceError(KenkuiError):
    """The source could not be accepted or inspected."""


class VoiceError(KenkuiError):
    """Voice resolution failed."""


class ModelError(KenkuiError):
    """Synthesis model setup failed."""


class RenderError(KenkuiError):
    """Speech rendering failed."""


class EncodingError(KenkuiError):
    """Artifact encoding or publication failed."""


class CancelledError(KenkuiError):
    """Execution stopped after cooperative cancellation."""
