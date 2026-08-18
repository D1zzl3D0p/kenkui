"""Pure immutable compilation of validated audiobook intent."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING, Protocol, TypeVar, cast

from kenkui._domain.operations import (
    AssignVoice,
    MetadataIntent,
    Operation,
    SynthesizeSpeech,
)
from kenkui._domain.text import NORMALIZATION_VERSION
from kenkui.errors import (
    ErrorCode,
    ModelError,
    ValidationError,
    VoiceError,
)

if TYPE_CHECKING:
    from kenkui.inspection import BookInspection, ChapterInspection
    from kenkui.voices import Voice

PARSER_SCHEMA_VERSION = "epub-visible-text-v1"
NORMALIZATION_SCHEMA_VERSION = NORMALIZATION_VERSION
PLANNING_SCHEMA_VERSION = "execution-plan-v2"
RENDER_SCHEMA_VERSION = "m4b-render-v1"
CHUNKING_SCHEMA_VERSION = "tts-chunks-v1"
MAX_TTS_SEGMENT_CHARACTERS = 1000
_SEGMENT_ID_VERSION = "v2"
_UTF8_HASH_CHUNK_CHARACTERS = 64 * 1024
_SHA256 = re.compile(r"[0-9a-f]{64}")
_OperationT = TypeVar("_OperationT", bound=Operation)


class _HashDigest(Protocol):
    """Minimal incremental digest interface used by bounded text hashing."""

    def update(self, value: bytes, /) -> None:
        """Add bytes to the digest."""
        ...

    def hexdigest(self) -> str:
        """Return the hexadecimal digest."""
        ...


class _PipelineIntent(Protocol):
    """Minimal structural boundary that excludes source and shell controls."""

    @property
    def operations(self) -> tuple[Operation, ...]:
        """Return ordered semantic operations."""
        ...


class CoverIntent(StrEnum):
    """Renderer-neutral choice for source cover inheritance."""

    SOURCE = "source"
    NONE = "none"


@dataclass(frozen=True, slots=True)
class SchemaVersions:
    """Versions of every stage whose semantics can affect an artifact."""

    parser: str
    normalization: str
    planning: str
    render: str


@dataclass(frozen=True, slots=True)
class VoicePlan:
    """Resolved single-voice content, compatibility, and rights metadata."""

    id: str
    name: str
    content_fingerprint: str
    language: str
    provenance: str
    license_id: str
    commercial_use_allowed: bool
    compatible_model_revisions: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class SpeechSegment:
    """One exact ordered synthesis input for an M1 spine chapter."""

    id: str
    chapter_id: str
    ordinal: int
    text: str
    character_count: int
    content_hash: str


@dataclass(frozen=True, slots=True)
class OutputChapter:
    """Semantic chapter marker metadata independent of output destination."""

    id: str
    source_index: int
    title: str
    speech_characters: int


@dataclass(frozen=True, slots=True)
class OutputMetadata:
    """Resolved bibliographic values and source-cover intent."""

    title: str | None
    author: str | None
    chapters: tuple[OutputChapter, ...]
    cover: CoverIntent
    source_cover_available: bool


@dataclass(frozen=True, slots=True)
class ExecutionPlan:
    """Complete renderer-neutral semantic plan safe to pickle for spawned work."""

    schema_versions: SchemaVersions
    source_bytes_hash: str
    model_revision: str
    voice: VoicePlan
    segments: tuple[SpeechSegment, ...]
    output: OutputMetadata
    total_speech_characters: int
    semantic_fingerprint: str


@dataclass(frozen=True, slots=True)
class _PlanMaterial:
    """Semantic fields canonicalized before attaching their fingerprint."""

    schemas: SchemaVersions
    source_bytes_hash: str
    model_revision: str
    voice: VoicePlan
    segments: tuple[SpeechSegment, ...]
    output: OutputMetadata
    total: int


def compile_execution_plan(
    pipeline: _PipelineIntent,
    inspection: BookInspection,
    *,
    source_bytes_hash: str,
    resolved_voice: Voice | None,
    model_revision: str,
) -> ExecutionPlan:
    """Compile supplied material without filesystem, provider, or process effects."""
    source_hash = _validated_hash(source_bytes_hash)
    revision = model_revision.strip()
    if not revision:
        raise ModelError(ErrorCode.INVALID_MODEL_REVISION)

    assigned = _one_operation(pipeline.operations, AssignVoice)
    if assigned is None:
        raise VoiceError(ErrorCode.VOICE_UNRESOLVED)
    if _one_operation(pipeline.operations, SynthesizeSpeech) is None:
        raise ValidationError(ErrorCode.TTS_REQUIRED)
    voice = _resolve_voice(assigned.voice_id, resolved_voice, revision)

    segments = _compile_segments(inspection.chapters)
    if not segments:
        raise ValidationError(ErrorCode.EMPTY_SPEECH)
    total = sum(segment.character_count for segment in segments)
    metadata = _output_metadata(
        inspection,
        _one_operation(pipeline.operations, MetadataIntent),
    )
    schemas = SchemaVersions(
        parser=PARSER_SCHEMA_VERSION,
        normalization=NORMALIZATION_SCHEMA_VERSION,
        planning=PLANNING_SCHEMA_VERSION,
        render=RENDER_SCHEMA_VERSION,
    )
    material = _PlanMaterial(
        schemas,
        source_hash,
        revision,
        voice,
        segments,
        metadata,
        total,
    )
    return ExecutionPlan(
        schema_versions=schemas,
        source_bytes_hash=source_hash,
        model_revision=revision,
        voice=voice,
        segments=segments,
        output=metadata,
        total_speech_characters=total,
        semantic_fingerprint=_fingerprint(material),
    )


def _one_operation(
    operations: tuple[Operation, ...], kind: type[_OperationT]
) -> _OperationT | None:
    matches = tuple(item for item in operations if isinstance(item, kind))
    if len(matches) > 1:
        raise ValidationError(ErrorCode.DUPLICATE_OPERATION)
    return matches[0] if matches else None


def _validated_hash(value: str) -> str:
    normalized = value.strip().lower()
    if _SHA256.fullmatch(normalized) is None:
        raise ValidationError(ErrorCode.INVALID_SOURCE_HASH)
    return normalized


def _resolve_voice(
    voice_id: str, voice: Voice | None, model_revision: str
) -> VoicePlan:
    if voice is None or voice.id != voice_id:
        raise VoiceError(ErrorCode.VOICE_UNRESOLVED)
    commercial_use_allowed = voice.commercial_use_allowed
    if not isinstance(voice.enabled, bool) or not isinstance(
        commercial_use_allowed, bool
    ):
        raise VoiceError(ErrorCode.VOICE_PROVENANCE_REQUIRED)
    if not voice.enabled:
        raise VoiceError(ErrorCode.VOICE_DISABLED)
    required_strings = (
        voice.id,
        voice.name,
        voice.provenance,
        voice.license_id,
        voice.language,
        voice.content_fingerprint,
    )
    if (
        any(value is None or not value.strip() for value in required_strings)
        or not voice.compatible_model_revisions
        or voice.content_fingerprint is None
        or _SHA256.fullmatch(voice.content_fingerprint.strip().lower()) is None
        or any(not revision.strip() for revision in voice.compatible_model_revisions)
    ):
        raise VoiceError(ErrorCode.VOICE_PROVENANCE_REQUIRED)
    compatibility = tuple(sorted(set(voice.compatible_model_revisions)))
    if model_revision not in compatibility:
        raise VoiceError(ErrorCode.VOICE_INCOMPATIBLE)
    return VoicePlan(
        id=voice.id,
        name=voice.name.strip(),
        content_fingerprint=voice.content_fingerprint.strip().lower(),
        language=cast("str", voice.language).strip(),
        provenance=cast("str", voice.provenance).strip(),
        license_id=cast("str", voice.license_id).strip(),
        commercial_use_allowed=commercial_use_allowed,
        compatible_model_revisions=compatibility,
    )


def _compile_segments(
    chapters: tuple[ChapterInspection, ...],
) -> tuple[SpeechSegment, ...]:
    """Split selected chapters while assigning one global plan-order ordinal."""
    result: list[SpeechSegment] = []
    for chapter in chapters:
        chunks = _chunk_text(chapter)
        for chunk_index, text in enumerate(chunks):
            result.append(_segment(chapter, len(result), chunk_index, text))
    return tuple(result)


def _chunk_text(chapter: ChapterInspection) -> tuple[str, ...]:
    if (
        not chapter.text
        or chapter.speech_characters is None
        or chapter.speech_characters != len(chapter.text)
    ):
        raise ValidationError(ErrorCode.EMPTY_SPEECH)
    text = chapter.text
    chunks: list[str] = []
    start = 0
    while start < len(text):
        hard_end = min(start + MAX_TTS_SEGMENT_CHARACTERS, len(text))
        end = hard_end
        if hard_end < len(text):
            # Keep separator characters in the preceding chunk so joining is exact.
            opportunities = tuple(
                match.end()
                for match in re.finditer(
                    r"(?:[.!?][\"')\]]*\s+|\s+)", text[start:hard_end]
                )
            )
            if opportunities:
                end = start + opportunities[-1]
        if end <= start:  # Defensive hard fallback for arbitrarily long tokens.
            end = hard_end
        chunks.append(text[start:end])
        start = end
    if not chunks or any(not chunk for chunk in chunks) or "".join(chunks) != text:
        raise ValidationError(ErrorCode.EMPTY_SPEECH)
    return tuple(chunks)


def _segment(
    chapter: ChapterInspection, ordinal: int, chunk_index: int, text: str
) -> SpeechSegment:
    content_hash = _hash_utf8(text)
    identity = json.dumps(
        {
            "chapter_id": _string_identity(chapter.id),
            "chunk_index": chunk_index,
            "chunking_schema": CHUNKING_SCHEMA_VERSION,
            "content_hash": content_hash,
            "normalization": NORMALIZATION_SCHEMA_VERSION,
            "ordinal": ordinal,
            "segment_id_version": _SEGMENT_ID_VERSION,
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    digest = _hash_utf8(identity)[:24]
    return SpeechSegment(
        id=f"seg-{NORMALIZATION_SCHEMA_VERSION}-{_SEGMENT_ID_VERSION}-{digest}",
        chapter_id=chapter.id,
        ordinal=ordinal,
        text=text,
        character_count=len(text),
        content_hash=content_hash,
    )


def _output_metadata(
    inspection: BookInspection, intent: MetadataIntent | None
) -> OutputMetadata:
    title = inspection.metadata.title
    author = inspection.metadata.author
    cover = CoverIntent.SOURCE
    if intent is not None:
        title = intent.title.strip() if intent.title is not None else title
        author = intent.author.strip() if intent.author is not None else author
        cover = CoverIntent.SOURCE if intent.cover == "source" else CoverIntent.NONE
    chapters = tuple(
        OutputChapter(
            id=chapter.id,
            source_index=chapter.index,
            title=chapter.title,
            speech_characters=len(chapter.text),
        )
        for chapter in inspection.chapters
    )
    return OutputMetadata(
        title=title,
        author=author,
        chapters=chapters,
        cover=cover,
        source_cover_available=inspection.metadata.cover_available,
    )


def _new_sha256() -> _HashDigest:
    """Create a SHA-256 digest; split out as a narrow test seam."""
    return hashlib.sha256()


def _hash_utf8(value: str) -> str:
    """Hash UTF-8 text while bounding each temporary encoded allocation."""
    digest = _new_sha256()
    for start in range(0, len(value), _UTF8_HASH_CHUNK_CHARACTERS):
        chunk = value[start : start + _UTF8_HASH_CHUNK_CHARACTERS]
        digest.update(chunk.encode("utf-8"))
    return digest.hexdigest()


def _string_identity(value: str) -> dict[str, int | str]:
    """Return a bounded collision-resistant identity for an arbitrary string."""
    return {"characters": len(value), "sha256": _hash_utf8(value)}


def _optional_string_identity(value: str | None) -> dict[str, int | str] | None:
    return None if value is None else _string_identity(value)


def _fingerprint(material: _PlanMaterial) -> str:
    schemas = material.schemas
    voice = material.voice
    segments = material.segments
    output = material.output
    payload = {
        "schema_versions": {
            "parser": schemas.parser,
            "normalization": schemas.normalization,
            "planning": schemas.planning,
            "render": schemas.render,
        },
        "source_bytes_hash": material.source_bytes_hash,
        "model_revision": _string_identity(material.model_revision),
        "voice": {
            "id": _string_identity(voice.id),
            "name": _string_identity(voice.name),
            "content_fingerprint": voice.content_fingerprint,
            "language": _string_identity(voice.language),
            "provenance": _string_identity(voice.provenance),
            "license_id": _string_identity(voice.license_id),
            "commercial_use_allowed": voice.commercial_use_allowed,
            "compatible_model_revisions": [
                _string_identity(revision)
                for revision in voice.compatible_model_revisions
            ],
        },
        "segments": [
            {
                "id": _string_identity(segment.id),
                "chapter_id": _string_identity(segment.chapter_id),
                "ordinal": segment.ordinal,
                "content_hash": segment.content_hash,
                "character_count": segment.character_count,
            }
            for segment in segments
        ],
        "output": {
            "title": _optional_string_identity(output.title),
            "author": _optional_string_identity(output.author),
            "chapters": [
                {
                    "id": _string_identity(chapter.id),
                    "source_index": chapter.source_index,
                    "title": _string_identity(chapter.title),
                    "speech_characters": chapter.speech_characters,
                }
                for chapter in output.chapters
            ],
            "cover": output.cover.value,
            "source_cover_available": output.source_cover_available,
        },
        "total_speech_characters": material.total,
    }
    canonical = json.dumps(
        payload,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    )
    return _hash_utf8(canonical)
