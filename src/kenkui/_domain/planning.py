"""Pure immutable compilation of validated audiobook intent."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING, Protocol, TypeVar, cast

from kenkui._domain.operations import (
    AssignVoices,
    MetadataIntent,
    Operation,
    SpokenForm,
    SynthesizeSpeech,
)
from kenkui._domain.spoken import (
    SPOKEN_FORM_VERSION,
    spoken_identity,
    to_spoken,
)
from kenkui._domain.text import NORMALIZATION_VERSION
from kenkui.errors import (
    ErrorCode,
    ModelError,
    ValidationError,
    VoiceError,
)

if TYPE_CHECKING:
    from collections.abc import Mapping

    from kenkui._domain.spoken.numbers import NumberTier
    from kenkui.inspection import BookInspection, ChapterInspection
    from kenkui.voices import Voice

PARSER_SCHEMA_VERSION = "epub-visible-text-v1"
NORMALIZATION_SCHEMA_VERSION = NORMALIZATION_VERSION
PLANNING_SCHEMA_VERSION = "execution-plan-v2"
RENDER_SCHEMA_VERSION = "m4b-render-v1"
CHUNKING_SCHEMA_VERSION = "tts-chunks-v2"
MAX_TTS_SEGMENT_CHARACTERS = 1000
# Pocket-TTS divides a segment on ".!?", sub-divides what is left on ",;:", and
# only then packs the pieces into bounded chunks. A run holding none of these is
# indivisible to it, so an over-long one is generated past the model's own limit
# and returns as unusable audio. Kenkui splits such a run itself while a natural
# boundary is still available.
POCKET_SEPARATORS = ".!?,;:"
# Calibrated against the engine's own tokenizer, counting the way it does: it
# replaces newlines with spaces before tokenizing, so a run is measured after
# that collapse. Over one 594k-character book, leaving runs unsplit produced a
# worst run of 320 tokens against a 50-token limit -- the case that generates
# past the limit and returns unusable audio. A 100-character budget holds the
# worst run to 58 tokens for about 30% more segments; tighter budgets fragment
# ordinary prose for little further gain.
MAX_SEPARATOR_FREE_CHARACTERS = 100
# Break points ranked by how natural the resulting pause sounds. Each tier keeps
# its separator in the preceding chunk so joining stays exact.
_BREAK_TIERS = (
    r"\n\s*",
    r"[.!?][\"')\]]*\s+|[,;:][\"')\]]*\s+",
    r"\s+",
    r"[-\u2010-\u2015]",
)
# A better boundary is only worth taking when it still fills the window. Without
# this, one early line break would strand a nearly empty chunk and multiply the
# per-segment synthesis overhead across a book. Measured over one 594k-character
# book, the share of breaks landing on a line break or clause boundary moves only
# from 71% to 64% across fills of 0.5 to 0.8, while segment count falls 1211 to
# 987, so the middle of that range buys most of the quality for less overhead.
MIN_BREAK_FILL = 0.7
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
    spoken_form: str | None = None


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
class SpeakerSpan:
    """One contiguous run of a chapter's normalized text with a single speaker."""

    chapter_id: str
    start: int
    end: int
    character_id: str | None  # None is narration


@dataclass(frozen=True, slots=True)
class CastPlan:
    """Resolved narrator, unknown, and per-character voices for one run."""

    narrator: VoicePlan
    unknown: VoicePlan
    voices: tuple[VoicePlan, ...]
    assignments: Mapping[str, str]  # character id -> voice id

    @classmethod
    def single(cls, voice: VoicePlan) -> CastPlan:
        """Return the degenerate cast: one voice narrates everything."""
        return cls(narrator=voice, unknown=voice, voices=(voice,), assignments={})

    def voice_for(self, speaker_id: str | None) -> VoicePlan:
        """Resolve the voice a speaker renders in.

        Narration takes the narrator. A character with no assignment takes the
        unknown voice, which defaults to the narrator's, so speech never
        silently vanishes because casting missed someone.
        """
        if speaker_id is None:
            return self.narrator
        voice_id = self.assignments.get(speaker_id)
        if voice_id is None:
            return self.unknown
        for voice in self.voices:
            if voice.id == voice_id:
                return voice
        return self.unknown


@dataclass(frozen=True, slots=True)
class SpeechSegment:
    """One exact ordered synthesis input for an M1 spine chapter."""

    id: str
    chapter_id: str
    ordinal: int
    text: str
    character_count: int
    content_hash: str
    speaker_id: str | None = None
    voice_id: str = ""


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
    # No separate singular voice: a lone VoicePlan beside the cast is a second
    # source of truth that can disagree with it, and disagreement here renders
    # well-formed audio in the wrong voice. cast.narrator is the authority.
    cast: CastPlan
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


def compile_execution_plan(  # noqa: PLR0913 - explicit compilation boundary.
    pipeline: _PipelineIntent,
    inspection: BookInspection,
    *,
    source_bytes_hash: str,
    resolved_voice: Voice | None,
    model_revision: str,
    cast_voices: tuple[Voice, ...] = (),
    assignments: Mapping[str, str] | None = None,
    unknown_voice_id: str | None = None,
    spans: tuple[SpeakerSpan, ...] = (),
) -> ExecutionPlan:
    """Compile supplied material without filesystem, provider, or process effects."""
    source_hash = _validated_hash(source_bytes_hash)
    revision = model_revision.strip()
    if not revision:
        raise ModelError(ErrorCode.INVALID_MODEL_REVISION)

    assigned = _one_operation(pipeline.operations, AssignVoices)
    if assigned is None:
        raise VoiceError(ErrorCode.VOICE_UNRESOLVED)
    if _one_operation(pipeline.operations, SynthesizeSpeech) is None:
        raise ValidationError(ErrorCode.TTS_REQUIRED)
    narrator = _resolve_voice(assigned.narrator_voice_id, resolved_voice, revision)
    # Single voice is the degenerate cast, not a separate path: one renderer
    # and one set of segment identities serve both.
    others = tuple(_resolve_voice(item.id, item, revision) for item in cast_voices)
    by_id = {plan.id: plan for plan in (narrator, *others)}
    cast = CastPlan(
        narrator=narrator,
        unknown=by_id.get(unknown_voice_id or narrator.id, narrator),
        voices=(narrator, *others),
        assignments=dict(assignments or {}),
    )
    voice = narrator

    spoken = _one_operation(pipeline.operations, SpokenForm)
    if spoken is not None and not narrator.language.lower().startswith("en"):
        # The number words and lexicon are English. Mangling a French book is
        # worse than leaving it, so the stage disables itself rather than
        # asking the caller to know this.
        spoken = None

    segments = _compile_segments(inspection.chapters, spans, cast, spoken)
    if not segments:
        raise ValidationError(ErrorCode.EMPTY_SPEECH)
    # Canonical characters, not spoken characters: this is the bill, and it
    # must describe the book the caller supplied.
    total = sum(len(chapter.text) for chapter in inspection.chapters)
    metadata = _output_metadata(
        inspection,
        _one_operation(pipeline.operations, MetadataIntent),
    )
    schemas = SchemaVersions(
        parser=PARSER_SCHEMA_VERSION,
        normalization=NORMALIZATION_SCHEMA_VERSION,
        planning=PLANNING_SCHEMA_VERSION,
        render=RENDER_SCHEMA_VERSION,
        spoken_form=SPOKEN_FORM_VERSION if spoken is not None else None,
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
        cast=cast,
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
    spans: tuple[SpeakerSpan, ...],
    cast_plan: CastPlan,
    spoken: SpokenForm | None = None,
) -> tuple[SpeechSegment, ...]:
    """Split each speaker span while assigning one global plan-order ordinal.

    Spans partition a chapter, and the frozen chunker runs inside each one, so
    concatenating every chunk still reproduces the chapter exactly. A chapter
    with no spans is one narration span, which is byte-for-byte the behaviour
    that existed before attribution.
    """
    result: list[SpeechSegment] = []
    for chapter in chapters:
        for span in _spans_for(chapter, spans):
            voice = cast_plan.voice_for(span.character_id)
            text = chapter.text[span.start : span.end]
            if spoken is not None:
                text = to_spoken(
                    text,
                    numbers=cast("NumberTier", spoken.numbers),
                    lexicon=spoken.lexicon,
                    builtin=spoken.builtin_lexicon,
                )
            for chunk_index, chunk in enumerate(_chunk_span(chapter, text)):
                result.append(
                    _segment(
                        chapter,
                        len(result),
                        chunk_index,
                        chunk,
                        speaker_id=span.character_id,
                        voice_id=voice.id,
                        spoken=spoken,
                    )
                )
    return tuple(result)


def _spans_for(
    chapter: ChapterInspection, spans: tuple[SpeakerSpan, ...]
) -> tuple[SpeakerSpan, ...]:
    """Return a chapter's ordered spans, or one narration span covering it."""
    owned = tuple(span for span in spans if span.chapter_id == chapter.id)
    if not owned:
        return (SpeakerSpan(chapter.id, 0, len(chapter.text), None),)
    ordered = tuple(sorted(owned, key=lambda span: span.start))
    covered = "".join(chapter.text[span.start : span.end] for span in ordered)
    if covered != chapter.text:
        raise ValidationError(ErrorCode.EMPTY_SPEECH)
    return ordered


def _break_offset(text: str, start: int, stop: int) -> int:
    """Offset past the best boundary in ``text[start:stop]``, or 0 when none fits."""
    window = text[start:stop]
    if not window:
        return 0
    threshold = len(window) * MIN_BREAK_FILL
    fullest = 0
    for pattern in _BREAK_TIERS:
        offsets = [match.end() for match in re.finditer(pattern, window)]
        if not offsets:
            continue
        best = offsets[-1]
        if best >= threshold:
            return best
        fullest = max(fullest, best)
    return fullest


def _separator_free_end(text: str, start: int, stop: int) -> int:
    """Return where a run the engine cannot divide outgrows its chunk budget."""
    run_start = start
    for index in range(start, stop):
        if text[index] in POCKET_SEPARATORS:
            run_start = index + 1
        elif index - run_start >= MAX_SEPARATOR_FREE_CHARACTERS:
            return index
    return stop


def _chunk_span(chapter: ChapterInspection, text: str) -> tuple[str, ...]:
    """Apply the frozen tts-chunks-v2 chunker to one speaker span.

    The chapter is still validated as a whole, because speech_characters
    describes the chapter and not the span.
    """
    if (
        not chapter.text
        or chapter.speech_characters is None
        or chapter.speech_characters != len(chapter.text)
        or not text
    ):
        raise ValidationError(ErrorCode.EMPTY_SPEECH)
    chunks: list[str] = []
    start = 0
    while start < len(text):
        hard_end = min(start + MAX_TTS_SEGMENT_CHARACTERS, len(text))
        end = hard_end
        if hard_end < len(text):
            offset = _break_offset(text, start, hard_end)
            if offset:
                end = start + offset
        budget_end = _separator_free_end(text, start, hard_end)
        if budget_end < end:
            # Cutting needs a natural boundary. Without one the character bound
            # still applies, which keeps an unbroken token from being split
            # mid-word into two mispronounced halves.
            forced = _break_offset(text, start, budget_end)
            if forced:
                end = start + forced
        if end <= start:  # Defensive hard fallback for arbitrarily long tokens.
            end = hard_end
        chunks.append(text[start:end])
        start = end
    if not chunks or any(not chunk for chunk in chunks) or "".join(chunks) != text:
        raise ValidationError(ErrorCode.EMPTY_SPEECH)
    return tuple(chunks)


def _segment(  # noqa: PLR0913 - each field is part of a distinct identity.
    chapter: ChapterInspection,
    ordinal: int,
    chunk_index: int,
    text: str,
    *,
    speaker_id: str | None = None,
    voice_id: str = "",
    spoken: SpokenForm | None = None,
) -> SpeechSegment:
    content_hash = _hash_utf8(text)
    fields: dict[str, object] = {
        "chapter_id": _string_identity(chapter.id),
        "chunk_index": chunk_index,
        "chunking_schema": CHUNKING_SCHEMA_VERSION,
        "content_hash": content_hash,
        "normalization": NORMALIZATION_SCHEMA_VERSION,
        "ordinal": ordinal,
        "segment_id_version": _SEGMENT_ID_VERSION,
    }
    if speaker_id is not None:
        # Added only for attributed speech, so single-voice identities -- and
        # therefore every existing cache entry -- stay byte-identical.
        fields["speaker_id"] = _string_identity(speaker_id)
        fields["voice_id"] = _string_identity(voice_id)
    if spoken is not None:
        # Added only when the stage is active, so a plain pipeline's identities
        # -- and therefore every existing cache entry -- stay byte-identical.
        fields.update(
            spoken_identity(
                numbers=cast("NumberTier", spoken.numbers),
                lexicon=spoken.lexicon,
                builtin=spoken.builtin_lexicon,
            )
        )
    identity = json.dumps(fields, sort_keys=True, separators=(",", ":"))
    digest = _hash_utf8(identity)[:24]
    return SpeechSegment(
        id=f"seg-{NORMALIZATION_SCHEMA_VERSION}-{_SEGMENT_ID_VERSION}-{digest}",
        chapter_id=chapter.id,
        ordinal=ordinal,
        text=text,
        character_count=len(text),
        content_hash=content_hash,
        speaker_id=speaker_id,
        voice_id=voice_id,
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
    schema_versions: dict[str, object] = {
        "parser": schemas.parser,
        "normalization": schemas.normalization,
        "planning": schemas.planning,
        "render": schemas.render,
    }
    if schemas.spoken_form is not None:
        # Present only when the stage is active. Emitting an explicit null
        # would change the canonical JSON -- and so the fingerprint -- for
        # every pipeline that never asked for spoken form.
        schema_versions["spoken_form"] = schemas.spoken_form
    payload = {
        "schema_versions": schema_versions,
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
