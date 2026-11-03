"""Immutable renderer-neutral plan values shared by compilation and execution."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from types import MappingProxyType
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from collections.abc import Mapping

    from kenkui.announcements import ChapterAnnouncement


class CoverIntent(StrEnum):
    """Renderer-neutral choice for source cover inheritance."""

    SOURCE = "source"
    FILE = "file"
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

    def __post_init__(self) -> None:
        """Snapshot assignments so later caller edits cannot change speech."""
        object.__setattr__(
            self, "assignments", MappingProxyType(dict(self.assignments))
        )

    def __reduce__(
        self,
    ) -> tuple[
        type[CastPlan],
        tuple[VoicePlan, VoicePlan, tuple[VoicePlan, ...], dict[str, str]],
    ]:
        """Reconstruct the immutable snapshot in spawned render processes."""
        return type(self), (
            self.narrator,
            self.unknown,
            self.voices,
            dict(self.assignments),
        )

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
    source_kind: Literal["source", "chapter_title"] = "source"


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
    # The content digest, never the path: core spec 14 requires that where an
    # output lives cannot change what it means.
    cover_content_hash: str | None = None


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
    # One entry per segment: the gap AFTER that segment, in milliseconds.
    # Excluded from segment identity -- silence never reaches a worker or the
    # cache, so retuning a duration costs no re-synthesis.
    trailing_silence_ms: tuple[int, ...] = ()


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
    trailing_silence: tuple[int, ...] = ()
    preparation_identity: str | None = None
    announcements: tuple[ChapterAnnouncement, ...] = ()
