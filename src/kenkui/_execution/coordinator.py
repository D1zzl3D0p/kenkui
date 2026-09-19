"""Deterministic coordinator with process-isolated synthesis."""

from __future__ import annotations

import hashlib
import logging
import os
import shutil
import stat
import tempfile
from contextlib import suppress
from dataclasses import dataclass, replace
from itertools import pairwise
from pathlib import Path
from typing import TYPE_CHECKING, Literal, cast

from kenkui._audio.cover import read_cover
from kenkui._audio.m4b import (
    ArtifactAssembler,
    AssemblyRequest,
    AssemblyResult,
    ChapterAudio,
    chapter_frame_boundaries_ms,
)
from kenkui._domain.planning import ExecutionPlan, compile_execution_plan
from kenkui._execution.checkpoints import restore_chapters, save_chapter
from kenkui._execution.process_pool import (
    EngineSpecification,
    render_spawned,
    resolve_workers,
)
from kenkui._progress import EventEmitter
from kenkui._source import snapshot_source
from kenkui._tts.fake import FAKE_CHANNELS, FAKE_SAMPLE_RATE_HZ
from kenkui._tts.protocols import (
    SegmentAudio,
    SynthesisTask,
    SynthesizedAudio,
    segment_audio,
)
from kenkui.api import ExecutionStats, Result
from kenkui.errors import (
    EncodingError,
    ErrorCode,
    KenkuiError,
    RenderError,
    SourceError,
)

# The bounds and estimates are public, so a caller can describe a run before
# starting it against the same numbers execution uses.
from kenkui.limits import (
    MAX_SEGMENT_PCM_BYTES,
    MAX_TOTAL_PCM_BYTES,
    estimated_audio_hours,
    is_long_chapter,
)
from kenkui.observability import get_logger, log_event

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator, Mapping

    from kenkui._domain.operations import MetadataIntent
    from kenkui._domain.planning import SpeakerSpan
    from kenkui._execution.cache import CacheStore, _RunContext
    from kenkui._execution.process_pool import WorkerRecord
    from kenkui.cancellation import CancellationToken
    from kenkui.events import ExecutionEvent
    from kenkui.pipeline import Pipeline
    from kenkui.voices import Voice

_READ_CHUNK_BYTES = 64 * 1024
_SHA256_HEX_LENGTH = 64
_PRIVATE_DIRECTORY_MODE = 0o700
MAX_ARTIFACT_BYTES = 16 * 1024 * 1024 * 1024
_LOGGER = get_logger(__name__)


@dataclass(frozen=True, slots=True)
class ExecutionBindings:
    """Private injectable production-resource bindings."""

    engine_specification: EngineSpecification
    assembler: ArtifactAssembler
    voice: Voice
    model_revision: str
    cache_store: CacheStore | None = None
    # The rest of the cast, if any. The narrator stays in `voice` because its
    # engine and revision govern the run.
    cast_voices: tuple[Voice, ...] = ()


def execute_sequential(  # noqa: PLR0913, PLR0915 - explicit orchestration boundary.
    pipeline: Pipeline,
    output: Path,
    *,
    bindings: ExecutionBindings,
    on_event: Callable[[ExecutionEvent], None] | None,
    cancel: CancellationToken | None,
    workers: int | Literal["auto"],
    overwrite: bool,
    keep_audio_cache: bool,
    assignments: Mapping[str, str] | None = None,
    unknown_voice_id: str | None = None,
    spans: tuple[SpeakerSpan, ...] = (),
    resolved_source_hash: str | None = None,
    emitter: EventEmitter | None = None,
    preview: bool = False,
) -> Result:
    """Execute one immutable plan in order and transactionally publish its artifact."""
    metadata_intent = pipeline.metadata_intent
    cover_file, cover_content_hash = (
        (None, None) if preview else _resolve_cover(metadata_intent)
    )
    _preflight_assembler(
        bindings.assembler,
        expect_cover=not preview
        and (
            metadata_intent is None
            or metadata_intent.cover == "source"
            or cover_file is not None
        ),
    )
    owns_emitter = emitter is None
    emitter = emitter if emitter is not None else EventEmitter(on_event)
    workspace = _make_workspace(output.parent)
    try:
        _check_cancel(cancel)
        if owns_emitter:
            emitter.emit_started()
        _check_cancel(cancel)

        emitter.emit_stage_started("planning")
        _check_cancel(cancel)
        snapshot = workspace / "source.epub"
        source_hash = snapshot_source(pipeline.source.path, snapshot, cancel)
        if resolved_source_hash is not None and source_hash != resolved_source_hash:
            raise SourceError(ErrorCode.SOURCE_CHANGED)
        _check_cancel(cancel)
        snapshot_pipeline = replace(
            pipeline,
            source=replace(pipeline.source, path=snapshot),
            _resolved=None,
            _roster=None,
        )
        inspection = snapshot_pipeline.inspect()
        _check_cancel(cancel)
        plan = compile_execution_plan(
            pipeline,
            inspection,
            source_bytes_hash=source_hash,
            resolved_voice=bindings.voice,
            model_revision=bindings.model_revision,
            cast_voices=bindings.cast_voices,
            assignments=assignments,
            unknown_voice_id=unknown_voice_id,
            spans=spans,
            cover_content_hash=cover_content_hash,
        )
        # An unusually long chapter is remarked on here, from the plan's own
        # character counts, while the run is still cheap to abandon. It is a
        # report, not a refusal: the chapter renders either way.
        _warn_about_long_chapters(plan, emitter)
        emitter.emit_cast_resolved(plan)
        # Emitted before the worker pool exists, so cancelling from the
        # callback costs the attribution already paid for and no rendering.
        _check_cancel(cancel)
        if bindings.engine_specification.kind == "pocket":
            config = bindings.engine_specification.pocket_config
            if config is None:
                raise RenderError(ErrorCode.SYNTHESIS_FAILED)
            from kenkui._tts.pocket import preflight_pocket  # noqa: PLC0415

            preflight_pocket(config, plan.cast.narrator, plan.model_revision)
        cache_context = (
            bindings.cache_store.prepare_run(plan)
            if bindings.cache_store is not None
            else None
        )
        emitter.emit_progress("planning", 1, 1)
        _check_cancel(cancel)
        emitter.emit_stage_completed("planning")
        _check_cancel(cancel)

        audio, pcm_parts = _render(
            plan,
            bindings.engine_specification,
            resolve_workers(workers, len(plan.segments)),
            emitter,
            cancel,
            workspace=workspace,
            cache_store=bindings.cache_store,
            cache_context=cache_context,
        )
        assembled = _assemble(
            plan,
            audio,
            pcm_parts,
            snapshot,
            workspace,
            bindings.assembler,
            emitter,
            cancel,
            cover_file,
        )
        result = _build_result(output, plan, audio, assembled)

        # Publication start/progress remain observable, fatal pre-commit events.
        emitter.emit_stage_started("publication")
        _check_cancel(cancel)
        emitter.emit_progress("publication", 1, 1)
        _check_cancel(cancel)

        # Freeze and validate the final private bytes, then make one last
        # cancellation decision. No observer runs from here through commit.
        publication = _snapshot_publication(assembled, workspace)
        _check_cancel(cancel)
        _publish(publication, output, overwrite=overwrite)

        # Publication is the commit. Nothing after this point may revoke success.
        # The render is done and published, so its audio cache has served its
        # purpose; release it unless the caller opted to keep it. Fail-open.
        if bindings.cache_store is not None and not keep_audio_cache:
            bindings.cache_store.clear_book(plan.source_bytes_hash)

        emitter.emit_stage_completed_best_effort("publication")
        emitter.emit_completed_best_effort()
        return result  # noqa: TRY300 - commit success remains in the guarded scope.
    except KenkuiError as error:
        # Deliberately omit traceback/context because backends may contain secrets.
        log_event(
            _LOGGER,
            "execution_failed",
            level=logging.ERROR,
            context={"boundary": "terminal_error", "code": error.code.value},
        )
        raise
    finally:
        shutil.rmtree(workspace, ignore_errors=True)


def _resolve_cover(
    metadata_intent: MetadataIntent | None,
) -> tuple[Path | None, str | None]:
    """Validate a caller-supplied cover before anything expensive starts.

    Resolved here rather than at assembly so an unreadable image fails while
    no worker is running and nothing has been written.
    """
    if metadata_intent is None or not isinstance(metadata_intent.cover, Path):
        return None, None
    return metadata_intent.cover, read_cover(metadata_intent.cover)[1]


def _make_workspace(parent: Path) -> Path:
    failed = False
    workspace_name = ""
    try:
        workspace_name = tempfile.mkdtemp(prefix=".kenkui-", dir=parent)
    except OSError:
        failed = True
    if failed:
        raise EncodingError(ErrorCode.PUBLICATION_FAILED)
    return Path(workspace_name)


def _build_result(
    output: Path,
    plan: ExecutionPlan,
    audio: tuple[SegmentAudio, ...],
    assembled: AssemblyResult,
) -> Result:
    stats = ExecutionStats(
        normalized_speech_characters=plan.total_speech_characters,
        synthesized_characters=sum(
            segment.character_count for segment in plan.segments
        ),
        synthesized_segments=len(audio),
        rendered_chapters=len(assembled.chapters),
        duration_ms=assembled.duration_ms,
    )
    return Result(output, stats)


def _cache_lookup(
    plan: ExecutionPlan,
    tasks: tuple[SynthesisTask, ...],
    engine_specification: EngineSpecification,
    cancel: CancellationToken | None,
    cache_store: CacheStore | None,
) -> tuple[dict[int, SynthesizedAudio], list[int], dict[int, str]]:
    """Resolve which segments are already rendered before any worker starts."""
    cached: dict[int, SynthesizedAudio] = {}
    miss_indices: list[int] = []
    cache_keys: dict[int, str] = {}
    if cache_store is None:
        miss_indices.extend(range(len(tasks)))
        return cached, miss_indices, cache_keys
    for index, (segment, task) in enumerate(zip(plan.segments, tasks, strict=True)):
        _check_cancel(cancel)
        key = cache_store.key_for(plan, segment, task, engine_specification)
        cache_keys[index] = key
        item = cache_store.lookup(key, segment, task)
        if item is None:
            miss_indices.append(index)
            log_event(_LOGGER, "cache_miss", context={"boundary": "cache"})
        else:
            cached[index] = item
            log_event(_LOGGER, "cache_hit", context={"boundary": "cache"})
    return cached, miss_indices, cache_keys


def _render(  # noqa: PLR0912, PLR0913, PLR0915, C901 - ordered synthesis and durable chapter commit.
    plan: ExecutionPlan,
    engine_specification: EngineSpecification,
    worker_count: int,
    emitter: EventEmitter,
    cancel: CancellationToken | None,
    *,
    workspace: Path,
    cache_store: CacheStore | None = None,
    cache_context: _RunContext | None = None,
) -> tuple[tuple[SegmentAudio, ...], tuple[Path, ...]]:
    emitter.emit_stage_started("render")
    _check_cancel(cancel)
    sample_rate_hz, channels = _audio_shape(engine_specification)
    tasks = tuple(
        SynthesisTask(
            segment.id,
            segment.chapter_id,
            segment.text,
            sample_rate_hz,
            channels,
            MAX_SEGMENT_PCM_BYTES,
            # VoicePlan.content_fingerprint is the asset digest the worker
            # routes its conditioning state by, so this bridge is a lookup
            # rather than new state.
            plan.cast.voice_for(segment.speaker_id).content_fingerprint,
        )
        for segment in plan.segments
    )
    cached, miss_indices, cache_keys = _cache_lookup(
        plan, tasks, engine_specification, cancel, cache_store
    )
    silence = plan.trailing_silence_ms or (0,) * len(plan.segments)
    rendered: list[SegmentAudio] = []
    parts: list[Path] = []
    spill: _ChapterSpill | None = None
    total_bytes = 0
    spill_directory = _fresh_spill_directory(workspace)
    chapter_ids = tuple(chapter.id for chapter in plan.output.chapters)
    last_segment = {
        chapter_id: max(
            index
            for index, segment in enumerate(plan.segments)
            if segment.chapter_id == chapter_id
        )
        for chapter_id in chapter_ids
    }
    restored = restore_chapters(
        plan,
        tasks,
        spill_directory,
        max_bytes=MAX_TOTAL_PCM_BYTES,
        engine=engine_specification,
        cancel=cancel,
    )
    miss_indices = [i for i in miss_indices if tasks[i].chapter_id not in restored]
    completed_chapters = 0
    chapter_start = 0
    missing_tasks = tuple(tasks[index] for index in miss_indices)
    records: Iterator[WorkerRecord] = iter(())
    if missing_tasks:
        records = render_spawned(
            missing_tasks,
            engine_specification,
            min(worker_count, len(missing_tasks)),
            cancel,
            max_total_bytes=MAX_TOTAL_PCM_BYTES,
        )
    try:
        for index, (segment, task) in enumerate(zip(plan.segments, tasks, strict=True)):
            _check_cancel(cancel)
            if segment.chapter_id in restored:
                if last_segment[segment.chapter_id] == index:
                    part, entries = restored[segment.chapter_id]
                    total_bytes += sum(entry.byte_count for entry in entries)
                    if total_bytes > MAX_TOTAL_PCM_BYTES:
                        raise RenderError(ErrorCode.BOOK_TOO_LONG)
                    rendered.extend(entries)
                    parts.append(part)
                    chapter_start = len(rendered)
                    completed_chapters += 1
                    emitter.emit_progress(
                        "render",
                        completed_chapters,
                        len(chapter_ids),
                        segment.chapter_id,
                    )
                continue
            item = cached.get(index)
            if item is None:
                item = next(records).audio
            item_bytes = _validate_audio(task, item)
            # Padding is applied only after the raw worker output has been
            # validated, so every existing audio invariant still governs what the
            # worker actually sent.
            entry, padding = _padded(segment_audio(item), silence[index])
            item_bytes += len(padding)
            total_bytes += item_bytes
            # A chapter has no length of its own to exceed: its samples stream
            # to disk. Only the run's bound on untrusted worker output remains.
            if total_bytes > MAX_TOTAL_PCM_BYTES:
                raise RenderError(ErrorCode.BOOK_TOO_LONG)
            rendered.append(entry)
            if spill is None:
                spill = _ChapterSpill(spill_directory, len(parts), segment.chapter_id)
            elif spill.chapter_id != segment.chapter_id:
                # A plan orders every chapter's segments together. Interleaving
                # would write two chapters' samples into one part.
                raise RenderError(ErrorCode.SYNTHESIS_FAILED)
            # Samples reach the chapter's part as they arrive, so the run holds
            # one segment rather than a whole chapter however long it runs.
            spill.write(item.pcm_s16le)
            if padding:
                spill.write(padding)
            if cache_store is not None and index in cache_keys and index not in cached:
                cache_store.store(cache_keys[index], segment, task, item, cache_context)
            if last_segment[segment.chapter_id] == index:
                parts.append(spill.commit())
                spill = None
                save_chapter(
                    plan,
                    segment.chapter_id,
                    parts[-1],
                    tuple(rendered[chapter_start:]),
                    engine_specification,
                )
                chapter_start = len(rendered)
                completed_chapters += 1
                emitter.emit_progress(
                    "render", completed_chapters, len(chapter_ids), segment.chapter_id
                )
            _check_cancel(cancel)
        if spill is not None or len(parts) != len(chapter_ids):
            raise RenderError(ErrorCode.SYNTHESIS_FAILED)
    finally:
        # An abandoned chapter's part is never assembled or checkpointed, so it
        # leaves with the run that failed rather than filling the workspace.
        if spill is not None:
            spill.discard()
    emitter.emit_stage_completed("render")
    _check_cancel(cancel)
    return tuple(rendered), tuple(parts)


def _fresh_spill_directory(workspace: Path) -> Path:
    """Create the private directory that holds this run's chapter PCM parts."""
    try:
        directory = Path(tempfile.mkdtemp(prefix=".render-", dir=workspace))
        directory.chmod(_PRIVATE_DIRECTORY_MODE)
    except OSError:
        raise RenderError(ErrorCode.SYNTHESIS_FAILED) from None
    return directory


def _padded(item: SegmentAudio, silence_ms: int) -> tuple[SegmentAudio, bytes]:
    """Extend one segment's metadata and produce its trailing silence bytes.

    SegmentAudio.byte_count is derived from frame_count, so this single
    adjustment keeps the assembler's part-size check, the chapter markers, and
    the reported duration in agreement without touching either.
    """
    if silence_ms <= 0:
        return item, b""
    frames = silence_ms * item.sample_rate_hz // 1000
    if frames <= 0:
        return item, b""
    total = item.frame_count + frames
    extended = replace(
        item,
        frame_count=total,
        duration_ms=total * 1000 // item.sample_rate_hz,
    )
    return extended, bytes(frames * item.channels * 2)


class _ChapterSpill:
    """One chapter's part, written as its segments arrive rather than at its end.

    Streaming is what lets a chapter be any length: the samples a chapter has
    already produced live in its part file, so only the segment in hand is in
    memory. A part is offered to assembly and checkpointing only once committed.
    """

    __slots__ = ("_stream", "chapter_id", "path")

    def __init__(self, directory: Path, index: int, chapter_id: str) -> None:
        """Create this chapter's private part, failing if one already exists."""
        self.chapter_id = chapter_id
        self.path = directory / f"chapter-{index:05d}.pcm"
        descriptor = -1
        try:
            flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0)
            descriptor = os.open(self.path, flags, 0o600)
            # Ownership passes to the stream, which closes it on every path.
            self._stream = os.fdopen(descriptor, "wb")
        except OSError:
            if descriptor >= 0:
                os.close(descriptor)
            raise RenderError(ErrorCode.SYNTHESIS_FAILED) from None

    def write(self, payload: bytes) -> None:
        """Append one validated segment's samples, or its trailing silence."""
        try:
            self._stream.write(payload)
        except OSError:
            raise RenderError(ErrorCode.SYNTHESIS_FAILED) from None

    def commit(self) -> Path:
        """Fsync and close, so the chapter is durable before it is reported."""
        try:
            self._stream.flush()
            os.fsync(self._stream.fileno())
            self._stream.close()
        except OSError:
            raise RenderError(ErrorCode.SYNTHESIS_FAILED) from None
        return self.path

    def discard(self) -> None:
        """Drop an unfinished chapter's bytes; its run is already failing."""
        with suppress(OSError):
            self._stream.close()
        with suppress(OSError):
            self.path.unlink(missing_ok=True)


def _audio_shape(engine_specification: EngineSpecification) -> tuple[int, int]:
    """Return the sample rate and channel count this engine renders at."""
    if (
        engine_specification.kind == "pocket"
        and engine_specification.pocket_config is not None
    ):
        return engine_specification.pocket_config.sample_rate_hz, 1
    return FAKE_SAMPLE_RATE_HZ, FAKE_CHANNELS


def _warn_about_long_chapters(plan: ExecutionPlan, emitter: EventEmitter) -> None:
    """Report each unusually long chapter without standing in its way.

    A caller that watches events can stop while stopping is still free; one that
    does not gets the same observation in the log and its audiobook regardless.
    """
    for chapter in plan.output.chapters:
        if not is_long_chapter(chapter.speech_characters):
            continue
        hours = estimated_audio_hours(chapter.speech_characters)
        log_event(
            _LOGGER,
            "long_chapter",
            level=logging.WARNING,
            context={
                "boundary": "planning",
                "chapter_id": chapter.id,
                "speech_characters": chapter.speech_characters,
                "estimated_hours": f"{hours:.1f}",
            },
        )
        emitter.emit_warning(
            "planning",
            "long_chapter",
            f"The chapter {chapter.title!r} is about {hours:.0f} hours of "
            f"audio ({chapter.speech_characters:,} speech characters). It will "
            "render; deselect or split it if that is not what you meant.",
            chapter.id,
        )


def _preflight_assembler(assembler: ArtifactAssembler, *, expect_cover: bool) -> None:
    """Fail before source snapshot or synthesis and sanitize backend details."""
    failure: ErrorCode | None = None
    try:
        assembler.preflight(expect_cover=expect_cover)
    except EncodingError as error:
        failure = error.code
    except Exception:  # noqa: BLE001 - arbitrary private backend is untrusted.
        failure = ErrorCode.ASSEMBLY_FAILED
    if failure is not None:
        raise EncodingError(failure) from None


def _assemble(  # noqa: PLR0913, PLR0917 - explicit effect boundary.
    plan: ExecutionPlan,
    audio: tuple[SegmentAudio, ...],
    pcm_parts: tuple[Path, ...],
    source_snapshot: Path,
    workspace: Path,
    assembler: ArtifactAssembler,
    emitter: EventEmitter,
    cancel: CancellationToken | None,
    cover_file: Path | None = None,
) -> AssemblyResult:
    emitter.emit_stage_started("assembly")
    _check_cancel(cancel)
    candidate = _fresh_assembly_candidate(workspace)
    failed = False
    encoding_failure: ErrorCode | None = None
    result: object | None = None
    try:
        result = assembler.assemble(
            AssemblyRequest(
                plan, audio, pcm_parts, candidate, source_snapshot, cover_file
            )
        )
    except EncodingError as error:
        encoding_failure = error.code
    except Exception:  # noqa: BLE001 - sanitize arbitrary assembler failures.
        failed = True
    if encoding_failure is not None:
        _check_cancel(cancel)
        raise EncodingError(encoding_failure) from None
    if failed or result is None:
        _check_cancel(cancel)
        raise EncodingError(ErrorCode.ASSEMBLY_FAILED) from None
    result = _validate_assembly(result, candidate, workspace, plan, audio)
    emitter.emit_progress("assembly", 1, 1)
    _check_cancel(cancel)
    emitter.emit_stage_completed("assembly")
    _check_cancel(cancel)
    return result


def _fresh_assembly_candidate(workspace: Path) -> Path:
    """Atomically create a private unpredictable directory, leaving output absent."""
    try:
        directory = Path(tempfile.mkdtemp(prefix=".assembly-", dir=workspace))
        directory.chmod(_PRIVATE_DIRECTORY_MODE)
        metadata = directory.lstat()
        candidate = directory / "candidate.m4b"
        _require_private_candidate_directory(metadata, candidate)
    except OSError:
        raise EncodingError(ErrorCode.ASSEMBLY_FAILED) from None
    return candidate


def _require_private_candidate_directory(
    metadata: os.stat_result, candidate: Path
) -> None:
    if (
        not stat.S_ISDIR(metadata.st_mode)
        or stat.S_IMODE(metadata.st_mode) != _PRIVATE_DIRECTORY_MODE
        or candidate.exists()
        or candidate.is_symlink()
    ):
        raise OSError


def _validate_audio(task: SynthesisTask, audio: SynthesizedAudio) -> int:
    if type(audio) is not SynthesizedAudio:
        raise RenderError(ErrorCode.INVALID_AUDIO)
    exact_ints = (
        audio.sample_rate_hz,
        audio.channels,
        audio.frame_count,
        audio.duration_ms,
    )
    if (
        type(audio.segment_id) is not str
        or type(audio.chapter_id) is not str
        or type(audio.pcm_s16le) is not bytes
        or any(type(value) is not int for value in exact_ints)
        or any(value <= 0 for value in exact_ints)
    ):
        raise RenderError(ErrorCode.INVALID_AUDIO)
    expected_bytes = audio.frame_count * audio.channels * 2
    expected_duration = audio.frame_count * 1000 // audio.sample_rate_hz
    if (
        audio.segment_id != task.segment_id
        or audio.chapter_id != task.chapter_id
        or audio.sample_rate_hz != task.sample_rate_hz
        or audio.channels != task.channels
        or expected_bytes > task.max_output_bytes
        or expected_bytes > MAX_SEGMENT_PCM_BYTES
        or len(audio.pcm_s16le) != expected_bytes
        or audio.duration_ms != expected_duration
    ):
        raise RenderError(ErrorCode.INVALID_AUDIO)
    return expected_bytes


def _validate_assembly(
    result: object,
    candidate: Path,
    workspace: Path,
    plan: ExecutionPlan,
    audio: tuple[SegmentAudio, ...],
) -> AssemblyResult:
    if type(result) is not AssemblyResult:
        _reject_assembly(candidate)
    typed_result = cast("AssemblyResult", result)
    try:
        integer_metadata = (
            typed_result.size_bytes,
            typed_result.sample_rate_hz,
            typed_result.channels,
            typed_result.duration_ms,
        )
        valid = (
            type(typed_result.artifact) is type(candidate)
            and typed_result.artifact == candidate
            and all(type(value) is int and value > 0 for value in integer_metadata)
            and type(typed_result.artifact_sha256) is str
            and len(typed_result.artifact_sha256) == _SHA256_HEX_LENGTH
            and all(
                character in "0123456789abcdef"
                for character in typed_result.artifact_sha256
            )
            and type(typed_result.chapters) is tuple
            and all(type(item) is ChapterAudio for item in typed_result.chapters)
            and all(
                type(item.chapter_id) is str
                and type(item.duration_ms) is int
                and item.duration_ms > 0
                for item in typed_result.chapters
            )
            and type(audio) is tuple
            and all(type(item) is SegmentAudio for item in audio)
            and all(
                type(item.segment_id) is str
                and type(item.chapter_id) is str
                and type(item.sample_rate_hz) is int
                and type(item.channels) is int
                and type(item.frame_count) is int
                and type(item.duration_ms) is int
                for item in audio
            )
        )
    except Exception:  # noqa: BLE001 - missing backend slots are untrusted.
        valid = False
    if not valid:
        _reject_assembly(candidate)

    try:
        size, digest = _read_candidate(candidate, workspace)
        expected_chapters = tuple(chapter.id for chapter in plan.output.chapters)
        boundaries = chapter_frame_boundaries_ms(plan, audio)
        expected_durations = tuple(end - start for start, end in pairwise(boundaries))
        valid = (
            typed_result.size_bytes == size
            and typed_result.artifact_sha256 == digest
            and typed_result.sample_rate_hz == audio[0].sample_rate_hz
            and typed_result.channels == audio[0].channels
            and typed_result.duration_ms == boundaries[-1]
            and tuple(item.chapter_id for item in typed_result.chapters)
            == expected_chapters
            and tuple(item.duration_ms for item in typed_result.chapters)
            == expected_durations
        )
    except Exception:  # noqa: BLE001 - malformed backend values are untrusted.
        valid = False
    if not valid:
        _reject_assembly(candidate)
    return typed_result


def _reject_assembly(candidate: Path) -> None:
    _discard_candidate(candidate)
    raise EncodingError(ErrorCode.ASSEMBLY_FAILED) from None


def _discard_candidate(candidate: Path) -> None:
    with suppress(OSError):
        candidate.unlink(missing_ok=True)


def _file_identity(metadata: os.stat_result) -> tuple[int, int, int, int, int]:
    return (
        metadata.st_dev,
        metadata.st_ino,
        metadata.st_size,
        metadata.st_mtime_ns,
        metadata.st_ctime_ns,
    )


def _open_verified_candidate(candidate: Path, workspace: Path) -> int:
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(candidate, flags)
    try:
        opened = os.fstat(descriptor)
        named = candidate.lstat()
        valid = (
            stat.S_ISREG(opened.st_mode)
            and not stat.S_ISLNK(named.st_mode)
            and opened.st_nlink == 1
            and opened.st_dev == named.st_dev
            and opened.st_ino == named.st_ino
            and 0 < opened.st_size <= MAX_ARTIFACT_BYTES
            and candidate.resolve(strict=True).is_relative_to(
                workspace.resolve(strict=True)
            )
        )
        if not valid:
            raise OSError  # noqa: TRY301 - normalized at this boundary.
    except Exception:
        os.close(descriptor)
        raise
    return descriptor


def _read_candidate(candidate: Path, workspace: Path) -> tuple[int, str]:
    descriptor = _open_verified_candidate(candidate, workspace)
    digest = hashlib.sha256()
    copied = 0
    with os.fdopen(descriptor, "rb") as source:
        before = os.fstat(source.fileno())
        while chunk := source.read(_READ_CHUNK_BYTES):
            copied += len(chunk)
            if copied > MAX_ARTIFACT_BYTES:
                raise OSError
            digest.update(chunk)
        after = os.fstat(source.fileno())
    if _file_identity(before) != _file_identity(after) or copied != after.st_size:
        raise OSError
    return copied, digest.hexdigest()


def _snapshot_publication(result: AssemblyResult, workspace: Path) -> Path:
    descriptor = -1
    snapshot: Path | None = None
    try:
        descriptor, name = tempfile.mkstemp(
            prefix=".publication-", suffix=".m4b", dir=workspace
        )
        snapshot = Path(name)
        source_descriptor = _open_verified_candidate(result.artifact, workspace)
        digest = hashlib.sha256()
        copied = 0
        with (
            os.fdopen(source_descriptor, "rb") as source,
            os.fdopen(descriptor, "wb") as target,
        ):
            descriptor = -1
            before = os.fstat(source.fileno())
            while chunk := source.read(_READ_CHUNK_BYTES):
                copied += len(chunk)
                if copied > MAX_ARTIFACT_BYTES or copied > result.size_bytes:
                    raise OSError  # noqa: TRY301 - normalized at this boundary.
                target.write(chunk)
                digest.update(chunk)
            target.flush()
            os.fsync(target.fileno())
            after = os.fstat(source.fileno())
        if (
            _file_identity(before) != _file_identity(after)
            or copied != result.size_bytes
            or digest.hexdigest() != result.artifact_sha256
        ):
            raise OSError  # noqa: TRY301 - normalized at this boundary.
        metadata = snapshot.lstat()
        if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
            raise OSError  # noqa: TRY301 - normalized at this boundary.
    except Exception:  # noqa: BLE001 - candidate and filesystem are untrusted.
        if descriptor >= 0:
            os.close(descriptor)
        if snapshot is not None:
            snapshot.unlink(missing_ok=True)
        raise EncodingError(ErrorCode.ASSEMBLY_FAILED) from None
    return snapshot


def _publish(candidate: Path, output: Path, *, overwrite: bool) -> None:
    failed = False
    exists = False
    try:
        if overwrite:
            candidate.replace(output)
        else:
            os.link(candidate, output, follow_symlinks=False)
    except FileExistsError:
        exists = True
    except OSError:
        failed = True
    if exists:
        raise EncodingError(ErrorCode.OUTPUT_EXISTS)
    if failed:
        raise EncodingError(ErrorCode.PUBLICATION_FAILED)


def _check_cancel(cancel: CancellationToken | None) -> None:
    if cancel is not None:
        cancel.raise_if_cancelled()
