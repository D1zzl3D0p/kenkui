"""Verified chapter PCM reuse keyed by the complete semantic rendering plan."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from typing import TYPE_CHECKING

from kenkui._tts.protocols import SegmentAudio
from kenkui.checkpoints import current_session

if TYPE_CHECKING:
    from pathlib import Path

    from kenkui._domain.planning import ExecutionPlan
    from kenkui._execution.process_pool import EngineSpecification
    from kenkui._tts.protocols import SynthesisTask
    from kenkui.cancellation import CancellationToken

# Bump when synthesis/padding semantics change without a plan schema change.
CHAPTER_SCHEMA = "chapter-pcm-v1"


def chapter_key(
    plan: ExecutionPlan, chapter_id: str, engine: EngineSpecification
) -> str:
    """Bind PCM to source, text, model revision, voice assets, and speech settings."""
    material = {
        "schema": CHAPTER_SCHEMA,
        "plan": plan.semantic_fingerprint,
        "chapter": chapter_id,
        "engine": engine.kind,
        "config": engine.pocket_config.semantic_material()
        if engine.pocket_config is not None
        else None,
        "voices": [
            asdict(plan.cast.voice_for(segment.speaker_id))
            for segment in plan.segments
            if segment.chapter_id == chapter_id
        ],
    }
    return hashlib.sha256(json.dumps(material, sort_keys=True).encode()).hexdigest()


def restore_chapters(  # noqa: PLR0913 - explicit validation inputs.
    plan: ExecutionPlan,
    tasks: tuple[SynthesisTask, ...],
    directory: Path,
    *,
    max_chapter_bytes: int,
    engine: EngineSpecification,
    cancel: CancellationToken | None,
) -> dict[str, tuple[Path, tuple[SegmentAudio, ...]]]:
    """Restore only chapters whose metadata and exact byte counts match the plan."""
    session = current_session()
    restored: dict[str, tuple[Path, tuple[SegmentAudio, ...]]] = {}
    if session is None:
        return restored
    for index, chapter in enumerate(plan.output.chapters):
        if cancel is not None:
            cancel.raise_if_cancelled()
        path = directory / f"restored-{index:05d}.pcm"
        metadata = session.store.restore(chapter_key(plan, chapter.id, engine), path)
        if metadata is None:
            continue
        try:
            audio = tuple(SegmentAudio(**entry) for entry in metadata["segments"])
            expected = tuple(task for task in tasks if task.chapter_id == chapter.id)
            valid = len(audio) == len(expected) and bool(audio)
            for item, task in zip(audio, expected, strict=True):
                valid = valid and (
                    item.segment_id == task.segment_id
                    and item.chapter_id == task.chapter_id
                    and item.sample_rate_hz == task.sample_rate_hz
                    and item.channels == task.channels
                    and type(item.frame_count) is int
                    and item.frame_count > 0
                    and item.duration_ms
                    == item.frame_count * 1000 // item.sample_rate_hz
                )
            size = sum(item.byte_count for item in audio)
            valid = (
                valid and 0 < size <= max_chapter_bytes and path.stat().st_size == size
            )
        except (KeyError, TypeError, ValueError, OSError, ZeroDivisionError):
            valid = False
        if valid:
            restored[chapter.id] = path, audio
        else:
            path.unlink(missing_ok=True)
    return restored


def save_chapter(
    plan: ExecutionPlan,
    chapter_id: str,
    path: Path,
    audio: tuple[SegmentAudio, ...],
    engine: EngineSpecification,
) -> None:
    """Commit durable audio before reporting a chapter as completed."""
    session = current_session()
    if session is not None:
        session.store.save(
            chapter_key(plan, chapter_id, engine),
            path,
            {"segments": [asdict(item) for item in audio]},
        )
