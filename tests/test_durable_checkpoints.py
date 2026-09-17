"""Recovery across fresh attempts without repeating paid synthesis or attribution."""

# ruff: noqa: D102, D103, PLR2004, ANN401, TC003, TC001
from __future__ import annotations

import shutil
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from contextvars import copy_context
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

import kenkui as kk
from kenkui._audio.m4b import AssemblyRequest, AssemblyResult, FakeArtifactAssembler
from kenkui._characters import store as casting
from kenkui._execution.checkpoints import chapter_key
from kenkui._execution.process_pool import (
    EngineSpecification,
    WorkerRecord,
    render_spawned,
)
from kenkui._tts.protocols import SynthesisTask
from kenkui.checkpoints import checkpointing
from test_execution import _bind, _pipeline


class Files:
    """Durable fake surviving destruction of every per-attempt workspace."""

    def __init__(self, root: Path) -> None:
        """Create an isolated store."""
        self.root = root
        root.mkdir()
        self.metadata: dict[str, dict[str, Any]] = {}

    def restore(self, key: str, destination: Path) -> dict[str, Any] | None:
        if key not in self.metadata:
            return None
        shutil.copyfile(self.root / key, destination)
        return self.metadata[key]

    def save(self, key: str, source: Path, metadata: dict[str, Any]) -> None:
        shutil.copyfile(source, self.root / key)
        self.metadata[key] = metadata


def test_interrupted_run_reuses_completed_chapter(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pipeline, _, _ = _pipeline(tmp_path)
    _bind(monkeypatch, EngineSpecification.fake(), FakeArtifactAssembler())
    files = Files(tmp_path / "durable")
    output = tmp_path / "book.m4b"

    def interrupt(event: kk.ExecutionEvent) -> None:
        if isinstance(event, kk.StageProgress) and event.stage == "render":
            raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt), checkpointing(files):
        pipeline.write(output, on_event=interrupt, workers=1)
    assert not output.exists()
    submitted: list[SynthesisTask] = []
    original = render_spawned

    def record(
        tasks: tuple[SynthesisTask, ...], *args: Any, **kwargs: Any
    ) -> Iterator[WorkerRecord]:
        submitted.extend(tasks)
        return original(tasks, *args, **kwargs)

    monkeypatch.setattr("kenkui._execution.coordinator.render_spawned", record)
    with checkpointing(files):
        pipeline.write(output, workers=1)
    assert len(submitted) == 1
    assert submitted[0].chapter_id == pipeline.inspect().chapters[1].id
    expected = tmp_path / "fresh.m4b"
    pipeline.write(expected, workers=1)
    assert output.read_bytes() == expected.read_bytes()


def test_completed_synthesis_resumes_assembly_without_workers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pipeline, _, _ = _pipeline(tmp_path)
    _bind(monkeypatch, EngineSpecification.fake(), FakeArtifactAssembler())
    files = Files(tmp_path / "durable")

    def interrupt(event: kk.ExecutionEvent) -> None:
        if (
            isinstance(event, kk.StageProgress)
            and event.stage == "render"
            and event.completed == 2
        ):
            raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt), checkpointing(files):
        pipeline.write(tmp_path / "first.m4b", on_event=interrupt)

    def forbidden(*_args: Any, **_kwargs: Any) -> None:
        pytest.fail("completed chapters must not start synthesis workers")

    monkeypatch.setattr("kenkui._execution.coordinator.render_spawned", forbidden)
    with checkpointing(files):
        pipeline.write(tmp_path / "resumed.m4b")
    assert (tmp_path / "resumed.m4b").exists()


def test_attribution_database_survives_new_session_and_is_isolated(
    tmp_path: Path,
) -> None:
    files = Files(tmp_path / "durable")
    with checkpointing(files):
        first = casting.default_store_path()
        casting.write_response("prompt-key", "model", {"answer": "speaker"})
        casting.write_identity("identity-key", "model", {"pairs": [], "excluded": []})
    assert not first.exists()
    with checkpointing(files):
        assert casting.read_response("prompt-key") == {"answer": "speaker"}
        assert casting.read_identity("identity-key") == {"pairs": [], "excluded": []}
    with checkpointing(Files(tmp_path / "other-job")):
        assert casting.read_response("prompt-key") is None


def test_changed_plan_does_not_reuse_chapters(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pipeline, _, _ = _pipeline(tmp_path)
    _bind(monkeypatch, EngineSpecification.fake(), FakeArtifactAssembler())
    files = Files(tmp_path / "durable")
    with checkpointing(files):
        pipeline.write(tmp_path / "first.m4b")
    submitted: list[SynthesisTask] = []
    original = render_spawned

    def record(
        tasks: tuple[SynthesisTask, ...], *args: Any, **kwargs: Any
    ) -> Iterator[WorkerRecord]:
        submitted.extend(tasks)
        return original(tasks, *args, **kwargs)

    monkeypatch.setattr("kenkui._execution.coordinator.render_spawned", record)
    with checkpointing(files):
        kk.epub(pipeline.source.path).assign_voice("narrator").pauses(
            chapter_ms=2000
        ).tts().write(tmp_path / "changed.m4b")
    assert len(submitted) == 2


def test_concurrent_attribution_writes_survive_restart(tmp_path: Path) -> None:
    files = Files(tmp_path / "durable")
    with checkpointing(files), ThreadPoolExecutor(max_workers=4) as pool:
        futures = [
            pool.submit(
                copy_context().run, casting.write_response, str(i), "model", {"i": i}
            )
            for i in range(8)
        ]
        for future in futures:
            future.result()
    with checkpointing(files):
        for i in range(8):
            assert casting.read_response(str(i)) == {"i": i}


def test_durable_attribution_upload_failure_is_not_swallowed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    files = Files(tmp_path / "durable")

    def fail(*_args: Any) -> None:
        raise OSError

    monkeypatch.setattr(files, "save", fail)
    with checkpointing(files), pytest.raises(RuntimeError, match="checkpoint"):
        casting.write_response("key", "model", {"answer": "speaker"})


def test_checkpoint_key_changes_when_resolved_voice_changes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class CheckingAssembler(FakeArtifactAssembler):
        def assemble(self, request: AssemblyRequest) -> AssemblyResult:
            plan = request.plan
            chapter = plan.output.chapters[0].id
            engine = EngineSpecification.fake()
            changed = replace(
                plan,
                cast=replace(
                    plan.cast,
                    narrator=replace(plan.cast.narrator, content_fingerprint="b" * 64),
                ),
            )
            assert chapter_key(plan, chapter, engine) != chapter_key(
                changed, chapter, engine
            )
            return super().assemble(request)

    pipeline, _, _ = _pipeline(tmp_path)
    _bind(monkeypatch, EngineSpecification.fake(), CheckingAssembler())
    pipeline.write(tmp_path / "output.m4b")


def test_invalid_checkpoint_metadata_is_not_used(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pipeline, _, _ = _pipeline(tmp_path)
    _bind(monkeypatch, EngineSpecification.fake(), FakeArtifactAssembler())
    files = Files(tmp_path / "durable")
    with checkpointing(files):
        pipeline.write(tmp_path / "first.m4b")
    for metadata in files.metadata.values():
        if "segments" in metadata:
            metadata["segments"][0]["frame_count"] = -1
    submitted: list[SynthesisTask] = []

    def record(
        tasks: tuple[SynthesisTask, ...], *args: Any, **kwargs: Any
    ) -> Iterator[WorkerRecord]:
        submitted.extend(tasks)
        return render_spawned(tasks, *args, **kwargs)

    monkeypatch.setattr("kenkui._execution.coordinator.render_spawned", record)
    with checkpointing(files):
        pipeline.write(tmp_path / "repaired.m4b")
    assert len(submitted) == 2
    assert (tmp_path / "repaired.m4b").read_bytes() == (
        tmp_path / "first.m4b"
    ).read_bytes()
