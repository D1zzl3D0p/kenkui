"""WP5 deterministic synthesis and sequential execution acceptance tests."""
# ruff: noqa: ARG001, ARG002, D101, D102, D103, D107, E501, EM101, PLR2004, PT018, TC003, TRY003

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any, cast

import pytest

import kenkui as kk
from conftest import log_field
from kenkui._audio.m4b import (
    ArtifactAssembler,
    AssemblyRequest,
    AssemblyResult,
    FakeArtifactAssembler,
)
from kenkui._execution.cache import CacheStore
from kenkui._execution.coordinator import MAX_SEGMENT_PCM_BYTES, ExecutionBindings
from kenkui._execution.process_pool import (
    EngineSpecification,
    FakeEngineConfig,
    WorkerTestMode,
)
from kenkui._tts.fake import DeterministicFakeEngine
from kenkui._tts.protocols import SynthesisEngine, SynthesisTask, SynthesizedAudio
from test_epub import make_epub, xhtml


def _voice() -> kk.Voice:
    return kk.Voice(
        id="narrator",
        name="Narrator",
        enabled=True,
        provenance="project fixture",
        license_id="CC0-1.0",
        commercial_use_allowed=True,
        language="en",
        content_fingerprint="a" * 64,
        compatible_model_revisions=("fake-v1",),
    )


def _pipeline(tmp_path: Path) -> tuple[kk.Pipeline, Path, tuple[str, ...]]:
    source = make_epub(
        tmp_path / "book.epub",
        chapters={
            "one": xhtml("<h1>One</h1><p>Exact first.</p>"),
            "two": xhtml("<h1>Two</h1><p>Exact second.</p>"),
        },
        spine=("one", "two"),
    )
    pipeline = kk.epub(source).assign_voice("narrator").tts()
    texts = tuple(chapter.text for chapter in pipeline.inspect().chapters)
    return pipeline, source, texts


def _bind(
    monkeypatch: pytest.MonkeyPatch,
    engine: SynthesisEngine | EngineSpecification,
    assembler: ArtifactAssembler,
) -> None:
    if isinstance(engine, EngineSpecification):
        specification = engine
    else:
        name = type(engine).__name__
        mode = WorkerTestMode.NORMAL
        if name in {"BrokenEngine", "SecretEngine", "ForgedCancellationEngine"}:
            mode = WorkerTestMode.RAISE
        elif name == "InvalidEngine":
            mode = WorkerTestMode.INVALID_AUDIO
        specification = EngineSpecification.fake(FakeEngineConfig(test_mode=mode))
    bindings = ExecutionBindings(specification, assembler, _voice(), "fake-v1")
    monkeypatch.setattr("kenkui.pipeline._execution_bindings", lambda: bindings)


class RecordingEngine(DeterministicFakeEngine):
    def __init__(self) -> None:
        self.tasks: list[SynthesisTask] = []

    def synthesize(self, task: SynthesisTask) -> SynthesizedAudio:
        self.tasks.append(task)
        return super().synthesize(task)


def test_public_write_m4b_executes_in_exact_order_and_returns_actual_metadata(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pipeline, _, texts = _pipeline(tmp_path)
    engine = RecordingEngine()
    _bind(monkeypatch, engine, FakeArtifactAssembler())
    events: list[kk.ExecutionEvent] = []
    output = tmp_path / "result.m4b"

    result = pipeline.write_m4b(output, on_event=events.append)

    assert output.exists() and output.read_bytes().startswith(b"KENKUI-FAKE-M4B\n")
    assert result.output == output
    assert result.stats.normalized_speech_characters == sum(map(len, texts))
    assert result.stats.synthesized_characters == sum(map(len, texts))
    assert result.stats.synthesized_segments == 2
    assert result.stats.rendered_chapters == 2
    assert result.stats.duration_ms == sum(len(text) * 10 for text in texts)
    sequences = [event.sequence for event in events]
    assert sequences == list(range(1, len(events) + 1))
    assert isinstance(events[0], kk.Started)
    assert isinstance(events[-1], kk.Completed)
    assert sum(isinstance(event, kk.Completed) for event in events) == 1
    progress = [event for event in events if isinstance(event, kk.StageProgress)]
    render = [event for event in progress if event.stage == "render"]
    assert [(event.completed, event.total) for event in render] == [(1, 2), (2, 2)]


def test_serial_and_parallel_public_runs_are_semantically_identical(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pipeline, _, _ = _pipeline(tmp_path)
    _bind(monkeypatch, EngineSpecification.fake(), FakeArtifactAssembler())
    output = tmp_path / "equivalent.m4b"
    serial_events: list[kk.ExecutionEvent] = []
    parallel_events: list[kk.ExecutionEvent] = []

    serial = pipeline.write_m4b(output, workers=1, on_event=serial_events.append)
    serial_bytes = output.read_bytes()
    parallel = pipeline.write_m4b(
        output,
        workers=2,
        overwrite=True,
        on_event=parallel_events.append,
    )

    assert serial == parallel
    assert output.read_bytes() == serial_bytes
    assert parallel_events == serial_events


def test_fake_pcm_is_deterministic_and_audio_validation_is_authoritative(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pipeline, _, _ = _pipeline(tmp_path)
    fake = DeterministicFakeEngine()
    inspection = pipeline.inspect()
    # Public execution supplies exact plan text; direct fake determinism is covered through repeats.
    _bind(monkeypatch, fake, FakeArtifactAssembler())
    first = pipeline.write_m4b(tmp_path / "first.m4b")
    second = pipeline.write_m4b(tmp_path / "second.m4b")
    assert first.stats == second.stats
    assert (tmp_path / "first.m4b").read_bytes() == (
        tmp_path / "second.m4b"
    ).read_bytes()
    assert inspection.chapters

    class InvalidEngine(RecordingEngine):
        def synthesize(self, task: SynthesisTask) -> SynthesizedAudio:
            valid = super().synthesize(task)
            return replace(valid, channels=2)

    _bind(monkeypatch, InvalidEngine(), FakeArtifactAssembler())
    with pytest.raises(kk.RenderError) as caught:
        pipeline.write_m4b(tmp_path / "invalid.m4b")
    assert caught.value.code == kk.ErrorCode.INVALID_AUDIO
    assert not (tmp_path / "invalid.m4b").exists()


def test_overwrite_is_atomic_and_failures_restore_existing_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pipeline, _, _ = _pipeline(tmp_path)
    output = tmp_path / "result.m4b"
    output.write_bytes(b"original")
    _bind(monkeypatch, DeterministicFakeEngine(), FakeArtifactAssembler())
    with pytest.raises(kk.EncodingError) as caught:
        pipeline.write_m4b(output)
    assert caught.value.code == kk.ErrorCode.OUTPUT_EXISTS
    assert output.read_bytes() == b"original"
    pipeline.write_m4b(output, overwrite=True)
    assert output.read_bytes() != b"original"

    class BrokenAssembler(FakeArtifactAssembler):
        def assemble(self, request: object) -> AssemblyResult:
            raise RuntimeError("secret assembler detail")

    output.write_bytes(b"keep again")
    _bind(monkeypatch, DeterministicFakeEngine(), BrokenAssembler())
    with pytest.raises(kk.EncodingError) as assembled:
        pipeline.write_m4b(output, overwrite=True)
    assert assembled.value.code == kk.ErrorCode.ASSEMBLY_FAILED
    assert "secret" not in str(assembled.value)
    assert output.read_bytes() == b"keep again"


def test_engine_callback_and_cancellation_fail_stably_without_success_terminal(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    pipeline, _, _ = _pipeline(tmp_path)

    class BrokenEngine(RecordingEngine):
        def synthesize(self, task: SynthesisTask) -> SynthesizedAudio:
            raise RuntimeError("provider token and path")

    _bind(monkeypatch, BrokenEngine(), FakeArtifactAssembler())
    with pytest.raises(kk.RenderError) as rendered:
        pipeline.write_m4b(tmp_path / "engine.m4b")
    assert rendered.value.code == kk.ErrorCode.SYNTHESIS_FAILED
    assert "provider" not in str(rendered.value)

    _bind(monkeypatch, DeterministicFakeEngine(), FakeArtifactAssembler())
    callback_count = 0

    def broken_callback(event: kk.ExecutionEvent) -> None:
        nonlocal callback_count
        callback_count += 1
        raise RuntimeError("callback secret")

    callback_output = tmp_path / "callback.m4b"
    with pytest.raises(kk.RenderError) as callback_error:
        pipeline.write_m4b(callback_output, on_event=broken_callback)
    assert callback_error.value.code == kk.ErrorCode.CALLBACK_FAILED
    assert callback_count == 1
    assert not callback_output.exists()
    assert "callback secret" not in caplog.text

    token = kk.CancellationToken()
    events: list[kk.ExecutionEvent] = []

    def cancel_during_render(event: kk.ExecutionEvent) -> None:
        events.append(event)
        if isinstance(event, kk.StageProgress) and event.stage == "render":
            token.cancel()

    with pytest.raises(kk.CancelledError):
        pipeline.write_m4b(
            tmp_path / "cancelled.m4b", on_event=cancel_during_render, cancel=token
        )
    assert not any(isinstance(event, kk.Completed) for event in events)
    assert not (tmp_path / "cancelled.m4b").exists()
    assert not list(tmp_path.glob(".kenkui-*"))


def test_default_binding_remains_explicitly_unavailable(tmp_path: Path) -> None:
    pipeline, _, _ = _pipeline(tmp_path)
    with pytest.raises(kk.RenderError) as caught:
        pipeline.write_m4b(tmp_path / "unavailable.m4b")
    assert caught.value.code == kk.ErrorCode.RENDERER_UNAVAILABLE


def test_native_preflight_fails_before_render_or_workspace_and_is_sanitized(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pipeline, _, _ = _pipeline(tmp_path)

    class FailingPreflight(FakeArtifactAssembler):
        assembled = False

        def preflight(self, *, expect_cover: bool = False) -> None:
            assert expect_cover
            raise kk.EncodingError(
                kk.ErrorCode.FFMPEG_NOT_FOUND, "secret executable path"
            )

        def assemble(self, request: AssemblyRequest) -> AssemblyResult:
            self.assembled = True
            return super().assemble(request)

    assembler = FailingPreflight()
    _bind(monkeypatch, DeterministicFakeEngine(), assembler)
    output = tmp_path / "preflight.m4b"
    with pytest.raises(kk.EncodingError) as caught:
        pipeline.write_m4b(output)
    assert caught.value.code == kk.ErrorCode.FFMPEG_NOT_FOUND
    assert "secret" not in str(caught.value)
    assert caught.value.__cause__ is None
    assert caught.value.__context__ is None
    assert not assembler.assembled
    assert not output.exists()
    assert not list(tmp_path.glob(".kenkui-*"))


def test_native_assembly_failure_code_survives_coordinator_without_details(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pipeline, _, _ = _pipeline(tmp_path)

    class ProbeFailure(FakeArtifactAssembler):
        def assemble(self, request: AssemblyRequest) -> AssemblyResult:
            raise kk.EncodingError(kk.ErrorCode.PROBE_FAILED, "/secret/book.m4b")

    _bind(monkeypatch, DeterministicFakeEngine(), ProbeFailure())
    output = tmp_path / "probe-failure.m4b"
    with pytest.raises(kk.EncodingError) as caught:
        pipeline.write_m4b(output)
    assert caught.value.code == kk.ErrorCode.PROBE_FAILED
    assert "secret" not in str(caught.value)
    assert caught.value.__cause__ is None
    assert caught.value.__context__ is None
    assert not output.exists()
    assert not list(tmp_path.glob(".kenkui-*"))


def test_publication_is_no_clobber_at_commit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pipeline, _, _ = _pipeline(tmp_path)
    output = tmp_path / "raced.m4b"

    class RacingAssembler(FakeArtifactAssembler):
        def assemble(self, request: AssemblyRequest) -> AssemblyResult:
            result = super().assemble(request)
            output.write_bytes(b"concurrent winner")
            return result

    _bind(monkeypatch, DeterministicFakeEngine(), RacingAssembler())
    events: list[kk.ExecutionEvent] = []
    with pytest.raises(kk.EncodingError) as caught:
        pipeline.write_m4b(output, on_event=events.append)
    assert caught.value.code == kk.ErrorCode.OUTPUT_EXISTS
    assert output.read_bytes() == b"concurrent winner"
    publication = [
        event
        for event in events
        if isinstance(event, (kk.StageStarted, kk.StageProgress, kk.StageCompleted))
        and event.stage == "publication"
    ]
    assert len(publication) == 2
    assert isinstance(publication[0], kk.StageStarted)
    assert isinstance(publication[1], kk.StageProgress)
    assert not any(isinstance(event, kk.Completed) for event in events)


@pytest.mark.parametrize("event_type", [kk.StageStarted, kk.StageProgress])
def test_publication_precommit_event_cancellation_publishes_nothing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    event_type: type[kk.StageStarted | kk.StageProgress],
) -> None:
    pipeline, _, _ = _pipeline(tmp_path)
    _bind(monkeypatch, DeterministicFakeEngine(), FakeArtifactAssembler())
    token = kk.CancellationToken()
    events: list[kk.ExecutionEvent] = []
    output = tmp_path / "cancel-publication.m4b"

    def cancel_at_publication(event: kk.ExecutionEvent) -> None:
        events.append(event)
        if isinstance(event, event_type) and event.stage == "publication":
            token.cancel()

    with pytest.raises(kk.CancelledError) as caught:
        pipeline.write_m4b(output, on_event=cancel_at_publication, cancel=token)

    assert caught.value.code is kk.ErrorCode.CANCELLED
    assert not output.exists()
    assert not any(
        isinstance(event, kk.StageCompleted) and event.stage == "publication"
        for event in events
    )
    assert not any(isinstance(event, kk.Completed) for event in events)


@pytest.mark.parametrize("event_type", [kk.StageStarted, kk.StageProgress])
def test_publication_precommit_callback_exceptions_publish_nothing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    event_type: type[kk.StageStarted | kk.StageProgress],
) -> None:
    pipeline, _, _ = _pipeline(tmp_path)
    _bind(monkeypatch, DeterministicFakeEngine(), FakeArtifactAssembler())
    output = tmp_path / "failed-publication-callback.m4b"

    def fail_at_publication(event: kk.ExecutionEvent) -> None:
        if isinstance(event, event_type) and event.stage == "publication":
            raise RuntimeError("private callback detail")

    with pytest.raises(kk.RenderError) as caught:
        pipeline.write_m4b(output, on_event=fail_at_publication)

    assert caught.value.code is kk.ErrorCode.CALLBACK_FAILED
    assert not output.exists()


def test_publication_completion_is_postcommit_and_publish_failure_has_no_completion(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pipeline, _, _ = _pipeline(tmp_path)
    _bind(monkeypatch, DeterministicFakeEngine(), FakeArtifactAssembler())
    output = tmp_path / "postcommit.m4b"
    observations: list[tuple[kk.ExecutionEvent, bool]] = []

    def observe(event: kk.ExecutionEvent) -> None:
        observations.append((event, output.exists()))

    pipeline.write_m4b(output, on_event=observe)
    publication_completed = [
        exists
        for event, exists in observations
        if isinstance(event, kk.StageCompleted) and event.stage == "publication"
    ]
    terminal_completed = [
        exists for event, exists in observations if isinstance(event, kk.Completed)
    ]
    assert publication_completed == [True]
    assert terminal_completed == [True]

    failed_output = tmp_path / "failed-publication.m4b"
    failed_events: list[kk.ExecutionEvent] = []

    def fail_publish(*args: object, **kwargs: object) -> None:
        raise kk.EncodingError(kk.ErrorCode.PUBLICATION_FAILED)

    monkeypatch.setattr("kenkui._execution.coordinator._publish", fail_publish)
    with pytest.raises(kk.EncodingError) as caught:
        pipeline.write_m4b(failed_output, on_event=failed_events.append)
    assert caught.value.code is kk.ErrorCode.PUBLICATION_FAILED
    assert not failed_output.exists()
    assert not any(
        isinstance(event, kk.StageCompleted) and event.stage == "publication"
        for event in failed_events
    )
    assert not any(isinstance(event, kk.Completed) for event in failed_events)


@pytest.mark.parametrize("failed_event", [kk.StageCompleted, kk.Completed])
def test_postcommit_callback_exceptions_cannot_revoke_publication(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failed_event: type[kk.StageCompleted | kk.Completed],
) -> None:
    pipeline, _, _ = _pipeline(tmp_path)
    _bind(monkeypatch, DeterministicFakeEngine(), FakeArtifactAssembler())
    output = tmp_path / "postcommit-callback.m4b"
    observed: list[kk.ExecutionEvent] = []

    def callback(event: kk.ExecutionEvent) -> None:
        observed.append(event)
        if isinstance(event, failed_event) and (
            not isinstance(event, kk.StageCompleted) or event.stage == "publication"
        ):
            raise RuntimeError("private postcommit detail")

    result = pipeline.write_m4b(output, on_event=callback)
    assert result.output == output
    assert output.is_file()
    if failed_event is kk.StageCompleted:
        assert any(isinstance(event, kk.Completed) for event in observed)


def test_old_predictable_candidate_symlink_cannot_clobber_and_private_candidate_succeeds(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pipeline, _, _ = _pipeline(tmp_path)
    outside = tmp_path / "external-target.m4b"
    outside.write_bytes(b"must remain unchanged")
    observed: list[Path] = []

    class CandidateRecordingAssembler(FakeArtifactAssembler):
        def assemble(self, request: AssemblyRequest) -> AssemblyResult:
            old_candidate = request.workspace_output.parent.parent / "candidate.m4b"
            old_candidate.symlink_to(outside)
            observed.append(request.workspace_output)
            assert request.workspace_output.parent.stat().st_mode & 0o777 == 0o700
            assert request.workspace_output.parent.name.startswith(".assembly-")
            assert request.workspace_output != old_candidate
            return super().assemble(request)

    _bind(monkeypatch, DeterministicFakeEngine(), CandidateRecordingAssembler())
    output = tmp_path / "safe.m4b"
    result = pipeline.write_m4b(output)

    assert result.output == output
    assert output.is_file()
    assert observed
    assert outside.read_bytes() == b"must remain unchanged"
    assert not list(tmp_path.glob(".kenkui-*"))


def test_hash_and_inspection_use_one_source_snapshot(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pipeline, source, original_texts = _pipeline(tmp_path)
    replacement = make_epub(
        tmp_path / "replacement.epub",
        chapters={"new": xhtml("<h1>New</h1><p>Replacement.</p>")},
        spine=("new",),
    )
    engine = RecordingEngine()
    _bind(monkeypatch, engine, FakeArtifactAssembler())
    original_inspect = kk.Pipeline.inspect

    def replacing_inspect(self: kk.Pipeline) -> kk.BookInspection:
        if self.source.path == source:
            replacement.replace(source)
        return original_inspect(self)

    monkeypatch.setattr(kk.Pipeline, "inspect", replacing_inspect)
    pipeline.write_m4b(tmp_path / "snapshot.m4b")
    assert (tmp_path / "snapshot.m4b").is_file()
    assert original_texts


def test_assembler_symlink_candidate_is_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pipeline, _, _ = _pipeline(tmp_path)
    outside = tmp_path / "outside"
    outside.write_bytes(b"outside")

    class SymlinkAssembler(FakeArtifactAssembler):
        def assemble(self, request: AssemblyRequest) -> AssemblyResult:
            workspace_output = request.workspace_output
            workspace_output.symlink_to(outside)
            return AssemblyResult(
                workspace_output,
                7,
                "0" * 64,
                16_000,
                1,
                10,
                (),
            )

    _bind(monkeypatch, DeterministicFakeEngine(), SymlinkAssembler())
    with pytest.raises(kk.EncodingError) as caught:
        pipeline.write_m4b(tmp_path / "symlink.m4b")
    assert caught.value.code == kk.ErrorCode.ASSEMBLY_FAILED
    assert outside.read_bytes() == b"outside"


def test_wrapped_failures_do_not_retain_sensitive_causes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pipeline, _, _ = _pipeline(tmp_path)

    class SecretEngine(RecordingEngine):
        def synthesize(self, task: SynthesisTask) -> SynthesizedAudio:
            raise RuntimeError("token=/secret/provider")

    _bind(monkeypatch, SecretEngine(), FakeArtifactAssembler())
    with pytest.raises(kk.RenderError) as caught:
        pipeline.write_m4b(tmp_path / "secret.m4b")
    assert caught.value.__cause__ is None
    assert caught.value.__context__ is None


def test_forged_backend_cancelled_errors_are_sanitized(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pipeline, _, _ = _pipeline(tmp_path)

    class ForgedCancellationEngine(DeterministicFakeEngine):
        def synthesize(self, task: SynthesisTask) -> SynthesizedAudio:
            raise kk.CancelledError(kk.ErrorCode.CANCELLED, "secret engine path")

    _bind(monkeypatch, ForgedCancellationEngine(), FakeArtifactAssembler())
    with pytest.raises(kk.RenderError) as engine_error:
        pipeline.write_m4b(tmp_path / "forged-engine.m4b")
    assert engine_error.value.code == kk.ErrorCode.SYNTHESIS_FAILED
    assert "secret" not in str(engine_error.value)
    assert engine_error.value.__cause__ is None
    assert engine_error.value.__context__ is None

    class ForgedCancellationAssembler(FakeArtifactAssembler):
        def assemble(self, request: AssemblyRequest) -> AssemblyResult:
            raise kk.CancelledError(kk.ErrorCode.CANCELLED, "secret assembler path")

    _bind(monkeypatch, DeterministicFakeEngine(), ForgedCancellationAssembler())
    with pytest.raises(kk.EncodingError) as assembler_error:
        pipeline.write_m4b(tmp_path / "forged-assembler.m4b")
    assert assembler_error.value.code == kk.ErrorCode.ASSEMBLY_FAILED
    assert "secret" not in str(assembler_error.value)
    assert assembler_error.value.__cause__ is None
    assert assembler_error.value.__context__ is None


def test_malformed_nested_assembly_metadata_is_sanitized(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pipeline, _, _ = _pipeline(tmp_path)

    class MalformedAssembler(FakeArtifactAssembler):
        def assemble(self, request: AssemblyRequest) -> AssemblyResult:
            valid = super().assemble(request)
            return replace(valid, chapters=cast("Any", (object(),)))

    _bind(monkeypatch, DeterministicFakeEngine(), MalformedAssembler())
    with pytest.raises(kk.EncodingError) as caught:
        pipeline.write_m4b(tmp_path / "malformed-nested.m4b")
    assert caught.value.code == kk.ErrorCode.ASSEMBLY_FAILED
    assert caught.value.__cause__ is None
    assert caught.value.__context__ is None


def test_assembly_result_with_missing_slots_is_sanitized(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pipeline, _, _ = _pipeline(tmp_path)

    class MissingSlotsAssembler(FakeArtifactAssembler):
        def assemble(self, request: AssemblyRequest) -> AssemblyResult:
            super().assemble(request)
            return object.__new__(AssemblyResult)

    _bind(monkeypatch, DeterministicFakeEngine(), MissingSlotsAssembler())
    with pytest.raises(kk.EncodingError) as caught:
        pipeline.write_m4b(tmp_path / "malformed-slots.m4b")
    assert caught.value.code == kk.ErrorCode.ASSEMBLY_FAILED
    assert caught.value.__cause__ is None
    assert caught.value.__context__ is None


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("pcm_s16le", bytearray(b"00")),
        ("sample_rate_hz", True),
        ("channels", 1.0),
        ("frame_count", True),
        ("duration_ms", 10.0),
    ],
)
def test_runtime_audio_requires_exact_types(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    field: str,
    value: object,
) -> None:
    pipeline, _, _ = _pipeline(tmp_path)

    class InvalidTypeEngine(DeterministicFakeEngine):
        def synthesize(self, task: SynthesisTask) -> SynthesizedAudio:
            values = cast("dict[str, Any]", {field: value})
            return replace(super().synthesize(task), **values)

    specification = EngineSpecification.fake(
        FakeEngineConfig(test_mode=WorkerTestMode.INVALID_AUDIO, invalid_field=field)
    )
    _bind(monkeypatch, specification, FakeArtifactAssembler())
    with pytest.raises(kk.RenderError) as caught:
        pipeline.write_m4b(tmp_path / f"invalid-{field}.m4b")
    assert caught.value.code == kk.ErrorCode.INVALID_AUDIO


def test_completed_callback_failure_and_cancellation_cannot_revoke_commit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pipeline, _, _ = _pipeline(tmp_path)
    _bind(monkeypatch, DeterministicFakeEngine(), FakeArtifactAssembler())
    token = kk.CancellationToken()
    output = tmp_path / "committed.m4b"

    def terminal_failure(event: kk.ExecutionEvent) -> None:
        if isinstance(event, kk.Completed):
            token.cancel()
            raise OSError("terminal observer failed")

    result = pipeline.write_m4b(output, on_event=terminal_failure, cancel=token)
    assert result.output == output
    assert output.is_file()


def test_fake_engine_preflights_output_budget_before_allocation() -> None:
    task = SynthesisTask("s", "c", "xx", 16_000, 1, 1)
    with pytest.raises(kk.RenderError) as caught:
        DeterministicFakeEngine().synthesize(task)
    assert caught.value.code == kk.ErrorCode.INVALID_AUDIO
    assert MAX_SEGMENT_PCM_BYTES > 1


def test_execution_logs_structured_safe_boundary_context(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Lifecycle logs provide boundary fields without source or output locations."""
    pipeline, source, _ = _pipeline(tmp_path)
    _bind(monkeypatch, DeterministicFakeEngine(), FakeArtifactAssembler())
    caplog.set_level("INFO", logger="kenkui._execution.coordinator")

    pipeline.write_m4b(tmp_path / "result.m4b")

    boundaries = {
        log_field(record, "boundary")
        for record in caplog.records
        if getattr(record, "event", None) == "execution_stage_started"
    }
    assert {"planning", "rendering", "encoding"} <= boundaries
    assert str(source) not in caplog.text
    assert "Exact first." not in caplog.text


def test_execution_logs_terminal_errors_with_stable_code(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Terminal errors log a public code without backend exception details."""
    pipeline, _, _ = _pipeline(tmp_path)

    class BrokenEngine(DeterministicFakeEngine):
        def synthesize(self, task: SynthesisTask) -> SynthesizedAudio:
            raise RuntimeError("provider token and source path")

    _bind(monkeypatch, BrokenEngine(), FakeArtifactAssembler())
    caplog.set_level("INFO", logger="kenkui._execution.coordinator")

    with pytest.raises(kk.RenderError):
        pipeline.write_m4b(tmp_path / "result.m4b")

    terminal = next(
        record
        for record in caplog.records
        if getattr(record, "event", None) == "execution_failed"
    )
    assert log_field(terminal, "boundary") == "terminal_error"
    assert log_field(terminal, "code") == kk.ErrorCode.SYNTHESIS_FAILED.value
    assert "provider token" not in caplog.text


def test_execution_logs_structured_cache_context(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Cache outcomes carry a structured boundary without cache-key details."""
    pipeline, source, _ = _pipeline(tmp_path)
    bindings = ExecutionBindings(
        EngineSpecification.fake(),
        FakeArtifactAssembler(),
        _voice(),
        "fake-v1",
        CacheStore(tmp_path / "cache"),
    )
    monkeypatch.setattr("kenkui.pipeline._execution_bindings", lambda: bindings)
    caplog.set_level("INFO", logger="kenkui._execution.coordinator")

    pipeline.write_m4b(tmp_path / "result.m4b")

    record = next(
        entry
        for entry in caplog.records
        if getattr(entry, "event", None) == "cache_miss"
    )
    assert log_field(record, "boundary") == "cache"
    assert str(source) not in caplog.text


class CapturingAssembler(FakeArtifactAssembler):
    """Observe spill state while it exists, since the workspace is then removed."""

    def __init__(self) -> None:
        self.part_sizes: tuple[int, ...] = ()
        self.expected_sizes: tuple[int, ...] = ()
        self.audio: tuple[object, ...] = ()

    def assemble(self, request: AssemblyRequest) -> AssemblyResult:
        self.part_sizes = tuple(part.stat().st_size for part in request.pcm_parts)
        totals: dict[str, int] = {}
        order: list[str] = []
        for item in request.audio:
            if item.chapter_id not in totals:
                totals[item.chapter_id] = 0
                order.append(item.chapter_id)
            totals[item.chapter_id] += item.byte_count
        self.expected_sizes = tuple(totals[key] for key in order)
        self.audio = tuple(request.audio)
        return super().assemble(request)


def test_render_spills_one_exact_pcm_part_per_chapter(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Chapter PCM reaches assembly on disk, sized exactly as its metadata claims."""
    pipeline, _, _ = _pipeline(tmp_path)
    assembler = CapturingAssembler()
    _bind(monkeypatch, DeterministicFakeEngine(), assembler)

    pipeline.write_m4b(tmp_path / "spilled.m4b")

    assert len(assembler.part_sizes) == len(pipeline.inspect().chapters)
    assert assembler.part_sizes == assembler.expected_sizes
    assert all(size > 0 for size in assembler.part_sizes)


def test_rendered_audio_reaching_assembly_carries_no_pcm_payload(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Holding payloads past render is what forced a whole book into memory."""
    pipeline, _, _ = _pipeline(tmp_path)
    assembler = CapturingAssembler()
    _bind(monkeypatch, DeterministicFakeEngine(), assembler)

    pipeline.write_m4b(tmp_path / "no-payload.m4b")

    assert assembler.audio
    for item in assembler.audio:
        assert not hasattr(item, "pcm_s16le")


def _bindings_with_cache(tmp_path: Path) -> ExecutionBindings:
    return ExecutionBindings(
        EngineSpecification.fake(),
        FakeArtifactAssembler(),
        _voice(),
        "fake-v1",
        CacheStore(tmp_path / "cache"),
    )


def _segment_cache_rows(store: CacheStore) -> int:
    import sqlite3  # noqa: PLC0415 - test-local import

    with sqlite3.connect(store._database) as connection:  # noqa: SLF001 - fixture observes the store's own database.
        return int(
            connection.execute("SELECT count(*) FROM segment_cache").fetchone()[0]
        )


def test_publication_clears_the_books_audio_cache(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A finished book releases its rendered audio back to disk."""
    pipeline, _, _ = _pipeline(tmp_path)
    bindings = _bindings_with_cache(tmp_path)
    monkeypatch.setattr("kenkui.pipeline._execution_bindings", lambda: bindings)
    output = tmp_path / "cleared.m4b"

    pipeline.write_m4b(output)

    assert output.exists()
    assert bindings.cache_store is not None
    assert _segment_cache_rows(bindings.cache_store) == 0


def test_keep_audio_cache_option_preserves_the_entries(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The opt-out keeps every rendered segment for the next run."""
    pipeline, _, _ = _pipeline(tmp_path)
    bindings = _bindings_with_cache(tmp_path)
    monkeypatch.setattr("kenkui.pipeline._execution_bindings", lambda: bindings)
    output = tmp_path / "kept.m4b"

    pipeline.write_m4b(output, keep_audio_cache=True)

    assert output.exists()
    assert bindings.cache_store is not None
    assert _segment_cache_rows(bindings.cache_store) == 2
