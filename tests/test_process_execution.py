"""WP6 spawned, bounded, deterministic rendering tests."""
# ruff: noqa: D103, EM101, PLR2004, S301, TRY003

from __future__ import annotations

import inspect
import multiprocessing
import pickle
from typing import TYPE_CHECKING

import pytest

from kenkui._execution import process_pool
from kenkui.cancellation import CancellationToken

if TYPE_CHECKING:
    from pathlib import Path

from kenkui._execution.process_pool import (
    MAX_RENDER_WORKERS,
    EngineSpecification,
    FakeEngineConfig,
    WorkerTestMode,
    render_spawned,
    resolve_workers,
)
from kenkui._tts.protocols import SynthesisTask
from kenkui.errors import CancelledError, ErrorCode, RenderError


def _tasks(count: int = 4) -> tuple[SynthesisTask, ...]:
    return tuple(
        SynthesisTask(f"s-{index}", f"c-{index}", f"text {index}", 16_000, 1, 1_000_000)
        for index in range(count)
    )


def test_specification_is_frozen_pickle_safe_and_workers_are_bounded(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spec = EngineSpecification.fake()
    assert pickle.loads(pickle.dumps(spec)) == spec
    monkeypatch.setattr(
        "kenkui._execution.process_pool._available_cpu_count", lambda: 99
    )
    assert resolve_workers("auto", 20) == MAX_RENDER_WORKERS
    assert resolve_workers(99, 20) == MAX_RENDER_WORKERS
    assert resolve_workers(2, 1) == 1


def test_spawned_serial_parallel_are_byte_identical_and_plan_ordered() -> None:
    tasks = _tasks()
    serial_records = list(render_spawned(tasks, EngineSpecification.fake(), 1, None))
    parallel_records = list(render_spawned(tasks, EngineSpecification.fake(), 2, None))
    assert [record.audio for record in serial_records] == [
        record.audio for record in parallel_records
    ]
    assert [record.audio.segment_id for record in parallel_records] == [
        task.segment_id for task in tasks
    ]
    assert {record.start_method for record in parallel_records} == {"spawn"}
    assert all(
        record.worker_pid != multiprocessing.current_process().pid
        for record in parallel_records
    )
    serial_pids = {record.worker_pid for record in serial_records}
    parallel_pids = {record.worker_pid for record in parallel_records}
    assert len(serial_pids) == 1 < len(tasks)
    assert len(parallel_pids) == 2 < len(tasks)
    assert {record.engine_initializations for record in parallel_records} == {1}


def test_many_segments_reuse_only_the_resolved_bounded_workers() -> None:
    tasks = _tasks(40)
    records = list(render_spawned(tasks, EngineSpecification.fake(), 2, None))
    assert len(records) == len(tasks)
    assert len({record.worker_pid for record in records}) == 2
    assert [record.audio.segment_id for record in records] == [
        task.segment_id for task in tasks
    ]


@pytest.mark.parametrize("mode", [WorkerTestMode.CRASH, WorkerTestMode.MALFORMED])
def test_child_failure_is_sanitized_and_reaped(mode: WorkerTestMode) -> None:
    before = {child.pid for child in multiprocessing.active_children()}
    spec = EngineSpecification("fake", FakeEngineConfig(test_mode=mode))
    with pytest.raises(RenderError) as caught:
        list(render_spawned(_tasks(), spec, 2, None))
    assert caught.value.code == ErrorCode.SYNTHESIS_FAILED
    assert caught.value.__cause__ is None
    assert {child.pid for child in multiprocessing.active_children()} == before


def test_child_timeout_is_sanitized_and_reaped() -> None:
    before = {child.pid for child in multiprocessing.active_children()}
    spec = EngineSpecification(
        "fake", FakeEngineConfig(test_mode=WorkerTestMode.HANG, timeout_seconds=0.05)
    )
    with pytest.raises(RenderError) as caught:
        list(render_spawned(_tasks(2), spec, 2, None))
    assert caught.value.code == ErrorCode.SYNTHESIS_FAILED
    assert {child.pid for child in multiprocessing.active_children()} == before


def test_start_failure_is_sanitized(monkeypatch: pytest.MonkeyPatch) -> None:
    class FailedProcess:
        def start(self) -> None:
            raise OSError

        def close(self) -> None:
            return

    class StartFailureContext:
        @staticmethod
        def Process(**_kwargs: object) -> FailedProcess:  # noqa: N802
            return FailedProcess()

    def failed_context(method: str) -> StartFailureContext:
        assert method == "spawn"
        return StartFailureContext()

    monkeypatch.setattr(
        "kenkui._execution.process_pool.multiprocessing.get_context", failed_context
    )
    with pytest.raises(RenderError) as caught:
        list(render_spawned(_tasks(1), EngineSpecification.fake(), 1, None))
    assert caught.value.code == ErrorCode.SYNTHESIS_FAILED


def test_parent_never_constructs_or_calls_engine(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def forbidden_parent_constructor() -> object:
        raise AssertionError("parent initialized synthesis engine")

    monkeypatch.setattr(
        "kenkui._execution.process_pool.DeterministicFakeEngine",
        forbidden_parent_constructor,
    )
    # Spawn imports the module afresh in the child; a fork would inherit this guard.
    records = list(render_spawned(_tasks(1), EngineSpecification.fake(), 1, None))
    assert len(records) == 1


def test_fake_config_does_not_accept_parent_resources(tmp_path: Path) -> None:
    # The frozen DTO contains values only, never a parent engine or process object.
    spec = EngineSpecification.fake()
    assert not hasattr(spec, "engine")
    assert tmp_path.exists()


def test_worker_results_never_use_connection_receive_or_pickle() -> None:
    source = inspect.getsource(process_pool)
    assert ".Pipe(" not in source
    assert ".recv(" not in source
    assert "Connection" not in source
    assert "pickle" not in source


@pytest.mark.parametrize(
    "mode",
    [
        WorkerTestMode.MALFORMED,
        WorkerTestMode.TRUNCATED_RESULT,
        WorkerTestMode.OVERSIZED_RESULT,
    ],
)
def test_unsafe_result_files_are_bounded_and_sanitized(mode: WorkerTestMode) -> None:
    spec = EngineSpecification.fake(FakeEngineConfig(test_mode=mode))
    with pytest.raises(RenderError) as caught:
        list(render_spawned(_tasks(1), spec, 1, None))
    assert caught.value.code == ErrorCode.SYNTHESIS_FAILED
    assert caught.value.__cause__ is None


def test_cancellation_is_polled_while_child_stalls(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class CountingCancellation(CancellationToken):
        def __init__(self) -> None:
            super().__init__()
            self.checks = 0

        def raise_if_cancelled(self) -> None:
            self.checks += 1
            if self.checks == 3:
                self.cancel()
            super().raise_if_cancelled()

    token = CountingCancellation()
    monkeypatch.setattr(process_pool, "_poll_wait", lambda _seconds: None)
    spec = EngineSpecification.fake(
        FakeEngineConfig(test_mode=WorkerTestMode.HANG, timeout_seconds=30)
    )
    with pytest.raises(CancelledError):
        list(render_spawned(_tasks(1), spec, 1, token))
    assert token.checks >= 3


def test_terminate_resistant_process_is_killed_and_all_joins_are_bounded() -> None:
    class ResistantProcess:
        def __init__(self) -> None:
            self.alive = True
            self.killed = False
            self.closed = False
            self.joins: list[float | None] = []

        def is_alive(self) -> bool:
            return self.alive

        def terminate(self) -> None:
            return

        def kill(self) -> None:
            self.killed = True
            self.alive = False

        def join(self, timeout: float | None = None) -> None:
            self.joins.append(timeout)

        def close(self) -> None:
            assert not self.alive
            self.closed = True

    process = ResistantProcess()
    process_pool._terminate_and_reap_process(process)  # type: ignore[arg-type]  # noqa: SLF001
    assert process.killed
    assert process.closed
    assert process.joins
    assert all(timeout is not None and timeout > 0 for timeout in process.joins)


def test_child_error_code_survives_the_process_boundary() -> None:
    """A coded child failure reaches the caller as its own code, not a generic one."""
    before = {child.pid for child in multiprocessing.active_children()}
    spec = EngineSpecification(
        "fake", FakeEngineConfig(test_mode=WorkerTestMode.RAISE_CODED)
    )
    with pytest.raises(RenderError) as caught:
        list(render_spawned(_tasks(), spec, 2, None))
    assert caught.value.code == ErrorCode.POCKET_INFERENCE_FAILED
    assert caught.value.__cause__ is None
    assert {child.pid for child in multiprocessing.active_children()} == before


def test_uncoded_child_failure_still_reports_the_generic_code() -> None:
    """A child failure carrying no stable code stays generic rather than guessing."""
    spec = EngineSpecification("fake", FakeEngineConfig(test_mode=WorkerTestMode.RAISE))
    with pytest.raises(RenderError) as caught:
        list(render_spawned(_tasks(), spec, 2, None))
    assert caught.value.code == ErrorCode.SYNTHESIS_FAILED


def test_worker_failure_is_logged_with_context(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """The stable code is observable in logs even though the CLI stays generic."""
    spec = EngineSpecification(
        "fake", FakeEngineConfig(test_mode=WorkerTestMode.RAISE_CODED)
    )
    with caplog.at_level("DEBUG", logger="kenkui"), pytest.raises(RenderError):
        list(render_spawned(_tasks(), spec, 2, None))
    records = [r for r in caplog.records if getattr(r, "event", "") == "worker_failed"]
    assert records
    assert records[0].error_code == ErrorCode.POCKET_INFERENCE_FAILED.value
    text = caplog.text
    assert "exact task text" not in text
    assert "Traceback" not in text


@pytest.mark.parametrize(
    ("cpus", "expected"),
    [(1, 1), (2, 1), (3, 1), (4, 2), (12, 10), (99, MAX_RENDER_WORKERS)],
)
def test_auto_workers_reserve_two_cpus_and_never_fall_below_one(
    monkeypatch: pytest.MonkeyPatch, cpus: int, expected: int
) -> None:
    """Auto leaves two CPUs for everything else without ever resolving to zero."""
    monkeypatch.setattr(
        "kenkui._execution.process_pool._available_cpu_count", lambda: cpus
    )
    assert resolve_workers("auto", 1000) == expected


def test_auto_workers_still_yield_to_chapter_count(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A short book never starts more workers than it has segments to render."""
    monkeypatch.setattr(
        "kenkui._execution.process_pool._available_cpu_count", lambda: 12
    )
    assert resolve_workers("auto", 3) == 3
    assert resolve_workers("auto", 0) == 0


@pytest.mark.parametrize(
    ("cpus", "workers", "expected"),
    [(12, 10, 1), (12, 2, 6), (12, 1, 12), (8, 3, 2), (2, 10, 1)],
)
def test_thread_limit_divides_cpus_across_workers(
    monkeypatch: pytest.MonkeyPatch, cpus: int, workers: int, expected: int
) -> None:
    """Each worker gets a share of the CPUs so the pool cannot oversubscribe."""
    monkeypatch.setattr(
        "kenkui._execution.process_pool._available_cpu_count", lambda: cpus
    )
    assert process_pool.resolve_thread_limit(workers) == expected


def test_thread_limit_never_drops_below_one(monkeypatch: pytest.MonkeyPatch) -> None:
    """A worker always gets at least one thread, whatever the arithmetic says."""
    monkeypatch.setattr(
        "kenkui._execution.process_pool._available_cpu_count", lambda: 1
    )
    assert process_pool.resolve_thread_limit(64) == 1
    assert process_pool.resolve_thread_limit(0) == 1
