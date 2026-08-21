"""Bounded reusable spawned-process synthesis scheduler."""

from __future__ import annotations

import json
import logging
import math
import multiprocessing
import os
import stat
import struct
import tempfile
import time
from contextlib import suppress
from dataclasses import dataclass, replace
from enum import StrEnum
from pathlib import Path
from typing import TYPE_CHECKING, Literal, NoReturn

from kenkui._tts.fake import DeterministicFakeEngine
from kenkui._tts.protocols import SynthesisTask, SynthesizedAudio
from kenkui.errors import ErrorCode, RenderError
from kenkui.observability import get_logger, log_event

if TYPE_CHECKING:
    from collections.abc import Iterator
    from multiprocessing.process import BaseProcess

    from kenkui._tts.pocket import PocketEngineConfig
    from kenkui.cancellation import CancellationToken

# Each worker copies a private model snapshot and holds its own model instance
# (~209 MiB for the english engine), so the ceiling is memory, not CPU.
MAX_RENDER_WORKERS = 16
# CPUs deliberately left to the rest of the system when resolving "auto".
RESERVED_CPU_COUNT = 2
_DEFAULT_WORKER_TIMEOUT_SECONDS = 30.0
_WAIT_SLICE_SECONDS = 0.05
_TERMINATE_GRACE_SECONDS = 0.1
_KILL_GRACE_SECONDS = 0.1
_DEFAULT_TOTAL_PCM_BYTES = 512 * 1024 * 1024
_WIRE_MAGIC = b"KENKUIR\0"
_WIRE_VERSION = 1
_WIRE_PREFIX = struct.Struct(">8sBI")
_MAX_HEADER_BYTES = 4096
_READ_CHUNK_BYTES = 64 * 1024
_MAX_WORKER_TIMEOUT_SECONDS = 3600.0
_MAX_TASK_COUNT = 100_000
_MAX_TASK_INPUT_BYTES = 256 * 1024 * 1024
_FAILURE_PREFIX = "failure-"
_MAX_FAILURE_BYTES = 128

_LOGGER = get_logger(__name__)


class WorkerTestMode(StrEnum):
    """Private deterministic failure controls for process-contract tests."""

    NORMAL = "normal"
    CRASH = "crash"
    MALFORMED = "malformed"
    TRUNCATED_RESULT = "truncated_result"
    OVERSIZED_RESULT = "oversized_result"
    HANG = "hang"
    RAISE = "raise"
    RAISE_CODED = "raise_coded"
    INVALID_AUDIO = "invalid_audio"


@dataclass(frozen=True, slots=True)
class FakeEngineConfig:
    """Pickle-safe values needed to construct the supported fake in a child."""

    test_mode: WorkerTestMode = WorkerTestMode.NORMAL
    timeout_seconds: float = _DEFAULT_WORKER_TIMEOUT_SECONDS
    invalid_field: str | None = None


@dataclass(frozen=True, slots=True)
class EngineSpecification:
    """Immutable engine construction specification; never an engine instance."""

    kind: Literal["fake", "pocket"]
    fake_config: FakeEngineConfig | None = None
    pocket_config: PocketEngineConfig | None = None

    @classmethod
    def fake(cls, config: FakeEngineConfig | None = None) -> EngineSpecification:
        """Create the private supported deterministic fake specification."""
        return cls("fake", config or FakeEngineConfig())

    @classmethod
    def pocket(cls, config: PocketEngineConfig) -> EngineSpecification:
        """Create an inert Pocket specification; construction occurs in children."""
        return cls("pocket", pocket_config=config)


@dataclass(frozen=True, slots=True)
class WorkerRecord:
    """Validated child response plus process-contract evidence."""

    audio: SynthesizedAudio
    worker_pid: int
    start_method: str
    engine_initializations: int


@dataclass(slots=True)
class _Active:
    process: BaseProcess
    assignments: tuple[tuple[int, SynthesisTask], ...]
    pending: set[int]
    deadline: float


@dataclass(frozen=True, slots=True)
class _ResultItem:
    process: BaseProcess
    task: SynthesisTask
    index: int
    result_path: Path


class _InvalidResultError(Exception):
    def __init__(self, code: ErrorCode = ErrorCode.SYNTHESIS_FAILED) -> None:
        self.code = code


def _available_cpu_count() -> int:
    """Return CPUs available to this process, conservatively falling back to one."""
    affinity = getattr(os, "sched_getaffinity", None)
    if affinity is not None:
        with suppress(OSError):
            return max(1, len(affinity(0)))
    return max(1, os.cpu_count() or 1)


# Inference libraries each start their own pool sized to the whole machine, so
# N worker processes otherwise contend for N times the available cores. Dividing
# the CPUs across workers keeps total threads near the core count.
_THREAD_LIMIT_VARIABLES = (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)


def resolve_thread_limit(worker_count: int) -> int:
    """Return the per-worker thread budget that keeps the pool from thrashing."""
    if worker_count < 1:
        return 1
    return max(1, _available_cpu_count() // worker_count)


def _apply_thread_limit(  # pragma: no cover - set inside spawned children.
    thread_limit: int,
) -> None:
    """Bound this child's inference threads before any such library is imported."""
    value = str(max(1, thread_limit))
    for name in _THREAD_LIMIT_VARIABLES:
        os.environ[name] = value


def resolve_workers(workers: int | Literal["auto"], chapter_count: int) -> int:
    """Resolve public policy against chapters, available CPUs, and the hard cap."""
    if chapter_count < 1:
        return 0
    if workers == "auto":
        requested = max(1, _available_cpu_count() - RESERVED_CPU_COUNT)
    else:
        requested = workers
    return min(requested, chapter_count, MAX_RENDER_WORKERS)


def _metadata(  # pragma: no cover - exercised only in independently spawned children.
    audio: SynthesizedAudio, index: int
) -> dict[str, object]:
    return {
        "channels": audio.channels,
        "chapter_id": audio.chapter_id,
        "duration_ms": audio.duration_ms,
        "engine_initializations": 1,
        "frame_count": audio.frame_count,
        "index": index,
        "pcm_bytes": len(audio.pcm_s16le),
        "pcm_type": type(audio.pcm_s16le).__name__,
        "sample_rate_hz": audio.sample_rate_hz,
        "segment_id": audio.segment_id,
        "start_method": multiprocessing.get_start_method(),
        "worker_pid": os.getpid(),
    }


def _write_wire_result(  # pragma: no cover - exercised by spawned integration tests.
    result_path: str,
    audio: SynthesizedAudio,
    index: int,
    mode: WorkerTestMode,
    max_output_bytes: int,
) -> None:
    path = Path(result_path)
    temporary = path.with_suffix(".tmp")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0)
    descriptor = os.open(temporary, flags, 0o600)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            if mode is WorkerTestMode.MALFORMED:
                stream.write(b"malformed")
            elif mode is WorkerTestMode.TRUNCATED_RESULT:
                stream.write(_WIRE_PREFIX.pack(_WIRE_MAGIC, _WIRE_VERSION, 100))
            elif mode is WorkerTestMode.OVERSIZED_RESULT:
                stream.truncate(
                    _WIRE_PREFIX.size + _MAX_HEADER_BYTES + max_output_bytes + 1
                )
            else:
                header = json.dumps(
                    _metadata(audio, index),
                    ensure_ascii=False,
                    separators=(",", ":"),
                    sort_keys=True,
                ).encode("utf-8")
                if len(header) > _MAX_HEADER_BYTES:
                    raise ValueError  # noqa: TRY301
                stream.write(_WIRE_PREFIX.pack(_WIRE_MAGIC, _WIRE_VERSION, len(header)))
                stream.write(header)
                stream.write(audio.pcm_s16le)
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(path)
    except BaseException:
        with suppress(OSError):
            temporary.unlink()
        raise


def _write_wire_failure(  # pragma: no cover - exercised by spawned tests.
    failure_path: str, error: BaseException
) -> None:
    """Publish only a stable code: never provider text, a path, or a traceback."""
    code = getattr(error, "code", None)
    value = code.value if type(code) is ErrorCode else ErrorCode.SYNTHESIS_FAILED.value
    with suppress(BaseException):
        path = Path(failure_path)
        temporary = path.with_suffix(".tmp")
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0)
        descriptor = os.open(temporary, flags, 0o600)
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(value.encode("ascii"))
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(path)


def _failure_code(path: Path, workspace: Path) -> ErrorCode:
    """Read a bounded child failure code, falling back to the generic failure."""
    try:
        if path.parent != workspace or not path.is_relative_to(workspace):
            raise OSError  # noqa: TRY301
        named = path.lstat()
        descriptor = os.open(
            path,
            os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0),
        )
        try:
            opened = os.fstat(descriptor)
            if (
                not stat.S_ISREG(opened.st_mode)
                or stat.S_ISLNK(named.st_mode)
                or opened.st_nlink != 1
                or opened.st_size > _MAX_FAILURE_BYTES
            ):
                raise OSError  # noqa: TRY301
            raw = os.read(descriptor, _MAX_FAILURE_BYTES)
        finally:
            os.close(descriptor)
        return ErrorCode(raw.decode("ascii").strip())
    except (OSError, UnicodeDecodeError, ValueError):
        return ErrorCode.SYNTHESIS_FAILED


def synthesis_worker_entry(  # noqa: C901, PLC0415, PLR0912, TRY301  # pragma: no cover
    workspace: str,
    specification: EngineSpecification,
    assignments: tuple[tuple[int, SynthesisTask], ...],
    failure_path: str = "",
    thread_limit: int = 1,
) -> None:
    """Construct one engine and publish acknowledged results for its static batch."""
    engine: object | None = None
    _apply_thread_limit(thread_limit)
    try:
        if specification.kind == "pocket":
            if specification.pocket_config is None:
                raise ValueError
            from kenkui._tts import pocket

            # Install the child capability and irreversible network audit hook
            # before importing Pocket or any of its inference dependencies.
            pocket._enter_spawned_worker(pocket._WORKER_TOKEN)  # noqa: SLF001
            engine = pocket.PocketTTSEngine(specification.pocket_config, reusable=True)
            mode = WorkerTestMode.NORMAL
        else:
            if specification.fake_config is None:
                raise ValueError
            mode = specification.fake_config.test_mode
            if mode is WorkerTestMode.CRASH:
                os._exit(71)
            if mode is WorkerTestMode.HANG:
                multiprocessing.Event().wait()
                return
            if mode is WorkerTestMode.RAISE:
                raise RuntimeError  # noqa: TRY301 - deterministic child failure mode.
            if mode is WorkerTestMode.RAISE_CODED:
                raise RenderError(  # noqa: TRY301 - deterministic coded child failure.
                    ErrorCode.POCKET_INFERENCE_FAILED
                )
            engine = DeterministicFakeEngine()

        for index, task in assignments:
            audio = engine.synthesize(task)
            if mode is WorkerTestMode.INVALID_AUDIO:
                values: dict[str, object] = {
                    "channels": 2,
                    "sample_rate_hz": True,
                    "frame_count": True,
                    "duration_ms": 10.0,
                    "pcm_s16le": bytearray(b"00"),
                }
                fake_config = specification.fake_config
                if fake_config is None:
                    raise ValueError
                field = fake_config.invalid_field or "channels"
                audio = replace(audio, **{field: values[field]})  # type: ignore[arg-type]
            pcm = audio.pcm_s16le
            if (
                not isinstance(pcm, (bytes, bytearray))
                or len(pcm) > task.max_output_bytes
            ):
                raise ValueError
            result_path = Path(workspace) / f"result-{index}.wire"
            _write_wire_result(
                str(result_path), audio, index, mode, task.max_output_bytes
            )
            # Deletion is the parent's acknowledgement: at most one result file
            # can exist per worker even when a static batch contains many tasks.
            while result_path.exists():
                time.sleep(_WAIT_SLICE_SECONDS)
    except BaseException as error:  # noqa: BLE001 - details never cross the boundary.
        _write_wire_failure(failure_path, error)
        raise SystemExit(72) from None
    finally:
        close = getattr(engine, "close", None)
        if close is not None:
            with suppress(BaseException):
                close()


def _poll_wait(seconds: float) -> None:
    time.sleep(seconds)


def _read_exact(descriptor: int, size: int) -> bytes:
    chunks: list[bytes] = []
    remaining = size
    while remaining:
        chunk = os.read(descriptor, min(remaining, _READ_CHUNK_BYTES))
        if not chunk:
            raise _InvalidResultError
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def _open_result(
    path: Path, workspace: Path, max_output_bytes: int
) -> tuple[int, os.stat_result]:
    try:
        if path.parent != workspace or not path.is_relative_to(workspace):
            raise OSError  # noqa: TRY301
        named = path.lstat()
        descriptor = os.open(
            path,
            os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0),
        )
        try:
            opened = os.fstat(descriptor)
            maximum = _WIRE_PREFIX.size + _MAX_HEADER_BYTES + max_output_bytes
            valid = (
                stat.S_ISREG(opened.st_mode)
                and not stat.S_ISLNK(named.st_mode)
                and opened.st_nlink == 1
                and (opened.st_dev, opened.st_ino) == (named.st_dev, named.st_ino)
                and _WIRE_PREFIX.size <= opened.st_size <= maximum
                and path.resolve(strict=True).is_relative_to(
                    workspace.resolve(strict=True)
                )
            )
            if not valid:
                raise OSError  # noqa: TRY301
        except BaseException:
            os.close(descriptor)
            raise
    except (OSError, ValueError):
        raise _InvalidResultError from None
    return descriptor, opened


def _exact_int(value: object) -> int:
    if type(value) is not int:
        raise _InvalidResultError(ErrorCode.INVALID_AUDIO)
    return value


def _exact_str(value: object) -> str:
    if type(value) is not str:
        raise _InvalidResultError(ErrorCode.INVALID_AUDIO)
    return value


def _parse_result(item: _ResultItem, workspace: Path) -> WorkerRecord:
    descriptor, before = _open_result(
        item.result_path, workspace, item.task.max_output_bytes
    )
    try:
        prefix = _read_exact(descriptor, _WIRE_PREFIX.size)
        magic, version, header_size = _WIRE_PREFIX.unpack(prefix)
        if (
            magic != _WIRE_MAGIC
            or version != _WIRE_VERSION
            or not 0 < header_size <= _MAX_HEADER_BYTES
        ):
            raise _InvalidResultError
        header_bytes = _read_exact(descriptor, header_size)
        try:
            metadata = json.loads(header_bytes.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            raise _InvalidResultError from None
        if type(metadata) is not dict:
            raise _InvalidResultError
        expected_keys = {
            "channels",
            "chapter_id",
            "duration_ms",
            "engine_initializations",
            "frame_count",
            "index",
            "pcm_bytes",
            "pcm_type",
            "sample_rate_hz",
            "segment_id",
            "start_method",
            "worker_pid",
        }
        if set(metadata) != expected_keys:
            raise _InvalidResultError
        pcm_size = _exact_int(metadata["pcm_bytes"])
        if pcm_size < 0 or pcm_size > item.task.max_output_bytes:
            raise _InvalidResultError(ErrorCode.INVALID_AUDIO)
        if before.st_size != _WIRE_PREFIX.size + header_size + pcm_size:
            raise _InvalidResultError
        pcm = _read_exact(descriptor, pcm_size)
        if os.read(descriptor, 1):
            raise _InvalidResultError
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
        with suppress(OSError):
            item.result_path.unlink()
    identity_before = (
        before.st_dev,
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
        before.st_ctime_ns,
    )
    identity_after = (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
        after.st_ctime_ns,
    )
    if identity_before != identity_after:
        raise _InvalidResultError

    index = _exact_int(metadata["index"])
    engine_initializations = _exact_int(metadata["engine_initializations"])
    worker_pid = _exact_int(metadata["worker_pid"])
    start_method = _exact_str(metadata["start_method"])
    segment_id = _exact_str(metadata["segment_id"])
    chapter_id = _exact_str(metadata["chapter_id"])
    sample_rate_hz = _exact_int(metadata["sample_rate_hz"])
    channels = _exact_int(metadata["channels"])
    frame_count = _exact_int(metadata["frame_count"])
    duration_ms = _exact_int(metadata["duration_ms"])
    pcm_type = _exact_str(metadata["pcm_type"])
    expected_bytes = frame_count * channels * 2
    expected_duration = (
        frame_count * 1000 // sample_rate_hz if sample_rate_hz > 0 else -1
    )
    if (
        index != item.index
        or engine_initializations != 1
        or worker_pid != item.process.pid
        or start_method != "spawn"
        or segment_id != item.task.segment_id
        or chapter_id != item.task.chapter_id
        or sample_rate_hz != item.task.sample_rate_hz
        or channels != item.task.channels
        or frame_count <= 0
        or duration_ms <= 0
        or pcm_type != "bytes"
        or expected_bytes != pcm_size
        or expected_duration != duration_ms
    ):
        raise _InvalidResultError(ErrorCode.INVALID_AUDIO)
    audio = SynthesizedAudio(
        segment_id, chapter_id, pcm, sample_rate_hz, channels, frame_count, duration_ms
    )
    return WorkerRecord(audio, worker_pid, start_method, engine_initializations)


def render_spawned(  # noqa: C901, PLC0415, PLR0912, PLR0915
    tasks: tuple[SynthesisTask, ...],
    specification: EngineSpecification,
    worker_count: int,
    cancel: CancellationToken | None,
    *,
    max_total_bytes: int = _DEFAULT_TOTAL_PCM_BYTES,
) -> Iterator[WorkerRecord]:
    """Yield plan-ordered results from a bounded set of reusable spawn workers."""
    if not tasks:
        return
    if (
        worker_count < 1
        or worker_count > min(len(tasks), MAX_RENDER_WORKERS)
        or len(tasks) > _MAX_TASK_COUNT
        or type(max_total_bytes) is not int
        or max_total_bytes <= 0
    ):
        raise RenderError(ErrorCode.SYNTHESIS_FAILED)
    try:
        input_bytes = sum(
            len(task.segment_id.encode("utf-8"))
            + len(task.chapter_id.encode("utf-8"))
            + len(task.text.encode("utf-8"))
            for task in tasks
        )
    except (AttributeError, UnicodeEncodeError):
        raise RenderError(ErrorCode.SYNTHESIS_FAILED) from None
    if input_bytes > _MAX_TASK_INPUT_BYTES:
        raise RenderError(ErrorCode.SYNTHESIS_FAILED)
    if specification.kind == "pocket":
        if specification.pocket_config is None:
            raise RenderError(ErrorCode.SYNTHESIS_FAILED)
        from kenkui._tts.pocket import preflight_pocket

        preflight_pocket(specification.pocket_config)
        timeout = specification.pocket_config.timeout_seconds
    else:
        if specification.fake_config is None:
            raise RenderError(ErrorCode.SYNTHESIS_FAILED)
        timeout = specification.fake_config.timeout_seconds
    if (
        type(timeout) not in (int, float)
        or not math.isfinite(timeout)
        or timeout <= 0
        or timeout > _MAX_WORKER_TIMEOUT_SECONDS
    ):
        raise RenderError(ErrorCode.SYNTHESIS_FAILED)

    # Contiguous static partitions serialize the engine specification once per
    # worker and each task once, instead of duplicating the corpus per worker.
    batches: list[tuple[tuple[int, SynthesisTask], ...]] = []
    start = 0
    for worker_index in range(worker_count):
        size = (len(tasks) - start + worker_count - worker_index - 1) // (
            worker_count - worker_index
        )
        stop = start + size
        batches.append(tuple(enumerate(tasks[start:stop], start)))
        start = stop

    context = multiprocessing.get_context("spawn")
    thread_limit = resolve_thread_limit(worker_count)
    active: dict[int, _Active] = {}
    ready: dict[int, WorkerRecord] = {}
    next_emit = 0
    total_bytes = 0
    with tempfile.TemporaryDirectory(prefix="kenkui-render-") as workspace_name:
        workspace = Path(workspace_name).resolve(strict=True)
        try:
            for worker_index, assignments in enumerate(batches):
                failure_path = workspace / f"{_FAILURE_PREFIX}{worker_index}"
                process = context.Process(
                    target=synthesis_worker_entry,
                    args=(
                        str(workspace),
                        specification,
                        assignments,
                        str(failure_path),
                        thread_limit,
                    ),
                    name=f"kenkui-render-worker-{worker_index}",
                )
                try:
                    process.start()
                except BaseException:  # noqa: BLE001 - sanitize process startup.
                    with suppress(BaseException):
                        process.close()
                    _fail(active)
                active[worker_index] = _Active(
                    process,
                    assignments,
                    {index for index, _task in assignments},
                    time.monotonic() + float(timeout),
                )

            while next_emit < len(tasks):
                _check_cancel(cancel)
                made_progress = False
                for worker_index, state in tuple(active.items()):
                    for index, task in state.assignments:
                        if index not in state.pending:
                            continue
                        result_path = workspace / f"result-{index}.wire"
                        if not result_path.exists():
                            continue
                        item = _ResultItem(state.process, task, index, result_path)
                        try:
                            record = _parse_result(item, workspace)
                        except _InvalidResultError as invalid:
                            _fail(active, invalid.code)
                        state.pending.remove(index)
                        # A deadline is a per-worker no-progress deadline. Only a
                        # fully validated result earns an extension.
                        state.deadline = time.monotonic() + float(timeout)
                        total_bytes += len(record.audio.pcm_s16le)
                        if total_bytes > max_total_bytes:
                            _fail(active, ErrorCode.INVALID_AUDIO)
                        ready[index] = record
                        made_progress = True
                        break  # backpressure permits at most one file per worker

                    if state.process.exitcode is not None:
                        state.process.join(timeout=_WAIT_SLICE_SECONDS)
                        if state.process.is_alive():
                            _fail(active)
                        if state.process.exitcode != 0 or state.pending:
                            _fail(
                                active,
                                _reported_failure(
                                    workspace, worker_index, state
                                ),
                            )
                        with suppress(BaseException):
                            state.process.close()
                        active.pop(worker_index)

                # Before exposing the final record, let acknowledged workers run
                # their one engine cleanup. This also makes generator abandonment
                # after the last ``next`` unable to strand Pocket snapshots.
                if active and all(not state.pending for state in active.values()):
                    while active:
                        _check_cancel(cancel)
                        for worker_index, state in tuple(active.items()):
                            if state.process.exitcode is None:
                                continue
                            state.process.join(timeout=_WAIT_SLICE_SECONDS)
                            if state.process.is_alive() or state.process.exitcode != 0:
                                _fail(active)
                            with suppress(BaseException):
                                state.process.close()
                            active.pop(worker_index)
                        if active:
                            now = time.monotonic()
                            if any(state.deadline <= now for state in active.values()):
                                _fail(active)
                            _poll_wait(_WAIT_SLICE_SECONDS)

                while next_emit in ready:
                    record = ready.pop(next_emit)
                    next_emit += 1
                    yield record
                    _check_cancel(cancel)
                if next_emit >= len(tasks):
                    break
                if not active:
                    _fail(active)
                now = time.monotonic()
                if any(state.deadline <= now for state in active.values()):
                    _fail(active)
                if not made_progress:
                    wait_for = min(
                        _WAIT_SLICE_SECONDS,
                        max(
                            0.0,
                            min(state.deadline for state in active.values()) - now,
                        ),
                    )
                    _poll_wait(wait_for)
        finally:
            _terminate_and_reap(active)


def _reported_failure(
    workspace: Path, worker_index: int, state: _Active
) -> ErrorCode:
    """Log why a worker failed, then return the code the caller should receive."""
    code = _failure_code(
        workspace / f"{_FAILURE_PREFIX}{worker_index}", workspace
    )
    pending = sorted(state.pending)
    chapter_id = ""
    for index, task in state.assignments:
        if pending and index == pending[0]:
            chapter_id = task.chapter_id
            break
    log_event(
        _LOGGER,
        "worker_failed",
        level=logging.DEBUG,
        context={
            "error_code": code.value,
            "worker_index": worker_index,
            "exit_code": state.process.exitcode or 0,
            "pending_segments": len(pending),
            "chapter_id": chapter_id,
        },
    )
    return code


def _check_cancel(cancel: CancellationToken | None) -> None:
    if cancel is not None:
        cancel.raise_if_cancelled()


def _fail(
    active: dict[int, _Active], code: ErrorCode = ErrorCode.SYNTHESIS_FAILED
) -> NoReturn:
    _terminate_and_reap(active)
    raise RenderError(code) from None


def _terminate_and_reap_process(process: BaseProcess) -> None:
    """Reap one process with bounded terminate and kill escalation."""
    with suppress(BaseException):
        if process.is_alive():
            process.terminate()
    with suppress(BaseException):
        process.join(timeout=_TERMINATE_GRACE_SECONDS)
    with suppress(BaseException):
        if process.is_alive():
            process.kill()
    with suppress(BaseException):
        process.join(timeout=_KILL_GRACE_SECONDS)
    with suppress(BaseException):
        if not process.is_alive():
            process.close()


def _terminate_and_reap(active: dict[int, _Active]) -> None:
    items = tuple(active.values())
    active.clear()
    for item in items:
        _terminate_and_reap_process(item.process)
