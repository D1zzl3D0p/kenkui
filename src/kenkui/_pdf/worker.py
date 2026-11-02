"""Bounded subprocess lifetime for optional PDF layout and OCR extraction."""

from __future__ import annotations

import contextlib
import json
import os
import signal
import subprocess
import time
from typing import TYPE_CHECKING

from kenkui.errors import ErrorCode, SourceError

from .protocol import read_result

_MAX_STAGE_BYTES = 256

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

    from kenkui.cancellation import CancellationToken

    from .models import PdfDocument
    from .options import PdfLayoutOptions


def _stop(process: subprocess.Popen[bytes]) -> None:
    """Terminate the entire process group, including any OCR subprocess."""
    if os.name == "posix":
        _signal_process_group(process, signal.SIGTERM)
    elif process.poll() is None:
        process.terminate()
    try:
        process.wait(timeout=1)
    except subprocess.TimeoutExpired:
        process.kill()
    finally:
        if os.name == "posix":
            _signal_process_group(process, signal.SIGKILL)
        process.wait()


def _signal_process_group(
    process: subprocess.Popen[bytes], number: signal.Signals
) -> None:
    """Retry a macOS zombie-group race after reaping the exited leader."""
    with contextlib.suppress(ProcessLookupError):
        try:
            os.killpg(process.pid, number)
        except PermissionError as error:
            # macOS can report EPERM while an exiting group leader is becoming
            # a zombie, just before waitpid can reap it. Allow the same bounded
            # grace as normal shutdown, then still signal surviving OCR children.
            try:
                process.wait(timeout=1)
            except subprocess.TimeoutExpired:
                raise error from None
            try:
                os.killpg(process.pid, number)
            except PermissionError:
                # A group containing only zombies also returns EPERM on macOS.
                # Never mistake a live or unreadable group for successful cleanup.
                if not _group_is_terminated(process.pid):
                    raise


def _group_is_terminated(group_id: int) -> bool:
    """Require positive evidence that no process-group member can execute."""
    import psutil  # noqa: PLC0415 - required only by the layout worker

    for pid in psutil.pids():
        if pid < 1:
            continue
        try:
            if os.getpgid(pid) != group_id:
                continue
            if psutil.Process(pid).status() not in (
                psutil.STATUS_ZOMBIE,
                psutil.STATUS_DEAD,
            ):
                return False
        except (ProcessLookupError, psutil.NoSuchProcess):
            continue
        except (PermissionError, psutil.AccessDenied):
            return False
    return True


def _stage(
    progress: Path | None, previous: str | None, callback: Callable[[str], None] | None
) -> str | None:
    if progress is None or callback is None or not progress.exists():
        return previous
    if progress.stat().st_size > _MAX_STAGE_BYTES:
        raise SourceError(ErrorCode.INVALID_PDF_OUTPUT)
    try:
        stage = json.loads(progress.read_text())
    except (ValueError, OSError):
        raise SourceError(ErrorCode.INVALID_PDF_OUTPUT) from None
    if stage not in ("pdf.extract", "pdf.layout", "pdf.ocr"):
        raise SourceError(ErrorCode.INVALID_PDF_OUTPUT)
    if stage != previous:
        callback(stage)
    return str(stage)


def run_worker(  # noqa: C901, PLR0913 - explicit process limits and optional progress sink
    command: list[str],
    result: Path,
    source_hash: str,
    limits: PdfLayoutOptions,
    cancel: CancellationToken | None,
    *,
    progress: Path | None = None,
    on_stage: Callable[[str], None] | None = None,
) -> PdfDocument:
    """Monitor time, combined RSS and result size; always reap before returning."""
    try:
        import psutil  # noqa: PLC0415 - optional PDF layout dependency
    except ImportError:
        raise SourceError(ErrorCode.PDF_LAYOUT_UNAVAILABLE) from None
    if cancel is not None:
        cancel.raise_if_cancelled()
    result.unlink(missing_ok=True)
    started = time.monotonic()
    try:
        process = subprocess.Popen(  # noqa: S603 - argv, caller-owned paths, no shell
            command,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=os.name == "posix",
        )
    except OSError:
        raise SourceError(ErrorCode.PDF_WORKER_FAILED) from None
    try:
        monitored = psutil.Process(process.pid)
        current_stage = None
        while process.poll() is None:
            current_stage = _stage(progress, current_stage, on_stage)
            if cancel is not None:
                cancel.raise_if_cancelled()
            if time.monotonic() - started > limits.timeout_seconds:
                raise SourceError(ErrorCode.PDF_TIMEOUT)
            try:
                family = [monitored, *monitored.children(recursive=True)]
                rss = sum(p.memory_info().rss for p in family if p.is_running())
            except psutil.NoSuchProcess:
                rss = 0
            if rss > limits.max_memory_mb * 1024 * 1024:
                raise SourceError(ErrorCode.PDF_LIMIT)
            if result.exists() and result.stat().st_size > limits.max_output_bytes:
                raise SourceError(ErrorCode.PDF_LIMIT)
            time.sleep(0.05)
        if cancel is not None:
            cancel.raise_if_cancelled()
        _stage(progress, current_stage, on_stage)
        if process.returncode != 0:
            raise SourceError(ErrorCode.PDF_WORKER_FAILED)
        return read_result(result, source_hash, limits.max_output_bytes)
    finally:
        _stop(process)
