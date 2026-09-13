"""Sanitized bounded native-command execution boundary."""
# ruff: noqa: BLE001, S110, SIM105

from __future__ import annotations

import logging
import subprocess
import threading
from typing import TYPE_CHECKING, BinaryIO, Protocol, cast, runtime_checkable

from kenkui.errors import EncodingError, ErrorCode

_STDOUT_LIMIT_BYTES = 2 * 1024 * 1024
_STDERR_LIMIT_BYTES: int | None = None  # Unlimited: ffmpeg stderr must never overflow.
_READ_CHUNK_BYTES = 64 * 1024
_KILL_REAP_TIMEOUT_SECONDS = 5.0
_READER_JOIN_TIMEOUT_SECONDS = 5.0
_LOGGER = logging.getLogger(__name__)

if TYPE_CHECKING:
    from collections.abc import Sequence


@runtime_checkable
class NativeCommandRunner(Protocol):
    """Injectable argv-only subprocess boundary."""

    def run(
        self, argv: Sequence[str], *, timeout: float
    ) -> subprocess.CompletedProcess[str]:
        """Run one command without a shell and capture bounded diagnostics."""
        ...


class _BoundedCapture:
    """Drain one pipe to EOF; retain at most its cap, or everything if None."""

    def __init__(self, limit: int | None) -> None:
        self._limit = limit
        self._retained = bytearray()
        self.overflow = False
        self.error = False

    @property
    def data(self) -> bytes:
        return bytes(self._retained)

    def drain(self, pipe: BinaryIO) -> None:
        try:
            while True:
                chunk = pipe.read(_READ_CHUNK_BYTES)
                if not chunk:
                    break
                if self._limit is None:
                    self._retained.extend(chunk)
                else:
                    available = self._limit - len(self._retained)
                    if available > 0:
                        self._retained.extend(chunk[:available])
                    if len(chunk) > available:
                        self.overflow = True
        except Exception:
            self.error = True
        finally:
            try:
                pipe.close()
            except Exception:
                self.error = True


def _stop_and_reap(process: subprocess.Popen[bytes]) -> bool:
    """Best-effort bounded termination; report whether cleanup was complete."""
    clean = True
    try:
        process.kill()
    except Exception:
        clean = False
    try:
        process.wait(timeout=_KILL_REAP_TIMEOUT_SECONDS)
    except Exception:
        clean = False
    return clean


def _finish_readers(
    threads: tuple[threading.Thread, threading.Thread], pipes: tuple[BinaryIO, BinaryIO]
) -> bool:
    """Join readers for bounded periods and close pipes to unblock stragglers."""
    clean = True
    for thread in threads:
        try:
            thread.join(timeout=_READER_JOIN_TIMEOUT_SECONDS)
        except Exception:
            clean = False
    for thread, pipe in zip(threads, pipes, strict=True):
        if thread.is_alive():
            clean = False
            try:
                pipe.close()
            except Exception:
                pass
            try:
                thread.join(timeout=_READER_JOIN_TIMEOUT_SECONDS)
            except Exception:
                pass
        if thread.is_alive():
            clean = False
        try:
            pipe.close()
        except Exception:
            clean = False
    return clean


class SubprocessRunner:
    """Production subprocess implementation with in-flight bounded capture."""

    def run(
        self, argv: Sequence[str], *, timeout: float
    ) -> subprocess.CompletedProcess[str]:
        """Execute argv-only while concurrently draining both bounded pipes."""
        command = tuple(argv)
        try:
            process = subprocess.Popen(  # noqa: S603 - executable is preflight-resolved.
                command,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                close_fds=True,
                shell=False,
            )
        except Exception:
            raise subprocess.SubprocessError from None

        stdout_pipe = process.stdout
        stderr_pipe = process.stderr
        if stdout_pipe is None or stderr_pipe is None:
            _stop_and_reap(process)
            raise subprocess.SubprocessError
        pipes = cast("tuple[BinaryIO, BinaryIO]", (stdout_pipe, stderr_pipe))
        stdout_capture = _BoundedCapture(_STDOUT_LIMIT_BYTES)
        stderr_capture = _BoundedCapture(_STDERR_LIMIT_BYTES)
        threads = (
            threading.Thread(
                target=stdout_capture.drain, args=(stdout_pipe,), daemon=True
            ),
            threading.Thread(
                target=stderr_capture.drain, args=(stderr_pipe,), daemon=True
            ),
        )
        try:
            for thread in threads:
                thread.start()
        except Exception:
            _stop_and_reap(process)
            _finish_readers(threads, pipes)
            raise subprocess.SubprocessError from None

        wait_error: Exception | None = None
        returncode: int | None = None
        try:
            returncode = process.wait(timeout=timeout)
        except Exception as exc:
            wait_error = exc
            _stop_and_reap(process)
        readers_clean = _finish_readers(threads, pipes)
        if (
            wait_error is not None
            or not readers_clean
            or stdout_capture.error
            or stderr_capture.error
            or stdout_capture.overflow
            or stderr_capture.overflow
            or type(returncode) is not int
        ):
            if isinstance(wait_error, subprocess.TimeoutExpired):
                raise wait_error
            raise subprocess.SubprocessError from None
        return subprocess.CompletedProcess(
            command,
            returncode,
            stdout_capture.data.decode("utf-8", errors="replace"),
            stderr_capture.data.decode("utf-8", errors="replace"),
        )


def _failure_reason(stderr: object) -> str:
    """Classify known FFmpeg failures without logging untrusted tool output."""
    if type(stderr) is not str:
        return "nonzero_exit"
    detail = stderr.casefold()
    for needle, reason in (
        ("no space left on device", "no_space"),
        ("permission denied", "permission_denied"),
        ("invalid argument", "invalid_argument"),
        ("unknown encoder", "unknown_encoder"),
        ("error initializing output stream", "encoder_initialization_failed"),
        ("error opening output", "output_open_failed"),
        ("conversion failed", "conversion_failed"),
    ):
        if needle in detail:
            return reason
    return "nonzero_exit"


def run_checked(
    runner: NativeCommandRunner,
    argv: Sequence[str],
    *,
    timeout: float,
    code: ErrorCode,
) -> subprocess.CompletedProcess[str]:
    """Run a command and map every process detail to one stable public error."""
    result: subprocess.CompletedProcess[str] | None = None
    failed = False
    try:
        result = runner.run(tuple(argv), timeout=timeout)
    except subprocess.TimeoutExpired:
        _LOGGER.warning("native_command_failed code=%s reason=timeout", code.value)
        raise EncodingError(code) from None
    except (OSError, subprocess.SubprocessError):
        failed = True
    if failed or result is None:
        _LOGGER.warning("native_command_failed code=%s runner_error=true", code.value)
        raise EncodingError(code) from None
    if type(result.returncode) is not int or result.returncode != 0:
        _LOGGER.warning(
            "native_command_failed code=%s returncode=%s reason=%s",
            code.value,
            result.returncode,
            _failure_reason(result.stderr),
        )
        raise EncodingError(code) from None
    return result
