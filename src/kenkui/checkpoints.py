"""Opt-in durable checkpoints for one job, independent of storage provider.

Stores must isolate jobs, verify downloaded bytes, and atomically publish a
checkpoint only after its upload completes. Storage failures are fatal: silently
continuing would spend money on work that cannot survive another interruption.
"""

from __future__ import annotations

import sqlite3
from contextlib import closing, contextmanager
from contextvars import ContextVar
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import RLock
from typing import TYPE_CHECKING, Any, Protocol

if TYPE_CHECKING:
    from collections.abc import Iterator


class CheckpointStore(Protocol):
    """Thread-safe, job-scoped durable files with JSON-compatible metadata."""

    def restore(self, key: str, destination: Path) -> dict[str, Any] | None:
        """Download and verify a committed checkpoint, or return None on a miss."""
        ...

    def save(self, key: str, source: Path, metadata: dict[str, Any]) -> None:
        """Upload then atomically commit a verified checkpoint."""
        ...


class CheckpointSession:
    """One attempt's private attribution database and durable storage binding."""

    def __init__(self, store: CheckpointStore, root: Path) -> None:
        """Create an isolated attempt scope."""
        self.store = store
        self.root = root
        self.casting_path = root / "casting.sqlite3"
        self.lock = RLock()

    def save_casting(self) -> None:
        """Use SQLite backup to capture a committed, internally consistent store."""
        snapshot = self.root / "casting-snapshot.sqlite3"
        with self.lock:
            with (
                closing(sqlite3.connect(self.casting_path)) as source,
                closing(sqlite3.connect(snapshot)) as destination,
            ):
                source.backup(destination)
            try:
                self.store.save("casting-v1", snapshot, {})
            except OSError as error:
                # Local casting caches deliberately swallow OSError. A durable
                # write must escape those fail-open handlers and stop this run.
                message = "Durable attribution checkpoint could not be saved."
                raise RuntimeError(message) from error


_current: ContextVar[CheckpointSession | None] = ContextVar("checkpoints", default=None)


def current_session() -> CheckpointSession | None:
    """Return this execution context's binding; never a process-wide job global."""
    return _current.get()


@contextmanager
def checkpointing(store: CheckpointStore) -> Iterator[None]:
    """Resume and persist checkpoints for pipeline operations inside this scope.

    Use a separate store namespace per job. Child attribution threads inherit
    this scope explicitly; synthesis subprocesses do not access cloud storage.
    """
    with TemporaryDirectory(prefix="kenkui-checkpoints-") as directory:
        session = CheckpointSession(store, Path(directory))
        store.restore("casting-v1", session.casting_path)
        token = _current.set(session)
        try:
            yield
        finally:
            _current.reset(token)
