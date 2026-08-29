"""Deterministic branch tests for the private fail-open cache."""
# ruff: noqa: ARG001, ARG005, D103, EM101, FBT001, PLR2004, RUF059, S301, SIM117, SLF001, TC003, TRY003

from __future__ import annotations

import hashlib
import os
import pickle
import sqlite3
import stat
import time
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING, cast

import pytest

import kenkui._execution.cache as cache_module
from kenkui._execution.cache import CacheStore
from kenkui._execution.process_pool import EngineSpecification
from test_cache import _material

if TYPE_CHECKING:
    from collections.abc import Iterator

    from kenkui._domain.planning import ExecutionPlan, SpeechSegment
    from kenkui._tts.protocols import SynthesisTask


def _key(
    store: CacheStore,
    plan: ExecutionPlan,
    segment: SpeechSegment,
    task: SynthesisTask,
) -> str:
    return store.key_for(plan, segment, task, EngineSpecification.fake())


def test_context_entry_and_exit_failures_are_fail_open(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _, plan, segment, task, audio = _material(tmp_path)
    store = CacheStore(tmp_path / "cache")
    key = _key(store, plan, segment, task)

    @contextmanager
    def entry_failure(self: CacheStore) -> Iterator[sqlite3.Connection]:
        raise sqlite3.OperationalError("entry")
        yield  # make this a generator context manager

    monkeypatch.setattr(CacheStore, "_connection", entry_failure)
    assert store.prepare_run(plan) is None
    assert store.lookup(key, segment, task) is None
    store.store(key, segment, task, audio, None)
    store._clear()
    store._prune_relational_metadata()
    assert store._discard_metadata(key) is None

    @contextmanager
    def exit_failure(self: CacheStore) -> Iterator[sqlite3.Connection]:
        connection = sqlite3.connect(store._database, isolation_level=None)
        connection.row_factory = sqlite3.Row
        try:
            yield connection
        finally:
            connection.close()
        raise OSError("exit")

    monkeypatch.setattr(CacheStore, "_connection", exit_failure)
    assert store.prepare_run(plan) is None
    assert store.lookup(key, segment, task) is None


def test_lookup_audio_validation_and_discard_failures(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _, plan, segment, task, audio = _material(tmp_path)
    store = CacheStore(tmp_path / "cache")
    key = _key(store, plan, segment, task)
    store.store(key, segment, task, audio, None)

    # The cache key and stored row derive chapter identity from the immutable
    # segment, not the transport task.  A caller-local task chapter label is
    # therefore restored onto the returned audio without invalidating the hit.
    changed_task = replace(task, chapter_id="different")
    assert _key(store, plan, segment, changed_task) == key
    hit = store.lookup(key, segment, changed_task)
    assert hit is not None
    assert hit.chapter_id == "different"

    original = CacheStore._discard_metadata

    def discard_then_raise(self: CacheStore, discarded_key: str) -> str | None:
        original(self, discarded_key)
        raise OSError("discard raced")

    monkeypatch.setattr(CacheStore, "_discard_metadata", discard_then_raise)
    with sqlite3.connect(store._database) as connection:
        connection.execute(
            "UPDATE segment_cache SET schema_version='bad' WHERE cache_key=?", (key,)
        )
    # A discard callback fault is caught by lookup; the second fail-open discard is
    # allowed to complete without manufacturing a hit.
    calls = 0

    def one_failure(self: CacheStore, discarded_key: str) -> str | None:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise OSError("first discard")
        return original(self, discarded_key)

    monkeypatch.setattr(CacheStore, "_discard_metadata", one_failure)
    assert store.lookup(key, segment, task) is None
    assert calls == 2


@pytest.mark.parametrize("preexisting", [False, True])
def test_metadata_failure_reclaims_only_new_publication(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, preexisting: bool
) -> None:
    _, plan, segment, task, audio = _material(tmp_path)
    store = CacheStore(tmp_path / "cache")
    changed = bytes([audio.pcm_s16le[0] ^ 1]) + audio.pcm_s16le[1:]
    changed_audio = replace(audio, pcm_s16le=changed)
    digest = hashlib.sha256(changed).hexdigest()
    payload_path = store._payload_path(digest)
    if preexisting:
        assert store._publish_payload(digest, changed)
        assert payload_path.exists()

    @contextmanager
    def broken_connection(self: CacheStore) -> Iterator[sqlite3.Connection]:
        raise sqlite3.OperationalError("metadata unavailable")
        yield

    monkeypatch.setattr(CacheStore, "_connection", broken_connection)
    store.store(
        hashlib.sha256(f"key-{preexisting}".encode()).hexdigest(),
        segment,
        task,
        changed_audio,
        None,
    )
    assert payload_path.exists() is preexisting


def test_delete_clear_reference_and_relational_prune_branches(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _, plan, segment, task, audio = _material(tmp_path)
    store = CacheStore(tmp_path / "cache")
    context = store.prepare_run(plan)
    assert context is not None
    first = hashlib.sha256(b"first").hexdigest()
    second = hashlib.sha256(b"second").hexdigest()
    store.store(first, segment, task, audio, context)
    store.store(second, segment, task, audio, context)
    digest = hashlib.sha256(audio.pcm_s16le).hexdigest()

    store._delete(first)
    assert store._payload_path(digest).exists()  # still referenced by second
    store._delete("absent")
    store._delete_digest_if_unreferenced("not-a-digest")

    monkeypatch.setattr(cache_module, "MAX_RUN_ROWS", 0)
    store._prune_relational_metadata()
    with sqlite3.connect(store._database) as connection:
        assert (
            connection.execute("SELECT run_id FROM segment_cache").fetchone()[0] is None
        )
        assert connection.execute("SELECT count(*) FROM runs").fetchone()[0] == 0
    store._clear()
    with sqlite3.connect(store._database) as connection:
        assert connection.execute("SELECT count(*) FROM books").fetchone()[0] == 0
        assert connection.execute("SELECT count(*) FROM voices").fetchone()[0] == 0


def test_root_payload_and_database_mode_ownership_identity_validation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "cache"
    store = CacheStore(root)
    root.chmod(0o755)
    store._create_or_validate_root()
    assert stat.S_IMODE(root.stat().st_mode) == 0o700
    payloads = root / "payloads"
    payloads.chmod(0o755)
    with store._root_fd() as root_fd:
        store._create_or_validate_payload_directory(root_fd)
    assert stat.S_IMODE(payloads.stat().st_mode) == 0o700

    database = root / "cache.sqlite3"
    database.chmod(0o640)
    with pytest.raises(OSError, match="unsafe database"):
        store._database_identity()
    database.chmod(0o600)

    monkeypatch.setattr(cache_module, "_owned", lambda metadata: False)
    with pytest.raises(OSError, match="unsafe cache root"):
        with store._root_fd():
            pass


def test_root_and_payload_identity_races_are_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = CacheStore(tmp_path / "cache")
    real_file_id = cache_module._file_id
    calls = 0

    def changing_identity(metadata: os.stat_result) -> tuple[int, int]:
        nonlocal calls
        calls += 1
        identity = real_file_id(metadata)
        if calls == 2:
            return identity[0], identity[1] + 1
        return identity

    monkeypatch.setattr(cache_module, "_file_id", changing_identity)
    with pytest.raises(OSError, match="cache root raced"):
        with store._root_fd():
            pass

    monkeypatch.setattr(cache_module, "_file_id", real_file_id)
    with store._root_fd() as root_fd:
        real_stat = os.stat

        def mismatched_stat(
            path: str,
            *,
            dir_fd: int | None = None,
            follow_symlinks: bool = True,
        ) -> os.stat_result:
            result = real_stat(path, dir_fd=dir_fd, follow_symlinks=follow_symlinks)
            if path == cache_module._PAYLOAD_DIRECTORY:
                values = list(result)
                values[1] += 1
                return os.stat_result(values)
            return result

        monkeypatch.setattr(os, "stat", mismatched_stat)
        with pytest.raises(OSError, match="unsafe payload directory"):
            store._create_or_validate_payload_directory(root_fd)


def test_payload_publication_existing_valid_mismatch_and_unsafe_paths(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _, _, _, _, audio = _material(tmp_path)
    store = CacheStore(tmp_path / "cache")
    payload = audio.pcm_s16le
    digest = hashlib.sha256(payload).hexdigest()
    assert store._publish_payload(digest, payload)
    assert store._publish_payload(digest, payload)

    with store._payload_fd() as payload_fd:
        monkeypatch.setattr(cache_module, "_read_regular", lambda *args: b"different")
        assert store._publish_payload_locked(payload_fd, digest, payload) is None

    with store._payload_fd() as payload_fd:
        monkeypatch.setattr(cache_module, "_read_regular", lambda *args: payload)
        real_stat = os.stat

        def unsafe_stat(
            path: str,
            *,
            dir_fd: int | None = None,
            follow_symlinks: bool = True,
        ) -> os.stat_result:
            result = real_stat(path, dir_fd=dir_fd, follow_symlinks=follow_symlinks)
            values = list(result)
            values[0] = stat.S_IFDIR | 0o700
            return os.stat_result(values)

        monkeypatch.setattr(os, "stat", unsafe_stat)
        assert store._publish_payload_locked(payload_fd, digest, payload) is None


def test_digest_lock_live_timeout_and_unsafe_creation_have_no_sleep(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = CacheStore(tmp_path / "cache")
    digest = "a" * 64
    lock_path = store._payload_path(digest).with_suffix(".pcm.lock")
    lock_path.write_bytes(b"live")
    moments = iter((0.0, 1.0))
    monkeypatch.setattr(time, "monotonic", lambda: next(moments))
    monkeypatch.setattr(time, "sleep", lambda delay: None)
    with store._payload_fd() as payload_fd:
        with store._digest_lock(payload_fd, digest) as locked:
            assert not locked
    lock_path.unlink()
    monkeypatch.undo()

    real_fstat = os.fstat

    def unsafe_fstat(fd: int) -> os.stat_result:
        result = real_fstat(fd)
        values = list(result)
        values[0] = stat.S_IFREG | 0o640
        return os.stat_result(values)

    with store._payload_fd() as payload_fd:
        monkeypatch.setattr(os, "fstat", unsafe_fstat)
        with store._digest_lock(payload_fd, digest) as locked:
            assert not locked


def test_connection_identity_replacement_and_connect_errors_retry_without_sleep(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = CacheStore(tmp_path / "cache")
    monkeypatch.setattr(time, "sleep", lambda delay: None)
    calls = 0

    def alternating_identity(self: CacheStore) -> tuple[int, int]:
        nonlocal calls
        calls += 1
        return (1, calls)

    monkeypatch.setattr(CacheStore, "_database_identity", alternating_identity)
    with pytest.raises(OSError, match="database replaced during connect"):
        store._connection()

    monkeypatch.undo()
    monkeypatch.setattr(time, "sleep", lambda delay: None)
    calls = 0

    def after_connect_identity(self: CacheStore) -> tuple[int, int]:
        nonlocal calls
        calls += 1
        cycle = (calls - 1) % 3
        return (1, 1) if cycle < 2 else (1, 2)

    monkeypatch.setattr(CacheStore, "_database_identity", after_connect_identity)
    with pytest.raises(OSError, match="database replaced after connect"):
        store._connection()


def test_pickle_restore_invalid_root_and_fsync_failures_are_fail_open(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = CacheStore(tmp_path / "cache")
    restored = CacheStore.__new__(CacheStore)
    blocked = tmp_path / "blocked"
    blocked.write_bytes(b"not a directory")
    restored.__setstate__({"directory": str(blocked), "enabled": True})
    assert not restored._enabled
    assert not pickle.loads(pickle.dumps(restored))._enabled

    _, _, _, _, audio = _material(tmp_path / "material")
    digest = hashlib.sha256(audio.pcm_s16le).hexdigest()
    monkeypatch.setattr(
        os, "fsync", lambda fd: (_ for _ in ()).throw(OSError("unsupported"))
    )
    assert not store._publish_payload(digest, audio.pcm_s16le)
    assert not store._payload_path(digest).exists()


def test_orphan_scan_is_bounded_and_tolerates_stat_and_directory_failures(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = CacheStore(tmp_path / "cache")
    payloads = tmp_path / "cache" / "payloads"
    first = payloads / ".tmp-a"
    second = payloads / ".tmp-b"
    first.write_bytes(b"a")
    second.write_bytes(b"b")
    store._scan_orphans(1, older_than=None)
    assert sum(path.exists() for path in (first, second)) == 1

    remaining = first if first.exists() else second
    real_stat = os.stat

    def failed_stat(
        path: str,
        *,
        dir_fd: int | None = None,
        follow_symlinks: bool = True,
    ) -> os.stat_result:
        if path == remaining.name:
            raise OSError("raced")
        return real_stat(path, dir_fd=dir_fd, follow_symlinks=follow_symlinks)

    monkeypatch.setattr(os, "stat", failed_stat)
    store._scan_orphans(10, older_than=None)
    assert remaining.exists()
    monkeypatch.undo()

    payloads.chmod(0o755)
    store._scan_orphans(10, older_than=None)
    payloads.chmod(0o700)


def test_low_level_row_read_unlink_and_ownership_validation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _, _, segment, task, audio = _material(tmp_path)
    assert not cache_module._row_matches(cast("sqlite3.Row", {}), segment, task)
    assert not cache_module._safe_hex(cast("str", 123))

    directory = tmp_path / "files"
    directory.mkdir()
    descriptor = os.open(directory, cache_module._DIR_FLAGS)
    try:
        name = "payload"
        path = directory / name
        path.write_bytes(audio.pcm_s16le)
        path.chmod(0o600)
        digest = hashlib.sha256(audio.pcm_s16le).hexdigest()
        assert (
            cache_module._read_regular(
                descriptor, name, len(audio.pcm_s16le) + 1, digest
            )
            is None
        )
        identity = cache_module._file_id(path.stat())
        assert not cache_module._unlink_safe_regular(
            descriptor, name, identity=(identity[0], identity[1] + 1)
        )
        monkeypatch.setattr(cache_module, "_owned", lambda metadata: False)
        assert not cache_module._safe_regular(path.stat())
    finally:
        os.close(descriptor)


def test_maintenance_keeps_every_entry_without_a_quota(tmp_path: Path) -> None:
    """The byte and entry caps are gone; maintenance must not evict anything."""
    _, plan, segment, task, audio = _material(tmp_path)
    store = CacheStore(tmp_path / "cache")
    first = _key(store, plan, segment, task)
    second = hashlib.sha256(b"cap-second").hexdigest()
    store.store(first, segment, task, audio, None)
    changed = bytes([audio.pcm_s16le[0] ^ 1]) + audio.pcm_s16le[1:]
    store.store(second, segment, task, replace(audio, pcm_s16le=changed), None)

    store._maintenance()

    with sqlite3.connect(store._database) as connection:
        count = connection.execute("SELECT count(*) FROM segment_cache").fetchone()[0]
    assert count == 2


def test_clear_book_removes_only_that_books_rows_and_unique_payloads(
    tmp_path: Path,
) -> None:
    """A finished book releases its audio; other books keep theirs."""
    _, plan, segment, task, audio = _material(tmp_path)
    store = CacheStore(tmp_path / "cache")
    context = store.prepare_run(plan)
    other_plan = replace(plan, source_bytes_hash="b" * 64)
    other_context = store.prepare_run(other_plan)

    shared = _key(store, plan, segment, task)
    store.store(shared, segment, task, audio, context)
    # The other book's segment shares the exact PCM bytes, so the payload is
    # content-addressed across books and must survive the first book's clear.
    other_key = hashlib.sha256(b"other-book-segment").hexdigest()
    store.store(other_key, segment, task, audio, other_context)
    # And one payload unique to the first book.
    changed = bytes([audio.pcm_s16le[0] ^ 1]) + audio.pcm_s16le[1:]
    unique_digest = hashlib.sha256(changed).hexdigest()
    unique_key = hashlib.sha256(b"unique-first-book").hexdigest()
    store.store(unique_key, segment, task, replace(audio, pcm_s16le=changed), context)

    store.clear_book(plan.source_bytes_hash)

    with sqlite3.connect(store._database) as connection:
        remaining = connection.execute("SELECT book_id FROM segment_cache").fetchall()
        assert [row[0] for row in remaining] == ["b" * 64]
        assert connection.execute("SELECT count(*) FROM books").fetchone()[0] == 1
        assert connection.execute("SELECT count(*) FROM runs").fetchone()[0] == 1
    payloads = store._directory / "payloads"
    assert not (payloads / f"{unique_digest}.pcm").exists()
    assert store.lookup(other_key, segment, task) is not None
