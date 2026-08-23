"""Private, fail-open SQLite metadata and content-addressed PCM cache.

All filesystem operations which can affect cache payloads are performed relative
 to validated, non-followed directory descriptors.  The cache is optional: any
 unsafe state, race, lock timeout, corruption, or I/O failure is a cache miss.
"""
# ruff: noqa: E501, EM101, PLR2004, RUF023, S101, SIM105, SIM117, TRY003, TRY300, TRY301

from __future__ import annotations

import hashlib
import json
import os
import secrets
import sqlite3
import stat
import time
from contextlib import closing, contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from kenkui._tts.protocols import SynthesisTask, SynthesizedAudio

if TYPE_CHECKING:
    from collections.abc import Iterator

    from kenkui._domain.planning import ExecutionPlan, SpeechSegment
    from kenkui._execution.process_pool import EngineSpecification

CACHE_SCHEMA_VERSION = "segment-cache-v2"
AUDIO_CONTRACT_VERSION = "pcm-s16le-v1"
_DB_NAME = "cache.sqlite3"
_PAYLOAD_DIRECTORY = "payloads"
_TEMP_PREFIX = ".tmp-"
_LOCK_SUFFIX = ".lock"
_PAYLOAD_SUFFIX = ".pcm"
_CHUNK_BYTES = 64 * 1024
_MAX_PAYLOAD_BYTES = 64 * 1024 * 1024
_BUSY_TIMEOUT_MS = 75
_RETRIES = 2
_RETRY_DELAY_SECONDS = 0.025
_LOCK_WAIT_SECONDS = 0.5
_ORPHAN_AGE_SECONDS = 24 * 60 * 60

# Finite persistent-retention policy.  Maintenance performs at most one bounded
# batch per call, so a poisoned database cannot force unbounded work or memory.
MAX_SEGMENT_ENTRIES = 4096
MAX_TOTAL_PAYLOAD_BYTES = 512 * 1024 * 1024
MAX_RUN_ROWS = 1024
MAINTENANCE_BATCH = 32
ORPHAN_SCAN_LIMIT = 64

_HEX = frozenset("0123456789abcdef")
_DIR_FLAGS = (
    os.O_RDONLY
    | getattr(os, "O_DIRECTORY", 0)
    | getattr(os, "O_NOFOLLOW", 0)
    | getattr(os, "O_CLOEXEC", 0)
)
_FILE_NOFOLLOW = getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_CLOEXEC", 0)


@dataclass(frozen=True, slots=True)
class _RunContext:
    """Opaque parent-only relational metadata; never part of a cache key."""

    run_id: str
    book_id: str
    voice_id: str


@dataclass(frozen=True, slots=True)
class _PayloadPublication:
    """A validated payload publication made while holding its digest lock."""

    newly_created: bool
    identity: tuple[int, int]


class CacheStore:
    """Private fail-open cache which never retains a connection or descriptor."""

    __slots__ = ("_directory", "_database", "_enabled")

    def __init__(self, directory: Path) -> None:
        self._directory = Path(directory)
        self._database = self._directory / _DB_NAME
        self._enabled = self._initialize()
        if self._enabled:
            self._maintenance()

    def __getstate__(self) -> dict[str, str | bool]:
        """Explicitly serialize only inert values, never connections or fds."""
        return {"directory": os.fspath(self._directory), "enabled": self._enabled}

    def __setstate__(self, state: dict[str, str | bool]) -> None:
        """Revalidate the store instead of trusting serialized enabled state."""
        self._directory = Path(str(state["directory"]))
        self._database = self._directory / _DB_NAME
        self._enabled = self._initialize()

    def prepare_run(self, plan: ExecutionPlan) -> _RunContext | None:
        """Record private book/voice/run metadata, failing open on any fault."""
        if not self._enabled:
            return None
        book_id = plan.source_bytes_hash
        voice_material = _voice_material(plan)
        voice_id = _sha256_json(voice_material)
        run_id = secrets.token_hex(16)
        now = time.time_ns()
        try:
            with self._connection() as connection:
                connection.execute("BEGIN IMMEDIATE")
                connection.execute(
                    "INSERT OR IGNORE INTO books(book_id,source_sha256,created_ns) VALUES(?,?,?)",
                    (book_id, plan.source_bytes_hash, now),
                )
                connection.execute(
                    "INSERT OR IGNORE INTO voices(voice_id,content_fingerprint,metadata_json,created_ns) VALUES(?,?,?,?)",
                    (
                        voice_id,
                        plan.cast.narrator.content_fingerprint,
                        _canonical_json(voice_material),
                        now,
                    ),
                )
                connection.execute(
                    "INSERT INTO runs(run_id,book_id,voice_id,plan_fingerprint,created_ns) VALUES(?,?,?,?,?)",
                    (run_id, book_id, voice_id, plan.semantic_fingerprint, now),
                )
                connection.commit()
            self._prune_relational_metadata()
            return _RunContext(run_id, book_id, voice_id)
        except (OSError, sqlite3.Error, ValueError):
            return None

    def key_for(
        self,
        plan: ExecutionPlan,
        segment: SpeechSegment,
        task: SynthesisTask,
        specification: EngineSpecification,
    ) -> str:
        """Return a key containing semantic PCM inputs and no shell/run controls."""
        material = {
            "audio_contract_version": AUDIO_CONTRACT_VERSION,
            "cache_schema_version": CACHE_SCHEMA_VERSION,
            "engine": _engine_material(
                specification,
                plan.cast.voice_for(segment.speaker_id).content_fingerprint,
            ),
            "model_revision": plan.model_revision,
            "normalization_schema": plan.schema_versions.normalization,
            "render_schema": plan.schema_versions.render,
            "segment": {
                "chapter_id_sha256": hashlib.sha256(
                    segment.chapter_id.encode()
                ).hexdigest(),
                "character_count": segment.character_count,
                "content_hash": segment.content_hash,
                "id": segment.id,
                "ordinal": segment.ordinal,
            },
            "voice": _voice_material(plan, segment),
            "pcm": {"channels": task.channels, "sample_rate_hz": task.sample_rate_hz},
        }
        return _sha256_json(material)

    def lookup(
        self, key: str, segment: SpeechSegment, task: SynthesisTask
    ) -> SynthesizedAudio | None:
        """Return fully verified PCM or evict bad metadata and report a miss."""
        if not self._enabled:
            return None
        row: sqlite3.Row | None = None
        try:
            with self._connection() as connection:
                row = connection.execute(
                    "SELECT * FROM segment_cache WHERE cache_key=?", (key,)
                ).fetchone()
            if row is None or not _row_matches(row, segment, task):
                if row is not None:
                    self._discard_metadata(key)
                return None
            payload = self._read_payload(
                str(row["payload_sha256"]), int(row["payload_bytes"])
            )
            if payload is None:
                self._discard_metadata(key)
                return None
            audio = SynthesizedAudio(
                task.segment_id,
                task.chapter_id,
                payload,
                int(row["sample_rate_hz"]),
                int(row["channels"]),
                int(row["frame_count"]),
                int(row["duration_ms"]),
            )
            if not _audio_matches(audio, task):
                self._discard_metadata(key)
                return None
            self._touch(key)
            return audio
        except (OSError, sqlite3.Error, TypeError, ValueError, OverflowError):
            if row is not None:
                self._discard_metadata(key)
            return None

    def store(
        self,
        key: str,
        segment: SpeechSegment,
        task: SynthesisTask,
        audio: SynthesizedAudio,
        context: _RunContext | None,
    ) -> None:
        """Publish payload and metadata under one digest lock, then maintain quotas."""
        if not self._enabled or not _safe_hex(key) or not _audio_matches(audio, task):
            return
        digest = hashlib.sha256(audio.pcm_s16le).hexdigest()
        try:
            with self._payload_fd() as payload_fd:
                with self._digest_lock(payload_fd, digest) as locked:
                    if not locked:
                        return
                    publication = self._publish_payload_locked(
                        payload_fd, digest, audio.pcm_s16le
                    )
                    if publication is None:
                        return
                    try:
                        now = time.time_ns()
                        with self._connection() as connection:
                            try:
                                connection.execute("BEGIN IMMEDIATE")
                                connection.execute(
                                    """INSERT INTO segment_cache(
                                    cache_key,schema_version,audio_contract_version,segment_id,
                                    chapter_id_sha256,ordinal,text_sha256,character_count,
                                    sample_rate_hz,channels,frame_count,duration_ms,payload_sha256,
                                    payload_bytes,book_id,voice_id,run_id,created_ns,access_ns)
                                    VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                                    ON CONFLICT(cache_key) DO UPDATE SET
                                    schema_version=excluded.schema_version,
                                    audio_contract_version=excluded.audio_contract_version,
                                    segment_id=excluded.segment_id,
                                    chapter_id_sha256=excluded.chapter_id_sha256,
                                    ordinal=excluded.ordinal,text_sha256=excluded.text_sha256,
                                    character_count=excluded.character_count,
                                    sample_rate_hz=excluded.sample_rate_hz,channels=excluded.channels,
                                    frame_count=excluded.frame_count,duration_ms=excluded.duration_ms,
                                    payload_sha256=excluded.payload_sha256,payload_bytes=excluded.payload_bytes,
                                    book_id=excluded.book_id,voice_id=excluded.voice_id,run_id=excluded.run_id,
                                    created_ns=excluded.created_ns,access_ns=excluded.access_ns""",
                                    (
                                        key,
                                        CACHE_SCHEMA_VERSION,
                                        AUDIO_CONTRACT_VERSION,
                                        segment.id,
                                        hashlib.sha256(
                                            segment.chapter_id.encode()
                                        ).hexdigest(),
                                        segment.ordinal,
                                        segment.content_hash,
                                        segment.character_count,
                                        audio.sample_rate_hz,
                                        audio.channels,
                                        audio.frame_count,
                                        audio.duration_ms,
                                        digest,
                                        len(audio.pcm_s16le),
                                        context.book_id if context else None,
                                        context.voice_id if context else None,
                                        context.run_id if context else None,
                                        now,
                                        now,
                                    ),
                                )
                                connection.commit()
                            except (
                                OSError,
                                sqlite3.Error,
                                TypeError,
                                ValueError,
                                OverflowError,
                            ):
                                connection.rollback()
                                raise
                    except (
                        OSError,
                        sqlite3.Error,
                        TypeError,
                        ValueError,
                        OverflowError,
                    ):
                        if publication.newly_created:
                            self._unlink_publication_locked(
                                payload_fd,
                                digest,
                                audio.pcm_s16le,
                                publication.identity,
                            )
                        return
            self._maintenance()
        except (OSError, sqlite3.Error, TypeError, ValueError, OverflowError):
            return

    def _delete(self, key: str) -> None:
        """Delete one metadata row and race-safely reclaim its payload."""
        digest = self._discard_metadata(key)
        if digest is not None:
            self._delete_digest_if_unreferenced(digest)
        self._prune_relational_metadata()

    def _clear(self) -> None:
        """Clear metadata, then boundedly reclaim payloads."""
        if not self._enabled:
            return
        try:
            with self._connection() as connection:
                connection.execute("BEGIN IMMEDIATE")
                connection.execute("DELETE FROM segment_cache")
                connection.execute("DELETE FROM runs")
                connection.execute("DELETE FROM books")
                connection.execute("DELETE FROM voices")
                connection.commit()
            self._scan_orphans(MAINTENANCE_BATCH, older_than=None)
        except (OSError, sqlite3.Error):
            return

    def _initialize(self) -> bool:
        try:
            self._create_or_validate_root()
            with self._root_fd() as root_fd:
                self._create_or_validate_payload_directory(root_fd)
                self._create_or_validate_database(root_fd)
            with self._connection() as connection:
                connection.executescript(_SCHEMA)
                row = connection.execute(
                    "SELECT value FROM schema_metadata WHERE name='schema_version'"
                ).fetchone()
                if row is None:
                    connection.execute(
                        "INSERT INTO schema_metadata(name,value) VALUES('schema_version',?)",
                        (CACHE_SCHEMA_VERSION,),
                    )
                    connection.commit()
                    return True
                return bool(row[0] == CACHE_SCHEMA_VERSION)
        except (OSError, sqlite3.Error, ValueError):
            return False

    def _create_or_validate_root(self) -> None:
        try:
            metadata = os.lstat(self._directory)
        except FileNotFoundError:
            self._directory.mkdir(parents=True, mode=0o700)
            metadata = os.lstat(self._directory)
        if not stat.S_ISDIR(metadata.st_mode) or not _owned(metadata):
            raise OSError("unsafe cache root")
        descriptor = os.open(self._directory, _DIR_FLAGS)
        try:
            opened = os.fstat(descriptor)
            if (
                _file_id(metadata) != _file_id(opened)
                or not stat.S_ISDIR(opened.st_mode)
                or not _owned(opened)
            ):
                raise OSError("cache root raced")
            if stat.S_IMODE(opened.st_mode) != 0o700:
                os.fchmod(descriptor, 0o700)
            after = os.fstat(descriptor)
            if (
                _file_id(opened) != _file_id(after)
                or stat.S_IMODE(after.st_mode) != 0o700
            ):
                raise OSError("cache root mode race")
        finally:
            os.close(descriptor)

    @contextmanager
    def _root_fd(self) -> Iterator[int]:
        before = os.lstat(self._directory)
        if (
            not stat.S_ISDIR(before.st_mode)
            or not _owned(before)
            or stat.S_IMODE(before.st_mode) != 0o700
        ):
            raise OSError("unsafe cache root")
        descriptor = os.open(self._directory, _DIR_FLAGS)
        try:
            after = os.fstat(descriptor)
            if (
                _file_id(before) != _file_id(after)
                or not stat.S_ISDIR(after.st_mode)
                or not _owned(after)
                or stat.S_IMODE(after.st_mode) != 0o700
            ):
                raise OSError("cache root raced")
            yield descriptor
        finally:
            os.close(descriptor)

    def _create_or_validate_payload_directory(self, root_fd: int) -> None:
        try:
            os.mkdir(_PAYLOAD_DIRECTORY, 0o700, dir_fd=root_fd)
        except FileExistsError:
            pass
        descriptor = os.open(_PAYLOAD_DIRECTORY, _DIR_FLAGS, dir_fd=root_fd)
        try:
            metadata = os.fstat(descriptor)
            named = os.stat(_PAYLOAD_DIRECTORY, dir_fd=root_fd, follow_symlinks=False)
            if (
                _file_id(metadata) != _file_id(named)
                or not stat.S_ISDIR(metadata.st_mode)
                or not _owned(metadata)
            ):
                raise OSError("unsafe payload directory")
            if stat.S_IMODE(metadata.st_mode) != 0o700:
                os.fchmod(descriptor, 0o700)
            if stat.S_IMODE(os.fstat(descriptor).st_mode) != 0o700:
                raise OSError("payload mode race")
        finally:
            os.close(descriptor)

    @contextmanager
    def _payload_fd(self) -> Iterator[int]:
        with self._root_fd() as root_fd:
            descriptor = os.open(_PAYLOAD_DIRECTORY, _DIR_FLAGS, dir_fd=root_fd)
            try:
                metadata = os.fstat(descriptor)
                named = os.stat(
                    _PAYLOAD_DIRECTORY, dir_fd=root_fd, follow_symlinks=False
                )
                if (
                    _file_id(metadata) != _file_id(named)
                    or not stat.S_ISDIR(metadata.st_mode)
                    or not _owned(metadata)
                    or stat.S_IMODE(metadata.st_mode) != 0o700
                ):
                    raise OSError("unsafe payload directory")
                yield descriptor
            finally:
                os.close(descriptor)

    def _create_or_validate_database(self, root_fd: int) -> None:
        try:
            metadata = os.stat(_DB_NAME, dir_fd=root_fd, follow_symlinks=False)
        except FileNotFoundError:
            descriptor = os.open(
                _DB_NAME,
                os.O_WRONLY | os.O_CREAT | os.O_EXCL | _FILE_NOFOLLOW,
                0o600,
                dir_fd=root_fd,
            )
            try:
                metadata = os.fstat(descriptor)
                if not _safe_regular(metadata, mode=0o600):
                    raise OSError("unsafe new database")
            finally:
                os.close(descriptor)
            return
        if not _safe_regular(metadata, mode=0o600) or not _owned(metadata):
            raise OSError("unsafe existing database")

    def _database_identity(self) -> tuple[int, int]:
        with self._root_fd() as root_fd:
            metadata = os.stat(_DB_NAME, dir_fd=root_fd, follow_symlinks=False)
            if not _safe_regular(metadata, mode=0o600) or not _owned(metadata):
                raise OSError("unsafe database")
            return _file_id(metadata)

    def _connection(self) -> closing[sqlite3.Connection]:
        last: BaseException | None = None
        for attempt in range(_RETRIES + 1):
            connection: sqlite3.Connection | None = None
            try:
                before = self._database_identity()
                connection = sqlite3.connect(
                    self._database,
                    timeout=_BUSY_TIMEOUT_MS / 1000,
                    isolation_level=None,
                )
                # Do not issue a write-capable pragma before proving that SQLite opened
                # the same private regular file we validated.
                after = self._database_identity()
                if before != after:
                    raise OSError("database replaced during connect")
                connection.row_factory = sqlite3.Row
                connection.execute(f"PRAGMA busy_timeout={_BUSY_TIMEOUT_MS}")
                connection.execute("PRAGMA foreign_keys=ON")
                # Metadata is disposable. A memory journal avoids opening
                # attacker-planted sidecar paths; corruption always fails open.
                connection.execute("PRAGMA journal_mode=MEMORY")
                if self._database_identity() != before:
                    raise OSError("database replaced after connect")
                return closing(connection)
            except (OSError, sqlite3.Error) as error:
                if connection is not None:
                    connection.close()
                last = error
                if attempt < _RETRIES:
                    time.sleep(_RETRY_DELAY_SECONDS * (attempt + 1))
        assert last is not None
        raise last

    def _payload_path(self, digest: str) -> Path:
        """Return the diagnostic path; security-sensitive operations never use it."""
        return self._directory / _PAYLOAD_DIRECTORY / f"{digest}{_PAYLOAD_SUFFIX}"

    def _read_payload(self, digest: str, expected_size: int) -> bytes | None:
        if (
            not _safe_hex(digest)
            or expected_size <= 0
            or expected_size > _MAX_PAYLOAD_BYTES
        ):
            return None
        try:
            with self._payload_fd() as payload_fd:
                return _read_regular(
                    payload_fd, _payload_name(digest), expected_size, digest
                )
        except OSError:
            return None

    def _publish_payload(self, digest: str, payload: bytes) -> bool:
        """Test/private helper that applies the same lock protocol as store."""
        if not _safe_hex(digest) or not payload or len(payload) > _MAX_PAYLOAD_BYTES:
            return False
        try:
            with self._payload_fd() as payload_fd:
                with self._digest_lock(payload_fd, digest) as locked:
                    return bool(
                        locked
                        and self._publish_payload_locked(payload_fd, digest, payload)
                        is not None
                    )
        except OSError:
            return False

    def _publish_payload_locked(  # noqa: C901, PLR0912, PLR0915 - linear defensive validation.
        self, payload_fd: int, digest: str, payload: bytes
    ) -> _PayloadPublication | None:
        name = _payload_name(digest)
        existing = _read_regular(payload_fd, name, len(payload), digest)
        if existing is not None:
            if existing != payload:
                return None
            metadata = os.stat(name, dir_fd=payload_fd, follow_symlinks=False)
            if not _safe_regular(metadata, mode=0o600):
                return None
            return _PayloadPublication(newly_created=False, identity=_file_id(metadata))
        preexisted = False
        try:
            named = os.stat(name, dir_fd=payload_fd, follow_symlinks=False)
        except FileNotFoundError:
            pass
        else:
            preexisted = True
            # Repair corrupt private regular files, but reject unsafe nodes. A
            # preexisting name is never reclaimed after a metadata failure.
            if not _safe_regular(named, mode=0o600):
                return None
        temporary = f"{_TEMP_PREFIX}{secrets.token_hex(16)}"
        descriptor = -1
        publication_identity: tuple[int, int] | None = None
        try:
            descriptor = os.open(
                temporary,
                os.O_WRONLY | os.O_CREAT | os.O_EXCL | _FILE_NOFOLLOW,
                0o600,
                dir_fd=payload_fd,
            )
            created = os.fstat(descriptor)
            named = os.stat(temporary, dir_fd=payload_fd, follow_symlinks=False)
            if _file_id(created) != _file_id(named) or not _safe_regular(
                created, mode=0o600
            ):
                raise OSError("unsafe temporary")
            hasher = hashlib.sha256()
            view = memoryview(payload)
            written = 0
            while written < len(view):
                count = os.write(descriptor, view[written : written + _CHUNK_BYTES])
                if count <= 0:
                    raise OSError("short write")
                hasher.update(view[written : written + count])
                written += count
            if written != len(payload) or hasher.hexdigest() != digest:
                raise OSError("payload digest mismatch")
            os.fsync(descriptor)
            final_state = os.fstat(descriptor)
            if _file_id(created) != _file_id(final_state) or not _safe_regular(
                final_state, mode=0o600
            ):
                raise OSError("temporary raced")
            os.close(descriptor)
            descriptor = -1
            os.replace(
                temporary,
                name,
                src_dir_fd=payload_fd,
                dst_dir_fd=payload_fd,
            )
            publication_identity = _file_id(created)
            os.fsync(payload_fd)
            if _read_regular(payload_fd, name, len(payload), digest) != payload:
                raise OSError("published payload failed validation")
            return _PayloadPublication(
                newly_created=not preexisted, identity=publication_identity
            )
        except OSError:
            if not preexisted and publication_identity is not None:
                self._unlink_publication_locked(
                    payload_fd, digest, payload, publication_identity
                )
            return None
        finally:
            if descriptor >= 0:
                os.close(descriptor)
            _unlink_safe_regular(payload_fd, temporary)

    def _unlink_publication_locked(
        self,
        payload_fd: int,
        digest: str,
        payload: bytes,
        identity: tuple[int, int],
    ) -> None:
        """Validate and unlink exactly one new payload under its held digest lock."""
        name = _payload_name(digest)
        if _read_regular(payload_fd, name, len(payload), digest) != payload:
            return
        if _unlink_safe_regular(payload_fd, name, identity=identity):
            os.fsync(payload_fd)

    @contextmanager
    def _digest_lock(self, payload_fd: int, digest: str) -> Iterator[bool]:
        name = _lock_name(digest)
        descriptor = -1
        deadline = time.monotonic() + _LOCK_WAIT_SECONDS
        while descriptor < 0:
            try:
                descriptor = os.open(
                    name,
                    os.O_WRONLY | os.O_CREAT | os.O_EXCL | _FILE_NOFOLLOW,
                    0o600,
                    dir_fd=payload_fd,
                )
                metadata = os.fstat(descriptor)
                named = os.stat(name, dir_fd=payload_fd, follow_symlinks=False)
                if _file_id(metadata) != _file_id(named) or not _safe_regular(
                    metadata, mode=0o600
                ):
                    raise OSError("unsafe lock")
            except FileExistsError:
                if time.monotonic() >= deadline:
                    yield False
                    return
                time.sleep(_RETRY_DELAY_SECONDS)
            except OSError:
                if descriptor >= 0:
                    os.close(descriptor)
                    descriptor = -1
                yield False
                return
        try:
            yield True
        finally:
            identity = _file_id(os.fstat(descriptor))
            os.close(descriptor)
            _unlink_safe_regular(payload_fd, name, identity=identity)

    def _touch(self, key: str) -> None:
        try:
            with self._connection() as connection:
                connection.execute(
                    "UPDATE segment_cache SET access_ns=? WHERE cache_key=?",
                    (time.time_ns(), key),
                )
        except (OSError, sqlite3.Error):
            pass

    def _discard_metadata(self, key: str) -> str | None:
        if not self._enabled:
            return None
        try:
            with self._connection() as connection:
                connection.execute("BEGIN IMMEDIATE")
                row = connection.execute(
                    "SELECT payload_sha256 FROM segment_cache WHERE cache_key=?", (key,)
                ).fetchone()
                connection.execute(
                    "DELETE FROM segment_cache WHERE cache_key=?", (key,)
                )
                connection.commit()
                return None if row is None else str(row[0])
        except (OSError, sqlite3.Error):
            return None

    def _delete_digest_if_unreferenced(
        self, digest: str, *, older_than: float | None = None
    ) -> None:
        if not _safe_hex(digest):
            return
        try:
            with self._payload_fd() as payload_fd:
                with self._digest_lock(payload_fd, digest) as locked:
                    if not locked:
                        return
                    with self._connection() as connection:
                        connection.execute("BEGIN IMMEDIATE")
                        referenced = connection.execute(
                            "SELECT 1 FROM segment_cache WHERE payload_sha256=? LIMIT 1",
                            (digest,),
                        ).fetchone()
                        if referenced is None:
                            name = _payload_name(digest)
                            try:
                                metadata = os.stat(
                                    name, dir_fd=payload_fd, follow_symlinks=False
                                )
                            except FileNotFoundError:
                                metadata = None
                            if metadata is not None and (
                                older_than is None or metadata.st_mtime < older_than
                            ):
                                _unlink_safe_regular(payload_fd, name)
                        connection.commit()
        except (OSError, sqlite3.Error):
            return

    def _maintenance(self) -> None:
        self._prune_entries()
        self._prune_relational_metadata()
        self._scan_orphans(
            ORPHAN_SCAN_LIMIT, older_than=time.time() - _ORPHAN_AGE_SECONDS
        )

    def _prune_entries(self) -> None:
        """Evict at most one deterministic LRU batch and safely reclaim payloads."""
        try:
            with self._connection() as connection:
                count = int(
                    connection.execute("SELECT count(*) FROM segment_cache").fetchone()[
                        0
                    ]
                )
                total = int(
                    connection.execute(
                        "SELECT COALESCE(SUM(payload_bytes),0) FROM (SELECT payload_sha256,MAX(payload_bytes) payload_bytes FROM segment_cache GROUP BY payload_sha256)"
                    ).fetchone()[0]
                )
                if count <= MAX_SEGMENT_ENTRIES and total <= MAX_TOTAL_PAYLOAD_BYTES:
                    return
                rows = connection.execute(
                    "SELECT cache_key,payload_sha256 FROM segment_cache ORDER BY access_ns,created_ns,cache_key LIMIT ?",
                    (MAINTENANCE_BATCH,),
                ).fetchall()
            for row in rows:
                digest = self._discard_metadata(str(row[0]))
                if digest is not None:
                    self._delete_digest_if_unreferenced(digest)
        except (OSError, sqlite3.Error, TypeError, ValueError):
            return

    def _prune_relational_metadata(self) -> None:
        try:
            with self._connection() as connection:
                connection.execute("BEGIN IMMEDIATE")
                excess = int(
                    connection.execute(
                        "SELECT MAX(count(*)-?,0) FROM runs", (MAX_RUN_ROWS,)
                    ).fetchone()[0]
                )
                limit = min(excess, MAINTENANCE_BATCH)
                if limit > 0:
                    run_rows = connection.execute(
                        "SELECT run_id FROM runs ORDER BY created_ns,run_id LIMIT ?",
                        (limit,),
                    ).fetchall()
                    for row in run_rows:
                        connection.execute(
                            "UPDATE segment_cache SET run_id=NULL WHERE run_id=?",
                            (row[0],),
                        )
                        connection.execute("DELETE FROM runs WHERE run_id=?", (row[0],))
                connection.execute(
                    "DELETE FROM books WHERE book_id IN (SELECT b.book_id FROM books b LEFT JOIN runs r ON r.book_id=b.book_id LEFT JOIN segment_cache s ON s.book_id=b.book_id WHERE r.run_id IS NULL AND s.cache_key IS NULL LIMIT ?)",
                    (MAINTENANCE_BATCH,),
                )
                connection.execute(
                    "DELETE FROM voices WHERE voice_id IN (SELECT v.voice_id FROM voices v LEFT JOIN runs r ON r.voice_id=v.voice_id LEFT JOIN segment_cache s ON s.voice_id=v.voice_id WHERE r.run_id IS NULL AND s.cache_key IS NULL LIMIT ?)",
                    (MAINTENANCE_BATCH,),
                )
                connection.commit()
        except (OSError, sqlite3.Error, TypeError, ValueError):
            return

    def _scan_orphans(self, limit: int, *, older_than: float | None) -> None:
        """Inspect only ``limit`` flat names and never follow or unlink unsafe nodes."""
        try:
            with self._payload_fd() as payload_fd:
                examined = 0
                with os.scandir(payload_fd) as entries:
                    for entry in entries:
                        if examined >= limit:
                            break
                        examined += 1
                        name = entry.name
                        if name.startswith(_TEMP_PREFIX):
                            try:
                                metadata = os.stat(
                                    name, dir_fd=payload_fd, follow_symlinks=False
                                )
                            except OSError:
                                continue
                            if older_than is None or metadata.st_mtime < older_than:
                                _unlink_safe_regular(payload_fd, name)
                        elif name.endswith(_PAYLOAD_SUFFIX):
                            digest = name[: -len(_PAYLOAD_SUFFIX)]
                            if _safe_hex(digest):
                                self._delete_digest_if_unreferenced(
                                    digest, older_than=older_than
                                )
        except OSError:
            return

    # Compatibility name retained for private acceptance tests.
    def _delete_unreferenced_payloads(
        self, limit: int, *, older_than: float | None = None
    ) -> None:
        self._scan_orphans(limit, older_than=older_than)

    def _bounded_orphan_cleanup(self) -> None:
        self._scan_orphans(
            ORPHAN_SCAN_LIMIT, older_than=time.time() - _ORPHAN_AGE_SECONDS
        )


def _safe_hex(value: str) -> bool:
    return (
        type(value) is str
        and len(value) == 64
        and all(character in _HEX for character in value)
    )


def _payload_name(digest: str) -> str:
    return f"{digest}{_PAYLOAD_SUFFIX}"


def _lock_name(digest: str) -> str:
    return f"{digest}{_PAYLOAD_SUFFIX}{_LOCK_SUFFIX}"


def _owned(metadata: os.stat_result) -> bool:
    getuid = getattr(os, "getuid", None)
    return getuid is None or metadata.st_uid == getuid()


def _file_id(metadata: os.stat_result) -> tuple[int, int]:
    return metadata.st_dev, metadata.st_ino


def _safe_regular(metadata: os.stat_result, *, mode: int | None = None) -> bool:
    return bool(
        stat.S_ISREG(metadata.st_mode)
        and metadata.st_nlink == 1
        and _owned(metadata)
        and (mode is None or stat.S_IMODE(metadata.st_mode) == mode)
    )


def _read_regular(
    directory_fd: int, name: str, expected_size: int, digest: str
) -> bytes | None:
    descriptor = -1
    try:
        descriptor = os.open(name, os.O_RDONLY | _FILE_NOFOLLOW, dir_fd=directory_fd)
        before = os.fstat(descriptor)
        named = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
        if (
            _file_id(before) != _file_id(named)
            or not _safe_regular(before, mode=0o600)
            or before.st_size != expected_size
        ):
            return None
        chunks: list[bytes] = []
        hasher = hashlib.sha256()
        size = 0
        while chunk := os.read(descriptor, _CHUNK_BYTES):
            size += len(chunk)
            if size > expected_size or size > _MAX_PAYLOAD_BYTES:
                return None
            hasher.update(chunk)
            chunks.append(chunk)
        after = os.fstat(descriptor)
        if (
            _file_id(before) != _file_id(after)
            or after.st_nlink != 1
            or size != expected_size
            or hasher.hexdigest() != digest
        ):
            return None
        return b"".join(chunks)
    except OSError:
        return None
    finally:
        if descriptor >= 0:
            os.close(descriptor)


def _unlink_safe_regular(
    directory_fd: int, name: str, *, identity: tuple[int, int] | None = None
) -> bool:
    try:
        metadata = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
        if not _safe_regular(metadata) or (
            identity is not None and _file_id(metadata) != identity
        ):
            return False
        os.unlink(name, dir_fd=directory_fd)
        return True
    except OSError:
        return False


def _row_matches(row: sqlite3.Row, segment: SpeechSegment, task: SynthesisTask) -> bool:
    try:
        return bool(
            row["schema_version"] == CACHE_SCHEMA_VERSION
            and row["audio_contract_version"] == AUDIO_CONTRACT_VERSION
            and row["segment_id"] == segment.id == task.segment_id
            and row["chapter_id_sha256"]
            == hashlib.sha256(segment.chapter_id.encode()).hexdigest()
            and row["ordinal"] == segment.ordinal
            and row["text_sha256"] == segment.content_hash
            and row["character_count"] == segment.character_count
            and row["sample_rate_hz"] == task.sample_rate_hz
            and row["channels"] == task.channels
            and row["payload_bytes"] == row["frame_count"] * row["channels"] * 2
            and row["payload_bytes"] <= task.max_output_bytes
            and row["duration_ms"] == row["frame_count"] * 1000 // row["sample_rate_hz"]
        )
    except (KeyError, TypeError, ZeroDivisionError):
        return False


def _audio_matches(audio: SynthesizedAudio, task: SynthesisTask) -> bool:
    return (
        type(audio) is SynthesizedAudio
        and type(audio.pcm_s16le) is bytes
        and audio.segment_id == task.segment_id
        and audio.chapter_id == task.chapter_id
        and audio.sample_rate_hz == task.sample_rate_hz
        and audio.channels == task.channels
        and type(audio.frame_count) is int
        and audio.frame_count > 0
        and type(audio.duration_ms) is int
        and audio.duration_ms == audio.frame_count * 1000 // audio.sample_rate_hz
        and len(audio.pcm_s16le) == audio.frame_count * audio.channels * 2
        and len(audio.pcm_s16le) <= task.max_output_bytes
        and len(audio.pcm_s16le) <= _MAX_PAYLOAD_BYTES
    )


def _voice_material(
    plan: ExecutionPlan, segment: SpeechSegment | None = None
) -> dict[str, object]:
    """Return the voice identity that renders one segment.

    Per segment rather than per plan: in a cast, two characters speaking the
    same words must not collide on one cache key.
    """
    voice = (
        plan.cast.voice_for(segment.speaker_id) if segment else plan.cast.narrator
    )
    return {
        "commercial_use_allowed": voice.commercial_use_allowed,
        "content_fingerprint": voice.content_fingerprint,
        "id_sha256": hashlib.sha256(voice.id.encode()).hexdigest(),
        "language_sha256": hashlib.sha256(voice.language.encode()).hexdigest(),
        "license_sha256": hashlib.sha256(voice.license_id.encode()).hexdigest(),
        "provenance_sha256": hashlib.sha256(voice.provenance.encode()).hexdigest(),
    }


def _engine_material(
    specification: EngineSpecification, voice_sha256: str = ""
) -> dict[str, object]:
    """Return engine identity, narrowed to the voice that renders one segment.

    The pocket config lists the whole cast. Keying every segment on all of it
    would make adding one character voice change the key of every segment in
    the book, including narration that did not change, so a re-cast would
    re-synthesize work already paid for. The rendering voice is carried by the
    separate "voice" key, which is where a cast difference belongs.
    """
    if specification.kind == "pocket":
        pocket = specification.pocket_config
        material = pocket.semantic_material() if pocket is not None else None
        voices = material.get("voices") if material is not None else None
        if voice_sha256 and isinstance(voices, tuple) and material is not None:
            material["voices"] = tuple(
                item
                for item in voices
                if isinstance(item, dict) and item.get("sha256") == voice_sha256
            )
        return {"kind": specification.kind, "semantic_config": material}
    config = specification.fake_config
    return {
        "kind": specification.kind,
        "invalid_field": config.invalid_field if config is not None else None,
        "test_mode": config.test_mode.value if config is not None else None,
    }


def _canonical_json(value: object) -> str:
    return json.dumps(value, ensure_ascii=True, separators=(",", ":"), sort_keys=True)


def _sha256_json(value: object) -> str:
    return hashlib.sha256(_canonical_json(value).encode()).hexdigest()


_SCHEMA = """
CREATE TABLE IF NOT EXISTS schema_metadata(name TEXT PRIMARY KEY,value TEXT NOT NULL) STRICT;
CREATE TABLE IF NOT EXISTS books(book_id TEXT PRIMARY KEY,source_sha256 TEXT NOT NULL,created_ns INTEGER NOT NULL) STRICT;
CREATE TABLE IF NOT EXISTS voices(voice_id TEXT PRIMARY KEY,content_fingerprint TEXT NOT NULL,metadata_json TEXT NOT NULL,created_ns INTEGER NOT NULL) STRICT;
CREATE TABLE IF NOT EXISTS runs(run_id TEXT PRIMARY KEY,book_id TEXT NOT NULL REFERENCES books(book_id),voice_id TEXT NOT NULL REFERENCES voices(voice_id),plan_fingerprint TEXT NOT NULL,created_ns INTEGER NOT NULL) STRICT;
CREATE TABLE IF NOT EXISTS segment_cache(
 cache_key TEXT PRIMARY KEY,schema_version TEXT NOT NULL,audio_contract_version TEXT NOT NULL,
 segment_id TEXT NOT NULL,chapter_id_sha256 TEXT NOT NULL,ordinal INTEGER NOT NULL,
 text_sha256 TEXT NOT NULL,character_count INTEGER NOT NULL,sample_rate_hz INTEGER NOT NULL,
 channels INTEGER NOT NULL,frame_count INTEGER NOT NULL,duration_ms INTEGER NOT NULL,
 payload_sha256 TEXT NOT NULL,payload_bytes INTEGER NOT NULL,
 book_id TEXT REFERENCES books(book_id),voice_id TEXT REFERENCES voices(voice_id),run_id TEXT REFERENCES runs(run_id),
 created_ns INTEGER NOT NULL,access_ns INTEGER NOT NULL
) STRICT;
CREATE INDEX IF NOT EXISTS segment_payload_index ON segment_cache(payload_sha256);
CREATE INDEX IF NOT EXISTS segment_lru_index ON segment_cache(access_ns,created_ns,cache_key);
CREATE INDEX IF NOT EXISTS runs_age_index ON runs(created_ns,run_id);
"""
