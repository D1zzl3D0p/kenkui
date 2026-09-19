"""WP8 private SQLite segment-cache acceptance tests."""
# ruff: noqa: ARG001, BLE001, D103, E501, EM101, PLR2004, PT018, SLF001, TC003, TRY003

from __future__ import annotations

import hashlib
import json
import os
import pickle
import sqlite3
import stat
import threading
import time
from dataclasses import replace
from pathlib import Path
from typing import NoReturn

import pytest

import kenkui as kk
import kenkui._execution.cache as cache_module
from kenkui._audio.m4b import FakeArtifactAssembler
from kenkui._domain.planning import (
    NORMALIZATION_SCHEMA_VERSION,
    CastPlan,
    ExecutionPlan,
    SpeechSegment,
    compile_execution_plan,
)
from kenkui._execution.cache import (
    AUDIO_CONTRACT_VERSION,
    CACHE_SCHEMA_VERSION,
    CacheStore,
)
from kenkui._execution.coordinator import ExecutionBindings
from kenkui._execution.process_pool import (
    EngineSpecification,
    FakeEngineConfig,
    WorkerTestMode,
)
from kenkui._tts.fake import FAKE_CHANNELS, FAKE_SAMPLE_RATE_HZ, DeterministicFakeEngine
from kenkui._tts.protocols import SynthesisTask, SynthesizedAudio
from kenkui.limits import MAX_SEGMENT_PCM_BYTES
from test_execution import _pipeline, _voice


def _material(
    tmp_path: Path,
) -> tuple[kk.Pipeline, ExecutionPlan, SpeechSegment, SynthesisTask, SynthesizedAudio]:
    tmp_path.mkdir(parents=True, exist_ok=True)
    pipeline, source, _ = _pipeline(tmp_path)
    plan = compile_execution_plan(
        pipeline,
        pipeline.inspect(),
        source_bytes_hash=__import__("hashlib").sha256(source.read_bytes()).hexdigest(),
        resolved_voice=_voice(),
        model_revision="fake-v1",
    )
    segment = plan.segments[0]
    task = SynthesisTask(
        segment.id,
        segment.chapter_id,
        segment.text,
        FAKE_SAMPLE_RATE_HZ,
        FAKE_CHANNELS,
        MAX_SEGMENT_PCM_BYTES,
    )
    return pipeline, plan, segment, task, DeterministicFakeEngine().synthesize(task)


def _legacy_plain_segment_id(segment: SpeechSegment) -> str:
    """Reproduce the retired tts-chunks-v4 ID for a first chapter segment."""
    assert segment.ordinal == 0
    chapter_id = {
        "characters": len(segment.chapter_id),
        "sha256": hashlib.sha256(segment.chapter_id.encode()).hexdigest(),
    }
    material = {
        "chapter_id": chapter_id,
        "chunk_index": 0,
        "chunking_schema": "tts-chunks-v4",
        "content_hash": segment.content_hash,
        "normalization": NORMALIZATION_SCHEMA_VERSION,
        "ordinal": segment.ordinal,
        "segment_id_version": "v2",
    }
    identity = json.dumps(material, sort_keys=True, separators=(",", ":"))
    digest = hashlib.sha256(identity.encode()).hexdigest()[:24]
    return f"seg-{NORMALIZATION_SCHEMA_VERSION}-v2-{digest}"


def test_cache_cold_warm_public_equivalence_and_no_warm_spawn(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pipeline, _, _ = _pipeline(tmp_path)
    cache = CacheStore(tmp_path / "private-cache")
    bindings = ExecutionBindings(
        EngineSpecification.fake(), FakeArtifactAssembler(), _voice(), "fake-v1", cache
    )
    monkeypatch.setattr(
        "kenkui._resolution._execution_bindings", lambda _voice_id, **_cast: bindings
    )
    first_events: list[kk.ExecutionEvent] = []
    first = pipeline.write_m4b(
        tmp_path / "cold.m4b",
        workers=2,
        keep_audio_cache=True,
        on_event=first_events.append,
    )
    cold_bytes = (tmp_path / "cold.m4b").read_bytes()

    def forbidden(*args: object, **kwargs: object) -> NoReturn:
        raise AssertionError("warm cache scheduled a worker")

    monkeypatch.setattr("kenkui._execution.coordinator.render_spawned", forbidden)
    second_events: list[kk.ExecutionEvent] = []
    second = pipeline.write_m4b(
        tmp_path / "warm.m4b",
        workers=1,
        keep_audio_cache=True,
        on_event=second_events.append,
    )

    assert first.stats == second.stats
    assert cold_bytes == (tmp_path / "warm.m4b").read_bytes()
    assert first_events == second_events
    with sqlite3.connect(tmp_path / "private-cache" / "cache.sqlite3") as connection:
        assert {
            row[0]
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            )
        } >= {"schema_metadata", "books", "voices", "runs", "segment_cache"}
        assert (
            connection.execute("SELECT count(*) FROM segment_cache").fetchone()[0] == 2
        )


def test_semantic_key_sensitivity_and_shell_insensitivity(tmp_path: Path) -> None:
    _, plan, segment, task, _ = _material(tmp_path)
    store = CacheStore(tmp_path / "cache")
    spec = EngineSpecification.fake()
    baseline = store.key_for(plan, segment, task, spec)
    # Worker count, output, callback, run/cache locations are not key inputs at all.
    assert (
        CacheStore(tmp_path / "elsewhere").key_for(plan, segment, task, spec)
        == baseline
    )
    assert (
        store.key_for(
            plan,
            segment,
            replace(task, max_output_bytes=task.max_output_bytes // 2),
            spec,
        )
        == baseline
    )
    assert (
        store.key_for(plan, replace(segment, content_hash="b" * 64), task, spec)
        != baseline
    )
    assert (
        store.key_for(replace(plan, model_revision="fake-v2"), segment, task, spec)
        != baseline
    )
    changed_voice = replace(plan.cast.narrator, content_fingerprint="c" * 64)
    assert (
        store.key_for(
            replace(plan, cast=CastPlan.single(changed_voice)), segment, task, spec
        )
        != baseline
    )
    assert (
        store.key_for(plan, segment, replace(task, sample_rate_hz=8_000), spec)
        != baseline
    )
    bad_spec = EngineSpecification.fake(
        FakeEngineConfig(test_mode=WorkerTestMode.INVALID_AUDIO)
    )
    assert store.key_for(plan, segment, task, bad_spec) != baseline
    assert CACHE_SCHEMA_VERSION and AUDIO_CONTRACT_VERSION


def test_grid_v1_misses_but_preserves_a_legacy_cache_entry(tmp_path: Path) -> None:
    """The new namespace cannot reuse old PCM and never deletes it implicitly."""
    _, plan, segment, task, audio = _material(tmp_path)
    store = CacheStore(tmp_path / "cache")
    specification = EngineSpecification.fake()
    legacy_id = _legacy_plain_segment_id(segment)
    assert legacy_id != segment.id
    legacy_segment = replace(segment, id=legacy_id)
    legacy_task = replace(task, segment_id=legacy_id)
    legacy_audio = replace(audio, segment_id=legacy_id)
    legacy_key = store.key_for(plan, legacy_segment, legacy_task, specification)
    current_key = store.key_for(plan, segment, task, specification)
    assert current_key != legacy_key

    store.store(legacy_key, legacy_segment, legacy_task, legacy_audio, None)
    assert store.lookup(current_key, segment, task) is None
    assert store.lookup(legacy_key, legacy_segment, legacy_task) == legacy_audio
    with sqlite3.connect(tmp_path / "cache" / "cache.sqlite3") as connection:
        assert connection.execute(
            "SELECT segment_id FROM segment_cache WHERE cache_key=?", (legacy_key,)
        ).fetchone() == (legacy_id,)


def test_payload_missing_corrupt_and_bad_metadata_recover_as_misses(
    tmp_path: Path,
) -> None:
    _, plan, segment, task, audio = _material(tmp_path)
    store = CacheStore(tmp_path / "cache")
    context = store.prepare_run(plan)
    key = store.key_for(plan, segment, task, EngineSpecification.fake())
    store.store(key, segment, task, audio, context)
    assert store.lookup(key, segment, task) == audio

    database = tmp_path / "cache" / "cache.sqlite3"
    with sqlite3.connect(database) as connection:
        digest = connection.execute(
            "SELECT payload_sha256 FROM segment_cache WHERE cache_key=?", (key,)
        ).fetchone()[0]
    payload = tmp_path / "cache" / "payloads" / f"{digest}.pcm"
    payload.write_bytes(b"truncated")
    assert store.lookup(key, segment, task) is None
    with sqlite3.connect(database) as connection:
        assert (
            connection.execute("SELECT count(*) FROM segment_cache").fetchone()[0] == 0
        )

    store.store(key, segment, task, audio, context)
    with sqlite3.connect(database) as connection:
        connection.execute(
            "UPDATE segment_cache SET schema_version='unknown-row-v2' WHERE cache_key=?",
            (key,),
        )
    assert store.lookup(key, segment, task) is None
    with sqlite3.connect(database) as connection:
        assert (
            connection.execute("SELECT count(*) FROM segment_cache").fetchone()[0] == 0
        )

    store.store(key, segment, task, audio, context)
    with sqlite3.connect(database) as connection:
        connection.execute(
            "UPDATE segment_cache SET duration_ms=duration_ms+1 WHERE cache_key=?",
            (key,),
        )
    assert store.lookup(key, segment, task) is None
    payload.unlink(missing_ok=True)
    store.store(key, segment, task, audio, context)
    payload.unlink()
    assert store.lookup(key, segment, task) is None


def test_private_delete_clear_and_schema_version_invalidation(tmp_path: Path) -> None:
    _, plan, segment, task, audio = _material(tmp_path)
    root = tmp_path / "cache"
    store = CacheStore(root)
    context = store.prepare_run(plan)
    key = store.key_for(plan, segment, task, EngineSpecification.fake())
    store.store(key, segment, task, audio, context)
    store._delete(key)
    assert store.lookup(key, segment, task) is None
    store.store(key, segment, task, audio, context)
    store._clear()
    assert store.lookup(key, segment, task) is None

    with sqlite3.connect(root / "cache.sqlite3") as connection:
        connection.execute(
            "UPDATE schema_metadata SET value='future-v99' WHERE name='schema_version'"
        )
    invalid = CacheStore(root)
    assert invalid.prepare_run(plan) is None
    assert invalid.lookup(key, segment, task) is None


def test_interrupted_files_concurrent_population_and_pickle_contract(
    tmp_path: Path,
) -> None:
    _, plan, segment, task, audio = _material(tmp_path)
    root = tmp_path / "cache"
    store = CacheStore(root)
    context = store.prepare_run(plan)
    key = store.key_for(plan, segment, task, EngineSpecification.fake())
    errors: list[BaseException] = []

    def populate() -> None:
        try:
            for _ in range(3):
                CacheStore(root).store(key, segment, task, audio, context)
        except (
            BaseException
        ) as error:  # pragma: no cover - assertion reports thread failures
            errors.append(error)

    threads = [threading.Thread(target=populate) for _ in range(4)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert not errors
    assert store.lookup(key, segment, task) == audio
    assert b"sqlite3.Connection" not in pickle.dumps(store)
    assert not any(
        isinstance(value, sqlite3.Connection) for value in store.__getstate__().values()
    )

    orphan_dir = root / "payloads"
    temporary = orphan_dir / ".tmp-interrupted"
    temporary.write_bytes(b"partial")
    old = time.time() - 100_000
    os.utime(temporary, (old, old))
    CacheStore(root)
    assert not temporary.exists()


def test_database_lock_and_unusable_location_degrade_without_render_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Exercise more failed publications than one orphan scan can cover without
    # repeating real SQLite lock timeouts for the production-sized scan budget.
    monkeypatch.setattr(cache_module, "ORPHAN_SCAN_LIMIT", 4)
    pipeline, _, segment, task, audio = _material(tmp_path)
    root = tmp_path / "cache"
    store = CacheStore(root)

    # A payload which predates the lock may already be shared by committed rows.
    existing_key = __import__("hashlib").sha256(b"existing-key").hexdigest()
    existing_shared_key = (
        __import__("hashlib").sha256(b"existing-shared-key").hexdigest()
    )
    store.store(existing_key, segment, task, audio, None)
    store.store(existing_shared_key, segment, task, audio, None)
    existing_digest = __import__("hashlib").sha256(audio.pcm_s16le).hexdigest()
    existing_payload = store._payload_path(existing_digest)
    assert existing_payload.exists()

    payloads_before = {path.name for path in (root / "payloads").glob("*.pcm")}
    lock = sqlite3.connect(root / "cache.sqlite3", timeout=0)
    lock.execute("BEGIN EXCLUSIVE")
    try:
        bindings = ExecutionBindings(
            EngineSpecification.fake(),
            FakeArtifactAssembler(),
            _voice(),
            "fake-v1",
            store,
        )
        monkeypatch.setattr(
            "kenkui._resolution._execution_bindings",
            lambda _voice_id, **_cast: bindings,
        )
        result = pipeline.write_m4b(tmp_path / "locked.m4b")
        assert result.stats.synthesized_segments == 2

        for index in range(cache_module.ORPHAN_SCAN_LIMIT * 2):
            changed = index.to_bytes(2, "little") + audio.pcm_s16le[2:]
            changed_audio = replace(audio, pcm_s16le=changed)
            key = __import__("hashlib").sha256(f"locked-{index}".encode()).hexdigest()
            store.store(key, segment, task, changed_audio, None)

        # A later failed metadata store of the already-valid digest must not
        # remove the payload belonging to the earlier committed row.
        shared_key = __import__("hashlib").sha256(b"shared-key").hexdigest()
        store.store(shared_key, segment, task, audio, None)
        assert existing_payload.read_bytes() == audio.pcm_s16le
        assert {
            path.name for path in (root / "payloads").glob("*.pcm")
        } == payloads_before
    finally:
        lock.rollback()
        lock.close()

    with sqlite3.connect(root / "cache.sqlite3") as connection:
        assert {
            row[0] for row in connection.execute("SELECT cache_key FROM segment_cache")
        } == {existing_key, existing_shared_key}

    blocked = tmp_path / "not-a-directory"
    blocked.write_bytes(b"x")
    disabled = CacheStore(blocked)
    other_plan = _material(tmp_path / "other")[1]
    assert disabled.prepare_run(other_plan) is None
    disabled._delete("absent")
    disabled._clear()


def test_private_cache_defensive_edge_paths(tmp_path: Path) -> None:
    """Malformed and interrupted private state always fails open."""
    _, plan, segment, task, audio = _material(tmp_path)
    root = tmp_path / "cache"
    store = CacheStore(root)
    key = store.key_for(plan, segment, task, EngineSpecification.fake())
    context = store.prepare_run(plan)

    store.store(key, segment, task, replace(audio, chapter_id="wrong"), context)
    assert store.lookup(key, segment, task) is None
    assert store._read_payload("bad", 1) is None
    assert store._read_payload("0" * 64, 0) is None
    assert not store._publish_payload("0" * 64, b"")

    payload = audio.pcm_s16le
    digest = __import__("hashlib").sha256(payload).hexdigest()
    final = store._payload_path(digest)
    lock_path = final.with_suffix(final.suffix + ".lock")
    lock_path.write_bytes(b"occupied")
    try:
        assert not store._publish_payload(digest, payload)
    finally:
        lock_path.unlink(missing_ok=True)

    orphan = root / "payloads" / f"{'f' * 64}.pcm"
    orphan.write_bytes(b"orphan")
    old = time.time() - 100_000
    os.utime(orphan, (old, old))
    CacheStore(root)
    assert not orphan.exists()


def test_cancellation_with_ordered_hit_then_miss_never_publishes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pipeline, plan, _, _, _ = _material(tmp_path)
    cache = CacheStore(tmp_path / "cache")
    bindings = ExecutionBindings(
        EngineSpecification.fake(), FakeArtifactAssembler(), _voice(), "fake-v1", cache
    )
    monkeypatch.setattr(
        "kenkui._resolution._execution_bindings", lambda _voice_id, **_cast: bindings
    )
    pipeline.write_m4b(tmp_path / "primed.m4b")

    second = plan.segments[1]
    second_task = SynthesisTask(
        second.id,
        second.chapter_id,
        second.text,
        FAKE_SAMPLE_RATE_HZ,
        FAKE_CHANNELS,
        MAX_SEGMENT_PCM_BYTES,
    )
    second_key = cache.key_for(plan, second, second_task, EngineSpecification.fake())
    cache._delete(second_key)
    token = kk.CancellationToken()

    def cancel_after_ordered_hit(event: kk.ExecutionEvent) -> None:
        if isinstance(event, kk.StageProgress) and event.stage == "render":
            token.cancel()

    output = tmp_path / "cancelled-mixed.m4b"
    with pytest.raises(kk.CancelledError):
        pipeline.write_m4b(output, cancel=token, on_event=cancel_after_ordered_hit)
    assert not output.exists()


def test_root_and_payload_directory_symlinks_never_touch_targets(
    tmp_path: Path,
) -> None:
    external_root = tmp_path / "external-root"
    external_root.mkdir(mode=0o755)
    marker = external_root / "marker"
    marker.write_bytes(b"outside")
    root_link = tmp_path / "root-link"
    root_link.symlink_to(external_root, target_is_directory=True)
    before_mode = stat.S_IMODE(external_root.stat().st_mode)
    disabled = CacheStore(root_link)
    assert not disabled._enabled
    assert marker.read_bytes() == b"outside"
    assert stat.S_IMODE(external_root.stat().st_mode) == before_mode
    assert not (external_root / "cache.sqlite3").exists()

    root = tmp_path / "cache-with-poisoned-payloads"
    root.mkdir(mode=0o700)
    external_payloads = tmp_path / "external-payloads"
    external_payloads.mkdir(mode=0o755)
    outside = external_payloads / "outside"
    outside.write_bytes(b"unchanged")
    (root / "payloads").symlink_to(external_payloads, target_is_directory=True)
    payload_mode = stat.S_IMODE(external_payloads.stat().st_mode)
    disabled_payloads = CacheStore(root)
    assert not disabled_payloads._enabled
    assert outside.read_bytes() == b"unchanged"
    assert stat.S_IMODE(external_payloads.stat().st_mode) == payload_mode
    assert not (root / "cache.sqlite3").exists()


def test_hardlinked_database_is_rejected_without_chmod_or_write(tmp_path: Path) -> None:
    external = tmp_path / "external.sqlite3"
    with sqlite3.connect(external) as connection:
        connection.execute("CREATE TABLE sentinel(value BLOB)")
        connection.execute("INSERT INTO sentinel VALUES(?)", (b"outside",))
    external.chmod(0o640)
    root = tmp_path / "cache"
    root.mkdir(mode=0o700)
    (root / "payloads").mkdir(mode=0o700)
    os.link(external, root / "cache.sqlite3")
    before = external.read_bytes()
    before_mode = stat.S_IMODE(external.stat().st_mode)
    store = CacheStore(root)
    assert not store._enabled
    assert external.read_bytes() == before
    assert stat.S_IMODE(external.stat().st_mode) == before_mode
    assert external.stat().st_nlink == 2


def test_symlinked_database_is_rejected_without_touching_target(tmp_path: Path) -> None:
    external = tmp_path / "external-db"
    external.write_bytes(b"not sqlite and must remain unchanged")
    external.chmod(0o640)
    root = tmp_path / "cache-db-link"
    root.mkdir(mode=0o700)
    (root / "payloads").mkdir(mode=0o700)
    (root / "cache.sqlite3").symlink_to(external)
    before_mode = stat.S_IMODE(external.stat().st_mode)
    store = CacheStore(root)
    assert not store._enabled
    assert external.read_bytes() == b"not sqlite and must remain unchanged"
    assert stat.S_IMODE(external.stat().st_mode) == before_mode


@pytest.mark.parametrize("attack", ["symlink", "hardlink"])
def test_payload_links_are_misses_and_external_files_are_untouched(
    tmp_path: Path, attack: str
) -> None:
    _, plan, segment, task, audio = _material(tmp_path)
    root = tmp_path / "cache"
    store = CacheStore(root)
    key = store.key_for(plan, segment, task, EngineSpecification.fake())
    store.store(key, segment, task, audio, store.prepare_run(plan))
    digest = __import__("hashlib").sha256(audio.pcm_s16le).hexdigest()
    payload = store._payload_path(digest)
    payload.unlink()
    external = tmp_path / f"external-{attack}"
    external.write_bytes(audio.pcm_s16le)
    external.chmod(0o640)
    if attack == "symlink":
        payload.symlink_to(external)
    else:
        os.link(external, payload)
    before = external.read_bytes()
    before_mode = stat.S_IMODE(external.stat().st_mode)
    assert store.lookup(key, segment, task) is None
    store._delete_unreferenced_payloads(cache_module.ORPHAN_SCAN_LIMIT)
    assert external.read_bytes() == before
    assert stat.S_IMODE(external.stat().st_mode) == before_mode
    assert payload.is_symlink() if attack == "symlink" else payload.exists()


def test_cleanup_rechecks_reference_after_concurrent_store_commit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _, plan, segment, task, audio = _material(tmp_path)
    root = tmp_path / "cache"
    store = CacheStore(root)
    first_key = store.key_for(plan, segment, task, EngineSpecification.fake())
    context = store.prepare_run(plan)
    store.store(first_key, segment, task, audio, context)
    with sqlite3.connect(root / "cache.sqlite3") as connection:
        connection.execute("DELETE FROM segment_cache")
    selected = threading.Event()
    committed = threading.Event()
    original = CacheStore._delete_digest_if_unreferenced

    def paused_delete(
        self: CacheStore, digest: str, *, older_than: float | None = None
    ) -> None:
        if not selected.is_set():
            selected.set()
            assert committed.wait(5)
        original(self, digest, older_than=older_than)

    monkeypatch.setattr(CacheStore, "_delete_digest_if_unreferenced", paused_delete)
    cleanup = threading.Thread(target=store._delete_unreferenced_payloads, args=(1,))
    cleanup.start()
    assert selected.wait(5)
    second_key = __import__("hashlib").sha256(b"concurrent-key").hexdigest()
    store.store(second_key, segment, task, audio, context)
    committed.set()
    cleanup.join(5)
    assert not cleanup.is_alive()
    assert store.lookup(second_key, segment, task) == audio


def test_run_rows_stay_bounded_and_audio_is_never_evicted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Runs keep a finite bound; rendered audio retention is per-book only."""
    _, plan, segment, task, audio = _material(tmp_path)
    monkeypatch.setattr(cache_module, "MAX_RUN_ROWS", 2)
    root = tmp_path / "cache"
    store = CacheStore(root)
    for index in range(6):
        context = store.prepare_run(plan)
        changed = bytes([index + 1]) + audio.pcm_s16le[1:]
        changed_audio = replace(audio, pcm_s16le=changed)
        key = __import__("hashlib").sha256(f"key-{index}".encode()).hexdigest()
        store.store(key, segment, task, changed_audio, context)
    with sqlite3.connect(root / "cache.sqlite3") as connection:
        assert (
            connection.execute("SELECT count(*) FROM segment_cache").fetchone()[0] == 6
        )
        assert connection.execute("SELECT count(*) FROM runs").fetchone()[0] <= 2


def test_poisoned_database_maintenance_evicts_nothing(
    tmp_path: Path,
) -> None:
    """A database full of junk rows is left alone, not pruned behind the user."""
    _, plan, segment, task, audio = _material(tmp_path)
    root = tmp_path / "cache"
    store = CacheStore(root)
    key = store.key_for(plan, segment, task, EngineSpecification.fake())
    store.store(key, segment, task, audio, store.prepare_run(plan))
    with sqlite3.connect(root / "cache.sqlite3") as connection:
        row = connection.execute(
            "SELECT * FROM segment_cache WHERE cache_key=?", (key,)
        ).fetchone()
        columns = [
            item[1] for item in connection.execute("PRAGMA table_info(segment_cache)")
        ]
        placeholders = ",".join("?" for _ in columns)
        for index in range(100):
            values = list(row)
            values[0] = (
                __import__("hashlib").sha256(f"poison-{index}".encode()).hexdigest()
            )
            connection.execute(
                f"INSERT INTO segment_cache VALUES({placeholders})",
                values,
            )

    store._maintenance()

    with sqlite3.connect(root / "cache.sqlite3") as connection:
        assert (
            connection.execute("SELECT count(*) FROM segment_cache").fetchone()[0]
            == 101
        )
