"""Tests for kenkui.server.worker.WorkerServer.

Coverage:
- _build_config() with single-voice and multi-voice fields
- _build_config() propagates _included_indices as a proper field
- _process_job() happy path: AudioBuilder success → complete_job()
- _process_job() AnnotatedChaptersCacheMissError → fail_job("CACHE_MISS:…")
- _process_job() generic exception → fail_job(error_message)
- Queue state management: add / start / complete / fail cycle
- get_server() / reset_server() singleton helpers
"""

from __future__ import annotations

import threading
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from kenkui.models import (
    AppConfig,
    ChapterPreset,
    ChapterSelection,
    JobConfig,
    JobStatus,
    NarrationMode,
    ProcessingConfig,
)
from kenkui.parsing import AnnotatedChaptersCacheMissError
from kenkui.server.worker import WorkerServer, get_server, reset_server


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_job(
    ebook_path: Path | None = None,
    narration_mode: NarrationMode = NarrationMode.SINGLE,
    speaker_voices: dict[str, str] | None = None,
    annotated_chapters_path: Path | None = None,
    included: list[int] | None = None,
) -> JobConfig:
    ep = ebook_path or Path("/tmp/book.epub")
    cs = ChapterSelection(preset=ChapterPreset.CONTENT_ONLY, included=included or [])
    return JobConfig(
        ebook_path=ep,
        voice="alba",
        chapter_selection=cs,
        narration_mode=narration_mode,
        speaker_voices=speaker_voices or {},
        annotated_chapters_path=annotated_chapters_path,
    )


def _make_server(tmp_path: Path) -> WorkerServer:
    """Create a fresh WorkerServer that stores state in tmp_path."""
    with patch("kenkui.server.worker.QUEUE_FILE", tmp_path / "queue.yaml"):
        server = WorkerServer()
    return server


# ---------------------------------------------------------------------------
# Queue state management
# ---------------------------------------------------------------------------


class TestWorkerServerQueue:
    def test_add_job_returns_queue_item(self, tmp_path):
        server = _make_server(tmp_path)
        job = _make_job()
        item = server.add_job(job)
        assert item.id
        assert item.status == JobStatus.PENDING

    def test_get_job_by_id(self, tmp_path):
        server = _make_server(tmp_path)
        job = _make_job()
        item = server.add_job(job)
        fetched = server.get_job(item.id)
        assert fetched is not None
        assert fetched.id == item.id

    def test_get_job_unknown_id_returns_none(self, tmp_path):
        server = _make_server(tmp_path)
        assert server.get_job("nonexistent") is None

    def test_start_next_job_sets_processing(self, tmp_path):
        server = _make_server(tmp_path)
        job = _make_job()
        item = server.add_job(job)
        started = server.start_next_job()
        assert started is not None
        assert started.id == item.id
        assert started.status == JobStatus.PROCESSING

    def test_complete_job(self, tmp_path):
        server = _make_server(tmp_path)
        item = server.add_job(_make_job())
        server.start_next_job()
        server.complete_job(item.id, output_path="/out/book.m4b")
        done = server.get_job(item.id)
        assert done is not None
        assert done.status == JobStatus.COMPLETED
        assert done.output_path == "/out/book.m4b"
        assert done.progress == 100.0

    def test_fail_job(self, tmp_path):
        server = _make_server(tmp_path)
        item = server.add_job(_make_job())
        server.start_next_job()
        server.fail_job(item.id, "something went wrong")
        failed = server.get_job(item.id)
        assert failed is not None
        assert failed.status == JobStatus.FAILED
        assert failed.error_message == "something went wrong"

    def test_remove_pending_job(self, tmp_path):
        server = _make_server(tmp_path)
        item = server.add_job(_make_job())
        assert server.remove_job(item.id) is True
        assert server.get_job(item.id) is None

    def test_cannot_remove_processing_job(self, tmp_path):
        server = _make_server(tmp_path)
        item = server.add_job(_make_job())
        server.start_next_job()
        assert server.remove_job(item.id) is False
        assert server.get_job(item.id) is not None

    def test_pending_items_property(self, tmp_path):
        server = _make_server(tmp_path)
        server.add_job(_make_job())
        server.add_job(_make_job())
        assert len(server.pending_items) == 2

    def test_clear_all_jobs(self, tmp_path):
        server = _make_server(tmp_path)
        server.add_job(_make_job())
        server.add_job(_make_job())
        server.clear_all_jobs()
        assert server.all_items == []


# ---------------------------------------------------------------------------
# _build_config — ProcessingConfig construction
# ---------------------------------------------------------------------------


class TestBuildConfig:
    def _build(
        self,
        tmp_path: Path,
        job: JobConfig | None = None,
        app_config: AppConfig | None = None,
    ) -> ProcessingConfig:
        server = _make_server(tmp_path)
        if app_config:
            server._app_config = app_config
        j = job or _make_job()
        return server._build_config(j)

    def test_basic_fields_populated(self, tmp_path):
        cfg = self._build(tmp_path)
        assert isinstance(cfg.voice, str)
        assert isinstance(cfg.ebook_path, Path)

    def test_multi_voice_fields_passed_through(self, tmp_path):
        cache_path = tmp_path / "cache.json"
        job = _make_job(
            narration_mode=NarrationMode.MULTI,
            speaker_voices={"ALICE-1": "alba", "NARRATOR": "cosette"},
            annotated_chapters_path=cache_path,
            included=[0, 2, 4],
        )
        cfg = self._build(tmp_path, job=job)
        assert cfg.speaker_voices == {"ALICE-1": "alba", "NARRATOR": "cosette"}
        assert cfg.annotated_chapters_path == cache_path

    def test_included_indices_propagated_as_field(self, tmp_path):
        """_included_indices must be a proper field (not a runtime attribute)."""
        job = _make_job(included=[1, 3, 5])
        cfg = self._build(tmp_path, job=job)
        # Access via the dataclass field — no getattr fallback needed
        assert cfg._included_indices == [1, 3, 5]

    def test_empty_included_means_all_chapters(self, tmp_path):
        """An empty included list should propagate as an empty _included_indices."""
        job = _make_job(included=[])
        cfg = self._build(tmp_path, job=job)
        assert cfg._included_indices == []

    def test_manual_preset_uses_index_operations(self, tmp_path):
        """MANUAL preset should produce index-based filter operations."""
        cs = ChapterSelection(preset=ChapterPreset.MANUAL, included=[2, 5])
        job = JobConfig(ebook_path=Path("/tmp/book.epub"), voice="alba", chapter_selection=cs)
        cfg = self._build(tmp_path, job=job)
        ops_values = [op.value for op in cfg.chapter_filters]
        assert "2" in ops_values
        assert "5" in ops_values

    def test_app_config_fields_respected(self, tmp_path):
        """AppConfig settings (workers, bitrate, pauses) flow into ProcessingConfig."""
        app_cfg = AppConfig(workers=4, m4b_bitrate="128k", pause_line_ms=200)
        cfg = self._build(tmp_path, app_config=app_cfg)
        assert cfg.workers == 4
        assert cfg.m4b_bitrate == "128k"
        assert cfg.pause_line_ms == 200


# ---------------------------------------------------------------------------
# _process_job — happy path and error cases
# ---------------------------------------------------------------------------


class TestProcessJob:
    def _run_process_job(
        self,
        tmp_path: Path,
        builder_result: bool = True,
        builder_side_effect: Exception | None = None,
    ) -> tuple[WorkerServer, object]:
        """Add a job, call _process_job with a mocked AudioBuilder, return server + item."""
        server = _make_server(tmp_path)
        item = server.add_job(_make_job(ebook_path=tmp_path / "book.epub"))
        item.status = JobStatus.PROCESSING  # manually mark as started

        mock_builder = MagicMock()
        if builder_side_effect is not None:
            mock_builder.return_value.run.side_effect = builder_side_effect
        else:
            mock_builder.return_value.run.return_value = builder_result

        with patch("kenkui.server.worker.AudioBuilder", mock_builder):
            server._process_job(item)

        return server, item

    def test_successful_job_marks_completed(self, tmp_path):
        server, item = self._run_process_job(tmp_path, builder_result=True)
        final = server.get_job(item.id)
        assert final is not None
        assert final.status == JobStatus.COMPLETED

    def test_failed_builder_run_marks_failed(self, tmp_path):
        """AudioBuilder.run() returning False marks the job as FAILED."""
        server, item = self._run_process_job(tmp_path, builder_result=False)
        final = server.get_job(item.id)
        assert final is not None
        assert final.status == JobStatus.FAILED
        assert "Conversion failed" in final.error_message

    def test_generic_exception_marks_failed(self, tmp_path):
        """An unexpected exception from AudioBuilder marks the job as FAILED."""
        server, item = self._run_process_job(tmp_path, builder_side_effect=RuntimeError("boom"))
        final = server.get_job(item.id)
        assert final is not None
        assert final.status == JobStatus.FAILED
        assert "boom" in final.error_message

    def test_cache_miss_error_includes_sentinel(self, tmp_path):
        """AnnotatedChaptersCacheMissError must produce a 'CACHE_MISS:' error message
        so the UI's QueueScreen._handle_cache_miss() recovery flow triggers."""
        server, item = self._run_process_job(
            tmp_path,
            builder_side_effect=AnnotatedChaptersCacheMissError("missing file"),
        )
        final = server.get_job(item.id)
        assert final is not None
        assert final.status == JobStatus.FAILED
        assert final.error_message.startswith("CACHE_MISS:")

    def test_progress_callback_called_during_processing(self, tmp_path):
        """The progress_callback passed to start_processing is invoked."""
        server = _make_server(tmp_path)
        server.add_job(_make_job(ebook_path=tmp_path / "book.epub"))

        progress_calls: list[tuple] = []

        def mock_run():
            # Simulate builder calling the progress callback
            if server._progress_callback:
                server._progress_callback(50.0, "Chapter 1", 30)
            return True

        mock_builder = MagicMock()
        mock_builder.return_value.run.side_effect = mock_run

        with patch("kenkui.server.worker.AudioBuilder", mock_builder):
            server.start_processing(
                progress_callback=lambda p, c, e: progress_calls.append((p, c, e))
            )
            # Wait briefly for the background thread to process
            server._processing_thread.join(timeout=5)

        assert any(p == 50.0 for p, c, e in progress_calls)


# ---------------------------------------------------------------------------
# Singleton helpers
# ---------------------------------------------------------------------------


class TestServerSingleton:
    def test_get_server_returns_same_instance(self, tmp_path):
        with patch("kenkui.server.worker.QUEUE_FILE", tmp_path / "queue.yaml"):
            reset_server()
            s1 = get_server()
            s2 = get_server()
            assert s1 is s2
            reset_server()

    def test_reset_server_creates_fresh_instance(self, tmp_path):
        with patch("kenkui.server.worker.QUEUE_FILE", tmp_path / "queue.yaml"):
            reset_server()
            s1 = get_server()
            reset_server()
            s2 = get_server()
            assert s1 is not s2
            reset_server()
