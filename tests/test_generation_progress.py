from __future__ import annotations

import os
import queue
from pathlib import Path
from types import SimpleNamespace

import pytest


def _processing_config(tmp_path: Path):
    from kenkui.models import ProcessingConfig

    return ProcessingConfig(
        voice="alba",
        ebook_path=tmp_path / "book.epub",
        output_path=tmp_path,
        pause_line_ms=0,
        pause_chapter_ms=0,
        workers=1,
        m4b_bitrate="96k",
        keep_temp=True,
        debug_html=False,
        chapter_filters=[],
        tts_provider="kokoro",
        tts_model="model-a",
    )


def test_progress_event_is_public_generation_callback_payload():
    import kenkui

    chapter = kenkui.ChapterProgress(
        index=2,
        title="Chapter 2",
        completed_units=10,
        total_units=100,
        status="advanced",
    )
    event = kenkui.ProgressEvent(
        stage="tts_synthesis",
        status="advanced",
        message="Chapter 1",
        completed_units=100,
        total_units=1000,
        unit="chars",
        active_chapters=(chapter,),
    )

    assert event.stage == "tts_synthesis"
    assert event.unit == "chars"
    assert event.active_chapters == (chapter,)
    assert isinstance(event.timestamp, float)


def test_audio_builder_emits_progress_event_with_generation_context(tmp_path):
    from kenkui.parsing import AudioBuilder
    from kenkui.progress import ProgressEvent

    events: list[ProgressEvent] = []
    builder = AudioBuilder(_processing_config(tmp_path), progress_callback=events.append)
    builder._book_hash = "book-hash"

    builder._emit_progress(
        "tts_synthesis",
        "advanced",
        "Chapter 1",
        completed_units=200,
        total_units=1000,
        unit="chars",
    )

    assert events == [
        ProgressEvent(
            stage="tts_synthesis",
            status="advanced",
            message="Chapter 1",
            completed_units=200,
            total_units=1000,
            unit="chars",
            timestamp=events[0].timestamp,
            book_hash="book-hash",
            provider="kokoro",
            model="model-a",
            active_chapters=(),
        )
    ]


def test_audio_builder_totals_duplicate_chapter_titles_by_position(tmp_path, monkeypatch):
    from kenkui.models import Chapter
    from kenkui.parsing import AudioBuilder

    chapters = [
        Chapter(index=0, title="Chapter", paragraphs=["first"]),
        Chapter(index=1, title="Chapter", paragraphs=["second"]),
    ]
    captured = {}

    class FakeReader:
        format_name = "fake"

        def configure_pdf_extraction(self, options):
            pass

        def get_chapters(self):
            return chapters

        def get_metadata(self):
            return SimpleNamespace(title="Book")

    def fake_batch_info(chapter, *, is_first_chapter=False):
        if chapter.index == 0:
            return 1, 10
        return 2, 20

    def fake_build(self, chapters_arg, output_file, chapter_batch_info, total_batches, total_chars):
        captured.update(
            chapters=chapters_arg,
            output_file=output_file,
            chapter_batch_info=chapter_batch_info,
            total_batches=total_batches,
            total_chars=total_chars,
        )
        return True

    monkeypatch.setattr("kenkui.parsing.get_reader", lambda *args, **kwargs: FakeReader())
    monkeypatch.setattr("kenkui.workers.get_batch_info", fake_batch_info)
    monkeypatch.setattr(AudioBuilder, "build", fake_build)

    assert AudioBuilder(_processing_config(tmp_path)).run() is True
    assert captured["chapters"] == chapters
    assert captured["chapter_batch_info"] == [(1, 10, True), (2, 20, False)]
    assert captured["total_batches"] == 3
    assert captured["total_chars"] == 30


def test_worker_malloc_env_sanitizer_only_removes_disabled_values(monkeypatch):
    from kenkui.workers import _sanitize_disabled_malloc_debug_env

    enabled_key = "MallocStackLoggingNoCompact"
    monkeypatch.setenv("MallocStackLogging", "0")
    monkeypatch.setenv(enabled_key, "YES")

    _sanitize_disabled_malloc_debug_env()

    assert "MallocStackLogging" not in os.environ
    assert os.environ[enabled_key] == "YES"


def test_stitching_progress_events_use_millisecond_units(tmp_path, monkeypatch):
    from kenkui.models import AudioResult
    from kenkui.parsing import AudioBuilder

    class FakeStdout:
        def __iter__(self):
            return iter(["out_time_ms=250\n", "out_time_ms=1000\n"])

    class FakeStderr:
        def read(self):
            return ""

    class FakeProcess:
        stdout = FakeStdout()
        stderr = FakeStderr()
        returncode = 0

        def wait(self):
            return None

    events = []
    builder = AudioBuilder(_processing_config(tmp_path), progress_callback=events.append)
    builder.temp_dir = tmp_path

    source = tmp_path / "chapter.wav"
    source.write_bytes(b"fake")
    monkeypatch.setattr("kenkui.parsing.imageio_ffmpeg.get_ffmpeg_exe", lambda: "ffmpeg")
    monkeypatch.setattr("kenkui.parsing.subprocess.Popen", lambda *args, **kwargs: FakeProcess())

    builder._stitch_files(
        [AudioResult(chapter_index=0, title="One", file_path=source, duration_ms=1000)],
        tmp_path / "book.m4b",
    )

    advanced = [event for event in events if event.stage == "stitching"]
    assert [event.unit for event in advanced] == ["milliseconds", "milliseconds"]
    assert [event.completed_units for event in advanced] == [250, 1000]
    assert all(not hasattr(event, "eta_seconds") for event in advanced)


def test_process_chapters_completion_is_driven_by_futures_not_done_messages(tmp_path, monkeypatch):
    from kenkui.models import AudioResult, Chapter
    from kenkui.parsing import AudioBuilder

    progress_events = []
    result = AudioResult(
        chapter_index=0,
        title="Chapter 1",
        file_path=tmp_path / "chapter.wav",
        duration_ms=1000,
    )

    class FakeFuture:
        def done(self):
            return True

        def result(self):
            return result

    class FakePool:
        _processes = {}

        def __init__(self, max_workers):
            self.max_workers = max_workers

        def submit(self, _fn, chapter, _cfg, _temp_dir, progress_queue, is_first):
            progress_queue.put(("START", 123, chapter.title, 1, 7, is_first, chapter.index))
            progress_queue.put(("UPDATE", 123, 1, 1, 1, 7))
            return FakeFuture()

        def shutdown(self, wait=False, cancel_futures=True):
            self.shutdown_args = (wait, cancel_futures)

    monkeypatch.setattr(
        "kenkui.parsing.multiprocessing.Manager",
        lambda: SimpleNamespace(Queue=lambda: queue.Queue()),
    )
    monkeypatch.setattr("kenkui.parsing.ProcessPoolExecutor", FakePool)
    monkeypatch.setattr("kenkui.parsing.as_completed", lambda futures: list(futures))

    builder = AudioBuilder(_processing_config(tmp_path), progress_callback=progress_events.append)
    builder.temp_dir = tmp_path

    results = builder._process_chapters(
        [Chapter(index=0, title="Chapter 1", paragraphs=["hello"])],
        total_batches=1,
        total_chars=7,
    )

    assert results == [result]
    assert progress_events[-1].active_chapters[0].status == "completed"


def test_process_chapters_broken_future_result_has_chapter_context(tmp_path, monkeypatch):
    from kenkui.models import Chapter
    from kenkui.parsing import AudioBuilder

    class FakeFuture:
        def done(self):
            return True

        def result(self):
            raise BrokenPipeError("closed")

    class FakePool:
        _processes = {}

        def __init__(self, max_workers):
            self.max_workers = max_workers

        def submit(self, *_args, **_kwargs):
            return FakeFuture()

        def shutdown(self, wait=False, cancel_futures=True):
            self.shutdown_args = (wait, cancel_futures)

    monkeypatch.setattr(
        "kenkui.parsing.multiprocessing.Manager",
        lambda: SimpleNamespace(Queue=lambda: queue.Queue()),
    )
    monkeypatch.setattr("kenkui.parsing.ProcessPoolExecutor", FakePool)
    monkeypatch.setattr("kenkui.parsing.as_completed", lambda futures: list(futures))

    builder = AudioBuilder(_processing_config(tmp_path))
    builder.temp_dir = tmp_path

    with pytest.raises(RuntimeError) as exc_info:
        builder._process_chapters(
            [Chapter(index=0, title="Pipe Chapter", paragraphs=["hello"])],
            total_batches=1,
            total_chars=5,
        )

    message = str(exc_info.value)
    assert "Pipe Chapter" in message
    assert "progress pipe closed unexpectedly" in message
    assert "workers set to 4 or fewer" in message
