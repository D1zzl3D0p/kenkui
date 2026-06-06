from __future__ import annotations

import os
from pathlib import Path


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
