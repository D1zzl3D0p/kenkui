"""Focused unit tests for the audio-assembly components extracted from
AudioBuilder (QP Task 8): AudioBatcher, MetadataWriter, M4BBuilder."""

from __future__ import annotations

import pytest


def _result(index, title, duration_ms, path):
    from kenkui.models import AudioResult

    return AudioResult(chapter_index=index, title=title, file_path=path, duration_ms=duration_ms)


class TestAudioBatcher:
    def test_compute_totals_and_first_chapter_flag(self, monkeypatch):
        from kenkui.audio_build import AudioBatcher
        from kenkui.models import Chapter

        chapters = [
            Chapter(index=0, title="A", paragraphs=["x"]),
            Chapter(index=1, title="B", paragraphs=["y"]),
        ]

        def fake_batch_info(chapter, *, is_first_chapter=False):
            return (1, 10) if chapter.index == 0 else (2, 20)

        monkeypatch.setattr("kenkui.workers.get_batch_info", fake_batch_info)

        info, total_batches, total_chars = AudioBatcher.compute(chapters)
        assert info == [(1, 10, True), (2, 20, False)]
        assert total_batches == 3
        assert total_chars == 30


class TestMetadataWriter:
    def test_write_concat_list_uses_resolved_posix_paths(self, tmp_path):
        from kenkui.audio_build import MetadataWriter

        src = tmp_path / "chapter one.wav"
        src.write_bytes(b"x")
        out = tmp_path / "files.txt"
        MetadataWriter().write_concat_list([_result(0, "One", 100, src)], out)

        text = out.read_text(encoding="utf-8")
        assert text == f"file '{src.resolve().as_posix()}'\n"

    def test_write_chapter_metadata_emits_ffmetadata_chapters(self, tmp_path):
        from kenkui.audio_build import MetadataWriter

        results = [
            _result(0, "One", 1000, tmp_path / "a.wav"),
            _result(1, "Two", 500, tmp_path / "b.wav"),
        ]
        out = tmp_path / "metadata.txt"
        MetadataWriter().write_chapter_metadata(results, out, narrator_label="alba")

        text = out.read_text(encoding="utf-8")
        assert text.startswith(";FFMETADATA1\n")
        assert "comment=Narrated by alba\n" in text
        assert "START=0\nEND=1000\ntitle=One\n" in text
        assert "START=1000\nEND=1500\ntitle=Two\n" in text

    def test_write_chapter_metadata_omits_comment_without_label(self, tmp_path):
        from kenkui.audio_build import MetadataWriter

        out = tmp_path / "metadata.txt"
        MetadataWriter().write_chapter_metadata(
            [_result(0, "One", 1000, tmp_path / "a.wav")], out, narrator_label=""
        )
        assert "comment=" not in out.read_text(encoding="utf-8")

    def test_embed_cover_missing_mutagen_reports_message(self, tmp_path, monkeypatch):
        import builtins

        from kenkui.audio_build import MetadataWriter

        real_import = builtins.__import__

        def fake_import(name, *args, **kwargs):
            if name == "mutagen.mp4":
                raise ImportError("no mutagen")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", fake_import)
        messages: list[str] = []
        MetadataWriter().embed_cover(
            tmp_path / "book.m4b", lambda: (b"data", "image/png"), messages.append
        )
        assert any("mutagen" in m for m in messages)


class TestM4BBuilder:
    def _fake_ffmpeg(self, monkeypatch, out_lines):
        class FakeStdout:
            def __iter__(self):
                return iter(out_lines)

        class FakeStderr:
            def read(self):
                return ""

        class FakeProcess:
            stdout = FakeStdout()
            stderr = FakeStderr()
            returncode = 0

            def wait(self):
                return None

        monkeypatch.setattr("kenkui.audio_build.imageio_ffmpeg.get_ffmpeg_exe", lambda: "ffmpeg")
        monkeypatch.setattr("kenkui.audio_build.subprocess.Popen", lambda *a, **k: FakeProcess())

    def test_stitch_emits_millisecond_progress(self, tmp_path, monkeypatch):
        from kenkui.audio_build import M4BBuilder

        self._fake_ffmpeg(monkeypatch, ["out_time_ms=250\n", "out_time_ms=1000\n"])
        events: list[dict] = []

        def emit(stage, status, message, **kwargs):
            events.append({"stage": stage, "status": status, **kwargs})

        src = tmp_path / "c.wav"
        src.write_bytes(b"x")
        M4BBuilder().stitch(
            [_result(0, "One", 1000, src)],
            tmp_path / "book.m4b",
            tmp_path / "files.txt",
            tmp_path / "metadata.txt",
            bitrate="96k",
            emit=emit,
        )
        assert [e["unit"] for e in events] == ["milliseconds", "milliseconds"]
        assert [e["completed_units"] for e in events] == [250, 1000]

    def test_stitch_raises_on_nonzero_returncode(self, tmp_path, monkeypatch):
        import subprocess

        from kenkui.audio_build import M4BBuilder

        class FakeStdout:
            def __iter__(self):
                return iter([])

        class FakeStderr:
            def read(self):
                return "boom"

        class FakeProcess:
            stdout = FakeStdout()
            stderr = FakeStderr()
            returncode = 1

            def wait(self):
                return None

        monkeypatch.setattr("kenkui.audio_build.imageio_ffmpeg.get_ffmpeg_exe", lambda: "ffmpeg")
        monkeypatch.setattr("kenkui.audio_build.subprocess.Popen", lambda *a, **k: FakeProcess())

        with pytest.raises(subprocess.CalledProcessError):
            M4BBuilder().stitch(
                [_result(0, "One", 1000, tmp_path / "c.wav")],
                tmp_path / "book.m4b",
                tmp_path / "files.txt",
                tmp_path / "metadata.txt",
                bitrate="96k",
                emit=lambda *a, **k: None,
            )
