"""Explicitly opt-in host FFmpeg acceptance; never runs in default gates."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import sqlite3
import subprocess
import tempfile
from html import escape
from typing import TYPE_CHECKING, BinaryIO, cast

import pytest

import kenkui as kk
from kenkui._audio.production import FFmpegM4BAssembler
from kenkui._execution.cache import (
    AUDIO_CONTRACT_VERSION,
    CACHE_SCHEMA_VERSION,
    CacheStore,
)
from kenkui._execution.coordinator import ExecutionBindings
from kenkui._execution.process_pool import EngineSpecification
from test_epub import make_epub, xhtml
from test_execution import _bind, _voice

if TYPE_CHECKING:
    from pathlib import Path

_EXPECTED_CHAPTERS = 2
_MAX_PROBE_BYTES = 1024 * 1024
_MAX_DECODED_PCM_BYTES = 64 * 1024 * 1024
_SHA256_HEX_LENGTH = 64
_PROBE_TIMEOUT_SECONDS = 30
_DECODE_TIMEOUT_SECONDS = 60
_TITLE = "Book = #; \\ title\ncontinued & complete"
_AUTHOR = "Author = #; \\ name & co."
_CHAPTER_TITLES = (
    "Opening = #; \\ & one",
    "Closing; = # \\ & two",
)
_MP4_FORMAT_NAMES = {"mov", "mp4", "m4a", "3gp", "3g2", "mj2"}

pytestmark = [
    pytest.mark.native,
    pytest.mark.skipif(
        os.environ.get("KENKUI_RUN_NATIVE") != "1",
        reason="set KENKUI_RUN_NATIVE=1 and select -m native explicitly",
    ),
]


def _read_bounded(stream: BinaryIO) -> bytes:
    """Read a tempfile only after proving its captured output is bounded."""
    size = stream.tell()
    assert size <= _MAX_PROBE_BYTES, f"native tool output exceeded {size} bytes"
    stream.seek(0)
    return stream.read(_MAX_PROBE_BYTES + 1)


def _host_tool(name: str) -> str:
    """Resolve a host executable independently of the production tool wrapper."""
    executable = shutil.which(name)
    assert executable is not None, f"host {name} is required for native acceptance"
    return executable


def _probe_with_host_ffprobe(output: Path) -> dict[str, object]:
    """Run host ffprobe by argv and parse only bounded tempfile output."""
    with tempfile.TemporaryFile() as stdout, tempfile.TemporaryFile() as stderr:
        completed = subprocess.run(  # noqa: S603 - explicit native opt-in acceptance.
            [
                _host_tool("ffprobe"),
                "-v",
                "error",
                "-show_format",
                "-show_streams",
                "-show_chapters",
                "-of",
                "json",
                str(output),
            ],
            stdin=subprocess.DEVNULL,
            stdout=stdout,
            stderr=stderr,
            check=False,
            timeout=_PROBE_TIMEOUT_SECONDS,
        )
        raw_stdout = _read_bounded(cast("BinaryIO", stdout))
        raw_stderr = _read_bounded(cast("BinaryIO", stderr))
    assert completed.returncode == 0, raw_stderr.decode("utf-8", errors="replace")
    payload = json.loads(raw_stdout)
    assert type(payload) is dict
    return cast("dict[str, object]", payload)


def _decode_with_host_ffmpeg(output: Path) -> None:
    """Require an independent full decode with no shell or captured output."""
    completed = subprocess.run(  # noqa: S603 - explicit native opt-in acceptance.
        [
            _host_tool("ffmpeg"),
            "-v",
            "error",
            "-i",
            str(output),
            "-map",
            "0",
            "-f",
            "null",
            "-",
        ],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
        timeout=_DECODE_TIMEOUT_SECONDS,
    )
    assert completed.returncode == 0


def _decoded_audio_fingerprint(output: Path) -> tuple[str, int]:
    """Decode only audio streams to bounded PCM and return its SHA-256 and size."""
    with tempfile.TemporaryFile() as pcm:
        completed = subprocess.run(  # noqa: S603 - explicit native opt-in acceptance.
            [
                _host_tool("ffmpeg"),
                "-v",
                "error",
                "-i",
                str(output),
                "-map",
                "0:a",
                "-c:a",
                "pcm_s16le",
                "-f",
                "s16le",
                "pipe:1",
            ],
            stdin=subprocess.DEVNULL,
            stdout=pcm,
            stderr=subprocess.DEVNULL,
            check=False,
            timeout=_DECODE_TIMEOUT_SECONDS,
        )
        size = pcm.tell()
        assert completed.returncode == 0
        assert 0 < size <= _MAX_DECODED_PCM_BYTES
        pcm.seek(0)
        digest = hashlib.sha256()
        observed_size = 0
        while chunk := pcm.read(64 * 1024):
            observed_size += len(chunk)
            assert observed_size <= _MAX_DECODED_PCM_BYTES
            digest.update(chunk)
    assert observed_size == size
    return digest.hexdigest(), size


def _probe_chapter_titles(payload: dict[str, object]) -> tuple[str, ...]:
    """Extract chapter ordering from an independently generated probe payload."""
    chapters = payload.get("chapters")
    assert type(chapters) is list
    titles: list[str] = []
    for chapter in chapters:
        assert type(chapter) is dict
        tags = chapter.get("tags")
        assert type(tags) is dict
        title = tags.get("title")
        assert isinstance(title, str)
        titles.append(title)
    return tuple(titles)


def _assert_valid_cache_rows(cache_directory: Path, *, expected_rows: int) -> None:
    """Check that every published cache row names intact, bounded PCM."""
    database = cache_directory / "cache.sqlite3"
    with sqlite3.connect(database) as connection:
        rows = connection.execute(
            """SELECT schema_version,audio_contract_version,payload_sha256,
            payload_bytes,frame_count,duration_ms,sample_rate_hz,channels
            FROM segment_cache ORDER BY ordinal"""
        ).fetchall()
    assert len(rows) == expected_rows
    for (
        schema_version,
        audio_contract_version,
        payload_sha256,
        payload_bytes,
        frame_count,
        duration_ms,
        sample_rate_hz,
        channels,
    ) in rows:
        assert schema_version == CACHE_SCHEMA_VERSION
        assert audio_contract_version == AUDIO_CONTRACT_VERSION
        assert isinstance(payload_sha256, str)
        assert len(payload_sha256) == _SHA256_HEX_LENGTH
        assert 0 < payload_bytes <= _MAX_DECODED_PCM_BYTES
        assert frame_count > 0
        assert duration_ms > 0
        assert sample_rate_hz > 0
        assert channels > 0
        payload = cache_directory / "payloads" / f"{payload_sha256}.pcm"
        assert payload.stat().st_size == payload_bytes
        assert hashlib.sha256(payload.read_bytes()).hexdigest() == payload_sha256


def _assert_probe(
    payload: dict[str, object], *, expect_cover: bool, chapter_titles: tuple[str, ...]
) -> None:
    """Assert independently observed container, codec, tags, chapters, and cover."""
    format_data = payload.get("format")
    streams = payload.get("streams")
    chapters = payload.get("chapters")
    assert type(format_data) is dict
    assert type(streams) is list
    assert type(chapters) is list

    typed_format = cast("dict[str, object]", format_data)
    format_name = typed_format.get("format_name")
    tags = typed_format.get("tags")
    assert isinstance(format_name, str)
    assert _MP4_FORMAT_NAMES & set(format_name.split(","))
    assert type(tags) is dict
    typed_tags = cast("dict[str, object]", tags)
    assert typed_tags.get("title") == _TITLE
    assert typed_tags.get("artist") == _AUTHOR

    typed_streams = [
        cast("dict[str, object]", stream) for stream in streams if type(stream) is dict
    ]
    assert len(typed_streams) == len(streams)
    audio_streams = [
        stream for stream in typed_streams if stream.get("codec_type") == "audio"
    ]
    assert len(audio_streams) == 1
    assert audio_streams[0].get("codec_name") == "aac"
    attached_pictures = [
        stream
        for stream in typed_streams
        if stream.get("codec_type") == "video"
        and type(stream.get("disposition")) is dict
        and cast("dict[str, object]", stream["disposition"]).get("attached_pic") == 1
    ]
    assert len(attached_pictures) == int(expect_cover)
    if not expect_cover:
        assert not [
            stream for stream in typed_streams if stream.get("codec_type") == "video"
        ]

    assert len(chapters) == len(chapter_titles) == _EXPECTED_CHAPTERS
    observed_titles: list[object] = []
    for chapter in chapters:
        assert type(chapter) is dict
        chapter_tags = chapter.get("tags")
        assert type(chapter_tags) is dict
        observed_titles.append(chapter_tags.get("title"))
    assert observed_titles == list(chapter_titles)


@pytest.mark.parametrize(
    ("expect_cover", "workers"),
    [(False, 1), (True, 2)],
    ids=("no-cover-serial", "source-cover-parallel"),
)
def test_public_write_is_real_mp4_aac_with_chapters_metadata_and_optional_cover(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    expect_cover: bool,
    workers: int,
) -> None:
    """Exercise public writes and independently probe and decode each artifact."""
    source = make_epub(
        tmp_path / f"source-{expect_cover}.epub",
        chapters={
            "one": xhtml(f"<h1>{escape(_CHAPTER_TITLES[0])}</h1><p>Exact first.</p>"),
            "two": xhtml(f"<h1>{escape(_CHAPTER_TITLES[1])}</h1><p>Exact second.</p>"),
        },
        spine=("one", "two"),
        cover=expect_cover,
    )
    pipeline = (
        kk.epub(source)
        .assign_voice("narrator")
        .tts()
        .metadata(
            title=_TITLE,
            author=_AUTHOR,
            cover="source" if expect_cover else None,
        )
    )
    inspection_titles = tuple(chapter.title for chapter in pipeline.inspect().chapters)
    assert inspection_titles == _CHAPTER_TITLES
    _bind(monkeypatch, EngineSpecification.fake(), FFmpegM4BAssembler())
    output = tmp_path / f"native-{expect_cover}.m4b"

    result = pipeline.write_m4b(output, workers=workers)

    assert result.output == output
    assert result.stats.rendered_chapters == _EXPECTED_CHAPTERS
    assert result.stats.duration_ms > 0
    assert output.is_file()
    _assert_probe(
        _probe_with_host_ffprobe(output),
        expect_cover=expect_cover,
        chapter_titles=inspection_titles,
    )
    _decode_with_host_ffmpeg(output)


def test_native_cold_and_warm_cache_outputs_are_decode_equivalent(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Prove cached PCM produces the same native AAC decode across worker counts."""
    source = make_epub(
        tmp_path / "cache-source.epub",
        chapters={
            "one": xhtml(f"<h1>{escape(_CHAPTER_TITLES[0])}</h1><p>Exact first.</p>"),
            "two": xhtml(f"<h1>{escape(_CHAPTER_TITLES[1])}</h1><p>Exact second.</p>"),
        },
        spine=("one", "two"),
    )
    pipeline = (
        kk.epub(source)
        .assign_voice("narrator")
        .tts()
        .metadata(title=_TITLE, author=_AUTHOR, cover=None)
    )
    inspection_titles = tuple(chapter.title for chapter in pipeline.inspect().chapters)
    assert inspection_titles == _CHAPTER_TITLES
    cache_directory = tmp_path / "private-cache"
    cache = CacheStore(cache_directory)
    bindings = ExecutionBindings(
        EngineSpecification.fake(),
        FFmpegM4BAssembler(),
        _voice(),
        "fake-v1",
        cache,
    )
    monkeypatch.setattr("kenkui.pipeline._execution_bindings", lambda: bindings)

    cold_output = tmp_path / "cold.m4b"
    warm_output = tmp_path / "warm.m4b"
    cold = pipeline.write_m4b(cold_output, workers=2)
    warm = pipeline.write_m4b(warm_output, workers=1)

    assert cold.output == cold_output
    assert warm.output == warm_output
    assert cold.stats == warm.stats
    assert cold.stats.rendered_chapters == _EXPECTED_CHAPTERS
    _assert_valid_cache_rows(cache_directory, expected_rows=_EXPECTED_CHAPTERS)

    cold_probe = _probe_with_host_ffprobe(cold_output)
    warm_probe = _probe_with_host_ffprobe(warm_output)
    _assert_probe(
        cold_probe,
        expect_cover=False,
        chapter_titles=inspection_titles,
    )
    _assert_probe(
        warm_probe,
        expect_cover=False,
        chapter_titles=inspection_titles,
    )
    assert _probe_chapter_titles(cold_probe) == _probe_chapter_titles(warm_probe)

    assert _decoded_audio_fingerprint(cold_output) == _decoded_audio_fingerprint(
        warm_output
    )
    _decode_with_host_ffmpeg(cold_output)
    _decode_with_host_ffmpeg(warm_output)


@pytest.mark.native
def test_native_write_with_pauses_is_longer_and_still_decodes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Generated silence must survive real FFmpeg encoding and decoding.

    The other native cases render without pauses, so padding is zero and
    FFmpeg never sees a padded chapter part. This is the only case that proves
    the part-size check, the chapter markers and the encoder all agree once
    silence is actually present.
    """
    chapters = {
        "one": xhtml(f"<h1>{escape(_CHAPTER_TITLES[0])}</h1><p>Exact first.</p>"),
        "two": xhtml(f"<h1>{escape(_CHAPTER_TITLES[1])}</h1><p>Exact second.</p>"),
    }
    source = make_epub(
        tmp_path / "paused.epub", chapters=chapters, spine=("one", "two")
    )
    _bind(monkeypatch, EngineSpecification.fake(), FFmpegM4BAssembler())

    plain = (
        kk.epub(source).assign_voice("narrator").tts().write_m4b(tmp_path / "plain.m4b")
    )
    paused = (
        kk.epub(source)
        .pauses(chapter_ms=2000)
        .assign_voice("narrator")
        .tts()
        .write_m4b(tmp_path / "paused.m4b")
    )

    assert paused.stats.duration_ms > plain.stats.duration_ms
    assert paused.stats.rendered_chapters == _EXPECTED_CHAPTERS
    _decode_with_host_ffmpeg(tmp_path / "paused.m4b")
    probed = _probe_with_host_ffprobe(tmp_path / "paused.m4b")
    assert _probe_chapter_titles(probed) == _CHAPTER_TITLES


# A genuinely valid 1x1 PNG. A synthetic header would satisfy read_cover's
# signature check and then be rejected by FFmpeg, so this case needs the real
# thing to prove anything.
_REAL_PNG = bytes.fromhex(
    "89504e470d0a1a0a0000000d4948445200000001000000010802000000907753de"
    "0000000c49444154789c63f8cfc0000003010100c9fe92ef0000000049454e44ae"
    "426082"
)


@pytest.mark.native
def test_native_write_embeds_a_caller_supplied_cover(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A caller-supplied image must actually reach the encoded artifact."""
    source = make_epub(
        tmp_path / "cover-source.epub",
        chapters={
            "one": xhtml(f"<h1>{escape(_CHAPTER_TITLES[0])}</h1><p>Exact first.</p>"),
            "two": xhtml(f"<h1>{escape(_CHAPTER_TITLES[1])}</h1><p>Exact second.</p>"),
        },
        spine=("one", "two"),
        cover=False,
    )
    art = tmp_path / "art.png"
    art.write_bytes(_REAL_PNG)
    _bind(monkeypatch, EngineSpecification.fake(), FFmpegM4BAssembler())
    output = tmp_path / "with-cover.m4b"

    kk.epub(source).assign_voice("narrator").tts().metadata(
        title=_TITLE, author=_AUTHOR, cover=art
    ).write_m4b(output)

    # The EPUB carries no cover of its own, so any video stream in the output
    # can only have come from the caller's file.
    _assert_probe(
        _probe_with_host_ffprobe(output),
        expect_cover=True,
        chapter_titles=_CHAPTER_TITLES,
    )
    _decode_with_host_ffmpeg(output)


@pytest.mark.native
def test_native_write_rejects_an_unreadable_cover(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An invalid caller cover fails the render rather than falling back."""
    source = make_epub(
        tmp_path / "bad-cover-source.epub",
        chapters={"one": xhtml("<h1>One</h1><p>Exact first.</p>")},
        spine=("one",),
        cover=True,
    )
    art = tmp_path / "not-an-image.png"
    art.write_bytes(b"definitely not a png")
    _bind(monkeypatch, EngineSpecification.fake(), FFmpegM4BAssembler())
    output = tmp_path / "rejected.m4b"

    with pytest.raises(kk.EncodingError) as error:
        kk.epub(source).assign_voice("narrator").tts().metadata(cover=art).write_m4b(
            output
        )

    assert error.value.code is kk.ErrorCode.COVER_INVALID
    assert not output.exists()
