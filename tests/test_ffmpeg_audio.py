"""WP7 FFmpeg assembly tests; every process and discovery effect is mocked."""
# ruff: noqa: D102, D103, FBT001, FBT003, PLR0911, PLR2004, PT018, RUF007, SIM300, SLF001

from __future__ import annotations

import io
import json
import subprocess
import sys
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

import kenkui as kk
from kenkui._audio import native as native_module
from kenkui._audio.ffmpeg import FFmpegTools
from kenkui._audio.m4b import AssemblyRequest, cumulative_frame_boundaries_ms
from kenkui._audio.native import NativeCommandRunner, SubprocessRunner, run_checked
from kenkui._audio.production import FFmpegM4BAssembler
from kenkui._domain.planning import (
    CoverIntent,
    ExecutionPlan,
    OutputChapter,
    OutputMetadata,
    SchemaVersions,
    SpeechSegment,
    VoicePlan,
)
from kenkui._execution.process_pool import EngineSpecification
from kenkui._tts.protocols import SynthesizedAudio, segment_audio
from test_epub import make_epub, xhtml
from test_execution import _bind, _pipeline

if TYPE_CHECKING:
    from collections.abc import Sequence


@dataclass
class MockRunner(NativeCommandRunner):
    """Record argv and emulate only the native outputs selected by a test."""

    probe_payload: dict[str, object] = field(default_factory=dict)
    fail_on: str | None = None
    timeout_on: str | None = None
    calls: list[tuple[tuple[str, ...], float]] = field(default_factory=list)
    metadata_text: str = ""
    cover_bytes: bytes | None = None
    encoders_stdout: str = (
        "Encoders:\n A..... aac AAC (Advanced Audio Coding)\n"
        " V..... mjpeg MJPEG encoder"
    )
    decoders_stdout: str = "Decoders:\n V..... png PNG image\n V..... mjpeg MJPEG image"

    def run(
        self, argv: Sequence[str], *, timeout: float
    ) -> subprocess.CompletedProcess[str]:
        command = tuple(argv)
        self.calls.append((command, timeout))
        operation = self._operation(command)
        if operation == self.timeout_on:
            raise subprocess.TimeoutExpired(command, timeout, stderr="/private/secret")
        returncode = 23 if operation == self.fail_on else 0
        stdout = ""
        stderr = "token=/private/secret" if returncode else ""
        if operation == "ffmpeg-version":
            stdout = "ffmpeg version 6.1 Copyright"
        elif operation == "encoders":
            stdout = self.encoders_stdout
        elif operation == "muxers":
            stdout = "Muxers:\n E mp4 MP4 (MPEG-4 Part 14)"
        elif operation == "decoders":
            stdout = self.decoders_stdout
        elif operation == "ffprobe-version":
            stdout = "ffprobe version 6.1 Copyright"
        elif operation == "encode":
            metadata = Path(command[command.index("-f", 8) + 3])
            self.metadata_text = metadata.read_text(encoding="utf-8")
            if "-disposition:v:0" in command:
                cover_index = command.index("-i", command.index(str(metadata)) + 1)
                self.cover_bytes = Path(command[cover_index + 1]).read_bytes()
            Path(command[-1]).write_bytes(b"mock-m4b")
        elif operation == "probe":
            stdout = json.dumps(self.probe_payload)
        return subprocess.CompletedProcess(command, returncode, stdout, stderr)

    @staticmethod
    def _operation(command: tuple[str, ...]) -> str:
        executable = Path(command[0]).name
        if executable == "ffprobe" and "-show_format" in command:
            return "probe"
        if executable == "ffprobe":
            return "ffprobe-version"
        if "-encoders" in command:
            return "encoders"
        if "-muxers" in command:
            return "muxers"
        if "-decoders" in command:
            return "decoders"
        if "-version" in command:
            return "ffmpeg-version"
        if command[-1] == "-":
            return "decode"
        return "encode"


def _plan(*, cover: CoverIntent = CoverIntent.NONE) -> ExecutionPlan:
    chapters = (
        OutputChapter("chapter-1", 0, "One = #; \\ title", 3),
        OutputChapter("chapter-2", 1, "Two\ncontinued", 3),
    )
    return ExecutionPlan(
        SchemaVersions("parser", "normalization", "planning", "render"),
        "a" * 64,
        "fake-v1",
        VoicePlan(
            "voice", "Voice", "b" * 64, "en", "fixture", "CC0", True, ("fake-v1",)
        ),
        (
            SpeechSegment("segment-1", "chapter-1", 0, "one", 3, "c" * 64),
            SpeechSegment("segment-2", "chapter-2", 1, "two", 3, "d" * 64),
        ),
        OutputMetadata(
            "Book = #; \\ title\ncontinued",
            "Author = #; \\ name",
            chapters,
            cover,
            cover is CoverIntent.SOURCE,
        ),
        6,
        "e" * 64,
    )


def _audio() -> tuple[SynthesizedAudio, ...]:
    return tuple(
        SynthesizedAudio(
            f"segment-{number}",
            f"chapter-{number}",
            b"\0\0" * 16_000,
            16_000,
            1,
            16_000,
            1_000,
        )
        for number in (1, 2)
    )


def _request(
    plan: ExecutionPlan,
    audio: tuple[SynthesizedAudio, ...],
    output: Path,
    source: Path | None = None,
) -> AssemblyRequest:
    """Spill in-memory audio to per-chapter parts the way rendering now does."""
    directory = output.parent / f"parts-{output.stem}"
    directory.mkdir(exist_ok=True)
    parts: list[Path] = []
    payloads: list[bytes] = []
    for index, item in enumerate(audio):
        payloads.append(item.pcm_s16le)
        last = index + 1 == len(audio) or audio[index + 1].chapter_id != item.chapter_id
        if last:
            part = directory / f"chapter-{len(parts):05d}.pcm"
            part.write_bytes(b"".join(payloads))
            parts.append(part)
            payloads.clear()
    return AssemblyRequest(
        plan, tuple(segment_audio(item) for item in audio), tuple(parts), output, source
    )


def _probe(
    *, cover: bool = False, data: bool = True, codec: str = "aac"
) -> dict[str, object]:
    streams: list[dict[str, object]] = [
        {
            "codec_type": "audio",
            "codec_name": codec,
            "sample_rate": "16000",
            "channels": 1,
            "duration": "2.000",
        }
    ]
    if data:
        streams.append(
            {
                "index": 1,
                "codec_name": "bin_data",
                "codec_type": "data",
                "codec_tag_string": "text",
            }
        )
    if cover:
        streams.append(
            {
                "index": len(streams),
                "codec_name": "mjpeg",
                "codec_type": "video",
                "width": 1,
                "height": 1,
                "disposition": {"attached_pic": 1},
            }
        )
    return {
        "format": {"format_name": "mov,mp4,m4a,3gp,3g2,mj2", "duration": "2.000"},
        "streams": streams,
        "chapters": [
            {
                "start_time": "0.000",
                "end_time": "1.000",
                "tags": {"title": "One = #; \\ title"},
            },
            {
                "start_time": "1.000",
                "end_time": "2.000",
                "tags": {"title": "Two\ncontinued"},
            },
        ],
    }


def _assembler(
    runner: MockRunner, *, resolver: object | None = None
) -> FFmpegM4BAssembler:
    paths = {"ffmpeg": "/mock tools/ffmpeg", "ffprobe": "/mock tools/ffprobe"}
    discover = resolver if resolver is not None else paths.get
    return FFmpegM4BAssembler(FFmpegTools(runner, discover=discover))  # type: ignore[arg-type]


def test_preflight_discovers_both_tools_and_checks_version_and_aac() -> None:
    runner = MockRunner()
    assembler = _assembler(runner)

    assembler.preflight()

    assert [call[0] for call in runner.calls] == [
        ("/mock tools/ffmpeg", "-hide_banner", "-version"),
        ("/mock tools/ffmpeg", "-hide_banner", "-encoders"),
        ("/mock tools/ffmpeg", "-hide_banner", "-muxers"),
        ("/mock tools/ffprobe", "-hide_banner", "-version"),
    ]
    assert all(timeout > 0 for _, timeout in runner.calls)


def test_cover_preflight_requires_mjpeg_encoder_and_relevant_image_decoders() -> None:
    no_cover = MockRunner(
        encoders_stdout="Encoders:\n A..... aac AAC",
        decoders_stdout="",
    )
    _assembler(no_cover).preflight(expect_cover=False)
    assert "decoders" not in [MockRunner._operation(call) for call, _ in no_cover.calls]

    missing_encoder = MockRunner(encoders_stdout="Encoders:\n A..... aac AAC")
    with pytest.raises(kk.EncodingError) as encoder_error:
        _assembler(missing_encoder).preflight(expect_cover=True)
    assert encoder_error.value.code == kk.ErrorCode.FFMPEG_UNSUPPORTED

    for decoder in ("png", "mjpeg"):
        runner = MockRunner(
            decoders_stdout=(
                "Decoders:\n V..... mjpeg MJPEG image"
                if decoder == "png"
                else "Decoders:\n V..... png PNG image"
            )
        )
        with pytest.raises(kk.EncodingError) as decoder_error:
            _assembler(runner).preflight(expect_cover=True)
        assert decoder_error.value.code == kk.ErrorCode.FFMPEG_UNSUPPORTED


@pytest.mark.parametrize(
    ("missing", "code"),
    [
        ("ffmpeg", kk.ErrorCode.FFMPEG_NOT_FOUND),
        ("ffprobe", kk.ErrorCode.FFPROBE_NOT_FOUND),
    ],
)
def test_preflight_missing_tools_have_distinct_stable_mappings(
    missing: str, code: kk.ErrorCode
) -> None:
    runner = MockRunner()
    resolver = lambda name: None if name == missing else f"/mock/{name}"  # noqa: E731

    with pytest.raises(kk.EncodingError) as caught:
        _assembler(runner, resolver=resolver).preflight()

    assert caught.value.code == code
    assert caught.value.__cause__ is None
    assert not runner.calls


@pytest.mark.parametrize(
    ("operation", "code"),
    [
        ("ffmpeg-version", kk.ErrorCode.FFMPEG_UNSUPPORTED),
        ("encoders", kk.ErrorCode.FFMPEG_UNSUPPORTED),
        ("ffprobe-version", kk.ErrorCode.FFPROBE_UNSUPPORTED),
    ],
)
def test_preflight_nonzero_and_timeout_are_sanitized(
    operation: str, code: kk.ErrorCode
) -> None:
    runner = MockRunner(fail_on=operation)
    with pytest.raises(kk.EncodingError) as caught:
        _assembler(runner).preflight()
    assert caught.value.code == code
    assert "secret" not in str(caught.value)
    assert caught.value.__cause__ is None

    timed = MockRunner(timeout_on=operation)
    with pytest.raises(kk.EncodingError) as timeout:
        _assembler(timed).preflight()
    assert timeout.value.code == code
    assert "secret" not in str(timeout.value)


def test_assembly_builds_exact_safe_argv_metadata_chapters_and_no_cover(
    tmp_path: Path,
) -> None:
    runner = MockRunner(probe_payload=_probe())
    assembler = _assembler(runner)
    output = tmp_path / "output with ünicode.m4b"

    result = assembler.assemble(_request(_plan(), _audio(), output))

    encode = next(call for call, _ in runner.calls if call[-1] == str(output))
    assert encode[:8] == (
        "/mock tools/ffmpeg",
        "-hide_banner",
        "-nostdin",
        "-n",
        "-f",
        "s16le",
        "-ar",
        "16000",
    )
    assert ("-ac", "1") == encode[8:10]
    assert "-map_metadata" in encode and "-map_chapters" in encode
    assert encode[encode.index("-c:a") + 1] == "aac"
    assert "-disposition:v:0" not in encode
    assert runner.cover_bytes is None
    assert runner.metadata_text == (
        ";FFMETADATA1\n"
        "title=Book \\= \\#\\; \\\\ title\\\ncontinued\n"
        "artist=Author \\= \\#\\; \\\\ name\n"
        "author=Author \\= \\#\\; \\\\ name\n"
        "[CHAPTER]\nTIMEBASE=1/1000\nSTART=0\nEND=1000\n"
        "title=One \\= \\#\\; \\\\ title\n"
        "[CHAPTER]\nTIMEBASE=1/1000\nSTART=1000\nEND=2000\n"
        "title=Two\\\ncontinued\n"
    )
    assert [Path(call[0]).name for call, _ in runner.calls[-2:]] == [
        "ffprobe",
        "ffmpeg",
    ]
    assert result.artifact == output
    assert result.duration_ms == 2_000
    assert [chapter.duration_ms for chapter in result.chapters] == [1_000, 1_000]


def test_requested_cover_is_materialized_from_epub_to_fixed_private_file(
    tmp_path: Path,
) -> None:
    source = make_epub(
        tmp_path / "book.epub",
        chapters={"one": xhtml("<h1>One</h1>text")},
        spine=("one",),
        cover=True,
    )
    runner = MockRunner(probe_payload=_probe(cover=True))
    output = tmp_path / "covered.m4b"

    _assembler(runner).assemble(
        _request(_plan(cover=CoverIntent.SOURCE), _audio(), output, source)
    )

    encode = next(call for call, _ in runner.calls if call[-1] == str(output))
    assert runner.cover_bytes == bytes.fromhex(
        "89504e470d0a1a0a0000000d4948445200000001000000010804000000"
        "b51c0c020000000b4944415478da6364f80f00010501012718e3660000"
        "000049454e44ae426082"
    )
    assert "-disposition:v:0" in encode
    assert "attached_pic" in encode
    assert not list(tmp_path.glob("*.cover"))


@pytest.mark.parametrize(
    ("failure", "code"),
    [
        ("encode", kk.ErrorCode.ENCODING_FAILED),
        ("probe", kk.ErrorCode.PROBE_FAILED),
        ("decode", kk.ErrorCode.DECODE_FAILED),
    ],
)
def test_nonzero_and_timeout_failures_never_leave_candidate(
    tmp_path: Path, failure: str, code: kk.ErrorCode
) -> None:
    for mode in ("fail_on", "timeout_on"):
        output = tmp_path / f"{failure}-{mode}.m4b"
        runner = (
            MockRunner(probe_payload=_probe(), fail_on=failure)
            if mode == "fail_on"
            else MockRunner(probe_payload=_probe(), timeout_on=failure)
        )
        with pytest.raises(kk.EncodingError) as caught:
            _assembler(runner).assemble(_request(_plan(), _audio(), output))
        assert caught.value.code == code
        assert "secret" not in str(caught.value)
        assert caught.value.__cause__ is None
        assert not output.exists()


@pytest.mark.parametrize(
    "payload",
    [
        {},
        {"format": "wrong", "streams": [], "chapters": []},
        _probe(codec="mp3"),
        {**_probe(), "chapters": []},
        {
            **_probe(),
            "chapters": [
                {"start_time": "1", "end_time": "0", "tags": {"title": "wrong"}},
                {
                    "start_time": "0",
                    "end_time": "2",
                    "tags": {"title": "Two\ncontinued"},
                },
            ],
        },
    ],
)
def test_corrupt_probe_payload_is_rejected_and_not_published(
    tmp_path: Path, payload: dict[str, object]
) -> None:
    output = tmp_path / "corrupt.m4b"
    with pytest.raises(kk.EncodingError) as caught:
        _assembler(MockRunner(probe_payload=payload)).assemble(
            _request(_plan(), _audio(), output)
        )
    assert caught.value.code == kk.ErrorCode.INVALID_ARTIFACT
    assert not output.exists()


def test_cover_probe_semantics_are_authoritative(tmp_path: Path) -> None:
    source = make_epub(
        tmp_path / "book.epub",
        chapters={"one": xhtml("text")},
        spine=("one",),
        cover=True,
    )
    output = tmp_path / "missing-cover.m4b"
    with pytest.raises(kk.EncodingError) as caught:
        _assembler(MockRunner(probe_payload=_probe(cover=False))).assemble(
            _request(_plan(cover=CoverIntent.SOURCE), _audio(), output, source)
        )
    assert caught.value.code == kk.ErrorCode.INVALID_ARTIFACT
    assert not output.exists()


def test_corrupt_attached_cover_fails_full_decode_of_every_mapped_stream(
    tmp_path: Path,
) -> None:
    source = make_epub(
        tmp_path / "book.epub",
        chapters={"one": xhtml("text")},
        spine=("one",),
        cover=True,
    )
    output = tmp_path / "corrupt-cover.m4b"
    runner = MockRunner(probe_payload=_probe(cover=True), fail_on="decode")

    with pytest.raises(kk.EncodingError) as caught:
        _assembler(runner).assemble(
            _request(_plan(cover=CoverIntent.SOURCE), _audio(), output, source)
        )

    assert caught.value.code == kk.ErrorCode.DECODE_FAILED
    decode = next(
        call for call, _ in runner.calls if MockRunner._operation(call) == "decode"
    )
    assert decode[decode.index("-map") + 1] == "0"
    assert not output.exists()


def test_hundred_fractional_frame_chapters_telescope_and_probe_within_tolerance(
    tmp_path: Path,
) -> None:
    count = 100
    rate = 44_100
    frames = 1_001
    base = _plan()
    segments = tuple(
        SpeechSegment(f"segment-{i}", f"chapter-{i}", i, "x", 1, f"{i:064x}")
        for i in range(count)
    )
    chapters = tuple(
        OutputChapter(f"chapter-{i}", i, f"Chapter {i}", 1) for i in range(count)
    )
    plan = replace(
        base,
        segments=segments,
        output=replace(base.output, chapters=chapters),
        total_speech_characters=count,
    )
    audio = tuple(
        SynthesizedAudio(
            f"segment-{i}",
            f"chapter-{i}",
            b"\0\0" * frames,
            rate,
            1,
            frames,
            frames * 1_000 // rate,
        )
        for i in range(count)
    )
    boundaries = cumulative_frame_boundaries_ms(tuple(map(segment_audio, audio)))
    assert boundaries[-1] == count * frames * 1_000 // rate
    assert (
        sum(
            end - start
            for start, end in zip(boundaries[:-1], boundaries[1:], strict=True)
        )
        == boundaries[-1]
    )
    payload: dict[str, object] = {
        "format": {
            "format_name": "mov,mp4,m4a,3gp,3g2,mj2",
            "duration": f"{boundaries[-1] / 1_000 + 0.074:.3f}",
        },
        "streams": [
            {
                "codec_type": "audio",
                "codec_name": "aac",
                "sample_rate": str(rate),
                "channels": 1,
            },
            {"codec_type": "data", "codec_name": "bin_data"},
        ],
        "chapters": [
            {
                "start_time": f"{start / 1_000 + 0.074:.3f}",
                "end_time": f"{end / 1_000 + 0.074:.3f}",
                "tags": {"title": chapter.title},
            }
            for chapter, start, end in zip(
                chapters, boundaries[:-1], boundaries[1:], strict=True
            )
        ],
    }
    runner = MockRunner(probe_payload=payload)
    output = tmp_path / "fractional.m4b"
    result = _assembler(runner).assemble(_request(plan, audio, output))
    assert result.duration_ms == boundaries[-1]
    metadata_bounds = [
        int(line.split("=", maxsplit=1)[1])
        for line in runner.metadata_text.splitlines()
        if line.startswith(("START=", "END="))
    ]
    assert metadata_bounds[::2] == list(boundaries[:-1])
    assert metadata_bounds[1::2] == list(boundaries[1:])


def test_mixed_sample_rates_are_rejected_before_any_subprocess(tmp_path: Path) -> None:
    runner = MockRunner(probe_payload=_probe())
    mixed = (_audio()[0], replace(_audio()[1], sample_rate_hz=22_050))
    with pytest.raises(kk.EncodingError) as caught:
        _assembler(runner).assemble(
            _request(_plan(), mixed, tmp_path / "mixed.m4b")
        )
    assert caught.value.code == kk.ErrorCode.ASSEMBLY_FAILED
    assert not runner.calls


def test_coordinator_publishes_only_after_mocked_probe_and_decode(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pipeline, _, texts = _pipeline(tmp_path)
    durations = [len(text) * 10 for text in texts]
    total = sum(durations)
    payload: dict[str, object] = {
        "format": {
            "format_name": "mov,mp4,m4a,3gp,3g2,mj2",
            "duration": str(total / 1_000),
        },
        "streams": [
            {
                "codec_type": "audio",
                "codec_name": "aac",
                "sample_rate": "16000",
                "channels": 1,
            }
        ],
        "chapters": [
            {
                "start_time": "0",
                "end_time": str(durations[0] / 1_000),
                "tags": {"title": "One"},
            },
            {
                "start_time": str(durations[0] / 1_000),
                "end_time": str(total / 1_000),
                "tags": {"title": "Two"},
            },
        ],
    }
    runner = MockRunner(probe_payload=payload)
    _bind(monkeypatch, EngineSpecification.fake(), _assembler(runner))
    output = tmp_path / "coordinated.m4b"

    result = pipeline.write_m4b(output)

    assert result.output == output
    assert result.stats.duration_ms == total
    assert output.read_bytes() == b"mock-m4b"
    assert [MockRunner._operation(call) for call, _ in runner.calls][-3:] == [
        "encode",
        "probe",
        "decode",
    ]


@pytest.mark.parametrize("oversized", ["stdout", "stderr"])
def test_subprocess_runner_drains_but_rejects_oversized_output(
    monkeypatch: pytest.MonkeyPatch, oversized: str
) -> None:
    monkeypatch.setattr(native_module, "_STDOUT_LIMIT_BYTES", 4096)
    monkeypatch.setattr(native_module, "_STDERR_LIMIT_BYTES", 4096)
    statement = (
        "import sys; "
        f"sys.{oversized}.buffer.write(b'secret-output' * 10000); "
        f"sys.{oversized}.flush()"
    )
    with pytest.raises(kk.EncodingError) as caught:
        run_checked(
            SubprocessRunner(),
            (sys.executable, "-c", statement),
            timeout=5,
            code=kk.ErrorCode.ENCODING_FAILED,
        )
    assert caught.value.code == kk.ErrorCode.ENCODING_FAILED
    assert caught.value.__cause__ is None
    assert "secret-output" not in str(caught.value)


def test_bounded_capture_handles_partial_reads_and_never_retains_over_cap() -> None:

    class PartialPipe:
        def __init__(self) -> None:
            self.parts = iter((b"abc", b"defgh", b"i", b""))
            self.closed = False

        def read(self, _size: int) -> bytes:
            return next(self.parts)

        def close(self) -> None:
            self.closed = True

    pipe = PartialPipe()
    capture = native_module._BoundedCapture(7)
    capture.drain(pipe)  # type: ignore[arg-type]
    assert capture.data == b"abcdefg"
    assert capture.overflow
    assert pipe.closed


@pytest.mark.parametrize("reap_times_out", [False, True])
def test_subprocess_timeout_kills_and_uses_bounded_reap(
    monkeypatch: pytest.MonkeyPatch, reap_times_out: bool
) -> None:
    class TimedOutPopen:
        instance: TimedOutPopen

        def __init__(self, command: Sequence[str], **_kwargs: object) -> None:
            self.command = command
            self.waits: list[float] = []
            self.killed = False
            self.stdout = io.BytesIO()
            self.stderr = io.BytesIO()
            type(self).instance = self

        def wait(self, *, timeout: float) -> int:
            self.waits.append(timeout)
            if len(self.waits) == 1 or reap_times_out:
                raise subprocess.TimeoutExpired(self.command, timeout)
            return -9

        def kill(self) -> None:
            self.killed = True

    monkeypatch.setattr("kenkui._audio.native.subprocess.Popen", TimedOutPopen)
    with pytest.raises(kk.EncodingError) as caught:
        run_checked(
            SubprocessRunner(),
            ("mock",),
            timeout=0.01,
            code=kk.ErrorCode.ENCODING_FAILED,
        )
    process = TimedOutPopen.instance
    assert process.killed
    assert len(process.waits) == 2
    assert all(wait > 0 for wait in process.waits)
    assert caught.value.code == kk.ErrorCode.ENCODING_FAILED
    assert caught.value.__cause__ is None


@pytest.mark.parametrize("delta", [-2, 2])
def test_pcm_part_disagreeing_with_metadata_is_rejected_before_any_subprocess(
    tmp_path: Path, delta: int
) -> None:
    """A spilled part is untrusted: its size must match the metadata exactly."""
    runner = MockRunner(probe_payload=_probe())
    output = tmp_path / "tampered.m4b"
    request = _request(_plan(), _audio(), output)
    part = request.pcm_parts[0]
    payload = part.read_bytes()
    part.write_bytes(payload[:delta] if delta < 0 else payload + b"\0" * delta)

    with pytest.raises(kk.EncodingError) as caught:
        _assembler(runner).assemble(request)
    assert caught.value.code == kk.ErrorCode.ASSEMBLY_FAILED
    assert not runner.calls


def test_missing_pcm_part_is_rejected_before_any_subprocess(tmp_path: Path) -> None:
    """A part that vanished between render and assembly fails closed."""
    runner = MockRunner(probe_payload=_probe())
    request = _request(_plan(), _audio(), tmp_path / "missing.m4b")
    request.pcm_parts[0].unlink()

    with pytest.raises(kk.EncodingError) as caught:
        _assembler(runner).assemble(request)
    assert caught.value.code == kk.ErrorCode.ASSEMBLY_FAILED
    assert not runner.calls
