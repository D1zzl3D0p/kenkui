"""Extraction workers return bounded primitive records and are always reaped."""
# ruff: noqa: PLC0415

from __future__ import annotations

import json
import sys
from dataclasses import asdict
from typing import TYPE_CHECKING

import pytest

from kenkui._pdf.models import PdfBlock, PdfDocument, PdfPage
from kenkui._pdf.protocol import read_result, write_result
from kenkui._pdf.worker import run_worker
from kenkui.errors import ErrorCode, SourceError
from kenkui.pdf_processing import PdfLayoutOptions

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

    from kenkui.cancellation import CancellationToken


def _document() -> PdfDocument:
    block = PdfBlock("1:0", 1, "Complete prose.")
    return PdfDocument("a" * 64, (PdfPage(1, 600, 800, (block,)),), (block,))


def test_protocol_round_trip_and_source_binding(tmp_path: Path) -> None:
    """No backend objects or unvalidated identities cross the worker boundary."""
    path = tmp_path / "result.json"
    write_result(path, _document(), 100_000)
    assert read_result(path, "a" * 64, 100_000) == _document()
    with pytest.raises(SourceError):
        read_result(path, "b" * 64, 100_000)


@pytest.mark.parametrize(
    "change", ["unknown", "nonfinite", "page", "role", "truncated"]
)
def test_invalid_protocol_records_are_rejected(tmp_path: Path, change: str) -> None:
    """Reject malformed and semantically inconsistent worker output."""
    path = tmp_path / "result.json"
    write_result(path, _document(), 100_000)
    payload = json.loads(path.read_text())
    if change == "unknown":
        payload["document"]["unexpected"] = True
    elif change == "nonfinite":
        payload["document"]["pages"][0]["width"] = float("nan")
    elif change == "page":
        payload["document"]["narration"][0]["page"] = 0
    elif change == "role":
        payload["document"]["narration"][0]["role"] = "invented"
    path.write_text("{" if change == "truncated" else json.dumps(payload))
    with pytest.raises(SourceError):
        read_result(path, "a" * 64, 100_000)


def test_protocol_size_limits_apply_on_both_sides(tmp_path: Path) -> None:
    """A successful worker exit cannot bypass the result bound."""
    path = tmp_path / "result.json"
    with pytest.raises(SourceError):
        write_result(path, _document(), 10)
    path.write_bytes(b" " * 100)
    with pytest.raises(SourceError):
        read_result(path, "a" * 64, 10)


def test_worker_success_and_crash(tmp_path: Path) -> None:
    """A real child exits before its result becomes usable by the parent."""
    pytest.importorskip("psutil")
    result = tmp_path / "result.json"
    document = asdict(_document())
    script = (
        "import json,sys;json.dump({'schema':'kenkui-pdf-worker-v1','document':"
        + repr(document)
        + "},open(sys.argv[1],'w'))"
    )
    output = run_worker(
        [sys.executable, "-c", script, str(result)],
        result,
        "a" * 64,
        PdfLayoutOptions(timeout_seconds=10),
        None,
    )
    assert output == _document()
    with pytest.raises(SourceError) as failure:
        run_worker(
            [sys.executable, "-c", "raise SystemExit(2)"],
            result,
            "a" * 64,
            PdfLayoutOptions(timeout_seconds=10),
            None,
        )
    assert failure.value.code == ErrorCode.PDF_WORKER_FAILED


def test_worker_timeout_and_cancellation(tmp_path: Path) -> None:
    """Hung children are killed and waited for on every exceptional path."""
    from kenkui import CancellationToken, CancelledError

    pytest.importorskip("psutil")
    command = [sys.executable, "-c", "import time;time.sleep(30)"]
    with pytest.raises(SourceError) as failure:
        run_worker(
            command,
            tmp_path / "out.json",
            "a" * 64,
            PdfLayoutOptions(timeout_seconds=0.1),
            None,
        )
    assert failure.value.code == ErrorCode.PDF_TIMEOUT
    token = CancellationToken()
    token.cancel()
    with pytest.raises(CancelledError):
        run_worker(command, tmp_path / "out.json", "a" * 64, PdfLayoutOptions(), token)


@pytest.mark.parametrize(
    ("script", "limit", "expected"),
    [
        ("import time;time.sleep(30)", 1, ErrorCode.PDF_LIMIT),
        (
            "import sys;open(sys.argv[1],'w').write('{')",
            4096,
            ErrorCode.INVALID_PDF_OUTPUT,
        ),
        ("import sys;open(sys.argv[1],'w').write('x'*1000)", 4096, ErrorCode.PDF_LIMIT),
    ],
)
def test_worker_memory_and_output_failures(
    tmp_path: Path,
    script: str,
    limit: int,
    expected: ErrorCode,
) -> None:
    """Resource and protocol failures never yield a prepared document."""
    pytest.importorskip("psutil")
    result = tmp_path / "result.json"
    with pytest.raises(SourceError) as failure:
        run_worker(
            [sys.executable, "-c", script, str(result)],
            result,
            "a" * 64,
            PdfLayoutOptions(max_memory_mb=limit, max_output_bytes=500),
            None,
        )
    assert failure.value.code == expected


def test_running_worker_is_reaped_when_cancelled(tmp_path: Path) -> None:
    """Cancellation after spawning does not leave extraction consuming resources."""
    import threading

    from kenkui import CancellationToken, CancelledError

    pytest.importorskip("psutil")
    token = CancellationToken()
    timer = threading.Timer(0.2, token.cancel)
    pid_file = tmp_path / "pid"
    command = [
        sys.executable,
        "-c",
        (
            "import os,sys,time;open(sys.argv[1],'w').write(str(os.getpid()));"
            "time.sleep(30)"
        ),
        str(pid_file),
    ]
    timer.start()
    try:
        with pytest.raises(CancelledError):
            run_worker(
                command, tmp_path / "out.json", "a" * 64, PdfLayoutOptions(), token
            )
    finally:
        timer.cancel()
        timer.join()
    if pid_file.exists():
        import os

        with pytest.raises(ProcessLookupError):
            os.kill(int(pid_file.read_text()), 0)


def test_auto_mode_missing_assets_fails_before_model_import(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A provisioned package alone cannot trigger an implicit model download."""
    from kenkui._pdf.layout_runner import extract_in_worker
    from kenkui._pdf.options import PdfOptions

    monkeypatch.delenv("KENKUI_PDF_MODELS", raising=False)
    monkeypatch.setattr(
        "kenkui._pdf.layout_runner.importlib.util.find_spec", lambda _: object()
    )
    with pytest.raises(SourceError) as failure:
        extract_in_worker(tmp_path / "source.pdf", "a" * 64, PdfOptions(), None)
    assert failure.value.code == ErrorCode.PDF_ASSETS_MISSING


def test_child_missing_assets_has_a_sanitized_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Child exceptions preserve stable errors without exposing backend tracebacks."""
    from kenkui._pdf.layout_runner import extract_in_worker
    from kenkui._pdf.options import PdfOptions

    pytest.importorskip("psutil")
    monkeypatch.setattr(
        "kenkui._pdf.layout_runner.importlib.util.find_spec", lambda _: object()
    )
    with pytest.raises(SourceError) as failure:
        extract_in_worker(
            tmp_path / "source.pdf",
            "a" * 64,
            PdfOptions(layout=PdfLayoutOptions(artifacts_path=str(tmp_path))),
            None,
        )
    assert failure.value.code == ErrorCode.PDF_ASSETS_MISSING


def test_worker_reports_validated_stages_before_returning(tmp_path: Path) -> None:
    """Optional progress survives a quick worker exit and cannot inject stages."""
    pytest.importorskip("psutil")
    from kenkui._pdf.worker import _stage

    path = tmp_path / "progress.json"
    stages: list[str] = []
    assert _stage(path, None, stages.append) is None
    path.write_text(json.dumps("pdf.ocr"))
    assert _stage(path, None, stages.append) == "pdf.ocr"
    assert _stage(path, "pdf.ocr", stages.append) == "pdf.ocr"
    assert stages == ["pdf.ocr"]
    for value in ('"unexpected"', "{", " " * 300):
        path.write_text(value)
        with pytest.raises(SourceError):
            _stage(path, None, stages.append)


def test_child_success_and_sanitized_failures(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The entrypoint serializes evidence and never exposes exception details."""
    from kenkui._pdf import child

    request = tmp_path / "request.json"
    result = tmp_path / "result.json"
    progress = tmp_path / "progress.json"
    request.write_text(
        json.dumps(
            {
                "layout": asdict(PdfLayoutOptions()),
                "language": "en",
                "max_pages": 100,
                "max_characters": 10000,
                "path": "source.pdf",
                "source_hash": "a" * 64,
                "progress": str(progress),
            }
        )
    )
    monkeypatch.setattr(sys, "argv", ["child", str(request), str(result)])

    def extract(*args: object) -> PdfDocument:
        callback = args[-1]
        assert callable(callback)
        callback("pdf.ocr")
        return _document()

    monkeypatch.setattr(child, "extract_layout", extract)
    child.main()
    assert read_result(result, "a" * 64, 100_000) == _document()
    assert json.loads(progress.read_text()) == "pdf.ocr"
    for error in (FileNotFoundError("private path"), RuntimeError("private prose")):

        def fail(*_args: object, error: Exception = error) -> PdfDocument:
            raise error

        monkeypatch.setattr(child, "extract_layout", fail)
        child.main()
        assert "private" not in result.read_text()
        with pytest.raises(SourceError):
            read_result(result, "a" * 64, 100_000)


def test_public_layout_preparation_reaps_child_before_speech_execution(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Exercise real IPC and public preparation with a deterministic extractor."""
    import kenkui as kk
    from kenkui._audio.m4b import FakeArtifactAssembler
    from kenkui._execution.process_pool import EngineSpecification
    from kenkui._pdf import layout_runner
    from pdf_helpers import make_pdf
    from test_execution import _bind

    psutil = pytest.importorskip("psutil")
    pid_file = tmp_path / "worker.pid"
    script = """
import json, os, sys
from pathlib import Path
from dataclasses import replace
from kenkui._pdf.native import extract_native
from kenkui._pdf.options import PdfOptions
from kenkui._pdf.protocol import write_result
request = json.loads(Path(sys.argv[1]).read_text())
Path(sys.argv[3]).write_text(str(os.getpid()))
Path(request["progress"]).write_text(json.dumps("pdf.layout"))
document = extract_native(
    Path(request["path"]), request["source_hash"], PdfOptions(), None
)
document = replace(document, capabilities=(*document.capabilities, "layout_blocks"))
write_result(Path(sys.argv[2]), document, 1000000)
"""

    def worker(  # noqa: PLR0913 - matches the worker injection boundary
        command: list[str],
        result: Path,
        source_hash: str,
        limits: PdfLayoutOptions,
        cancel: CancellationToken | None,
        *,
        progress: Path,
        on_stage: Callable[[str], None],
    ) -> PdfDocument:
        return run_worker(
            [sys.executable, "-c", script, command[-2], command[-1], str(pid_file)],
            result,
            source_hash,
            limits,
            cancel,
            progress=progress,
            on_stage=on_stage,
        )

    monkeypatch.setattr(layout_runner, "run_worker", worker)
    monkeypatch.setattr(
        "kenkui._pdf.layout_runner.importlib.util.find_spec", lambda _: object()
    )
    _bind(monkeypatch, EngineSpecification.fake(), FakeArtifactAssembler())
    path = make_pdf(tmp_path / "book.pdf", ("Original prose remains.",))
    events: list[kk.ExecutionEvent] = []
    prepared = (
        kk.pdf(path)
        .pdf_processing(
            layout=PdfLayoutOptions(artifacts_path=str(tmp_path)),
        )
        .prepare(on_event=events.append)
    )
    assert not psutil.pid_exists(int(pid_file.read_text()))
    stages = [event.stage for event in events if isinstance(event, kk.StageCompleted)]
    assert "pdf.layout" in stages
    assert "pdf.cleanup" in stages
    assert "pdf.validation" in stages
    result = prepared.assign_voice("narrator").tts().write(tmp_path / "out.m4b")
    assert result.output.exists()
