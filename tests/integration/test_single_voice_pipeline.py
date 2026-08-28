"""Opt-in vertical acceptance for the local single-voice M4B path."""
# ruff: noqa: S603

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path

import json
import os
import shutil
import subprocess
from dataclasses import dataclass
from typing import cast
from zipfile import ZIP_STORED, ZipFile

import pytest

import kenkui as kk
from kenkui._audio.production import FFmpegM4BAssembler
from kenkui._execution.coordinator import ExecutionBindings
from kenkui._execution.process_pool import EngineSpecification

pytestmark = [
    pytest.mark.native,
    pytest.mark.skipif(
        os.environ.get("KENKUI_RUN_NATIVE") != "1",
        reason="set KENKUI_RUN_NATIVE=1 and select -m native explicitly",
    ),
]


@dataclass(frozen=True, slots=True)
class M4BProbe:
    """The chapter titles independently observed from a published M4B."""

    chapter_titles: tuple[str, ...]


def probe_m4b(output: Path) -> M4BProbe:
    """Probe chapter titles through the host FFprobe executable."""
    ffprobe = shutil.which("ffprobe")
    assert ffprobe is not None
    completed = subprocess.run(
        [
            ffprobe,
            "-v",
            "error",
            "-show_chapters",
            "-show_entries",
            "chapter_tags=title",
            "-of",
            "json",
            str(output),
        ],
        check=True,
        capture_output=True,
        timeout=30,
    )
    payload = cast("dict[str, object]", json.loads(completed.stdout))
    chapters = cast("list[dict[str, object]]", payload["chapters"])
    return M4BProbe(
        tuple(cast("dict[str, str]", chapter["tags"])["title"] for chapter in chapters)
    )


def _decode_m4b(output: Path) -> None:
    """Require the host FFmpeg executable to fully decode the M4B."""
    ffmpeg = shutil.which("ffmpeg")
    assert ffmpeg is not None
    subprocess.run(
        [ffmpeg, "-v", "error", "-i", str(output), "-f", "null", "-"],
        check=True,
        capture_output=True,
        timeout=60,
    )

def _chapter_document(title: str, sentence: str) -> str:
    """Return one minimal XHTML chapter for the fixture EPUB."""
    return (
        '<html xmlns="http://www.w3.org/1999/xhtml">'
        f"<head><title>{title}</title></head>"
        f"<body><h1>{title}</h1><p>{sentence}</p></body></html>"
    )


def _fixture_epub(path: Path) -> Path:
    """Write the minimal fixed EPUB that exercises ordered chapter assembly."""
    container = """<?xml version="1.0"?>
<container xmlns="urn:oasis:names:tc:opendocument:xmlns:container"
 version="1.0">
 <rootfiles><rootfile full-path="OPS/package.opf"
  media-type="application/oebps-package+xml"/></rootfiles>
</container>"""
    package = """<?xml version="1.0"?>
<package xmlns="http://www.idpf.org/2007/opf" version="3.0">
 <metadata xmlns:dc="http://purl.org/dc/elements/1.1/">
  <dc:title>Fixture</dc:title><dc:creator>Fixture Author</dc:creator>
 </metadata>
 <manifest>
  <item id="one" href="text/one.xhtml" media-type="application/xhtml+xml"/>
  <item id="two" href="text/two.xhtml" media-type="application/xhtml+xml"/>
 </manifest>
 <spine><itemref idref="one"/><itemref idref="two"/></spine>
</package>"""
    chapters = {
        "one": _chapter_document("One", "First fixture sentence."),
        "two": _chapter_document("Two", "Second fixture sentence."),
    }
    with ZipFile(path, "w") as archive:
        archive.writestr("mimetype", "application/epub+zip", compress_type=ZIP_STORED)
        archive.writestr("META-INF/container.xml", container)
        archive.writestr("OPS/package.opf", package)
        for name, chapter in chapters.items():
            archive.writestr(f"OPS/text/{name}.xhtml", chapter)
    return path


def test_fixture_epub_to_single_voice_m4b(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A fixture EPUB publishes decodable chapters in source order."""
    fixture_epub = _fixture_epub(tmp_path / "fixture.epub")
    bindings = ExecutionBindings(
        EngineSpecification.fake(),
        FFmpegM4BAssembler(),
        kk.Voice(
            id="fixture-voice",
            name="Fixture Voice",
            enabled=True,
            provenance="bundled local metadata",
            license_id="CC0-1.0",
            commercial_use_allowed=True,
            language="en",
            content_fingerprint="0" * 64,
            compatible_model_revisions=("fixture-v1",),
        ),
        "fixture-v1",
    )
    monkeypatch.setattr("kenkui.pipeline._execution_bindings", lambda: bindings)
    events: list[kk.ExecutionEvent] = []

    result = (
        kk.epub(fixture_epub)
        .assign_voice("fixture-voice")
        .tts()
        .write(tmp_path / "fixture.m4b", on_event=events.append, workers=1)
    )

    assert probe_m4b(result.output).chapter_titles == ("One", "Two")
    _decode_m4b(result.output)
    assert result.stats.normalized_speech_characters > 0
    assert type(events[-1]).__name__ == "Completed"
