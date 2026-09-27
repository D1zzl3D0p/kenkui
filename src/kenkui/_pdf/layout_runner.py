"""Parent-side layout dispatch without importing models or PDF backends."""

from __future__ import annotations

import importlib.util
import json
import os
import sys
from dataclasses import asdict, replace
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import TYPE_CHECKING

from kenkui.errors import ErrorCode, SourceError

from .worker import run_worker

if TYPE_CHECKING:
    from kenkui._progress import EventEmitter
    from kenkui.cancellation import CancellationToken

    from .models import PdfDocument
    from .options import PdfOptions


def extract_in_worker(
    path: Path,
    source_hash: str,
    options: PdfOptions,
    cancel: CancellationToken | None,
    emitter: EventEmitter | None = None,
) -> PdfDocument:
    """Resolve local assets and reap the worker before returning its evidence."""
    if importlib.util.find_spec("docling") is None:
        raise SourceError(ErrorCode.PDF_LAYOUT_UNAVAILABLE)
    assets = options.layout.artifacts_path or os.environ.get("KENKUI_PDF_MODELS")
    if not assets or not Path(assets).is_dir():
        raise SourceError(ErrorCode.PDF_ASSETS_MISSING)
    layout = replace(options.layout, artifacts_path=str(Path(assets).resolve()))
    current_stage: str | None = None

    def on_stage(stage: str) -> None:
        nonlocal current_stage
        if emitter is not None:
            if current_stage is not None:
                emitter.emit_stage_completed(current_stage)
            emitter.emit_stage_started(stage)
        current_stage = stage

    with TemporaryDirectory(prefix="kenkui-pdf-layout-") as workspace:
        request, output = (
            Path(workspace) / "request.json",
            Path(workspace) / "result.json",
        )
        progress = Path(workspace) / "progress.json"
        request.write_text(
            json.dumps(
                {
                    "path": str(path.resolve()),
                    "source_hash": source_hash,
                    "language": options.language,
                    "max_pages": options.max_pages,
                    "max_characters": options.max_characters,
                    "layout": asdict(layout),
                    "progress": str(progress),
                }
            )
        )
        document = run_worker(
            [sys.executable, "-m", "kenkui._pdf.child", str(request), str(output)],
            output,
            source_hash,
            layout,
            cancel,
            progress=progress,
            on_stage=on_stage,
        )
        if emitter is not None and current_stage is not None:
            emitter.emit_stage_completed(current_stage)
        if (
            len(document.pages) > options.max_pages
            or sum(len(b.text) for b in document.narration) > options.max_characters
        ):
            raise SourceError(ErrorCode.PDF_LIMIT)
        return document
