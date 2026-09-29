"""Format dispatch over bounded source snapshots, independent of pipeline intent."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from ._epub.parser import inspect_epub
from ._source import snapshot_source
from .errors import ErrorCode, SourceError

if TYPE_CHECKING:
    from pathlib import Path

    from ._pdf.models import PdfDocument
    from ._pdf.options import PdfOptions
    from ._progress import EventEmitter
    from .cancellation import CancellationToken
    from .inspection import BookInspection


@dataclass(frozen=True, slots=True)
class SourceSnapshot:
    """A copied source whose workspace remains owned by the calling shell."""

    path: Path
    format: str
    source_hash: str


def _validate_format(source_format: str) -> None:
    if source_format not in ("epub", "pdf"):
        raise SourceError(ErrorCode.UNSUPPORTED_FORMAT)


def inspect_document(path: Path, source_format: str) -> BookInspection:
    """Inspect an entire source without applying pipeline selection or tuning."""
    _validate_format(source_format)
    if source_format == "pdf":
        raise SourceError(ErrorCode.PDF_PREPARATION_REQUIRED)
    return inspect_epub(path)


def snapshot_document(
    path: Path,
    source_format: str,
    workspace: Path,
    cancel: CancellationToken | None,
) -> SourceSnapshot:
    """Copy into caller-owned storage and hash the exact bytes being inspected."""
    _validate_format(source_format)
    if cancel is not None:
        cancel.raise_if_cancelled()
    target = workspace / f"source.{source_format}"
    digest = snapshot_source(path, target, cancel)
    return SourceSnapshot(target, source_format, digest)


@dataclass(frozen=True, slots=True)
class PreparedSource:
    """A frozen full-source inspection, with separate byte and preparation identity."""

    source_hash: str
    inspection: BookInspection
    identity: str | None = None
    pdf_document: PdfDocument | None = None


def prepare_document(
    snapshot: SourceSnapshot,
    options: PdfOptions | None,
    cancel: CancellationToken | None,
    emitter: EventEmitter | None = None,
) -> PreparedSource:
    """Prepare the exact snapshot; no temporary paths escape this boundary."""
    if snapshot.format == "epub":
        return PreparedSource(
            snapshot.source_hash, inspect_document(snapshot.path, "epub")
        )
    from ._pdf.options import PdfOptions  # noqa: PLC0415 - optional format boundary
    from ._pdf.preparation import prepare_pdf  # noqa: PLC0415

    inspection, document, identity = prepare_pdf(
        snapshot.path, snapshot.source_hash, options or PdfOptions(), cancel, emitter
    )
    return PreparedSource(snapshot.source_hash, inspection, identity, document)
