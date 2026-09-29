"""PDF preparation over a caller-owned source snapshot."""

from __future__ import annotations

from dataclasses import replace
from importlib.metadata import version
from typing import TYPE_CHECKING

from .identity import preparation_key, prepared_identity
from .inspection import inspect_pdf_document
from .layout_runner import extract_in_worker
from .native import extract_native
from .recipe import PdfStep, apply_steps, validate_recipe
from .steps import (
    omit_visual_material,
    reconstruct_paragraphs,
    remove_furniture,
    remove_note_sections,
    remove_notes,
    repair_line_break_words,
)

if TYPE_CHECKING:
    from pathlib import Path

    from kenkui._progress import EventEmitter
    from kenkui.cancellation import CancellationToken
    from kenkui.inspection import BookInspection

    from .models import PdfDocument
    from .options import PdfOptions

_NATIVE_STEPS = (
    PdfStep("remove_furniture", "2", remove_furniture, requires=("native_lines",)),
    PdfStep(
        "remove_notes",
        "1",
        remove_notes,
        requires=("native_lines", "character_geometry"),
    ),
    PdfStep(
        "reconstruct_paragraphs",
        "2",
        reconstruct_paragraphs,
        requires=("native_lines",),
    ),
    PdfStep(
        "repair_line_break_words",
        "1",
        repair_line_break_words,
        requires=("native_lines",),
    ),
)


_LAYOUT_STEPS = (
    PdfStep("remove_furniture", "3", remove_furniture, requires=("native_lines",)),
    PdfStep("remove_notes", "2", remove_notes, requires=("native_lines",)),
    PdfStep(
        "remove_note_sections", "1", remove_note_sections, requires=("layout_blocks",)
    ),
    PdfStep(
        "omit_visual_material", "1", omit_visual_material, requires=("layout_blocks",)
    ),
    *_NATIVE_STEPS[2:],
)


def prepare_pdf(
    path: Path,
    source_hash: str,
    options: PdfOptions,
    cancel: CancellationToken | None,
    emitter: EventEmitter | None = None,
) -> tuple[BookInspection, PdfDocument, str]:
    """Extract and freeze a narration projection using only local resources."""
    default_steps = _NATIVE_STEPS if options.mode == "native" else _LAYOUT_STEPS
    steps = default_steps if options.steps is None else options.steps
    capabilities: tuple[str, ...] = ("native_lines", "character_geometry")
    if options.mode == "auto":
        capabilities += ("layout_blocks",)
    validate_recipe(steps, capabilities)
    if emitter is not None and options.mode == "native":
        emitter.emit_stage_started("pdf.extract")
    native = (
        extract_native(path, source_hash, options, cancel)
        if options.mode == "native"
        else extract_in_worker(path, source_hash, options, cancel, emitter)
    )
    if emitter is not None:
        if options.mode == "native":
            emitter.emit_stage_completed("pdf.extract")
        emitter.emit_stage_started("pdf.cleanup")
    if cancel is not None:
        cancel.raise_if_cancelled()
    document = apply_steps(native, steps)
    if emitter is not None:
        emitter.emit_stage_completed("pdf.cleanup")
        emitter.emit_stage_started("pdf.validation")
    inspection = inspect_pdf_document(document)
    resources = native.resources or (
        ("pdfplumber", version("pdfplumber")),
        ("adapter", "native-v1"),
    )
    request = preparation_key(
        source_hash, options.mode, steps, options.language, resources
    )
    # Unversioned callables have no persistent request identity. The actual
    # output remains identifiable for this explicitly prepared in-memory value.
    identity = prepared_identity(request or "uncacheable", document)
    if cancel is not None:
        cancel.raise_if_cancelled()
    if emitter is not None:
        emitter.emit_stage_completed("pdf.validation")
    return replace(inspection, _preparation_identity=identity), document, identity
