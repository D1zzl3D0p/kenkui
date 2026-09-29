"""Optional Docling extraction; imported and executed only in the PDF child."""

from __future__ import annotations

import gc
import hashlib
import importlib.metadata
import itertools
import math
import os
from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

from kenkui.errors import ErrorCode, SourceError

from .layout import merge_layout
from .models import PdfBlock, PdfRole
from .native import extract_native

if TYPE_CHECKING:
    from collections.abc import Callable

    from .models import PdfDocument
    from .options import PdfOptions

_ROLES: dict[str, PdfRole] = {
    "text": "text",
    "paragraph": "text",
    "list_item": "text",
    "title": "heading",
    "section_header": "heading",
    "page_header": "header",
    "page_footer": "footer",
    "footnote": "note",
    "caption": "caption",
    "table": "table",
    "picture": "picture",
    "formula": "formula",
    "code": "code",
}
_OCR_IMAGE_COVERAGE = 0.7
_MODEL_FOLDER = "docling-project--docling-layout-heron"


def asset_identity(root: Path) -> str:
    """Require the selected local model and identify its exact contents."""
    model = root / _MODEL_FOLDER
    paths = (
        model / "config.json",
        model / "model.safetensors",
        model / "preprocessor_config.json",
    )
    if any(not path.is_file() for path in paths):
        raise SourceError(ErrorCode.PDF_ASSETS_MISSING)
    digest = hashlib.sha256()
    for path in sorted([*model.rglob("*"), *(root / "RapidOcr").rglob("*")]):
        if path.is_file() and ".cache" not in path.relative_to(root).parts:
            digest.update(str(path.relative_to(root)).encode())
            with path.open("rb") as handle:
                for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                    digest.update(chunk)
    return digest.hexdigest()


def _box(raw: dict[str, Any], height: float) -> tuple[float, float, float, float]:
    left, right, top, bottom = (float(raw[key]) for key in ("l", "r", "t", "b"))
    if raw.get("coord_origin", "BOTTOMLEFT").upper() == "BOTTOMLEFT":
        top, bottom = height - top, height - bottom
    if (
        not all(math.isfinite(x) for x in (left, right, top, bottom))
        or left > right
        or top > bottom
    ):
        raise ValueError
    return left, top, right, bottom


def _ordered_items(payload: dict[str, Any]) -> list[dict[str, Any]]:
    items = {
        item["self_ref"]: item
        for collection in ("texts", "tables", "pictures", "groups")
        for item in payload.get(collection, [])
    }
    pending = list(
        reversed(
            [
                *payload.get("body", {}).get("children", []),
                *payload.get("furniture", {}).get("children", []),
            ]
        )
    )
    seen: set[str] = set()
    ordered: list[dict[str, Any]] = []
    while pending:
        reference = pending.pop()["$ref"]
        if reference in seen:
            raise ValueError
        seen.add(reference)
        item = items[reference]
        pending.extend(reversed(item.get("children", [])))
        if item.get("prov"):
            ordered.append(item)
        elif item.get("text", "").strip():
            raise ValueError
    if any(
        item.get("text", "").strip() and key not in seen for key, item in items.items()
    ):
        raise ValueError
    return ordered


def _item_blocks(
    item: dict[str, Any],
    pages: dict[str, Any],
    mapping: dict[int, int],
    prefix: int,
) -> tuple[PdfBlock, ...]:
    # List provenance includes markers that Docling strips from normalized text.
    text = str(
        item.get("orig", item.get("text", ""))
        if item.get("label") == "list_item"
        else item.get("text", "")
    )
    accounted: set[int] = set()
    blocks = []
    for index, prov in enumerate(item["prov"]):
        number = int(prov["page_no"])
        start, end = prov.get("charspan", (0, len(text)))
        if not 0 <= start <= end <= len(text):
            raise ValueError
        positions = {i for i in range(start, end) if not text[i].isspace()}
        if positions & accounted:
            raise ValueError
        accounted.update(positions)
        height = float(pages[str(number)]["size"]["height"])
        blocks.append(
            PdfBlock(
                f"layout:{prefix}:{item['self_ref']}:{index}",
                mapping[number],
                text[start:end],
                role=_ROLES.get(item["label"], "text"),
                box=_box(prov["bbox"], height),
                origin="layout",
            )
        )
    if accounted != {i for i, char in enumerate(text) if not char.isspace()}:
        raise ValueError
    return tuple(blocks)


def _page_mapping(supplied: set[int], expected: tuple[int, ...]) -> dict[int, int]:
    if supplied == set(expected):
        return {number: number for number in expected}
    if supplied == set(range(1, len(expected) + 1)):
        return dict(enumerate(expected, 1))
    raise ValueError


def export_blocks(
    payload: dict[str, Any],
    expected_pages: tuple[int, ...],
) -> tuple[PdfBlock, ...]:
    """Map ordered Docling items and complete provenance spans to immutable blocks."""
    try:
        pages = payload["pages"]
        mapping = _page_mapping({int(number) for number in pages}, expected_pages)
        return tuple(
            block
            for item in _ordered_items(payload)
            for block in _item_blocks(item, pages, mapping, expected_pages[0])
        )
    except (KeyError, ValueError, TypeError, OverflowError, IndexError):
        raise SourceError(ErrorCode.PDF_EXTRACTION_INCOMPLETE) from None


def extract_layout(  # noqa: C901 - explicit offline setup and bounded page routing
    path: Path,
    source_hash: str,
    options: PdfOptions,
    progress: Callable[[str], None] | None = None,
) -> PdfDocument:
    """Run CPU layout and selective full-page OCR using already provisioned assets."""
    root = Path(options.layout.artifacts_path or "")
    identity = asset_identity(root)
    if progress is not None:
        progress("pdf.layout")
    # Set before importing either transformers or the Hugging Face client.
    os.environ.update(
        HF_HUB_OFFLINE="1",
        TRANSFORMERS_OFFLINE="1",
        TOKENIZERS_PARALLELISM="false",
        OMP_NUM_THREADS=str(options.layout.threads),
    )
    try:
        from docling.backend.pypdfium2_backend import (  # noqa: PLC0415
            PyPdfiumDocumentBackend,
        )
        from docling.datamodel.accelerator_options import (  # noqa: PLC0415
            AcceleratorDevice,
            AcceleratorOptions,
        )
        from docling.datamodel.base_models import InputFormat  # noqa: PLC0415
        from docling.datamodel.pipeline_options import (  # noqa: PLC0415
            PdfPipelineOptions,
            RapidOcrOptions,
        )
        from docling.document_converter import (  # noqa: PLC0415
            DocumentConverter,
            PdfFormatOption,
        )
    except ImportError:
        raise SourceError(ErrorCode.PDF_LAYOUT_UNAVAILABLE) from None
    if progress is not None:
        progress("pdf.extract")
    native = extract_native(path, source_hash, options, None)
    ocr_pages = frozenset(
        page.number
        for page in native.pages
        if page.image_coverage >= _OCR_IMAGE_COVERAGE
        or page.disposition == "unresolved"
    )
    if ocr_pages and not (root / "RapidOcr").is_dir():
        raise SourceError(ErrorCode.PDF_ASSETS_MISSING)
    converters: dict[bool, Any] = {}
    blocks: list[PdfBlock] = []
    for needs_ocr, group in itertools.groupby(
        native.pages, key=lambda page: page.number in ocr_pages
    ):
        numbers = [page.number for page in group]
        if progress is not None:
            progress("pdf.ocr" if needs_ocr else "pdf.layout")
        if needs_ocr not in converters:
            policy = PdfPipelineOptions(artifacts_path=root)
            policy.accelerator_options = AcceleratorOptions(
                num_threads=options.layout.threads,
                device=AcceleratorDevice.CPU,
            )
            policy.do_ocr = needs_ocr
            policy.do_table_structure = False
            policy.generate_page_images = needs_ocr
            policy.layout_batch_size = policy.ocr_batch_size = (
                policy.table_batch_size
            ) = 1
            policy.queue_max_size = 2
            policy.enable_remote_services = False
            policy.allow_external_plugins = False
            if needs_ocr:
                policy.ocr_options = RapidOcrOptions(
                    lang=[options.layout.ocr_language],
                    force_full_page_ocr=True,
                    backend="onnxruntime",
                )
            converters[needs_ocr] = DocumentConverter(
                format_options={
                    InputFormat.PDF: PdfFormatOption(
                        pipeline_options=policy,
                        backend=PyPdfiumDocumentBackend,
                    ),
                }
            )
        converter = converters[needs_ocr]
        for offset in range(0, len(numbers), options.layout.batch_pages):
            selected = tuple(numbers[offset : offset + options.layout.batch_pages])
            result = converter.convert(
                path,
                page_range=(selected[0], selected[-1]),
                max_num_pages=options.max_pages,
                raises_on_error=False,
            )
            if result.status.value != "success" or result.errors:
                raise SourceError(ErrorCode.PDF_EXTRACTION_INCOMPLETE)
            payload = cast("dict[str, Any]", result.document.export_to_dict())
            blocks.extend(export_blocks(payload, selected))
            if sum(len(block.text) for block in blocks) > options.max_characters:
                raise SourceError(ErrorCode.PDF_LIMIT)
            del result, payload
            gc.collect()
    document = merge_layout(native, tuple(blocks), ocr_pages)
    return replace(
        document,
        resources=(
            ("docling", importlib.metadata.version("docling")),
            ("pdfplumber", importlib.metadata.version("pdfplumber")),
            ("layout_model", identity),
            ("ocr_language", options.layout.ocr_language),
            ("adapter", "layout-v1"),
        ),
    )
