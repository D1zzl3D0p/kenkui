"""Offline backend tests exercise routing and limits without installing models."""
# ruff: noqa: ANN401, PLR2004

from __future__ import annotations

import sys
from dataclasses import replace
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any

import pytest

from kenkui._pdf.docling import asset_identity, extract_layout
from kenkui._pdf.options import PdfOptions
from kenkui.errors import ErrorCode, SourceError
from kenkui.pdf_processing import PdfBlock, PdfLayoutOptions
from test_pdf_steps import _document, _page

if TYPE_CHECKING:
    from pathlib import Path


def _assets(root: Path) -> None:
    model = root / "docling-project--docling-layout-heron"
    model.mkdir()
    for name in ("config.json", "preprocessor_config.json", "model.safetensors"):
        (model / name).write_text(name)
    (root / "RapidOcr").mkdir()


def test_assets_require_complete_local_model_and_track_changes(tmp_path: Path) -> None:
    """Missing assets fail early; model content participates in identity."""
    with pytest.raises(SourceError):
        asset_identity(tmp_path)
    _assets(tmp_path)
    before = asset_identity(tmp_path)
    (tmp_path / "RapidOcr" / "recognizer.onnx").write_bytes(b"changed model")
    assert before != asset_identity(tmp_path)


@pytest.mark.parametrize("failure", [None, "partial", "errors", "limit", "assets"])
def test_backend_routes_batches_and_rejects_incomplete_results(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failure: str | None
) -> None:
    """Native and scanned routes use offline CPU policies and account for pages."""
    _assets(tmp_path)
    pages = tuple(
        replace(_page(i, ()), disposition="unresolved", image_coverage=1)
        if i == 3
        else _page(i, (("Original prose.", 60, 150, 500, 12),))
        for i in range(1, 4)
    )
    raw = _document(*pages)
    calls = []
    policies = []

    def convert(_path: Path, **kwargs: Any) -> SimpleNamespace:
        calls.append(kwargs["page_range"])
        return SimpleNamespace(
            status=SimpleNamespace(
                value="partial_success" if failure == "partial" else "success"
            ),
            errors=["failure"] if failure == "errors" else [],
            document=SimpleNamespace(export_to_dict=dict),
        )

    def converter(**kwargs: Any) -> SimpleNamespace:
        policies.append(kwargs["format_options"]["pdf"].pipeline_options)
        return SimpleNamespace(convert=convert)

    modules: dict[str, dict[str, Any]] = {
        "docling.backend.pypdfium2_backend": {"PyPdfiumDocumentBackend": object},
        "docling.datamodel.accelerator_options": {
            "AcceleratorDevice": SimpleNamespace(CPU="cpu"),
            "AcceleratorOptions": SimpleNamespace,
        },
        "docling.datamodel.base_models": {"InputFormat": SimpleNamespace(PDF="pdf")},
        "docling.datamodel.pipeline_options": {
            "PdfPipelineOptions": SimpleNamespace,
            "RapidOcrOptions": SimpleNamespace,
        },
        "docling.document_converter": {
            "DocumentConverter": converter,
            "PdfFormatOption": SimpleNamespace,
        },
    }
    for name, values in modules.items():
        monkeypatch.setitem(sys.modules, name, SimpleNamespace(**values))
    monkeypatch.setattr("kenkui._pdf.docling.extract_native", lambda *_args: raw)
    monkeypatch.setattr(
        "kenkui._pdf.docling.importlib.metadata.version", lambda _: "test"
    )
    monkeypatch.setattr(
        "kenkui._pdf.docling.export_blocks",
        lambda _payload, selected: tuple(
            PdfBlock(
                f"layout:{i}",
                i,
                "Original prose.",
                box=(55, 145, 505, 180),
                origin="layout",
            )
            for i in selected
        ),
    )
    options = PdfOptions(
        max_characters=1 if failure == "limit" else 1000,
        layout=PdfLayoutOptions(artifacts_path=str(tmp_path), batch_pages=1),
    )
    if failure == "assets":
        (tmp_path / "RapidOcr").rmdir()
    stages: list[str] = []
    if failure:
        with pytest.raises(SourceError) as caught:
            extract_layout(
                tmp_path / "source.pdf", raw.source_hash, options, stages.append
            )
        expected = {
            "assets": ErrorCode.PDF_ASSETS_MISSING,
            "limit": ErrorCode.PDF_LIMIT,
        }.get(failure, ErrorCode.PDF_EXTRACTION_INCOMPLETE)
        assert caught.value.code == expected
        return
    result = extract_layout(
        tmp_path / "source.pdf", raw.source_hash, options, stages.append
    )
    assert calls == [(1, 1), (2, 2), (3, 3)]
    assert [policy.do_ocr for policy in policies] == [False, True]
    assert all(not policy.enable_remote_services for policy in policies)
    assert all(policy.accelerator_options.device == "cpu" for policy in policies)
    assert policies[1].ocr_options.force_full_page_ocr
    assert len(result.narration) == 3
    assert "pdf.ocr" in stages
