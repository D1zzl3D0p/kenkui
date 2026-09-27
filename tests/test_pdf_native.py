"""Native PDF preparation is lazy, bounded and consistent with spoken input."""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING

import pytest

import kenkui as kk
from kenkui._pdf.options import PdfOptions
from pdf_helpers import make_pdf

if TYPE_CHECKING:
    from pathlib import Path

    from kenkui.pdf_processing import PdfDocument

pytest.importorskip("pdfplumber")


def test_pdf_constructor_and_preparation_are_separate(tmp_path: Path) -> None:
    """Declaring intent opens nothing; inspection requires explicit preparation."""
    path = tmp_path / "source.PDF"
    book = kk.book(path).pdf_processing(mode="native")
    assert book.source.format == "pdf"
    make_pdf(path, ("A quiet voice.", "Another page."))
    with pytest.raises(kk.SourceError) as failure:
        book.inspect()
    assert failure.value.code == kk.ErrorCode.PDF_PREPARATION_REQUIRED
    prepared = book.prepare()
    inspection = prepared.inspect()
    assert inspection.chapters[0].text == "A quiet voice.\n\nAnother page."
    assert not inspection.metadata.cover_available
    assert len(prepared.pdf_report().pages) == 2  # noqa: PLR2004
    assert len(list(prepared.script())) > 0
    assert prepared.assign_voice("narrator").inspect() == inspection
    with pytest.raises(kk.SourceError):
        book.pdf_report()


def test_auto_mode_does_not_silently_downgrade(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The intermediate native-only implementation explicitly rejects auto mode."""
    monkeypatch.setattr(
        "kenkui._pdf.layout_runner.importlib.util.find_spec", lambda _: None
    )
    path = make_pdf(tmp_path / "book.pdf", ("Native text.",))
    with pytest.raises(kk.SourceError) as failure:
        kk.pdf(path).prepare()
    assert failure.value.code == kk.ErrorCode.PDF_LAYOUT_UNAVAILABLE


def test_native_does_not_render_pages(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Native mode cannot quietly call PDFium page rendering."""
    import pdfplumber.page  # noqa: PLC0415

    def forbidden(*_args: object, **_kwargs: object) -> None:
        pytest.fail("native extraction rasterized a page")

    monkeypatch.setattr(pdfplumber.page.Page, "to_image", forbidden)
    path = make_pdf(tmp_path / "book.pdf", ("Embedded text.",))
    assert (
        kk.pdf(path)
        .pdf_processing(mode="native", steps=())
        .prepare()
        .inspect()
        .chapters
    )


def test_unknown_graphic_page_blocks_narration(tmp_path: Path) -> None:
    """A page with unextracted drawing content cannot be treated as blank."""
    path = make_pdf(tmp_path / "book.pdf", ("",), graphic=True)
    with pytest.raises(kk.SourceError) as failure:
        kk.pdf(path).pdf_processing(mode="native").prepare()
    assert failure.value.code == kk.ErrorCode.PDF_EXTRACTION_INCOMPLETE


def test_true_blank_page_is_retained_in_inventory(tmp_path: Path) -> None:
    """Blank pages are accounted for without creating empty audiobook chapters."""
    path = make_pdf(tmp_path / "book.pdf", ("Prose.", ""))
    prepared = kk.pdf(path).pdf_processing(mode="native").prepare()
    assert prepared.pdf_report().pages[1].disposition == "blank"
    assert prepared.inspect().chapters[0].text == "Prose."


def test_policy_changes_invalidate_prepared_text(tmp_path: Path) -> None:
    """Changing a recipe requires preparing and reviewing a new source projection."""
    path = make_pdf(tmp_path / "book.pdf", ("Prose.",))
    prepared = kk.pdf(path).pdf_processing(mode="native").prepare()
    branch = prepared.pdf_processing(mode="native", steps=())
    with pytest.raises(kk.SourceError):
        branch.inspect()
    newer = branch.prepare()
    assert (
        newer.inspect()._preparation_identity  # noqa: SLF001
        != prepared.inspect()._preparation_identity  # noqa: SLF001
    )


def test_native_limits_and_malformed_input(tmp_path: Path) -> None:
    """Limits are enforced on actual parsed pages and text."""
    path = make_pdf(tmp_path / "book.pdf", ("One.", "Two."))
    base = kk.pdf(path)
    limited = replace(
        base,
        source=replace(base.source, pdf_options=PdfOptions(mode="native", max_pages=1)),
    )
    with pytest.raises(kk.SourceError) as failure:
        limited.prepare()
    assert failure.value.code == kk.ErrorCode.PDF_LIMIT
    path.write_bytes(b"not a pdf")
    with pytest.raises(kk.SourceError) as failure:
        kk.pdf(path).pdf_processing(mode="native").prepare()
    assert failure.value.code == kk.ErrorCode.MALFORMED_PDF


def test_epub_rejects_pdf_policy(tmp_path: Path) -> None:
    """A PDF-only setting must not be ignored on an EPUB source."""
    with pytest.raises(kk.ValidationError):
        kk.epub(tmp_path / "book.epub").pdf_processing(mode="native")


def test_pdf_write_prepares_before_planning(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The ordinary write path consumes the same prepared prose as inspection."""
    from kenkui._audio.m4b import FakeArtifactAssembler  # noqa: PLC0415
    from kenkui._execution.process_pool import EngineSpecification  # noqa: PLC0415
    from test_execution import _bind  # noqa: PLC0415

    _bind(monkeypatch, EngineSpecification.fake(), FakeArtifactAssembler())
    path = make_pdf(tmp_path / "book.pdf", ("A quiet voice.", "Another page."))
    pipeline = kk.pdf(path).pdf_processing(mode="native").assign_voice("narrator").tts()
    result = pipeline.write(tmp_path / "out.m4b")
    expected = pipeline.prepare().inspect().chapters[0].text
    assert result.stats.normalized_speech_characters == len(expected)
    assert result.output.exists()


def test_changed_pdf_checkpoint_cannot_be_rendered(tmp_path: Path) -> None:
    """New bytes require a new preparation, including after successful inspection."""
    path = make_pdf(tmp_path / "book.pdf", ("Old prose.",))
    prepared = kk.pdf(path).pdf_processing(mode="native").prepare()
    make_pdf(path, ("New prose.",))
    with pytest.raises(kk.SourceError) as failure:
        prepared.assign_voice("narrator").tts().resolve()
    assert failure.value.code == kk.ErrorCode.SOURCE_CHANGED


def test_scanned_body_with_embedded_header_is_not_complete(tmp_path: Path) -> None:
    """A small text layer must not hide an unread scanned page body."""
    path = make_pdf(tmp_path / "scan.pdf", ("Running header",), image=True)
    with pytest.raises(kk.SourceError) as failure:
        kk.pdf(path).pdf_processing(mode="native").prepare()
    assert failure.value.code == kk.ErrorCode.PDF_EXTRACTION_INCOMPLETE


def test_pdf_plan_identity_tracks_recipe_without_changing_source_hash(
    tmp_path: Path,
) -> None:
    """Same speech under different recipes cannot reuse an old semantic plan."""
    from kenkui._domain.planning import compile_execution_plan  # noqa: PLC0415
    from test_execution import _voice  # noqa: PLC0415

    path = make_pdf(tmp_path / "book.pdf", ("The same prose.",))
    base = kk.pdf(path).pdf_processing(mode="native").assign_voice("narrator").tts()
    prepared = base.prepare()
    other = base.pdf_processing(mode="native", steps=()).prepare()
    plans = [
        compile_execution_plan(
            item,
            item.inspect(),
            source_bytes_hash=item.pdf_report().source_hash,
            resolved_voice=_voice(),
            model_revision="fake-v1",
        )
        for item in (prepared, other)
    ]
    assert plans[0].source_bytes_hash == plans[1].source_bytes_hash
    assert plans[0].schema_versions.parser == "pdf-prepared-v1"
    assert plans[0].semantic_fingerprint != plans[1].semantic_fingerprint


def test_cancelled_preparation_returns_no_checkpoint(tmp_path: Path) -> None:
    """Cancellation at the stage callback stops before any PDF parsing."""
    path = make_pdf(tmp_path / "book.pdf", ("Prose.",))
    token = kk.CancellationToken()
    pipeline = kk.pdf(path).pdf_processing(mode="native")

    def cancel_on_stage(event: kk.ExecutionEvent) -> None:
        if isinstance(event, kk.StageStarted):
            token.cancel()

    with pytest.raises(kk.CancelledError):
        pipeline.prepare(cancel=token, on_event=cancel_on_stage)
    with pytest.raises(kk.SourceError):
        pipeline.inspect()


def test_default_recipe_cleans_extracted_words_and_notes(tmp_path: Path) -> None:
    """Real PDF characters feed cleanup, inspection, and an idempotent recipe."""
    from kenkui._pdf.preparation import _NATIVE_STEPS  # noqa: PLC0415
    from kenkui.pdf_processing import apply_steps  # noqa: PLC0415

    stream = b" ".join(
        (
            b"BT /F1 12 Tf 60 640 Td (The investment remains a sound invest-) Tj ET",
            b"BT /F1 12 Tf 60 624 Td (ment when the prose continues.) Tj ET",
            (
                b"BT /F1 12 Tf 60 600 Td (A claim) Tj /F1 8 Tf 4 Ts (2) Tj "
                b"/F1 12 Tf 0 Ts ( remains in the main prose.) Tj ET"
            ),
            b"BT /F1 8 Tf 60 132 Td (2 This is a footer note.) Tj ET",
            b"BT /F1 8 Tf 60 121 Td (Its continuation is also omitted.) Tj ET",
        )
    )
    path = make_pdf(tmp_path / "layout.pdf", ("",), streams=(stream,))
    prepared = kk.pdf(path).pdf_processing(mode="native").prepare()
    text = prepared.inspect().chapters[0].text
    assert "a sound investment when the prose continues." in text
    assert "A claim remains in the main prose." in text
    assert "footer note" not in text
    assert "continuation is also omitted" not in text
    report = prepared.pdf_report()
    assert apply_steps(report, _NATIVE_STEPS) == report
    raw = kk.pdf(path).pdf_processing(mode="native", steps=()).prepare()
    assert "footer note" in raw.inspect().chapters[0].text
    assert raw.pdf_report().pages == report.pages


def test_cancellation_at_cleanup_boundary_prevents_custom_transforms(
    tmp_path: Path,
) -> None:
    """A callback can stop preparation before user cleanup functions execute."""
    token = kk.CancellationToken()
    calls: list[str] = []

    def transform(document: PdfDocument) -> PdfDocument:
        calls.append("called")
        return document

    def on_event(event: kk.ExecutionEvent) -> None:
        if isinstance(event, kk.StageStarted) and event.stage == "pdf.cleanup":
            token.cancel()

    path = make_pdf(tmp_path / "cancel.pdf", ("Preserve this prose.",))
    pipeline = kk.pdf(path).pdf_processing(mode="native", steps=(transform,))
    with pytest.raises(kk.CancelledError):
        pipeline.prepare(cancel=token, on_event=on_event)
    assert calls == []
