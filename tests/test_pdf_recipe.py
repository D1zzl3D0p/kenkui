"""PDF transforms preserve evidence and bind identities to ordered recipes."""

from __future__ import annotations

from dataclasses import FrozenInstanceError, replace
from pathlib import Path
from typing import Literal

import pytest

from kenkui._pdf.identity import preparation_key, prepared_identity
from kenkui._pdf.models import PdfBlock, PdfDocument, PdfEdit, PdfPage
from kenkui._pdf.options import PdfOptions
from kenkui._pdf.preparation import prepare_pdf
from kenkui._pdf.recipe import PdfStep, apply_steps
from kenkui.errors import ValidationError


def _document() -> PdfDocument:
    block = PdfBlock("p1-b1", 1, "A broken-\nword.")
    return PdfDocument("a" * 64, (PdfPage(1, 100, 200, (block,)),), (block,))


def _join(document: PdfDocument) -> PdfDocument:
    before = document.narration[0]
    after = replace(before, text=before.text.replace("-\n", ""))
    if before == after:
        return document
    return replace(
        document,
        narration=(after,),
        edits=(*document.edits, PdfEdit("join", (before.id,), before.text, after.text)),
    )


def test_step_returns_new_narration_and_preserves_evidence() -> None:
    """A transformed projection must not overwrite the extracted original."""
    original = _document()
    step = PdfStep("join", "1", _join)
    result = apply_steps(original, (step,))
    assert original.narration[0].text == "A broken-\nword."
    assert result.narration[0].text == "A brokenword."
    assert result.pages == original.pages
    assert apply_steps(result, (step,)) == result
    with pytest.raises(FrozenInstanceError):
        result.language = "fr"  # type: ignore[misc]


def test_recipe_cannot_replace_archived_evidence() -> None:
    """Even an otherwise valid custom step cannot quietly alter source records."""

    def corrupt(document: PdfDocument) -> PdfDocument:
        return replace(document, pages=())

    with pytest.raises(ValidationError):
        apply_steps(_document(), (PdfStep("corrupt", "1", corrupt),))


def test_unlogged_text_change_is_rejected() -> None:
    """The projection's changed text must have edit evidence."""

    def corrupt(document: PdfDocument) -> PdfDocument:
        return replace(document, narration=())

    with pytest.raises(ValidationError):
        apply_steps(_document(), (PdfStep("corrupt", "1", corrupt),))


def test_empty_recipe_is_literal() -> None:
    """No optional cleanup occurs when the caller chooses zero steps."""
    document = _document()
    assert apply_steps(document, ()) == document


def test_recipe_validates_requirements_before_any_step_runs() -> None:
    """Reject an invalid recipe before invoking custom functions."""
    calls: list[str] = []

    def record(document: PdfDocument) -> PdfDocument:
        calls.append("called")
        return document

    with pytest.raises(ValidationError):
        apply_steps(
            _document(),
            (
                PdfStep("first", "1", record),
                PdfStep("second", "1", record, requires=("character_geometry",)),
            ),
        )
    assert calls == []


def test_identity_tracks_configuration_version_order_and_output() -> None:
    """Raw input bytes cannot authorize reuse after a recipe or text change."""
    document = _document()
    one = PdfStep("one", "1", _join)
    two = PdfStep("two", "1", _join)
    base = preparation_key(document.source_hash, "native", (one, two), "en", ())
    assert base is not None
    variants = (
        preparation_key(document.source_hash, "native", (two, one), "en", ()),
        preparation_key(
            document.source_hash, "native", (replace(one, version="2"), two), "en", ()
        ),
        preparation_key(document.source_hash, "native", (one, two), "fr", ()),
        preparation_key(document.source_hash, "auto", (one, two), "en", ()),
        preparation_key(
            document.source_hash, "native", (one, two), "en", (("backend", "2"),)
        ),
    )
    assert len({base, *variants}) == len(variants) + 1
    assert prepared_identity(base, document) != prepared_identity(base, _join(document))
    assert preparation_key(document.source_hash, "native", (one, two), "en", ()) == base


def test_unversioned_custom_step_is_not_cacheable() -> None:
    """Executing a local callable does not imply portable cache identity."""
    assert apply_steps(_document(), (_join,)).narration[0].text == "A brokenword."
    assert preparation_key("a" * 64, "native", (_join,), "en", ()) is None


@pytest.mark.parametrize(
    ("step_id", "version", "configuration"),
    [
        ("", "1", ()),
        ("join", "", ()),
        ("join", "1", (("language", "en"), ("language", "fr"))),
    ],
)
def test_invalid_descriptors_fail_early(
    step_id: str, version: str, configuration: tuple[tuple[str, str], ...]
) -> None:
    """Portable recipes cannot contain ambiguous identifiers/configuration."""
    with pytest.raises(ValidationError):
        PdfStep(step_id, version, _join, configuration=configuration)


def test_order_and_duplicate_steps_are_validated() -> None:
    """Explicit dependencies are not silently reordered or run twice."""
    first = PdfStep("first", "1", _join)
    second = PdfStep("second", "1", _join, after=("first",))
    for steps in ((second, first), (first, first)):
        with pytest.raises(ValidationError):
            apply_steps(_document(), steps)
    assert (
        apply_steps(_document(), (first, second)).narration[0].text == "A brokenword."
    )


def test_prior_audit_cannot_be_replaced() -> None:
    """A later stage cannot conceal the history of earlier transformations."""

    def corrupt(document: PdfDocument) -> PdfDocument:
        return replace(document, edits=())

    with pytest.raises(ValidationError):
        apply_steps(_join(_document()), (corrupt,))


def test_descriptor_configuration_is_canonical() -> None:
    """Mapping order is irrelevant; values and source bytes are significant."""
    first = PdfStep("join", "1", _join, configuration=(("a", "1"), ("b", "2")))
    reordered = replace(first, configuration=tuple(reversed(first.configuration)))
    changed = replace(first, configuration=(("a", "3"), ("b", "2")))
    original = preparation_key("a" * 64, "native", (first,), "en", ())
    assert preparation_key("a" * 64, "native", (reordered,), "en", ()) == original
    assert preparation_key("a" * 64, "native", (changed,), "en", ()) != original
    assert preparation_key("b" * 64, "native", (first,), "en", ()) != original


@pytest.mark.parametrize("mode", ["native", "auto"])
def test_invalid_recipe_is_rejected_before_extraction(
    mode: Literal["native", "auto"],
) -> None:
    """Bad dependencies must not parse a file or start an expensive worker."""
    options = PdfOptions(
        mode=mode,
        steps=(PdfStep("invalid", "1", _join, requires=("unknown_capability",)),),
    )
    with pytest.raises(ValidationError):
        prepare_pdf(Path("does-not-exist.pdf"), "a" * 64, options, None)


def test_noncallable_recipe_is_rejected_before_running_valid_steps() -> None:
    """Runtime mistakes use the library error without partially applying a recipe."""
    calls: list[str] = []

    def record(document: PdfDocument) -> PdfDocument:
        calls.append("called")
        return document

    with pytest.raises(ValidationError):
        apply_steps(_document(), (record, None))  # type: ignore[arg-type]
    assert calls == []


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("max_pages", 1.5),
        ("max_pages", float("nan")),
        ("max_characters", 2.5),
        ("max_characters", True),
        ("language", None),
        ("layout", None),
    ],
)
def test_pdf_options_reject_invalid_runtime_limits(field: str, value: object) -> None:
    """Configuration errors do not escape as backend type/arithmetic failures."""
    with pytest.raises(ValidationError):
        replace(PdfOptions(), **{field: value})  # type: ignore[arg-type]


def test_step_requires_an_executable_function() -> None:
    """A callable descriptor must not hide a missing implementation."""
    with pytest.raises(ValidationError):
        PdfStep("missing", "1", None)  # type: ignore[arg-type]
