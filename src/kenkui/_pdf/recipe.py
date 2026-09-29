"""Ordered PDF transforms with explicit versions and prerequisite checks."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TypeAlias

from kenkui.errors import ErrorCode, ValidationError

from .models import PdfDocument

PdfTransform: TypeAlias = Callable[[PdfDocument], PdfDocument]


@dataclass(frozen=True, slots=True)
class PdfStep:
    """A callable transform with an explicit semantic identity for persistence.

    The version and configuration are the author's compatibility contract, not
    a fingerprint inferred from Python bytecode or a function's name.
    """

    id: str
    version: str
    function: PdfTransform = field(repr=False, compare=False)
    configuration: tuple[tuple[str, str], ...] = ()
    requires: tuple[str, ...] = ()
    after: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        """Reject incomplete or ambiguous semantic descriptors."""
        keys = [key for key, _ in self.configuration]
        if (
            not callable(self.function)
            or not self.id.strip()
            or not self.version.strip()
            or len(keys) != len(set(keys))
        ):
            raise ValidationError(ErrorCode.INVALID_PDF_RECIPE)

    def __call__(self, document: PdfDocument) -> PdfDocument:
        """Apply the same function available through the recipe runner."""
        return self.function(document)


PdfRecipeStep: TypeAlias = PdfStep | PdfTransform


def validate_recipe(
    steps: tuple[PdfRecipeStep, ...], capabilities: tuple[str, ...]
) -> None:
    """Check callable steps and declared dependencies before extraction or edits."""
    seen: set[str] = set()
    for step in steps:
        if not callable(step):
            raise ValidationError(ErrorCode.INVALID_PDF_RECIPE)
        if isinstance(step, PdfStep):
            if (
                step.id in seen
                or not set(step.after).issubset(seen)
                or not set(step.requires).issubset(capabilities)
            ):
                raise ValidationError(ErrorCode.INVALID_PDF_RECIPE)
            seen.add(step.id)


def apply_steps(document: PdfDocument, steps: tuple[PdfRecipeStep, ...]) -> PdfDocument:
    """Validate the entire recipe, then apply it without changing source evidence.

    Generic checks enforce an append-only audit trail. Each built-in transform
    must additionally validate its own omission, character or ordering evidence.
    """
    validate_recipe(steps, document.capabilities)
    current = document
    for step in steps:
        updated = step(current)
        if (
            not isinstance(updated, PdfDocument)
            or updated.source_hash != current.source_hash
            or updated.pages != current.pages
            or updated.language != current.language
            or updated.capabilities != current.capabilities
            or updated.title != current.title
            or updated.author != current.author
            or updated.resources != current.resources
            or updated.edits[: len(current.edits)] != current.edits
            or updated.issues[: len(current.issues)] != current.issues
            or (
                updated.narration != current.narration
                and len(updated.edits) == len(current.edits)
            )
        ):
            raise ValidationError(ErrorCode.INVALID_PDF_OUTPUT)
        current = updated
    return current
