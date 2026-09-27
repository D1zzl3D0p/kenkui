"""Immutable PDF extraction policy, separate from downstream speech intent."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal

from kenkui.errors import ErrorCode, ValidationError

if TYPE_CHECKING:
    from .recipe import PdfRecipeStep


@dataclass(frozen=True, slots=True)
class PdfLayoutOptions:
    """Local layout assets and limits for the isolated extraction process."""

    artifacts_path: str | None = None
    ocr_language: str = "en"
    timeout_seconds: float = 1800
    max_memory_mb: int = 4096
    max_output_bytes: int = 128 * 1024 * 1024
    threads: int = 2
    batch_pages: int = 16

    def __post_init__(self) -> None:
        """Reject invalid limits before any worker or model is started."""
        integers = (
            self.max_memory_mb,
            self.max_output_bytes,
            self.threads,
            self.batch_pages,
        )
        if (
            any(type(value) is not int or value < 1 for value in integers)
            or isinstance(self.timeout_seconds, bool)
            or not isinstance(self.timeout_seconds, (int, float))
            or not math.isfinite(self.timeout_seconds)
            or self.timeout_seconds <= 0
            or not isinstance(self.ocr_language, str)
            or not self.ocr_language.isalpha()
            or (
                self.artifacts_path is not None
                and (
                    not isinstance(self.artifacts_path, str)
                    or not self.artifacts_path.strip()
                )
            )
        ):
            raise ValidationError(ErrorCode.INVALID_PDF_RECIPE)


@dataclass(frozen=True, slots=True)
class PdfOptions:
    """Choose extraction and optional cleanup without opening the source."""

    mode: Literal["auto", "native"] = "auto"
    steps: tuple[PdfRecipeStep, ...] | None = None
    language: str = "en"
    max_pages: int = 5000
    max_characters: int = 20_000_000
    layout: PdfLayoutOptions = field(default_factory=PdfLayoutOptions)

    def __post_init__(self) -> None:
        """Validate the cheap, source-independent policy."""
        if (
            self.mode not in ("auto", "native")
            or not isinstance(self.language, str)
            or not self.language.strip()
            or type(self.max_pages) is not int
            or self.max_pages <= 0
            or type(self.max_characters) is not int
            or self.max_characters <= 0
            or not isinstance(self.layout, PdfLayoutOptions)
        ):
            raise ValidationError(ErrorCode.INVALID_PDF_RECIPE)
        if self.steps is not None:
            object.__setattr__(self, "steps", tuple(self.steps))
