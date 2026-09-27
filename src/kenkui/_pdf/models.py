"""Immutable PDF evidence, narration projections and transformation records."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, TypeAlias

PdfRole: TypeAlias = Literal[
    "text",
    "heading",
    "header",
    "footer",
    "note",
    "caption",
    "table",
    "formula",
    "picture",
    "verse",
    "code",
]
PdfBox: TypeAlias = tuple[float, float, float, float]


@dataclass(frozen=True, slots=True)
class PdfCharacter:
    """A native character with geometry for reference-marker corroboration."""

    text: str
    box: PdfBox
    size: float
    font: str


@dataclass(frozen=True, slots=True)
class PdfLine:
    """One extracted line, before paragraph reconstruction or word repair."""

    id: str
    page: int
    text: str
    box: PdfBox
    size: float
    font: str
    characters: tuple[PdfCharacter, ...] = ()


@dataclass(frozen=True, slots=True)
class PdfBlock:
    """An ordered text block, linked to its original extraction evidence."""

    id: str
    page: int
    text: str
    role: PdfRole = "text"
    sources: tuple[str, ...] = ()
    box: PdfBox | None = None
    origin: Literal["native", "layout"] = "native"


@dataclass(frozen=True, slots=True)
class PdfPage:
    """Original page evidence, retained independently of spoken content."""

    number: int
    width: float
    height: float
    blocks: tuple[PdfBlock, ...]
    lines: tuple[PdfLine, ...] = ()
    image_coverage: float = 0.0
    disposition: Literal["text", "blank", "unresolved"] = "text"
    layout_blocks: tuple[PdfBlock, ...] = ()


@dataclass(frozen=True, slots=True)
class PdfEdit:
    """A transform's declared change; correctness needs step-specific checks."""

    step: str
    block_ids: tuple[str, ...]
    before: str
    after: str
    reason: str = ""


@dataclass(frozen=True, slots=True)
class PdfIssue:
    """An unresolved extraction or cleanup decision with source coordinates."""

    code: str
    message: str
    pages: tuple[int, ...] = ()
    blocks: tuple[str, ...] = ()
    severity: Literal["warning", "error"] = "warning"


@dataclass(frozen=True, slots=True)
class PdfDocument:
    """Immutable extraction archive plus a separately transformable narration."""

    source_hash: str
    pages: tuple[PdfPage, ...]
    narration: tuple[PdfBlock, ...]
    language: str = "en"
    capabilities: tuple[str, ...] = ()
    edits: tuple[PdfEdit, ...] = ()
    issues: tuple[PdfIssue, ...] = ()
    title: str | None = None
    author: str | None = None
    resources: tuple[tuple[str, str], ...] = ()
