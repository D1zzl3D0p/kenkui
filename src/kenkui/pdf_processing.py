"""Composable PDF cleanup over immutable source evidence.

Importing this module does not load a PDF parser or a layout/OCR model.
Native and optional layout extraction share the conservative transforms below.
"""

from ._pdf.models import (
    PdfBlock,
    PdfBox,
    PdfCharacter,
    PdfDocument,
    PdfEdit,
    PdfIssue,
    PdfLine,
    PdfPage,
    PdfRole,
)
from ._pdf.options import PdfLayoutOptions, PdfOptions
from ._pdf.recipe import PdfRecipeStep, PdfStep, PdfTransform, apply_steps
from ._pdf.steps import (
    omit_visual_material,
    reconstruct_paragraphs,
    remove_furniture,
    remove_note_sections,
    remove_notes,
    repair_line_break_words,
)

__all__ = [
    "PdfBlock",
    "PdfBox",
    "PdfCharacter",
    "PdfDocument",
    "PdfEdit",
    "PdfIssue",
    "PdfLayoutOptions",
    "PdfLine",
    "PdfOptions",
    "PdfPage",
    "PdfRecipeStep",
    "PdfRole",
    "PdfStep",
    "PdfTransform",
    "apply_steps",
    "omit_visual_material",
    "reconstruct_paragraphs",
    "remove_furniture",
    "remove_note_sections",
    "remove_notes",
    "repair_line_break_words",
]
