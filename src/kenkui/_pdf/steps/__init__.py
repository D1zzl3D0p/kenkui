"""Conservative, evidence-based transforms over immutable PDF documents."""

from .furniture import remove_furniture
from .notes import remove_notes
from .paragraphs import reconstruct_paragraphs
from .sections import remove_note_sections
from .visuals import omit_visual_material
from .words import repair_line_break_words

__all__ = [
    "omit_visual_material",
    "reconstruct_paragraphs",
    "remove_furniture",
    "remove_note_sections",
    "remove_notes",
    "repair_line_break_words",
]
