"""Repair confirmed line-break words without dictionaries or spelling guesses."""

from __future__ import annotations

import itertools
import re
from dataclasses import replace
from typing import TYPE_CHECKING

from kenkui._pdf.models import PdfEdit, PdfIssue

if TYPE_CHECKING:
    from kenkui._pdf.models import PdfBlock, PdfDocument, PdfLine

_WORD = r"[^\W\d_]+"
_TOKEN = re.compile(rf"(?<![\w-]){_WORD}(?:-{_WORD})*(?![\w-])")
_END = re.compile(rf"({_WORD})([-\u00ad])$")
_START = re.compile(rf"^({_WORD})")


def _patches(
    parts: tuple[str, ...],
    vocabulary: set[str],
) -> tuple[list[tuple[int, int, str]], bool]:
    """Locate separator-only changes against the exact original line projection."""
    patches: list[tuple[int, int, str]] = []
    ambiguous = False
    offset = 0
    for left, right in itertools.pairwise(parts):
        offset += len(left)
        end, start = _END.search(left), _START.match(right)
        if end is not None and start is not None and start[1][0].islower():
            whole = (end[1] + start[1]).casefold()
            compound = (end[1] + "-" + start[1]).casefold()
            if end[2] == "\u00ad":
                patches.append((offset - 1, offset + 1, ""))
            elif (whole in vocabulary) != (compound in vocabulary):
                patches.append(
                    (offset - 1, offset + 1, "" if whole in vocabulary else "-")
                )
            else:
                ambiguous = True
        offset += 1
    return patches, ambiguous


def _source_parts(block: PdfBlock, lines: dict[str, PdfLine]) -> tuple[str, ...]:
    """Require an unmodified source-line projection before using text offsets."""
    if block.role != "text" or len(block.sources) < 2:  # noqa: PLR2004
        return ()
    if any(source not in lines for source in block.sources):
        return ()
    parts = tuple(lines[source].text.strip() for source in block.sources)
    return parts if block.text == " ".join(parts) else ()


def repair_line_break_words(document: PdfDocument) -> PdfDocument:
    """Repair only proven boundaries inside already reconstructed prose.

    Hard hyphens require exactly one attested complete spelling in the original
    document. Soft hyphens explicitly mark discretionary breaks. Unknown and
    conflicting spellings are retained and reported. Original evidence and all
    letters are unchanged; this is not OCR spelling correction.
    """
    lines = {line.id: line for page in document.pages for line in page.lines}
    vocabulary = {
        match[0].casefold()
        for line in lines.values()
        for match in _TOKEN.finditer(line.text)
    }
    blocks = []
    edits = list(document.edits)
    issues = list(document.issues)
    for block in document.narration:
        parts = _source_parts(block, lines)
        patches, ambiguous = _patches(parts, vocabulary)
        text = block.text
        for start, end, replacement in reversed(patches):
            text = text[:start] + replacement + text[end:]
        blocks.append(replace(block, text=text) if patches else block)
        if patches:
            edits.append(
                PdfEdit(
                    "repair_line_break_words",
                    (block.id,),
                    block.text,
                    text,
                    "source_boundary_and_attested_spelling_or_soft_hyphen",
                )
            )
        if ambiguous:
            issue = PdfIssue(
                "ambiguous_line_break_word",
                "A line-break spelling lacks unambiguous source support; retained.",
                pages=(block.page,),
                blocks=(block.id,),
            )
            if issue not in issues:
                issues.append(issue)
    return replace(
        document, narration=tuple(blocks), edits=tuple(edits), issues=tuple(issues)
    )
