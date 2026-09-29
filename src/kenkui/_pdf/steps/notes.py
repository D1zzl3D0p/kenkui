"""Omit corroborated native footnotes and their raised prose references together."""

from __future__ import annotations

import re
import statistics
from collections import Counter
from dataclasses import replace
from typing import TYPE_CHECKING

from kenkui._pdf.models import PdfEdit

if TYPE_CHECKING:
    from kenkui._pdf.models import PdfDocument, PdfLine, PdfPage

_NOTE = re.compile(r"^(\d{1,3})[.)]?\s+\S")
_NOTE_END = re.compile(r"[.!?][\u201d\u2019\"\')]*$")
_PRECEDING_WORD = re.compile(r"[^\W\d_]{2,}[\u201d\u2019\"'),.;:]*$")
_MATH_OR_CODE = re.compile(
    r"Math|Symbol|CM(?:MI|SY|EX)|Courier|Mono|Typewriter", re.IGNORECASE
)
_SIZE_RATIO = 0.82
_MIN_BODY_LINES = 3
_MIN_BODY_CHARACTERS = 20
_MIN_ALPHA_CHARACTERS = 4
_FONT_TOLERANCE = 0.5

# marker, inclusive start, exclusive end in the extracted line text
Reference = tuple[str, int, int]


def _references(line: PdfLine) -> tuple[Reference, ...]:
    """Align characters exactly, allowing only whitespace inserted by extraction."""
    chars = line.characters
    normal = [c for c in chars if c.text.isalpha()]
    if len(normal) < _MIN_ALPHA_CHARACTERS or any(len(c.text) != 1 for c in chars):
        return ()
    visible = [(i, c.text) for i, c in enumerate(chars) if not c.text.isspace()]
    textual = [(i, c) for i, c in enumerate(line.text) if not c.isspace()]
    if [c for _, c in visible] != [c for _, c in textual]:
        return ()
    mapping = {raw[0]: text[0] for raw, text in zip(visible, textual, strict=True)}
    size = statistics.median(c.size for c in normal)
    bottom = statistics.median(c.box[3] for c in normal)
    raw = "".join(c.text for c in chars)
    references = []
    for match in re.finditer(r"\d{1,3}", raw):
        start, end = match.span()
        if (
            _PRECEDING_WORD.search(line.text[: mapping[start]]) is not None
            and not (end < len(raw) and raw[end].isdigit())
            and all(
                c.size <= size * _SIZE_RATIO and c.box[3] <= bottom - size * 0.12
                for c in chars[start:end]
            )
            and not any(
                _MATH_OR_CODE.search(c.font) for c in chars[max(0, start - 2) : end]
            )
        ):
            references.append((match[0], mapping[start], mapping[end - 1] + 1))
    return tuple(references)


def _small_note(line: PdfLine, page: PdfPage, body_size: float) -> bool:
    return (
        line.box[1] >= page.height * 0.65
        and 0 < line.size <= body_size * _SIZE_RATIO
        and not _MATH_OR_CODE.search(line.font)
    )


def _note_groups(
    page: PdfPage,
    available: set[str],
) -> tuple[dict[str, tuple[Reference, ...]], dict[str, tuple[str, ...]]]:
    """Require body support, a separate footer zone, and an exact marker match."""
    body = [
        line
        for line in page.lines
        if page.height * 0.12 <= line.box[1] < page.height * 0.65
        and len(line.text) > _MIN_BODY_CHARACTERS
        and line.size > 0
    ]
    if len(body) < _MIN_BODY_LINES:
        return {}, {}
    size = statistics.median(line.size for line in body)
    references = {
        line.id: _references(line)
        for line in body
        if line.id in available and abs(line.size - size) < _FONT_TOLERANCE
    }
    markers = {marker for refs in references.values() for marker, _, _ in refs}
    anchors = [
        (i, match[1])
        for i, line in enumerate(page.lines)
        if _small_note(line, page, size)
        and (match := _NOTE.match(line.text)) is not None
    ]
    counts = Counter(marker for _, marker in anchors)
    groups = {}
    for index, marker in anchors:
        anchor = page.lines[index]
        if marker not in markers or counts[marker] != 1:
            continue
        if anchor.box[1] - max(line.box[3] for line in body) < size * 2:
            continue
        group = [anchor.id]
        previous = anchor
        for line in page.lines[index + 1 :]:
            if (
                not _small_note(line, page, size)
                or _NOTE.match(line.text) is not None
                or abs(line.size - anchor.size) >= _FONT_TOLERANCE
                or line.font != anchor.font
                or not 0 < line.box[1] - previous.box[1] < anchor.size * 1.8
                or abs(line.box[0] - anchor.box[0]) > anchor.size * 2
            ):
                break
            group.append(line.id)
            previous = line
        if set(group).issubset(available) and _NOTE_END.search(previous.text.rstrip()):
            groups[marker] = tuple(group)
    return references, groups


def _remove_native_notes(document: PdfDocument) -> PdfDocument:
    """Omit native footer notes only with a corresponding raised prose marker.

    This initial rule handles numbered notes on the same page. It intentionally
    retains ambiguous small text, baseline numbers, exponents, endnotes and
    potentially unfinished notes. Cross-page note chains are not followed.
    Run before paragraph reconstruction; edited or merged
    source blocks are left alone rather than remapped by fuzzy matching.
    """
    lines = {line.id: line for page in document.pages for line in page.lines}
    occurrences = Counter(
        source for block in document.narration for source in block.sources
    )
    available = {
        block.sources[0]
        for block in document.narration
        if len(block.sources) == 1
        and block.role in ("text", "note")
        and block.sources[0] in lines
        and block.text == lines[block.sources[0]].text
        and occurrences[block.sources[0]] == 1
    }
    omitted: set[str] = set()
    replacements: dict[str, tuple[Reference, ...]] = {}
    for page in document.pages:
        references, groups = _note_groups(page, available)
        omitted.update(source for group in groups.values() for source in group)
        for source, refs in references.items():
            selected = tuple(ref for ref in refs if ref[0] in groups)
            if selected:
                replacements[source] = selected
    blocks = []
    edits = list(document.edits)
    for block in document.narration:
        block_source = block.sources[0] if len(block.sources) == 1 else None
        if block_source in omitted:
            edits.append(
                PdfEdit(
                    "remove_notes",
                    (block.id,),
                    block.text,
                    "",
                    "small_footer_note_with_raised_prose_reference",
                )
            )
            continue
        text = block.text
        for _, start, end in reversed(replacements.get(block_source or "", ())):
            text = text[:start] + text[end:]
        if text != block.text:
            edits.append(
                PdfEdit(
                    "remove_notes",
                    (block.id,),
                    block.text,
                    text,
                    "raised_reference_to_omitted_note",
                )
            )
        blocks.append(replace(block, text=text) if text != block.text else block)
    return replace(document, narration=tuple(blocks), edits=tuple(edits))


def remove_notes(document: PdfDocument) -> PdfDocument:
    """Remove corroborated notes from native lines or exact layout projections."""
    if not any(block.origin == "layout" for block in document.narration):
        return _remove_native_notes(document)
    lines = {line.id: line for page in document.pages for line in page.lines}
    eligible = {
        block.id: block
        for block in document.narration
        if block.role in ("text", "note")
        and block.sources
        and all(source in lines for source in block.sources)
        and block.text
        == " ".join(lines[source].text.strip() for source in block.sources)
    }
    sources = {source for block in eligible.values() for source in block.sources}
    native_blocks = tuple(
        block for page in document.pages for block in page.blocks if block.id in sources
    )
    shadow = replace(document, narration=native_blocks)
    cleaned = _remove_native_notes(shadow)
    updated = {block.id: block.text for block in cleaned.narration}
    blocks = []
    edits = list(document.edits)
    for block in document.narration:
        if block.id not in eligible:
            blocks.append(block)
            continue
        text = " ".join(
            updated[source].strip() for source in block.sources if source in updated
        )
        if text != block.text:
            edits.append(
                PdfEdit(
                    "remove_notes",
                    (block.id,),
                    block.text,
                    text,
                    "native_note_and_reference_evidence_in_layout_block",
                )
            )
        if text:
            blocks.append(replace(block, text=text))
    return replace(document, narration=tuple(blocks), edits=tuple(edits))
