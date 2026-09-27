"""Remove repeated edge furniture from the narration projection only."""

from __future__ import annotations

import re
from dataclasses import replace
from typing import TYPE_CHECKING

from kenkui._pdf.models import PdfEdit

if TYPE_CHECKING:
    from kenkui._pdf.models import PdfDocument

_MIN_REPEAT_PAGES = 3
_MIN_FOLIO_PAGES = 5
_EDGE_FRACTION = 0.12


_FOLIO = re.compile(r"^(?:(\d{1,4})(?:\s+(.+))?|(.+?)\s+(\d{1,4}))$")
_MIN_BODY_CHARACTERS = 20


def _folio_labels(document: PdfDocument) -> dict[str, str]:
    """Find endpoint numbers supported by a shared offset on distinct pages."""
    candidates: dict[str, tuple[str, int]] = {}
    offsets: dict[int, set[int]] = {}
    for page in document.pages:
        if not page.lines or not any(
            len(line.text) > _MIN_BODY_CHARACTERS
            and page.height * _EDGE_FRACTION <= line.box[1]
            and line.box[3] <= page.height * (1 - _EDGE_FRACTION)
            for line in page.lines
        ):
            continue
        for line in (page.lines[0], page.lines[-1]):
            edge = line.box[1] < page.height * _EDGE_FRACTION or (
                line.box[3] > page.height * (1 - _EDGE_FRACTION)
            )
            match = _FOLIO.fullmatch(line.text.strip()) if edge else None
            if match is not None:
                number = int(match[1] or match[4])
                label = match[2] or match[3] or ""
                offset = page.number - number
                candidates[line.id] = (label, offset)
                offsets.setdefault(offset, set()).add(page.number)
    return {
        source: label
        for source, (label, offset) in candidates.items()
        if len(offsets[offset]) >= _MIN_FOLIO_PAGES
    }


def remove_furniture(document: PdfDocument) -> PdfDocument:
    """Corroborate repeated edge text and endpoint folios across source pages."""
    keys: dict[str, tuple[str, str, int, int]] = {}
    support: dict[tuple[str, str, int, int], set[int]] = {}
    folios = _folio_labels(document)
    lines = {line.id: line for page in document.pages for line in page.lines}
    for page in document.pages:
        for line in page.lines:
            edge = line.box[1] < page.height * _EDGE_FRACTION or (
                line.box[3] > page.height * (1 - _EDGE_FRACTION)
            )
            text = re.sub(r"\s+", " ", folios.get(line.id, line.text)).strip()
            if edge and len(text) > 3 and not text.isdigit():  # noqa: PLR2004
                key = (
                    text.casefold(),
                    line.font,
                    round(line.size * 10),
                    round(line.box[1] / page.height * 100),
                )
                keys[line.id] = key
                support.setdefault(key, set()).add(page.number)
    removed: set[str] = set()
    edits = list(document.edits)
    for block in document.narration:
        if len(block.sources) != 1 or block.role not in ("text", "header", "footer"):
            continue
        source = block.sources[0]
        candidate_key = keys.get(source)
        repeated = (
            candidate_key is not None
            and len(support[candidate_key]) >= _MIN_REPEAT_PAGES
        )
        original = lines.get(source)
        folio = source in folios and not folios[source]
        unchanged = original is not None and block.text == original.text
        if unchanged and (repeated or folio):
            removed.add(block.id)
            edits.append(
                PdfEdit(
                    "remove_furniture",
                    (block.id,),
                    block.text,
                    "",
                    "repeated_edge" if repeated else "folio_sequence",
                )
            )
    return _remove_layout_furniture(
        replace(
            document,
            narration=tuple(b for b in document.narration if b.id not in removed),
            edits=tuple(edits),
        )
    )


def _remove_layout_furniture(document: PdfDocument) -> PdfDocument:
    """Require layout labels, edge placement and repeated geometry on source pages."""
    support: dict[tuple[str, str, int, int], set[int]] = {}
    keys: dict[str, tuple[str, str, int, int]] = {}
    originals = {}
    offsets: dict[int, set[int]] = {}
    for page in document.pages:
        for block in page.layout_blocks:
            if block.role not in ("header", "footer") or block.box is None:
                continue
            _, top, _, bottom = block.box
            if not (
                top < page.height * _EDGE_FRACTION
                or bottom > page.height * (1 - _EDGE_FRACTION)
            ):
                continue
            text = re.sub(r"\s+", " ", block.text).strip()
            key = (
                re.sub(r"\d+", "#", text.casefold()),
                block.role,
                round(top / page.height * 50),
                round((bottom - top) / page.height * 100),
            )
            keys[block.id] = key
            originals[block.id] = text
            support.setdefault(key, set()).add(page.number)
            if text.isdigit():
                offsets.setdefault(page.number - int(text), set()).add(page.number)
    removed = set()
    edits = list(document.edits)
    for block in document.narration:
        candidate_key = keys.get(block.id)
        original = originals.get(block.id)
        if candidate_key is None or original != re.sub(r"\s+", " ", block.text).strip():
            continue
        confirmed = len(support[candidate_key]) >= _MIN_REPEAT_PAGES
        if original.isdigit():
            confirmed = len(offsets[block.page - int(original)]) >= _MIN_FOLIO_PAGES
        if confirmed:
            removed.add(block.id)
            edits.append(
                PdfEdit(
                    "remove_furniture",
                    (block.id,),
                    block.text,
                    "",
                    "repeated_layout_furniture",
                )
            )
    return replace(
        document,
        narration=tuple(b for b in document.narration if b.id not in removed),
        edits=tuple(edits),
    )
