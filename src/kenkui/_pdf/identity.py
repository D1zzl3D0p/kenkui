"""Versioned PDF request and output identities independent of filesystem paths."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from typing import TYPE_CHECKING

from .recipe import PdfRecipeStep, PdfStep

if TYPE_CHECKING:
    from .models import PdfDocument

PREPARATION_SCHEMA = "pdf-preparation-v1"


def _digest(value: object) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def preparation_key(
    source_hash: str,
    mode: str,
    steps: tuple[PdfRecipeStep, ...],
    language: str,
    resources: tuple[tuple[str, str], ...],
) -> str | None:
    """Identify versioned requests, abstaining for unversioned custom callables."""
    if any(not isinstance(step, PdfStep) for step in steps):
        return None
    descriptors = [
        {
            "id": step.id,
            "version": step.version,
            "configuration": sorted(step.configuration),
            "requires": sorted(step.requires),
            "after": sorted(step.after),
        }
        for step in steps
        if isinstance(step, PdfStep)
    ]
    return _digest(
        {
            "schema": PREPARATION_SCHEMA,
            "source_hash": source_hash,
            "mode": mode,
            "steps": descriptors,
            "language": language,
            "resources": sorted(resources),
        }
    )


def prepared_identity(request_key: str, document: PdfDocument) -> str:
    """Bind a request to actual narration and source-supported structure.

    The inspection adapter's schema is versioned by the preparation request.
    Geometry/source IDs are included because they can affect derived structure;
    timing and diagnostic messages do not affect spoken output.
    """
    return _digest(
        {
            "request": request_key,
            "source_hash": document.source_hash,
            "language": document.language,
            "title": document.title,
            "author": document.author,
            "narration": [asdict(block) for block in document.narration],
        }
    )
