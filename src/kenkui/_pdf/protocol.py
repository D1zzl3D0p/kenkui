"""Bounded JSON transport for immutable extraction records; never pickle IPC."""

from __future__ import annotations

import json
import math
import types
from dataclasses import asdict, fields, is_dataclass
from functools import lru_cache
from typing import (
    TYPE_CHECKING,
    Any,
    Literal,
    Union,
    get_args,
    get_origin,
    get_type_hints,
)

from kenkui.errors import ErrorCode, SourceError

from .models import PdfDocument

if TYPE_CHECKING:
    from pathlib import Path

SCHEMA = "kenkui-pdf-worker-v1"
_ALLOWED_ERRORS = frozenset(
    {
        ErrorCode.PDF_LAYOUT_UNAVAILABLE,
        ErrorCode.PDF_ASSETS_MISSING,
        ErrorCode.PDF_EXTRACTION_INCOMPLETE,
        ErrorCode.PDF_LIMIT,
        ErrorCode.MALFORMED_PDF,
        ErrorCode.PDF_ENCRYPTED,
        ErrorCode.PDF_PACKAGE_MISSING,
        ErrorCode.PDF_WORKER_FAILED,
    }
)


@lru_cache(maxsize=16)
def _record_types(record: type[Any]) -> dict[str, Any]:
    return get_type_hints(record)


def _decode(value: object, expected: Any) -> Any:  # noqa: ANN401, C901, PLR0911, PLR0912 - closed recursive type grammar
    origin, arguments = get_origin(expected), get_args(expected)
    if origin in (types.UnionType, Union):
        for option in arguments:
            try:
                return _decode(value, option)
            except (TypeError, ValueError):
                pass
        raise ValueError
    if origin is Literal:
        if value not in arguments:
            raise ValueError
        return value
    if origin is tuple:
        if not isinstance(value, list):
            raise ValueError
        if len(arguments) == 2 and arguments[1] is Ellipsis:  # noqa: PLR2004
            return tuple(_decode(item, arguments[0]) for item in value)
        if len(arguments) != len(value):
            raise ValueError
        return tuple(
            _decode(item, kind) for item, kind in zip(value, arguments, strict=True)
        )
    if isinstance(expected, type) and is_dataclass(expected):
        if not isinstance(value, dict) or set(value) != {
            f.name for f in fields(expected)
        }:
            raise ValueError
        hints = _record_types(expected)
        return expected(
            **{key: _decode(item, hints[key]) for key, item in value.items()}
        )
    if (
        expected is float
        and isinstance(value, (float, int))
        and not isinstance(value, bool)
        and math.isfinite(value)
    ):
        return float(value)
    if expected in (str, int, type(None)) and type(value) is expected:
        return value
    raise ValueError


def _validate(document: PdfDocument, source_hash: str) -> None:
    numbers = [p.number for p in document.pages]
    if document.source_hash != source_hash or numbers != list(
        range(1, len(numbers) + 1)
    ):
        raise ValueError
    blocks = [b for p in document.pages for b in (*p.blocks, *p.layout_blocks)]
    lines = [line for p in document.pages for line in p.lines]
    line_ids = {line.id for line in lines}
    if (
        any(p.width <= 0 or p.height <= 0 for p in document.pages)
        or len({b.id for b in blocks}) != len(blocks)
        or len({line.id for line in lines}) != len(lines)
        or any(b.page not in numbers for b in (*blocks, *document.narration))
        or any(line.page != p.number for p in document.pages for line in p.lines)
        or any(
            b.page != p.number
            for p in document.pages
            for b in (*p.blocks, *p.layout_blocks)
        )
        or len({b.id for b in document.narration}) != len(document.narration)
        or any(
            source not in line_ids for b in document.narration for source in b.sources
        )
    ):
        raise ValueError


def write_result(path: Path, document: PdfDocument, limit: int) -> None:
    """Write a size-bounded stream; incomplete results are never successful."""
    payload = {"schema": SCHEMA, "document": asdict(document)}
    total = 0
    with path.open("wb") as output:
        for part in json.JSONEncoder(ensure_ascii=False, allow_nan=False).iterencode(
            payload
        ):
            encoded = part.encode("utf-8")
            total += len(encoded)
            if total > limit:
                raise SourceError(ErrorCode.PDF_LIMIT)
            output.write(encoded)


def _payload_document(payload: object, source_hash: str) -> PdfDocument:
    if not isinstance(payload, dict) or payload.get("schema") != SCHEMA:
        raise ValueError
    if set(payload) == {"schema", "error"}:
        code = ErrorCode(payload["error"])
        if code not in _ALLOWED_ERRORS:
            raise ValueError
        raise SourceError(code)
    if set(payload) != {"schema", "document"}:
        raise ValueError
    document: PdfDocument = _decode(payload["document"], PdfDocument)
    _validate(document, source_hash)
    return document


def read_result(path: Path, source_hash: str, limit: int) -> PdfDocument:
    """Decode exact known types, then validate source identity and page inventory."""
    try:
        with path.open("rb") as handle:
            raw = handle.read(limit + 1)
        if len(raw) > limit:
            raise SourceError(ErrorCode.PDF_LIMIT)
        return _payload_document(json.loads(raw), source_hash)
    except (OSError, ValueError, TypeError, RecursionError, OverflowError):
        raise SourceError(ErrorCode.INVALID_PDF_OUTPUT) from None
