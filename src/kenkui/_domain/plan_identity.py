"""Canonical persisted identity for resolved plans."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable

    from kenkui._domain.plan_models import _PlanMaterial


def fingerprint(material: _PlanMaterial, hash_utf8: Callable[[str], str]) -> str:
    """Canonicalize semantic fields without changing persistent cache identity."""

    def _string_identity(value: str) -> dict[str, int | str]:
        return {"characters": len(value), "sha256": hash_utf8(value)}

    def _optional_string_identity(value: str | None) -> dict[str, int | str] | None:
        return None if value is None else _string_identity(value)

    schemas = material.schemas
    voice = material.voice
    segments = material.segments
    output = material.output
    schema_versions: dict[str, object] = {
        "parser": schemas.parser,
        "normalization": schemas.normalization,
        "planning": schemas.planning,
        "render": schemas.render,
    }
    if schemas.spoken_form is not None:
        # Present only when the stage is active. Emitting an explicit null
        # would change the canonical JSON -- and so the fingerprint -- for
        # every pipeline that never asked for spoken form.
        schema_versions["spoken_form"] = schemas.spoken_form
    payload = {
        "schema_versions": schema_versions,
        "source_bytes_hash": material.source_bytes_hash,
        "model_revision": _string_identity(material.model_revision),
        "voice": {
            "id": _string_identity(voice.id),
            "name": _string_identity(voice.name),
            "content_fingerprint": voice.content_fingerprint,
            "language": _string_identity(voice.language),
            "provenance": _string_identity(voice.provenance),
            "license_id": _string_identity(voice.license_id),
            "commercial_use_allowed": voice.commercial_use_allowed,
            "compatible_model_revisions": [
                _string_identity(revision)
                for revision in voice.compatible_model_revisions
            ],
        },
        "segments": [
            {
                "id": _string_identity(segment.id),
                "chapter_id": _string_identity(segment.chapter_id),
                "ordinal": segment.ordinal,
                "content_hash": segment.content_hash,
                "character_count": segment.character_count,
                **(
                    {"source_kind": segment.source_kind}
                    if segment.source_kind != "source"
                    else {}
                ),
            }
            for segment in segments
        ],
        "output": {
            "title": _optional_string_identity(output.title),
            "author": _optional_string_identity(output.author),
            "chapters": [
                {
                    "id": _string_identity(chapter.id),
                    "source_index": chapter.source_index,
                    "title": _string_identity(chapter.title),
                    "speech_characters": chapter.speech_characters,
                }
                for chapter in output.chapters
            ],
            "cover": output.cover.value,
            "source_cover_available": output.source_cover_available,
        },
        "total_speech_characters": material.total,
    }
    if material.announcements:
        payload["chapter_announcements"] = [
            (a.chapter_id, a.text, a.kind, a.heading_end)
            for a in material.announcements
        ]
    if material.preparation_identity is not None:
        payload["preparation_identity"] = material.preparation_identity
    if any(material.trailing_silence):
        payload["trailing_silence_ms"] = list(material.trailing_silence)
    if output.cover_content_hash is not None:
        # Absent rather than null: an explicit null would change the
        # fingerprint of every pipeline that never supplied a cover file.
        payload["cover_content_hash"] = output.cover_content_hash
    canonical = json.dumps(
        payload,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hash_utf8(canonical)
