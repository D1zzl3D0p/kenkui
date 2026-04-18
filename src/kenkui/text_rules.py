"""Shared text-pattern helpers used across NLP and rendering layers."""

from __future__ import annotations

import re

SCENE_BREAK_RE = re.compile(
    r"^\s*(\*\s*){2,}\s*$"
    r"|^\s*[-\u2014]{2,}\s*$"
    r"|^\s*#\s*$",
)


def is_scene_break(text: str) -> bool:
    """Return True if *text* is a scene-break marker or pure whitespace."""
    stripped = text.strip()
    return not stripped or bool(SCENE_BREAK_RE.match(stripped))


def split_at_scene_breaks(paragraphs: list[str]) -> list[list[str]]:
    """Split paragraphs into groups separated by scene-break markers."""
    groups: list[list[str]] = []
    current: list[str] = []
    for para in paragraphs:
        if is_scene_break(para):
            if current:
                groups.append(current)
                current = []
        else:
            current.append(para)
    if current:
        groups.append(current)
    return groups or [[]]


__all__ = [
    "SCENE_BREAK_RE",
    "is_scene_break",
    "split_at_scene_breaks",
]
