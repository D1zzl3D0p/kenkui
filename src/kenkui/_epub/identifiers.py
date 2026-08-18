"""Stable EPUB semantic chapter identifiers."""

from __future__ import annotations

import hashlib

CHAPTER_ID_VERSION = "v1"


def chapter_id(member_path: str, occurrence: int, fragment: str) -> str:
    """Hash only versioned canonical semantic spine identity."""
    identity = f"{CHAPTER_ID_VERSION}\0{member_path}\0{occurrence}\0{fragment}"
    digest = hashlib.sha256(identity.encode("utf-8")).hexdigest()[:24]
    return f"ch-{CHAPTER_ID_VERSION}-{digest}"
