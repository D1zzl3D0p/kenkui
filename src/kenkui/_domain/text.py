"""Deterministic versioned speech-text normalization."""

from __future__ import annotations

import re
import unicodedata

NORMALIZATION_VERSION = "nfc-space-newline-v1"
# Unicode spacing characters intentionally mapped to an ordinary space.
SPACE_CODEPOINTS = frozenset(
    {
        "\u00a0",  # NO-BREAK SPACE
        "\u1680",  # OGHAM SPACE MARK
        "\u2000",
        "\u2001",
        "\u2002",
        "\u2003",
        "\u2004",
        "\u2005",
        "\u2006",
        "\u2007",
        "\u2008",
        "\u2009",
        "\u200a",
        "\u202f",  # NARROW NO-BREAK SPACE
        "\u205f",
        "\u3000",
        "\ufeff",  # BOM/zero-width no-break space
    }
)
_HORIZONTAL_RUN = re.compile(r"[^\S\n]+")
_MANY_NEWLINES = re.compile(r"\n{3,}")


def normalize_text(text: str) -> str:
    """Return NFC v1 speech text while preserving case and punctuation."""
    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    normalized = "".join(
        " " if char in SPACE_CODEPOINTS else char for char in normalized
    )
    normalized = unicodedata.normalize("NFC", normalized)
    lines = (
        _HORIZONTAL_RUN.sub(" ", line).strip(" ") for line in normalized.split("\n")
    )
    normalized = "\n".join(lines).strip()
    return _MANY_NEWLINES.sub("\n\n", normalized)
