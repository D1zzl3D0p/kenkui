"""Shared title data for deterministic text and name partitioning."""

from __future__ import annotations

# Honorifics that PRECEDE a name separate individuals: "Mr Elliot" and "Miss
# Elliot" are two Elliots, "Mr Geary" and "Mrs Geary" a husband and wife who
# both speak. They also guard sentence splitting from treating title periods as
# sentence endings.
PREFIX_TITLES: frozenset[str] = frozenset(
    {
        "mr",
        "mrs",
        "miss",
        "ms",
        "master",
        "mistress",
        "lord",
        "lady",
        "sir",
        "dame",
        "dr",
        "doctor",
        "captain",
        "admiral",
        "colonel",
        "major",
        "general",
        "inspector",
        "sergeant",
        "king",
        "queen",
        "prince",
        "princess",
        "goodman",
        "goodwife",
        "mother",
        "father",
        "elder",
        "mayor",
    }
)
