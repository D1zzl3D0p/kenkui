"""Thin service wrapper around series.py for use by the API layer.

Public API:
  list_series() -> ListSeriesResult
  load_series(slug: str) -> SeriesEntry  — raises KeyError if not found
  save_series(entry: SeriesEntry) -> None
  delete_series(slug: str) -> bool       — returns False if not found
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from .. import series as _series

if TYPE_CHECKING:
    from kenkui.nlp.models import CharacterRoster


# ---------------------------------------------------------------------------
# Return-type dataclasses
# ---------------------------------------------------------------------------


@dataclass
class SeriesCharacterEntry:
    canonical: str
    aliases: list[str] = field(default_factory=list)
    voice: str = ""
    gender: str = ""


@dataclass
class SeriesEntry:
    slug: str
    name: str
    updated_at: str = ""
    characters: list[SeriesCharacterEntry] = field(default_factory=list)


@dataclass
class ListSeriesResult:
    series: list[SeriesEntry]
    total: int


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _manifest_to_entry(manifest: _series.SeriesManifest) -> SeriesEntry:
    return SeriesEntry(
        slug=manifest.slug,
        name=manifest.name,
        updated_at=manifest.updated_at,
        characters=[
            SeriesCharacterEntry(
                canonical=c.canonical,
                aliases=list(c.aliases),
                voice=c.voice,
                gender=c.gender,
            )
            for c in manifest.characters
        ],
    )


def _entry_to_manifest(entry: SeriesEntry) -> _series.SeriesManifest:
    return _series.SeriesManifest(
        name=entry.name,
        slug=entry.slug,
        updated_at=entry.updated_at,
        characters=[
            _series.SeriesCharacter(
                canonical=c.canonical,
                aliases=list(c.aliases),
                voice=c.voice,
                gender=c.gender,
            )
            for c in entry.characters
        ],
    )


def _series_path(slug: str):
    return _series.series_dir() / f"{slug}.toml"


# ---------------------------------------------------------------------------
# Public service functions
# ---------------------------------------------------------------------------


def list_series() -> ListSeriesResult:
    """Return all series manifests as a ListSeriesResult."""
    manifests = _series.list_series()
    entries = [_manifest_to_entry(m) for m in manifests]
    return ListSeriesResult(series=entries, total=len(entries))


def load_series(slug: str) -> SeriesEntry:
    """Load a series manifest by slug.

    Raises KeyError(slug) if not found.
    """
    manifest = _series.load_series(slug)
    if manifest is None:
        raise KeyError(slug)
    return _manifest_to_entry(manifest)


def save_series(entry: SeriesEntry) -> None:
    """Persist a SeriesEntry to disk via the underlying series.py save."""
    manifest = _entry_to_manifest(entry)
    _series.save_series(manifest)


def delete_series(slug: str) -> bool:
    """Delete the series manifest file for *slug*.

    Returns True if the file existed and was deleted; False if not found.
    """
    path = _series_path(slug)
    try:
        path.unlink()
        return True
    except FileNotFoundError:
        return False


def get_roster(series_slug: str) -> "CharacterRoster":
    """Return the cumulative CharacterRoster for *series_slug*.

    Returns an empty roster for the first book in a series (no prior roster file).
    """
    from kenkui.nlp.models import CharacterRoster

    existing = _series.load_series_roster(series_slug)
    return existing if existing is not None else CharacterRoster(characters=[])


def update_roster(series_slug: str, new_roster: "CharacterRoster", book_slug: str) -> None:
    """Merge *new_roster* into the series roster and persist.

    Existing characters are updated (new aliases unioned, counts summed,
    last_appearance advanced). New characters are appended with
    ``first_appearance = (book_slug, first_chapter_index)``.
    """
    existing = get_roster(series_slug)
    merged = _series.merge_into_series_roster(existing, new_roster, book_slug)
    _series.save_series_roster(series_slug, merged)


__all__ = [
    "SeriesCharacterEntry",
    "SeriesEntry",
    "ListSeriesResult",
    "list_series",
    "load_series",
    "save_series",
    "delete_series",
    "get_roster",
    "update_roster",
]
