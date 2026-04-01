"""Series voice consistency — manifest I/O and character matching.

A series manifest lives at:
    ~/.config/kenkui/series/{slug}.toml

It accumulates character→voice assignments across books, so the same
character always gets the same voice in subsequent installments.
"""
from __future__ import annotations

import json
import re
import tomllib

import tomli_w
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .models import CharacterInfo, FastScanResult

# Test seam: override the series directory in tests via monkeypatch.
_series_dir_override: Path | None = None


def series_dir() -> Path:
    """Return (and create if needed) the series manifest directory."""
    if _series_dir_override is not None:
        _series_dir_override.mkdir(parents=True, exist_ok=True)
        return _series_dir_override
    from .config import CONFIG_DIR
    d = CONFIG_DIR / "series"
    d.mkdir(parents=True, exist_ok=True)
    return d


def slugify(name: str) -> str:
    """Convert a series name to a filesystem-safe slug."""
    s = name.lower().strip()
    s = re.sub(r"[^\w\s-]", "", s)
    s = re.sub(r"[\s_]+", "-", s)
    s = re.sub(r"-+", "-", s)
    return s.strip("-")


@dataclass
class SeriesCharacter:
    canonical: str
    aliases: list[str] = field(default_factory=list)
    voice: str = ""
    gender: str = ""


@dataclass
class SeriesManifest:
    name: str
    slug: str
    updated_at: str
    characters: list[SeriesCharacter] = field(default_factory=list)


def load_series(slug: str) -> SeriesManifest | None:
    """Load a series manifest by slug. Returns None if not found."""
    path = series_dir() / f"{slug}.toml"
    if not path.exists():
        return None
    try:
        data = tomllib.loads(path.read_text(encoding="utf-8"))
        return SeriesManifest(
            name=data["name"],
            slug=slug,
            updated_at=data.get("updated_at", ""),
            characters=[
                SeriesCharacter(
                    canonical=c["canonical"],
                    aliases=c.get("aliases", []),
                    voice=c.get("voice", ""),
                    gender=c.get("gender", ""),
                )
                for c in data.get("characters", [])
            ],
        )
    except Exception:
        return None


def save_series(manifest: SeriesManifest) -> None:
    """Persist a SeriesManifest to disk atomically."""
    manifest.updated_at = datetime.now(timezone.utc).isoformat()
    data: dict = {
        "name": manifest.name,
        "updated_at": manifest.updated_at,
        "characters": [
            {
                "canonical": c.canonical,
                "aliases": c.aliases,
                "voice": c.voice,
                "gender": c.gender,
            }
            for c in manifest.characters
        ],
    }
    path = series_dir() / f"{manifest.slug}.toml"
    tmp = path.with_suffix(".tmp")
    tmp.write_bytes(tomli_w.dumps(data).encode("utf-8"))
    tmp.replace(path)


def list_series() -> list[SeriesManifest]:
    """Return all series manifests sorted by mtime descending (most recent first)."""
    d = series_dir()
    results = []
    for f in sorted(d.glob("*.toml"), key=lambda p: p.stat().st_mtime, reverse=True):
        m = load_series(f.stem)
        if m is not None:
            results.append(m)
    return results


def match_characters(
    characters: list["CharacterInfo"],
    roster: "FastScanResult",
    manifest: SeriesManifest,
) -> tuple[dict[str, str], set[str]]:
    """Match new-book characters to series manifest entries via alias-overlap.

    Returns:
        inherited_voices: char_id → voice for matched characters
        pinned: set of char_ids that received an inherited voice
    """
    alias_map: dict[str, list[str]] = {
        g.canonical: g.aliases for g in roster.roster.characters
    }

    inherited_voices: dict[str, str] = {}
    pinned: set[str] = set()

    for char in characters:
        aliases = alias_map.get(char.character_id, [])
        char_names = {char.character_id.lower()} | {a.lower() for a in aliases}

        best_score = 0.0
        best_entry: SeriesCharacter | None = None

        for entry in manifest.characters:
            entry_names = {entry.canonical.lower()} | {a.lower() for a in entry.aliases}
            score = _score_names(char_names, entry_names)
            if score > best_score:
                best_score = score
                best_entry = entry

        if best_score >= 0.7 and best_entry is not None and best_entry.voice:
            inherited_voices[char.character_id] = best_entry.voice
            pinned.add(char.character_id)

    return inherited_voices, pinned


def _score_names(a_names: set[str], b_names: set[str]) -> float:
    """Score similarity between two sets of name strings. Returns 0.0–1.0."""
    if a_names & b_names:
        return 1.0
    best = 0.0
    for a in a_names:
        for b in b_names:
            s = _word_overlap(a, b)
            if s > best:
                best = s
    return best


def _word_overlap(a: str, b: str) -> float:
    """Fraction of words in the shorter string that appear in the longer string.

    Exact word match scores full credit. A word scores credit if it is a
    prefix of a longer word AND is at least 3 characters (to avoid false
    positives from single-letter or two-letter abbreviations).
    """
    a_words = set(a.lower().split())
    b_words = set(b.lower().split())
    if not a_words or not b_words:
        return 0.0
    if len(a_words) <= len(b_words):
        shorter, longer = a_words, b_words
    else:
        shorter, longer = b_words, a_words
    matched = sum(
        1
        for w in shorter
        if w in longer or (len(w) >= 3 and any(lw.startswith(w) for lw in longer))
    )
    return matched / len(shorter)


def list_roster_candidates() -> list[dict]:
    """Return books with cached roster files, sorted by mtime descending.

    Each entry: {hash, title, path, speaker_voices, roster_path}
    Scans nlp_cache for roster files; cross-references queue.toml for names
    and voice assignments from completed multi-voice jobs.
    """
    from .config import CONFIG_DIR
    cache_dir = CONFIG_DIR / "nlp_cache"
    if not cache_dir.exists():
        return []

    roster_files = sorted(
        cache_dir.glob("*-roster.json"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )

    queue_file = CONFIG_DIR / "queue.toml"
    job_by_hash: dict[str, dict] = {}
    if queue_file.exists():
        try:
            data = tomllib.loads(queue_file.read_text(encoding="utf-8"))
            from .models import JobStatus, QueueItem
            from .nlp import book_hash as _book_hash
            for item_data in data.get("items", []):
                item = QueueItem.from_dict(item_data)
                if item.status != JobStatus.COMPLETED:
                    continue
                try:
                    h = _book_hash(Path(item.job.ebook_path))
                    job_by_hash[h] = {
                        "title": item.job.name,
                        "path": str(item.job.ebook_path),
                        "speaker_voices": item.job.speaker_voices,
                    }
                except OSError:
                    pass
        except Exception:
            pass

    candidates = []
    for f in roster_files:
        h = f.stem.replace("-roster", "")
        info = job_by_hash.get(h, {})
        candidates.append({
            "hash": h,
            "title": info.get("title") or h[:8],
            "path": info.get("path", ""),
            "speaker_voices": info.get("speaker_voices", {}),
            "roster_path": f,
        })
    return candidates


def build_manifest_from_predecessor(candidate: dict, name: str) -> SeriesManifest:
    """Create a new SeriesManifest seeded from a previously processed book.

    ``candidate`` is a dict from ``list_roster_candidates()``.
    Characters without an assigned voice are omitted.
    """
    from .models import FastScanResult

    data = json.loads(Path(candidate["roster_path"]).read_text(encoding="utf-8"))
    roster = FastScanResult.from_dict(data)
    alias_map = {g.canonical: g.aliases for g in roster.roster.characters}
    speaker_voices: dict[str, str] = candidate.get("speaker_voices", {})

    characters = []
    for char in roster.characters:
        voice = speaker_voices.get(char.character_id, "")
        if not voice:
            continue
        characters.append(
            SeriesCharacter(
                canonical=char.character_id,
                aliases=alias_map.get(char.character_id, []),
                voice=voice,
                gender=char.gender_pronoun,
            )
        )

    return SeriesManifest(
        name=name,
        slug=slugify(name),
        updated_at="",
        characters=characters,
    )


def update_manifest(
    manifest: SeriesManifest,
    characters: list["CharacterInfo"],
    roster: "FastScanResult",
    speaker_voices: dict[str, str],
    pinned: set[str],
) -> SeriesManifest:
    """Return an updated copy of *manifest* incorporating new book's assignments.

    - Pinned characters with changed voices → existing entry updated
    - New characters (not pinned) → appended if not already present
    Does not mutate the original manifest.
    """
    alias_map = {g.canonical: g.aliases for g in roster.roster.characters}

    updated_chars = [
        SeriesCharacter(
            canonical=c.canonical,
            aliases=list(c.aliases),
            voice=c.voice,
            gender=c.gender,
        )
        for c in manifest.characters
    ]

    for char in characters:
        voice = speaker_voices.get(char.character_id, "")
        if not voice:
            continue
        aliases = alias_map.get(char.character_id, [])

        if char.character_id in pinned:
            for entry in updated_chars:
                if entry.canonical.lower() == char.character_id.lower():
                    entry.voice = voice
                    existing_lower = {a.lower() for a in entry.aliases}
                    for a in aliases:
                        if a.lower() not in existing_lower:
                            entry.aliases.append(a)
                    break
        else:
            char_names = {char.character_id.lower()} | {a.lower() for a in aliases}
            already_present = any(
                _score_names(
                    char_names,
                    {e.canonical.lower()} | {a.lower() for a in e.aliases},
                ) >= 0.7
                for e in updated_chars
            )
            if not already_present:
                updated_chars.append(
                    SeriesCharacter(
                        canonical=char.character_id,
                        aliases=aliases,
                        voice=voice,
                        gender=char.gender_pronoun,
                    )
                )

    return SeriesManifest(
        name=manifest.name,
        slug=manifest.slug,
        updated_at=manifest.updated_at,
        characters=updated_chars,
    )
