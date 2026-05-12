"""kenkui NLP pipeline — replaces the old BookNLP integration.

Architecture
------------
Stage 1  quotes.py      Regex extracts every dialogue quote deterministically.
Stage 2  entities.py    spaCy finds PERSON names; LLM clusters aliases into
                        a per-book character roster.
Stage 3  chunker.py     Chapter split into overlapping ~700-word chunks at
                        paragraph boundaries.
Stage 4  attribution.py LLM reads each chunk and assigns speaker + emotion
                        to every regex-found quote.  last_speakers state is
                        threaded between chunks for A-B-A-B continuity.

The single public entry point ``run_analysis()`` mirrors the old
``booknlp_processor.run_analysis`` signature so callers need only swap the
import.  Results are cached as JSON keyed by a SHA-256 hash of the book path
and mtime.

Public API
----------
run_analysis(chapters, book_path, nlp_model, progress_callback) → NLPResult
get_cached_result(book_path)    → NLPResult | None
cache_result(result, book_path) → Path
get_cached_roster(book_path)    → FastScanResult | None
cache_roster(result, book_path) → Path
CACHE_DIR                        Path
book_hash(book_path)             str

CONFIG_DIR sentinel
-------------------
``CONFIG_DIR = None`` is a module-level test-seam.  All cache helpers call
``_get_config_dir()``, which returns the patched value when a test sets
``kenkui.nlp.CONFIG_DIR`` to a temporary path, and falls back to the real
``kenkui.config.CONFIG_DIR`` in production.  This lets every cache helper be
controlled with a single ``patch("kenkui.nlp.CONFIG_DIR", tmp_path)`` call.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import sys
import time
from collections import defaultdict
from collections.abc import Callable
from dataclasses import replace as _replace
from pathlib import Path

from ..text_rules import SCENE_BREAK_RE, is_scene_break

logger = logging.getLogger(__name__)


def _is_scene_break(text: str) -> bool:
    """Return True if *text* is a scene-break marker or pure whitespace."""
    return is_scene_break(text)


_SCENE_BREAK_RE = SCENE_BREAK_RE


def _load_spacy_model():
    """Load en_core_web_sm, downloading it automatically if not installed."""
    import spacy
    try:
        return spacy.load("en_core_web_sm")
    except OSError:
        from spacy.cli import download as _spacy_download
        _spacy_download("en_core_web_sm")
        return spacy.load("en_core_web_sm")


def _normalize_gender_pronoun(value: str) -> str:
    """Map BookNLP gender variants to canonical pronoun form.

    infer_gender_pronouns() returns one of "he/him", "she/her", "they/them", or "".
    BookNLP may return uppercase or alternative forms — normalise to the same vocabulary.
    """
    v = value.lower().strip()
    if v in ("he/him", "he", "him", "his", "m", "male", "man"):
        return "he/him"
    if v in ("she/her", "she", "her", "hers", "f", "female", "woman"):
        return "she/her"
    if v in ("they/them", "they", "them", "their", "theirs", "nonbinary", "non-binary"):
        return "they/them"
    return ""


def _resolve_gender(group, full_text: str) -> str:
    """Return the best pronoun set for *group*, cross-validating BookNLP against pronouns.

    Always runs infer_gender_pronouns() and uses its result when it contradicts
    the BookNLP-assigned gender. If pronoun inference returns empty (no clear
    majority or no name mentions found), BookNLP's value is kept as-is.
    """
    from .entities import infer_gender_pronouns
    inferred = infer_gender_pronouns(group.canonical_name, group.aliases, full_text)
    booknlp = _normalize_gender_pronoun(group.gender) if group.gender else ""
    if not booknlp:
        return inferred
    if inferred and inferred != booknlp:
        # Pronoun evidence contradicts BookNLP — trust pronouns
        return inferred
    return booknlp


# ---------------------------------------------------------------------------
# Cache helpers
# ---------------------------------------------------------------------------


def __getattr__(name: str):
    """PEP 562 module-level __getattr__ for lazy attributes.

    ``CACHE_DIR`` is resolved lazily so importing sub-modules (quotes, chunker,
    entities, attribution) does NOT trigger the config / tomli_w import chain.
    """
    if name == "CACHE_DIR":
        from ..config import CONFIG_DIR
        return CONFIG_DIR / "nlp_cache"
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def book_hash(book_path: Path) -> str:
    """Return a stable SHA-256 hex digest for *book_path* (path + mtime)."""
    stat = book_path.stat()
    key = f"{book_path.resolve()}:{stat.st_mtime}"
    return hashlib.sha256(key.encode()).hexdigest()[:32]


def _attribution_cache_name(book_path: Path, provider: "str | None" = None) -> str:
    """Return the cache filename stem for a book + provider combination.

    Ollama (or no provider) uses the legacy ``{hash}.json`` name so existing
    caches remain valid.  Cloud providers use ``{hash}-{provider}.json`` so
    their results are stored separately and are never confused with Ollama output.
    """
    h = book_hash(book_path)
    if provider and provider != "ollama":
        return f"{h}-{provider}.json"
    return f"{h}.json"


def get_cached_result(book_path: Path, provider: "str | None" = None) -> "NLPResult | None":
    """Return a cached ``NLPResult`` if a valid cache file exists, else None.

    When *provider* is a cloud provider name the lookup uses a provider-specific
    cache file (``{hash}-{provider}.json``) so Ollama results are never reused
    for cloud-provider jobs and vice versa.
    """
    from ..models import NLPResult

    cache_dir = _get_config_dir() / "nlp_cache"
    cache_file = cache_dir / _attribution_cache_name(book_path, provider)
    if not cache_file.exists():
        return None
    try:
        data = json.loads(cache_file.read_text(encoding="utf-8"))
        return NLPResult.from_dict(data)
    except Exception as exc:
        logger.warning("Failed to load NLP cache %s: %s", cache_file, exc)
        return None


def cache_result(result: "NLPResult", book_path: Path, provider: "str | None" = None) -> Path:
    """Serialise *result* to disk and return the cache file path.

    Uses a provider-specific filename for cloud providers so results from
    different providers are stored independently.
    """
    cache_dir = _get_config_dir() / "nlp_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file = cache_dir / _attribution_cache_name(book_path, provider)
    cache_file.write_text(
        json.dumps(result.to_dict(), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    logger.debug("NLP cache written: %s", cache_file)
    return cache_file


# CONFIG_DIR is exposed at module level so that patch("kenkui.nlp.CONFIG_DIR", ...) works in
# tests.  The real value is populated lazily by _get_config_dir() to avoid importing
# kenkui.config (and its tomli_w dependency) at module import time.
CONFIG_DIR: "Path | None" = None


def _get_config_dir() -> Path:
    """Return CONFIG_DIR, respecting any test patches applied to this module."""
    val = sys.modules[__name__].CONFIG_DIR
    if val is not None:
        return val  # type: ignore[return-value]
    from ..config import CONFIG_DIR as _cfg
    return _cfg


def _roster_cache_name(
    book_path: Path,
    method: "str | None" = None,
    provider: "str | None" = None,
) -> str:
    """Return roster cache filename for the given method/provider combination.

    New format: ``{hash}-roster-{method}-{provider}.json``
    Legacy format: ``{hash}-roster.json`` (treated as method=llm, provider=ollama on read).
    """
    h = book_hash(book_path)
    m = (method or "llm").lower()
    p = (provider or "ollama").lower() if m != "booknlp" else "none"
    return f"{h}-roster-{m}-{p}.json"


def _legacy_roster_cache_name(book_path: Path) -> str:
    return f"{book_hash(book_path)}-roster.json"


class RosterCacheMeta:
    """Metadata for a single cached roster file."""

    def __init__(
        self,
        path: Path,
        method: str,
        provider: str,
        model: str,
        description: str,
        created_at: str,
        data: dict,
    ):
        self.path = path
        self.method = method
        self.provider = provider
        self.model = model
        self.description = description
        self.created_at = created_at
        self._data = data

    def load(self) -> "FastScanResult":
        from ..models import FastScanResult
        return FastScanResult.from_dict(self._data.get("roster_data") or self._data)


def list_cached_rosters(book_path: Path) -> "list[RosterCacheMeta]":
    """Return all available roster cache files for *book_path*, newest first."""
    h = book_hash(book_path)
    cache_dir = _get_config_dir() / "nlp_cache"
    results: list[RosterCacheMeta] = []

    # New-format files: {hash}-roster-{method}-{provider}.json
    for p in cache_dir.glob(f"{h}-roster-*.json"):
        try:
            raw = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            continue
        if "method" in raw and "created_at" in raw:
            results.append(RosterCacheMeta(
                path=p,
                method=raw.get("method", "llm"),
                provider=raw.get("provider", "ollama"),
                model=raw.get("model", ""),
                description=raw.get("description", ""),
                created_at=raw.get("created_at", ""),
                data=raw,
            ))

    # Legacy file: {hash}-roster.json treated as llm/ollama
    legacy = cache_dir / f"{h}-roster.json"
    if legacy.exists():
        try:
            raw = json.loads(legacy.read_text(encoding="utf-8"))
            if "method" not in raw:
                results.append(RosterCacheMeta(
                    path=legacy,
                    method="llm",
                    provider="ollama",
                    model="",
                    description="llm · ollama (legacy)",
                    created_at="",
                    data=raw,
                ))
        except Exception:
            pass

    results.sort(key=lambda r: r.created_at, reverse=True)
    return results


def get_cached_roster(book_path: Path, method: "str | None" = None, provider: "str | None" = None) -> "FastScanResult | None":
    """Return a cached ``FastScanResult`` if a valid roster cache file exists, else None.

    When *method* and *provider* are given, looks up only the matching file.
    When both are None, falls back to the first available roster (legacy behaviour
    for callers that don't have per-step context).
    """
    from ..models import FastScanResult

    cache_dir = _get_config_dir() / "nlp_cache"

    if method is not None or provider is not None:
        cache_file = cache_dir / _roster_cache_name(book_path, method, provider)
        if not cache_file.exists():
            return None
        try:
            raw = json.loads(cache_file.read_text(encoding="utf-8"))
            data = raw.get("roster_data") or raw
            return FastScanResult.from_dict(data)
        except Exception as exc:
            logger.warning("Failed to load roster cache %s: %s", cache_file, exc)
            return None

    # No method/provider specified — try legacy file first, then any new-format file
    legacy_file = cache_dir / _legacy_roster_cache_name(book_path)
    if legacy_file.exists():
        try:
            data = json.loads(legacy_file.read_text(encoding="utf-8"))
            return FastScanResult.from_dict(data)
        except Exception as exc:
            logger.warning("Failed to load roster cache %s: %s", legacy_file, exc)

    # Try first new-format match
    metas = list_cached_rosters(book_path)
    for meta in metas:
        if meta.path != legacy_file:
            try:
                return meta.load()
            except Exception as exc:
                logger.warning("Failed to load roster cache %s: %s", meta.path, exc)

    return None


def get_cached_roster_or_prompt(
    book_path: Path,
    method: "str | None" = None,
    provider: "str | None" = None,
) -> "FastScanResult | None":
    """Return a matching cached roster, or show an InquirerPy picker when multiple exist.

    - 0 matches → return None (caller should run fresh)
    - 1 match → return silently
    - 2+ matches, method+provider disambiguate → return matching one
    - 2+ matches, ambiguous → show InquirerPy picker with description + timestamp
    """
    from ..models import FastScanResult

    metas = list_cached_rosters(book_path)
    if not metas:
        return None

    # Try exact match first
    if method is not None or provider is not None:
        for meta in metas:
            if (method is None or meta.method == method) and (provider is None or meta.provider == provider):
                try:
                    return meta.load()
                except Exception:
                    pass

    if len(metas) == 1:
        try:
            return metas[0].load()
        except Exception:
            return None

    # Multiple candidates — show picker
    try:
        from InquirerPy import inquirer

        choices = []
        for meta in metas:
            label = meta.description or f"{meta.method} · {meta.provider} · {meta.model}"
            if meta.created_at:
                label = f"{label}  [{meta.created_at[:19]}]"
            choices.append({"name": label, "value": meta})

        choices.append({"name": "(run fresh — skip cache)", "value": None})

        selected = inquirer.select(
            message="Multiple cached rosters found — which would you like to use?",
            choices=choices,
        ).execute()

        if selected is None:
            return None
        try:
            return selected.load()
        except Exception:
            return None
    except Exception:
        # If InquirerPy is unavailable, fall back to newest
        try:
            return metas[0].load()
        except Exception:
            return None


def cache_roster(
    result: "FastScanResult",
    book_path: Path,
    method: "str | None" = None,
    provider: "str | None" = None,
    model: "str | None" = None,
    description: "str | None" = None,
) -> Path:
    """Serialise *result* to disk with a metadata envelope and return the cache file path."""
    import datetime

    cache_dir = _get_config_dir() / "nlp_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    _method = method or "llm"
    _provider = provider or "ollama"
    _model = model or ""

    cache_file = cache_dir / _roster_cache_name(book_path, _method, _provider)

    if _method == "booknlp":
        auto_desc = "booknlp"
    else:
        parts = [_method, _provider]
        if _model:
            parts.append(_model.split("/")[-1])
        auto_desc = " · ".join(parts)

    envelope = {
        "created_at": datetime.datetime.now().isoformat(timespec="seconds"),
        "description": description or auto_desc,
        "method": _method,
        "provider": _provider,
        "model": _model,
        "roster_data": result.to_dict(),
    }
    cache_file.write_text(
        json.dumps(envelope, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    logger.debug("Roster cache written: %s", cache_file)
    return cache_file


# ---------------------------------------------------------------------------
# Per-chunk checkpoint cache
# ---------------------------------------------------------------------------


def _chunk_cache_key(book_h: str, chapter_indices: list[int]) -> str:
    """Deterministic 16-hex cache key for one roster chunk."""
    indices_str = ",".join(str(i) for i in sorted(chapter_indices))
    return hashlib.sha256(f"{book_h}:{indices_str}".encode()).hexdigest()[:16]


def get_cached_chunk_roster(book_path: Path, chapter_indices: list[int]) -> "CharacterRoster | None":
    """Return a cached ``CharacterRoster`` for one chunk, or None."""
    from kenkui.nlp.models import CharacterRoster

    bh = book_hash(book_path)
    cache_dir = _get_config_dir() / "nlp_cache"
    key = _chunk_cache_key(bh, chapter_indices)
    cache_file = cache_dir / f"{bh}-chunk-{key}.json"
    if not cache_file.exists():
        return None
    try:
        data = json.loads(cache_file.read_text(encoding="utf-8"))
        return CharacterRoster.model_validate(data)
    except Exception as exc:
        logger.warning("Failed to load chunk cache %s: %s", cache_file, exc)
        return None


def cache_chunk_roster(roster: "CharacterRoster", book_path: Path, chapter_indices: list[int]) -> Path:
    """Write a partial ``CharacterRoster`` for one chunk to disk."""
    bh = book_hash(book_path)
    cache_dir = _get_config_dir() / "nlp_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    key = _chunk_cache_key(bh, chapter_indices)
    cache_file = cache_dir / f"{bh}-chunk-{key}.json"
    cache_file.write_text(
        json.dumps(roster.model_dump(), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    logger.debug("Chunk roster cache written: %s", cache_file)
    return cache_file


# ---------------------------------------------------------------------------
# Mention counting helper
# ---------------------------------------------------------------------------


def _count_mentions(roster: "CharacterRoster", full_text: str) -> dict[str, int]:
    """Count word-boundary occurrences of each character's aliases in *full_text*.

    Returns a mapping of canonical name → total mention count across all aliases.
    """
    counts: dict[str, int] = {}
    for group in roster.characters:
        total = 0
        for alias in group.aliases:
            pattern = re.compile(r"\b" + re.escape(alias) + r"\b", re.IGNORECASE)
            total += len(pattern.findall(full_text))
        counts[group.canonical_name] = total
    return counts


# ---------------------------------------------------------------------------
# Fast scan entry point (Stage 1-2 only)
# ---------------------------------------------------------------------------


def run_fast_scan(
    chapters: list,
    book_path: Path,
    nlp_model: str,
    use_cache: bool = True,
    progress_callback: Callable[[str], None] | None = None,
    method: str = "auto",
) -> "FastScanResult":
    """Run Stage 1-2 only: quote extraction + entity clustering + mention counting.

    Significantly faster than ``run_analysis()`` — no LLM attribution over
    individual chapters. Results are cached to ``nlp_cache/{hash}-roster.json``.

    Args:
        chapters:          List of ``Chapter`` objects (paragraphs populated).
        book_path:         Path to the source ebook (used for cache key).
        nlp_model:         Ollama model name (e.g. ``"llama3.2"``).
        use_cache:         Return cached result if available.
        progress_callback: Optional callable receiving status strings.

    Returns:
        ``FastScanResult`` with characters sorted by mention_count descending.
    """
    import spacy

    from ..models import CharacterInfo, FastScanResult
    from .entities import build_roster_with_llm
    from .llm import LLMClient

    if use_cache:
        cached = get_cached_roster(book_path)
        if cached is not None:
            return cached

    _cb: Callable[[str], None] = progress_callback or (lambda _: None)

    # Verify Ollama is reachable (needed for Stage 2 LLM cleanup passes)
    import ollama as _ollama
    try:
        _ollama.list()
    except Exception as exc:
        raise RuntimeError(
            f"Cannot reach Ollama at localhost:11434 — is it running? ({exc})"
        ) from exc

    llm = LLMClient(nlp_model)

    # Load spaCy
    _cb("Loading spaCy language model…")
    nlp = _load_spacy_model()

    # Stage 2: Build character roster
    _cb("Building character roster…")
    full_text = " ".join(" ".join(ch.paragraphs) for ch in chapters)
    roster = build_roster_with_llm(full_text, nlp, llm, method=method)

    char_names = ", ".join(g.canonical_name for g in roster.characters[:8])
    overflow = len(roster.characters) - 8
    suffix = f" (+{overflow} more)" if overflow > 0 else ""
    _cb(f"Character roster: {len(roster.characters)} characters — {char_names}{suffix}")

    # Count name mentions
    _cb("Counting character mentions…")
    mention_counts = _count_mentions(roster, full_text)

    characters: list[CharacterInfo] = [
        CharacterInfo(
            character_id=group.canonical_name,
            display_name=group.canonical_name,
            mention_count=mention_counts.get(group.canonical_name, 0),
            gender_pronoun=_resolve_gender(group, full_text),
        )
        for group in roster.characters
    ]
    characters.sort(key=lambda c: c.mention_count, reverse=True)

    result = FastScanResult(roster=roster, characters=characters, book_hash=book_hash(book_path))
    cache_roster(result, book_path)
    return result


# ---------------------------------------------------------------------------
# Stage 3-4 entry point
# ---------------------------------------------------------------------------


def run_attribution(
    roster: "CharacterRoster",
    chapters: list,
    book_path: Path,
    nlp_model: str,
    use_cache: bool = True,
    progress_callback: Callable[[str], None] | None = None,
    confidence_threshold: int = 0,
    review_model: str = "",
) -> "NLPResult":
    """Run Stage 3-4: LLM speaker attribution using a pre-built roster.

    Cache-aware: returns a cached ``NLPResult`` from ``nlp_cache/{hash}.json``
    if one exists and ``use_cache`` is True.

    Args:
        roster:            ``CharacterRoster`` from a prior ``run_fast_scan()`` call.
        chapters:          List of ``Chapter`` objects (paragraphs populated).
        book_path:         Path to the source ebook (used for cache key).
        nlp_model:         Ollama model name (e.g. ``"llama3.2"``).
        use_cache:         Return cached NLPResult if available.
        progress_callback: Optional callable receiving status strings.
        confidence_threshold: Quotes with confidence below this value are
                              re-attributed in a second pass using *review_model*.
                              0 disables the second pass (default).
        review_model:      Ollama model name for the second-pass review.
                           Falls back to *nlp_model* when empty (default).

    Returns:
        Full ``NLPResult`` with annotated chapters and quote counts.

    Note:
        The cached ``NLPResult`` written by this function has ``mention_count=0``
        on all characters — mention counts are a Stage 1-2 concern populated by
        ``run_fast_scan()``. When called via ``run_analysis()``, the cache is
        re-written with ``mention_count`` populated. Direct callers should be aware
        of this if they read the cache independently afterward.
    """
    import spacy

    from ..models import Chapter, CharacterInfo, NLPResult, Segment
    from .attribution import attribute_all_chunks
    from .chunker import chunk_paragraphs
    from .entities import extract_person_names
    from .llm import LLMClient
    from .quotes import extract_quotes, strip_scare_quotes

    if use_cache:
        cached = get_cached_result(book_path)
        if cached is not None:
            return cached

    _cb: Callable[[str], None] = progress_callback or (lambda _: None)

    # Verify Ollama is reachable
    import ollama as _ollama
    try:
        _ollama.list()
    except Exception as exc:
        raise RuntimeError(
            f"Cannot reach Ollama at localhost:11434 — is it running? ({exc})"
        ) from exc

    llm = LLMClient(nlp_model)
    review_llm = LLMClient(review_model) if review_model else None

    # Load spaCy
    _cb("Loading spaCy language model…")
    nlp = _load_spacy_model()

    full_text = " ".join(" ".join(ch.paragraphs) for ch in chapters)

    # Build alias → canonical lookup
    alias_to_canonical: dict[str, str] = {}
    for group in roster.characters:
        alias_to_canonical[group.canonical_name.lower()] = group.canonical_name
        for alias in group.aliases:
            alias_to_canonical[alias.lower()] = group.canonical_name

    roster_aliases: dict[str, list[str]] = {
        group.canonical_name: group.aliases for group in roster.characters
    }

    # Stage 1: Extract quotes per chapter (using scare-quote-stripped paragraphs)
    _cb("Extracting dialogue quotes…")
    chapter_quotes: dict[int, list] = {}
    chapter_clean_paras: dict[int, list[str]] = {}
    for chapter in chapters:
        clean_paras = strip_scare_quotes(chapter.paragraphs)
        chapter_clean_paras[chapter.index] = clean_paras
        chapter_quotes[chapter.index] = extract_quotes(clean_paras)

    # Stages 3+4: Per-chapter chunking and attribution
    attributed_chapters: list[Chapter] = []
    attribution_counts: dict[str, int] = defaultdict(int)

    total_chapters = len(chapters)
    attrib_start = time.monotonic()

    for ch_done, chapter in enumerate(chapters):
        title = chapter.title or f"Chapter {chapter.index}"
        elapsed = time.monotonic() - attrib_start
        if ch_done > 0:
            avg = elapsed / ch_done
            remaining = avg * (total_chapters - ch_done)
            eta_min = int(remaining // 60)
            eta_sec = int(remaining % 60)
            _cb(f"Attributing: {title}… (ETA {eta_min:02d}:{eta_sec:02d})")
        else:
            _cb(f"Attributing: {title}…")

        quotes = chapter_quotes[chapter.index]
        clean_paras = chapter_clean_paras[chapter.index]

        if not quotes:
            segments = [
                Segment(
                    text="\n\n".join(clean_paras),
                    speaker="NARRATOR",
                    index=0,
                )
            ]
        else:
            chapter_text = " ".join(clean_paras)
            raw_names = extract_person_names(chapter_text, nlp)
            chapter_canonicals = sorted({
                canonical
                for n in raw_names
                if (canonical := alias_to_canonical.get(n.lower())) is not None
            })
            roster_names = chapter_canonicals + ["NARRATOR", "Unknown"]

            chunks = chunk_paragraphs(clean_paras, quotes)
            all_attributions = attribute_all_chunks(
                chunks, quotes, roster_names, llm, roster_aliases=roster_aliases,
                confidence_threshold=confidence_threshold, review_llm=review_llm,
            )

            for item in all_attributions.values():
                item.speaker = _normalize_speaker(
                    item.speaker, alias_to_canonical, chapter_canonicals
                )

            segments = _build_segments(clean_paras, quotes, all_attributions)

            for item in all_attributions.values():
                if item.speaker not in ("NARRATOR", "Unknown"):
                    attribution_counts[item.speaker] += 1

        attributed_chapters.append(_replace(chapter, segments=segments))

    # Build CharacterInfo with quote_count
    characters: list[CharacterInfo] = [
        CharacterInfo(
            character_id=group.canonical_name,
            display_name=group.canonical_name,
            quote_count=attribution_counts.get(group.canonical_name, 0),
            gender_pronoun=_resolve_gender(group, full_text),
        )
        for group in roster.characters
    ]
    characters.sort(key=lambda c: c.quote_count, reverse=True)

    result = NLPResult(
        characters=characters,
        chapters=attributed_chapters,
        book_hash=book_hash(book_path),
    )
    cache_result(result, book_path)
    return result


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def run_analysis(
    chapters: list,
    book_path: Path,
    nlp_model: str,
    progress_callback: Callable[[str], None] | None = None,
    confidence_threshold: int = 0,
    review_model: str = "",
) -> "NLPResult":
    """Run the full NLP speaker-attribution pipeline on *chapters*.

    Delegates to ``run_fast_scan()`` (Stage 1-2) then ``run_attribution()``
    (Stage 3-4), merging mention_count from the fast scan into the final result.

    Args:
        chapters:          List of ``Chapter`` objects (paragraphs populated).
        book_path:         Path to the source ebook (used for cache key).
        nlp_model:         Ollama model name (e.g. ``"llama3.2"``).
        progress_callback: Optional callable receiving status strings.

    Returns:
        ``NLPResult`` with both ``mention_count`` and ``quote_count`` populated.
    """
    _cb: Callable[[str], None] = progress_callback or (lambda _: None)

    # Stage 1-2: fast scan (may use roster cache)
    fast_result = run_fast_scan(
        chapters=chapters,
        book_path=book_path,
        nlp_model=nlp_model,
        use_cache=True,
        progress_callback=_cb,
    )

    # Stage 3-4: attribution (may use full NLP cache)
    nlp_result = run_attribution(
        roster=fast_result.roster,
        chapters=chapters,
        book_path=book_path,
        nlp_model=nlp_model,
        use_cache=True,
        progress_callback=_cb,
        confidence_threshold=confidence_threshold,
        review_model=review_model,
    )

    # Patch mention_count from fast scan into NLP result characters
    mention_by_id = {c.character_id: c.mention_count for c in fast_result.characters}
    nlp_result = _replace(
        nlp_result,
        characters=sorted(
            [
                _replace(c, mention_count=mention_by_id.get(c.character_id, 0))
                for c in nlp_result.characters
            ],
            key=lambda c: c.prominence,
            reverse=True,
        ),
    )

    cache_result(nlp_result, book_path)
    return nlp_result


# ---------------------------------------------------------------------------
# Segment assembly helper
# ---------------------------------------------------------------------------


def _split_paragraph_by_quotes(
    para: str, para_quotes: list
) -> list[tuple[str, str]]:
    """Split a paragraph into (text, speaker) spans at attributed quote/italic boundaries.

    Runs both the dialogue regex (_QUOTE_RE) and the italic regex (_ITALIC_RE)
    on *para*, merges all matches in document order, and looks up each span's
    speaker from *para_quotes* (a list of ``(Quote, AttributionItem)`` pairs).

    Lookup key consistency:
    - Dialogue Quote.text includes quote marks  → matches _QUOTE_RE.group(0)
    - Italic   Quote.text is plain content      → matches _ITALIC_RE.group(1)

    For italic spans the marker characters (\\x02/\\x03) are stripped from the
    final segment text so TTS receives clean content.

    Unattributed spans and surrounding narrative text both become NARRATOR.
    Concatenating all returned texts reconstructs *para* exactly (minus the
    \\x02/\\x03 marker characters from italic spans).

    Returns:
        A list of ``(text, speaker)`` 2-tuples.  Falls back to
        ``[(para, "NARRATOR")]`` when the paragraph is empty or no matches align.
    """
    from .quotes import _ITALIC_RE, _QUOTE_RE

    if not para:
        return [(para, "NARRATOR")]

    # Build a text → speaker map from the attributed quotes in this paragraph.
    # Dialogue key = full quoted string with marks; italic key = plain content.
    # If the same text appears more than once we keep the first attribution.
    text_to_speaker: dict[str, str] = {}
    for q, attr in para_quotes:
        if q.text not in text_to_speaker:
            text_to_speaker[q.text] = attr.speaker

    # Collect all matches from both patterns, tagged with kind and output text.
    all_matches: list[tuple[int, int, str, str]] = []  # (start, end, out_text, lookup_key)
    for m in _QUOTE_RE.finditer(para):
        all_matches.append((m.start(), m.end(), m.group(0), m.group(0)))
    for m in _ITALIC_RE.finditer(para):
        # out_text has markers stripped; lookup_key is the plain content
        all_matches.append((m.start(), m.end(), m.group(1), m.group(1)))
    all_matches.sort(key=lambda t: t[0])

    spans: list[tuple[str, str]] = []
    last_end = 0

    for start, end, out_text, lookup_key in all_matches:
        if start < last_end:
            continue

        speaker = text_to_speaker.get(lookup_key, "NARRATOR")

        # Narrative text before this span
        if start > last_end:
            narrator_text = para[last_end:start]
            if narrator_text:
                spans.append((narrator_text, "NARRATOR"))

        if out_text:
            spans.append((out_text, speaker))

        last_end = end

    # Trailing narrative text after the last span
    if last_end < len(para):
        trailing = para[last_end:]
        if trailing:
            spans.append((trailing, "NARRATOR"))

    return spans if spans else [(para, "NARRATOR")]


def _merge_consecutive_segments(segments: list) -> list:
    """Merge adjacent segments that share the same speaker.

    Narrator spans are joined with ``"\\n\\n"``; character spans with ``" "``.
    Indices are rewritten to be contiguous starting from 0.
    """
    from ..models import Segment

    if not segments:
        return segments

    merged: list[Segment] = []
    for seg in segments:
        # Scene-break segments are never merged with adjacent segments
        if merged and merged[-1].speaker == seg.speaker and not seg.is_scene_break and not merged[-1].is_scene_break:
            prev = merged[-1]
            sep = "\n\n" if seg.speaker == "NARRATOR" else " "
            merged[-1] = Segment(
                text=prev.text + sep + seg.text,
                speaker=prev.speaker,
                index=prev.index,
                is_scene_break=prev.is_scene_break,
            )
        else:
            merged.append(Segment(text=seg.text, speaker=seg.speaker, index=seg.index, is_scene_break=seg.is_scene_break))

    # Rewrite indices to be contiguous
    for i, seg in enumerate(merged):
        merged[i] = Segment(text=seg.text, speaker=seg.speaker, index=i, is_scene_break=seg.is_scene_break)

    return merged


def _build_segments(paragraphs: list[str], quotes: list, attributions: dict) -> list:
    """Convert paragraphs + quote attributions into a flat Segment list.

    Strategy:
    - Walk paragraphs in order.
    - If a paragraph has no attributed quotes, buffer it as NARRATOR.
    - If a paragraph has attributed quotes, split it at quote boundaries
      (via _split_paragraph_by_quotes) so each quoted span gets its own
      character Segment and surrounding narrative goes to the NARRATOR buffer.
    - After the main loop, merge any consecutive same-speaker Segments to
      reduce TTS call overhead and end-of-clip artifacts.
    """
    from ..models import Segment

    # para_index → list of (quote, attribution_item)
    para_to_attr: dict[int, list] = defaultdict(list)
    for q in quotes:
        if q.id in attributions:
            para_to_attr[q.para_index].append((q, attributions[q.id]))

    segments: list[Segment] = []
    seg_idx = 0
    narrator_buf: list[str] = []

    def _flush_narrator() -> None:
        nonlocal seg_idx
        if narrator_buf:
            segments.append(
                Segment(text="\n\n".join(narrator_buf), speaker="NARRATOR", index=seg_idx)
            )
            seg_idx += 1
            narrator_buf.clear()

    for para_idx, para in enumerate(paragraphs):
        if _is_scene_break(para):
            _flush_narrator()
            segments.append(Segment(text="", speaker="SCENE_BREAK", index=seg_idx, is_scene_break=True))
            seg_idx += 1
        elif para_idx not in para_to_attr:
            narrator_buf.append(para)
        else:
            spans = _split_paragraph_by_quotes(para, para_to_attr[para_idx])
            for span_text, speaker in spans:
                if speaker == "NARRATOR":
                    narrator_buf.append(span_text)
                else:
                    _flush_narrator()
                    segments.append(Segment(text=span_text, speaker=speaker, index=seg_idx))
                    seg_idx += 1

    _flush_narrator()

    return _merge_consecutive_segments(segments)


def _attribution_to_segments(
    chapter: "Chapter",
    attr_result: "AttributionResult",
    roster: "CharacterRoster",
) -> list["Segment"]:
    """Convert AttributionResult + Chapter paragraphs into a Segment list.

    Used by NLPService when dispatching through the provider protocol.
    """
    from kenkui.nlp.quotes import extract_quotes, strip_scare_quotes
    clean_paragraphs = strip_scare_quotes(chapter.paragraphs)
    quotes = extract_quotes(clean_paragraphs)
    quote_by_id = {q.id: q for q in quotes}

    # Verify LLM-returned char positions match pre-extracted quote positions.
    # On mismatch, clear the position fields so downstream falls back to id-based matching.
    for item in attr_result.attributions:
        if item.char_start is not None and item.quote_id in quote_by_id:
            expected = quote_by_id[item.quote_id]
            if item.char_start != expected.char_offset:
                logger.warning(
                    "Attribution position mismatch for quote_id=%d "
                    "(expected offset %d, got %d) — using id-based match",
                    item.quote_id, expected.char_offset, item.char_start,
                )
                item.char_start = None
                item.char_end = None

    attributions = {item.quote_id: item for item in attr_result.attributions}

    # Fill in any quote IDs the LLM skipped using last-known-speaker fallback.
    # Walk quotes in document order so each missing quote can inherit from its
    # nearest resolved predecessor rather than defaulting to Unknown.
    from kenkui.nlp.models import AttributionItem as _AttributionItem
    ordered_quotes = sorted(quotes, key=lambda q: (q.para_index, q.char_offset))
    resolved_in_order: list[str] = []

    for q in ordered_quotes:
        if q.id not in attributions:
            # Walk backwards for last non-NARRATOR/Unknown speaker
            last_speaker = next(
                (s for s in reversed(resolved_in_order) if s not in ("NARRATOR", "Unknown")),
                None,
            )
            speaker = last_speaker if last_speaker else "NARRATOR"
            logger.warning(
                "Chapter %d: quote id=%d missing from LLM attribution — "
                "fallback speaker=%s", chapter.index, q.id, speaker,
            )
            attributions[q.id] = _AttributionItem(
                quote_id=q.id, speaker=speaker, emotion="neutral", confidence=1
            )
        resolved_in_order.append(attributions[q.id].speaker)

    return _build_segments(clean_paragraphs, quotes, attributions)


def _normalize_speaker(
    speaker: str,
    alias_to_canonical: dict[str, str],
    chapter_canonicals: list[str],
) -> str:
    """Map an LLM-returned speaker name back to its book-level canonical.

    Strategy:
    1. Pass-through ``NARRATOR`` and ``Unknown`` unchanged.
    2. Exact lowercase lookup in ``alias_to_canonical``.
    3. Fuzzy word-overlap fallback: find the chapter canonical whose
       significant words overlap most with the returned name.
    """
    if speaker in ("NARRATOR", "Unknown"):
        return speaker

    canonical = alias_to_canonical.get(speaker.lower())
    if canonical:
        return canonical

    # Fuzzy fallback using the same word-overlap logic as alias clustering.
    from .entities import _significant_words

    speaker_words = set(_significant_words(speaker))
    if not speaker_words:
        return speaker

    best: str | None = None
    best_overlap = 0
    for name in chapter_canonicals:
        overlap = len(speaker_words & set(_significant_words(name)))
        if overlap > best_overlap:
            best_overlap = overlap
            best = name

    return best if best else speaker


__all__ = [
    "run_analysis",
    "run_fast_scan",
    "run_attribution",
    "get_cached_result",
    "cache_result",
    "get_cached_roster",
    "get_cached_roster_or_prompt",
    "list_cached_rosters",
    "cache_roster",
    "get_cached_chunk_roster",
    "cache_chunk_roster",
    "RosterCacheMeta",
    "CACHE_DIR",
    "book_hash",
    "_count_mentions",
    "_split_paragraph_by_quotes",
    "_merge_consecutive_segments",
    "_build_segments",
    "_attribution_to_segments",
    "_is_scene_break",
    "_SCENE_BREAK_RE",
]
