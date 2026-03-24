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
get_cached_result(book_path)   → NLPResult | None
cache_result(result, book_path) → Path
CACHE_DIR                       Path
book_hash(book_path)            str
"""

from __future__ import annotations

import hashlib
import json
import logging
from collections import defaultdict
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path

logger = logging.getLogger(__name__)


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


def get_cached_result(book_path: Path) -> "NLPResult | None":
    """Return a cached ``NLPResult`` if a valid cache file exists, else None."""
    from ..config import CONFIG_DIR
    from ..models import NLPResult

    cache_dir = CONFIG_DIR / "nlp_cache"
    cache_file = cache_dir / f"{book_hash(book_path)}.json"
    if not cache_file.exists():
        return None
    try:
        data = json.loads(cache_file.read_text(encoding="utf-8"))
        return NLPResult.from_dict(data)
    except Exception as exc:
        logger.warning("Failed to load NLP cache %s: %s", cache_file, exc)
        return None


def cache_result(result: "NLPResult", book_path: Path) -> Path:
    """Serialise *result* to disk and return the cache file path."""
    from ..config import CONFIG_DIR

    cache_dir = CONFIG_DIR / "nlp_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file = cache_dir / f"{book_hash(book_path)}.json"
    cache_file.write_text(
        json.dumps(result.to_dict(), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    logger.debug("NLP cache written: %s", cache_file)
    return cache_file


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def run_analysis(
    chapters: list,
    book_path: Path,
    nlp_model: str,
    progress_callback: Callable[[str], None] | None = None,
) -> "NLPResult":
    """Run the full NLP speaker-attribution pipeline on *chapters*.

    Args:
        chapters:          List of ``Chapter`` objects (paragraphs populated).
        book_path:         Path to the source ebook (used for cache key).
        nlp_model:         Ollama model name (e.g. ``"llama3.2"``).
        progress_callback: Optional callable receiving status strings.

    Returns:
        ``NLPResult`` with ``characters`` populated and ``chapters`` having
        ``segments`` set on every chapter.

    Raises:
        RuntimeError: If Ollama is not reachable.
    """
    import spacy

    from ..models import Chapter, CharacterInfo, NLPResult, Segment
    from .attribution import attribute_all_chunks
    from .chunker import chunk_paragraphs
    from .entities import build_roster, build_roster_with_llm, extract_person_names
    from .llm import LLMClient
    from .quotes import extract_quotes

    _cb: Callable[[str], None] = progress_callback or (lambda _: None)

    # ---- Verify Ollama is reachable ----------------------------------------
    import ollama as _ollama
    try:
        _ollama.list()
    except Exception as exc:
        raise RuntimeError(
            f"Cannot reach Ollama at localhost:11434 — is it running? ({exc})"
        ) from exc

    llm = LLMClient(nlp_model)

    # ---- Load spaCy --------------------------------------------------------
    _cb("Loading spaCy language model…")
    try:
        nlp = spacy.load("en_core_web_sm")
    except OSError:
        raise RuntimeError(
            "spaCy model 'en_core_web_sm' not found. "
            "Install it with: uv pip install https://github.com/explosion/spacy-models"
            "/releases/download/en_core_web_sm-3.8.0/en_core_web_sm-3.8.0-py3-none-any.whl"
        )

    # ---- Stage 1: Extract quotes from every chapter -----------------------
    _cb("Extracting dialogue quotes…")
    chapter_quotes: dict[int, list] = {}
    for chapter in chapters:
        chapter_quotes[chapter.index] = extract_quotes(chapter.paragraphs)

    total_q = sum(len(qs) for qs in chapter_quotes.values())
    _cb(f"Found {total_q} quotes across {len(chapters)} chapters")

    # ---- Stage 2: Build character roster from full book text --------------
    _cb("Building character roster…")
    full_text = " ".join(" ".join(ch.paragraphs) for ch in chapters)
    roster = build_roster_with_llm(full_text, nlp, llm)

    char_names = ", ".join(g.canonical for g in roster.characters[:8])
    overflow = len(roster.characters) - 8
    suffix = f" (+{overflow} more)" if overflow > 0 else ""
    _cb(f"Character roster: {len(roster.characters)} characters — {char_names}{suffix}")
    logger.info("Stage 2 complete: %d characters", len(roster.characters))

    # Alias → canonical lookup (lower-cased for fuzzy matching)
    alias_to_canonical: dict[str, str] = {}
    for group in roster.characters:
        alias_to_canonical[group.canonical.lower()] = group.canonical
        for alias in group.aliases:
            alias_to_canonical[alias.lower()] = group.canonical

    # Roster aliases dict for LLM prompt (canonical → all aliases including itself)
    roster_aliases: dict[str, list[str]] = {
        group.canonical: group.aliases
        for group in roster.characters
    }

    # ---- Stages 3 + 4: Per-chapter chunking and attribution ---------------
    attributed_chapters: list[Chapter] = []
    attribution_counts: dict[str, int] = defaultdict(int)

    for chapter in chapters:
        title = chapter.title or f"Chapter {chapter.index}"
        _cb(f"Attributing: {title}…")

        quotes = chapter_quotes[chapter.index]

        if not quotes:
            # No dialogue — entire chapter is a single narrator segment.
            segments = [
                Segment(
                    text="\n\n".join(chapter.paragraphs),
                    speaker="NARRATOR",
                    index=0,
                )
            ]
        else:
            # Build a chapter-local roster: spaCy finds names in this chapter,
            # then each is resolved to its book-level canonical.  Names that
            # spaCy surfaces but that don't appear in the book roster are
            # discarded — keeping the per-chapter list a strict subset of the
            # book-wide character set.
            chapter_text = " ".join(chapter.paragraphs)
            raw_names = extract_person_names(chapter_text, nlp)
            chapter_canonicals = sorted({
                canonical
                for n in raw_names
                if (canonical := alias_to_canonical.get(n.lower())) is not None
            })
            roster_names = chapter_canonicals + ["NARRATOR", "Unknown"]

            # Chunk and attribute (pass aliases so the LLM prompt shows them).
            chunks = chunk_paragraphs(chapter.paragraphs, quotes)
            all_attributions = attribute_all_chunks(
                chunks, quotes, roster_names, llm, roster_aliases=roster_aliases
            )

            # Normalize LLM-returned speaker names back to canonicals.
            for item in all_attributions.values():
                item.speaker = _normalize_speaker(
                    item.speaker, alias_to_canonical, chapter_canonicals
                )

            # Build segments from paragraphs + attributions.
            segments = _build_segments(chapter.paragraphs, quotes, all_attributions)

            for item in all_attributions.values():
                if item.speaker not in ("NARRATOR", "Unknown"):
                    attribution_counts[item.speaker] += 1

        attributed_chapters.append(replace(chapter, segments=segments))

    # ---- Build CharacterInfo list ------------------------------------------
    from .entities import infer_gender_pronouns

    characters: list[CharacterInfo] = [
        CharacterInfo(
            character_id=group.canonical,
            display_name=group.canonical,
            quote_count=attribution_counts.get(group.canonical, 0),
            gender_pronoun=infer_gender_pronouns(
                group.canonical, group.aliases, full_text
            ),
        )
        for group in roster.characters
    ]
    characters.sort(key=lambda c: c.quote_count, reverse=True)

    return NLPResult(
        characters=characters,
        chapters=attributed_chapters,
        book_hash=book_hash(book_path),
    )


# ---------------------------------------------------------------------------
# Segment assembly helper
# ---------------------------------------------------------------------------


def _build_segments(paragraphs: list[str], quotes: list, attributions: dict) -> list:
    """Convert paragraphs + quote attributions into a flat Segment list.

    Strategy:
    - Walk paragraphs in order.
    - If a paragraph contains at least one attributed quote, emit it as a
      character Segment using the speaker of the longest quote in that para.
    - Consecutive un-attributed paragraphs are merged into a single NARRATOR
      Segment to minimise TTS call overhead.
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
        if para_idx not in para_to_attr:
            narrator_buf.append(para)
        else:
            _flush_narrator()
            # Pick the attribution of the longest quote in this paragraph.
            _, best_attr = max(para_to_attr[para_idx], key=lambda x: len(x[0].text))
            segments.append(Segment(text=para, speaker=best_attr.speaker, index=seg_idx))
            seg_idx += 1

    _flush_narrator()
    return segments


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
    "get_cached_result",
    "cache_result",
    "CACHE_DIR",
    "book_hash",
]
