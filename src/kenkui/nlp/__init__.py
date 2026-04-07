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

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Scene-break helpers (duplicated from workers.py to avoid circular import)
# ---------------------------------------------------------------------------

_SCENE_BREAK_RE = re.compile(
    r"^\s*(\*\s*){2,}\s*$"
    r"|^\s*[-\u2014]{2,}\s*$"
    r"|^\s*#\s*$",
)


def _is_scene_break(text: str) -> bool:
    """Return True if *text* is a scene-break marker or pure whitespace."""
    stripped = text.strip()
    return not stripped or bool(_SCENE_BREAK_RE.match(stripped))


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
    inferred = infer_gender_pronouns(group.canonical, group.aliases, full_text)
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


def get_cached_result(book_path: Path) -> "NLPResult | None":
    """Return a cached ``NLPResult`` if a valid cache file exists, else None."""
    from ..models import NLPResult

    cache_dir = _get_config_dir() / "nlp_cache"
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
    cache_dir = _get_config_dir() / "nlp_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file = cache_dir / f"{book_hash(book_path)}.json"
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


def get_cached_roster(book_path: Path) -> "FastScanResult | None":
    """Return a cached ``FastScanResult`` if a valid roster cache file exists, else None."""
    from ..models import FastScanResult

    cache_dir = _get_config_dir() / "nlp_cache"
    cache_file = cache_dir / f"{book_hash(book_path)}-roster.json"
    if not cache_file.exists():
        return None
    try:
        data = json.loads(cache_file.read_text(encoding="utf-8"))
        return FastScanResult.from_dict(data)
    except Exception as exc:
        logger.warning("Failed to load roster cache %s: %s", cache_file, exc)
        return None


def cache_roster(result: "FastScanResult", book_path: Path) -> Path:
    """Serialise *result* to disk and return the roster cache file path."""
    cache_dir = _get_config_dir() / "nlp_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file = cache_dir / f"{book_hash(book_path)}-roster.json"
    cache_file.write_text(
        json.dumps(result.to_dict(), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    logger.debug("Roster cache written: %s", cache_file)
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
        counts[group.canonical] = total
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
    roster = build_roster_with_llm(full_text, nlp, llm)

    char_names = ", ".join(g.canonical for g in roster.characters[:8])
    overflow = len(roster.characters) - 8
    suffix = f" (+{overflow} more)" if overflow > 0 else ""
    _cb(f"Character roster: {len(roster.characters)} characters — {char_names}{suffix}")

    # Count name mentions
    _cb("Counting character mentions…")
    mention_counts = _count_mentions(roster, full_text)

    characters: list[CharacterInfo] = [
        CharacterInfo(
            character_id=group.canonical,
            display_name=group.canonical,
            mention_count=mention_counts.get(group.canonical, 0),
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
    from .quotes import extract_quotes

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

    # Load spaCy
    _cb("Loading spaCy language model…")
    nlp = _load_spacy_model()

    full_text = " ".join(" ".join(ch.paragraphs) for ch in chapters)

    # Build alias → canonical lookup
    alias_to_canonical: dict[str, str] = {}
    for group in roster.characters:
        alias_to_canonical[group.canonical.lower()] = group.canonical
        for alias in group.aliases:
            alias_to_canonical[alias.lower()] = group.canonical

    roster_aliases: dict[str, list[str]] = {
        group.canonical: group.aliases for group in roster.characters
    }

    # Stage 1: Extract quotes per chapter
    _cb("Extracting dialogue quotes…")
    chapter_quotes: dict[int, list] = {}
    for chapter in chapters:
        chapter_quotes[chapter.index] = extract_quotes(chapter.paragraphs)

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

        if not quotes:
            segments = [
                Segment(
                    text="\n\n".join(chapter.paragraphs),
                    speaker="NARRATOR",
                    index=0,
                )
            ]
        else:
            chapter_text = " ".join(chapter.paragraphs)
            raw_names = extract_person_names(chapter_text, nlp)
            chapter_canonicals = sorted({
                canonical
                for n in raw_names
                if (canonical := alias_to_canonical.get(n.lower())) is not None
            })
            roster_names = chapter_canonicals + ["NARRATOR", "Unknown"]

            chunks = chunk_paragraphs(chapter.paragraphs, quotes)
            all_attributions = attribute_all_chunks(
                chunks, quotes, roster_names, llm, roster_aliases=roster_aliases
            )

            for item in all_attributions.values():
                item.speaker = _normalize_speaker(
                    item.speaker, alias_to_canonical, chapter_canonicals
                )

            segments = _build_segments(chapter.paragraphs, quotes, all_attributions)

            for item in all_attributions.values():
                if item.speaker not in ("NARRATOR", "Unknown"):
                    attribution_counts[item.speaker] += 1

        attributed_chapters.append(_replace(chapter, segments=segments))

    # Build CharacterInfo with quote_count
    characters: list[CharacterInfo] = [
        CharacterInfo(
            character_id=group.canonical,
            display_name=group.canonical,
            quote_count=attribution_counts.get(group.canonical, 0),
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
    "cache_roster",
    "CACHE_DIR",
    "book_hash",
    "_count_mentions",
    "_split_paragraph_by_quotes",
    "_merge_consecutive_segments",
    "_build_segments",
    "_is_scene_break",
    "_SCENE_BREAK_RE",
]
