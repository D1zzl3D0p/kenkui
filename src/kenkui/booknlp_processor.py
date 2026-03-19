"""BookNLP integration for multi-voice audiobook generation.

This module wraps the BookNLP NLP pipeline to extract speaker-attributed
speech segments from ebook chapters.  It is intentionally structured as
an optional feature: if the ``booknlp`` package is not installed,
``BOOKNLP_AVAILABLE`` will be ``False`` and callers can present the user
with install instructions rather than crashing.

Usage::

    from kenkui.booknlp_processor import BOOKNLP_AVAILABLE, run_analysis

    if BOOKNLP_AVAILABLE:
        result = run_analysis(chapters, book_path, model_size="small",
                              progress_callback=lambda msg: print(msg))
        cache_path = cache_result(result, book_path)
    else:
        # Show install instructions to user

Design notes
------------
- BookNLP operates on a flat plain-text file.  We write all chapters to a
  temp file, recording the byte range of each paragraph so we can map
  BookNLP token IDs back to (chapter_index, paragraph_index) tuples after
  analysis completes.

- The ``.quotes`` output file produced by BookNLP lists every attributed
  direct-speech span with its speaker's coreference ID.  We use this to
  split each chapter into alternating NARRATOR and character segments.

- Consecutive narrator paragraphs that are not covered by any quote span
  are merged into a single Segment to reduce the number of TTS calls.

- Results are cached as JSON in ``~/.config/kenkui/booknlp_cache/`` keyed
  by a SHA-256 hash of (book path + mtime).  A cache hit skips re-analysis
  entirely.
"""

from __future__ import annotations

import hashlib
import json
import logging
import subprocess
import sys
import tempfile
import traceback
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Lazy import guard - checked at runtime, not import time
# ---------------------------------------------------------------------------

_booknlp_available: bool | None = None


def BOOKNLP_AVAILABLE() -> bool:
    """Check whether BookNLP is fully operational: package importable AND spaCy model present.

    Two separate conditions must both be true before we consider BookNLP ready:

    1. ``booknlp`` and its Python dependencies are importable.
    2. The ``en_core_web_sm`` spaCy model is installed.  BookNLP calls
       ``spacy.load("en_core_web_sm")`` inside its *constructor* (not at
       import time), so a successful ``from booknlp.booknlp import BookNLP``
       does NOT guarantee that analysis will succeed.  Without this second
       check, ``BOOKNLP_AVAILABLE()`` returns ``True`` even when the model is
       absent, causing ``NarrationModeScreen`` to bypass ``MultiVoiceSetupScreen``
       and route the user directly to ``BookNLPAnalysisScreen`` where
       ``BookNLP.__init__`` immediately raises an ``OSError``.

    The result is cached so the import+model check only runs once per session.
    Call :func:`reset_booknlp_check` to invalidate the cache (e.g. after
    ``MultiVoiceSetupScreen`` finishes installing the model).
    """
    global _booknlp_available
    if _booknlp_available is not None:
        logger.debug(f"BOOKNLP_AVAILABLE → cached result={_booknlp_available}")
        return _booknlp_available

    logger.debug("BOOKNLP_AVAILABLE → no cached result, running full check")
    try:
        from booknlp.booknlp import BookNLP  # noqa: F401

        logger.debug("BOOKNLP_AVAILABLE → booknlp package import: OK")

        # Package imported OK — now verify the spaCy model is actually loadable.
        # BookNLP calls spacy.load(_SPACY_MODEL) in its constructor, so if the
        # model is absent the user will hit an OSError at analysis time even
        # though the import above succeeded.
        import spacy as _spacy  # type: ignore[import]

        model_present = _spacy.util.is_package(_SPACY_MODEL)
        logger.debug(
            f"BOOKNLP_AVAILABLE → spacy={getattr(_spacy, '__version__', 'unknown')} "
            f"en_core_web_sm present={model_present}"
        )
        if not model_present:
            logger.debug(f"BOOKNLP_AVAILABLE → False (en_core_web_sm not installed)")
            logger.debug(
                "BookNLP package is importable but spaCy model '%s' is not installed.",
                _SPACY_MODEL,
            )
            _booknlp_available = False
            return _booknlp_available

        logger.debug("BOOKNLP_AVAILABLE → True (package importable + model present)")
        _booknlp_available = True
    except Exception as exc:
        logger.debug(
            f"BOOKNLP_AVAILABLE → False (exception: {type(exc).__name__}: {exc})\n"
            f"{traceback.format_exc()}"
        )
        logger.debug("BookNLP not available: %s: %s", type(exc).__name__, exc)
        _booknlp_available = False

    return _booknlp_available


def reset_booknlp_check() -> None:
    """Reset the cached availability check.

    Useful for testing or if dependencies are installed after initial check.
    """
    global _booknlp_available
    _booknlp_available = None


# ---------------------------------------------------------------------------
# Cache location (reuses the existing kenkui config directory)
# ---------------------------------------------------------------------------

from .config import CONFIG_DIR  # noqa: E402 – after optional import

CACHE_DIR = CONFIG_DIR / "booknlp_cache"

# ---------------------------------------------------------------------------
# Public result type
# ---------------------------------------------------------------------------


@dataclass
class BookNLPResult:
    """The output of a BookNLP analysis run.

    ``characters`` is sorted by ``quote_count`` descending so the most
    prominent speakers appear first in the MultiVoiceScreen.

    ``chapters`` are the same Chapter objects passed in but with their
    ``.segments`` list populated.

    ``book_hash`` is the cache key used to locate the on-disk JSON file.
    """

    characters: list  # list[CharacterInfo] — avoid circular import at module level
    chapters: list  # list[Chapter]
    book_hash: str


# ---------------------------------------------------------------------------
# Cache helpers
# ---------------------------------------------------------------------------


def _book_hash(book_path: Path) -> str:
    """Compute a short SHA-256 hash of path + mtime for cache invalidation."""
    try:
        mtime = str(book_path.stat().st_mtime)
    except OSError:
        mtime = "0"
    raw = f"{book_path}{mtime}"
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def get_cached_result(book_path: Path) -> BookNLPResult | None:
    """Return a cached BookNLPResult if a valid cache file exists, else None."""
    h = _book_hash(book_path)
    cache_file = CACHE_DIR / f"{h}.json"
    if not cache_file.exists():
        return None
    try:
        data = json.loads(cache_file.read_text(encoding="utf-8"))
        return _result_from_dict(data)
    except Exception as exc:
        logger.warning("Failed to load BookNLP cache %s: %s", cache_file, exc)
        return None


def cache_result(result: BookNLPResult, book_path: Path) -> Path:
    """Serialise a BookNLPResult to disk and return the cache file path."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_file = CACHE_DIR / f"{result.book_hash}.json"
    data = _result_to_dict(result)
    cache_file.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.debug("BookNLP cache written: %s", cache_file)
    return cache_file


def _result_to_dict(result: BookNLPResult) -> dict[str, Any]:
    from .models import CharacterInfo

    return {
        "book_hash": result.book_hash,
        "characters": [
            {
                "character_id": c.character_id,
                "display_name": c.display_name,
                "quote_count": c.quote_count,
                "gender_pronoun": c.gender_pronoun,
            }
            for c in result.characters
        ],
        "chapters": [ch.to_dict() for ch in result.chapters],
    }


def _result_from_dict(data: dict[str, Any]) -> BookNLPResult:
    from .models import Chapter, CharacterInfo

    characters = [
        CharacterInfo(
            character_id=c["character_id"],
            display_name=c["display_name"],
            quote_count=c.get("quote_count", 0),
            gender_pronoun=c.get("gender_pronoun", ""),
        )
        for c in data.get("characters", [])
    ]
    chapters = [Chapter.from_dict(ch) for ch in data.get("chapters", [])]
    return BookNLPResult(
        characters=characters,
        chapters=chapters,
        book_hash=data["book_hash"],
    )


# ---------------------------------------------------------------------------
# Text serialisation helpers
# ---------------------------------------------------------------------------


@dataclass
class _ParaLocation:
    """Records byte span of a paragraph in the BookNLP input file."""

    chapter_index: int
    para_index: int
    byte_start: int
    byte_end: int


def _write_input_file(chapters: list, file_path: Path) -> list[_ParaLocation]:
    """Write all chapter paragraphs to a plain-text file for BookNLP.

    Returns a list of _ParaLocation entries (one per paragraph) in file order.
    Paragraphs are separated by double newlines so BookNLP treats them as
    separate blocks.

    Implementation note: the file is opened in **binary** mode so that
    ``fh.tell()`` returns true byte offsets.  Text-mode ``tell()`` on
    Windows returns opaque position cookies that are not comparable to the
    raw byte offsets BookNLP records in its ``.tokens`` output file.  By
    writing UTF-8 bytes directly we guarantee the recorded ``byte_start``
    and ``byte_end`` values align with BookNLP's ``byte_onset``/``byte_offset``
    columns regardless of platform.
    """
    _SEPARATOR = b"\n\n"
    locations: list[_ParaLocation] = []
    with open(file_path, "wb") as fh:
        for chapter in chapters:
            for para_idx, para in enumerate(chapter.paragraphs):
                encoded = para.encode("utf-8")
                start = fh.tell()
                fh.write(encoded)
                end = fh.tell()
                locations.append(
                    _ParaLocation(
                        chapter_index=chapter.index,
                        para_index=para_idx,
                        byte_start=start,
                        byte_end=end,
                    )
                )
                fh.write(_SEPARATOR)
    return locations


# ---------------------------------------------------------------------------
# Quote-file parsing
# ---------------------------------------------------------------------------


def _parse_quotes_file(quotes_path: Path) -> list[dict[str, Any]]:
    """Parse BookNLP .quotes TSV file into a list of quote dicts.

    Columns (tab-separated, with header row):
        quote_start  quote_end  mention_start  mention_end
        mention_text  char_id  quote_text
    """
    quotes: list[dict[str, Any]] = []
    if not quotes_path.exists():
        return quotes
    try:
        lines = quotes_path.read_text(encoding="utf-8").splitlines()
    except Exception as exc:
        logger.warning("Could not read quotes file %s: %s", quotes_path, exc)
        return quotes

    # Skip header line
    for line in lines[1:]:
        parts = line.split("\t")
        if len(parts) < 7:
            continue
        try:
            quotes.append(
                {
                    "quote_start": int(parts[0]),
                    "quote_end": int(parts[1]),
                    "char_id": parts[5].strip(),
                    "quote_text": parts[6].strip(),
                }
            )
        except (ValueError, IndexError):
            continue
    return quotes


def _parse_tokens_file(tokens_path: Path) -> list[dict[str, Any]]:
    """Parse BookNLP .tokens TSV file.

    We only need token_id_doc and byte_onset / byte_offset.
    Columns (tab-separated, with header row):
        paragraph_ID  sentence_ID  token_ID_within_sentence  token_ID_doc
        word  lemma  byte_onset  byte_offset  POS  dep  dep_head  event
    """
    tokens: list[dict[str, Any]] = []
    if not tokens_path.exists():
        return tokens
    try:
        lines = tokens_path.read_text(encoding="utf-8").splitlines()
    except Exception as exc:
        logger.warning("Could not read tokens file %s: %s", tokens_path, exc)
        return tokens

    for line in lines[1:]:
        parts = line.split("\t")
        if len(parts) < 8:
            continue
        try:
            tokens.append(
                {
                    "token_id": int(parts[3]),
                    "byte_onset": int(parts[6]),
                    "byte_offset": int(parts[7]),
                }
            )
        except (ValueError, IndexError):
            continue
    return tokens


# ---------------------------------------------------------------------------
# Token-ID → paragraph location mapping
# ---------------------------------------------------------------------------


def _build_token_to_para_map(
    tokens: list[dict[str, Any]],
    locations: list[_ParaLocation],
) -> dict[int, _ParaLocation]:
    """Map each token_id to the paragraph it belongs to by byte offset."""
    mapping: dict[int, _ParaLocation] = {}
    for tok in tokens:
        onset = tok["byte_onset"]
        # Binary search over sorted paragraph locations
        loc = _find_para_by_byte(locations, onset)
        if loc is not None:
            mapping[tok["token_id"]] = loc
    return mapping


def _find_para_by_byte(locations: list[_ParaLocation], byte_pos: int) -> _ParaLocation | None:
    """Return the _ParaLocation whose byte range contains byte_pos."""
    lo, hi = 0, len(locations) - 1
    while lo <= hi:
        mid = (lo + hi) // 2
        loc = locations[mid]
        if byte_pos < loc.byte_start:
            hi = mid - 1
        elif byte_pos > loc.byte_end:
            lo = mid + 1
        else:
            return loc
    return None


# ---------------------------------------------------------------------------
# Segment assembly
# ---------------------------------------------------------------------------


def _assemble_segments(
    chapter,  # Chapter
    chapter_quotes: list[dict[str, Any]],
    token_to_para: dict[int, _ParaLocation],
    para_to_quote: dict[int, str],
) -> list:  # list[Segment]
    """Build the segment list for a single chapter.

    Strategy:
    1. Determine which (para_index) rows are covered by at least one quote,
       and record which character ID spoke them.
    2. Walk paragraphs in order.  Consecutive un-quoted paragraphs are merged
       into one NARRATOR Segment.  Quoted paragraphs each become their own
       character Segment.

    ``para_to_quote`` maps para_index → character_id for paragraphs that
    contain attributed direct speech.
    """
    from .models import Segment

    segments: list[Segment] = []
    narrator_buf: list[str] = []
    seg_idx = 0

    def _flush_narrator():
        nonlocal seg_idx
        if narrator_buf:
            segments.append(
                Segment(
                    text="\n\n".join(narrator_buf),
                    speaker="NARRATOR",
                    index=seg_idx,
                )
            )
            seg_idx += 1
            narrator_buf.clear()

    for para_idx, para in enumerate(chapter.paragraphs):
        char_id = para_to_quote.get(para_idx)
        if char_id is not None:
            _flush_narrator()
            segments.append(Segment(text=para, speaker=char_id, index=seg_idx))
            seg_idx += 1
        else:
            narrator_buf.append(para)

    _flush_narrator()
    return segments


def _build_para_to_quote(
    chapter_index: int,
    quotes: list[dict[str, Any]],
    token_to_para: dict[int, _ParaLocation],
    char_id_map: dict[str, str] | None = None,
) -> dict[int, str]:
    """Return {para_index: character_id} for paragraphs with attributed quotes.

    Args:
        chapter_index:  Only quotes whose start token falls in this chapter
                        are included.
        quotes:         Parsed rows from the BookNLP ``.quotes`` file.
        token_to_para:  Mapping from token_id to its _ParaLocation.
        char_id_map:    Optional mapping from raw BookNLP char_id (e.g. ``"42"``)
                        to the composite ``CharacterInfo.character_id``
                        (e.g. ``"ELIZABETH_BENNETT-42"``).  When provided,
                        ``Segment.speaker`` will carry the composite id so that
                        ``speaker_voices`` lookups in the worker succeed.
                        When ``None`` the raw char_id is stored (legacy behaviour).
    """
    para_to_quote: dict[int, str] = {}
    for q in quotes:
        # Map quote start token to a paragraph location
        start_loc = token_to_para.get(q["quote_start"])
        if start_loc is None or start_loc.chapter_index != chapter_index:
            continue
        raw_id = q["char_id"]
        if not raw_id or raw_id.strip() == "-1":
            continue
        # Resolve to the composite coref id used as CharacterInfo.character_id
        # so that speaker_voices dict lookups in workers.py succeed.
        resolved = (char_id_map or {}).get(raw_id)
        char_id: str = resolved if resolved is not None else raw_id
        # Only record the first attribution for a given paragraph
        if start_loc.para_index not in para_to_quote:
            para_to_quote[start_loc.para_index] = char_id
    return para_to_quote


# ---------------------------------------------------------------------------
# Character extraction from BookNLP .book JSON
# ---------------------------------------------------------------------------


def _extract_characters(book_json_path: Path) -> list:
    """Parse the .book JSON file and return a sorted list of CharacterInfo.

    BookNLP 1.0.8 output schema (actual):
    ::

        {
          "characters": [
            {
              "id": 13,
              "mentions": {
                "proper":  [{"n": "Jane", "c": 9}],
                "common":  [],
                "pronoun": [{"n": "her", "c": 8}, ...]
              },
              "count": 25,           # total mention count
              "g": {
                "argmax": "she/her",
                "inference": {...}
              }
            }
          ]
        }

    Older / hypothetical schema used ``"names"`` instead of
    ``"mentions.proper"`` and ``"quoteCount"`` instead of ``"count"``.
    Both are supported for forward-compatibility.
    """
    from .models import CharacterInfo

    if not book_json_path.exists():
        return []
    try:
        data = json.loads(book_json_path.read_text(encoding="utf-8"))
    except Exception as exc:
        logger.warning("Could not read .book JSON %s: %s", book_json_path, exc)
        return []

    characters: list[CharacterInfo] = []
    for char_data in data.get("characters", []):
        char_id: str = str(char_data.get("id", ""))

        # ── Display name ──────────────────────────────────────────────────
        # BookNLP 1.0.8: mentions.proper is a list of {"n": str, "c": int}
        # where "c" is the frequency of that name form.  Fall back to the
        # legacy "names" field if present (older schema).
        mentions: dict = char_data.get("mentions", {})
        name_candidates: list[dict] = mentions.get("proper", []) or char_data.get("names", [])

        display_name = ""
        if name_candidates:
            top = max(name_candidates, key=lambda n: n.get("c", 0))
            display_name = str(top.get("n") or "")

        if not display_name:
            # Fall back to the most common pronoun form when no proper name found
            pronoun_candidates: list[dict] = mentions.get("pronoun", [])
            if pronoun_candidates:
                top_p = max(pronoun_candidates, key=lambda n: n.get("c", 0))
                display_name = str(top_p.get("n") or "")

        if not display_name:
            display_name = char_id

        # ── Prominence / quote count ──────────────────────────────────────
        # BookNLP 1.0.8 uses "count" (total mentions).  Older schema used
        # "quoteCount".  We prefer "quoteCount" when present (more precise),
        # otherwise fall back to "count".
        quote_count: int = (
            char_data["quoteCount"]
            if char_data.get("quoteCount") is not None
            else int(char_data.get("count", 0) or 0)
        )

        # ── Referential gender ────────────────────────────────────────────
        gender: str = char_data.get("g", {}).get("argmax", "") if char_data.get("g") else ""

        # ── Stable coref-style composite ID ──────────────────────────────
        coref_id = f"{display_name.upper().replace(' ', '_')}-{char_id}"

        characters.append(
            CharacterInfo(
                character_id=coref_id,
                display_name=display_name,
                quote_count=quote_count,
                gender_pronoun=gender,
            )
        )

    # Sort by prominence descending so the most-spoken characters appear first
    # in MultiVoiceScreen.
    characters.sort(key=lambda c: c.quote_count, reverse=True)
    return characters


# ---------------------------------------------------------------------------
# spaCy model auto-download
# ---------------------------------------------------------------------------

_SPACY_MODEL = "en_core_web_sm"


def _get_spacy_model_wheel_url() -> str:
    """Return the wheel URL for ``en_core_web_sm`` matching the installed spaCy version.

    Mirrors the URL-construction logic inside ``spacy.cli.download`` so we can
    invoke an installer directly rather than going through spaCy's CLI, which
    calls ``sys.exit()`` on failure and therefore cannot be used safely inside
    a background thread.
    """
    try:
        import spacy.about as _about  # type: ignore[import]
        from spacy.cli.download import (  # type: ignore[import]
            get_compatibility,
            get_model_filename,
            get_version,
        )
        from urllib.parse import urljoin

        compat = get_compatibility()
        version = get_version(_SPACY_MODEL, compat)
        filename = get_model_filename(_SPACY_MODEL, version, sdist=False)
        base_url = _about.__download_url__
        if not base_url.endswith("/"):
            base_url += "/"
        return urljoin(base_url, filename)
    except Exception as exc:
        # Fall back to a pinned URL that matches spaCy 3.8.x
        logger.debug("Could not resolve spaCy model URL dynamically: %s", exc)
        return (
            "https://github.com/explosion/spacy-models/releases/download/"
            f"{_SPACY_MODEL}-3.8.0/{_SPACY_MODEL}-3.8.0-py3-none-any.whl"
        )


def ensure_spacy_model(
    progress_callback: Callable[[str], None] | None = None,
) -> None:
    """Ensure the required spaCy language model is present, installing it if not.

    BookNLP calls ``spacy.load("en_core_web_sm")`` inside its constructor, so
    the model must be installed before :func:`run_analysis` is called.

    **Why we do not use ``spacy.cli.download``:**
    ``spacy.cli.download`` installs the model by running
    ``sys.executable -m pip install <url>`` via ``subprocess``.  When ``pip``
    is absent (e.g. in a ``uv``-managed tool environment, which is the default
    install method for kenkui), the subprocess exits with code 1 and
    ``spacy.cli.run_command`` calls ``sys.exit(1)``.  ``sys.exit`` raises
    ``SystemExit``, a *subclass of BaseException, not Exception*, so it
    escapes any ``except Exception`` handler — including the one in the
    background download thread — causing the thread to die silently while the
    UI spinner runs forever.

    Instead we build the wheel URL ourselves and invoke the best available
    installer directly, raising a clean ``RuntimeError`` (never ``SystemExit``)
    on failure so the UI can surface a proper error message.

    Installer priority:
    1. ``uv pip install --python <sys.executable> <url>`` — works in uv envs
    2. ``<sys.executable> -m pip install <url>`` — works in standard venvs

    Args:
        progress_callback: Optional callable for UI status updates.

    Raises:
        RuntimeError: If spaCy itself is not importable, or if the model
                      cannot be installed by any available installer.
    """
    logger.debug("ensure_spacy_model: entry")
    try:
        import spacy  # type: ignore[import]
    except ImportError as exc:
        logger.debug(f"ensure_spacy_model: spaCy not importable — {exc}")
        raise RuntimeError(
            "spaCy is not installed. Install it with: pip install kenkui[multivoice]"
        ) from exc

    model_present = spacy.util.is_package(_SPACY_MODEL)
    logger.debug(
        f"ensure_spacy_model: spacy={getattr(spacy, '__version__', 'unknown')} "
        f"python={sys.executable} "
        f"model_present={model_present}"
    )

    if model_present:
        logger.debug("spaCy model '%s' already present.", _SPACY_MODEL)
        logger.debug("ensure_spacy_model: model already present, nothing to do")
        return

    def _progress(msg: str) -> None:
        logger.info(msg)
        logger.debug(f"ensure_spacy_model progress: {msg}")
        if progress_callback:
            progress_callback(msg)

    wheel_url = _get_spacy_model_wheel_url()
    logger.debug(f"ensure_spacy_model: wheel URL = {wheel_url}")
    _progress(f"Downloading spaCy model '{_SPACY_MODEL}' (one-time setup, ~12 MB)…")
    logger.debug("spaCy model wheel URL: %s", wheel_url)

    # Each candidate is a complete argv list.  We try them in order and stop
    # at the first one that succeeds (returncode == 0).
    candidates = [
        # uv-managed environments (no pip available)
        ["uv", "pip", "install", "--python", sys.executable, wheel_url],
        # Standard venvs / pip-enabled environments
        [sys.executable, "-m", "pip", "install", wheel_url],
    ]

    last_error: str = ""
    for cmd in candidates:
        logger.debug(f"ensure_spacy_model: trying installer: {' '.join(cmd[:3])} ...")
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
            )
            if result.returncode == 0:
                logger.debug(f"ensure_spacy_model: installer succeeded: {cmd[0]}")
                logger.debug("Installed '%s' via: %s", _SPACY_MODEL, " ".join(cmd[:2]))
                break
            last_error = (result.stderr or result.stdout or "").strip()
            logger.debug(
                f"ensure_spacy_model: installer FAILED code={result.returncode} "
                f"cmd={cmd[0]}\nstdout={result.stdout[:400]}\nstderr={result.stderr[:400]}"
            )
            logger.debug(
                "Installer '%s' failed (code %d): %s",
                cmd[0],
                result.returncode,
                last_error[:200],
            )
        except FileNotFoundError:
            logger.debug(f"ensure_spacy_model: installer not found (FileNotFoundError): {cmd[0]}")
            logger.debug("Installer not found: %s", cmd[0])
            continue
        except SystemExit as exc:
            # Guard against third-party code (e.g. a monkey-patched subprocess)
            # calling sys.exit() — convert to RuntimeError so the caller always
            # gets a catchable Exception rather than a silent thread death.
            logger.debug(
                f"ensure_spacy_model: installer called sys.exit({exc.code}) — "
                f"converting to RuntimeError"
            )
            raise RuntimeError(
                f"Installer '{cmd[0]}' called sys.exit({exc.code}). "
                f"Install manually: uv pip install --python {sys.executable} {wheel_url}"
            ) from exc
    else:
        # All candidates exhausted without success
        logger.debug(f"ensure_spacy_model: ALL installers failed.\nlast_error={last_error}")
        raise RuntimeError(
            f"Could not install spaCy model '{_SPACY_MODEL}'.\n"
            f"Last error: {last_error}\n\n"
            f"Install manually with one of:\n"
            f"  uv pip install --python {sys.executable} {wheel_url}\n"
            f"  pip install {wheel_url}"
        )

    # Final verification — the installer may have succeeded without the
    # importable package appearing (edge case: wrong --target, etc.)
    model_present_after = spacy.util.is_package(_SPACY_MODEL)
    logger.debug(f"ensure_spacy_model: post-install model_present={model_present_after}")
    if not model_present_after:
        raise RuntimeError(
            f"spaCy model '{_SPACY_MODEL}' still not importable after install.\n"
            f"Try manually: uv pip install --python {sys.executable} {wheel_url}"
        )

    _progress(f"spaCy model '{_SPACY_MODEL}' installed successfully.")
    logger.debug("ensure_spacy_model: complete OK")


# ---------------------------------------------------------------------------
# Main public entry point
# ---------------------------------------------------------------------------


def run_analysis(
    chapters: list,
    book_path: Path,
    model_size: str = "small",
    progress_callback: Callable[[str], None] | None = None,
) -> BookNLPResult:
    """Run BookNLP analysis on a list of Chapter objects.

    Args:
        chapters:          Chapters to analyse (from the existing ebook readers).
        book_path:         Original ebook file path (used for cache key only).
        model_size:        ``"small"`` (faster, CPU-friendly) or ``"big"``
                           (more accurate, better with GPU / multi-core).
        progress_callback: Optional callable receiving human-readable status
                           strings for display in the UI.

    Returns:
        A :class:`BookNLPResult` with ``characters`` populated and each
        Chapter's ``.segments`` list filled in.

    Raises:
        RuntimeError: If BookNLP is not available.
    """
    logger.debug(
        f"run_analysis: entry — book={book_path} model_size={model_size} "
        f"chapters={len(chapters)} thread={__import__('threading').current_thread().name}"
    )

    avail = BOOKNLP_AVAILABLE()
    logger.debug(f"run_analysis: BOOKNLP_AVAILABLE()={avail}")
    if not avail:
        logger.debug("run_analysis: raising RuntimeError — BookNLP not available")
        raise RuntimeError(
            "BookNLP is not available. Run kenkui and use Multi-Voice to trigger setup."
        )

    from booknlp.booknlp import BookNLP  # type: ignore[import]

    logger.debug("run_analysis: booknlp import OK")

    def _progress(msg: str) -> None:
        logger.debug("BookNLP: %s", msg)
        logger.debug(f"run_analysis progress: {msg}")
        if progress_callback:
            progress_callback(msg)

    book_hash = _book_hash(book_path)
    logger.debug(f"run_analysis: book_hash={book_hash}")
    _progress("Starting BookNLP analysis…")

    with tempfile.TemporaryDirectory(prefix="kenkui_booknlp_") as tmp_str:
        tmp = Path(tmp_str)
        input_txt = tmp / "book.txt"
        output_dir = tmp / "output"
        output_dir.mkdir()
        book_id = "book"

        # --- Write plain-text input file and record paragraph locations ---
        _progress("Preparing text for analysis…")
        locations = _write_input_file(chapters, input_txt)
        logger.debug(
            f"run_analysis: input file written — "
            f"{input_txt.stat().st_size} bytes, {len(locations)} para locations"
        )

        # --- Run BookNLP ---
        model_params = {
            "pipeline": "entity,quote,coref",
            "model": model_size,
        }
        # Phase 1 of 2: load model weights into memory.  This is the slow part
        # (~8–15 s on first run per session, faster on repeat due to OS page
        # cache).  Reporting it as a distinct phase gives the user a clear
        # signal that something is happening and sets expectations correctly.
        _progress(f"[1/2] Loading NLP models ({model_size})…")

        # Compatibility shim for transformers ≥5.0 / booknlp 1.0.8
        # ----------------------------------------------------------------
        # booknlp's BERT checkpoints were saved with transformers ~4.x, where
        # BertEmbeddings registered `position_ids` as a *persistent* buffer
        # (included in state_dict).  In transformers 5.x it became
        # non-persistent (excluded from state_dict).  PyTorch's default
        # strict=True load therefore raises:
        #   RuntimeError: Unexpected key(s) in state_dict: "bert.embeddings.position_ids"
        # Passing strict=False tells PyTorch to ignore that extra key.
        # position_ids is a simple pre-computed arange buffer — not learned
        # weights — so dropping it is safe and the model behaves identically.
        import torch.nn as _nn

        _orig_load_state_dict = _nn.Module.load_state_dict
        logger.debug("run_analysis: patching nn.Module.load_state_dict for transformers 5.x compat")

        def _permissive_load_state_dict(self, state_dict, strict=True, **kwargs):  # type: ignore[override]
            return _orig_load_state_dict(self, state_dict, strict=False, **kwargs)

        _nn.Module.load_state_dict = _permissive_load_state_dict  # type: ignore[method-assign]
        try:
            # Pre-initialize tqdm's multiprocessing write lock before the first
            # tqdm() call inside BookNLP's BertModel.from_pretrained() weight
            # loading.  tqdm creates this lock lazily (on first tqdm() call).
            # When that first call happens inside a daemon thread on macOS,
            # the lazy creation triggers multiprocessing.resource_tracker to
            # spawn a background subprocess via fork_exec — which fails with:
            #   ValueError: bad value(s) in fds_to_keep
            # because Textual's event loop file descriptors are open in the
            # parent but invalid in the forked child.
            # create_mp_lock() is idempotent (guards with hasattr), so calling
            # it here from the worker thread, before tqdm is ever used, ensures
            # the lock exists and tqdm never attempts the deferred fork.
            import tqdm.std as _tqdm_std

            _tqdm_std.TqdmDefaultWriteLock.create_mp_lock()
            logger.debug("run_analysis: tqdm mp_lock pre-initialized")

            logger.debug("run_analysis: calling BookNLP() constructor …")
            booknlp_instance = BookNLP("en", model_params)
            logger.debug("run_analysis: BookNLP() constructor SUCCEEDED")
        except Exception as exc:
            logger.debug(
                f"run_analysis: BookNLP() constructor FAILED — "
                f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}"
            )
            raise
        finally:
            _nn.Module.load_state_dict = _orig_load_state_dict  # type: ignore[method-assign]
            logger.debug("run_analysis: load_state_dict patch removed")

        # Phase 2 of 2: run the NLP pipeline over the book text.  Time scales
        # roughly linearly with word count (~0.5 s/10 k words for small model).
        _progress("[2/2] Analysing text…")
        logger.debug("run_analysis: calling booknlp_instance.process() …")
        booknlp_instance.process(str(input_txt), str(output_dir), book_id)
        logger.debug("run_analysis: booknlp_instance.process() COMPLETED")

        # --- Parse outputs ---
        _progress("Parsing BookNLP output…")
        tokens_path = output_dir / f"{book_id}.tokens"
        quotes_path = output_dir / f"{book_id}.quotes"
        book_json_path = output_dir / f"{book_id}.book"

        tokens = _parse_tokens_file(tokens_path)
        quotes = _parse_quotes_file(quotes_path)
        characters = _extract_characters(book_json_path)
        logger.debug(
            f"run_analysis: parsed — "
            f"tokens={len(tokens)} quotes={len(quotes)} characters={len(characters)}"
        )

        # --- Build token → paragraph mapping ---
        _progress("Mapping speakers to paragraphs…")
        token_to_para = _build_token_to_para_map(tokens, locations)
        logger.debug(f"run_analysis: token_to_para entries={len(token_to_para)}")

        # Build raw-char_id → composite-coref_id lookup so that Segment.speaker
        # values match the CharacterInfo.character_id keys used in speaker_voices.
        # e.g. "42" → "ELIZABETH_BENNETT-42"
        # The composite id is constructed by _extract_characters() from the
        # display name, so we parse it from the character list here.
        char_id_map: dict[str, str] = {}
        for char in characters:
            # character_id format: "DISPLAY_NAME-raw_id"  (see _extract_characters)
            raw = char.character_id.rsplit("-", 1)[-1]
            char_id_map[raw] = char.character_id
        logger.debug(
            f"run_analysis: char_id_map built — {len(char_id_map)} entries: {list(char_id_map.items())[:5]}"
        )

        # --- Assemble segments per chapter ---
        _progress("Building voice segments…")
        annotated_chapters = []
        for chapter in chapters:
            para_to_quote = _build_para_to_quote(chapter.index, quotes, token_to_para, char_id_map)
            segments = _assemble_segments(chapter, quotes, token_to_para, para_to_quote)
            # Clone chapter with segments populated
            import copy

            ch_copy = copy.copy(chapter)
            ch_copy.segments = segments
            annotated_chapters.append(ch_copy)

        logger.debug(
            f"run_analysis: segments assembled — "
            f"{sum(len(c.segments or []) for c in annotated_chapters)} total segments"
        )
        _progress("Analysis complete.")

    logger.debug(
        f"run_analysis: returning BookNLPResult — "
        f"characters={len(characters)} chapters={len(annotated_chapters)}"
    )
    return BookNLPResult(
        characters=characters,
        chapters=annotated_chapters,
        book_hash=book_hash,
    )


__all__ = [
    "BOOKNLP_AVAILABLE",
    "reset_booknlp_check",
    "BookNLPResult",
    "ensure_spacy_model",
    "run_analysis",
    "get_cached_result",
    "cache_result",
]
