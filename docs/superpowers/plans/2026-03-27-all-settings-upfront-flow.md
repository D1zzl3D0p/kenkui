# All-Settings-Upfront Conversion Flow — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Restructure the wizard so all passive settings are collected before any NLP runs, then a fast entity scan surfaces characters for voice assignment, then unattended processing handles the slow LLM attribution.

**Architecture:** Split `run_analysis()` at the stage boundary into `run_fast_scan()` (Stage 1-2, seconds, no LLM attribution) and `run_attribution()` (Stage 3-4, minutes). The wizard calls the fast scan; the worker calls attribution as a pre-TTS phase. A `prominence` property on `CharacterInfo` makes all sort/display code work regardless of which count is populated.

**Tech Stack:** Python dataclasses, Pydantic (existing nlp/models.py), spaCy, Ollama (via LLMClient), InquirerPy, Rich

---

## File Map

| File | Change |
|------|--------|
| `src/kenkui/models.py` | Add `mention_count`, `prominence` to `CharacterInfo`; add `FastScanResult`; add `roster_cache_path` to `JobConfig` |
| `src/kenkui/nlp/__init__.py` | Add `run_fast_scan()`, `run_attribution()`, `cache_roster()`, `get_cached_roster()`; refactor `run_analysis()` |
| `src/kenkui/cli/add.py` | Reorder wizard steps; add `_run_fast_scan_wizard()`; update all sort keys to `prominence` |
| `src/kenkui/server/worker.py` | Add pre-TTS attribution phase in `_process_job()` |
| `tests/test_models.py` | Extend existing `TestCharacterInfo` and add `TestJobConfig`, `TestFastScanResult` |
| `tests/test_nlp_pipeline_split.py` | New: test `cache_roster`, `get_cached_roster`, mention counting |

---

## Task 1: Add `mention_count` and `prominence` to `CharacterInfo`

**Files:**
- Modify: `src/kenkui/models.py:62-91`
- Test: `tests/test_models.py:185-202`

- [ ] **Step 1: Write the failing tests**

In `tests/test_models.py`, add to the existing `TestCharacterInfo` class:

```python
def test_mention_count_default_zero(self):
    c = CharacterInfo(character_id="Alice", display_name="Alice")
    assert c.mention_count == 0

def test_mention_count_round_trip(self):
    c = CharacterInfo(character_id="Alice", display_name="Alice", mention_count=150)
    restored = CharacterInfo.from_dict(c.to_dict())
    assert restored.mention_count == 150

def test_mention_count_backward_compat(self):
    """Old cache files without mention_count should load with 0."""
    c = CharacterInfo.from_dict({"character_id": "Alice"})
    assert c.mention_count == 0

def test_prominence_prefers_mention_count(self):
    c = CharacterInfo(character_id="Alice", display_name="Alice",
                      mention_count=100, quote_count=5)
    assert c.prominence == 100

def test_prominence_falls_back_to_quote_count(self):
    c = CharacterInfo(character_id="Alice", display_name="Alice",
                      mention_count=0, quote_count=42)
    assert c.prominence == 42

def test_prominence_zero_when_both_zero(self):
    c = CharacterInfo(character_id="Alice", display_name="Alice")
    assert c.prominence == 0
```

- [ ] **Step 2: Run the tests to verify they fail**

```bash
uv run pytest tests/test_models.py::TestCharacterInfo -v
```

Expected: 6 failures (`mention_count`, `prominence` not defined)

- [ ] **Step 3: Implement the changes in `src/kenkui/models.py`**

Replace the `CharacterInfo` dataclass (lines 62-91):

```python
@dataclass
class CharacterInfo:
    """Metadata for a character identified by the NLP pipeline.

    Used for per-character voice assignment in multi-voice mode.
    Not persisted directly in JobConfig — only the resulting
    ``speaker_voices`` mapping is stored.
    """

    character_id: str  # Canonical name, e.g. "Elizabeth Bennet"
    display_name: str  # Human-readable label shown in the UI
    quote_count: int = 0
    mention_count: int = 0  # Name occurrences in full text; populated by fast scan
    gender_pronoun: str = ""  # "he", "she", "they", etc. (optional)

    @property
    def prominence(self) -> int:
        """Best available count for sorting — prefer mention_count, fall back to quote_count."""
        return self.mention_count or self.quote_count

    def to_dict(self) -> dict[str, Any]:
        return {
            "character_id": self.character_id,
            "display_name": self.display_name,
            "quote_count": self.quote_count,
            "mention_count": self.mention_count,
            "gender_pronoun": self.gender_pronoun,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "CharacterInfo":
        return cls(
            character_id=data["character_id"],
            display_name=data.get("display_name", data["character_id"]),
            quote_count=data.get("quote_count", 0),
            mention_count=data.get("mention_count", 0),
            gender_pronoun=data.get("gender_pronoun", ""),
        )
```

- [ ] **Step 4: Run the tests to verify they pass**

```bash
uv run pytest tests/test_models.py::TestCharacterInfo -v
```

Expected: all pass

- [ ] **Step 5: Commit**

```bash
git add src/kenkui/models.py tests/test_models.py
git commit -m "feat: add mention_count and prominence to CharacterInfo"
```

---

## Task 2: Add `FastScanResult` to `kenkui.models`

**Files:**
- Modify: `src/kenkui/models.py` (add after `NLPResult`)
- Test: `tests/test_models.py` (add new `TestFastScanResult` class)

First, find `NLPResult` in `src/kenkui/models.py` to know the line number to insert after it. It contains `characters`, `chapters`, `book_hash`.

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_models.py`:

```python
class TestFastScanResult:
    def _make_result(self):
        from kenkui.models import FastScanResult
        from kenkui.nlp.models import AliasGroup, CharacterRoster
        roster = CharacterRoster(characters=[
            AliasGroup(canonical="Alice", aliases=["Alice", "Al"], gender="she/her"),
        ])
        chars = [CharacterInfo(
            character_id="Alice", display_name="Alice", mention_count=99
        )]
        return FastScanResult(roster=roster, characters=chars, book_hash="abc")

    def test_round_trip(self):
        from kenkui.models import FastScanResult
        result = self._make_result()
        restored = FastScanResult.from_dict(result.to_dict())
        assert restored.book_hash == "abc"
        assert restored.characters[0].mention_count == 99
        assert restored.roster.characters[0].canonical == "Alice"

    def test_roster_aliases_preserved(self):
        from kenkui.models import FastScanResult
        result = self._make_result()
        restored = FastScanResult.from_dict(result.to_dict())
        assert "Al" in restored.roster.characters[0].aliases

    def test_empty_result(self):
        from kenkui.models import FastScanResult
        from kenkui.nlp.models import CharacterRoster
        empty = FastScanResult(
            roster=CharacterRoster(characters=[]),
            characters=[],
            book_hash="",
        )
        restored = FastScanResult.from_dict(empty.to_dict())
        assert restored.characters == []
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
uv run pytest tests/test_models.py::TestFastScanResult -v
```

Expected: `ImportError` — `FastScanResult` not defined

- [ ] **Step 3: Add `FastScanResult` to `src/kenkui/models.py`**

Find where `NLPResult` ends (look for `__all__` or the next class definition after it). Add immediately after `NLPResult`:

```python
@dataclass
class FastScanResult:
    """Result of the Stage 1-2 fast scan (no LLM attribution).

    Contains the character roster with name-mention counts. Used by the wizard
    for voice assignment and passed to the worker via ``roster_cache_path`` so
    Stage 3-4 attribution can use the same canonical names.
    """

    roster: "CharacterRoster"  # Pydantic model from nlp.models
    characters: list[CharacterInfo]  # Sorted by mention_count descending
    book_hash: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "book_hash": self.book_hash,
            "roster": self.roster.model_dump(),
            "characters": [c.to_dict() for c in self.characters],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "FastScanResult":
        from .nlp.models import CharacterRoster
        return cls(
            book_hash=data.get("book_hash", ""),
            roster=CharacterRoster.model_validate(data["roster"]),
            characters=[CharacterInfo.from_dict(c) for c in data.get("characters", [])],
        )
```

Add `"CharacterRoster"` to the `TYPE_CHECKING` imports block if one exists, or leave as a string annotation (already done above).

- [ ] **Step 4: Run tests to verify they pass**

```bash
uv run pytest tests/test_models.py::TestFastScanResult -v
```

Expected: all pass

- [ ] **Step 5: Commit**

```bash
git add src/kenkui/models.py tests/test_models.py
git commit -m "feat: add FastScanResult dataclass"
```

---

## Task 3: Add `roster_cache_path` to `JobConfig`

**Files:**
- Modify: `src/kenkui/models.py:135-209`
- Test: `tests/test_models.py` (add `TestJobConfigRosterCachePath`)

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_models.py`:

```python
class TestJobConfigRosterCachePath:
    def _make_job(self, roster_path=None):
        from kenkui.models import JobConfig
        return JobConfig(
            ebook_path=Path("/tmp/book.epub"),
            roster_cache_path=roster_path,
        )

    def test_default_is_none(self):
        job = self._make_job()
        assert job.roster_cache_path is None

    def test_round_trip_with_path(self):
        from kenkui.models import JobConfig
        job = self._make_job(Path("/tmp/cache/roster.json"))
        restored = JobConfig.from_dict(job.to_dict())
        assert restored.roster_cache_path == Path("/tmp/cache/roster.json")

    def test_round_trip_with_none(self):
        from kenkui.models import JobConfig
        job = self._make_job(None)
        restored = JobConfig.from_dict(job.to_dict())
        assert restored.roster_cache_path is None

    def test_backward_compat_missing_field(self):
        """Old queue.toml without roster_cache_path should load cleanly."""
        from kenkui.models import JobConfig
        old_data = {"ebook_path": "/tmp/book.epub"}
        job = JobConfig.from_dict(old_data)
        assert job.roster_cache_path is None
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
uv run pytest tests/test_models.py::TestJobConfigRosterCachePath -v
```

Expected: 4 failures — `roster_cache_path` not a field

- [ ] **Step 3: Add `roster_cache_path` to `JobConfig` in `src/kenkui/models.py`**

In the `JobConfig` dataclass, add after `annotated_chapters_path` (line 145):

```python
roster_cache_path: Path | None = None  # Path to Stage 2 roster cache (set by wizard fast scan)
```

In `JobConfig.to_dict()`, add alongside `annotated_chapters_path`:

```python
"roster_cache_path": str(self.roster_cache_path) if self.roster_cache_path else None,
```

In `JobConfig.from_dict()`, add alongside `annotated_chapters_path`:

```python
roster_cache_path=Path(data["roster_cache_path"])
if data.get("roster_cache_path")
else None,
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
uv run pytest tests/test_models.py::TestJobConfigRosterCachePath -v
```

Expected: all pass

- [ ] **Step 5: Run full test suite to check no regressions**

```bash
uv run pytest tests/test_models.py -v
```

Expected: all pass

- [ ] **Step 6: Commit**

```bash
git add src/kenkui/models.py tests/test_models.py
git commit -m "feat: add roster_cache_path to JobConfig"
```

---

## Task 4: Add `cache_roster` / `get_cached_roster` to `nlp/__init__.py`

**Files:**
- Modify: `src/kenkui/nlp/__init__.py` (after line 113, before `run_analysis`)
- Test: `tests/test_nlp_pipeline_split.py` (new file)

- [ ] **Step 1: Write the failing tests**

Create `tests/test_nlp_pipeline_split.py`:

```python
"""Tests for the split NLP pipeline: cache_roster / get_cached_roster."""
from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch
import pytest


def _make_fast_scan_result():
    from kenkui.models import CharacterInfo, FastScanResult
    from kenkui.nlp.models import AliasGroup, CharacterRoster
    roster = CharacterRoster(characters=[
        AliasGroup(canonical="Alice", aliases=["Alice"], gender="she/her"),
    ])
    characters = [CharacterInfo(character_id="Alice", display_name="Alice", mention_count=50)]
    return FastScanResult(roster=roster, characters=characters, book_hash="testhash")


class TestRosterCache:
    def test_cache_and_retrieve(self, tmp_path):
        from kenkui.nlp import cache_roster, get_cached_roster

        # Make a fake ebook path; book_hash uses path + mtime
        ebook = tmp_path / "book.epub"
        ebook.write_text("fake")

        result = _make_fast_scan_result()

        with patch("kenkui.nlp.CONFIG_DIR", tmp_path):
            with patch("kenkui.config.CONFIG_DIR", tmp_path):
                cache_path = cache_roster(result, ebook)
                assert cache_path.exists()
                assert cache_path.name.endswith("-roster.json")

                restored = get_cached_roster(ebook)
                assert restored is not None
                assert restored.characters[0].mention_count == 50
                assert restored.roster.characters[0].canonical == "Alice"

    def test_get_cached_roster_returns_none_when_missing(self, tmp_path):
        from kenkui.nlp import get_cached_roster

        ebook = tmp_path / "book.epub"
        ebook.write_text("fake")

        with patch("kenkui.nlp.CONFIG_DIR", tmp_path):
            with patch("kenkui.config.CONFIG_DIR", tmp_path):
                result = get_cached_roster(ebook)
                assert result is None

    def test_get_cached_roster_returns_none_on_corrupt_file(self, tmp_path):
        from kenkui.nlp import book_hash, get_cached_roster

        ebook = tmp_path / "book.epub"
        ebook.write_text("fake")

        cache_dir = tmp_path / "nlp_cache"
        cache_dir.mkdir()
        corrupt = cache_dir / f"{book_hash(ebook)}-roster.json"
        corrupt.write_text("not json {{{{")

        with patch("kenkui.nlp.CONFIG_DIR", tmp_path):
            with patch("kenkui.config.CONFIG_DIR", tmp_path):
                result = get_cached_roster(ebook)
                assert result is None
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
uv run pytest tests/test_nlp_pipeline_split.py::TestRosterCache -v
```

Expected: `ImportError` — `cache_roster`, `get_cached_roster` not in `kenkui.nlp`

- [ ] **Step 3: Add `cache_roster` and `get_cached_roster` to `src/kenkui/nlp/__init__.py`**

After `cache_result` (line 113), add:

```python
def get_cached_roster(book_path: Path) -> "FastScanResult | None":
    """Return a cached ``FastScanResult`` if a valid roster cache file exists, else None."""
    from ..config import CONFIG_DIR
    from ..models import FastScanResult

    cache_dir = CONFIG_DIR / "nlp_cache"
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
    from ..config import CONFIG_DIR

    cache_dir = CONFIG_DIR / "nlp_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file = cache_dir / f"{book_hash(book_path)}-roster.json"
    cache_file.write_text(
        json.dumps(result.to_dict(), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    logger.debug("Roster cache written: %s", cache_file)
    return cache_file
```

Also add `"get_cached_roster"` and `"cache_roster"` to the `__all__` list at the bottom of the file.

- [ ] **Step 4: Run tests to verify they pass**

```bash
uv run pytest tests/test_nlp_pipeline_split.py::TestRosterCache -v
```

Expected: all pass

- [ ] **Step 5: Commit**

```bash
git add src/kenkui/nlp/__init__.py tests/test_nlp_pipeline_split.py
git commit -m "feat: add cache_roster / get_cached_roster to NLP module"
```

---

## Task 5: Add `run_fast_scan()` to `nlp/__init__.py`

**Files:**
- Modify: `src/kenkui/nlp/__init__.py`
- Test: `tests/test_nlp_pipeline_split.py`

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_nlp_pipeline_split.py`:

```python
class TestMentionCounting:
    """Test the internal _count_mentions helper."""

    def test_counts_canonical_occurrences(self):
        from kenkui.nlp import _count_mentions
        from kenkui.nlp.models import AliasGroup, CharacterRoster

        roster = CharacterRoster(characters=[
            AliasGroup(canonical="Alice", aliases=["Alice", "Al"]),
            AliasGroup(canonical="Bob", aliases=["Bob"]),
        ])
        text = "Alice went to the store. Al was there too. Bob walked in. Alice left."
        counts = _count_mentions(roster, text)
        assert counts["Alice"] == 3  # "Alice" x2 + "Al" x1
        assert counts["Bob"] == 1

    def test_case_insensitive(self):
        from kenkui.nlp import _count_mentions
        from kenkui.nlp.models import AliasGroup, CharacterRoster

        roster = CharacterRoster(characters=[
            AliasGroup(canonical="Alice", aliases=["alice"]),
        ])
        counts = _count_mentions(roster, "ALICE said hello to alice.")
        assert counts["Alice"] == 2

    def test_word_boundary(self):
        from kenkui.nlp import _count_mentions
        from kenkui.nlp.models import AliasGroup, CharacterRoster

        roster = CharacterRoster(characters=[
            AliasGroup(canonical="Al", aliases=["Al"]),
        ])
        # "Al" should not match inside "Alice" or "pal"
        counts = _count_mentions(roster, "Alice and Al and pal and Al")
        assert counts["Al"] == 2

    def test_empty_text(self):
        from kenkui.nlp import _count_mentions
        from kenkui.nlp.models import AliasGroup, CharacterRoster

        roster = CharacterRoster(characters=[
            AliasGroup(canonical="Alice", aliases=["Alice"]),
        ])
        counts = _count_mentions(roster, "")
        assert counts["Alice"] == 0
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
uv run pytest tests/test_nlp_pipeline_split.py::TestMentionCounting -v
```

Expected: `ImportError` — `_count_mentions` not defined

- [ ] **Step 3: Add `_count_mentions` and `run_fast_scan()` to `src/kenkui/nlp/__init__.py`**

After `cache_roster`, add the helper:

```python
def _count_mentions(roster: "CharacterRoster", full_text: str) -> dict[str, int]:
    """Count word-boundary occurrences of each character's aliases in *full_text*.

    Returns a mapping of canonical name → total mention count across all aliases.
    """
    import re

    counts: dict[str, int] = {}
    for group in roster.characters:
        total = 0
        for alias in group.aliases:
            pattern = re.compile(r"\b" + re.escape(alias) + r"\b", re.IGNORECASE)
            total += len(pattern.findall(full_text))
        counts[group.canonical] = total
    return counts
```

Then add `run_fast_scan()` before `run_analysis()`:

```python
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
    from .entities import build_roster_with_llm, infer_gender_pronouns
    from .llm import LLMClient
    from .quotes import extract_quotes

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
    try:
        nlp = spacy.load("en_core_web_sm")
    except OSError:
        raise RuntimeError(
            "spaCy model 'en_core_web_sm' not found. "
            "Install it with: uv pip install https://github.com/explosion/spacy-models"
            "/releases/download/en_core_web_sm-3.8.0/en_core_web_sm-3.8.0-py3-none-any.whl"
        )

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

    def _resolve_gender(group) -> str:
        if group.gender and group.gender.lower() not in ("", "unknown"):
            return group.gender
        return infer_gender_pronouns(group.canonical, group.aliases, full_text)

    characters: list[CharacterInfo] = [
        CharacterInfo(
            character_id=group.canonical,
            display_name=group.canonical,
            mention_count=mention_counts.get(group.canonical, 0),
            gender_pronoun=_resolve_gender(group),
        )
        for group in roster.characters
    ]
    characters.sort(key=lambda c: c.mention_count, reverse=True)

    result = FastScanResult(roster=roster, characters=characters, book_hash=book_hash(book_path))
    cache_roster(result, book_path)
    return result
```

Note: Stage 1 quote extraction is not needed in the fast scan output itself — quotes are re-extracted by `run_attribution` when it runs. The comment above explains this.

Add `"run_fast_scan"` and `"_count_mentions"` to `__all__`.

- [ ] **Step 4: Run tests to verify they pass**

```bash
uv run pytest tests/test_nlp_pipeline_split.py::TestMentionCounting -v
```

Expected: all pass

- [ ] **Step 5: Commit**

```bash
git add src/kenkui/nlp/__init__.py tests/test_nlp_pipeline_split.py
git commit -m "feat: add run_fast_scan and _count_mentions to NLP module"
```

---

## Task 6: Add `run_attribution()` and refactor `run_analysis()`

**Files:**
- Modify: `src/kenkui/nlp/__init__.py`

No new tests needed — `run_analysis()` is integration-tested via `tests/test_full_flow.py`; the refactor must not change its observable output. The key invariant: `run_analysis()` output before and after refactor must be identical.

- [ ] **Step 1: Add `run_attribution()` before `run_analysis()`**

```python
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
    """
    import spacy
    import time
    from collections import defaultdict
    from dataclasses import replace as _replace

    from ..models import Chapter, CharacterInfo, NLPResult, Segment
    from .attribution import attribute_all_chunks
    from .chunker import chunk_paragraphs
    from .entities import extract_person_names, infer_gender_pronouns
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
    try:
        nlp = spacy.load("en_core_web_sm")
    except OSError:
        raise RuntimeError(
            "spaCy model 'en_core_web_sm' not found. "
            "Install it with: uv pip install https://github.com/explosion/spacy-models"
            "/releases/download/en_core_web_sm-3.8.0/en_core_web_sm-3.8.0-py3-none-any.whl"
        )

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
    def _resolve_gender(group) -> str:
        if group.gender and group.gender.lower() not in ("", "unknown"):
            return group.gender
        return infer_gender_pronouns(group.canonical, group.aliases, full_text)

    characters: list[CharacterInfo] = [
        CharacterInfo(
            character_id=group.canonical,
            display_name=group.canonical,
            quote_count=attribution_counts.get(group.canonical, 0),
            gender_pronoun=_resolve_gender(group),
        )
        for group in roster.characters
    ]
    characters.sort(key=lambda c: c.quote_count, reverse=True)

    return NLPResult(
        characters=characters,
        chapters=attributed_chapters,
        book_hash=book_hash(book_path),
    )
```

Add `"run_attribution"` to `__all__`.

- [ ] **Step 2: Refactor `run_analysis()` to delegate to the two new functions**

Replace the body of `run_analysis()` (lines 121-301) with:

```python
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
    from dataclasses import replace as _replace
    nlp_result = _replace(
        nlp_result,
        characters=[
            _replace(c, mention_count=mention_by_id.get(c.character_id, 0))
            for c in nlp_result.characters
        ],
    )

    cache_result(nlp_result, book_path)
    return nlp_result
```

- [ ] **Step 3: Run existing NLP tests**

```bash
uv run pytest tests/test_nlp_quotes.py tests/test_nlp_chunker.py tests/test_nlp_attribution.py tests/test_nlp_segments.py tests/test_nlp_booknlp_roster.py -v
```

Expected: all pass (no changes to these units)

- [ ] **Step 4: Commit**

```bash
git add src/kenkui/nlp/__init__.py
git commit -m "feat: add run_attribution; refactor run_analysis to delegate to split pipeline"
```

---

## Task 7: Update wizard step order in `add.py`

**Files:**
- Modify: `src/kenkui/cli/add.py`

This task restructures `_run_wizard()` (lines 900-1094). The logic inside each step is mostly unchanged — the key changes are:
1. Quality overrides and output directory move before the multi-voice block
2. `_run_nlp_analysis()` is replaced with a new `_run_fast_scan_wizard()` wrapper
3. `annotated_chapters_path` is no longer set in the wizard; `roster_cache_path` is set instead

- [ ] **Step 1: Add `_run_fast_scan_wizard()` helper above `_run_wizard()`**

Add after `_run_nlp_analysis()` (around line 368):

```python
def _run_fast_scan_wizard(
    book_path: Path,
    nlp_model: str,
    use_cache: bool = True,
    chapter_selection: dict | None = None,
):
    """Run Stage 1-2 fast scan with a progress spinner. Returns FastScanResult or None."""
    from ..nlp import cache_roster, get_cached_roster, run_fast_scan
    from ..readers import get_reader

    # Check roster cache first
    if use_cache:
        cached = get_cached_roster(book_path)
        if cached is not None:
            console.print("[green]Using cached character roster.[/green]")
            return cached

    console.print(f"[cyan]Scanning characters in '{book_path.name}'…[/cyan]")

    try:
        reader = get_reader(book_path, verbose=False)
        all_chapters = reader.get_chapters()
    except Exception as exc:
        console.print(f"[red]Could not read ebook: {exc}[/red]")
        return None

    # Restrict to selected chapters
    included_indices: set[int] | None = None
    if chapter_selection:
        raw = chapter_selection.get("included")
        if raw:
            included_indices = set(raw)

    if included_indices:
        chapters = [ch for ch in all_chapters if ch.index in included_indices]
        if not chapters:
            chapters = all_chapters
    else:
        chapters = all_chapters

    result = None
    with Progress(
        SpinnerColumn(),
        TextColumn("{task.description}"),
        TimeElapsedColumn(),
        console=console,
    ) as prog:
        task = prog.add_task("Scanning characters…", total=None)
        try:
            result = run_fast_scan(
                chapters=chapters,
                book_path=book_path,
                nlp_model=nlp_model,
                use_cache=False,  # cache already checked above
                progress_callback=lambda msg: prog.update(task, description=msg),
            )
            prog.update(task, description="Character scan complete")
        except Exception as exc:
            console.print(f"[red]Character scan failed: {exc}[/red]")
            return None

    return result
```

- [ ] **Step 2: Rewrite `_run_wizard()` with the new step order**

Replace the body of `_run_wizard()` (lines 900-1094) with:

```python
def _run_wizard(book_path: Path, app_config) -> dict | None:
    """Run the interactive wizard.

    Returns a kwargs dict suitable for client.add_job(), or None if the user
    cancels.
    """
    from InquirerPy import inquirer

    from ..models import NarrationMode

    console.rule(f"[bold]kenkui — {book_path.name}[/bold]")

    # Step 1 — Chapter preset / custom selection.
    chapter_selection = _prompt_chapter_preset_and_selection(book_path)

    # Step 2 — Narration mode.
    mode_val = inquirer.select(
        message="Narration mode:",
        choices=[
            {"name": "Single Voice", "value": "single"},
            {"name": "Multi-Voice  (per-character, requires local LLM)", "value": "multi"},
            {"name": "Chapter Voice  (assign a voice per chapter)", "value": "chapter"},
        ],
    ).execute()

    # Step 3 — Quality overrides (passive, no computation).
    quality_overrides = _prompt_quality_overrides(app_config)

    # Step 4 — Output directory (passive).
    default_out = str(app_config.default_output_dir or book_path.parent)
    output_dir = (
        inquirer.text(
            message="Output directory:",
            default=default_out,
        )
        .execute()
        .strip()
        or default_out
    )

    # --- Mode-specific setup ---
    speaker_voices: dict[str, str] = {}
    chapter_voices: dict[str, str] = {}
    roster_cache_path: str | None = None
    narration_mode = mode_val

    if mode_val == "multi":
        from ..config import save_app_config, DEFAULT_CONFIG_PATH
        from ..nlp import CACHE_DIR, book_hash, get_cached_roster
        from ..nlp.setup import check_llm_available, run_setup_dialogue

        # Step 5a — Show requirements status and confirm readiness.
        console.print()
        if not _check_multivoice_requirements(app_config):
            console.print("[yellow]Falling back to single-voice mode.[/yellow]")
            narration_mode = "single"
            mode_val = "single"

        if mode_val == "multi":
            # Step 5b — Ensure spaCy model is available.
            if not _ensure_spacy():
                console.print("[red]Cannot proceed without spaCy model.[/red]")
                return None

            # Step 5c — NLP model: show current model and offer reconfigure.
            console.print()
            console.print(
                f"NLP model for speaker inference: [bold]{app_config.nlp_model}[/bold]"
            )
            reconfigure_action = inquirer.select(
                message="Continue with this model or reconfigure?",
                choices=[
                    {"name": f"Continue with {app_config.nlp_model}", "value": "continue"},
                    {"name": "Reconfigure NLP model…", "value": "reconfigure"},
                ],
            ).execute()

            if reconfigure_action == "reconfigure" or not check_llm_available(app_config):
                updated = run_setup_dialogue(app_config)
                if updated is None:
                    console.print("[yellow]Falling back to single-voice mode.[/yellow]")
                    narration_mode = "single"
                else:
                    app_config = updated
                    save_app_config(app_config, DEFAULT_CONFIG_PATH)
                    console.print(
                        f"[green]NLP model set to [bold]{app_config.nlp_model}[/bold][/green]"
                    )

        if narration_mode == "multi":
            # Step 6 — Fast character scan (Stage 1-2 only, seconds).
            use_cache = True
            if get_cached_roster(book_path) is not None:
                use_cache = inquirer.select(
                    message="Cached character roster found for this book.",
                    choices=[
                        {"name": "Use cached roster  (fast)", "value": True},
                        {
                            "name": "Regenerate  (re-runs spaCy + LLM roster building)",
                            "value": False,
                        },
                    ],
                ).execute()

            fast_result = _run_fast_scan_wizard(
                book_path,
                app_config.nlp_model,
                use_cache=use_cache,
                chapter_selection=chapter_selection,
            )
            if fast_result is None:
                console.print("[yellow]Falling back to single-voice mode.[/yellow]")
                narration_mode = "single"
            else:
                # Step 7 — Character voice assignment (simple or advanced).
                speaker_voices = _prompt_multivoice_character_voices(
                    fast_result, app_config.default_voice
                )
                roster_cache_path = str(CACHE_DIR / f"{book_hash(book_path)}-roster.json")

    elif mode_val == "chapter":
        # Chapter-voice mode: assign a voice per chapter.
        narration_mode = "single"
        console.print()
        console.print("[bold]Voice for narrator / unassigned chapters:[/bold]")
        voice = _prompt_voice(default=app_config.default_voice, message="Default voice:")
        _check_hf_auth(voice)
        try:
            from ..readers import get_reader
            reader = get_reader(book_path, verbose=False)
            all_chapters = reader.get_chapters()
            chapter_voices = _prompt_chapter_voices(all_chapters, voice)
        except Exception as exc:
            console.print(f"[red]Could not load chapters for assignment: {exc}[/red]")
            chapter_voices = {}

    # Step 8 — Narrator voice (single/chapter modes; multi already selected it).
    if narration_mode == "multi":
        voice = speaker_voices.get("NARRATOR", app_config.default_voice)
    elif mode_val != "chapter":
        console.print()
        console.print("[bold]Voice:[/bold]")
        voice = _prompt_voice(default=app_config.default_voice, message="Select voice:")
        _check_hf_auth(voice)

    # Step 9 — Confirmation table.
    console.print()
    summary = Table(title="Job Summary", show_header=False, box=None)
    summary.add_column("Field", style="bold", width=20)
    summary.add_column("Value")
    summary.add_row("Book", str(book_path))
    preset_label = chapter_selection.get("preset", "content-only")
    included = chapter_selection.get("included", [])
    if included:
        summary.add_row("Chapters", f"{preset_label} ({len(included)} selected)")
    else:
        summary.add_row("Chapters", preset_label)
    display_mode = mode_val if mode_val != "chapter" else "chapter-voice"
    summary.add_row("Mode", display_mode)
    summary.add_row("Narrator voice", voice)
    if narration_mode == "multi" and speaker_voices:
        non_narrator = {k: v for k, v in speaker_voices.items() if k != "NARRATOR"}
        summary.add_row("Character voices", f"{len(non_narrator)} assigned")
    if chapter_voices:
        summary.add_row("Chapter voices", f"{len(chapter_voices)} chapters assigned")
    if quality_overrides:
        summary.add_row("Quality overrides", ", ".join(
            f"{k.replace('job_', '')}={v}" for k, v in quality_overrides.items()
        ))
    summary.add_row("Output", output_dir)
    console.print(summary)
    console.print()

    confirmed = inquirer.confirm(message="Queue this job?", default=True).execute()
    if not confirmed:
        console.print("Cancelled.")
        return None

    job_kwargs = dict(
        ebook_path=str(book_path),
        voice=voice,
        chapter_selection=chapter_selection,
        output_path=output_dir,
        narration_mode=narration_mode,
        speaker_voices=speaker_voices or None,
        annotated_chapters_path=None,     # Worker populates this during processing
        roster_cache_path=roster_cache_path,
        chapter_voices=chapter_voices or None,
        **quality_overrides,
    )
    return job_kwargs
```

- [ ] **Step 3: Update `_prompt_multivoice_character_voices` to work with `FastScanResult`**

`FastScanResult` has no `.chapters` attribute (chapters aren't attributed yet). The function currently does:

```python
speaker_voices = _resolve_chapter_voice_conflicts(
    speaker_voices, characters, nlp_result.chapters, male_pool, female_pool, narrator_voice
)
```

`_get_chapter_cooccurrence` (called by `_resolve_chapter_voice_conflicts`) checks `ch.segments`; with no attributed chapters it returns `{}` — meaning conflict resolution is a no-op. This is acceptable: post-attribution conflict resolution can be a future improvement.

Change the single `nlp_result.chapters` reference to:

```python
chapters = getattr(scan_result, "chapters", None) or []
speaker_voices = _resolve_chapter_voice_conflicts(
    speaker_voices, characters, chapters, male_pool, female_pool, narrator_voice
)
```

Also rename the parameter: `def _prompt_multivoice_character_voices(scan_result, default_voice: str):`
and replace `nlp_result.characters` with `scan_result.characters`.

- [ ] **Step 4: Manually test the wizard flow end-to-end**

```bash
uv run kenkui /path/to/any-epub.epub
```

Walk through the wizard:
1. Select chapters
2. Select multi-voice mode
3. Set quality overrides (or skip)
4. Set output directory
5. Confirm requirements / NLP model
6. Observe fast scan spinner (seconds, not minutes)
7. Complete voice assignment
8. Confirm and queue

Verify the job appears in queue: `uv run kenkui queue list`

- [ ] **Step 5: Commit**

```bash
git add src/kenkui/cli/add.py
git commit -m "feat: reorder wizard to all-settings-upfront; replace full NLP with fast scan"
```

---

## Task 8: Update sort keys to use `prominence` throughout `add.py`

**Files:**
- Modify: `src/kenkui/cli/add.py`

Four functions use `quote_count` for sorting/display and must be updated to use `c.prominence`.

- [ ] **Step 1: Update `_prompt_character_voice_review()` (lines 533-598)**

Change every reference to `quote_count` to use `.prominence`:

Line ~555: `for ch in sorted(characters, key=lambda c: c.quote_count, reverse=True):`
→ `for ch in sorted(characters, key=lambda c: c.prominence, reverse=True):`

Line ~560: `{"name": f"{ch.display_name} ({ch.quote_count} quotes, {gender})  →  {current}",`
→ `{"name": f"{ch.display_name} ({ch.prominence} mentions, {gender})  →  {current}",`

Line ~593-595 (the review_choices update inside the while loop):
```python
rc["name"] = (
    f"{ch.display_name} ({ch.quote_count} quotes, {gender})  →  {new_voice}"
)
```
→
```python
rc["name"] = (
    f"{ch.display_name} ({ch.prominence} mentions, {gender})  →  {new_voice}"
)
```

- [ ] **Step 2: Update `_auto_assign_character_voices()` (lines 483-530)**

Line ~506: `for ch in sorted(characters, key=lambda c: c.quote_count, reverse=True):`
→ `for ch in sorted(characters, key=lambda c: c.prominence, reverse=True):`

Lines ~512, 516: `male_quotes += ch.quote_count` and `female_quotes += ch.quote_count`
→ `male_quotes += ch.prominence` and `female_quotes += ch.prominence`

Lines ~521, 524: same pattern in the else branch.

- [ ] **Step 3: Update `_resolve_chapter_voice_conflicts()` (lines 420-480)**

Line ~439: `char_quotes: dict[str, int] = {ch.character_id: ch.quote_count for ch in characters}`
→ `char_quotes: dict[str, int] = {ch.character_id: ch.prominence for ch in characters}`

- [ ] **Step 4: Update `_top_gender_matched_voice()` (lines 371-401)**

Line ~379: `by_quotes = sorted(characters, key=lambda c: c.quote_count, reverse=True)`
→ `by_quotes = sorted(characters, key=lambda c: c.prominence, reverse=True)`

Lines ~386-390:
```python
if g == "male":
    top_male_quotes = ch.quote_count
    break
if g == "female":
    top_female_quotes = ch.quote_count
    break
```
→
```python
if g == "male":
    top_male_quotes = ch.prominence
    break
if g == "female":
    top_female_quotes = ch.prominence
    break
```

- [ ] **Step 5: Run existing tests**

```bash
uv run pytest tests/test_multi_voice_models.py tests/test_workers.py tests/test_cli.py -v
```

Expected: all pass

- [ ] **Step 6: Commit**

```bash
git add src/kenkui/cli/add.py
git commit -m "feat: use CharacterInfo.prominence for all sort/display in wizard"
```

---

## Task 9: Add pre-TTS attribution phase to the worker

**Files:**
- Modify: `src/kenkui/server/worker.py:267-288`

- [ ] **Step 1: Rewrite `_process_job()` to run attribution if `annotated_chapters_path` is None**

Replace `_process_job()` (lines 267-288):

```python
def _process_job(self, item: QueueItem):
    """Process a single job."""
    job = item.job

    try:
        # Pre-TTS phase: for multi-voice jobs, ensure speaker attribution is cached.
        if job.narration_mode.value == "multi" and not (
            job.annotated_chapters_path and job.annotated_chapters_path.exists()
        ):
            self._run_attribution_phase(item)
            # Reload job reference (attribution phase may have updated annotated_chapters_path)
            job = item.job

        cfg = self._build_config(job)
        builder = AudioBuilder(cfg, progress_callback=self._progress_callback)

        result = builder.run()

        if result:
            output_path = str(cfg.output_path / f"{job.name}.m4b")
            self.complete_job(item.id, output_path)
        else:
            self.fail_job(item.id, "Conversion failed")

    except AnnotatedChaptersCacheMissError as e:
        self.fail_job(item.id, f"CACHE_MISS: {e}")
    except Exception as e:
        self.fail_job(item.id, str(e))


def _run_attribution_phase(self, item: QueueItem) -> None:
    """Run Stage 3-4 speaker attribution and update item.job.annotated_chapters_path.

    Loads the roster from ``roster_cache_path`` if available; falls back to a
    fresh Stage 1-2 scan if not (e.g., job submitted via API without wizard).
    """
    from ..nlp import (
        CACHE_DIR,
        book_hash,
        cache_result,
        get_cached_result,
        run_attribution,
        run_fast_scan,
    )
    from ..readers import get_reader

    job = item.job
    book_path = job.ebook_path

    # Return early if full NLP cache already exists
    cached = get_cached_result(book_path)
    if cached is not None:
        h = book_hash(book_path)
        job.annotated_chapters_path = CACHE_DIR / f"{h}.json"
        return

    def _cb(msg: str) -> None:
        if self._progress_callback:
            self._progress_callback(0.0, f"Analyzing speakers: {msg}", 0)

    # Load chapters
    _cb("reading ebook…")
    try:
        reader = get_reader(book_path, verbose=False)
        all_chapters = reader.get_chapters()
    except Exception as exc:
        raise RuntimeError(f"Could not read ebook for attribution: {exc}") from exc

    # Filter to selected chapters
    included = set(job.chapter_selection.included)
    if included:
        chapters = [ch for ch in all_chapters if ch.index in included] or all_chapters
    else:
        chapters = all_chapters

    # Get roster — load from roster_cache_path, or re-run fast scan as fallback
    roster = None
    if job.roster_cache_path and job.roster_cache_path.exists():
        try:
            import json
            from ..models import FastScanResult
            data = json.loads(job.roster_cache_path.read_text(encoding="utf-8"))
            roster = FastScanResult.from_dict(data).roster
        except Exception as exc:
            logger.warning("Could not load roster cache %s: %s — re-scanning", job.roster_cache_path, exc)

    if roster is None:
        # Fallback: run fast scan to rebuild roster
        _cb("rebuilding character roster…")
        fast_result = run_fast_scan(
            chapters=chapters,
            book_path=book_path,
            nlp_model=self._app_config.nlp_model,
            use_cache=False,
            progress_callback=_cb,
        )
        roster = fast_result.roster

    # Run Stage 3-4 attribution
    nlp_result = run_attribution(
        roster=roster,
        chapters=chapters,
        book_path=book_path,
        nlp_model=self._app_config.nlp_model,
        use_cache=False,
        progress_callback=_cb,
    )

    cache_file = cache_result(nlp_result, book_path)
    job.annotated_chapters_path = cache_file
    self._save()
```

- [ ] **Step 2: Run worker tests**

```bash
uv run pytest tests/test_worker_server/ tests/test_workers.py -v
```

Expected: all pass

- [ ] **Step 3: End-to-end verification**

Run a full multi-voice job:
1. `uv run kenkui /path/to/book.epub` — complete wizard (fast scan + voice assignment)
2. Queue starts processing
3. Live dashboard shows "Analyzing speakers…" phase
4. Attribution runs (slow) then TTS generation begins
5. Job completes with M4B output

On second run of the same book, attribution is skipped (cache hit) — TTS begins immediately.

- [ ] **Step 4: Commit**

```bash
git add src/kenkui/server/worker.py
git commit -m "feat: run Stage 3-4 attribution in worker as pre-TTS phase"
```

---

## Task 10: Final regression check

- [ ] **Step 1: Run full test suite**

```bash
uv run pytest -v
```

Expected: all pass

- [ ] **Step 2: Verify single-voice wizard is unchanged**

Run `uv run kenkui /any/book.epub`, select single-voice. Confirm no NLP runs, job completes normally.

- [ ] **Step 3: Commit spec note**

Update spec status in `docs/superpowers/specs/2026-03-27-all-settings-upfront-flow-design.md`: Change `**Status:** Approved` to `**Status:** Implemented`.

```bash
git add docs/superpowers/specs/2026-03-27-all-settings-upfront-flow-design.md
git commit -m "docs: mark all-settings-upfront spec as implemented"
```
