# Design: All-Settings-Upfront Conversion Flow

**Date:** 2026-03-27
**Status:** Approved

## Problem

The current interactive wizard interleaves user decisions with computation. In multi-voice mode the wizard blocks on the full NLP pipeline (Stage 1–4, including LLM speaker attribution) before the user can assign voices. This means:

- User picks chapters and mode, then waits 5–20 minutes for LLM attribution to finish
- Voice assignment only becomes available after the slow phase completes
- Quality and output settings come after the voice screen — passive settings are buried behind computation

The goal is a clean linear flow: **all settings first, then extract characters, then assign voices, then let the rest run unattended.**

## Proposed Flow

```
TODAY
─────
Wizard: chapters → mode → [Stage 1-4 NLP, slow] → voice assignment → quality → output → submit
Worker: TTS generation → assembly

PROPOSED
────────
Wizard: chapters → mode → quality → output → [Stage 1-2, fast] → voice assignment → confirm → submit
Worker: [Stage 3-4 attribution] → TTS generation → assembly
```

The NLP pipeline is split at the stage boundary. Stage 1–2 (quote extraction + entity clustering, no LLM required) runs in the wizard in seconds. Stage 3–4 (LLM chunking + speaker attribution) moves to the worker as a pre-TTS phase.

## Wizard Step Reorder

| # | Step | Notes |
|---|------|-------|
| 1 | Chapter selection | Unchanged |
| 2 | Mode selection | Unchanged |
| 3 | Quality overrides | Moved up — was step 7 |
| 4 | Output directory | Moved up — was step 8 |
| 5 | Requirements check + NLP model config | Multi-voice only; unchanged logic |
| 6 | Fast scan (Stage 1–2) | Multi-voice only; replaces full NLP |
| 7 | Voice assignment | Multi-voice only; uses mention count |
| 8 | Narrator voice | Single/chapter modes only |
| 9 | Confirm + submit | Unchanged |

Steps 1–4 are all passive decisions requiring no computation. The fast scan (step 6) is the only blocking computation in the wizard and completes in seconds.

## NLP Pipeline Split

### New public API in `nlp/__init__.py`

**`run_fast_scan(book_path, chapter_selection, nlp_model, use_cache=True) → FastScanResult`**

Runs Stage 1–2 only:
- Stage 1: Quote extraction (pure regex, instant)
- Stage 2: Entity extraction + alias clustering (spaCy, seconds)
- Counts name mentions per canonical character
- Caches result to `nlp_cache/{book_hash}-roster.json`
- Returns `FastScanResult` containing the `CharacterRoster` with mention counts

**`run_attribution(book_path, roster, chapter_selection, nlp_model, use_cache=True) → NLPResult`**

Runs Stage 3–4 using a pre-built roster:
- Stage 3: Chapter chunking
- Stage 4: LLM speaker attribution
- Cache-aware: checks `nlp_cache/{book_hash}.json` before running
- Returns full `NLPResult` with annotated chapters (existing format, unchanged)

The existing `run_analysis()` is refactored to call these two functions sequentially and is kept for the headless path and any callers that need the full pipeline in one shot.

### Cache files

| File | Written by | Used by |
|------|-----------|--------|
| `nlp_cache/{hash}-roster.json` | Wizard (fast scan) | Worker (loads roster for attribution) |
| `nlp_cache/{hash}.json` | Worker (attribution) | Worker (TTS phase), future re-runs |

On re-runs, the worker finds `{hash}.json` already present and skips attribution entirely.

## Data Model Changes

### `CharacterInfo` — `src/kenkui/models.py`

Add `mention_count: int = 0` alongside the existing `quote_count`:

```python
@dataclass
class CharacterInfo:
    character_id: str
    display_name: str
    quote_count: int = 0       # Populated by worker after attribution
    mention_count: int = 0     # Populated by fast scan; used for voice assignment sort
    gender_pronoun: str = ""
```

The fast scan populates `mention_count`. Attribution (in the worker) populates `quote_count`. They coexist without conflict.

### `JobConfig` — `src/kenkui/models.py`

Add `roster_cache_path`:

```python
@dataclass
class JobConfig:
    # ... existing fields ...
    annotated_chapters_path: Path | None = None   # Unchanged; now set by worker, not wizard
    roster_cache_path: Path | None = None         # NEW: path to Stage 2 roster cache
```

The wizard sets `roster_cache_path` at submit time. `annotated_chapters_path` starts `None` for all new jobs and is set internally by the worker after attribution completes.

Storing the roster path in `JobConfig` ensures the worker runs attribution against the same canonical names the user saw during voice assignment — preventing any divergence between what was displayed and what gets used.

## Worker Changes — `src/kenkui/server/worker.py`

For multi-voice jobs, the worker gains a pre-TTS phase:

```
Job received (multi-voice)
  ↓
Check annotated_chapters_path
  ├─ Valid cache exists → skip attribution
  └─ Missing or stale:
       Load roster from roster_cache_path
       Run run_attribution(book_path, roster, chapter_selection, nlp_model)
       Cache full result → set annotated_chapters_path internally
  ↓
TTS generation (unchanged)
  ↓
Assembly (unchanged)
```

The live dashboard gains a new phase indicator: `Analyzing speakers…` shown during attribution. This is the only user-visible worker change.

## Voice Assignment Screen

`_prompt_character_voice_review()` in `cli/add.py`:

- Sort characters by `mention_count` (descending) instead of `quote_count`
- Display label changes from `X lines` to `X mentions`
- All other behavior (fuzzy search, gender-matched defaults, conflict resolution) unchanged

The auto-assignment and conflict resolution logic in `_auto_assign_character_voices()` and `_resolve_chapter_voice_conflicts()` remains unchanged — it operates on the character roster regardless of which count field is used for sorting.

## Backwards Compatibility

- **Headless single-voice**: No NLP involved. Unchanged.
- **Cached jobs / re-runs**: Worker finds `{hash}.json` on first run and skips attribution on subsequent runs. Identical behavior to today for repeated processing of the same book.
- **Jobs submitted before this change**: `annotated_chapters_path` may already be populated (old format). Worker treats a valid existing path as a cache hit and skips attribution. No migration needed.
- **Headless multi-voice**: Currently unimplemented interactively. The worker-side attribution phase means headless multi-voice becomes viable in a future iteration — the job just needs a `roster_cache_path` and the worker handles the rest.

## What We Lose

- **Quote count on the voice assignment screen**: Replaced by mention count. Mention count is a direct proxy for character prominence and serves the same organizational purpose (most-mentioned characters appear first).

## Files Changed

| File | Change |
|------|--------|
| `src/kenkui/nlp/__init__.py` | Add `run_fast_scan()`, `run_attribution()`; refactor `run_analysis()` to use them |
| `src/kenkui/nlp/models.py` | Add `FastScanResult` dataclass |
| `src/kenkui/models.py` | Add `mention_count` to `CharacterInfo`; add `roster_cache_path` to `JobConfig` |
| `src/kenkui/cli/add.py` | Reorder wizard steps; replace `_run_nlp_analysis()` with fast scan call; update voice review sort |
| `src/kenkui/server/worker.py` | Add pre-TTS attribution phase for multi-voice jobs |
