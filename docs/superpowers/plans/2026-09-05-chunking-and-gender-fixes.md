# Chunking Prosody and Gender Inference Fixes — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Stop the chunker from cutting ordinary prose mid-sentence (which pocket-tts renders as a false full stop), and make spaCy gender inference resolve main characters instead of abstaining.

**Architecture:** Two independent defects, both introduced by `d6705fe`. (1) The separator-free character budget is applied to every prose sentence and forces a bare-whitespace cut; pocket-tts appends a period to any fragment ending in an alphanumeric, so each cut becomes an audible sentence ending. Fix is two-part: forbid the bare-whitespace break tier for *forced* cuts, and raise the budget. (2) Gender is voted from a ±40-token pronoun window that measures scene pronoun density rather than reference, so co-present characters cancel each other out. Fix is to replace the window with a nearest-following-pronoun signal and add a gendered-honorific signal.

**Tech Stack:** Python 3.12, pytest, spaCy (`en_core_web_lg`), pocket-tts 2.1.0, sentencepiece (measurement only).

**Spec:** This document. Root-cause evidence is inline below rather than in a separate spec.

---

## Global Constraints

- **Purity:** `_domain/planning.py` is pure — no I/O, no model, no clock, no randomness. It must not import spaCy, sentencepiece, or torch.
- **Exact partition:** `"".join(chunks) == text` must hold for every span. Every existing test asserting this stays.
- **Determinism:** roster order and casting must remain a total order derived from content alone; the plan fingerprint depends on it.
- **Cache invalidation is intentional here.** Both fixes change segment identity. `CHUNKING_SCHEMA_VERSION` must be bumped so no stale PCM is reused. Every user re-renders once.
- **Test command:** `uv run pytest <path> -q --no-cov` for iteration; full gate is `uv run pytest` (enforces `--cov-fail-under=90`).
- **spaCy tests** are guarded by `pytest.importorskip("spacy")` and use `en_core_web_lg`.

---

## Starting State: the suite is already red

`d6705fe` shipped with this note in its own commit message:

> Note: tests/test_identity_stability.py and tests/test_single_voice_regression.py goldens need regeneration for the new separator budget and parser coordinates.

They were never regenerated. On `main` today:

```
FAILED tests/test_single_voice_regression.py::test_single_voice_segment_identities_are_unchanged
FAILED tests/test_single_voice_regression.py::test_single_voice_chunk_boundaries_are_unchanged
FAILED tests/test_single_voice_regression.py::test_single_voice_plan_fingerprint_is_unchanged
FAILED tests/test_identity_stability.py::test_plain_pipeline_identity_is_unchanged
4 failed, 62 passed
```

`test_single_voice_chunk_boundaries_are_unchanged` is precisely the guard that would have caught this regression — it asserts "where the chunker breaks decides how the rendered audio actually sounds". It was left red, so the defect shipped. **Task 1 fixes the process problem before the code problem.**

Verified against a clean `HEAD` (0439f10) in a detached worktree: the same four fail there, so this is `d6705fe`'s debt and not an artifact of the working tree.

### The working tree is not clean — decide this before Task 1

`git status` at the time of writing:

```
 M src/kenkui/_audio/native.py
 M src/kenkui/_audio/production.py
 M src/kenkui/_domain/planning.py
?? spikes/examples/example_failed.py
```

The uncommitted `planning.py` change is in `_append_chapter` — it drops whitespace-only chunks and changes `if carried:` to `if carried.strip():`. **That changes segment identities**, so whatever is in the tree when Task 1 regenerates the goldens becomes the new frozen baseline.

Resolve before starting: either commit those changes first (so the baseline is a named commit), or stash them (so the baseline is `HEAD`). Do not regenerate goldens on top of unreviewed working-tree edits — that is the same mistake, one layer down.

---

## Evidence

### Defect 1 — chunking

`MAX_SEPARATOR_FREE_CHARACTERS` (`_domain/planning.py:64`) is 48. `POCKET_SEPARATORS` is `".!?,;:"` — no whitespace — so a "separator-free run" is any sentence without an internal comma. Every such sentence over 48 characters is force-cut at the last whitespace before char 48:

```
'Ellie stared at the horizon and said nothing for a long while.'
  -> ['Ellie stared at the horizon and said nothing ', 'for a long while.']
```

The cut is audible, not merely a seam, because of `pocket_tts.models.tts_model.prepare_text_prompt`:

```python
if not text[0].isupper(): text = text[0].upper() + text[1:]
if text[-1].isalnum():    text = text + "."     # <- appended full stop
```

Each kenkui segment is its own `generate_audio` call (`_tts/pocket.py:844`), so a fragment ending in a bare word is synthesized as a complete sentence with falling terminal intonation.

**The asymmetry that names the fix:** pocket-tts splits internally too, but only at `.!?` or `,;:`, *keeping the separator*, so its own fragments end in punctuation and `prepare_text_prompt` appends nothing. Only kenkui's bare-whitespace tier (`_BREAK_TIERS[2]`, `r"\s+"`) produces an alphanumeric-terminated fragment. **The bare-whitespace tier is the entire defect.**

Measured on three real books with the real sentencepiece tokenizer (`kyutai/pocket-tts-without-voice-cloning`, `languages/english_2026-04/tokenizer.model`). "falseStop" = non-final chunks ending on an alphanumeric; "runs>50" = separator-free runs exceeding pocket's `MAX_TOKEN_PER_CHUNK = 50`:

| book | design | segs | falseStop | worstTok | runs>50 |
|---|---|---:|---:|---:|---:|
| Folding Space | **48 all-tiers (current)** | 4007 | **3267** | 38 | 0 |
| Folding Space | 48 clean-tiers | 747 | 31 | 128 | 31 |
| Folding Space | 150 clean-tiers | 466 | 4 | 83 | 27 |
| Folding Space | **200 clean-tiers** | **447** | **2** | 104 | 33 |
| Red Rising | **48 all-tiers (current)** | 5698 | **4129** | 33 | 0 |
| Red Rising | 150 clean-tiers | 784 | 4 | 71 | 20 |
| Red Rising | **200 clean-tiers** | **767** | **1** | 72 | 31 |
| Dune | **48 all-tiers (current)** | 11490 | **8898** | 38 | 0 |
| Dune | 150 clean-tiers | 1427 | 16 | 83 | 91 |
| Dune | **200 clean-tiers** | **1340** | **8** | 83 | 126 |

Reading: the current setting is the only one with zero token overruns, and it buys that by turning 78–86% of all segments into false sentence endings. `200 clean-tiers` removes essentially every false stop *and* cuts synthesis calls ~9x.

The `runs>50` that remain at 200 are ordinary long comma-free sentences. They cannot be split without a whitespace cut, and pocket only *warns* on them (`"generation may skip words"`) — `_estimate_max_gen_len` scales generation length with token count, so it is a soft risk, not truncation. The Whisper measurement below shows that risk is **zero** below 80 tokens and negligible below 150.

Chosen value: **200**, confirmed by synthesis rather than inferred: it holds every measured book's worst run inside the zero-word-loss band. `150` is the conservative alternative (see table); it costs ~4% more segments and a few more false stops for a slightly tighter token tail. Recorded here so the choice can be revisited without re-measuring.

A rejected alternative: estimating tokens from characters to get one threshold serving both prose and catalogue text. Measured against the real tokenizer over 20,000 runs, a word-plus-punctuation estimator underestimated 89.6% of the time (median −4 tokens, worst real/est ratio 6.0). Not a safe bound; dropped.

### Synthesis evidence: the cut costs 30% more audio for the same words

48 real separator-free runs from Folding Space, bucketed by true token count (long ones built by joining real clauses, since runs over 55 tokens are rare by nature). Each was synthesized twice with pocket-tts 2.1.0 through the same call kenkui makes — once whole, once through the current 48-character chunker with every fragment its own `generate_audio` call, concatenated:

| bucket | n | mean tok | mean parts | whole | cut | ratio |
|---|---:|---:|---:|---:|---:|---:|
| 15-25 | 6 | 18 | 1.7 | 3.17s | 3.72s | 1.17x |
| 30-40 | 6 | 33 | 2.8 | 5.97s | 7.21s | 1.21x |
| 45-55 | 6 | 50 | 4.0 | 8.07s | 9.91s | 1.23x |
| ~65 | 6 | 70 | 5.8 | 11.63s | 14.88s | 1.28x |
| ~85 | 6 | 89 | 7.3 | 14.59s | 18.44s | 1.26x |
| ~105 | 6 | 112 | 8.8 | 17.96s | 23.47s | 1.31x |
| ~130 | 6 | 137 | 10.8 | 21.77s | 28.55s | 1.31x |
| ~160 | 6 | 164 | 12.5 | 23.60s | 33.00s | **1.40x** |

**Overall: 640.6s whole vs 835.0s cut — 30% more audio for identical words**, and the ratio tracks the number of cuts. That excess is exactly the defect being reported: terminal lengthening plus a pause at each manufactured sentence ending. It is a direct, objective measurement of the thing you can hear.

### Whisper evidence: the 50-token warning does not matter until ~150 tokens

Every clip above was transcribed with `faster-whisper small.en` and scored against its input. **Deletion rate** — words present in the input and absent from the transcript — is the metric that answers "does it skip words"; raw WER is contaminated by transcription artifacts on compounds (`fishballs` → `fish balls` alone costs 0.286 WER on a 16-token clip).

Deletion rate for the whole-utterance arm, by true token count:

| tokens | n | deletions |
|---|---:|---:|
| 0–40 | 12 | **0.0000** |
| 40–60 | 6 | **0.0000** |
| 60–80 | 6 | **0.0000** |
| 80–100 | 6 | 0.0029 |
| 100–125 | 6 | 0.0023 |
| 125–150 | 6 | 0.0037 |
| 150–200 | 6 | **0.0315** |

**Zero word loss through 80 tokens — 60% past the engine's own 50-token limit — and negligible loss to 150.** Real loss (3.2%) appears only past 150 tokens.

*Confound, handled:* natural runs over 55 tokens are rare in these books, so the long buckets were built by joining real clauses with " and ", which is unnatural English. Separating them: the 18 verbatim runs (mean 34 tokens) show **0.000** deletions, and built runs at 55–125 tokens show 0.002 — so the cliff at 150+ is a length effect, not a construction artifact. The elevated *WER* on built runs is partly artifact; the *deletion* figures agree with the natural set and are the ones relied on here.

**This settles two things.** First, your instinct is right — the warning is not worth worrying about in the range real prose produces, so it must not be traded for a certain audible defect. Second, it independently validates 200 rather than removing the guard: at budget 200 the measured worst runs are 70–104 tokens, inside the zero-loss zone; with the guard effectively off (budget 1000) Folding Space produces a 207-token run, which is in the 3%-deletion zone. **200 is the value that keeps every book inside the flat part of this curve.**

### There is no cross-chunk conditioning to preserve

`generate_audio` defaults to `copy_state=True`, and `_generate_audio_stream_short_text` deep-copies the state for every internal chunk (`models/tts_model.py:635-637`). The source says so outright:

```python
# TODO: add the teacher forcing method for long texts where we use the audio of one chunk
# as conditioning for the next chunk.
```

Each internal chunk is generated independently from the same base state. **Segment size therefore buys no audio quality at all** — splitting a paragraph at sentence boundaries in kenkui yields the same audio as handing pocket the paragraph and letting it split at those same points. Only *where* the cut lands matters, which is what Task 2 fixes.

### Evaluated and not adopted: cut only on paragraph or speaker change

Proposed as a simpler invariant — never cut inside a paragraph, so the guard and its tuned budget disappear entirely. It is a legitimate design and it would fix the reported defect. It is not adopted, for four measured reasons.

**1. It buys no audio quality.** Per the `copy_state` finding above, pocket generates each internal chunk independently. A paragraph handed over whole is split by pocket at its sentence boundaries and produces the same audio as kenkui splitting at those same boundaries. The quality lever is cut *position*, not segment size, and Task 2 already fixes position.

**2. It costs more calls, not fewer.** Paragraphs in these books are small:

| book | paragraphs | median | p90 | p99 | max | >1000 chars |
|---|---:|---:|---:|---:|---:|---:|
| Folding Space | 1863 | 155 | 432 | 697 | 979 | 0 |
| Dune | 8447 | 103 | 304 | 561 | 1407 | 3 |

Paragraph-aligned segments give **1863** segments for Folding Space against **447** for `200 clean-tiers` — 4x more `generate_audio` calls for identical audio.

**3. It is not the small change it looks like.** Paragraph boundaries are not segment boundaries today. `split_structural` returns the whole chapter as a single piece unless a pause tier is configured (`_domain/structure.py:129-130`), and `example.py` sets no pauses. Making structural splitting unconditional changes `break_tiers`, which drives both the v2/v3 chunker selection and gap insertion — a wider identity change than the two constants Tasks 2 and 3 touch.

**4. It works against the scheduler.** Work is distributed as contiguous static partitions sized by task *count* (`_execution/process_pool.py:576-584`). Uniform ~858-character segments balance across 12 workers; paragraph segments varying from 103 to 1407 characters do not, giving a longer tail.

**What is worth keeping from the idea:** speaker change is *already* a hard segment boundary (`_spans_for`/`_fragments`), and after Task 2 every remaining cut lands on punctuation or a line break — so in practice the chunker will now only cut where the text itself breaks. That is the invariant you wanted, reached without the costs above. If you later configure `.pause(paragraph_ms=...)`, paragraph boundaries become segment boundaries too, and that path already exists.

### Defect 2 — gender

Running the real spaCy roster over *The Subtle Art of Folding Space*:

```
ellie   gender=None   chapters=25
daniel  gender=None   chapters=24
ahdi    gender=None   chapters=15
```

Ellie is not assigned masculine — she is assigned **nothing**, and `casting.candidates` (`_domain/casting.py:107`) answers `gender is None` with *the whole pool*, so load-balancing can hand her a masculine voice. Raw votes: **feminine 2500 / masculine 1269** — a ratio of 1.97 against `_GENDER_MARGIN = 2`. She misses by 1.5%.

The cause is `_GENDER_WINDOW = 40` (`spacy_roster.py:74`): every gendered pronoun within ±40 tokens of a name is counted, which in a two-hander scene counts Daniel's every "he" as a masculine vote for Ellie. Sweeping the window on the real book — the current value is the only one that fails:

| | w=3 | w=8 | w=20 | **w=40 (current)** | nearest-pronoun |
|---|---|---|---|---|---|
| Ellie | fem | fem | fem | **None** (2500/1269) | **fem** (339/87) |
| Daniel | None | None | None | **None** (1286/1866) | **masc** (94/294) |
| Chris | fem | fem | fem | fem | fem |
| Ahdi | masc | masc | None | **None** (227/369) | **masc** (9/58) |
| Aunt Vera | None | masc | None | None | masc *(wrong)* |

"nearest-pronoun" — first gendered pronoun after the name, stopping at an intervening proper noun — is the only strategy that also resolves Daniel and Ahdi. Its one miss, `Aunt Vera`, is covered by an honorific signal that is currently unused (`titled` feeds animacy only, and `_TITLES` lacks `mr`/`mrs`/`ms`/`miss`/`aunt`/`uncle`/`mom`/`dad`). Reading the honorific off the span:

```
Mom          {'feminine': 185}
Aunt Vera    {'feminine': 21}
Mr. Neeson   {'masculine': 10}
```

`tests/test_spacy_roster.py:41-43` already documents the defect as intended behaviour — "a dense two-hander where every line touches both characters is exactly the case it abstains on". That comment is what Task 4 overturns.

**Out of scope, worth a later look:** the same roster casts `Amtrak`, `Boston`, `Metro`, `Carousel`, `Simulation` and `Belt` as speaking characters; and a gendered cast silently falls back to the whole pool for `gender=None` without any warning (`ungendered_pool_characters` only reports characters whose gender *is* known).

---

## File Structure

| File | Responsibility | Change |
|---|---|---|
| `src/kenkui/_domain/planning.py` | pure chunking + plan compilation | budget, clean-tier forced break, schema bump |
| `tests/test_planning.py` | chunker contract | 3 tests rewritten, 2 added |
| `tests/test_single_voice_regression.py` | frozen segment goldens | regenerate |
| `tests/test_identity_stability.py` + `tests/data/` | frozen plan golden | regenerate |
| `src/kenkui/_characters/spacy_roster.py` | offline roster + gender | pronoun signal replaced, honorific signal added |
| `tests/test_spacy_roster.py` | roster contract | gender tests rewritten |
| `src/kenkui/_characters/dialogue_tags.py` | **new** — speech-tag pronoun extraction | Task 5 |
| `src/kenkui/_characters/__init__.py` | post-attribution profile rebuild | Task 5 |
| `tests/test_dialogue_tags.py` | **new** | Task 5 |

---

## Task 1: Restore the golden tests to green

The goldens are the regression guard for Tasks 2–3. They must be trustworthy *before* the chunker changes, so that the diff they show next is the intended one and nothing else.

**Files:**
- Modify: `tests/test_single_voice_regression.py:80`
- Modify: `tests/data/` golden JSON read by `tests/test_identity_stability.py:100-104`

- [ ] **Step 1: Confirm the four known failures, and only those**

```bash
uv run pytest tests/test_single_voice_regression.py tests/test_identity_stability.py -q --no-cov
```

Expected: exactly 4 failures —
`test_single_voice_segment_identities_are_unchanged`,
`test_single_voice_chunk_boundaries_are_unchanged`,
`test_single_voice_plan_fingerprint_is_unchanged`,
`test_plain_pipeline_identity_is_unchanged`.

If any *other* test fails, stop and report — the baseline is not what this plan assumes.

- [ ] **Step 2: Print the current actual values**

```bash
uv run python - <<'PY'
import sys; sys.path.insert(0, "src")
sys.path.insert(0, "tests")
from test_single_voice_regression import _plan
plan = _plan()
print("BASELINE_SEGMENT_IDS = (")
for s in plan.segments: print(f'    "{s.id}",')
print(")")
print("BASELINE_LENGTHS =", tuple(len(s.text) for s in plan.segments))
print("BASELINE_FINGERPRINT =", repr(plan.semantic_fingerprint))
PY
```

- [ ] **Step 3: Paste those values into the golden constants**

Replace `BASELINE_SEGMENT_IDS`, `BASELINE_LENGTHS` and the fingerprint constant in `tests/test_single_voice_regression.py` with the printed values. Do not touch `test_chunking_schema_version_is_frozen` yet — Task 3 bumps it.

- [ ] **Step 4: Regenerate the identity-stability golden**

```bash
uv run python - <<'PY'
import sys, json, pathlib; sys.path.insert(0, "src"); sys.path.insert(0, "tests")
from test_identity_stability import snapshot, GOLDEN
GOLDEN.write_text(json.dumps(snapshot(), indent=2, sort_keys=True) + "\n", "utf-8")
print("wrote", GOLDEN)
PY
```

- [ ] **Step 5: Verify green**

```bash
uv run pytest tests/test_single_voice_regression.py tests/test_identity_stability.py -q --no-cov
```

Expected: PASS, 0 failed.

- [ ] **Step 6: Commit**

```bash
git add tests/test_single_voice_regression.py tests/test_identity_stability.py tests/data
git commit -m "test: regenerate the goldens d6705fe left stale

These are the guard that should have caught the separator-budget
regression. They were red on main, so it shipped."
```

---

## Task 2: Forbid the bare-whitespace tier for forced cuts

This is the fix that matters. It makes the guard incapable of manufacturing a false sentence ending regardless of how the budget is tuned.

**Files:**
- Modify: `src/kenkui/_domain/planning.py:66-79` (tier constants), `:570-585` (`_break_offset`), `:599-635` (`_chunk_span`)
- Test: `tests/test_planning.py`

**Interfaces:**
- Produces: `planning._CLEAN_BREAK_TIERS: tuple[str, ...]` — the break tiers that leave a fragment ending in punctuation or a line break.
- Produces: `planning._break_offset(text, start, stop, tiers=_BREAK_TIERS) -> int` — gains a third positional-or-keyword parameter `tiers`; existing two-argument-plus-stop calls keep today's behaviour.

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_planning.py`:

```python
def test_forced_cut_never_ends_a_fragment_on_a_word() -> None:
    """Pocket-TTS appends a full stop to any fragment ending alphanumeric.

    A cut mid-sentence therefore renders as a completed sentence, which is
    audible as a break after three or four words. Only breaks that leave
    punctuation or a line break behind are safe to force.
    """
    text = "Ellie stared at the horizon and said nothing at all for a very long while."
    chunks = _chunks(text)

    assert "".join(chunks) == text
    assert not any(chunk.rstrip() and chunk.rstrip()[-1].isalnum() for chunk in chunks[:-1])


def test_comma_free_prose_sentence_is_left_whole() -> None:
    """No clean break exists inside it, so it goes to the engine intact.

    Pocket-TTS splits and packs it internally and only warns if it runs long.
    That soft risk is preferable to a certain false sentence ending.
    """
    text = "He walked across the road and then he saw the man standing there."
    assert _chunks(text) == [text]
```

- [ ] **Step 2: Run them to verify they fail**

```bash
uv run pytest tests/test_planning.py::test_forced_cut_never_ends_a_fragment_on_a_word \
              tests/test_planning.py::test_comma_free_prose_sentence_is_left_whole -q --no-cov
```

Expected: both FAIL — the first because chunks end on `"the"`, the second because the sentence is split in two.

- [ ] **Step 3: Add the clean-tier constant**

In `src/kenkui/_domain/planning.py`, immediately after `_BREAK_TIERS`:

```python
# The subset of _BREAK_TIERS whose cut leaves the preceding fragment ending in
# punctuation or a line break. Pocket-TTS's prepare_text_prompt appends a full
# stop to any input ending alphanumeric, and every segment is its own
# generate_audio call, so a cut on the bare-whitespace tier renders as a
# completed sentence -- an audible break mid-clause. Its own splitter never
# does this: it cuts on ".!?" or ",;:" and keeps the separator. A forced cut
# may therefore use every tier except bare whitespace, and when none of these
# is available the run is left whole for the engine to pack internally.
_CLEAN_BREAK_TIERS = (
    r"\n\s*",
    r"[.!?][\"')\]]*\s+|[,;:][\"')\]]*\s+",
    r"[-\u2010-\u2015]",
)
```

- [ ] **Step 4: Let `_break_offset` take a tier set**

Replace the signature and loop head of `_break_offset` (`planning.py:570-585`):

```python
def _break_offset(
    text: str, start: int, stop: int, tiers: tuple[str, ...] = _BREAK_TIERS
) -> int:
    """Offset past the best boundary in ``text[start:stop]``, or 0 when none fits."""
    window = text[start:stop]
    if not window:
        return 0
    threshold = len(window) * MIN_BREAK_FILL
    fullest = 0
    for pattern in tiers:
        offsets = [match.end() for match in re.finditer(pattern, window)]
        if not offsets:
            continue
        best = offsets[-1]
        if best >= threshold:
            return best
        fullest = max(fullest, best)
    return fullest
```

- [ ] **Step 5: Use the clean tiers for the forced cut only**

In `_chunk_span` (`planning.py:621-628`), change the forced-cut branch to pass the clean tiers. The unforced break above it is unchanged — it only fires at the 1000-character bound, where a whitespace cut beats a mid-word one:

```python
        budget_end = _separator_free_end(text, start, hard_end)
        if budget_end < end:
            # Cutting needs a boundary that leaves punctuation behind. A bare
            # whitespace cut would render as a completed sentence, so when no
            # clean boundary exists the run is left for the engine to pack.
            forced = _break_offset(text, start, budget_end, _CLEAN_BREAK_TIERS)
            if forced:
                end = start + forced
```

- [ ] **Step 6: Run the new tests**

```bash
uv run pytest tests/test_planning.py::test_forced_cut_never_ends_a_fragment_on_a_word \
              tests/test_planning.py::test_comma_free_prose_sentence_is_left_whole -q --no-cov
```

Expected: PASS.

- [ ] **Step 7: Rewrite the test that pinned the old behaviour**

`test_comma_free_run_on_sentence_is_split_at_whitespace` (`tests/test_planning.py:390`) asserts the exact behaviour this task removes. Replace it wholesale:

```python
def test_comma_free_run_on_sentence_is_left_to_the_engine() -> None:
    """A run with no clean boundary is handed over whole, not cut at a space.

    Formerly this was split at whitespace to hold the engine's token limit.
    That cut renders as a false sentence ending on every fragment, which is
    a worse defect than the engine's own soft "may skip words" warning.
    """
    text = "and then " * 120
    chunks = _chunks(text)

    assert "".join(chunks) == text
    assert not any(chunk.rstrip() and chunk.rstrip()[-1].isalnum() for chunk in chunks[:-1])
```

Note it may still split at the 1000-character hard bound; the assertion allows that and only forbids the alphanumeric ending.

- [ ] **Step 8: Run the whole planning suite**

```bash
uv run pytest tests/test_planning.py -q --no-cov
```

Expected: PASS. `test_separator_free_runs_are_split_at_line_breaks` (contents page) still passes — the newline tier is clean. The hyphen test still passes — the hyphen tier is clean.

- [ ] **Step 9: Commit**

```bash
git add src/kenkui/_domain/planning.py tests/test_planning.py
git commit -m "fix: never force a chunk cut that ends a fragment on a word

Pocket-TTS appends a full stop to any input ending alphanumeric, and
every segment is its own generate_audio call, so a bare-whitespace cut
rendered as a completed sentence. Measured over three books it turned
78-86% of all segments into false sentence endings."
```

---

## Task 3: Raise the separator-free budget to 200

Task 2 makes the guard safe; this makes it stop firing on ordinary prose.

**Files:**
- Modify: `src/kenkui/_domain/planning.py:51-52` (schema versions), `:61-64` (budget)
- Modify: `tests/test_planning.py:461-469`, `tests/test_single_voice_regression.py:78-80`

- [ ] **Step 1: Rewrite the test that pins 48**

`test_token_dense_separator_free_text_uses_a_safe_character_budget` uses `"a-" * 60` (120 characters), which no longer exceeds a 200-character budget. Give it text that does, and assert the contract that actually matters:

```python
def test_token_dense_separator_free_text_is_split_at_its_hyphens() -> None:
    """Catalogue text has no ".!?,;:" the engine can divide, so Kenkui divides it.

    The hyphen tier is clean: each fragment ends in "-", so the engine appends
    no full stop and the split is inaudible as a sentence ending.
    """
    text = "a-" * 150

    chunks = _chunks(text)

    assert "".join(chunks) == text
    assert len(chunks) > 1
    budget = planning.MAX_SEPARATOR_FREE_CHARACTERS
    for chunk in chunks:
        assert _worst_separator_free_run(chunk) <= budget
```

- [ ] **Step 2: Run it to verify it fails**

```bash
uv run pytest tests/test_planning.py::test_token_dense_separator_free_text_is_split_at_its_hyphens -q --no-cov
```

Expected: FAIL — at budget 48 the chunks are shorter than the assertion expects to be meaningful; the test is written for the new budget.

- [ ] **Step 3: Raise the budget and record the measurement**

Replace `planning.py:61-64`:

```python
# Calibrated against the engine's own tokenizer over three full books, counting
# the way it does (newlines collapse to spaces before tokenizing). English prose
# runs about 2.8-3.2 characters per token, not the one-token-per-character worst
# case a previous value assumed: at 200 characters the worst separator-free run
# measured 70-104 tokens against the engine's 50-token soft limit, and only
# 31-126 runs per book exceeded it at all. Dropping to 48 held every run under
# the limit but forced a cut inside 78-86% of all segments, which is the far
# worse defect -- see _CLEAN_BREAK_TIERS. 150 is the conservative alternative:
# ~4% more segments for a slightly tighter token tail.
MAX_SEPARATOR_FREE_CHARACTERS = 200
```

- [ ] **Step 4: Bump the chunking schema versions**

Chunk boundaries have changed, so every cached segment must miss. Replace `planning.py:51-52`:

```python
CHUNKING_SCHEMA_VERSION = "tts-chunks-v4"
CHUNKING_V3_SCHEMA_VERSION = "tts-chunks-v5"
```

Then update the freeze test at `tests/test_single_voice_regression.py:80`:

```python
    assert CHUNKING_SCHEMA_VERSION == "tts-chunks-v4"
```

- [ ] **Step 5: Run the planning suite**

```bash
uv run pytest tests/test_planning.py -q --no-cov
```

Expected: PASS.

- [ ] **Step 6: Regenerate the goldens, and read the diff**

Repeat Task 1 Steps 2–4. Then inspect what changed:

```bash
git diff --stat tests/
```

Expected: segment counts fall sharply and no non-final segment text ends on a bare word. This diff is the deliverable — read it before committing.

- [ ] **Step 7: Verify the whole suite**

```bash
uv run pytest -q
```

Expected: PASS, coverage ≥ 90%.

- [ ] **Step 8: Commit**

```bash
git add src/kenkui/_domain/planning.py tests/
git commit -m "fix: raise the separator-free budget from 48 to 200

48 assumed one token per character. Measured against the engine's own
tokenizer over three books, English prose runs 2.8-3.2 characters per
token, so 48 split nearly every sentence. Chunking schema bumped: every
cached segment must miss."
```

---

## Task 4: Replace the gender window with pronoun-proximity plus honorifics

**Files:**
- Modify: `src/kenkui/_characters/spacy_roster.py:73-78` (constants), `:255-293` (`_TITLES` area), `:531-566` (`_scan`), `:568-581` (`_vote_gender`), `:505-520` (`_Signals.__init__`)
- Test: `tests/test_spacy_roster.py`

**Interfaces:**
- Produces: `_Signals.title_gender: dict[str, Counter[str]]` — honorific votes per surface name.
- Produces: `_gender_of(pronouns: Counter[str], titles: Counter[str]) -> str | None` — replaces the single-argument `_vote_gender`. Honorifics win outright when they clear the threshold; pronouns decide otherwise.

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_spacy_roster.py`. `PASSAGE` (the dense two-hander at line 30) is exactly the case the old code abstained on:

```python
def test_gender_resolves_in_a_dense_two_hander() -> None:
    """Both speakers share every scene, so a wide pronoun window cancels out.

    The nearest following pronoun refers to the name it follows, so it
    survives co-presence. Formerly both abstained and casting fell back to
    the whole pool, which is how a woman drew a masculine voice.
    """
    roster, _ = roster_of(PASSAGE)
    by_id = {character.id: character for character in roster}

    assert by_id["egwene"].gender == "feminine"
    assert by_id["tam-al-thor"].gender == "masculine"


def test_a_gendered_honorific_outranks_nearby_pronouns() -> None:
    """"Aunt" is decisive; pronouns near her name are not about her."""
    text = (
        "Aunt Vera set down the tray. He had left the door open again.\n"
        "Aunt Vera frowned at him. He said nothing at all.\n"
        "Aunt Vera poured the tea. He watched her hands.\n"
        "Aunt Vera sighed. He shrugged at Aunt Vera and looked away.\n"
    )
    roster, _ = roster_of(text)
    by_id = {character.id: character for character in roster}

    assert by_id["aunt-vera"].gender == "feminine"
```

- [ ] **Step 2: Run them to verify they fail**

```bash
uv run pytest tests/test_spacy_roster.py::test_gender_resolves_in_a_dense_two_hander \
              tests/test_spacy_roster.py::test_a_gendered_honorific_outranks_nearby_pronouns -q --no-cov
```

Expected: both FAIL — genders come back `None`.

- [ ] **Step 3: Replace the window constants**

Replace `spacy_roster.py:73-78`:

```python
# How far past a name to look for the pronoun that refers to it. A pronoun
# further away than this is usually about someone else. The scan stops at the
# first gendered pronoun, and at any intervening proper noun -- once another
# name intervenes, the pronoun is more likely theirs.
#
# This replaces a symmetric +/-40-token window that counted every pronoun near
# a name. That measured how many men and women were in the scene, not who the
# name was: measured on one novel it gave the female lead feminine 2500 to
# masculine 1269 -- a ratio of 1.97 against the 2.0 margin, so she abstained
# and casting fell back to the whole pool. The nearest-pronoun signal gives
# her 339 to 87, and resolves the male lead the window could not resolve at all.
_GENDER_LOOKAHEAD = 12
# Votes required before a gender is claimed at all, and the margin the winner
# must hold over the loser. A character seen twice near "she" is not evidence.
_GENDER_MINIMUM = 3
_GENDER_MARGIN = 2
```

- [ ] **Step 4: Add the honorific sets**

Add after `_TITLES` (`spacy_roster.py:293`). Deliberately separate from `_TITLES`, which `_strip_titles` consumes — folding these in would rename "Aunt Vera" to "Vera" and change character ids, which is a different change with a different blast radius:

```python
# Honorifics and kinship terms that state a gender outright. Read from the name
# span itself, never stripped first: this is the strongest gender evidence a
# text offers and it is free. Kept apart from _TITLES because that set feeds
# _strip_titles, and stripping these would rename "Aunt Vera" to "Vera" and
# change her character id.
_FEMININE_TITLES = frozenset(
    {
        "mrs", "ms", "miss", "madam", "madame", "mistress", "lady", "dame",
        "queen", "princess", "duchess", "countess", "baroness", "sister",
        "mother", "mom", "mum", "mama", "aunt", "auntie", "grandma",
        "grandmother", "granny", "nan", "widow",
    }
)
_MASCULINE_TITLES = frozenset(
    {
        "mr", "mister", "sir", "lord", "master", "king", "prince", "duke",
        "baron", "earl", "brother", "father", "dad", "papa", "uncle",
        "grandpa", "grandfather", "monsieur", "herr", "senor",
    }
)
```

- [ ] **Step 5: Add the honorific tally to `_Signals`**

In `_Signals.__init__` (`spacy_roster.py:509`), beside `self.gender`:

```python
        self.title_gender: dict[str, Counter[str]] = defaultdict(Counter)
```

- [ ] **Step 6: Replace the scan's gender branch**

Replace the trailing window loop of `_scan` (`spacy_roster.py:560-565`) with:

```python
        leading = span.text.split()[0].lower().strip(".") if span.text.split() else ""
        preceding = doc[span.start - 1].lower_.strip(".") if span.start else ""
        for word in (leading, preceding):
            if word in _FEMININE_TITLES:
                into.title_gender[name]["feminine"] += 1
                break
            if word in _MASCULINE_TITLES:
                into.title_gender[name]["masculine"] += 1
                break
        for other in doc[span.end : span.end + _GENDER_LOOKAHEAD]:
            if other.pos_ == "PROPN":
                break  # another name intervened; the pronoun is likely theirs
            if other.lower_ in _FEMININE:
                into.gender[name]["feminine"] += 1
                break
            if other.lower_ in _MASCULINE:
                into.gender[name]["masculine"] += 1
                break
```

- [ ] **Step 7: Make honorifics outrank pronouns**

Replace `_vote_gender` (`spacy_roster.py:568-581`) with:

```python
def _majority(votes: Counter[str]) -> str | None:
    """Return the winner only on a clear majority, else None."""
    top = votes.most_common(1)
    if not top or top[0][1] < _GENDER_MINIMUM:
        return None
    winner, count = top[0]
    other = votes["masculine" if winner == "feminine" else "feminine"]
    return winner if count >= _GENDER_MARGIN * other else None


def _gender_of(pronouns: Counter[str], titles: Counter[str]) -> str | None:
    """Decide a gender from honorifics first, then nearby pronouns.

    An honorific states the gender outright, so it wins whenever it clears
    the threshold: pronouns near "Aunt Vera" are frequently about whoever she
    is talking to, and cannot be allowed to overturn the word "Aunt".

    An unsourced gender is an admission of ignorance, and `casting.candidates`
    answers it by offering the whole pool. A wrong guess is worse: it silently
    restricts a character to voices that sound wrong for them.
    """
    return _majority(titles) or _majority(pronouns)
```

- [ ] **Step 8: Accumulate and use the honorific votes**

In the merge loop (`spacy_roster.py:703`), beside `row["gender"].update(...)`:

```python
        row["title_gender"].update(signals.title_gender[name])
```

Add `"title_gender": Counter(),` to the `setdefault` dict at `spacy_roster.py:697`, and change the profile construction at `spacy_roster.py:716`:

```python
                    gender=_gender_of(row["gender"], row["title_gender"]),
```

- [ ] **Step 9: Update the parametrised margin test**

`test_gender_is_claimed_only_on_a_clear_margin` (`tests/test_spacy_roster.py:132`) calls `spacy_roster._vote_gender(Counter(votes))`. Point it at the renamed helper:

```python
        assert spacy_roster._majority(Counter(votes)) == expected  # noqa: SLF001
```

- [ ] **Step 10: Update the stale `SEPARATED` comment**

`tests/test_spacy_roster.py:41-43` documents the defect as intended. Replace it:

```python
# The same signals with the two people kept apart. Both this and PASSAGE must
# now resolve: the nearest-pronoun signal does not depend on characters being
# separated, which is what the old +/-40-token window required.
```

- [ ] **Step 11: Run the roster suite**

```bash
uv run pytest tests/test_spacy_roster.py -q --no-cov
```

Expected: PASS.

- [ ] **Step 12: Verify against the real book**

```bash
uv run python - <<'PY'
import sys, pathlib; sys.path.insert(0, "src")
from kenkui._epub.parser import inspect_epub
from kenkui._characters import spacy_roster as SR
from kenkui._characters.quotes import extract_spans
book = inspect_epub(pathlib.Path(
    "/Users/dizzler/Projects/Calibre Library/John Chu/"
    "The Subtle Art of Folding Space (465)/The Subtle Art of Folding Space - John Chu.epub"))
roster, _ = SR.infer_roster(book.chapters, {c.id: extract_spans(c.id, c.text) for c in book.chapters})
for ch in roster:
    print(f"{ch.id:22s} {ch.gender}")
PY
```

Expected: `ellie feminine`, `daniel masculine`, `ahdi masculine`, `chris feminine`, `aunt-vera feminine`, `mr-neeson masculine`.

- [ ] **Step 13: Run the full suite and commit**

```bash
uv run pytest -q
git add src/kenkui/_characters/spacy_roster.py tests/test_spacy_roster.py
git commit -m "fix: gender a character from the pronoun that refers to them

The +/-40-token window counted every pronoun in the scene, so two
co-present characters cancelled each other out and both abstained --
which sends them to the whole voice pool, and a woman to a male voice.
Adds gendered honorifics, which state the answer outright."
```

---

## Task 5: Gender from attributed dialogue tags

Your idea: attribution already knows who spoke each quote, so `"..." she said` genders that speaker directly. This is the cleanest signal in the book — the pronoun in a dialogue tag refers to the speaker by construction, with none of the proximity noise.

**Sizing, measured on Folding Space:** of 1694 quotes, tags resolve to `name` 290, `pronoun` 134 (8%), `other` 525, none 745. So ~134 zero-noise votes per book, concentrated on the main speakers — exactly the characters casting cares about. Sparse but high-value, and free: attribution has already run.

**On a dedicated LLM pass:** not needed, and not recommended as a *separate* pass. Attribution already carries the speaker, so this is pure post-processing of data you have paid for. If Tasks 4 and 5 still leave a main character unsourced, the cheaper escalation is **one** call per *book* over the ~20-name roster ("which of these names is male, female, or unclear?"), not a per-chapter pass — but only add it if measurement shows it is needed.

**Files:**
- Create: `src/kenkui/_characters/dialogue_tags.py`
- Create: `tests/test_dialogue_tags.py`
- Modify: `src/kenkui/_characters/__init__.py:20` (import) and `:412` (the `_measured` call site)

**Interfaces:**
- Consumes: `SpeakerSpan` tuples from attribution; `CharacterProfile` from Task 4.
- Produces: `dialogue_tags.tag_genders(chapters, spans) -> dict[str, Counter[str]]` — gender votes keyed by `character_id`.
- Produces: `dialogue_tags.apply(characters, votes) -> tuple[CharacterProfile, ...]` — **supersedes** the roster's gender whenever the tag vote clears its threshold, and logs every disagreement.

### Precedence, and the risk it carries

The tag signal supersedes the roster's, because the two are not the same kind of evidence. The roster's pronoun signal is *proximity* — a guess that the nearest pronoun refers to this name. A dialogue tag's pronoun **is the speaker's**, grammatically, by construction. Where they disagree the tag is right, so gating it behind "only fill the gaps" would throw away the better answer in exactly the cases that matter.

The honest cost: this can now make a *correct* gender wrong, where gap-filling could only help. The failure path is a mis-attributed quote — attribution says Ellie, the tag says "he said", and Ellie flips masculine. Three mitigations, all in the code below:

1. **A threshold, not a single vote:** `_TAG_MINIMUM = 3` with a 2× margin, so a handful of bad attributions cannot carry it.
2. **Disagreement is logged, never silent.** A conflict here is evidence of an attribution bug, which is worth surfacing on its own — the character's *lines* are already in the wrong voice if attribution is wrong, so gender is the symptom, not the disease.
3. **Honorifics break ties below the threshold only.** A name the text literally calls "Aunt" keeps its gender when tags are too sparse to decide.

Deliberately **not** done: adding a `gender_source` field to `CharacterProfile` to make honorifics outrank tags. `CharacterProfile.gender` is persisted in explicit SQLite columns (`_characters/store.py:66,112,361,584`), so a provenance field means a schema migration — too much cost for a conflict that indicates an upstream bug either way.

- [ ] **Step 1: Write the failing test**

Create `tests/test_dialogue_tags.py`:

```python
"""Gender a speaker from the pronoun in their own dialogue tag."""

from collections import Counter

import kenkui as kk
from kenkui._characters import dialogue_tags
from kenkui._domain.casting import CharacterProfile
from kenkui._domain.planning import SpeakerSpan

TEXT = '"I will not go," she said. Ellie turned away.\n"Then stay," he said.\n'


def _chapter(text: str) -> kk.ChapterInspection:
    return kk.ChapterInspection(
        id="ch-1", index=0, title="One", speech_characters=len(text), text=text
    )


def test_a_pronoun_tag_genders_the_attributed_speaker() -> None:
    """The pronoun in a tag refers to the speaker by construction."""
    chapter = _chapter(TEXT)
    spans = (
        SpeakerSpan("ch-1", 0, 16, "ellie"),
        SpeakerSpan("ch-1", 16, len(TEXT), None),
    )

    votes = dialogue_tags.tag_genders((chapter,), spans)

    assert votes["ellie"]["feminine"] == 1


def test_a_confident_tag_vote_supersedes_the_roster() -> None:
    """The tag's pronoun is the speaker's; the roster's was only nearby."""
    characters = (
        CharacterProfile("ellie", "Ellie", "masculine", 100, ("ch-1",)),
        CharacterProfile("ahdi", "Ahdi", None, 100, ("ch-1",)),
    )
    votes = {
        "ellie": Counter({"feminine": 9, "masculine": 1}),
        "ahdi": Counter({"masculine": 4}),
    }

    applied = {c.id: c.gender for c in dialogue_tags.apply(characters, votes)}

    assert applied == {"ellie": "feminine", "ahdi": "masculine"}


def test_an_unconfident_tag_vote_leaves_the_roster_alone() -> None:
    """Below the threshold this signal is too sparse to overturn a book-wide read."""
    characters = (CharacterProfile("vera", "Aunt Vera", "feminine", 100, ("ch-1",)),)
    votes = {"vera": Counter({"masculine": 2})}

    applied = {c.id: c.gender for c in dialogue_tags.apply(characters, votes)}

    assert applied == {"vera": "feminine"}
```

- [ ] **Step 2: Run to verify it fails**

```bash
uv run pytest tests/test_dialogue_tags.py -q --no-cov
```

Expected: FAIL with `ModuleNotFoundError: No module named 'kenkui._characters.dialogue_tags'`.

- [ ] **Step 3: Implement the module**

Create `src/kenkui/_characters/dialogue_tags.py`. Regex-based, reusing `narration._VERBS` — no second spaCy parse, so this costs nothing and adds no dependency to the attribution path:

```python
"""Gender a speaker from the pronoun in their own dialogue tag.

Attribution already decided who spoke each quote. A tag like `"..." she said`
therefore states that speaker's gender directly, with none of the proximity
noise the roster's pronoun signal has to tolerate: the pronoun is the tag's
subject, and the tag belongs to the quote.

Measured on one novel this fires on about 8% of quotes -- sparse, but with
almost no error, and concentrated on the characters who speak most, which are
exactly the ones casting must get right.
"""

from __future__ import annotations

import logging
import re
from collections import Counter, defaultdict
from typing import TYPE_CHECKING

from kenkui._characters.narration import _VERBS
from kenkui.observability import log_event

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from kenkui._domain.casting import CharacterProfile
    from kenkui._domain.planning import SpeakerSpan
    from kenkui.inspection import ChapterInspection

_LOGGER = logging.getLogger(__name__)

# Votes required before a tag-sourced gender is claimed, and the margin the
# winner must hold. Matches the roster's thresholds: this signal is cleaner,
# but it is also sparser, and one stray parse should not decide a voice.
_TAG_MINIMUM = 3
_TAG_MARGIN = 2

# How much narration either side of a quote can hold its tag.
_TAG_WINDOW = 120

_FEMININE = ("she",)
_MASCULINE = ("he",)
# "she said" and "said she", the two orders English puts a tag in.
_AFTER = re.compile(rf"^[\s,]*(?:(?P<f>she)|(?P<m>he))\s+(?:{_VERBS})\b", re.IGNORECASE)
_BEFORE = re.compile(rf"(?:(?P<f>she)|(?P<m>he))\s+(?:{_VERBS})[\s,:]*$", re.IGNORECASE)


def _tag_gender(before: str, after: str) -> str | None:
    """Return the gender a quote's adjacent tag states, if any."""
    for pattern, text in ((_AFTER, after), (_BEFORE, before)):
        match = pattern.search(text)
        if match is None:
            continue
        return "feminine" if match.group("f") else "masculine"
    return None


def tag_genders(
    chapters: Sequence[ChapterInspection], spans: Sequence[SpeakerSpan]
) -> dict[str, Counter[str]]:
    """Tally gender votes per character from their own dialogue tags."""
    text_by_id = {chapter.id: chapter.text for chapter in chapters}
    votes: dict[str, Counter[str]] = defaultdict(Counter)
    for span in spans:
        if span.character_id is None:
            continue
        text = text_by_id.get(span.chapter_id)
        if text is None:
            continue
        gender = _tag_gender(
            text[max(0, span.start - _TAG_WINDOW) : span.start],
            text[span.end : span.end + _TAG_WINDOW],
        )
        if gender is not None:
            votes[span.character_id][gender] += 1
    return dict(votes)


def _confident(tally: Counter[str]) -> str | None:
    """Return the tag-sourced gender when the vote is clear, else None."""
    if not tally:
        return None
    winner, count = tally.most_common(1)[0]
    other = tally["masculine" if winner == "feminine" else "feminine"]
    if count >= _TAG_MINIMUM and count >= _TAG_MARGIN * other:
        return winner
    return None


def apply(
    characters: Sequence[CharacterProfile], votes: Mapping[str, Counter[str]]
) -> tuple[CharacterProfile, ...]:
    """Let a confident tag vote decide, superseding whatever the roster read.

    The roster's pronoun signal is proximity -- a guess that the nearest
    pronoun refers to this name. A dialogue tag's pronoun *is* the speaker's,
    grammatically, so where the two disagree the tag is the better evidence
    and gap-filling would discard the better answer.

    A disagreement is still worth saying out loud: the only way a tag can be
    wrong is a mis-attributed quote, and that character's lines are already
    in the wrong voice if so. The log line names the bug, not the symptom.
    """
    filled = []
    for character in characters:
        gender = character.gender
        decided = _confident(votes.get(character.id, Counter()))
        if decided is not None:
            if gender is not None and gender != decided:
                log_event(
                    _LOGGER,
                    "dialogue_tag_gender_conflict",
                    # LogContext is str | int | bool, so the tally is flattened
                    # rather than passed as a dict.
                    context={
                        "boundary": "characters",
                        "character": character.id,
                        "roster": gender,
                        "tags": decided,
                        "feminine_votes": votes[character.id]["feminine"],
                        "masculine_votes": votes[character.id]["masculine"],
                    },
                )
            gender = decided
        filled.append(
            type(character)(
                id=character.id,
                display_name=character.display_name,
                gender=gender,
                spoken_characters=character.spoken_characters,
                chapter_ids=character.chapter_ids,
                aliases=character.aliases,
            )
        )
    return tuple(filled)
```

- [ ] **Step 4: Run the tests**

```bash
uv run pytest tests/test_dialogue_tags.py -q --no-cov
```

Expected: PASS.

- [ ] **Step 5: Wire it into the post-attribution rebuild**

`_measured` (`src/kenkui/_characters/__init__.py:141`) already rebuilds every profile after attribution — filling `spoken_characters` and `chapter_ids` — and passes `gender=character.gender` through untouched. Its single call site is `__init__.py:412`, where `spans` has just been assembled. Apply at the call site rather than inside `_measured`, so that function keeps one job.

Add to the imports at `__init__.py:20`:

```python
from kenkui._characters import dialogue_tags, spacy_roster, store
```

Then change the `characters=` argument at `__init__.py:412`:

```python
        # Attribution has just decided who speaks each quote, so `"..." she
        # said` now genders a known character. Done here, not in the roster:
        # the roster runs before any speaker is known.
        characters=dialogue_tags.apply(
            _measured(characters, spans),
            dialogue_tags.tag_genders(inspection.chapters, spans),
        ),
```

- [ ] **Step 6: Verify against the real book end to end**

```bash
uv run pytest tests/test_attribution.py tests/test_spacy_roster.py tests/test_dialogue_tags.py -q --no-cov
```

Expected: PASS.

- [ ] **Step 7: Run the full suite and commit**

```bash
uv run pytest -q
git add src/kenkui/_characters/dialogue_tags.py tests/test_dialogue_tags.py src/kenkui/_characters/__init__.py
git commit -m "feat: gender a speaker from their own dialogue tag

Attribution knows who spoke each quote, so '\"...\" she said' genders
that speaker with no proximity noise. A confident tag vote supersedes
the roster's proximity read, and any disagreement is logged: the only
way a tag is wrong is a mis-attributed quote, which is worth surfacing."
```

---

## Task 6: Confirm by ear

The defect was reported by listening; it has to be signed off by listening.

- [ ] **Step 1: Re-render the book**

The chunking schema bump means nothing is cached. Expect a full render.

```bash
uv run python spikes/examples/example.py
```

- [ ] **Step 2: Check the roster the run derived**

Confirm from the log that `ellie` is `feminine` and drew a feminine voice.

- [ ] **Step 3: Listen**

Spot-check several dialogue passages and several narration paragraphs in
`/Users/dizzler/Projects/Calibre Library/John Chu/The Subtle Art of Folding Space (465)/The Subtle Art of Folding Space - John Chu.m4b`.

Expected: sentences run to their natural end with no full stop landing three or four words in; Ellie is voiced by a feminine voice.

- [ ] **Step 4: Commit any tuning**

If a residual break is still audible, the next lever is `MIN_BREAK_FILL` (currently 0.7), not the budget — report before changing it.

---

## Self-Review

**Spec coverage.** Defect 1 → Tasks 2 (mechanism) and 3 (threshold), verified in Task 6. Defect 2 → Task 4 (both selected signals), extended by Task 5 (the dialogue-tag idea), verified in Tasks 4.12 and 6.2. Stale goldens → Task 1. Cache invalidation → Task 3.4.

**Placeholders.** None: every code step carries the literal code, every test step the literal test, every verification step the exact command and expected output.

**Type consistency.** `_break_offset` gains `tiers: tuple[str, ...] = _BREAK_TIERS` in Task 2.4 and is called with `_CLEAN_BREAK_TIERS` in 2.5. `_vote_gender` is split into `_majority` and `_gender_of` in Task 4.7; both call sites are updated (4.8 for production, 4.9 for the test). `_Signals.title_gender` is declared in 4.5, written in 4.6, read in 4.8. `tag_genders`/`apply`/`_confident` are defined in 5.3 and called in 5.5 with the signatures the tests in 5.1 assert; `log_event` takes `Mapping[str, str | int | bool]` (`observability.py:11`), so the conflict tally is flattened to two int fields rather than passed as a dict.

**Designs evaluated and rejected, with the measurement that settled each:** paragraph/speaker-only cutting (no quality gain — no cross-chunk conditioning; 4x the calls; wider identity blast radius; worse static-partition balance); a character-based token estimator (underestimates 89.6% of the time); a `gender_source` provenance field to rank honorifics above tags (would require a SQLite schema migration).

**Known gap accepted:** Task 3 removes the guard for token-dense text shorter than 200 characters (e.g. `"a-" * 60`, ~120 tokens). Measured across three real books this costs nothing — contents pages split on the newline tier and catalogue runs on the hyphen tier, both clean — and 120 tokens sits in the 0.002-deletion band, not the 150+ cliff. Recorded rather than fixed.

**Reproducing the synthesis measurement.** The harness is not committed (it needs a provisioned model, a compiled voice, and `faster-whisper`), so the method is recorded here instead: sample separator-free runs from a book bucketed by true token count using `languages/english_2026-04/tokenizer.model`; synthesize each twice through `TTSModel.generate_audio` — once whole, once as the chunker's fragments concatenated; transcribe with `faster-whisper small.en`; score **deletion rate**, not WER. Re-run it before changing `MAX_SEPARATOR_FREE_CHARACTERS` again.

**Deliberately out of scope:** non-person entities in the roster (`Amtrak`, `Boston`, `Metro`, `Carousel`, `Simulation`, `Belt`); and the silent whole-pool fallback for `gender=None`, which produces no warning today.
