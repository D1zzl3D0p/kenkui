# Book Tuning and the Dial-In Loop — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make an individual line of a book correctable — its speaker, the silence after it, and how its words are pronounced — through a per-book file that survives re-parsing and re-attribution.

**Architecture:** A deterministic addressable grid (Layer 0) derived from chapter text alone, and a tuning layer (Layer 1) of path-anchored rules that override machine attribution during planning rather than during resolution. Corrections are ordinary pipeline operations, serialized to a JSON sidecar beside the EPUB.

**Tech Stack:** Python 3.11–3.13, `uv`, pytest, mypy (strict), ruff, hatchling. No new runtime dependencies.

**Spec:** `docs/superpowers/specs/2026-09-08-grid-and-tuning-design.md`

## Scope

This plan implements spec sections 1–4, 5, 6, 8, and 9, plus the parser
prerequisite. It deliberately **excludes spec section 7 (The folds)**, which
is the only cache-invalidating work and belongs to a second plan.

Consequence: grid boundaries are not yet TTS segment boundaries. Attribution
overrides work regardless, because a speaker change forces a span edge and
`_chunk_span` chunks within spans. Manual silences force a span edge the same
way. Nothing in this plan re-renders the existing library.

Deferred to plan 2: folding `extract_spans`, `split_structural`, and
`_chunk_span` into the grid; `grid-v1` schema version; the
`pronounce`/`spoken_form` split; the break-quality metric.

Also deferred: `GridEdits` (manual splits and merges). The spec raises this
as an open question — the eager-split bias is designed to make under-splitting
impossible, and if it succeeds, splits never fire and merges are cosmetic.
Measure on Dune with `script()` from Task 12 before building the operation.

## Global Constraints

- **The repository is not under version control.** Run `git init` in
  `/Users/dizzler/Projects/Repos/kenkui-v2` before Task 1, or every commit
  step is a no-op.
- **Never `git add -A`.** Every commit step lists its files explicitly.
- **Never push.** Commits stay local.
- **Commits are test-gated.** Do not commit unless the task's tests pass.
- Working directory for all commands is `/Users/dizzler/Projects/Repos/kenkui-v2/kenkui`.
- **mypy strict** (`pyproject.toml:145`). Every new function is fully annotated.
- **Coverage floor is 90%** (`pyproject.toml:174`). New modules need real tests.
- **pytest runs with `--strict-config --strict-markers`.** Any new marker must
  be registered in `pyproject.toml` under `[tool.pytest.ini_options] markers`.
- Full gate, run before each commit:
  `uv run ruff format --check . && uv run ruff check . && uv run mypy && uv run pytest`
- **Grid level indices are 1-based.** Paragraph 1 is the first paragraph.
- **Chapter components are stable spine IDs (`str`), never indices.**
- **No new runtime dependencies.** `regex`, `spacy`-based splitting, and
  third-party sentence tokenizers are all out of scope; the splitter is `re`.
- Private modules live under `src/kenkui/_domain/` and `src/kenkui/_characters/`;
  public value types live at `src/kenkui/`. Tests are flat in `tests/`,
  named `test_<topic>.py`, matching the existing convention.

---

## File Structure

**Create:**

| file | responsibility |
|---|---|
| `src/kenkui/_domain/grid.py` | Layer 0: `Unit`, the splitter, grid construction from chapter text |
| `src/kenkui/_domain/paths.py` | `Path` and `Pattern` types, parsing, matching, subset ordering |
| `src/kenkui/_domain/tuning.py` | Precedence resolution: provenance layers, specificity, warnings |
| `src/kenkui/_domain/sidecar.py` | Sidecar JSON read/write and digest |
| `src/kenkui/script.py` | Public `Script` / `ScriptRow` read model |
| `src/kenkui/_domain/tiers.py` | Total identity/tuning/style classification of operations |

**Modify:**

| file | change |
|---|---|
| `src/kenkui/_epub/parser.py` | emit emphasis boundaries for `<em>`/`<i>` |
| `src/kenkui/_domain/operations.py` | new tuning operations; replace-or-create helper |
| `src/kenkui/pipeline.py` | `.attribute()`, `.silence()`, `.pronounce(where=)`, `.annotations()`, `.write_annotations()`, `.script()`, `.select()`, `.preview()`, tier properties |
| `src/kenkui/_domain/planning.py` | merge machine spans with tuning to produce effective spans |
| `src/kenkui/api.py` | `ValidationIssue.severity`; `is_valid` means "no errors" |
| `src/kenkui/__init__.py` | export new public types |

---

### Task 1: Shared test fixtures

Nine later tasks need a parseable EPUB, a resolved pipeline, and a chapter
with known machine attribution. `make_epub` exists today but is local to
`tests/test_epub.py`, and `conftest.py` has no book fixtures at all. Building
them once here keeps every later task's tests short and consistent.

**Files:**
- Create: `tests/helpers.py`
- Modify: `tests/test_epub.py`
- Modify: `tests/conftest.py`

**Interfaces:**
- Consumes: nothing.
- Produces, all importable from `tests/helpers.py` or available as fixtures:
  - `make_epub(path, *, chapters, spine, hrefs=None, title=..., author=..., cover=False)`
    and `xhtml(body, *, title=...)` — both moved verbatim from `test_epub.py`
  - `epub_path` — a two-chapter EPUB on disk, chapter IDs `ch08` and `ch09`
  - `chapter_ch08` — a `ChapterInspection` with known text
  - `machine_spans` — `tuple[SpeakerSpan, ...]` covering `chapter_ch08` exactly
  - `siblings` — empty `SiblingCounts`
  - `unit_ch08_p3_s2`, `unit_ch08_p1_s1` — `Unit` values
  - `unresolved_book`, `resolved_book` — pipelines over `epub_path`

- [ ] **Step 1: Move `make_epub` and `xhtml` into a shared helper**

Cut `make_epub` (`tests/test_epub.py:36`) and `xhtml`
(`tests/test_epub.py:88`) into a new `tests/helpers.py` unchanged — same
parameters, same defaults, same `noqa: PLR0913` comment — along with the
`CONTAINER` constant and imports they need, and import them back into
`test_epub.py`.

Chapter IDs come from the manifest item names, so `chapters={"ch08": ...}`
with `spine=["ch08", ...]` is what makes `chapter.id == "ch08"`. Step 4
verifies this; if the parser derives IDs from the href instead, change the
two constants in `conftest.py` and nothing else.

- [ ] **Step 2: Verify the move changed nothing**

Run: `uv run pytest tests/test_epub.py -v`
Expected: PASS, with the same test count as before the move.

- [ ] **Step 3: Add the fixtures**

```python
# tests/conftest.py
from kenkui._domain.planning import SpeakerSpan
from kenkui.inspection import ChapterInspection
from tests.helpers import make_epub, xhtml

CH08_TEXT = 'Alpha one. Alpha two.\n\n"Beta," she said. "Gamma," he answered.'
CH08_BODY = '<p>Alpha one. Alpha two.</p><p>"Beta," she said. "Gamma," he answered.</p>'


@pytest.fixture
def epub_path(tmp_path: Path) -> Path:
    return make_epub(
        tmp_path / "book.epub",
        chapters={"ch08": xhtml(CH08_BODY), "ch09": xhtml("<p>Delta.</p>")},
        spine=["ch08", "ch09"],
    )


@pytest.fixture
def chapter_ch08() -> ChapterInspection:
    return ChapterInspection(
        id="ch08",
        index=8,
        title="Chapter Eight",
        speech_characters=None,
        text=CH08_TEXT,
    )


@pytest.fixture
def machine_spans(chapter_ch08: ChapterInspection) -> tuple[SpeakerSpan, ...]:
    """One narration span covering the chapter exactly, as attribution emits."""
    return (SpeakerSpan("ch08", 0, len(chapter_ch08.text), None),)


@pytest.fixture
def siblings() -> dict[tuple[str, ...], int]:
    return {}


@pytest.fixture
def unresolved_book(epub_path: Path) -> kk.Pipeline:
    return kk.book(epub_path).assign_voice("ivy")
```

`resolved_book` builds on `unresolved_book` and must not reach the network.
Attribution is a model call, so the fixture stubs it: monkeypatch the
attribution provider to return narration for every span, then call
`resolve()`. Follow whatever stubbing `tests/test_attribution.py` already
does rather than inventing a second mechanism — run
`grep -n "monkeypatch\|stub\|fake" tests/test_attribution.py` first.

`unit_ch08_p3_s2` and `unit_ch08_p1_s1` are added in Task 7, when `Unit`
exists; leave them out here.

- [ ] **Step 4: Verify the fixtures resolve**

```python
# tests/test_fixtures.py
def test_epub_fixture_parses(epub_path) -> None:
    chapters = kk.book(epub_path).inspect().chapters
    assert [chapter.id for chapter in chapters] == ["ch08", "ch09"]


def test_machine_spans_tile_the_chapter(chapter_ch08, machine_spans) -> None:
    rebuilt = "".join(chapter_ch08.text[s.start : s.end] for s in machine_spans)
    assert rebuilt == chapter_ch08.text


def test_resolved_book_needs_no_network(resolved_book) -> None:
    assert resolved_book.inspect().casting is not None
```

Run: `uv run pytest tests/test_fixtures.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add tests/helpers.py tests/conftest.py tests/test_epub.py tests/test_fixtures.py
git commit -m "test: share EPUB and pipeline fixtures across the suite"
```

---

### Task 2: Recover emphasis in the EPUB parser

`_emit_element` (`_epub/parser.py:196`) handles `br` and block elements and
lets everything else fall through with no boundary and no marker, so `<em>`
and `<i>` vanish into plain text. For Dune this loses Herbert's italicized
interior monologue — a large share of the book, and exactly the text that
should not be in the narrator's voice.

This task records where emphasis was, without changing canonical text.

**Files:**
- Modify: `src/kenkui/_epub/parser.py`
- Modify: `src/kenkui/inspection.py`
- Test: `tests/test_emphasis.py`

**Interfaces:**
- Consumes: `make_epub`, `xhtml` (Task 1).
- Produces: `ChapterInspection.emphasis: tuple[tuple[int, int], ...]` — canonical
  `(start, end)` offset pairs, in document order, non-overlapping, sorted by start.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_emphasis.py
from pathlib import Path

from kenkui._epub.parser import inspect_epub
from tests.helpers import make_epub, xhtml


def only_chapter(tmp_path: Path, body: str):
    source = make_epub(tmp_path / "e.epub", chapters={"c1": xhtml(body)}, spine=["c1"])
    return inspect_epub(source).chapters[0]


def test_emphasis_offsets_recorded_without_changing_text(tmp_path: Path) -> None:
    chapter = only_chapter(
        tmp_path, "<p>He thought <em>I must not fear</em> and stopped.</p>"
    )
    assert chapter.text == "He thought I must not fear and stopped."
    assert chapter.emphasis == ((11, 26),)
    assert chapter.text[11:26] == "I must not fear"


def test_nested_emphasis_is_flattened_to_one_range(tmp_path: Path) -> None:
    chapter = only_chapter(tmp_path, "<p><em>Outer <i>inner</i> tail</em></p>")
    assert len(chapter.emphasis) == 1
    start, end = chapter.emphasis[0]
    assert chapter.text[start:end] == "Outer inner tail"


def test_no_emphasis_yields_empty_tuple(tmp_path: Path) -> None:
    assert only_chapter(tmp_path, "<p>Plain text.</p>").emphasis == ()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_emphasis.py -v`
Expected: FAIL — `ChapterInspection` has no attribute `emphasis`.

- [ ] **Step 3: Add the field**

```python
# src/kenkui/inspection.py — in ChapterInspection
    # Canonical (start, end) offsets of emphasised runs, in document order.
    # Recorded rather than marked inline: emphasis must not alter canonical
    # text, which billing, attribution offsets, and segment identity all use.
    emphasis: tuple[tuple[int, int], ...] = ()
```

- [ ] **Step 4: Emit emphasis in the parser**

In `_TextEmitter`, track the emitted length. In `_emit_element`, open a range
before descending into an emphasis tag and close it after, merging nested
ranges rather than nesting them:

```python
_EMPHASIS_ELEMENTS = frozenset({"em", "i", "cite", "dfn", "var"})
```

```python
    emphasis = tag in _EMPHASIS_ELEMENTS
    if emphasis:
        emitter.open_emphasis()
    ...
    if emphasis:
        emitter.close_emphasis()
```

`open_emphasis` increments a depth counter and records the start offset when
depth goes 0 → 1; `close_emphasis` decrements and appends `(start, current)`
when depth returns to 0. Nesting therefore flattens to one range, and a range
that is empty after normalization is discarded.

- [ ] **Step 5: Run tests to verify they pass**

Run: `uv run pytest tests/test_emphasis.py -v`
Expected: PASS

- [ ] **Step 6: Verify canonical text is unchanged for the whole test suite**

Run: `uv run pytest -v`
Expected: PASS. Any failure here means emphasis leaked into canonical text —
the field must be purely additive.

- [ ] **Step 7: Commit**

```bash
git add src/kenkui/_epub/parser.py src/kenkui/inspection.py tests/test_emphasis.py
git commit -m "feat(parser): record emphasis offsets without altering canonical text"
```

---

### Task 3: Unit type and the sentence/phrase splitter

The two new levels. Everything above them (`paragraph`, `line`) already
exists in `_domain/structure.py`; this task adds only what is missing, as
pure functions over a string.

**Files:**
- Create: `src/kenkui/_domain/grid.py`
- Test: `tests/test_grid_splitter.py`

**Interfaces:**
- Consumes: nothing.
- Produces:
  - `split_sentences(text: str) -> tuple[str, ...]`
  - `split_phrases(text: str) -> tuple[str, ...]`
  - Both preserve trailing separators so `"".join(parts) == text`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_grid_splitter.py
import pytest

from kenkui._domain.grid import split_phrases, split_sentences


def test_sentences_split_on_terminal_punctuation() -> None:
    assert split_sentences("One. Two! Three?") == ("One. ", "Two! ", "Three?")


def test_sentence_split_is_exact() -> None:
    text = "One.  Two!\tThree?"
    assert "".join(split_sentences(text)) == text


def test_abbreviation_does_not_split() -> None:
    assert split_sentences("Dr. Yueh smiled.") == ("Dr. Yueh smiled.",)


def test_single_initial_does_not_split() -> None:
    assert split_sentences("J. R. Smith left.") == ("J. R. Smith left.",)


def test_closing_quote_stays_with_its_sentence() -> None:
    assert split_sentences('"Go." He left.') == ('"Go." ', "He left.")


def test_ellipsis_splits_once() -> None:
    assert split_sentences("Wait... Then go.") == ("Wait... ", "Then go.")


def test_phrases_split_on_clause_punctuation() -> None:
    assert split_phrases("Yes, he said; then left.") == (
        "Yes, ",
        "he said; ",
        "then left.",
    )


def test_phrase_split_is_exact() -> None:
    text = "A, b; c: d"
    assert "".join(split_phrases(text)) == text


def test_empty_text_yields_no_parts() -> None:
    assert split_sentences("") == ()
    assert split_phrases("") == ()


@pytest.mark.parametrize(
    "text",
    [
        "No punctuation here",
        "Mr. and Mrs. Smith.",
        "He said 'no.' She left.",
        "A.B.C. Corp. filed.",
        "...",
        "  ",
    ],
)
def test_splitters_are_exact_over_awkward_input(text: str) -> None:
    assert "".join(split_sentences(text)) == text
    assert "".join(split_phrases(text)) == text
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_grid_splitter.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'kenkui._domain.grid'`

- [ ] **Step 3: Write the splitter**

```python
# src/kenkui/_domain/grid.py
"""The addressable grid: a partition of chapter text that depends on nothing else.

Every other partition in this codebase takes a setting, a budget, or a model.
This one takes only the text, which is what lets an annotation anchored to it
survive a change to pause tiers, chunk budgets, or the attribution provider.

Biased to split. Over-splitting costs one extra row in a reviewer's view and
nothing in the audio, because a grid boundary becomes a segment boundary only
when an annotation attaches to it. Under-splitting traps two speakers in one
addressable unit and makes the correction inexpressible. The two failures are
not symmetric, so the guard lists below are allowed to be short.
"""

from __future__ import annotations

import re

from kenkui._characters.identity import PREFIX_TITLES

# A terminator, any closing brackets or quotes, then whitespace. The lookahead
# keeps the whitespace with the preceding part so joining stays exact.
_SENTENCE = re.compile(r"[.!?…]+[\"'’\)\]]*\s+")
_PHRASE = re.compile(r"[,;:][\"'’\)\]]*\s+")

# A single capital before the period is an initial ("J. R. Smith"), not a
# sentence end.
_INITIAL = re.compile(r"(?:^|\s)[A-Z]\.$")


def _is_abbreviation(prefix: str) -> bool:
    """Whether a candidate sentence end is really an abbreviation."""
    if _INITIAL.search(prefix):
        return True
    trailing = re.search(r"([A-Za-z]+)\.$", prefix)
    return trailing is not None and trailing.group(1).casefold() in PREFIX_TITLES


def _split(text: str, pattern: re.Pattern[str], *, guard: bool) -> tuple[str, ...]:
    if not text:
        return ()
    parts: list[str] = []
    position = 0
    for match in pattern.finditer(text):
        if guard and _is_abbreviation(text[position : match.start() + 1]):
            continue
        parts.append(text[position : match.end()])
        position = match.end()
    if position < len(text):
        parts.append(text[position:])
    return tuple(parts) if parts else (text,)


def split_sentences(text: str) -> tuple[str, ...]:
    """Split on sentence-terminal punctuation, guarding abbreviations."""
    return _split(text, _SENTENCE, guard=True)


def split_phrases(text: str) -> tuple[str, ...]:
    """Split on clause punctuation. No guard: commas do not abbreviate."""
    return _split(text, _PHRASE, guard=False)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_grid_splitter.py -v`
Expected: PASS. If `PREFIX_TITLES` does not contain a title the tests need,
extend the test rather than the frozenset — the guard list is deliberately
short, and a miss here is cosmetic by design.

- [ ] **Step 5: Run the full gate**

Run: `uv run ruff format --check . && uv run ruff check . && uv run mypy && uv run pytest`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add src/kenkui/_domain/grid.py tests/test_grid_splitter.py
git commit -m "feat(grid): add exact sentence and phrase splitters"
```

---

### Task 4: Grid construction and the exactness property

Assembles the full hierarchy from existing pieces plus Task 2's splitters,
and asserts the invariant this codebase lives by.

**Files:**
- Modify: `src/kenkui/_domain/grid.py`
- Test: `tests/test_grid.py`
- Test: `tests/test_grid_exactness.py`

**Interfaces:**
- Consumes: `split_sentences`, `split_phrases` (Task 2);
  `structure._blocks`, `structure._lines`; `quotes.extract_spans`.
- Produces:
  - `Unit` — frozen dataclass with fields `chapter_id: str`, `paragraph: int`,
    `line: int`, `sentence: int`, `phrase: int`, `start: int`, `end: int`,
    `is_dialogue: bool`, `is_emphasised: bool`. All level indices 1-based.
  - `build_grid(chapter: ChapterInspection) -> tuple[Unit, ...]`
  - `unit_text(unit: Unit, chapter_text: str) -> str`
  - `unit_digest(unit: Unit, chapter_text: str) -> str` — `"sha256:<hex16>"`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_grid.py
from kenkui._domain.grid import build_grid, unit_digest, unit_text
from kenkui.inspection import ChapterInspection


def chapter(text: str, **kwargs: object) -> ChapterInspection:
    return ChapterInspection(
        id="ch01", index=0, title="One", speech_characters=None, text=text, **kwargs
    )


def test_grid_is_exact() -> None:
    text = 'Alpha one. Alpha two.\n\n"Beta," she said. "Gamma," he answered.'
    units = build_grid(chapter(text))
    assert "".join(unit_text(unit, text) for unit in units) == text


def test_levels_are_one_based_and_reset_under_their_parent() -> None:
    units = build_grid(chapter("One. Two.\n\nThree."))
    assert (units[0].paragraph, units[0].sentence) == (1, 1)
    assert (units[1].paragraph, units[1].sentence) == (1, 2)
    assert (units[2].paragraph, units[2].sentence) == (2, 1)


def test_quote_edges_force_boundaries() -> None:
    text = '"Yes," she said. "No," he answered.'
    units = build_grid(chapter(text))
    spoken = [unit_text(u, text) for u in units if u.is_dialogue]
    assert '"Yes,"' in "".join(spoken)
    assert '"No,"' in "".join(spoken)


def test_two_speakers_are_never_trapped_in_one_unit() -> None:
    text = '"Yes," she said. "No," he answered.'
    units = build_grid(chapter(text))
    for unit in units:
        rendered = unit_text(unit, text)
        assert not ("Yes" in rendered and "No" in rendered)


def test_line_level_separates_verse() -> None:
    text = "Line one\nLine two\n\nProse."
    units = build_grid(chapter(text))
    assert {unit.line for unit in units if unit.paragraph == 1} == {1, 2}


def test_emphasis_marks_units() -> None:
    text = "He thought I must not fear and stopped."
    units = build_grid(chapter(text, emphasis=((11, 26),)))
    assert any(unit.is_emphasised for unit in units)


def test_empty_chapter_yields_no_units() -> None:
    assert build_grid(chapter("")) == ()


def test_digest_is_stable_and_content_derived() -> None:
    text = "One. Two."
    units = build_grid(chapter(text))
    assert unit_digest(units[0], text) == unit_digest(units[0], text)
    assert unit_digest(units[0], text) != unit_digest(units[1], text)
    assert unit_digest(units[0], text).startswith("sha256:")
```

```python
# tests/test_grid_exactness.py
"""The grid's only safety property, asserted over real books.

Boundary placement is a quality metric to tune. Partition exactness is not
negotiable: a gap silently drops audio and an overlap silently duplicates it,
which is the same reasoning as _characters/quotes.py:7-10.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from kenkui._domain.grid import build_grid, unit_text
from kenkui.api import book

LIBRARY = Path("/Users/dizzler/Projects/Calibre Library")


def _epubs() -> list[Path]:
    if not LIBRARY.is_dir():
        return []
    return sorted(LIBRARY.rglob("*.epub"))[:40]


@pytest.mark.corpus
@pytest.mark.skipif(
    not os.environ.get("KENKUI_RUN_CORPUS"), reason="set KENKUI_RUN_CORPUS=1 to run"
)
@pytest.mark.parametrize("epub", _epubs(), ids=lambda p: p.stem)
def test_grid_partitions_every_chapter_exactly(epub: Path) -> None:
    for chapter in book(epub).inspect().chapters:
        units = build_grid(chapter)
        rebuilt = "".join(unit_text(unit, chapter.text) for unit in units)
        assert rebuilt == chapter.text, f"{epub.stem} / {chapter.id}"
```

- [ ] **Step 2: Register the corpus marker**

```toml
# pyproject.toml, under [tool.pytest.ini_options] markers
  "corpus: opt-in property test over the local Calibre library",
```

Required because pytest runs with `--strict-markers`.

- [ ] **Step 3: Run tests to verify they fail**

Run: `uv run pytest tests/test_grid.py -v`
Expected: FAIL — `build_grid` is not defined.

- [ ] **Step 4: Implement grid construction**

Append to `src/kenkui/_domain/grid.py`. Descend paragraph → line → sentence →
phrase, then force a boundary at every quote edge. Track a running offset so
each unit records `(start, end)` into the canonical chapter text and nothing
is copied.

```python
@dataclass(frozen=True, slots=True)
class Unit:
    """One addressable run of a chapter's canonical text."""

    chapter_id: str
    paragraph: int
    line: int
    sentence: int
    phrase: int
    start: int
    end: int
    is_dialogue: bool
    is_emphasised: bool


def unit_text(unit: Unit, chapter_text: str) -> str:
    """Return a unit's canonical text."""
    return chapter_text[unit.start : unit.end]


def unit_digest(unit: Unit, chapter_text: str) -> str:
    """Return a short content hash for anchor verification."""
    payload = unit_text(unit, chapter_text).encode("utf-8")
    return f"sha256:{hashlib.sha256(payload).hexdigest()[:16]}"


def _cut_points(text: str, offset: int, edges: frozenset[int]) -> list[int]:
    """Return offsets where a phrase must be broken by a quote edge."""
    return sorted(edge for edge in edges if offset < edge < offset + len(text))


def build_grid(chapter: ChapterInspection) -> tuple[Unit, ...]:
    """Partition one chapter into addressable units.

    Depends only on the chapter's canonical text, its recorded emphasis, and
    quote extraction -- which is itself a pure function of the text. No pause
    setting, chunk budget, or model can change the result.
    """
    text = chapter.text
    if not text:
        return ()
    spans = extract_spans(chapter.id, text)
    edges = frozenset({span.start for span in spans} | {span.end for span in spans})
    dialogue = tuple((s.start, s.end) for s in spans if s.is_dialogue)
    units: list[Unit] = []
    offset = 0
    for p_index, (body, chunk) in enumerate(_blocks(text), start=1):
        for l_index, line in enumerate(_lines(chunk, body), start=1):
            for s_index, sentence in enumerate(split_sentences(line), start=1):
                ph_index = 0
                for phrase in split_phrases(sentence):
                    for piece in _apply_cuts(phrase, offset, edges):
                        ph_index += 1
                        units.append(
                            Unit(
                                chapter.id,
                                p_index,
                                l_index,
                                s_index,
                                ph_index,
                                offset,
                                offset + len(piece),
                                is_dialogue=_covers(offset, len(piece), dialogue),
                                is_emphasised=_covers(
                                    offset, len(piece), chapter.emphasis
                                ),
                            )
                        )
                        offset += len(piece)
    return tuple(units)
```

`_apply_cuts(text, offset, edges)` splits `text` at each cut point from
`_cut_points`, yielding the pieces in order; with no cut points it yields
`text` unchanged. `_covers(start, length, ranges)` returns whether the unit's
midpoint falls inside any range — the midpoint rather than the start so a
leading quote character does not mark a narration unit as dialogue.

Add `import hashlib` and `from dataclasses import dataclass`, and import
`_blocks`, `_lines` from `kenkui._domain.structure`, `extract_spans` from
`kenkui._characters.quotes`, and `ChapterInspection` from
`kenkui.inspection` under `TYPE_CHECKING`.

- [ ] **Step 5: Run tests to verify they pass**

Run: `uv run pytest tests/test_grid.py -v`
Expected: PASS

- [ ] **Step 6: Run the corpus property test**

Run: `KENKUI_RUN_CORPUS=1 uv run pytest tests/test_grid_exactness.py -v --no-cov`
Expected: PASS for every book that parses. A failure here is a real defect in
the splitter, not a flaky test — fix `grid.py`, never the assertion. Books
that fail to parse at all are a pre-existing condition and out of scope; if
`inspect()` raises, note the title and skip it.

- [ ] **Step 7: Run the full gate**

Run: `uv run ruff format --check . && uv run ruff check . && uv run mypy && uv run pytest`
Expected: PASS

- [ ] **Step 8: Commit**

```bash
git add src/kenkui/_domain/grid.py tests/test_grid.py tests/test_grid_exactness.py pyproject.toml
git commit -m "feat(grid): build the addressable grid and assert partition exactness"
```

---

### Task 5: Paths and subtree addressing

**Files:**
- Create: `src/kenkui/_domain/paths.py`
- Test: `tests/test_paths.py`

**Interfaces:**
- Consumes: `Unit` (Task 3).
- Produces:
  - `LEVELS: tuple[str, ...]` = `("chapter", "paragraph", "line", "sentence", "phrase")`
  - `Path` — frozen dataclass, `chapter: str | None`, then
    `paragraph/line/sentence/phrase: int | None`; `None` means unspecified
  - `parse_path(mapping: Mapping[str, object]) -> Path`
  - `path_of(unit: Unit) -> Path` — the full leaf path
  - `render_path(path: Path) -> str` — display form, degenerate levels elided
  - `contains(outer: Path, inner: Path) -> bool` — subtree test

- [ ] **Step 1: Write the failing test**

```python
# tests/test_paths.py
import pytest

from kenkui._domain.paths import Path, contains, parse_path, render_path
from kenkui.errors import ValidationError


def test_parse_full_path() -> None:
    parsed = parse_path({"chapter": "ch08", "paragraph": 3, "sentence": 2})
    assert parsed == Path(
        chapter="ch08", paragraph=3, line=None, sentence=2, phrase=None
    )


def test_empty_mapping_is_the_whole_book() -> None:
    assert parse_path({}).chapter is None


def test_holes_are_allowed_and_mean_any_of_that_level() -> None:
    parsed = parse_path({"chapter": "ch08", "sentence": 2})
    assert parsed.paragraph is None
    assert parsed.sentence == 2


def test_unknown_level_is_refused() -> None:
    with pytest.raises(ValidationError):
        parse_path({"chapter": "ch08", "stanza": 1})


def test_zero_and_negative_indices_are_refused_in_paths() -> None:
    with pytest.raises(ValidationError):
        parse_path({"chapter": "ch08", "paragraph": 0})


def test_subtree_containment() -> None:
    paragraph = Path("ch08", 3, None, None, None)
    sentence = Path("ch08", 3, 1, 2, None)
    assert contains(paragraph, sentence)
    assert not contains(sentence, paragraph)


def test_containment_is_reflexive() -> None:
    path = Path("ch08", 3, None, None, None)
    assert contains(path, path)


def test_book_contains_everything() -> None:
    assert contains(Path(None, None, None, None, None), Path("ch08", 3, 1, 2, 1))


def test_render_elides_degenerate_middle_levels() -> None:
    assert render_path(Path("ch08", 3, 1, 2, None)) == "ch08  ¶3  s2"
    assert render_path(Path("ch08", 3, None, None, None)) == "ch08  ¶3"
    assert render_path(Path(None, None, None, None, None)) == "whole book"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_paths.py -v`
Expected: FAIL — no module `kenkui._domain.paths`.

- [ ] **Step 3: Implement paths**

`Path` is a frozen slots dataclass with all five fields defaulting to `None`.
`parse_path` rejects any key outside `LEVELS`, requires `chapter` to be a
`str` and the rest to be positive `int`s, and raises
`ValidationError(ErrorCode.INVALID_PATH)`. Add `INVALID_PATH` to `ErrorCode`
in `src/kenkui/errors.py`.

`contains(outer, inner)` returns `True` when, for every level, `outer`'s
component is either `None` or equal to `inner`'s. `render_path` joins the
non-`None` components with two spaces, labelling paragraph as `¶`, sentence
as `s`, phrase as `p`, and line as `l` — and omits `line` when it is `1`,
which is the display-time elision of degenerate levels. The stored `Path`
keeps the level.

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_paths.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/kenkui/_domain/paths.py src/kenkui/errors.py tests/test_paths.py
git commit -m "feat(paths): add labeled paths with subtree addressing"
```

---

### Task 6: Patterns and subset ordering

**Files:**
- Modify: `src/kenkui/_domain/paths.py`
- Test: `tests/test_patterns.py`

**Interfaces:**
- Consumes: `Path`, `LEVELS`, `Unit`.
- Produces:
  - `Selector` union: `Exact(value)`, `Any()`, `OneOf(values)`, `Span(lo, hi)`, `Last()`
  - `Pattern` — frozen mapping of level name to `Selector`
  - `parse_pattern(mapping: Mapping[str, object]) -> Pattern`
  - `matches(pattern: Pattern, unit: Unit, siblings: SiblingCounts) -> bool`
  - `subset(a: Pattern, b: Pattern) -> bool | None` — `None` means incomparable
  - `SiblingCounts = Mapping[tuple[str, ...], int]` — parent path to child count,
    used to resolve `Last()`
  - `Pattern.is_whole_book() -> bool`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_patterns.py
import pytest

from kenkui._domain.paths import parse_pattern, subset
from kenkui.errors import ValidationError


def test_wildcard_parses() -> None:
    assert parse_pattern({"chapter": "*", "paragraph": 1}) is not None


def test_list_and_range_and_last_parse() -> None:
    assert parse_pattern({"chapter": "ch08", "paragraph": [2, 4, 7]}) is not None
    assert parse_pattern({"chapter": "ch08", "paragraph": "2..4"}) is not None
    assert parse_pattern({"chapter": "ch08", "paragraph": -1}) is not None


def test_chapter_range_is_refused() -> None:
    # Stable spine IDs are unordered; ranges belong to select_chapter_range.
    with pytest.raises(ValidationError):
        parse_pattern({"chapter": "2..4"})


def test_concrete_is_a_strict_subset_of_wildcard() -> None:
    narrow = parse_pattern({"chapter": "ch08", "paragraph": 3})
    wide = parse_pattern({"chapter": "ch08", "paragraph": "*"})
    assert subset(narrow, wide) is True
    assert subset(wide, narrow) is False


def test_deeper_path_is_a_subset_of_its_ancestor() -> None:
    deep = parse_pattern({"chapter": "ch08", "paragraph": 3, "sentence": 2})
    shallow = parse_pattern({"chapter": "ch08", "paragraph": 3})
    assert subset(deep, shallow) is True


def test_crossing_patterns_are_incomparable() -> None:
    # The Irulan case: neither match set contains the other.
    a = parse_pattern({"chapter": "ch08", "paragraph": "*"})
    b = parse_pattern({"chapter": "*", "paragraph": 1})
    assert subset(a, b) is None
    assert subset(b, a) is None


def test_last_is_incomparable_with_a_concrete_index() -> None:
    # Resolving -1 needs a sibling count, so this cannot be decided statically.
    last = parse_pattern({"chapter": "ch08", "paragraph": -1})
    third = parse_pattern({"chapter": "ch08", "paragraph": 3})
    assert subset(last, third) is None


def test_last_is_a_subset_of_wildcard() -> None:
    last = parse_pattern({"chapter": "ch08", "paragraph": -1})
    wide = parse_pattern({"chapter": "ch08", "paragraph": "*"})
    assert subset(last, wide) is True


def test_list_subset_of_wider_list() -> None:
    narrow = parse_pattern({"chapter": "ch08", "paragraph": [2, 4]})
    wide = parse_pattern({"chapter": "ch08", "paragraph": [2, 4, 7]})
    assert subset(narrow, wide) is True
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_patterns.py -v`
Expected: FAIL — `parse_pattern` is not defined.

- [ ] **Step 3: Implement selectors, matching, and subset**

Each `Selector` implements `covers(value, sibling_count) -> bool` and
`within(other) -> bool | None`. `subset(a, b)` returns `True` when every
level's selector in `a` is within `b`'s, `False` when some level in `b` is
strictly narrower than `a`'s, and `None` when the levels disagree in
different directions — the incomparable case.

`Last().within(Any())` is `True`; `Last().within(Exact(n))` and
`Exact(n).within(Last())` are `None`, because deciding them needs a sibling
count that is not available at rule-authoring time.

An absent level in a pattern is `Any()`, so a shallow pattern is
automatically wider than a deeper one at the levels they share.

`parse_pattern` refuses a range or list at the `chapter` level and raises
`ValidationError(ErrorCode.INVALID_PATTERN)`.

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_patterns.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/kenkui/_domain/paths.py src/kenkui/errors.py tests/test_patterns.py
git commit -m "feat(paths): add set-valued patterns with partial subset ordering"
```

---

### Task 7: Precedence resolution

**Files:**
- Create: `src/kenkui/_domain/tuning.py`
- Test: `tests/test_precedence.py`

**Interfaces:**
- Consumes: `Unit`, `Pattern`, `subset`, `matches`.
- Produces:
  - `Rule` — frozen dataclass: `where: Pattern`, `value: object`, `index: int`
  - `Provenance` — `Literal["default", "unresolved", "machine", "rule"]`.
    There is no separate override rank: a fully concrete pattern is a subset
    of everything matching it, so exact overrides already win on specificity.
  - `Decision` — frozen dataclass: `value: object | None`, `provenance: Provenance`, `rule_index: int | None`
  - `resolve_rules(unit, machine_value, rules, siblings) -> Decision`
  - `overlap_warnings(rules) -> tuple[tuple[int, int], ...]` — incomparable pairs

- [ ] **Step 1: Write the failing test**

```python
# tests/test_precedence.py
from kenkui._domain.paths import parse_pattern
from kenkui._domain.tuning import Rule, overlap_warnings, resolve_rules


def rule(index: int, where: dict[str, object], value: str) -> Rule:
    return Rule(where=parse_pattern(where), value=value, index=index)


def test_no_rules_falls_through_to_machine(unit_ch08_p3_s2, siblings) -> None:
    decision = resolve_rules(unit_ch08_p3_s2, "narrator", (), siblings)
    assert decision.value == "narrator"
    assert decision.provenance == "machine"


def test_no_rules_and_no_machine_is_the_default(unit_ch08_p3_s2, siblings) -> None:
    decision = resolve_rules(unit_ch08_p3_s2, None, (), siblings)
    assert decision.provenance == "default"


def test_any_rule_beats_machine(unit_ch08_p3_s2, siblings) -> None:
    rules = (rule(0, {"chapter": "*"}, "irulan"),)
    decision = resolve_rules(unit_ch08_p3_s2, "paul", rules, siblings)
    assert decision.value == "irulan"
    assert decision.provenance == "rule"


def test_subset_wins_regardless_of_order(unit_ch08_p3_s2, siblings) -> None:
    wide = rule(0, {"chapter": "ch08"}, "paul")
    narrow = rule(1, {"chapter": "ch08", "paragraph": 3, "sentence": 2}, "jessica")
    assert (
        resolve_rules(unit_ch08_p3_s2, None, (wide, narrow), siblings).value
        == "jessica"
    )
    assert (
        resolve_rules(unit_ch08_p3_s2, None, (narrow, wide), siblings).value
        == "jessica"
    )


def test_incomparable_overlap_falls_back_to_declaration_order(
    unit_ch08_p1_s1, siblings
) -> None:
    a = rule(0, {"chapter": "ch08", "paragraph": "*"}, "paul")
    b = rule(1, {"chapter": "*", "paragraph": 1}, "irulan")
    assert resolve_rules(unit_ch08_p1_s1, None, (a, b), siblings).value == "irulan"
    assert resolve_rules(unit_ch08_p1_s1, None, (b, a), siblings).value == "paul"


def test_incomparable_overlap_is_reported() -> None:
    a = rule(0, {"chapter": "ch08", "paragraph": "*"}, "paul")
    b = rule(1, {"chapter": "*", "paragraph": 1}, "irulan")
    assert overlap_warnings((a, b)) == ((0, 1),)


def test_nested_rules_do_not_warn() -> None:
    wide = rule(0, {"chapter": "ch08"}, "paul")
    narrow = rule(1, {"chapter": "ch08", "paragraph": 3}, "jessica")
    assert overlap_warnings((wide, narrow)) == ()


def test_rule_index_is_reported_for_provenance(unit_ch08_p3_s2, siblings) -> None:
    rules = (rule(0, {"chapter": "*"}, "a"), rule(1, {"chapter": "ch08"}, "b"))
    assert resolve_rules(unit_ch08_p3_s2, None, rules, siblings).rule_index == 1
```

Add the two `Unit` fixtures to `tests/conftest.py` — `siblings` is already
there from Task 1, and `Unit` only exists as of Task 3:

```python
@pytest.fixture
def unit_ch08_p3_s2() -> Unit:
    return Unit("ch08", 3, 1, 2, 1, 0, 10, is_dialogue=False, is_emphasised=False)


@pytest.fixture
def unit_ch08_p1_s1() -> Unit:
    return Unit("ch08", 1, 1, 1, 1, 0, 10, is_dialogue=False, is_emphasised=False)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_precedence.py -v`
Expected: FAIL — no module `kenkui._domain.tuning`.

- [ ] **Step 3: Implement resolution**

```python
def resolve_rules(
    unit: Unit,
    machine_value: object | None,
    rules: tuple[Rule, ...],
    siblings: SiblingCounts,
) -> Decision:
    """Resolve one unit through the provenance layers.

    Provenance beats specificity. Machine attribution emits fully concrete
    paths, so under subset ordering alone it would beat every hand-written
    pattern -- inverting the model. Human rules are therefore considered as a
    layer above it, and specificity applies only within that layer.
    """
    candidates = [r for r in rules if matches(r.where, unit, siblings)]
    if not candidates:
        if machine_value is None:
            return Decision(None, "default", None)
        return Decision(machine_value, "machine", None)
    winner = candidates[0]
    for candidate in candidates[1:]:
        relation = subset(candidate.where, winner.where)
        if relation is True or (relation is None and candidate.index > winner.index):
            winner = candidate
    return Decision(winner.value, "rule", winner.index)
```

`overlap_warnings` returns every pair of rules that both could match the same
unit and for which `subset` is `None` in both directions. Two rules that
cannot co-match — different concrete chapters, for instance — do not warn.

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_precedence.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/kenkui/_domain/tuning.py tests/test_precedence.py tests/conftest.py
git commit -m "feat(tuning): resolve rules by provenance then subset specificity"
```

---

### Task 8: Tuning operations, tier classification, and pipeline methods

**Files:**
- Modify: `src/kenkui/_domain/operations.py`
- Create: `src/kenkui/_domain/tiers.py`
- Modify: `src/kenkui/pipeline.py`
- Test: `tests/test_tuning_operations.py`
- Test: `tests/test_tiers.py`

**Interfaces:**
- Consumes: `Pattern`, `parse_pattern`, `Rule`.
- Produces:
  - Operations `Attributions`, `Silences`, `Pronunciations`, `Annotations`,
    each frozen with a `rules: tuple[Rule, ...]` field (`Annotations` instead
    carries `path: Path`, `digest: str`, `loaded: Mapping[str, int]`).
  - `replace_or_append(operations, operation) -> tuple[Operation, ...]`
  - `tier_of(operation: type[Operation]) -> Literal["identity", "tuning", "style"] | None`
  - `Pipeline.attribute(character_id, *, where=None) -> Pipeline`
  - `Pipeline.silence(duration_ms, *, where=None) -> Pipeline`
  - `Pipeline.pronounce(lexicon=None, *, where=None, numbers=..., builtin=..., **features)`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_tuning_operations.py
import pytest

import kenkui as kk
from kenkui._domain.operations import Attributions, Silences
from kenkui.errors import ValidationError


def test_attribute_twice_accumulates(epub_path) -> None:
    book = (
        kk.book(epub_path)
        .attribute("irulan")
        .attribute("paul", where={"chapter": "ch08"})
    )
    ops = [op for op in book.operations if isinstance(op, Attributions)]
    assert len(ops) == 1
    assert len(ops[0].rules) == 2


def test_accumulation_preserves_declaration_order(epub_path) -> None:
    book = kk.book(epub_path).attribute("a").attribute("b")
    rules = next(op for op in book.operations if isinstance(op, Attributions)).rules
    assert [rule.value for rule in rules] == ["a", "b"]
    assert [rule.index for rule in rules] == [0, 1]


def test_pipeline_stays_immutable(epub_path) -> None:
    base = kk.book(epub_path)
    branched = base.attribute("irulan")
    assert base.operations != branched.operations


def test_style_operation_replaces_rather_than_raising(epub_path) -> None:
    book = kk.book(epub_path).pauses(paragraph_ms=400).pauses(paragraph_ms=500)
    pauses = [op for op in book.operations if type(op).__name__ == "Pauses"]
    assert len(pauses) == 1
    assert pauses[0].paragraph_ms == 500


def test_silence_accumulates_and_zero_is_allowed(epub_path) -> None:
    book = (
        kk.book(epub_path)
        .silence(900, where={"chapter": "ch08"})
        .silence(0, where={"chapter": "ch09"})
    )
    rules = next(op for op in book.operations if isinstance(op, Silences)).rules
    assert [rule.value for rule in rules] == [900, 0]


def test_negative_silence_is_refused(epub_path) -> None:
    with pytest.raises(ValidationError):
        kk.book(epub_path).silence(-1)


def test_where_defaults_to_the_whole_book(epub_path) -> None:
    book = kk.book(epub_path).attribute("narrator")
    rule = next(op for op in book.operations if isinstance(op, Attributions)).rules[0]
    assert rule.where.is_whole_book()
```

```python
# tests/test_tiers.py
from kenkui._domain import operations as ops
from kenkui._domain.tiers import tier_of


def _concrete_operations() -> list[type]:
    return [
        cls
        for cls in vars(ops).values()
        if isinstance(cls, type)
        and issubclass(cls, ops.Operation)
        and cls is not ops.Operation
    ]


def test_every_operation_is_classified() -> None:
    """A new operation must not vanish from all three tier views."""
    unclassified = [
        cls.__name__ for cls in _concrete_operations() if tier_of(cls) is None
    ]
    assert unclassified == []


def test_tuning_operations_are_the_ones_that_serialize() -> None:
    assert tier_of(ops.Attributions) == "tuning"
    assert tier_of(ops.Silences) == "tuning"
    assert tier_of(ops.Pronunciations) == "tuning"
    assert tier_of(ops.Annotations) == "tuning"


def test_style_and_identity_are_separated() -> None:
    assert tier_of(ops.Pauses) == "style"
    assert tier_of(ops.AssignVoices) == "style"
    assert tier_of(ops.SynthesizeSpeech) == "style"
    assert tier_of(ops.MetadataIntent) == "identity"
    assert tier_of(ops.Series) == "identity"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_tuning_operations.py tests/test_tiers.py -v`
Expected: FAIL — `Pipeline` has no attribute `attribute`.

- [ ] **Step 3: Add the operations and the replace-or-append helper**

```python
# src/kenkui/_domain/operations.py
def replace_or_append(
    operations: tuple[Operation, ...], operation: _OperationT
) -> tuple[Operation, ...]:
    """Return operations with one of this type replaced, or appended if absent.

    append_unique refuses a second operation of the same type because two
    conflicting settings must not both survive. Tuning and style both need to
    be re-callable, so they replace instead: tuning callers pass an extended
    rule tuple, style callers pass the new setting.
    """
    if any(type(item) is type(operation) for item in operations):
        return tuple(
            operation if type(item) is type(operation) else item for item in operations
        )
    if any(isinstance(item, SynthesizeSpeech) for item in operations):
        raise ValidationError(ErrorCode.INVALID_OPERATION_ORDER)
    return (*operations, operation)
```

Switch `pauses`, `pronounce`, `assign_voices`, `metadata`, and `series` from
`append_unique` to `replace_or_append`. Leave `select_chapters` and
`select_chapter_range` on `append_unique` — those two are genuinely
incompatible with each other, which is the case the error is for.

- [ ] **Step 4: Add the pipeline methods**

```python
def attribute(self, character_id: str, *, where: WhereArg = None) -> Pipeline:
    """Return a branch attributing matched units to one character.

    Accumulates: calling this twice extends the rule list rather than
    replacing it, and later rules break ties among rules whose match sets
    overlap without one containing the other.
    """
    return self._add_rule(Attributions, character_id, where)


def silence(self, duration_ms: int, *, where: WhereArg = None) -> Pipeline:
    """Return a branch setting the silence after matched positions.

    Replaces the derived gap duration rather than adding to it, so zero
    removes a pause. Anchors to the gap *after* a path; a path addressing
    a subtree normalizes to its last leaf.
    """
    if isinstance(duration_ms, bool) or not isinstance(duration_ms, int):
        raise ValidationError(ErrorCode.INVALID_PAUSE)
    if duration_ms < 0 or duration_ms > _MAX_PAUSE_MS:
        raise ValidationError(ErrorCode.INVALID_PAUSE)
    return self._add_rule(Silences, duration_ms, where)
```

`_add_rule` parses `where` (defaulting to the whole-book pattern), finds any
existing operation of that type, and returns
`replace_or_append(self.operations, kind(rules=(*existing, Rule(...))))` with
`index` set to `len(existing)`. It preserves `_resolved` — tuning consumes
resolved attribution without changing it, exactly as `_append`'s allowlist
does for `Pauses`.

Extend `pronounce()` with a `where` keyword that routes the `lexicon`
argument into a `Pronunciations` rule while leaving `numbers`, `builtin`, and
`**features` on `SpokenForm`. Splitting `SpokenForm` properly is plan 2.

- [ ] **Step 5: Add tier classification**

`tiers.py` holds one mapping from operation type to tier and a `tier_of`
returning `None` for anything unclassified, so `test_tiers.py` fails loudly
when an operation is added without a tier.

- [ ] **Step 6: Run tests to verify they pass**

Run: `uv run pytest tests/test_tuning_operations.py tests/test_tiers.py -v`
Expected: PASS

- [ ] **Step 7: Run the full suite — this step changes existing behaviour**

Run: `uv run pytest -v`
Expected: PASS, except tests asserting that a repeated `pauses()`,
`pronounce()`, `assign_voices()`, `metadata()`, or `series()` raises
`DUPLICATE_OPERATION`. Those assertions are now wrong by design — update them
to assert last-call-wins, and do not weaken the two selection operations.

- [ ] **Step 8: Commit**

```bash
git add src/kenkui/_domain/operations.py src/kenkui/_domain/tiers.py src/kenkui/pipeline.py tests/test_tuning_operations.py tests/test_tiers.py
git commit -m "feat(pipeline): add accumulating tuning operations and tier classification"
```

---

### Task 9: The sidecar

**Files:**
- Create: `src/kenkui/_domain/sidecar.py`
- Modify: `src/kenkui/pipeline.py`
- Test: `tests/test_sidecar.py`

**Interfaces:**
- Consumes: tuning operations (Task 7).
- Produces:
  - `SIDECAR_VERSION = 1`
  - `serialize(operations) -> dict[str, object]`
  - `deserialize(payload) -> tuple[Operation, ...]`
  - `sidecar_path(epub: Path) -> Path` — `<stem>.kenkui.json` beside the EPUB
  - `Pipeline.annotations(path=None) -> Pipeline`
  - `Pipeline.write_annotations(path=None) -> Path`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_sidecar.py
import json

import pytest

import kenkui as kk
from kenkui.errors import ValidationError


def test_round_trip_preserves_tuning(epub_path, tmp_path) -> None:
    book = (
        kk.book(epub_path)
        .attribute("irulan", where={"chapter": "*", "paragraph": 1})
        .silence(900, where={"chapter": "ch08", "paragraph": 3})
        .pronounce({"Atreides": "Ah-tray-deez"})
    )
    written = book.write_annotations(tmp_path / "b.kenkui.json")
    reloaded = kk.book(epub_path).annotations(written)
    assert repr(reloaded.tuning) == repr(book.tuning)


def test_sidecar_defaults_beside_the_epub(epub_path) -> None:
    book = kk.book(epub_path).attribute("irulan")
    written = book.write_annotations()
    assert written.parent == epub_path.parent
    assert written.name.endswith(".kenkui.json")
    written.unlink()


def test_digest_changes_the_plan_fingerprint(epub_path, tmp_path) -> None:
    first = kk.book(epub_path).attribute("a").write_annotations(tmp_path / "s.json")
    one = kk.book(epub_path).annotations(first)
    kk.book(epub_path).attribute("b").write_annotations(tmp_path / "s.json")
    two = kk.book(epub_path).annotations(tmp_path / "s.json")
    assert one.operations != two.operations


def test_loading_twice_is_refused(epub_path, tmp_path) -> None:
    written = kk.book(epub_path).attribute("a").write_annotations(tmp_path / "s.json")
    with pytest.raises(ValidationError):
        kk.book(epub_path).annotations(written).annotations(written)


def test_malformed_sidecar_is_refused(tmp_path, epub_path) -> None:
    bad = tmp_path / "bad.kenkui.json"
    bad.write_text("[]", encoding="utf-8")
    with pytest.raises(ValidationError):
        kk.book(epub_path).annotations(bad)


def test_unknown_version_is_refused(tmp_path, epub_path) -> None:
    bad = tmp_path / "v9.kenkui.json"
    bad.write_text(json.dumps({"kenkui_sidecar": 9}), encoding="utf-8")
    with pytest.raises(ValidationError):
        kk.book(epub_path).annotations(bad)


def test_only_tuning_serializes(epub_path, tmp_path) -> None:
    book = (
        kk.book(epub_path).assign_voice("ivy").pauses(paragraph_ms=400).attribute("a")
    )
    payload = json.loads(book.write_annotations(tmp_path / "s.json").read_text())
    assert "attributions" in payload
    assert "voices" not in payload
    assert "pauses" not in payload
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_sidecar.py -v`
Expected: FAIL — `Pipeline` has no attribute `write_annotations`.

- [ ] **Step 3: Implement the format**

```json
{
  "kenkui_sidecar": 1,
  "attributions": [
    {"where": {"chapter": "*", "paragraph": 1}, "character": "irulan", "matched": 42},
    {"where": {"chapter": "ch08", "paragraph": 3, "sentence": 2}, "character": "jessica",
     "digest": "sha256:1a2b3c4d5e6f7081"}
  ],
  "silences": [{"where": {"chapter": "ch08", "paragraph": 3}, "ms": 900}],
  "pronunciations": [{"where": {}, "words": {"Atreides": "Ah-tray-deez"}}],
}
```

`"where": {}` is the whole book. A rule whose pattern is fully concrete
carries `digest`; a rule containing any set-valued component carries
`matched`, the count of units it matched at authoring time. `deserialize`
refuses a payload that is not an object, whose `kenkui_sidecar` is not
`SIDECAR_VERSION`, or whose rules fail `parse_pattern`, raising
`ValidationError(ErrorCode.INVALID_SIDECAR)`.

`annotations()` reads the file, appends the deserialized operations, and
records an `Annotations` operation carrying the path, the file's content
digest, and the per-kind load counts. The digest is what makes an edited
sidecar change the plan fingerprint — without it a stale render would be
served from cache, which is the same reasoning as `resolve()`'s use of
`source_digest` at `pipeline.py:383`. `annotations()` uses `append_unique`,
so loading twice raises.

`write_annotations()` serializes exactly the operations `tier_of` classifies
as tuning, excluding `Annotations` itself.

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_sidecar.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/kenkui/_domain/sidecar.py src/kenkui/pipeline.py src/kenkui/errors.py tests/test_sidecar.py
git commit -m "feat(sidecar): persist per-book tuning beside the EPUB"
```

---

### Task 10: Warning severity in validation

**Files:**
- Modify: `src/kenkui/api.py`
- Modify: `src/kenkui/pipeline.py`
- Test: `tests/test_validation_severity.py`

**Interfaces:**
- Consumes: `overlap_warnings` (Task 6).
- Produces:
  - `ValidationIssue.severity: Literal["error", "warning"]`, defaulting to `"error"`
  - `ValidationResult.errors` and `ValidationResult.warnings` properties
  - `is_valid` means "no errors"

- [ ] **Step 1: Write the failing test**

```python
# tests/test_validation_severity.py
import kenkui as kk


def test_warning_does_not_invalidate(epub_path) -> None:
    book = (
        kk.book(epub_path)
        .assign_voice("ivy")
        .attribute("paul", where={"chapter": "ch08", "paragraph": "*"})
        .attribute("irulan", where={"chapter": "*", "paragraph": 1})
    )
    result = book.validate()
    assert result.warnings
    assert result.is_valid


def test_error_still_invalidates(tmp_path) -> None:
    result = kk.book(tmp_path / "missing.epub").validate()
    assert not result.is_valid
    assert result.errors


def test_existing_issues_default_to_error() -> None:
    assert (
        kk.ValidationIssue(code=kk.ErrorCode.INVALID_OUTPUT, message="x").severity
        == "error"
    )
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_validation_severity.py -v`
Expected: FAIL — `ValidationResult` has no attribute `warnings`.

- [ ] **Step 3: Add severity**

`severity` defaults to `"error"` so every existing construction site keeps its
meaning. `is_valid` becomes `not self.errors`.

- [ ] **Step 4: Verify the render path still refuses on errors**

`write_m4b` raises on `validation.issues[0]` at `pipeline.py:549`. Change it
to inspect `validation.errors[0]` instead, or a warning will abort a render
that should have proceeded.

- [ ] **Step 5: Emit the tuning warnings from `validate()`**

Report incomparable rule overlap as a warning. Anchor-digest mismatch and
pattern match-count drift require the grid, so they are reported by
`script()` in Task 11 — `validate()` is documented as inexpensive and must
not start parsing.

- [ ] **Step 6: Run the full suite**

Run: `uv run pytest -v`
Expected: PASS. Failures here mean a call site assumed `is_valid` meant "no
issues at all"; each one needs reading, not a blanket edit.

- [ ] **Step 7: Commit**

```bash
git add src/kenkui/api.py src/kenkui/pipeline.py tests/test_validation_severity.py
git commit -m "feat(validation): separate warnings from errors"
```

---

### Task 11: Merge tuning into planning

The task the loop's speed depends on. Resolution must keep producing only the
machine layer; planning merges.

**Files:**
- Modify: `src/kenkui/_domain/planning.py`
- Test: `tests/test_tuning_merge.py`

**Interfaces:**
- Consumes: `build_grid`, `resolve_rules`, tuning operations.
- Produces:
  - `effective_spans(chapter, machine_spans, operations) -> tuple[SpeakerSpan, ...]`
  - `manual_gaps(chapter, operations) -> Mapping[int, int]` — unit index → forced ms

- [ ] **Step 1: Write the failing test**

```python
# tests/test_tuning_merge.py
from kenkui._domain.operations import Attributions, Silences
from kenkui._domain.paths import parse_pattern
from kenkui._domain.planning import effective_spans, manual_gaps
from kenkui._domain.tuning import Rule


def attributions(*pairs: tuple[str, dict[str, object]]) -> tuple[Attributions]:
    """Build the operation tuple directly; no test hook on the public API."""
    rules = tuple(
        Rule(where=parse_pattern(where), value=value, index=index)
        for index, (value, where) in enumerate(pairs)
    )
    return (Attributions(rules=rules),)


def silences(*pairs: tuple[int, dict[str, object]]) -> tuple[Silences]:
    rules = tuple(
        Rule(where=parse_pattern(where), value=value, index=index)
        for index, (value, where) in enumerate(pairs)
    )
    return (Silences(rules=rules),)


def test_override_replaces_machine_attribution(chapter_ch08, machine_spans) -> None:
    book = attributions(("jessica", {"chapter": "ch08", "paragraph": 3, "sentence": 2}))
    spans = effective_spans(chapter_ch08, machine_spans, book)
    covering = [s for s in spans if s.character_id == "jessica"]
    assert len(covering) == 1


def test_spans_still_tile_the_chapter(chapter_ch08, machine_spans) -> None:
    book = attributions(("paul", {"chapter": "ch08"}))
    spans = effective_spans(chapter_ch08, machine_spans, book)
    rebuilt = "".join(chapter_ch08.text[s.start : s.end] for s in spans)
    assert rebuilt == chapter_ch08.text


def test_adjacent_units_with_one_speaker_merge_into_one_span(
    chapter_ch08, machine_spans
) -> None:
    book = attributions(("paul", {"chapter": "ch08"}))
    spans = effective_spans(chapter_ch08, machine_spans, book)
    assert len(spans) == 1


def test_no_tuning_returns_machine_spans_unchanged(chapter_ch08, machine_spans) -> None:
    assert effective_spans(chapter_ch08, machine_spans, ()) == machine_spans


def test_silence_normalizes_to_the_last_leaf(chapter_ch08) -> None:
    paragraph = silences((900, {"chapter": "ch08", "paragraph": 3}))
    sentence = silences((900, {"chapter": "ch08", "paragraph": 3, "sentence": -1}))
    assert manual_gaps(chapter_ch08, paragraph) == manual_gaps(chapter_ch08, sentence)


def test_zero_removes_a_pause(chapter_ch08) -> None:
    ops = silences((0, {"chapter": "ch08", "paragraph": 3}))
    assert 0 in manual_gaps(chapter_ch08, ops).values()
```

`chapter_ch08` and `machine_spans` come from Task 1's fixtures.

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_tuning_merge.py -v`
Expected: FAIL — `effective_spans` is not defined.

- [ ] **Step 3: Implement the merge**

```python
def effective_spans(
    chapter: ChapterInspection,
    machine_spans: tuple[SpeakerSpan, ...],
    operations: tuple[Operation, ...],
) -> tuple[SpeakerSpan, ...]:
    """Merge machine attribution with tuning rules into the spans planning compiles.

    Deliberately here and not in resolution. Resolution produces the machine
    layer only, which is what lets a correction preserve a resolved pipeline
    and cost no model calls, no store read, and no re-resolution -- just a
    re-plan and the two or three segments whose identity actually changed.
    """
    rules = _rules_of(operations, Attributions)
    if not rules:
        return machine_spans
    grid = build_grid(chapter)
    siblings = _sibling_counts(grid)
    by_offset = _machine_lookup(machine_spans)
    decided = [
        resolve_rules(unit, by_offset(unit), rules, siblings).value for unit in grid
    ]
    return _coalesce(chapter.id, grid, decided)
```

`_coalesce` merges adjacent units sharing a character into one `SpeakerSpan`,
which keeps the span count — and therefore the segment count — close to
today's. `manual_gaps` resolves each `Silences` rule's pattern to its matched
subtrees, takes each subtree's last unit, and maps that unit's index to the
duration; later rules win on collision through the same `resolve_rules` path.

Then wire both into `_append_chapter` (`planning.py:537`), forcing a chunk
boundary at any unit index present in `manual_gaps`.

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_tuning_merge.py -v`
Expected: PASS

- [ ] **Step 5: Assert the resolved pipeline survives a correction**

```python
# tests/test_tuning_operations.py — append
def test_correction_preserves_resolution(resolved_book) -> None:
    """A correction must not cost a re-resolve; it layers over the machine result."""
    corrected = resolved_book.attribute("jessica", where={"chapter": "ch08"})
    assert corrected._resolved is resolved_book._resolved
```

Run: `uv run pytest tests/test_tuning_operations.py -v`
Expected: PASS. A failure means the merge landed in resolution rather than
planning, which silently costs a full resolve on every correction.

- [ ] **Step 6: Run the full gate**

Run: `uv run ruff format --check . && uv run ruff check . && uv run mypy && uv run pytest`
Expected: PASS

- [ ] **Step 7: Commit**

```bash
git add src/kenkui/_domain/planning.py tests/test_tuning_merge.py tests/test_tuning_operations.py tests/conftest.py
git commit -m "feat(planning): merge tuning over machine attribution at plan time"
```

---

### Task 12: The `script()` read model

**Files:**
- Create: `src/kenkui/script.py`
- Modify: `src/kenkui/pipeline.py`
- Modify: `src/kenkui/__init__.py`
- Test: `tests/test_script.py`

**Interfaces:**
- Consumes: `build_grid`, `resolve_rules`, `render_path`.
- Produces:
  - `ScriptRow` — frozen: `path: Path`, `text: str`, `character: str | None`,
    `provenance: Provenance`, `rule_index: int | None`, `silence_after_ms: int`,
    `is_dialogue: bool`, `is_emphasised: bool`
  - `Script` — `Mapping`-like: `script[path]`, `script.at(pattern)`, iteration,
    `script.warnings`
  - `Pipeline.script() -> Script`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_script.py
import kenkui as kk


def test_rows_cover_the_chapter_in_order(resolved_book) -> None:
    rows = list(resolved_book.script().at({"chapter": "ch08"}))
    assert rows == sorted(
        rows, key=lambda row: (row.path.paragraph or 0, row.path.sentence or 0)
    )
    assert "".join(row.text for row in rows)


def test_provenance_reports_the_winning_rule(resolved_book) -> None:
    book = resolved_book.attribute("irulan", where={"chapter": "*", "paragraph": 1})
    row = next(iter(book.script().at({"chapter": "ch08", "paragraph": 1})))
    assert row.character == "irulan"
    assert row.provenance == "rule"
    assert row.rule_index == 0


def test_machine_provenance_when_no_rule_matches(resolved_book) -> None:
    rows = list(resolved_book.script().at({"chapter": "ch08"}))
    assert {row.provenance for row in rows} <= {"machine", "default"}


def test_script_works_before_resolution(unresolved_book) -> None:
    rows = list(unresolved_book.script().at({"chapter": "ch08"}))
    assert rows
    assert all(row.provenance in {"unresolved", "rule"} for row in rows)


def test_chapters_materialize_lazily(resolved_book) -> None:
    script = resolved_book.script()
    assert script.materialized == ()
    list(script.at({"chapter": "ch08"}))
    assert script.materialized == ("ch08",)


def test_anchor_digest_mismatch_is_reported(resolved_book) -> None:
    """A stale anchor must surface a conflict, never silently move."""
    stale_rule = Rule(
        where=parse_pattern({"chapter": "ch08", "paragraph": 3, "sentence": 2}),
        value="jessica",
        index=0,
        digest="sha256:deadbeefdeadbeef",
    )
    stale = replace(
        resolved_book,
        operations=(*resolved_book.operations, Attributions(rules=(stale_rule,))),
    )
    assert any("digest" in warning.message for warning in stale.script().warnings)
```

Imports for this module: `from dataclasses import replace`,
`from kenkui._domain.operations import Attributions`,
`from kenkui._domain.paths import parse_pattern`,
`from kenkui._domain.tuning import Rule`.

`Rule` gains an optional `digest: str | None = None` field in Task 7 for
exactly this check; add it there if it is not already present.

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_script.py -v`
Expected: FAIL — `Pipeline` has no attribute `script`.

- [ ] **Step 3: Implement `Script`**

A method, not a property: it parses, and this codebase reserves methods for
work — `metadata_intent` is a property because it is a tuple scan, `inspect()`
is a method because it parses.

`Script` holds the pipeline and a mutable per-chapter cache. `at(pattern)`
resolves which chapters the pattern selects, builds the grid for those only,
and yields rows. `materialized` exposes which chapters have been built, which
is what the laziness test asserts. `warnings` accumulates digest mismatches
and pattern match-count drift found while materializing — the two findings
`validate()` cannot report without parsing.

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_script.py -v`
Expected: PASS

- [ ] **Step 5: Export the public types**

```python
# src/kenkui/__init__.py
from kenkui.script import Script, ScriptRow
```

Add both to `__all__`. `tests/test_package.py` checks the public surface — run
it and update the expected export list.

- [ ] **Step 6: Run the full gate**

Run: `uv run ruff format --check . && uv run ruff check . && uv run mypy && uv run pytest`
Expected: PASS

- [ ] **Step 7: Commit**

```bash
git add src/kenkui/script.py src/kenkui/pipeline.py src/kenkui/__init__.py tests/test_script.py tests/test_package.py
git commit -m "feat(script): add the per-unit read model with provenance"
```

---

### Task 13: `select()` and `preview()`

**Files:**
- Modify: `src/kenkui/pipeline.py`
- Test: `tests/test_select_preview.py`

**Interfaces:**
- Consumes: `Pattern`, `build_grid`.
- Produces:
  - `Pipeline.select(*patterns) -> Pipeline`
  - `Pipeline.preview(output, *, on_event=None, workers="auto", overwrite=False) -> Result`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_select_preview.py
import pytest

import kenkui as kk
from kenkui.errors import ValidationError


def test_select_narrows_to_a_paragraph(resolved_book) -> None:
    probe = resolved_book.select({"chapter": "ch08", "paragraph": 3})
    rows = list(probe.script())
    assert {row.path.paragraph for row in rows} == {3}


def test_select_accepts_several_patterns(resolved_book) -> None:
    probe = resolved_book.select(
        {"chapter": "ch08", "paragraph": 3}, {"chapter": "ch08", "paragraph": 5}
    )
    assert {row.path.paragraph for row in probe.script()} == {3, 5}


def test_select_conflicts_with_chapter_selection(resolved_book) -> None:
    with pytest.raises(ValidationError):
        resolved_book.select_chapters("ch08").select({"chapter": "ch09"})


def test_preview_refuses_an_m4b_extension(resolved_book, tmp_path) -> None:
    with pytest.raises(ValidationError):
        resolved_book.select({"chapter": "ch08"}).preview(tmp_path / "probe.m4b")


@pytest.mark.pocket_real
def test_probe_segments_match_the_full_render(resolved_book) -> None:
    """A probe must warm the cache the full render will read.

    Grid boundaries are a pure function of chapter text, so narrowing the
    selection must not change segment identity -- otherwise every probe is
    wasted work. The final gap is exempt: it has no following unit.
    """
    whole = _segment_ids(resolved_book.select({"chapter": "ch08"}))
    part = _segment_ids(resolved_book.select({"chapter": "ch08", "paragraph": 3}))
    assert set(part[:-1]) <= set(whole)
```

`_segment_ids` compiles a plan without synthesizing and returns its segment
identities in order:

```python
def _segment_ids(pipeline: kk.Pipeline) -> list[str]:
    from kenkui._domain.planning import compile_plan

    return [segment.identity for segment in compile_plan(pipeline).segments]
```

`compile_plan` and `identity` are illustrative names for the plan-building
entry point in `_domain/planning.py`. Confirm them with
`grep -n "^def \|identity" src/kenkui/_domain/planning.py` and use the real
ones — the assertion is what matters, not the accessor spelling.

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_select_preview.py -v`
Expected: FAIL — `Pipeline` has no attribute `select`.

- [ ] **Step 3: Implement `select` and `preview`**

`select` records a `Select` operation via `append_unique` and raises
`DUPLICATE_OPERATION` alongside `SelectChapters` and `SelectChapterRange` —
three selection modes are mutually exclusive for the same reason two are
today. `inspect()` gains a branch narrowing chapters to those the patterns
touch, and planning trims each chapter's units to the matched range.

`preview` reuses `write_m4b`'s validation and execution, writing WAV instead
of M4B and skipping chaptering and metadata. It refuses an `.m4b` extension
so a probe is never confused with a publication.

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_select_preview.py -v -m "not pocket_real"`
Expected: PASS

- [ ] **Step 5: Run the cache-warming assertion**

Run: `uv run pytest tests/test_select_preview.py -v -m pocket_real --no-cov`
Expected: PASS. This is the property that makes the loop worth building; if it
fails, selection is changing segment identity and every probe is wasted work.

- [ ] **Step 6: Commit**

```bash
git add src/kenkui/pipeline.py tests/test_select_preview.py
git commit -m "feat(pipeline): add sub-chapter selection and lightweight preview"
```

---

### Task 14: Tier introspection

**Files:**
- Modify: `src/kenkui/pipeline.py`
- Create: `src/kenkui/_domain/summary.py`
- Test: `tests/test_introspection.py`

**Interfaces:**
- Consumes: `tier_of` (Task 7), `render_path` (Task 4).
- Produces: `Pipeline.identity`, `Pipeline.tuning`, `Pipeline.style` — properties
  returning objects whose `__repr__` is the summary and which iterate their rules.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_introspection.py
import pytest

import kenkui as kk


def test_tuning_lists_rules_in_declaration_order(epub_path) -> None:
    book = kk.book(epub_path).attribute("a").attribute("b")
    rendered = repr(book.tuning)
    assert rendered.index("a") < rendered.index("b")


def test_tuning_marks_unsaved_rules(epub_path, tmp_path) -> None:
    saved = kk.book(epub_path).attribute("a").write_annotations(tmp_path / "s.json")
    book = kk.book(epub_path).annotations(saved).attribute("b")
    assert "unsaved" in repr(book.tuning)
    assert repr(book.tuning).count("unsaved") == 1


def test_display_elides_degenerate_line_level(epub_path) -> None:
    book = kk.book(epub_path).attribute(
        "jessica", where={"chapter": "ch08", "paragraph": 3, "line": 1, "sentence": 2}
    )
    assert "¶3  s2" in repr(book.tuning)
    assert "l1" not in repr(book.tuning)


def test_long_rule_lists_are_truncated_but_iterable(epub_path) -> None:
    book = kk.book(epub_path)
    for index in range(50):
        book = book.attribute(f"c{index}", where={"chapter": f"ch{index:02d}"})
    assert "more" in repr(book.tuning)
    assert len(list(book.tuning)) == 50


def test_style_shows_effective_values_after_replacement(epub_path) -> None:
    book = kk.book(epub_path).pauses(paragraph_ms=400).pauses(paragraph_ms=500)
    assert "500ms" in repr(book.style)
    assert "400ms" not in repr(book.style)


def test_introspection_does_not_parse(epub_path, monkeypatch) -> None:
    """These are properties, so they must not reach the filesystem."""
    monkeypatch.setattr(
        "kenkui._epub.inspect_epub", lambda *a, **k: pytest.fail("parsed")
    )
    repr(kk.book(epub_path).attribute("a").tuning)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_introspection.py -v`
Expected: FAIL — `Pipeline` has no attribute `tuning`.

- [ ] **Step 3: Implement the summaries**

`__repr__`, not `__str__` — the point is typing `book.tuning` in a REPL, where
`repr` is what shows. Rules render in declaration order and are never sorted;
order is the precedence tiebreaker, so a sorted view would misrepresent the
render. Lists longer than three rules per kind truncate with `… N more` while
iteration still yields everything. `unsaved` is derived from the `Annotations`
operation's per-kind load counts: rules at an index at or beyond the loaded
count were added in code.

No match counts. The property is a tuple scan with no parse and no I/O; drift
belongs to `script()`, which already does the work.

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_introspection.py -v`
Expected: PASS

- [ ] **Step 5: Update the example to the new shape**

Rewrite `spikes/examples/example.py` to use a `house_style` function passed
through `.pipe()`, with `LEXICONS` still supplied per series in code and
`.annotations()` loading each book's sidecar. This is the spec's worked
example and the file most likely to be read first.

- [ ] **Step 6: Run the full gate**

Run: `uv run ruff format --check . && uv run ruff check . && uv run mypy && uv run pytest`
Expected: PASS, with coverage at or above 90%.

- [ ] **Step 7: Commit**

```bash
git add src/kenkui/pipeline.py src/kenkui/_domain/summary.py tests/test_introspection.py spikes/examples/example.py
git commit -m "feat(pipeline): add identity, tuning, and style introspection"
```

---

## Verification

After Task 14, the loop from the spec runs end to end:

```bash
cd /Users/dizzler/Projects/Repos/kenkui-v2/kenkui
uv run ruff format --check . && uv run ruff check . && uv run mypy && uv run pytest
KENKUI_RUN_CORPUS=1 uv run pytest tests/test_grid_exactness.py --no-cov -q
```

Then, manually, against a real book:

```python
import kenkui as kk
from pathlib import Path

epub = Path(
    "/Users/dizzler/Projects/Calibre Library/Frank Herbert/Dune (466)/Dune - Frank Herbert.epub"
)
book = kk.book(epub).pipe(house_style).annotations().resolve()

for row in book.script().at({"chapter": book.inspect().chapters[8].id}):
    print(row.path, row.character, row.provenance, row.text[:60])
```

Expect: one row per grid unit, provenance naming where each speaker came
from, and no unit containing two speakers' dialogue.
