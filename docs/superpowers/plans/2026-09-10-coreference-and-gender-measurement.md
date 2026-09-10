# Coreference and Gender Measurement Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make both defects measurable — one person getting several voices, and a
character getting the wrong gender — then run every candidate method against that
measurement and report what each costs.

**Architecture:** A new eval track alongside the existing attribution eval. It
scores *partitions of surface names* rather than ballots of ids, weighted by
spoken volume, reporting splits and merges separately. Pure scoring and citation
verification live in tested modules; the paid runners are thin scripts over them.
Two non-contingent observability changes land in `src/`; every other production
change waits on the numbers this plan produces.

**Tech Stack:** Python 3.12, `uv`, pytest, spaCy (`en_core_web_lg`), LiteLLM via
OpenRouter, the existing `evals/attribution/` harness.

**Spec:** `docs/superpowers/specs/2026-09-10-coreference-and-gender-design.md`

## Why this is one of two plans

The spec's section 4 ("What ships") is deliberately **not** in this plan. Its
content is contingent: clusters compose onto `identity.py` only if Tier 2
citation precision clears its gate, the prompt-layout change ships only if the
caching arm pays, and `include_aliases` goes unconditional only if its arm wins.
Writing those tasks now would mean inventing specifics for methods that may not
survive measurement — exactly the placeholder failure this skill forbids.

Two production changes are **not** contingent and are included here: recording
cached-token counts (Task 5, without which the cost axis cannot be measured) and
reporting ungendered spoken volume (Task 10, correct regardless of which arm
wins, and the thing that made Nefud invisible).

Plan 2 gets written after Task 11 reports.

## Global Constraints

- **Sequencing:** Tasks 5-11 start only after grid-folds lands. Tasks 1-4 are
  startable immediately; they read chapter text and touch no pipeline internals.
- **Baseline capture:** arm 0 must be captured on post-grid-folds `main`.
  Grid-folds ships `grid-v1` and deliberately changes segment identities, so any
  baseline taken before it is not comparable afterwards.
- **Coverage gate:** `--cov=kenkui --cov-fail-under=90` applies to `src/` only
  (`pyproject.toml:157`). `testpaths = ["tests"]`, so eval tests are not
  collected by default and are run explicitly with `--no-cov`.
- **Eval invocation:** every eval command runs from
  `kenkui/evals/attribution/` with `PY=../../.venv/bin/python`, per `RUNBOOK.md`.
- **Book scoping:** `KENKUI_EVAL_BOOK=dune` scopes `work/` and `runs/`
  (`common.py:22`). Never run a Dune step without it.
- **Attribution held fixed:** one stored attribution run underlies every
  coreference and gender arm. An arm may not re-attribute.
- **Gender vocabulary:** `"masculine"`, `"feminine"`, or `None` in code;
  `"unstated"` in gold JSON for a gender the text genuinely never gives.
- **`evals/` is git-ignored and entirely untracked** (`.gitignore:39`, 0 files
  in `git ls-files evals`). `pyproject.toml:98` also excludes it from ruff, and
  it is described there as "local evaluation scratch, not part of the package".
  So a plain `git add evals/...` is a silent no-op. This plan does **not**
  overturn that decision. Only these eval files are force-added, because they
  are durable artefacts rather than scratch — pure modules that every decision
  rests on, and hand-written truth that is expensive to recreate:
  `clusters.py`, `test_clusters.py`, `citations.py`, `test_citations.py`,
  `work/<book>/seed-clusters.json`, `work/<book>/adjudicated-clusters.json`.
  Runners, README and RUNBOOK stay untracked, matching `run.py`, `gold.py` and
  `score.py` which already are.
- **Ruff and mypy do not run over `evals/`.** The eval modules still follow
  house style, but the project gate in Task 10 covers `src/` and `tests/` only.
- **No pushing.** Commit locally on a feature branch; do not push.

---

## File Structure

| File | Responsibility |
|---|---|
| `evals/attribution/clusters.py` | Pure cluster algebra and metrics: B-cubed, splits, merges. No I/O. |
| `evals/attribution/citations.py` | Pure citation verification: tier assignment for a proposed merge. No I/O. |
| `evals/attribution/test_clusters.py` | Unit tests for the metrics. |
| `evals/attribution/test_citations.py` | Unit tests for tier assignment. |
| `evals/attribution/gold_clusters.py` | Builds `work/gold-clusters.json` from strong-model consensus plus a hand seed. |
| `evals/attribution/identity_run.py` | Runs one coreference arm, writes `runs/<arm>/clusters.json`. |
| `evals/attribution/score_identity.py` | Scores arms against gold, prints the table. |
| `evals/attribution/gender_run.py` | Runs one gender arm, writes `runs/<arm>/genders.json`. |
| `evals/attribution/work/dune/seed-clusters.json` | Hand-written known-broken cases. Checked in. |
| `src/kenkui/_domain/casting.py` | Gains `ungendered_volume`. |
| `src/kenkui/_resolution.py` | Gains the matching log event. |
| `evals/attribution/client.py` | Gains cached-token recording. |

---

## Task 1: Prepare Dune as an eval book

**Files:**
- Modify: `evals/attribution/RUNBOOK.md` (append a Dune section)
- Produces: `evals/attribution/work/dune/{book.epub,chapters.json,quotes.json}`

**Interfaces:**
- Consumes: nothing.
- Produces: `work/dune/chapters.json` as `[{"id": str, "text": str, "characters": int}]`
  and `work/dune/quotes.json` as `[{"chapter_id", "index", "start", "end", "text"}]`,
  loadable via `common.load_quotes()`.

Dune is the book that exhibits both defects and is not currently in the corpus.
Coreference needs the *whole book* — aliases like `Usul` appear hundreds of pages
from `Paul`, so the 6-chapter default would hide the defect being measured.

- [ ] **Step 1: Prepare the whole book**

```bash
cd /Users/dizzler/Projects/Repos/kenkui-v2/kenkui/evals/attribution
PY=../../.venv/bin/python
export KENKUI_EVAL_BOOK=dune
$PY prepare_book.py \
  "/Users/dizzler/Projects/Calibre Library/Frank Herbert/Dune (466)/Dune - Frank Herbert.epub" \
  --chapters 0
```

`--chapters 0` keeps every chapter meeting `--min-quotes`. `strip_doctype` runs
automatically; Dune is EPUB2 and will not parse without it
(`_epub/parser.py:132`, `forbid_dtd=True`).

- [ ] **Step 2: Verify the output is sane**

```bash
$PY -c "
from common import read_json, WORK, load_quotes
ch = read_json(WORK / 'chapters.json')
q = load_quotes()
print(f'{len(ch)} chapters, {len(q)} quotes, {sum(c[\"characters\"] for c in ch):,} chars')
assert len(ch) > 20, 'too few chapters -- did --chapters 0 apply?'
assert len(q) > 2000, 'too few quotes for a novel this size'
"
```

Expected: on the order of 30-50 chapters and several thousand quotes. If chapter
count is under 20, `prepare_book.py` filtered too hard — lower `--min-quotes`.

- [ ] **Step 3: Confirm the target names survived extraction**

```bash
$PY -c "
from common import read_json, WORK
text = ' '.join(c['text'] for c in read_json(WORK / 'chapters.json'))
for name in ('Paul', 'Usul', \"Muad'Dib\", 'Nefud', 'Feyd', 'Chani', 'Jessica'):
    print(f'{name:10} {text.count(name):5}')
"
```

Expected: every name non-zero. A zero for `Usul` or `Muad'Dib` means the selected
chapters exclude the back half of the book, and the coreference defect will not
be reproducible — fix that before going further.

- [ ] **Step 4: Append a Dune section to RUNBOOK.md**

Match the existing style: the command, then a `Result on 2026-09-10:` line with
the real numbers from Step 2.

- [ ] **Step 5: Commit**

No commit. `RUNBOOK.md` and everything under `work/` are untracked by design
(see Global Constraints). Record the numbers in `RUNBOOK.md` anyway — it is the
run log the next person reads, tracked or not.

---

## Task 2: Cluster metrics

**Files:**
- Create: `evals/attribution/clusters.py`
- Test: `evals/attribution/test_clusters.py`

**Interfaces:**
- Consumes: nothing.
- Produces:
  - `Partition = Mapping[str, str]` — surface name to cluster id.
  - `bcubed(gold: Partition, system: Partition, volume: Mapping[str, int]) -> Scores`
  - `Scores` — a frozen dataclass with `precision: float`, `recall: float`,
    `f1: float`.
  - `splits(gold: Partition, system: Partition, volume) -> tuple[Split, ...]`
    where `Split` has `cluster: str`, `pieces: int`, `volume: int`.
  - `merges(gold: Partition, system: Partition, volume) -> tuple[Merge, ...]`
    where `Merge` has `cluster: str`, `folded: tuple[str, ...]`, `volume: int`.

This is the module every decision in the plan rests on, so it is pure and tested
first. B-cubed recall maps to splits and B-cubed precision maps to merges — that
correspondence is what makes a two-number report interpretable, and the tests
below pin it.

- [ ] **Step 1: Write the failing tests**

Create `evals/attribution/test_clusters.py`:

```python
"""Unit tests for the cluster metrics. Run with --no-cov; not in testpaths."""

from __future__ import annotations

from clusters import bcubed, merges, splits

# Paul's four surface forms, with the volume that makes a split expensive.
VOLUME = {
    "Paul": 40000,
    "Paul Atreides": 5000,
    "Usul": 12000,
    "Muad'Dib": 9000,
    "Jessica": 30000,
    "Nefud": 800,
}
GOLD = {
    "Paul": "paul",
    "Paul Atreides": "paul",
    "Usul": "paul",
    "Muad'Dib": "paul",
    "Jessica": "jessica",
    "Nefud": "nefud",
}


def test_a_perfect_partition_scores_one() -> None:
    scores = bcubed(GOLD, dict(GOLD), VOLUME)
    assert scores.precision == 1.0
    assert scores.recall == 1.0
    assert scores.f1 == 1.0


def test_a_split_costs_recall_not_precision() -> None:
    """One person in four clusters: every cluster is pure, none is complete."""
    system = {
        "Paul": "a",
        "Paul Atreides": "a",
        "Usul": "b",
        "Muad'Dib": "c",
        "Jessica": "jessica",
        "Nefud": "nefud",
    }
    scores = bcubed(GOLD, system, VOLUME)
    assert scores.precision == 1.0
    assert scores.recall < 1.0


def test_a_merge_costs_precision_not_recall() -> None:
    """Two people in one cluster: complete, not pure."""
    system = {
        "Paul": "x",
        "Paul Atreides": "x",
        "Usul": "x",
        "Muad'Dib": "x",
        "Jessica": "x",
        "Nefud": "nefud",
    }
    scores = bcubed(GOLD, system, VOLUME)
    assert scores.recall == 1.0
    assert scores.precision < 1.0


def test_volume_weighting_punishes_the_expensive_split() -> None:
    """Splitting Paul must score worse than splitting Nefud off a pair."""
    small = {
        "Paul": "a",
        "Paul Atreides": "a",
        "Usul": "a",
        "Muad'Dib": "a",
        "Jessica": "jessica",
        "Nefud": "nefud",
    }
    big = {
        "Paul": "a",
        "Paul Atreides": "a",
        "Usul": "b",
        "Muad'Dib": "b",
        "Jessica": "jessica",
        "Nefud": "nefud",
    }
    flat = dict.fromkeys(VOLUME, 1)
    assert bcubed(GOLD, big, VOLUME).recall < bcubed(GOLD, small, VOLUME).recall
    # And the weighting is doing real work: unweighted differs from weighted.
    assert bcubed(GOLD, big, VOLUME).recall != bcubed(GOLD, big, flat).recall


def test_splits_names_the_broken_cluster_and_its_volume() -> None:
    system = {
        "Paul": "a",
        "Paul Atreides": "a",
        "Usul": "b",
        "Muad'Dib": "c",
        "Jessica": "jessica",
        "Nefud": "nefud",
    }
    found = splits(GOLD, system, VOLUME)
    assert len(found) == 1
    assert found[0].cluster == "paul"
    assert found[0].pieces == 3
    assert found[0].volume == 66000


def test_merges_names_the_conflated_clusters() -> None:
    system = dict(GOLD) | {"Jessica": "paul"}
    found = merges(GOLD, system, VOLUME)
    assert len(found) == 1
    assert found[0].folded == ("jessica", "paul")


def test_names_absent_from_gold_are_ignored() -> None:
    """Gold covers only multi-form speaking characters; extras must not score."""
    system = dict(GOLD) | {"Thufir Hawat": "thufir"}
    assert bcubed(GOLD, system, VOLUME | {"Thufir Hawat": 5000}).f1 == 1.0
```

- [ ] **Step 2: Run the tests to verify they fail**

```bash
cd /Users/dizzler/Projects/Repos/kenkui-v2/kenkui/evals/attribution
../../.venv/bin/python -m pytest test_clusters.py -v --no-cov -p no:cacheprovider
```

Expected: FAIL — `ModuleNotFoundError: No module named 'clusters'`.

- [ ] **Step 3: Implement `clusters.py`**

```python
"""Cluster algebra for scoring who-is-who against gold.

Scores a *partition of surface names*, not a ballot of ids. Two models that
group the same names score identically whatever they call the group, which is
what lets arms with different id vocabularies be compared at all -- the problem
that forced a single spaCy roster on the attribution eval.

Weighted by spoken volume throughout. Standard coreference metrics treat every
mention alike; a split costing Paul 40% of his lines and a split on a two-line
servant are not the same defect, and only the weighting says so.

B-cubed precision falls when two people share a cluster (a merge) and recall
falls when one person spans several (a split). Reporting both, plus the named
splits and merges below, is the whole output.
"""

from __future__ import annotations

from collections import defaultdict
# Mapping is imported at runtime, not under TYPE_CHECKING: the Partition alias
# below is evaluated at import time whatever `from __future__ import
# annotations` does to signatures.
from collections.abc import Mapping
from dataclasses import dataclass

# Surface name -> cluster id. Cluster ids are opaque and never compared across
# partitions; only the grouping they induce is.
Partition = Mapping[str, str]


@dataclass(frozen=True, slots=True)
class Scores:
    """B-cubed over surface names, weighted by spoken volume."""

    precision: float
    recall: float
    f1: float


@dataclass(frozen=True, slots=True)
class Split:
    """One gold cluster the system broke into several."""

    cluster: str
    pieces: int
    volume: int


@dataclass(frozen=True, slots=True)
class Merge:
    """One system cluster holding names from several gold clusters."""

    cluster: str
    folded: tuple[str, ...]
    volume: int


def _members(partition: Partition, names: set[str]) -> dict[str, set[str]]:
    """Invert a partition to cluster id -> names, restricted to `names`."""
    grouped: dict[str, set[str]] = defaultdict(set)
    for name, cluster in partition.items():
        if name in names:
            grouped[cluster].add(name)
    return grouped


def _weight(names: set[str], volume: Mapping[str, int]) -> int:
    return sum(volume.get(name, 0) for name in names)


def bcubed(gold: Partition, system: Partition, volume: Mapping[str, int]) -> Scores:
    """Return volume-weighted B-cubed precision, recall, and F1.

    Scored only over names gold knows about. Gold covers the multi-form speaking
    characters, so a system that also names Thufir Hawat is not penalised for
    knowing someone gold did not label -- that is a roster question, measured by
    the attribution eval, not an identity one.
    """
    scored = set(gold) & set(system)
    if not scored:
        return Scores(0.0, 0.0, 0.0)
    gold_members = _members(gold, scored)
    system_members = _members(system, scored)

    total = 0
    precision_acc = recall_acc = 0.0
    for name in scored:
        g = gold_members[gold[name]]
        s = system_members[system[name]]
        shared = _weight(g & s, volume)
        weight = volume.get(name, 0)
        # A name with no recorded volume still exists; give it weight 1 so a
        # partition is never scored over an empty denominator.
        weight = weight or 1
        gold_weight = _weight(g, volume) or len(g)
        system_weight = _weight(s, volume) or len(s)
        shared = shared or len(g & s)
        precision_acc += weight * (shared / system_weight)
        recall_acc += weight * (shared / gold_weight)
        total += weight
    precision = precision_acc / total
    recall = recall_acc / total
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall
        else 0.0
    )
    return Scores(precision, recall, f1)


def splits(
    gold: Partition, system: Partition, volume: Mapping[str, int]
) -> tuple[Split, ...]:
    """Return the gold clusters the system broke apart, worst volume first."""
    scored = set(gold) & set(system)
    found = []
    for cluster, names in _members(gold, scored).items():
        pieces = {system[name] for name in names}
        if len(pieces) > 1:
            found.append(Split(cluster, len(pieces), _weight(names, volume)))
    return tuple(sorted(found, key=lambda row: -row.volume))


def merges(
    gold: Partition, system: Partition, volume: Mapping[str, int]
) -> tuple[Merge, ...]:
    """Return the system clusters holding several people, worst volume first.

    This is the failure the attribution design is organised against, so it is
    reported by name rather than folded into an aggregate.
    """
    scored = set(gold) & set(system)
    found = []
    for cluster, names in _members(system, scored).items():
        folded = {gold[name] for name in names}
        if len(folded) > 1:
            found.append(
                Merge(cluster, tuple(sorted(folded)), _weight(names, volume))
            )
    return tuple(sorted(found, key=lambda row: -row.volume))
```

- [ ] **Step 4: Run the tests to verify they pass**

```bash
../../.venv/bin/python -m pytest test_clusters.py -v --no-cov -p no:cacheprovider
```

Expected: 7 passed.

- [ ] **Step 5: Commit**

```bash
git add -f evals/attribution/clusters.py evals/attribution/test_clusters.py
git commit -m "evals: volume-weighted cluster metrics for coreference scoring"
```

`-f` is required: `evals/` is git-ignored. See Global Constraints for why these
two files are the exception and the runners are not.

---

## Task 3: Citation verification

**Files:**
- Create: `evals/attribution/citations.py`
- Test: `evals/attribution/test_citations.py`

**Interfaces:**
- Consumes: nothing.
- Produces: `verify(proposal: Proposal, chapter_text: str, window: int = 200) -> Tier`
  - `Proposal` — frozen dataclass: `alias: str`, `canonical: str`, `evidence: str`.
  - `Tier` — `str` enum-like constants `TIER1`, `TIER2`, `TIER3`, `HALLUCINATED`.

The untried lever the whole coreference argument rests on, so it is pure and
tested before a single paid call is made. `window` is swept in Task 8, not
guessed here; 200 is the starting value.

- [ ] **Step 1: Write the failing tests**

Create `evals/attribution/test_citations.py`:

```python
"""Unit tests for citation tier assignment. Run with --no-cov."""

from __future__ import annotations

from citations import HALLUCINATED, TIER1, TIER2, TIER3, Proposal, verify

def test_both_names_in_one_verified_span_is_tier1() -> None:
    text = "They called him Paul-Muad'Dib and the name carried."
    proposal = Proposal("Muad'Dib", "Paul", "They called him Paul-Muad'Dib")
    assert verify(proposal, text) == TIER1


def test_a_naming_event_near_the_canonical_name_is_tier2() -> None:
    text = (
        "Paul stood among them in the dark. Stilgar raised his hand. "
        "You shall be known among us as Usul, the base of the pillar."
    )
    proposal = Proposal("Usul", "Paul", "You shall be known among us as Usul")
    assert verify(proposal, text) == TIER2


def test_a_greeting_is_not_a_naming_event() -> None:
    """This is the dai-shan/agelmar error: an honorific, not an introduction."""
    text = (
        "Agelmar bowed low. Beside him Lan said nothing at all, and the men "
        "greeted him: Peace favour your sword, Dai Shan."
    )
    proposal = Proposal("Dai Shan", "Agelmar", "Peace favour your sword, Dai Shan")
    assert verify(proposal, text) == TIER3


def test_a_third_party_between_the_names_does_not_block_tier2() -> None:
    """The speaker of a naming event is usually not the person being named."""
    text = (
        "Paul stood among them in the dark. Stilgar raised his hand. "
        "You shall be known among us as Usul, the base of the pillar."
    )
    proposal = Proposal("Usul", "Paul", "You shall be known among us as Usul")
    assert verify(proposal, text) == TIER2


def test_evidence_absent_from_the_text_is_hallucinated() -> None:
    text = "Paul stood among them in the dark."
    proposal = Proposal("Usul", "Paul", "Paul was called Usul by the Fremen")
    assert verify(proposal, text) == HALLUCINATED


def test_whitespace_differences_do_not_defeat_verification() -> None:
    text = "They called him\n   Paul-Muad'Dib   and the name\ncarried."
    proposal = Proposal("Muad'Dib", "Paul", "They called him Paul-Muad'Dib")
    assert verify(proposal, text) == TIER1


def test_evidence_without_the_alias_is_tier3() -> None:
    text = "Paul stood among them in the dark. Stilgar raised his hand."
    proposal = Proposal("Usul", "Paul", "Stilgar raised his hand")
    assert verify(proposal, text) == TIER3


def test_a_canonical_name_beyond_the_window_is_tier3() -> None:
    text = "Paul stood." + " and they waited." * 40 + " He is called Usul."
    proposal = Proposal("Usul", "Paul", "He is called Usul")
    assert verify(proposal, text, window=50) == TIER3
```

- [ ] **Step 2: Run the tests to verify they fail**

```bash
../../.venv/bin/python -m pytest test_citations.py -v --no-cov -p no:cacheprovider
```

Expected: FAIL — `ModuleNotFoundError: No module named 'citations'`.

- [ ] **Step 3: Implement `citations.py`**

```python
"""Grade the evidence a model offers for saying two names are one person.

The four routes rejected in `identity.py`'s "Under-merge stays" note all asked
a model to *assert* an identity. An assertion is unfalsifiable, so a wrong one
(dai-shan folded into agelmar) is indistinguishable from a right one. This asks
for a verbatim span from the book instead, which is checkable: the span either
occurs in the text or the model invented it, and the names either sit in the
right relation to it or they do not.

Three tiers, because the strongest check is not the one that matters. Tier 1
wants both names inside one verified span, which catches "Paul-Muad'Dib" and
misses the case this exists for: "You shall be known among us as Usul" contains
no "Paul" at all. Tier 2 is that case, and is therefore both the tier doing the
work and the tier most likely to leak. Its precision is measured separately.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence

TIER1 = "tier1"
TIER2 = "tier2"
TIER3 = "tier3"
HALLUCINATED = "hallucinated"

Tier = str

_SPACE = re.compile(r"\s+")


@dataclass(frozen=True, slots=True)
class Proposal:
    """One claimed identity, with the span the model says establishes it."""

    alias: str
    canonical: str
    evidence: str


def _flat(value: str) -> str:
    """Collapse whitespace so a span survives the text's own line breaks."""
    return _SPACE.sub(" ", value).strip()


def verify(
    proposal: Proposal,
    chapter_text: str,
    window: int = 200,
) -> Tier:
    """Return the tier the evidence earns.

    `window` is how far back the canonical name may sit from the evidence span
    and still be read as its antecedent. It is a swept parameter, not a
    constant: too small loses real naming events, too large lets any nearby
    name claim one.

    The discriminator for Tier 2 is the *construction*, not proximity. An
    earlier draft required the canonical name to be the nearest preceding
    named entity, which is wrong for precisely the case this exists to catch:
    in "Stilgar raised his hand. You shall be known among us as Usul", the
    nearest name is the speaker, not the person being named. Naming events
    routinely put a third party between the two names.

    So Tier 2 asks for a naming cue inside the span instead. "You shall be
    known among us as Usul" carries one; "Peace favour your sword, Dai Shan"
    -- a greeting, and the shape of the merge that folded Dai Shan into
    Agelmar -- does not.

    The known leak, and the reason Tier 2 precision is measured separately: a
    genuine naming event whose subject is someone other than the proposed
    canonical name within the window still passes. Only measurement settles
    whether that is rare.
    """
    text = _flat(chapter_text)
    evidence = _flat(proposal.evidence)
    alias = _flat(proposal.alias)
    canonical = _flat(proposal.canonical)
    if not evidence:
        return HALLUCINATED

    start = text.find(evidence)
    if start < 0:
        # The model produced a span the book does not contain. Counted, not
        # merely rejected: the rate is a model-quality signal in its own right.
        return HALLUCINATED

    if alias in evidence and canonical in evidence:
        return TIER1
    if alias not in evidence or not _CUE.search(evidence):
        return TIER3
    prefix = text[max(0, start - window) : start]
    return TIER2 if canonical in prefix else TIER3
```

- [ ] **Step 4: Add the naming-cue pattern**

Above `Proposal`, beside `_SPACE`:

```python
# Words that mark a span as a naming event rather than mere proximity. Kept
# loose on purpose: "known as" would miss "known among us as Usul", which is
# the exact sentence this tier exists for.
_CUE = re.compile(
    r"\b(?:known|call|calls|called|name|named|names|styled|dubbed|titled)\b",
    re.IGNORECASE,
)
```

- [ ] **Step 4: Run the tests to verify they pass**

```bash
../../.venv/bin/python -m pytest test_citations.py -v --no-cov -p no:cacheprovider
```

Expected: 8 passed.

- [ ] **Step 5: Commit**

```bash
git add -f evals/attribution/citations.py evals/attribution/test_citations.py
git commit -m "evals: tiered verification for cited alias merges"
```

---

## Task 4: The gold cluster set

**Files:**
- Create: `evals/attribution/work/dune/seed-clusters.json`
- Create: `evals/attribution/gold_clusters.py`
- Produces: `work/dune/gold-clusters.json`, `work/dune/cluster-disagreements.json`

**Interfaces:**
- Consumes: `common.read_json`, `common.write_json`, `common.WORK`, `common.load_env`.
- Produces: `work/dune/gold-clusters.json` as
  `{"book": str, "clusters": [{"id": str, "names": [str], "gender": "masculine"|"feminine"|"unstated", "source": "seed"|"consensus"|"adjudicated"}]}`.

Consensus across strong models, hand adjudication for the rest, nothing guessed
— the same discipline as `gold.py`. Two departures, both in the spec: the set is
seeded with the known-broken cases so the harness is proven to catch them, and
agreement is required over whole clusters rather than pairs, because pairwise
votes can produce an intransitive gold set.

**This is the long pole.** It involves hand adjudication and is the one
workstream not blocked by grid-folds. Start it first.

- [ ] **Step 1: Write the seed file**

The cases you already know are broken. Create
`evals/attribution/work/dune/seed-clusters.json`:

```json
{
  "book": "dune",
  "clusters": [
    {
      "id": "paul-atreides",
      "names": ["Paul", "Paul Atreides", "Usul", "Muad'Dib", "Paul-Muad'Dib"],
      "gender": "masculine",
      "source": "seed"
    },
    {
      "id": "iakin-nefud",
      "names": ["Nefud", "Iakin Nefud", "Captain Nefud"],
      "gender": "masculine",
      "source": "seed"
    }
  ]
}
```

Add any other case you already know is wrong from listening. Every name must
appear verbatim in the book — Step 2 checks that.

- [ ] **Step 2: Verify every seeded name occurs in the prepared text**

```bash
cd /Users/dizzler/Projects/Repos/kenkui-v2/kenkui/evals/attribution
export KENKUI_EVAL_BOOK=dune
../../.venv/bin/python -c "
from common import read_json, WORK
text = ' '.join(c['text'] for c in read_json(WORK / 'chapters.json'))
seed = read_json(WORK / 'seed-clusters.json')
bad = [n for c in seed['clusters'] for n in c['names'] if n not in text]
print('missing from text:', bad or 'none')
assert not bad, 'a seeded surface form does not occur in the book'
"
```

Expected: `missing from text: none`. A miss means the surface form is spelled
differently in this edition — fix the seed, not the check.

- [ ] **Step 3: Write `gold_clusters.py`**

```python
"""Build the gold cluster set: who is one person, and what gender.

Consensus across strong models, hand adjudication for the rest, nothing
guessed -- the discipline `gold.py` established for quote labels, applied to
identity.

Two departures from `gold.py`. The set is seeded from a hand-written file of
cases already known to be broken, so the harness is proven to catch the
defects that motivated it rather than only the ones models happen to agree on.
And agreement is required over a whole cluster rather than pair by pair: three
models can each agree to a pair while disagreeing about the group those pairs
imply, and a gold set assembled from pairwise votes can be intransitive.

The known weakness is `gold.py`'s: an error every adjudicator shares is
invisible, so this is an upper bound. Read `cluster-disagreements.json`, and
read a random slice of the agreed clusters too.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict

from common import WORK, load_env, read_json, write_json

PROMPT = """\
You are cataloguing the speaking characters of a novel.

Two questions, for the whole book:
1. Which surface names denote ONE person? A character may be named several
   ways -- a forename, a full name, a title, an epithet, a name given to them
   partway through the story.
2. What gender does the TEXT give each person? Not what a name suggests.

Return ONLY JSON:
{{"clusters": [{{"id": "...", "names": ["..."], "gender": "..."}}]}}

- "id": lowercase, hyphen-joined, from the fullest form of the name.
- "names": every surface form the book uses for that person, spelled exactly
  as the text spells them.
- "gender": "masculine", "feminine", or "unstated" when the text genuinely
  never says. Do not guess from a name.
- Include only characters who speak.
- One person per cluster. If you are not sure two names are the same person,
  put them in separate clusters -- a wrong grouping is worse than a missing one.

NAMES OBSERVED IN THE BOOK
{names}

PASSAGES
---
{passages}
---
"""


def _normalise(payload: object) -> dict[str, frozenset[str]]:
    """Model response to cluster id -> frozen set of surface names."""
    if not isinstance(payload, dict):
        return {}
    out: dict[str, frozenset[str]] = {}
    for item in payload.get("clusters", []):
        if not isinstance(item, dict):
            continue
        names = item.get("names")
        cluster_id = item.get("id")
        if not isinstance(names, list) or not isinstance(cluster_id, str):
            continue
        clean = frozenset(n.strip() for n in names if isinstance(n, str) and n.strip())
        if clean:
            out[cluster_id] = clean
    return out


def _genders(payload: object) -> dict[frozenset[str], str]:
    if not isinstance(payload, dict):
        return {}
    out: dict[frozenset[str], str] = {}
    for item in payload.get("clusters", []):
        if not isinstance(item, dict):
            continue
        names = item.get("names")
        gender = item.get("gender")
        if not isinstance(names, list) or gender not in (
            "masculine",
            "feminine",
            "unstated",
        ):
            continue
        clean = frozenset(n.strip() for n in names if isinstance(n, str) and n.strip())
        if clean:
            out[clean] = gender
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("models", nargs="+", help="strong models, as adjudicators")
    parser.add_argument(
        "--adjudicated",
        default=None,
        help="hand-resolved JSON in gold-clusters.json shape, merged in last",
    )
    args = parser.parse_args()
    load_env()

    from client import RecordingClient

    chapters = read_json(WORK / "chapters.json")
    seed = read_json(WORK / "seed-clusters.json")

    # Candidate surface names: everything the spaCy roster found, so the model
    # is grouping observed names rather than inventing them.
    roster = read_json(WORK / "roster.json")
    names = sorted({row["display_name"] for row in roster})
    passages = "\n\n".join(c["text"] for c in chapters)

    prompt = PROMPT.format(
        names=json.dumps(names, ensure_ascii=False, indent=2),
        passages=passages.replace("{", "{{").replace("}", "}}"),
    )

    ballots: list[dict[str, frozenset[str]]] = []
    gender_ballots: list[dict[frozenset[str], str]] = []
    client = RecordingClient(label="gold-clusters")
    for model in args.models:
        raw = client.complete(model, prompt)
        payload = json.loads(raw[raw.find("{") : raw.rfind("}") + 1])
        ballots.append(_normalise(payload))
        gender_ballots.append(_genders(payload))

    # Whole-cluster agreement: a set of names is gold only when every model
    # returned exactly that set.
    tally: dict[frozenset[str], int] = defaultdict(int)
    for ballot in ballots:
        for members in ballot.values():
            tally[members] += 1

    agreed = {m for m, votes in tally.items() if votes == len(args.models)}
    contested = {m for m, votes in tally.items() if 0 < votes < len(args.models)}

    clusters = list(seed["clusters"])
    claimed = {n for c in clusters for n in c["names"]}
    for members in sorted(agreed, key=lambda m: sorted(m)):
        if members & claimed:
            continue  # the seed wins; it is hand-written truth
        genders = {g.get(members) for g in gender_ballots} - {None}
        clusters.append(
            {
                "id": min(sorted(members), key=lambda n: -len(n)).lower().replace(" ", "-"),
                "names": sorted(members),
                "gender": genders.pop() if len(genders) == 1 else "unstated",
                "source": "consensus",
            }
        )
        claimed |= members

    if args.adjudicated:
        for cluster in read_json(args.adjudicated)["clusters"]:
            clusters = [c for c in clusters if not set(c["names"]) & set(cluster["names"])]
            clusters.append({**cluster, "source": "adjudicated"})

    write_json(WORK / "gold-clusters.json", {"book": seed["book"], "clusters": clusters})
    write_json(
        WORK / "cluster-disagreements.json",
        [sorted(m) for m in sorted(contested, key=lambda m: sorted(m))],
    )
    print(f"gold: {len(clusters)} clusters   contested: {len(contested)}")
    print(f"cost: ${client.total_cost:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Build the spaCy roster Dune needs as candidate names**

```bash
export KENKUI_EVAL_BOOK=dune
../../.venv/bin/python roster_spacy.py
```

Inspect `work/dune/roster.json` before spending anything. The eval README is
emphatic on this: a missing character caps every arm at once and reads as model
caution. Confirm Paul, Jessica, Stilgar, Chani, Feyd, Nefud are present.

- [ ] **Step 5: Run the consensus pass**

```bash
export OPENROUTER_API_KEY=...
../../.venv/bin/python gold_clusters.py \
  openrouter/anthropic/claude-opus-5 \
  openrouter/google/gemini-2.5-pro \
  openrouter/openai/gpt-5
```

Dune is ~300k characters; each call is a whole-book pass and these are expensive
models. Expect single-digit dollars total. If a model refuses the length, drop it
and use two adjudicators, recording that in RUNBOOK.md.

- [ ] **Step 6: Adjudicate the disagreements by hand**

Read `work/dune/cluster-disagreements.json`. Write the ones you can resolve into
`work/dune/adjudicated-clusters.json` in `gold-clusters.json` shape, then re-run
Step 5 with `--adjudicated work/dune/adjudicated-clusters.json`.

Also spot-check a random ten of the agreed clusters. Consensus gold's blind spot
is the error every adjudicator shares, and it is only found by reading.

- [ ] **Step 7: Verify gold contains the motivating defects**

```bash
../../.venv/bin/python -c "
from common import read_json, WORK
gold = read_json(WORK / 'gold-clusters.json')
by_name = {n: c for c in gold['clusters'] for n in c['names']}
paul = by_name['Paul']
assert by_name[\"Muad'Dib\"] is paul and by_name['Usul'] is paul, 'Paul is not one cluster'
assert by_name['Nefud']['gender'] == 'masculine'
print(f\"{len(gold['clusters'])} clusters; Paul has {len(paul['names'])} names\")
"
```

Expected: passes. This is the assertion that the measurement can see the bugs.

- [ ] **Step 8: Commit**

```bash
git add -f evals/attribution/work/dune/seed-clusters.json
git add -f evals/attribution/work/dune/adjudicated-clusters.json  # if it exists
git commit -m "evals: hand-written gold cluster seed for Dune"
```

`gold_clusters.py` is a runner and stays untracked, like `gold.py` beside it.
The seed and adjudication files are hand-written truth and are force-added.

---

## Task 5: Record cached tokens

**Files:**
- Modify: `evals/attribution/client.py:77-82` (the `Call` dataclass), `:173-183` (usage capture)

**Interfaces:**
- Consumes: nothing.
- Produces: `Call.cached_tokens: int | None`, present in `client.ledger()` output
  and therefore in every `run.json`.

Prerequisite for the whole cost axis. `client.py` records `prompt_tokens` but not
`prompt_tokens_details.cached_tokens`, so a cache hit is invisible and can only be
inferred from cost — which conflates it with a price change.

**Blocked on grid-folds.**

- [ ] **Step 1: Add the field to `Call`**

In the `Call` dataclass beside `prompt_tokens`:

```python
    # OpenRouter passes through the provider's cached-prompt count. Absent on
    # providers that do not report it, which is different from zero: None means
    # "not told", 0 means "told, and nothing was cached".
    cached_tokens: int | None = None
```

- [ ] **Step 2: Capture it from the response**

In `complete`, replace the success-path `self.calls.append(Call(...))` block's
`prompt_tokens=` line region so the call reads:

```python
        usage = getattr(response, "usage", None)
        details = getattr(usage, "prompt_tokens_details", None)
        self.calls.append(
            Call(
                model=model,
                label=self.label,
                prompt_chars=len(prompt),
                prompt_tokens=getattr(usage, "prompt_tokens", None),
                completion_tokens=getattr(usage, "completion_tokens", None),
                cached_tokens=getattr(details, "cached_tokens", None),
                cost_usd=cost,
                cost_source=source,
                latency_s=latency,
            )
        )
```

Leave the failure path's `Call(...)` alone; `cached_tokens` defaults to `None`.

- [ ] **Step 3: Verify against a real call**

```bash
cd /Users/dizzler/Projects/Repos/kenkui-v2/kenkui/evals/attribution
export OPENROUTER_API_KEY=...
../../.venv/bin/python -c "
from common import load_env; load_env()
from client import RecordingClient
c = RecordingClient(label='probe')
prompt = 'Say OK. ' + 'Context padding sentence. ' * 400
c.complete('openrouter/openai/gpt-4.1-mini', prompt)
c.complete('openrouter/openai/gpt-4.1-mini', prompt)
for call in c.calls:
    print(call.prompt_tokens, call.cached_tokens, call.cost_usd)
"
```

Expected: two rows. The second may show a non-zero `cached_tokens` on a provider
with automatic caching. `None` on both rows means this provider does not report
it — record that in RUNBOOK.md rather than treating it as a bug.

- [ ] **Step 4: Commit**

No commit; `client.py` is untracked. Note the change in `RUNBOOK.md` so the
next person knows `cached_tokens` exists in the ledger.

---

## Task 6: The free baseline diagnostic

**Files:**
- Create: `evals/attribution/diagnose_gender.py`
- Produces: `work/dune/gender-diagnosis.json`

**Interfaces:**
- Consumes: `kenkui._characters.spacy_roster.infer_roster`, `pipeline_for`.
- Produces: `work/dune/gender-diagnosis.json` as
  `[{"id", "display_name", "gender", "chapters", "aliases"}]`.

Costs nothing and may resolve much of the gender defect on its own. The spec's
preliminary table was an approximation with a regex tokenizer; this is the real
pipeline. Three questions: did Nefud, Feyd and Chani reach the roster and under
which surface forms; is the gender vote keyed on the folded entity or the raw
surface name; and does relaxing the proper-noun stop or the 2:1 margin resolve
them without breaking the two-hander case those constants were tuned for
(`tests/test_spacy_roster.py::test_gender_resolves_in_a_dense_two_hander`).

**Blocked on grid-folds.**

- [ ] **Step 1: Write the diagnostic**

```python
"""Run the real spaCy roster over Dune and report what it decided about gender.

The design document's gender table was an approximation -- a regex tokenizer
with capitalisation standing in for proper-noun detection. This is the shipped
pipeline. Nothing here is paid, so it runs before anything that is.
"""

from __future__ import annotations

import argparse

from common import WORK, read_json, write_json


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="en_core_web_lg")
    args = parser.parse_args()

    from kenkui._characters import spacy_roster
    from kenkui.inspection import ChapterInspection

    chapters = tuple(
        ChapterInspection(id=row["id"], text=row["text"], characters=row["characters"])
        for row in read_json(WORK / "chapters.json")
    )
    pipeline = spacy_roster.pipeline_for(args.model)
    characters, narrator = spacy_roster.infer_roster(chapters, pipeline)

    rows = [
        {
            "id": c.id,
            "display_name": c.display_name,
            "gender": c.gender,
            "chapters": len(c.chapter_ids),
            "aliases": list(c.aliases),
        }
        for c in characters
    ]
    write_json(WORK / "gender-diagnosis.json", rows)

    ungendered = [r for r in rows if r["gender"] is None]
    print(f"{len(rows)} characters, {len(ungendered)} with no gender, narrator={narrator}")
    print(f"\n{'id':32} {'gender':10} chapters")
    for row in sorted(rows, key=lambda r: -r["chapters"])[:40]:
        print(f"{row['id'][:32]:32} {str(row['gender']):10} {row['chapters']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

The exact `ChapterInspection` constructor signature must match
`src/kenkui/inspection.py` after grid-folds. Read it before running; if the
fields differ, use the real ones rather than adapting this snippet blindly.

- [ ] **Step 2: Run it**

```bash
export KENKUI_EVAL_BOOK=dune
../../.venv/bin/python diagnose_gender.py
```

Expected: a table. Note whether Nefud, Feyd and Chani appear, and their gender.

- [ ] **Step 3: Answer the three questions and record them**

```bash
../../.venv/bin/python -c "
from common import read_json, WORK
rows = read_json(WORK / 'gender-diagnosis.json')
for target in ('nefud', 'feyd', 'chani', 'paul', 'jessica'):
    hits = [r for r in rows if target in r['id']]
    print(target, '->', [(h['id'], h['gender'], h['aliases']) for h in hits] or 'ABSENT')
"
```

Record in RUNBOOK.md: which are absent, which are ungendered, and whether any
appears under more than one id (which would confirm the vote-splitting
hypothesis for Nefud).

- [ ] **Step 4: Sweep the two constants, without breaking the tuned case**

`_GENDER_LOOKAHEAD`, `_GENDER_MINIMUM` and `_GENDER_MARGIN` are in
`src/kenkui/_characters/spacy_roster.py`. For each candidate value, edit, re-run
Step 2, then run the guard test:

```bash
cd /Users/dizzler/Projects/Repos/kenkui-v2/kenkui
uv run pytest tests/test_spacy_roster.py -q --no-cov
```

Expected: PASS for any value you would keep. A value that resolves Feyd and
breaks `test_gender_resolves_in_a_dense_two_hander` is not an improvement — it is
the 2026-09-05 defect coming back. Revert the source edits when the sweep ends;
this task produces a recorded finding, not a change.

- [ ] **Step 5: Commit the diagnostic and the finding**

No commit; both are untracked. **Record the finding in `RUNBOOK.md`** — it is
the input to Task 9's arm selection and to Plan 2.

---

## Task 7: The coreference arm runner and the two baselines

**Files:**
- Create: `evals/attribution/identity_run.py`
- Create: `evals/attribution/score_identity.py`
- Produces: `runs/dune/<arm>/clusters.json`

**Interfaces:**
- Consumes: `clusters.Partition`, `clusters.bcubed`, `clusters.splits`, `clusters.merges`.
- Produces: `runs/dune/<arm>/clusters.json` as
  `{"arm": str, "model": str|null, "cost_usd": float, "partition": {surface_name: cluster_id}, "volume": {surface_name: int}, "calls": [...]}`.
- `score_identity.py` reads every `runs/dune/*/clusters.json` plus
  `work/dune/gold-clusters.json` and prints one row per arm.

Arm 0 is production (`_model_roster` through `merge_rosters`), arm 0b is
`spacy_roster.infer_roster`. Without them there is no improvement to claim.

**Blocked on grid-folds. Arm 0 must be captured on post-grid-folds `main`.**

- [ ] **Step 1: Write `identity_run.py` with the two baseline arms**

The runner dispatches on `--arm`. Baselines produce a partition by mapping every
surface form a character carries — `display_name` plus `aliases` — to that
character's id.

```python
"""Run one coreference arm and write the partition it produces.

An arm answers one question: given this book, which surface names denote one
person? It writes a partition -- name to opaque cluster id -- and never a
ballot of ids, because two arms with different id vocabularies must still be
comparable. `score_identity.py` grades the grouping, not the labels.

Volume travels with the partition. A split costing Paul 40% of his lines and a
split on a two-line servant are not the same defect, and the weight is what
says so.
"""

from __future__ import annotations

import argparse

from common import RUNS, WORK, load_env, read_json, write_json


def _partition_from_characters(characters) -> tuple[dict[str, str], dict[str, int]]:
    """Every surface form a character carries maps to that character's id."""
    partition: dict[str, str] = {}
    volume: dict[str, int] = {}
    for character in characters:
        for name in (character.display_name, *character.aliases):
            partition[name] = character.id
            volume[name] = character.spoken_characters
    return partition, volume


def _arm_spacy(_args) -> tuple[dict[str, str], dict[str, int], float, list]:
    from kenkui._characters import spacy_roster
    from kenkui.inspection import ChapterInspection

    chapters = tuple(
        ChapterInspection(id=r["id"], text=r["text"], characters=r["characters"])
        for r in read_json(WORK / "chapters.json")
    )
    characters, _ = spacy_roster.infer_roster(
        chapters, spacy_roster.pipeline_for("en_core_web_lg")
    )
    partition, volume = _partition_from_characters(characters)
    return partition, volume, 0.0, []


def _arm_production(args) -> tuple[dict[str, str], dict[str, int], float, list]:
    """Per-chapter LLM rosters folded by merge_rosters -- what ships today."""
    from client import RecordingClient
    from kenkui._characters import _roster_for
    from kenkui._characters.infer import merge_rosters

    client = RecordingClient(label=f"identity-{args.arm}", repair=True)
    rosters = []
    for row in read_json(WORK / "chapters.json"):
        roster, _ = _roster_for(row["text"], args.model, client, first_person=False)
        rosters.append(roster)
    characters = merge_rosters(tuple(rosters))
    partition, volume = _partition_from_characters(characters)
    return partition, volume, client.total_cost, client.ledger()


ARMS = {"spacy": _arm_spacy, "production": _arm_production}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("arm", choices=sorted(ARMS))
    parser.add_argument("--tag", required=True, help="run directory name")
    parser.add_argument("--model", default="openrouter/deepseek/deepseek-v3.2")
    args = parser.parse_args()
    load_env()

    partition, volume, cost, calls = ARMS[args.arm](args)
    write_json(
        RUNS / args.tag / "clusters.json",
        {
            "arm": args.arm,
            "model": args.model if args.arm != "spacy" else None,
            "cost_usd": cost,
            "partition": partition,
            "volume": volume,
            "calls": calls,
        },
    )
    groups = len(set(partition.values()))
    print(f"{args.tag}: {len(partition)} names in {groups} clusters, ${cost:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

`_roster_for` is private; import it exactly as `roster_spacy.py` and `run.py`
already reach into `kenkui._characters`. If grid-folds changed its signature,
read the current one and adapt — do not guess.

- [ ] **Step 2: Write `score_identity.py`**

```python
"""Score every coreference arm against the gold cluster set.

Two numbers and two lists. B-cubed precision falls when two people share a
cluster; recall falls when one person spans several. The named splits and
merges below say which characters, and what they cost in spoken volume.

Reported weighted and unweighted. Weighted predicts what a listener hears;
unweighted is what the coreference literature reports, and the pair together
show whether a result depends on the weighting.
"""

from __future__ import annotations

import argparse

from clusters import bcubed, merges, splits
from common import RUNS, WORK, read_json, write_json


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("tags", nargs="+")
    args = parser.parse_args()

    gold_file = read_json(WORK / "gold-clusters.json")
    gold = {
        name: cluster["id"]
        for cluster in gold_file["clusters"]
        for name in cluster["names"]
    }

    rows = []
    print(
        f"{'arm':22} {'P':>6} {'R':>6} {'F1':>6} "
        f"{'P/raw':>6} {'R/raw':>6} {'splits':>7} {'merges':>7} {'$':>8}"
    )
    print("-" * 88)
    for tag in args.tags:
        run = read_json(RUNS / tag / "clusters.json")
        partition, volume = run["partition"], run["volume"]
        flat = dict.fromkeys(volume, 1)
        weighted = bcubed(gold, partition, volume)
        raw = bcubed(gold, partition, flat)
        broken = splits(gold, partition, volume)
        conflated = merges(gold, partition, volume)
        rows.append(
            {
                "tag": tag,
                "precision": weighted.precision,
                "recall": weighted.recall,
                "f1": weighted.f1,
                "precision_raw": raw.precision,
                "recall_raw": raw.recall,
                "splits": [s.__dict__ for s in broken],
                "merges": [m.__dict__ for m in conflated],
                "cost_usd": run["cost_usd"],
            }
        )
        print(
            f"{tag[:22]:22} {weighted.precision:6.3f} {weighted.recall:6.3f} "
            f"{weighted.f1:6.3f} {raw.precision:6.3f} {raw.recall:6.3f} "
            f"{len(broken):7} {len(conflated):7} {run['cost_usd']:8.4f}"
        )

    for row in rows:
        if row["splits"]:
            print(f"\n{row['tag']} splits (worst volume first):")
            for split in row["splits"][:8]:
                print(f"  {split['cluster']:30} {split['pieces']} pieces, {split['volume']:,} chars")
        if row["merges"]:
            print(f"\n{row['tag']} MERGES -- these fail the gate:")
            for merge in row["merges"][:8]:
                print(f"  {merge['cluster']:30} {' + '.join(merge['folded'])}")

    write_json(WORK / "identity-scores.json", rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 3: Run both baselines**

```bash
export KENKUI_EVAL_BOOK=dune
export OPENROUTER_API_KEY=...
../../.venv/bin/python identity_run.py spacy --tag base-spacy
../../.venv/bin/python identity_run.py production --tag base-production \
  --model openrouter/deepseek/deepseek-v3.2
../../.venv/bin/python score_identity.py base-spacy base-production
```

- [ ] **Step 4: Confirm the harness reproduces the known defect**

```bash
../../.venv/bin/python -c "
from common import read_json, WORK
rows = read_json(WORK / 'identity-scores.json')
for row in rows:
    paul = [s for s in row['splits'] if s['cluster'] == 'paul-atreides']
    print(row['tag'], '-> Paul split into', paul[0]['pieces'] if paul else 1, 'pieces')
"
```

Expected: at least one baseline shows Paul in more than one piece. **If both
show 1, stop.** Either gold or the runner is wrong, and every later number would
be meaningless. Do not proceed to paid arms until this reproduces.

- [ ] **Step 5: Record the baselines in RUNBOOK.md and commit**

No commit; runners are untracked. Record the baseline numbers in `RUNBOOK.md`.

---

## Task 8: The cited-alias arm and its control

**Files:**
- Modify: `evals/attribution/identity_run.py` (add two arms)
- Produces: `runs/dune/cited-*/clusters.json`, `runs/dune/uncited/clusters.json`

**Interfaces:**
- Consumes: `citations.Proposal`, `citations.verify`, `citations.TIER1/TIER2/TIER3/HALLUCINATED`.
- Produces: `clusters.json` gains `"proposals": [{"alias", "canonical", "evidence", "tier", "accepted": bool}]`.

The recommendation, and the control that tells you whether the citation gate is
doing the work or is ceremony. **Run both.** Arm `uncited` is the route already
rejected in `identity.py`'s "Under-merge stays" note; if `cited` does not beat
it, the gate should not be built.

**Blocked on grid-folds.**

- [ ] **Step 1: Add the alias-proposal prompt to `identity_run.py`**

```python
ALIAS_PROMPT = """\
Some characters in a novel are named more than one way: a forename, a full
name, a title, an epithet, or a name given to them partway through the story.

From this passage, list every pair of names that denote ONE person.

Return ONLY JSON:
{{"aliases": [{{"alias": "...", "canonical": "...", "evidence": "..."}}]}}

- "alias": the alternative name, exactly as the passage spells it.
- "canonical": the character's fullest or commonest name in this passage.
- "evidence": a span copied VERBATIM from the passage that establishes the
  two names are one person. Copy it exactly, character for character. Do not
  paraphrase, summarise, or write a sentence of your own.

Return nothing for a pair you cannot support with a verbatim span. An omitted
pair costs little; an unsupported one puts two people in one voice.

Passage:
---
{passage}
---
"""

UNCITED_PROMPT = """\
Some characters in a novel are named more than one way: a forename, a full
name, a title, an epithet, or a name given to them partway through the story.

From this passage, list every pair of names that denote ONE person.

Return ONLY JSON:
{{"aliases": [{{"alias": "...", "canonical": "..."}}]}}

Passage:
---
{passage}
---
"""
```

- [ ] **Step 2: Add the two arms**

```python
def _propose(args, prompt_template, client) -> list[dict]:
    """Collect alias proposals chapter by chapter."""
    import json

    proposals = []
    for row in read_json(WORK / "chapters.json"):
        passage = row["text"].replace("{", "{{").replace("}", "}}")
        try:
            raw = client.complete(args.model, prompt_template.format(passage=passage))
        except Exception:  # noqa: BLE001 - a failed chapter loses its proposals only
            continue
        try:
            payload = json.loads(raw[raw.find("{") : raw.rfind("}") + 1])
        except ValueError:
            continue
        for item in payload.get("aliases", []):
            if isinstance(item, dict) and item.get("alias") and item.get("canonical"):
                proposals.append({**item, "chapter_id": row["id"]})
    return proposals


def _fold(base: dict[str, str], accepted: list[tuple[str, str]]) -> dict[str, str]:
    """Apply accepted merges to a baseline partition.

    Only ever adds a fold. The spec's rule: a cluster may not dissolve a merge
    `identity.py` already made, so the baseline is a strict lower bound and a
    bad reconciliation degrades to today's output rather than to something
    worse.
    """
    partition = dict(base)
    for alias, canonical in accepted:
        if alias not in partition or canonical not in partition:
            continue
        source, target = partition[alias], partition[canonical]
        if source == target:
            continue
        for name, cluster in list(partition.items()):
            if cluster == source:
                partition[name] = target
    return partition


def _arm_cited(args):
    from citations import HALLUCINATED, TIER1, TIER2, TIER3, Proposal, verify
    from client import RecordingClient

    client = RecordingClient(label=f"identity-{args.arm}", repair=True)
    base, volume, _, _ = _arm_spacy(args)
    chapters = {r["id"]: r["text"] for r in read_json(WORK / "chapters.json")}

    proposals = _propose(args, ALIAS_PROMPT, client)
    accepted, graded = [], []
    allowed = {TIER1, TIER2} if not args.accept_tier3 else {TIER1, TIER2, TIER3}
    for item in proposals:
        tier = verify(
            Proposal(item["alias"], item["canonical"], item.get("evidence", "")),
            chapters[item["chapter_id"]],
            window=args.window,
        )
        ok = tier in allowed
        graded.append({**item, "tier": tier, "accepted": ok})
        if ok:
            accepted.append((item["alias"], item["canonical"]))

    partition = _fold(base, accepted)
    hallucinated = sum(1 for g in graded if g["tier"] == HALLUCINATED)
    print(f"  proposals {len(graded)}, accepted {len(accepted)}, hallucinated {hallucinated}")
    return partition, volume, client.total_cost, client.ledger(), graded


def _arm_uncited(args):
    from client import RecordingClient

    client = RecordingClient(label=f"identity-{args.arm}", repair=True)
    base, volume, _, _ = _arm_spacy(args)
    proposals = _propose(args, UNCITED_PROMPT, client)
    accepted = [(p["alias"], p["canonical"]) for p in proposals]
    partition = _fold(base, accepted)
    graded = [{**p, "tier": "uncited", "accepted": True} for p in proposals]
    return partition, volume, client.total_cost, client.ledger(), graded
```

- [ ] **Step 2b: Rewire `ARMS` and `main()`**

Give every arm the same five-value shape so the call site needs no special
case, then write the proposals through — Step 5 reads them back.

```python
ARMS = {
    "spacy": lambda args: (*_arm_spacy(args), []),
    "production": lambda args: (*_arm_production(args), []),
    "cited": _arm_cited,
    "uncited": _arm_uncited,
}
```

In `main()`, add the two options and widen the unpacking and the payload:

```python
    parser.add_argument("--window", type=int, default=200)
    parser.add_argument("--accept-tier3", action="store_true")
    parser.add_argument(
        "--gate-disjoint",
        action="store_true",
        help="apply a merge only where the two never speak in one chapter",
    )
    ...
    partition, volume, cost, calls, graded = ARMS[args.arm](args)
    write_json(
        RUNS / args.tag / "clusters.json",
        {
            "arm": args.arm,
            "model": args.model if args.arm != "spacy" else None,
            "cost_usd": cost,
            "partition": partition,
            "volume": volume,
            "calls": calls,
            "proposals": graded,
        },
    )
```

- [ ] **Step 2c: Add the chapter-disjoint gate**

The spec's modifier, applied over any arm rather than being one. A merge whose
two names never speak in the same chapter is inaudible if wrong — collisions in
`_domain/casting.py` are chapter-scoped, so those two were already permitted to
share a voice. It cannot help Paul, who shares chapters with Muad'Dib
throughout; it is the safety net for merges that fail the citation tiers.

```python
def _chapters_of(name: str, chapters: dict[str, str]) -> set[str]:
    """Which chapters a surface name appears in at all.

    Appearance, not attributed speech: a name the extractor never tied to a
    quote can still be the one the merge is about, and the gate is meant to be
    conservative about *permitting* merges, not about finding them.
    """
    return {cid for cid, text in chapters.items() if name in text}


def _inaudible(alias: str, canonical: str, chapters: dict[str, str]) -> bool:
    return not (_chapters_of(alias, chapters) & _chapters_of(canonical, chapters))
```

In `_arm_cited`, when `args.gate_disjoint` is set, accept a Tier 3 proposal
that `_inaudible` allows, in addition to Tiers 1 and 2:

```python
        ok = tier in allowed or (
            args.gate_disjoint
            and tier == TIER3
            and _inaudible(item["alias"], item["canonical"], chapters)
        )
```

Score `--gate-disjoint` as a separate tag for each arm it modifies, so its
contribution is visible rather than baked in:

```bash
../../.venv/bin/python identity_run.py cited --tag cited-gated --gate-disjoint \
  --model openrouter/deepseek/deepseek-v3.2
../../.venv/bin/python identity_run.py uncited --tag uncited-gated --gate-disjoint \
  --model openrouter/deepseek/deepseek-v3.2
```

**Watch the merges column.** The gate exists to make wrong merges inaudible,
not to make them disappear — if `cited-gated` shows more merges than `cited`,
that is the gate admitting errors and it must be judged on whether those merges
are genuinely chapter-disjoint.

- [ ] **Step 3: Run the cited arm and its control**

```bash
export KENKUI_EVAL_BOOK=dune
../../.venv/bin/python identity_run.py cited --tag cited-w200 \
  --model openrouter/deepseek/deepseek-v3.2
../../.venv/bin/python identity_run.py uncited --tag uncited \
  --model openrouter/deepseek/deepseek-v3.2
../../.venv/bin/python score_identity.py base-spacy base-production cited-w200 uncited
```

- [ ] **Step 4: Sweep the window**

```bash
for w in 80 150 200 400 800; do
  ../../.venv/bin/python identity_run.py cited --tag cited-w$w --window $w \
    --model openrouter/deepseek/deepseek-v3.2
done
../../.venv/bin/python score_identity.py cited-w80 cited-w150 cited-w200 cited-w400 cited-w800
```

The window is the whole experiment for Tier 2, in the way the 30/45/60-character
window was for spaCy attribution. Too small loses real naming events; too large
lets any nearby name claim one. Record the curve in RUNBOOK.md.

- [ ] **Step 5: Measure per-tier precision — the number the approach lives on**

```bash
../../.venv/bin/python -c "
import collections
from common import RUNS, WORK, read_json
gold = {n: c['id'] for c in read_json(WORK / 'gold-clusters.json')['clusters'] for n in c['names']}
run = read_json(RUNS / 'cited-w200' / 'clusters.json')
by_tier = collections.defaultdict(lambda: [0, 0])
for p in run['proposals']:
    a, c = p['alias'], p['canonical']
    if a not in gold or c not in gold:
        continue
    row = by_tier[p['tier']]
    row[0] += 1
    row[1] += gold[a] == gold[c]
for tier, (n, right) in sorted(by_tier.items()):
    print(f'{tier:14} {right}/{n} correct  precision {right/n if n else 0:.3f}')
"
```

Expected shape: Tier 1 near-perfect and rare, Tier 2 decisive, Tier 3 poor.
**Tier 2 precision against the threshold in the acceptance gates is the decision
this whole plan exists to inform.** Record it prominently.

- [ ] **Step 6: Commit**

No commit. Record the window curve and per-tier precision in `RUNBOOK.md`.

---

## Task 9: Gender arms

**Files:**
- Create: `evals/attribution/gender_run.py`
- Modify: `evals/attribution/score_gender.py` (add gold-based scoring)
- Produces: `runs/dune/<arm>/genders.json`
- Depends on: Task 6's `work/dune/gender-diagnosis.json`, Task 7's fixed attribution run

**Interfaces:**
- Consumes: `work/dune/gold-clusters.json`, one fixed stored attribution run.
- Produces: `runs/dune/<arm>/genders.json` as
  `{"arm": str, "model": str|null, "cost_usd": float, "genders": {cluster_or_character_id: "masculine"|"feminine"|null}}`.
- `score_gender.py` gains `--gold`, scoring inference correctness per gold
  cluster and reporting ungendered spoken volume.

Ordered by the corrected hypothesis: named-character abstention first. The
escalation pass covers role speakers only — its prompt says "Omit it for
`roster`" — so it would never have fixed Nefud.

**Blocked on grid-folds. Uses the fixed attribution run from Task 7.**

- [ ] **Step 1: Extend `score_gender.py` to score against gold**

Add a `--gold` option. When given, read `work/<book>/gold-clusters.json`, map
each cast character to its gold cluster by surface name, and report four numbers
alongside the existing table:

```python
    correct = sum(1 for row in rows if inferred(row) == gold_gender(row) != "unstated")
    wrong = sum(
        1 for row in rows
        if inferred(row) and gold_gender(row) not in (None, "unstated")
        and inferred(row) != gold_gender(row)
    )
    abstained = sum(1 for row in rows if not inferred(row) and gold_gender(row) != "unstated")
    abstained_volume = sum(
        row["spoken_characters"] for row in rows
        if not inferred(row) and gold_gender(row) != "unstated"
    )
```

`abstained_volume` is the number that made Nefud invisible: how much speech is
being cast from the whole pool on a coin flip. Print it in the totals line.

- [ ] **Step 2: Write `gender_run.py`**

```python
"""Run one gender arm and write the gender it assigns each character.

Ordered by the corrected hypothesis. The escalation pass prototyped in
`escalate.py` covers `role:` speakers only -- its prompt says "Omit it for
roster" -- so it would never have fixed Captain Nefud, who is named and
rostered. Named-character abstention is the primary arm here; roles are the
secondary one.

Every arm reads the same stored attribution run. An arm may not re-attribute:
`dialogue_tags.apply` can turn a correct gender wrong when a quote goes to the
wrong mouth, so an arm that shifted attribution underneath itself would be
measuring two things at once.
"""

from __future__ import annotations

import argparse
import json

from common import RUNS, WORK, load_env, read_json, write_json

NAMED_PROMPT = """\
For each character below, decide the gender the TEXT gives them.

Return ONLY JSON:
{{"genders": [{{"id": "...", "gender": "..."}}]}}

- "gender": "masculine", "feminine", or null when these passages genuinely do
  not say. Do not guess from a name.
- Judge only from the passages given. Each passage is text surrounding a line
  that character speaks.

CHARACTERS AND THEIR PASSAGES
{blocks}
"""


def _windows(character_id, spans, chapters, radius, limit):
    """Text around up to `limit` of one character's own quotes."""
    found = []
    for span in spans:
        if span["character_id"] != character_id:
            continue
        text = chapters.get(span["chapter_id"], "")
        found.append(
            text[max(0, span["start"] - radius) : span["end"] + radius]
        )
        if len(found) >= limit:
            break
    return found


def _arm_baseline(args, _client):
    """Genders exactly as production composes them, with nothing added."""
    diagnosis = read_json(WORK / "gender-diagnosis.json")
    return {row["id"]: row["gender"] for row in diagnosis}, 0.0


def _arm_named(args, client):
    """Ask once about the rostered characters the pipeline abstained on.

    The arm aimed at Nefud, Feyd and Chani: named, high-volume, and left at
    None by the roster vote, which sends them to `candidates` and out to the
    whole pool on a coin flip.
    """
    diagnosis = read_json(WORK / "gender-diagnosis.json")
    attribution = read_json(RUNS / args.attribution / "run.json")
    chapters = {c["id"]: c["text"] for c in read_json(WORK / "chapters.json")}
    spans = attribution["spans"]

    genders = {row["id"]: row["gender"] for row in diagnosis}
    unknown = [
        row
        for row in diagnosis
        if row["gender"] is None
        and sum(1 for s in spans if s["character_id"] == row["id"]) >= args.min_quotes
    ]

    for batch in range(0, len(unknown), args.batch):
        rows = unknown[batch : batch + args.batch]
        blocks = "\n\n".join(
            f"### {row['id']}\n"
            + "\n---\n".join(
                _windows(row["id"], spans, chapters, args.radius, args.windows)
            )
            for row in rows
        )
        try:
            raw = client.complete(
                args.model,
                NAMED_PROMPT.format(blocks=blocks.replace("{", "{{").replace("}", "}}")),
            )
            payload = json.loads(raw[raw.find("{") : raw.rfind("}") + 1])
        except (ValueError, KeyError):
            continue
        for item in payload.get("genders", []):
            if isinstance(item, dict) and item.get("gender") in ("masculine", "feminine"):
                genders[item["id"]] = item["gender"]
    return genders, client.total_cost


def _arm_roles(args, client):
    """The escalate.py pass, ported. Covers `role:` speakers only."""
    genders, _ = _arm_baseline(args, client)
    escalation = read_json(RUNS / args.escalation / "escalation.json")["resolved"]
    for value in escalation.values():
        if value["kind"] == "role" and value.get("gender"):
            genders[f"role:{value['speaker']}"] = value["gender"]
    return genders, 0.0


ARMS = {"baseline": _arm_baseline, "named": _arm_named, "roles": _arm_roles}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("arm", choices=sorted(ARMS))
    parser.add_argument("--tag", required=True)
    parser.add_argument("--model", default="openrouter/deepseek/deepseek-v3.2")
    parser.add_argument("--attribution", default="base-production")
    parser.add_argument("--escalation", default="escalated-gender")
    parser.add_argument("--min-quotes", type=int, default=3)
    parser.add_argument("--windows", type=int, default=6)
    parser.add_argument("--radius", type=int, default=200)
    parser.add_argument("--batch", type=int, default=12)
    args = parser.parse_args()
    load_env()

    from client import RecordingClient

    client = RecordingClient(label=f"gender-{args.arm}", repair=True)
    genders, cost = ARMS[args.arm](args, client)
    write_json(
        RUNS / args.tag / "genders.json",
        {
            "arm": args.arm,
            "model": args.model,
            "cost_usd": cost,
            "genders": genders,
            "calls": client.ledger(),
        },
    )
    unknown = sum(1 for value in genders.values() if value is None)
    print(f"{args.tag}: {len(genders)} characters, {unknown} ungendered, ${cost:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

The `attribution` arm from the spec — gender folded into the attribution prompt,
zero extra passes — is measured in Task 11 Step 2 as a *packing* arm rather than
here, because its whole point is what it does to cost and to attribution
precision, and that needs the packing matrix to be interpretable.

`attribution["spans"]` must match what `run.py` writes. Read `run.py` before
running this; if the stored shape differs, adapt to the real one.

- [ ] **Step 3: Run the four arms**

```bash
export KENKUI_EVAL_BOOK=dune
for arm in baseline named roles; do
  ../../.venv/bin/python gender_run.py $arm --tag gender-$arm \
    --model openrouter/deepseek/deepseek-v3.2
done
../../.venv/bin/python score_gender.py --gold \
  --runs gender-baseline gender-named gender-roles
```

- [ ] **Step 4: Check the motivating cases specifically**

```bash
../../.venv/bin/python -c "
from common import RUNS, read_json
for tag in ('gender-baseline', 'gender-named', 'gender-roles'):
    g = read_json(RUNS / tag / 'genders.json')['genders']
    hits = {k: v for k, v in g.items() if any(t in k for t in ('nefud', 'feyd', 'chani'))}
    print(f'{tag:22}', hits)
"
```

Expected: `baseline` shows at least one `null` or wrong value; at least one paid
arm resolves them. If no arm fixes Nefud, the hypothesis in the spec is wrong and
Plan 2 must be rethought rather than written.

- [ ] **Step 5: Commit**

No commit. Record which arm resolved Nefud, Feyd and Chani in `RUNBOOK.md`.

---

## Task 10: Report ungendered spoken volume in production

**Files:**
- Modify: `src/kenkui/_domain/casting.py` (add `ungendered_volume` after
  `ungendered_pool_characters`, ends line 149)
- Modify: `src/kenkui/_resolution.py` (add `_log_ungendered_inference` beside
  `_log_ungendered_cast:360`, and call it from the same site)
- Test: `tests/test_casting_solver.py`

**Interfaces:**
- Consumes: `CharacterProfile`.
- Produces: `ungendered_volume(method: str, characters: Sequence[CharacterProfile]) -> tuple[tuple[str, int], ...]`
  — `(character_id, spoken_characters)` pairs, highest volume first, empty for
  non-gendered methods.

Non-contingent: correct whatever the arms say. `ungendered_pool_characters`
deliberately excludes `gender is None` — "the whole pool is the designed answer
for those, not a degradation" — which is right while inference is weak, and is
exactly why a misgendered Nefud produced no signal. Its contract is left alone;
this is a sibling.

**Blocked on grid-folds.**

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_casting_solver.py`:

```python
def test_ungendered_volume_reports_the_coin_flip_surface() -> None:
    characters = (
        _character("nefud", gender=None, spoken_characters=8000),
        _character("paul", gender="masculine", spoken_characters=40000),
        _character("chani", gender=None, spoken_characters=12000),
    )
    assert ungendered_volume("gendered", characters) == (
        ("chani", 12000),
        ("nefud", 8000),
    )


def test_ungendered_volume_is_silent_for_a_random_cast() -> None:
    characters = (_character("nefud", gender=None, spoken_characters=8000),)
    assert ungendered_volume("random", characters) == ()


def test_ungendered_volume_is_empty_when_every_gender_is_known() -> None:
    characters = (_character("paul", gender="masculine", spoken_characters=40000),)
    assert ungendered_volume("gendered", characters) == ()
```

Use the file's existing helper for building a `CharacterProfile`; read the top of
`tests/test_casting_solver.py` and match it rather than introducing `_character`
if a different helper is already there. Add `ungendered_volume` to the import at
line 19.

- [ ] **Step 2: Run the tests to verify they fail**

```bash
cd /Users/dizzler/Projects/Repos/kenkui-v2/kenkui
uv run pytest tests/test_casting_solver.py -q --no-cov
```

Expected: FAIL — `ImportError: cannot import name 'ungendered_volume'`.

- [ ] **Step 3: Implement `ungendered_volume`**

In `src/kenkui/_domain/casting.py`, after `ungendered_pool_characters`:

```python
def ungendered_volume(
    method: str, characters: Sequence[CharacterProfile]
) -> tuple[tuple[str, int], ...]:
    """Return (id, spoken characters) for cast characters with no gender.

    `ungendered_pool_characters` deliberately excludes these: the whole pool is
    the designed answer for a character whose gender was never inferred, not a
    degradation of a gendered cast. That reasoning holds while inference is
    weak and `None` is an honest answer.

    It stops holding once inference is meant to be good. A character with
    thousands of spoken characters and no gender is then a defect wearing the
    same shape as the designed case, and `candidates` answers both identically
    -- with the whole pool, so the voice is a coin flip. This reports the
    surface area of that flip without changing what gets rendered.

    Ordered by volume because that is the order in which the flips matter.
    """
    if method != "gendered":
        return ()
    rows = [
        (character.id, character.spoken_characters)
        for character in characters
        if character.gender is None
    ]
    return tuple(sorted(rows, key=lambda row: (-row[1], row[0])))
```

- [ ] **Step 4: Run the tests to verify they pass**

```bash
uv run pytest tests/test_casting_solver.py -q --no-cov
```

Expected: PASS.

- [ ] **Step 5: Add the log event**

In `src/kenkui/_resolution.py`, beside `_log_ungendered_cast`:

```python
def _log_ungendered_inference(
    method: str, characters: tuple[CharacterProfile, ...]
) -> None:
    """Record how much speech a gendered cast is placing on a coin flip.

    `_log_ungendered_cast` warns when the *pool* cannot honour a known gender.
    This warns about the other half: a character whose gender was never
    inferred reaches `candidates` and gets the whole pool, so a listener hears
    an arbitrary voice for a character the text may well have gendered plainly.
    Silent until now, which is how a misgendered named character survived a
    full render without a single failing check.
    """
    affected = ungendered_volume(method, characters)
    if not affected:
        return
    log_event(
        _LOGGER,
        "ungendered_inference",
        level=logging.WARNING,
        context={
            "boundary": "casting",
            "method": method,
            "characters": len(affected),
            "spoken_characters": sum(volume for _, volume in affected),
            "sample": ", ".join(name for name, _ in affected[:5]),
        },
    )
```

Import `ungendered_volume` alongside `ungendered_pool_characters` at line 18, and
call `_log_ungendered_inference(method, characters)` immediately after the
existing `_log_ungendered_cast(...)` call.

- [ ] **Step 6: Run the full gate**

```bash
uv run ruff format --check . && uv run ruff check . && uv run mypy . && uv run pytest -q
```

Expected: all pass, coverage at or above 90%. If the new branch drops coverage,
add the missing case to `tests/test_casting_solver.py` rather than lowering the
gate.

- [ ] **Step 7: Commit**

```bash
git add src/kenkui/_domain/casting.py src/kenkui/_resolution.py tests/test_casting_solver.py
git commit -m "feat: warn when a gendered cast places speech on an uninferred gender"
```

---

## Task 11: The cost sweep and the findings write-up

**Files:**
- Modify: `evals/attribution/README.md` (a "Measured findings" section, matching
  the existing style)
- Modify: `evals/attribution/RUNBOOK.md` (the full command log)

**Interfaces:**
- Consumes: every `runs/dune/*/clusters.json` and `genders.json`.
- Produces: the written findings Plan 2 is built from.

The spec's section 2 axes. Fix one cheap workhorse model, sweep packing and
layout, then sweep models across only the packings that survive — the full
cross-product is waste.

**Blocked on grid-folds and on Tasks 7-9.** Task 12 is independent and may run
in parallel with Tasks 8-11.

- [ ] **Step 1: Measure the layout arm**

Run the winning coreference and gender arms twice: once with the passage after
the instructions (as `prompts.py` writes them today), once with the passage
first. Compare `cached_tokens` share and `cost_usd`.

```bash
../../.venv/bin/python identity_run.py cited --tag cited-passage-last  --layout last
../../.venv/bin/python identity_run.py cited --tag cited-passage-first --layout first
```

Add `--layout` to `identity_run.py`. It reorders the template and nothing else:

```python
PASSAGE_FIRST = """\
Passage:
---
{passage}
---

{task}
"""


def _laid_out(template: str, passage: str, layout: str) -> str:
    """Render a prompt with the passage last (as shipped) or first.

    Prefix caching pays only when the shared content sits at the front, and
    both shipped prompts put the passage after the instructions -- so today two
    calls about one chapter share no cacheable prefix on any provider. This is
    the arm that measures whether moving it is worth a PROMPT_VERSION bump.
    """
    if layout == "last":
        return template.format(passage=passage)
    task = template.split("Passage:")[0].strip()
    return PASSAGE_FIRST.format(passage=passage, task=task)
```

Note in RUNBOOK.md whether the provider reported any cached tokens at all. Task
5 may have established that it does not, in which case this arm answers "no"
cheaply — and that is a real finding, not a failed experiment.

- [ ] **Step 2: Measure the packing arms**

Compare separate calls against gender folded into attribution against aliases
folded into roster, on: dollars per book, passes per book, cached-token share,
latency. **And on attribution precision**, via the existing `score.py` — the
failure this matrix exists to catch is a combined call that saves 40% and costs
two points of precision.

```bash
../../.venv/bin/python score.py <every attribution tag from the packing arms>
```

- [ ] **Step 3: Sweep models across the surviving packings only**

Use the spread already in `runs/`: deepseek, haiku, gemini-flash, gpt-4.1-mini,
qwen, minimax, plus one frontier reference. Report the hallucinated-citation rate
per model from Task 8 Step 5 alongside the cluster scores — it is a sharp
quality discriminator the harness did not previously have.

- [ ] **Step 4: Run the whole-book ceiling once**

One long-context model, the entire book, clusters out. Almost certainly not
shippable. It measures what the cheap arms leave on the table, which is the
number that decides whether to keep pushing.

- [ ] **Step 5: Write the findings against the acceptance gates**

For each surviving arm, state explicitly:

- did merges regress? (any increase fails the arm outright)
- did volume-weighted splits improve, and is Paul one cluster?
- is attribution precision unchanged?
- what is Tier 2 citation precision, against the threshold set before the run?
- what is the cost per book, in passes and dollars?

Match the README's existing "Measured findings" voice: the number, the mechanism,
and the trap that makes it easy to get wrong.

- [ ] **Step 6: State the verdict plainly**

Per the spec's agreed fallback: ship gender and aliases-to-attribution
regardless; coreference ships only if Tier 2 clears. **"Under-merge stays" is a
legitimate outcome** — write it as a finding, not a failure, if that is what the
numbers say.

- [ ] **Step 7: Commit**

No commit; both files are untracked. **The findings are the deliverable** —
if they matter beyond this machine, copy the summary into
`docs/superpowers/specs/` where it is tracked, and commit that instead.

---

## Task 12: Coreference model arms

**Files:**
- Create: `evals/attribution/coref_run.py`
- Produces: `runs/dune/coref-*/clusters.json`

**Interfaces:**
- Consumes: `clusters.Partition` shape, same `clusters.json` schema as Task 7.
- Produces: `runs/dune/coref-booknlp/clusters.json`,
  `runs/dune/coref-fastcoref/clusters.json`.

The spec's coreference arm 3, and gender arm 4. **Independent of Tasks 8-11** —
it needs only Task 7's scorer and can run in parallel with them, or be dropped
with a written reason if the cited-alias arm already clears the gates.

The dependency objection recorded in the eval README no longer binds: `torch
2.13` is already a hard transitive dependency via `pocket-tts`
(`uv.lock:3176`), so this costs a model download and a `transformers` dep, not
a new dependency class.

**The structural limit to expect, stated up front so a poor result is read
correctly:** coreference resolves mentions to antecedents within a window. It
will link "he" and "the young man" back to Paul inside a scene, and will link
"Usul" to Paul only where the text contains a linking construction. A model
working a 512-4096 token window forms a *local* Usul cluster in one chapter and
a *local* Paul cluster in another; merging those across 200 pages is not what
these models are trained to do. LitBank's own coreference annotations cover
roughly 2,000-word excerpts. A weak score here is a finding about scope, not
about the model.

- [ ] **Step 1: Install one coreference model**

```bash
cd /Users/dizzler/Projects/Repos/kenkui-v2/kenkui
uv pip install --python .venv/bin/python booknlp
# or, for the OntoNotes-trained contrast:
uv pip install --python .venv/bin/python fastcoref
```

Install into the eval venv only. Do **not** add either to `pyproject.toml` —
the spec puts replacing pipeline components with BookNLP out of scope for this
work, whatever it scores.

- [ ] **Step 2: Write `coref_run.py`**

It must produce the same `clusters.json` schema as `identity_run.py` so
`score_identity.py` grades it unchanged. The projection from mention clusters to
*name* clusters is the whole adapter, and is where a mention-level model loses
information:

```python
def _name_clusters(mention_clusters, volume):
    """Project mention clusters onto surface names.

    A mention cluster holds pronouns, common-noun phrases, and proper names.
    Only the proper names can be scored against gold, which is a partition of
    surface names -- so each cluster contributes the set of names it contains,
    and clusters contributing no name are dropped rather than counted as
    misses. Dropping them is right: a cluster of nothing but pronouns is real
    coreference work that this metric simply cannot see.
    """
    partition = {}
    for index, mentions in enumerate(mention_clusters):
        names = [m for m in mentions if m in volume]
        for name in names:
            partition[name] = f"coref-{index}"
    return partition
```

Take `volume` from the Task 7 spaCy baseline run so both arms weight names
identically; an arm that supplied its own volumes would not be comparable.

- [ ] **Step 3: Run and score against the same gold**

```bash
cd /Users/dizzler/Projects/Repos/kenkui-v2/kenkui/evals/attribution
export KENKUI_EVAL_BOOK=dune
../../.venv/bin/python coref_run.py booknlp --tag coref-booknlp
../../.venv/bin/python score_identity.py base-spacy base-production cited-w200 coref-booknlp
```

- [ ] **Step 4: Record the within-scene win separately**

BookNLP may lose on whole-book clustering and still be worth having for a
different reason the README already quantified: 48 quotes gold names and
gpt-4o-mini called unknown, 37 of them (77%) carrying a pronoun within 90
characters after the quote. Those are within-scene coreference, which is what
these models are actually good at.

Measure it: feed the model's within-scene resolutions into the attribution
prompt as anchors and score with the existing `score.py`. Report coverage and
precision against the arm-0 attribution baseline.

That is a *different* win from the one this plan is chasing, so record it as a
separate finding rather than folding it into the cluster table.

- [ ] **Step 5: Record BookNLP's referential gender**

BookNLP infers gender from pronouns as part of its entity pass. Emit it in
`genders.json` shape and score it with Task 9's extended `score_gender.py` — the
spec's gender arm 4, free once the model has run.

- [ ] **Step 6: Record the verdict**

No commit; `coref_run.py` is untracked. Write into `RUNBOOK.md`: the cluster
scores, the within-scene attribution delta, the gender score, wall-clock time
over a whole novel, and whether the projection dropped so many clusters that the
comparison is unfair. **If the arm is dropped, write why** — the spec's
acceptance requires every arm to have a score or a written reason.

---

## Definition of done

- `pytest test_clusters.py test_citations.py --no-cov` passes from `evals/attribution/`.
- `uv run ruff format --check . && uv run ruff check . && uv run mypy . && uv run pytest -q` passes at 90% coverage.
- `work/dune/gold-clusters.json` exists, contains Paul as one cluster and Nefud as masculine, and its disagreements have been adjudicated by hand.
- Both baselines reproduce Paul as more than one cluster.
- Every arm in the spec's section 3 has a row in `identity-scores.json` or a gender score, or a written reason it was dropped — including the coreference-model arms (Task 12) and the chapter-disjoint gate variants (Task 8, Step 2c).
- Tier 2 citation precision is a recorded number.
- `README.md` carries the findings; `RUNBOOK.md` carries every command that produced them.
- Plan 2 is writable: for each production change in the spec's section 4, the numbers say ship or do not ship.
