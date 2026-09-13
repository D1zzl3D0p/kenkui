# Roster Identity Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** One character, one voice: fix the roster's over-merge bugs, add a reasoning identity pass that merges entries naming one person and removes entries that are not people, and keep the rule-based filters as an offline fallback.

**Architecture:** spaCy finds candidate names. Deterministic safety rules (spelling normalisation, the title-residue rule, the spouse split with a gender veto) run first and are never skipped. A reasoning model then answers two questions about the numbered cast list — which entries are one person, which are not people — twice; only what both runs agree on is applied. When no identity model is configured or the pass fails, five rule-based merge/filter heuristics run instead. Attribution resolves an answer that matches exactly one entry's alias to that entry.

**Tech Stack:** Python 3.11+, spaCy (`en_core_web_lg`), LiteLLM via OpenRouter, SQLite store, pytest, Ruff, mypy.

**Spec:** `docs/superpowers/specs/2026-09-11-roster-identity-design.md`

## Global Constraints

- **Start from post-grid-folds `main`.** `2026-09-10-grid-folds-design.md` moves the quote scanner (`_characters.quotes.extract_spans`) and prefix-title data (`identity.PREFIX_TITLES`) into `_domain`. Wherever this plan names them, use their post-grid-folds location; `_domain` must not import `_characters`.
- Ruff `select = ["ALL"]`, strict mypy, pytest `--cov-fail-under=90` (`pyproject.toml`). Target `py311`.
- **No new word lists.** Only `identity.PREFIX_TITLES`, `identity.SUFFIX_TITLES`, `spacy_roster._TITLES`, `_FEMININE_TITLES`, `_MASCULINE_TITLES`, and closed grammar words ("the", prepositions).
- Identity model default `openrouter/z-ai/glm-5.3-flash`; identity calls use `reasoning_effort="high"`; attribution and model-roster calls keep `"none"`.
- Two identity runs; act only on what both agree on; never on a single run.
- `PROMPT_VERSION = "characters-v6"`.
- Every "most-mentioned" choice sorts names first, then takes `max`, so ties are deterministic.
- Offline fallback thresholds, verbatim: tie share **0.1**; group "the"-share **0.3**; place share **0.25** over **{in, on, at, from, into, onto, across, upon}**; personhood floor **0.05**; address vocative share **0.8**; lowercase attestation **≥ 2** occurrences.
- Identity excerpts: **3** per entry at the 25th/50th/75th percentile of occurrences, **110** characters either side, whitespace collapsed.
- `evals/` stays untracked. Fixtures derived from it are committed under `tests/data/roster_identity/`.
- Never push. Commit each task with explicit paths; never `git add -A`.

## File Structure

| File | Responsibility |
|---|---|
| `src/kenkui/_characters/identity.py` | `same_person` empty-residue guard; public `name_tokens`, `residue` |
| `src/kenkui/_characters/entries.py` (new) | `RosterEntry`, `most_mentioned`, the `IdentityResolver` protocol |
| `src/kenkui/_characters/spacy_roster.py` | normalisation, signal collection with spouse split, fold rules, fallback filters, entry and profile building |
| `src/kenkui/_characters/identity_pass.py` (new) | identity prompt building, two reasoning calls, parsing, agreement, applying, caching |
| `src/kenkui/_characters/prompts.py` | `IDENTITY_PROMPT`, `PROMPT_VERSION` bump |
| `src/kenkui/_characters/llm.py` | per-client reasoning effort |
| `src/kenkui/_characters/store.py` | identity decision cache; identity material in `attribution_key` |
| `src/kenkui/_characters/attribution.py` | alias answer resolution |
| `src/kenkui/_characters/__init__.py` | wiring `identity_model_id`; aliases always shown to attribution |
| `src/kenkui/_domain/operations.py`, `pipeline.py`, `_domain/summary.py`, `_resolution.py` | public `identity=` option |
| `tests/test_identity.py`, `tests/test_spacy_roster.py`, `tests/test_identity_pass.py` (new), `tests/test_attribution.py`, `tests/test_character_llm.py`, `tests/test_roster_corpus.py` (new) | tests |
| `tests/data/roster_identity/*.json` (new) | recorded identity responses for the corpus test |

`spacy_roster.py` grows by roughly 300 lines. The rules stay beside the fold logic they extend; the model boundary stays out of it (the identity pass is handed in as an object, so `spacy_roster` never imports `identity_pass`).

---

### Task 1: A name made only of titles matches no one

**Files:**
- Modify: `src/kenkui/_characters/identity.py` (`same_person`, new helpers after `_tokens`)
- Test: `tests/test_identity.py`

**Interfaces:**
- Produces: `name_tokens(name: str) -> set[str]`; `residue(name: str, titles: frozenset[str]) -> set[str]`; `same_person` returns `False` when either residue is empty (identical token sets still return `True`).

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_identity.py`:

```python
from kenkui._characters.identity import PREFIX_TITLES, name_tokens, residue


@pytest.mark.parametrize(
    ("first", "second"),
    [
        ("Great House", "Bene Gesserit"),
        ("Great House", "Rautha Harkonnen"),
        ("Aes Sedai", "Rand al'Thor"),
    ],
)
def test_a_name_made_only_of_titles_matches_no_one(first: str, second: str) -> None:
    titles = PREFIX_TITLES | {"great", "house"}
    assert same_person(first, second, titles) is False


def test_identical_title_only_names_are_still_one_name() -> None:
    assert same_person("Great House", "Great House", PREFIX_TITLES | {"great"}) is True


def test_group_full_names_has_no_title_only_wildcard() -> None:
    # "house" and "great" each lead three names, so detect_titles learns both
    # and "Great House" is left with nothing. Before the guard it matched every
    # later full name and pulled them all into Bene Gesserit.
    names = [
        "Bene Gesserit",
        "Great House",
        "Rautha Harkonnen",
        "Shadout Mapes",
        "House Atreides",
        "House Harkonnen",
        "House Corrino",
        "Great Houses",
        "Great Convention",
    ]
    entity = group_full_names(names)
    assert entity["Rautha Harkonnen"] == "Rautha Harkonnen"
    assert entity["Shadout Mapes"] == "Shadout Mapes"


def test_residue_sets_titles_aside() -> None:
    assert residue("Mr Elliot", PREFIX_TITLES) == {"elliot"}
    assert residue("Moiraine Aes Sedai", PREFIX_TITLES) == {"moiraine"}
    assert name_tokens("Dr. Kynes") == {"dr", "kynes"}
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/test_identity.py -q --no-cov`
Expected: FAIL — `ImportError: cannot import name 'name_tokens'`.

- [ ] **Step 3: Implement**

In `src/kenkui/_characters/identity.py`, after `_tokens`:

```python
def name_tokens(name: str) -> set[str]:
    """The lowercased, dot-stripped words of a name."""
    return _tokens(name)


def residue(name: str, titles: frozenset[str]) -> set[str]:
    """A name's words once prefix and suffix titles are set aside."""
    return _tokens(name) - titles - SUFFIX_TITLES
```

In `same_person`, replace the last three lines with:

```python
    rest_a = tokens_a - titles - SUFFIX_TITLES
    rest_b = tokens_b - titles - SUFFIX_TITLES
    # A name made only of titles denotes no one in particular. An empty residue
    # is a subset of every name, so without this "Great House" matched every
    # full name in Dune and folded Kynes, Nefud, Rabban and Mapes into Bene
    # Gesserit; "Aes Sedai" did the same to Rand, Mat and Perrin.
    if not rest_a or not rest_b:
        return False
    return rest_a == rest_b or rest_a < rest_b or rest_b < rest_a
```

- [ ] **Step 4: Run to verify pass**

Run: `uv run pytest tests/test_identity.py -q --no-cov`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/kenkui/_characters/identity.py tests/test_identity.py
git commit -m "fix: a title-only name no longer matches every other name"
```

---

### Task 2: Spelling normalisation

**Files:**
- Modify: `src/kenkui/_characters/spacy_roster.py` (`_clean` ~:510, `_proper_noun_spans` ~:548)
- Test: `tests/test_spacy_roster.py`

**Interfaces:**
- Produces: `_clean` folds `‘ ’ ʼ` to `'` before any other cleaning; `_proper_noun_spans` bridges proper nouns across an unspaced hyphen.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_spacy_roster.py`:

```python
class TestSpelling:
    """One name, however the typesetter spelled it."""

    def test_apostrophe_variants_clean_to_one_name(self) -> None:
        clean = spacy_roster._clean  # noqa: SLF001
        assert clean("Muad‘Dib") == clean("Muad’Dib") == "Muad'Dib"

    def test_hyphenated_names_stay_whole(self) -> None:
        nlp = spacy_roster._load(spacy_roster.DEFAULT_PIPELINE)  # noqa: SLF001
        doc = nlp('"Enough," said Feyd-Rautha Harkonnen, and Feyd-Rautha smiled.')
        spans = [s.text for s in spacy_roster._proper_noun_spans(doc)]  # noqa: SLF001
        assert "Feyd-Rautha Harkonnen" in spans
        assert "Rautha Harkonnen" not in spans

    def test_a_spaced_dash_still_separates_names(self) -> None:
        nlp = spacy_roster._load(spacy_roster.DEFAULT_PIPELINE)  # noqa: SLF001
        doc = nlp("Paul - Jessica watched him - said nothing.")
        spans = [s.text for s in spacy_roster._proper_noun_spans(doc)]  # noqa: SLF001
        assert "Paul - Jessica" not in spans
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/test_spacy_roster.py -q --no-cov -k TestSpelling`
Expected: the first two FAIL.

- [ ] **Step 3: Implement**

Near the other module constants in `spacy_roster.py`:

```python
# Curly and modifier apostrophes are one character to a reader and three to a
# string compare: "Muad‘Dib" and "Muad’Dib" were two roster entries.
_APOSTROPHES = str.maketrans({"‘": "'", "’": "'", "ʼ": "'"})
```

Replace `_clean`:

```python
def _clean(name: str) -> str:
    """Fold apostrophes, strip punctuation, the possessive, and honorifics."""
    name = name.translate(_APOSTROPHES)
    cleaned = name.strip().strip(" ,.;:!?-—’'").replace("’s", "")
    return _strip_titles(cleaned)
```

Replace `_proper_noun_spans`:

```python
def _proper_noun_spans(doc: Any) -> list[Any]:  # noqa: ANN401 - a spaCy Doc
    """Group adjacent proper nouns into one name, across an unspaced hyphen too.

    Without the grouping "Padan Fain" becomes two candidates. Without the
    hyphen bridge "Feyd-Rautha Harkonnen" becomes "Feyd" and "Rautha
    Harkonnen", two entries for one man.
    """
    spans = []
    start: int | None = None
    index, length = 0, len(doc)
    while index < length:
        token = doc[index]
        if token.pos_ == "PROPN":
            start = token.i if start is None else start
            index += 1
            continue
        bridge = (
            start is not None
            and token.text == "-"
            and not doc[index - 1].whitespace_
            and not token.whitespace_
            and index + 1 < length
            and doc[index + 1].pos_ == "PROPN"
        )
        if not bridge and start is not None:
            spans.append(doc[start:index])
            start = None
        index += 1
    if start is not None:
        spans.append(doc[start:])
    return spans
```

- [ ] **Step 4: Run to verify pass**

Run: `uv run pytest tests/test_spacy_roster.py -q --no-cov`
Expected: PASS, including every pre-existing test.

- [ ] **Step 5: Commit**

```bash
git add src/kenkui/_characters/spacy_roster.py tests/test_spacy_roster.py
git commit -m "fix: fold apostrophe variants and keep hyphenated names whole"
```

---

### Task 3: Signal collection with the spouse split

**Files:**
- Modify: `src/kenkui/_characters/spacy_roster.py` (`_Signals` ~:569, `_scan` ~:605, `infer_roster` ~:761)
- Test: `tests/test_spacy_roster.py`

**Interfaces:**
- Produces: `collect_signals(chapters, dialogue, *, pipeline=DEFAULT_PIPELINE) -> _Signals`. `_Signals` gains `flag_hosts: dict[str, set[str]]`, `titled_forms: dict[str, str]`, `title_hosts: dict[str, Counter[str]]`, `the_det: Counter[str]`, `prep: dict[str, Counter[str]]`, and `rename(source: str, target: str | None) -> None`.

The spouse split is one pass with a fold-back: every mention that opens with a title is tallied under its titled form ("Count Fenring"); once the whole book is read, titled forms of surnames held by a feminine title **and** another title stay separate, and every other titled form folds back into the bare name. This reproduces the two-pass harness exactly, because every tally is additive.

- [ ] **Step 1: Write the failing tests**

```python
# Count Fenring needs three pronoun votes (``_GENDER_MINIMUM``) for the gender
# veto to fire: each "Count Fenring ... He" with no name between is one vote.
SPOUSES = """
Count Fenring bowed low. "My dear Baron," Count Fenring said. He smiled thinly.
Lady Fenring laughed. "You are cruel," Lady Fenring said. She turned away.
"Enough," Count Fenring said. He waved a hand.
"As you wish," Lady Fenring said. She sat down.
Count Fenring rose. He left the room.
Count Fenring sighed. He was tired.
Count Fenring frowned, and Lady Fenring watched him.
"""

PROMOTED = """
Captain Nefud saluted. "Yes, my Lord," Captain Nefud said. He waited.
Lieutenant Nefud had once said the same. "At once," Nefud said. Nefud turned.
"It is done," Captain Nefud said.
"""

GROUPS = """
The Fremen came at dawn. The Fremen fought on Arrakis. "Go," Stilgar said.
The Fremen said nothing. They lived on Arrakis. From Arrakis came spice.
"Stay," Stilgar said. Stilgar waited. The Fremen watched Stilgar.
"""


def signals_of(text: str) -> spacy_roster._Signals:  # noqa: SLF001
    chapter = kk.ChapterInspection(
        id="ch-1", index=0, title="One", speech_characters=len(text), text=text
    )
    return spacy_roster.collect_signals(
        (chapter,), {"ch-1": extract_spans("ch-1", text)}
    )


class TestSignals:
    """What one read of the book records, and how titled forms fold."""

    def test_a_couple_sharing_a_surname_are_two_candidates(self) -> None:
        signals = signals_of(SPOUSES)
        assert signals.mentions["Count Fenring"] >= 3
        assert signals.mentions["Lady Fenring"] >= 3
        assert "Fenring" not in signals.mentions

    def test_a_man_promoted_keeps_one_name(self) -> None:
        signals = signals_of(PROMOTED)
        assert signals.mentions["Nefud"] >= 5
        assert "Captain Nefud" not in signals.mentions

    def test_group_and_place_evidence_is_recorded(self) -> None:
        signals = signals_of(GROUPS)
        assert signals.the_det["Fremen"] >= 2
        assert signals.prep["Arrakis"]["on"] >= 1

    def test_rename_moves_every_tally(self) -> None:
        signals = spacy_roster._Signals()  # noqa: SLF001
        signals.mentions["A"] = 2
        signals.gender["A"]["masculine"] = 3
        signals.chapters["A"].add("ch-1")
        signals.title_hosts["count"]["A"] = 1
        signals.rename("A", "B")
        assert signals.mentions["B"] == 2
        assert signals.gender["B"]["masculine"] == 3
        assert signals.chapters["B"] == {"ch-1"}
        assert signals.title_hosts["count"]["B"] == 1
        assert "A" not in signals.mentions
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/test_spacy_roster.py -q --no-cov -k TestSignals`
Expected: FAIL — `AttributeError: module ... has no attribute 'collect_signals'`.

- [ ] **Step 3: Implement**

In `_Signals.__init__`, after the existing fields:

```python
        # Which titles each cleaned name was seen under: decides the spouse split.
        self.flag_hosts: dict[str, set[str]] = defaultdict(set)
        # Titled tally keys ("Count Fenring") and the bare name each folds into.
        self.titled_forms: dict[str, str] = {}
        # Offline-fallback evidence, keyed like every other tally.
        self.title_hosts: dict[str, Counter[str]] = defaultdict(Counter)
        self.the_det: Counter[str] = Counter()
        self.prep: dict[str, Counter[str]] = defaultdict(Counter)
```

Add to `_Signals`:

```python
def rename(self, source: str, target: str | None) -> None:
    """Move every tally kept for ``source`` onto ``target``, or drop it."""
    for counter in (
        self.mentions,
        self.speech,
        self.agency,
        self.vocative,
        self.animate,
        self.titled,
        self.the_det,
    ):
        value = counter.pop(source, 0)
        if target is not None and value:
            counter[target] += value
    for table in (self.gender, self.title_gender, self.prep):
        votes = table.pop(source, None)
        if target is not None and votes:
            table[target].update(votes)
    seen = self.chapters.pop(source, None)
    if target is not None and seen:
        self.chapters[target] |= seen
    for tally in (
        *self.addressed_in.values(),
        *self.speaking_in.values(),
        *self.title_hosts.values(),
    ):
        value = tally.pop(source, 0)
        if target is not None and value:
            tally[target] += value
```

Add helpers before `_scan`:

```python
def _usable(name: str) -> bool:
    return _plausible(name) and name.lower() not in PRONOUNS


def _only_titles(name: str) -> bool:
    return all(token.lower().strip(".") in _TITLES for token in name.split())


def _attached_title(raw: str, before: str) -> str | None:
    """The title opening a mention, or standing just before it."""
    words = raw.split()
    leading = words[0].lower().strip(".") if words else ""
    if leading in _TITLES:
        return leading
    preceding = before.strip(".")
    return preceding if preceding in _TITLES else None


def _leading_title(raw: str, cleaned: str) -> str | None:
    """The title word a mention opens with, spelled as the text spells it."""
    words = raw.strip().split()
    if not words:
        return None
    head = words[0].strip(".,")
    if head.lower() in _TITLES and head.lower() != cleaned.lower():
        return head
    return None
```

Replace `_scan`:

```python
def _scan(
    doc: Any,  # noqa: ANN401 - a spaCy Doc; the package ships no stubs
    chapter_id: str,
    quote_bounds: Sequence[tuple[int, int]],
    into: _Signals,
) -> None:
    """Accumulate every signal one parsed chapter carries."""
    for span in _proper_noun_spans(doc):
        token = span[0]
        cleaned = _clean(span.text)
        before = doc[span.start - 1].lower_ if span.start else ""
        attached = _attached_title(span.text, before)
        if attached and _usable(cleaned) and not _only_titles(cleaned):
            into.flag_hosts[cleaned].add(attached)
        head = _leading_title(span.text, cleaned)
        if head is None:
            if not _usable(cleaned):
                continue
            name = cleaned
        else:
            # Tallied under the titled form until the book is read: only then is
            # it known whether "Fenring" is one man or a husband and wife.
            name = f"{head} {cleaned}"
            into.titled_forms[name] = cleaned
        into.mentions[name] += 1
        into.chapters[name].add(chapter_id)
        inside_quote = any(start <= token.idx < end for start, end in quote_bounds)
        if inside_quote and _addressed(token):
            into.vocative[name] += 1
            into.addressed_in[chapter_id][name] += 1
        anchor = span.root
        if anchor.dep_ in ("nsubj", "nsubjpass") and anchor.head.pos_ == "VERB":
            if anchor.head.lemma_.lower() in _SPEECH_LEMMAS:
                into.speech[name] += 1
                into.speaking_in[chapter_id][name] += 1
            else:
                into.agency[name] += 1
        if anchor.dep_ == "poss" and anchor.head.lemma_.lower() in _ANIMATE_NOUNS:
            into.animate[name] += 1
        if token.i and token.nbor(-1).lower_ in _TITLES:
            into.titled[name] += 1
        _gender_signals(doc, span, name, into)
        if attached and not _only_titles(name):
            into.title_hosts[attached][name] += 1
        if before == "the" and span[0].lower_.strip(".") not in _TITLES:
            into.the_det[name] += 1
        if anchor.dep_ == "pobj":
            into.prep[name][anchor.head.lower_] += 1


def _fold_titles(signals: _Signals) -> None:
    """Keep a couple apart; fold every other titled form into the bare name.

    A surname held by a feminine title and any other ("Lady" and "Count"
    Fenring, "Mistress" and "Master" Luhhan) names two people. One held only by
    other titles ("Corporal", then "Captain" Nefud) is one person promoted.
    """
    flagged = {
        name
        for name, held in signals.flag_hosts.items()
        if held & _FEMININE_TITLES and held - _FEMININE_TITLES
    }
    for key, cleaned in sorted(signals.titled_forms.items()):
        target = key if cleaned in flagged else cleaned
        if not _usable(target):
            signals.rename(key, None)
        elif target != key:
            signals.rename(key, target)


def collect_signals(
    chapters: Sequence[ChapterInspection],
    dialogue: Mapping[str, Sequence[TextSpan]],
    *,
    pipeline: str = DEFAULT_PIPELINE,
) -> _Signals:
    """Parse the book once and return every tally the roster reads."""
    nlp = _load(pipeline)
    signals = _Signals()
    for chapter in chapters:
        bounds = [
            (span.start, span.end)
            for span in dialogue.get(chapter.id, ())
            if span.is_dialogue
        ]
        _scan(nlp(chapter.text), chapter.id, bounds, signals)
    _fold_titles(signals)
    return signals
```

In `infer_roster`, replace the `nlp = _load(...)` through the scanning loop with:

```python
    signals = collect_signals(chapters, dialogue, pipeline=pipeline)
```

- [ ] **Step 4: Run to verify pass**

Run: `uv run pytest tests/test_spacy_roster.py -q --no-cov`
Expected: PASS. If a pre-existing test fails, read it: a test asserting "Lady X" and "Lord X" fold together encoded the spouse merge and is updated with a comment citing the Fenrings; anything else is a regression to fix in the code.

- [ ] **Step 5: Commit**

```bash
git add src/kenkui/_characters/spacy_roster.py tests/test_spacy_roster.py
git commit -m "feat: collect roster signals once, keeping couples who share a surname apart"
```

---

### Task 4: Fold rules

**Files:**
- Modify: `src/kenkui/_characters/spacy_roster.py` (replace `_canonical_names` ~:689; imports; `infer_roster`)
- Test: `tests/test_spacy_roster.py`

**Interfaces:**
- Consumes: `identity.detect_titles`, `group_full_names`, `name_tokens`, `residue` (Task 1); `_Signals.gender`, `.mentions`, `.title_hosts` (Task 3).
- Produces: `_fold_names(signals: _Signals, kept: set[str], *, fallback: bool) -> tuple[dict[str, str], dict[str, str]]` — `(canonical name -> target name, removed name -> reason)`.

On the identity path (`fallback=False`) ambiguous bare names and bare titles stay as their own entries for the model to place. In the fallback (`fallback=True`) the tie rule and bare-title rule decide.

- [ ] **Step 1: Write the failing tests**

```python
def signals_with(
    mentions: Mapping[str, int],
    *,
    gender: Mapping[str, Mapping[str, int]] | None = None,
    title_hosts: Mapping[str, Mapping[str, int]] | None = None,
) -> spacy_roster._Signals:  # noqa: SLF001
    signals = spacy_roster._Signals()  # noqa: SLF001
    signals.mentions.update(mentions)
    # One act each, so every name clears MIN_EVIDENCE; folding ignores agency.
    signals.agency.update(dict.fromkeys(mentions, 1))
    for name, votes in (gender or {}).items():
        signals.gender[name].update(votes)
    for title, hosts in (title_hosts or {}).items():
        signals.title_hosts[title].update(hosts)
    return signals


def fold(signals: spacy_roster._Signals, *, fallback: bool = True):  # noqa: ANN201, SLF001
    return spacy_roster._fold_names(  # noqa: SLF001
        signals, set(signals.mentions), fallback=fallback
    )


class TestFold:
    """Which names are one person, decided by rule."""

    def test_title_and_surname_do_not_absorb_the_family(self) -> None:
        canonical, _ = fold(
            signals_with(
                {"Mr Elliot": 40, "Anne Elliot": 30, "Walter Elliot": 20, "Anne": 400}
            )
        )
        assert canonical["Mr Elliot"] == "Mr Elliot"
        assert canonical["Anne"] == "Anne Elliot"

    def test_a_single_claimant_owns_a_titled_short_form(self) -> None:
        canonical, _ = fold(
            signals_with({"Iakin Nefud": 3, "Captain Nefud": 5, "Nefud": 50})
        )
        assert canonical["Captain Nefud"] == "Iakin Nefud"
        assert canonical["Nefud"] == "Iakin Nefud"

    def test_a_wife_never_folds_into_her_husband(self) -> None:
        signals = signals_with(
            {"Perrin Aybara": 20, "Perrin": 900, "Mistress Aybara": 5},
            gender={"Perrin": {"masculine": 90}, "Perrin Aybara": {"masculine": 5}},
        )
        canonical, _ = fold(signals)
        assert canonical["Mistress Aybara"] == "Mistress Aybara"
        assert canonical["Perrin"] == "Perrin Aybara"

    def test_tie_rule_ignores_tiny_claimants_in_the_fallback(self) -> None:
        paul = signals_with({"Paul": 1678, "Paul Atreides": 18, "Paul Muad'Dib": 14})
        assert fold(paul)[0]["Paul"] == "Paul"
        seldon = signals_with({"Seldon": 77, "Hari Seldon": 45, "Raven Seldon": 4})
        assert fold(seldon)[0]["Seldon"] == "Hari Seldon"

    def test_a_real_tie_drops_the_bare_name_in_the_fallback(self) -> None:
        charles = signals_with(
            {"Charles": 111, "Charles Hayter": 30, "Charles Musgrove": 40}
        )
        canonical, removed = fold(charles)
        assert "Charles" not in canonical
        assert "Charles" in removed

    def test_the_identity_path_keeps_ambiguous_names_for_the_model(self) -> None:
        charles = signals_with(
            {"Charles": 111, "Charles Hayter": 30, "Charles Musgrove": 40}
        )
        assert fold(charles, fallback=False)[0]["Charles"] == "Charles"

    def test_bare_titles_in_the_fallback(self) -> None:
        signals = signals_with(
            {
                "Baron": 500,
                "Vladimir Harkonnen": 16,
                "Duke": 480,
                "Leto Atreides": 15,
                "Paul Atreides": 18,
                "Mayor": 70,
            },
            title_hosts={
                "baron": {"Vladimir Harkonnen": 8},
                "duke": {"Leto Atreides": 50, "Paul Atreides": 5},
            },
        )
        canonical, removed = fold(signals)
        assert canonical["Baron"] == "Vladimir Harkonnen"
        assert "Duke" in removed
        assert canonical["Mayor"] == "Mayor"

    def test_bare_titles_on_the_identity_path_are_left_for_the_model(self) -> None:
        signals = signals_with(
            {"Baron": 500, "Vladimir Harkonnen": 16},
            title_hosts={"baron": {"Vladimir Harkonnen": 8}},
        )
        assert fold(signals, fallback=False)[0]["Baron"] == "Baron"

    def test_couple_stays_apart_end_to_end(self) -> None:
        characters, _ = roster_of(SPOUSES)
        by_name = {character.display_name: character for character in characters}
        assert "Count Fenring" in by_name and "Lady Fenring" in by_name
        assert by_name["Lady Fenring"].gender == "feminine"
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/test_spacy_roster.py -q --no-cov -k TestFold`
Expected: FAIL — `AttributeError: ... '_fold_names'`.

- [ ] **Step 3: Implement**

Replace the identity import with:

```python
from kenkui._characters.identity import (
    detect_titles,
    group_full_names,
    name_tokens,
    residue,
)
```

Near the thresholds:

```python
# Fallback only. A claimant under this share of a bare name's mentions does not
# compete for it: "Paul" (1,678) beside "Paul Atreides" (18). Used only to break
# a tie between several claimants; a single claimant is simply the owner.
_TIE_SHARE = 0.1
```

Replace `_canonical_names` with:

```python
def _contradicts(
    name: str, host: str, canonical: Mapping[str, str], signals: _Signals
) -> bool:
    """Whether a titled name's gender contradicts its host's pronouns."""
    lead = name.split()[0].lower().strip(".")
    said = (
        "feminine"
        if lead in _FEMININE_TITLES
        else "masculine"
        if lead in _MASCULINE_TITLES
        else None
    )
    if said is None:
        return False
    votes: Counter[str] = Counter()
    for alias, target in canonical.items():
        if target == host:
            votes.update(signals.gender.get(alias, {}))
    known = _majority(votes)
    return known is not None and known != said


def _title_holders(
    bare: str, canonical: Mapping[str, str], signals: _Signals
) -> Counter[str]:
    """Who a bare title belongs to: stripped titles and kept ones alike."""
    title = bare.split()[0].lower().strip(".")
    holders: Counter[str] = Counter()
    for host, count in signals.title_hosts.get(title, {}).items():
        if host in canonical:
            holders[canonical[host]] += count
    for other in canonical:
        parts = other.split()
        if len(parts) > 1 and parts[0].lower().strip(".") == title:
            holders[canonical[other]] += signals.mentions[other]
    return holders


def _fold_names(  # noqa: C901 - one branch per rule, kept together on purpose
    signals: _Signals, kept: set[str], *, fallback: bool
) -> tuple[dict[str, str], dict[str, str]]:
    """Fold the surface forms that denote one person onto a single name.

    Full names (two or more words once titles are set aside) group first. A
    name with one word or fewer left is not a full name: it attaches only to a
    single claimant, or it is not attached at all -- the rule that stops "Mr
    Elliot" absorbing Anne and Sir Walter. A titled name never folds into a host
    of the opposite gender. In the fallback a tie between claimants is broken by
    ignoring tiny ones, and a bare title folds into its only holder.
    """
    mentions = signals.mentions
    titles = detect_titles([name for name in kept if len(name.split()) > 1])
    fulls = sorted(
        (name for name in kept if len(residue(name, titles)) >= 2),
        key=lambda name: (-mentions[name], name),
    )
    entity = group_full_names(fulls)
    canonical = {name: entity[name] for name in fulls}
    removed: dict[str, str] = {}
    bare: list[str] = []
    # Untitled names first, so a host's gender is read from all of its forms.
    for name in sorted(
        set(kept) - set(fulls), key=lambda n: (bool(name_tokens(n) & titles), n)
    ):
        rest = residue(name, titles)
        only_titles = all(token in (titles | _TITLES) for token in name_tokens(name))
        if not rest or (fallback and only_titles):
            bare.append(name)
            continue
        token = next(iter(rest))
        claims: Counter[str] = Counter()
        for full in fulls:
            if token in name_tokens(full):
                claims[entity[full]] += mentions[full]
        titled = bool(name_tokens(name) & titles)
        tie_break = fallback and not titled and len(claims) > 1
        hosts = {
            host
            for host, count in claims.items()
            if not tie_break or count >= _TIE_SHARE * mentions[name]
        }
        if len(hosts) == 1:
            host = next(iter(hosts))
            canonical[name] = (
                name
                if titled and _contradicts(name, host, canonical, signals)
                else host
            )
        elif not hosts or titled or not fallback:
            canonical[name] = name
        else:
            removed[name] = "ambiguous bare name"
    for name in bare:
        if not fallback:
            canonical[name] = name
            continue
        holders = _title_holders(name, canonical, signals)
        if len(holders) == 1:
            canonical[name] = next(iter(holders))
        elif not holders:
            canonical[name] = name  # a role nobody names: "the Mayor"
        else:
            removed[name] = "title held by several"
    return canonical, removed
```

In `infer_roster`, replace `canonical = _canonical_names(signals, kept)` with `canonical, _ = _fold_names(signals, kept, fallback=True)`.

- [ ] **Step 4: Run to verify pass**

Run: `uv run pytest tests/test_spacy_roster.py tests/test_identity.py -q --no-cov`
Expected: PASS. Pre-existing tests that asserted a title-plus-surname or a title-only name folded into a family member encoded the over-merge; update them with a comment citing Mr Elliot / Great House.

- [ ] **Step 5: Commit**

```bash
git add src/kenkui/_characters/spacy_roster.py tests/test_spacy_roster.py
git commit -m "fix: fold short names only onto a single claimant, with a gender veto"
```

---

### Task 5: Roster entries and the offline filters

**Files:**
- Create: `src/kenkui/_characters/entries.py`
- Modify: `src/kenkui/_characters/spacy_roster.py` (`infer_roster`; new `_base_entries`, `_profiles`, filters)
- Test: `tests/test_spacy_roster.py`

**Interfaces:**
- Produces (`entries.py`): `RosterEntry(display_name: str, aliases: frozenset[str], mentions: int, evidence: int)`; `most_mentioned(names: Iterable[str], mentions: Mapping[str, int]) -> str`; `class IdentityResolver(Protocol): def resolve(self, entries: Sequence[RosterEntry], text: str, mentions: Mapping[str, int]) -> tuple[RosterEntry, ...] | None`.
- Produces (`spacy_roster.py`): `_base_entries(signals, text: str, *, fallback: bool) -> list[RosterEntry]`; `_profiles(entries, signals, chapters, dialogue, pipeline) -> tuple[tuple[CharacterProfile, ...], str | None]`.

- [ ] **Step 1: Create `entries.py`**

```python
"""A roster entry before it becomes a character, and who may reshape the list."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping, Sequence


@dataclass(frozen=True, slots=True)
class RosterEntry:
    """One candidate character: every surface form the rules folded together."""

    display_name: str
    aliases: frozenset[str]
    mentions: int
    evidence: int


def most_mentioned(names: Iterable[str], mentions: Mapping[str, int]) -> str:
    """The name used most, ties broken alphabetically so the choice is stable."""
    return max(sorted(names), key=lambda name: mentions.get(name, 0))


class IdentityResolver(Protocol):
    """Decides which entries are one person and which are not people."""

    def resolve(
        self, entries: Sequence[RosterEntry], text: str, mentions: Mapping[str, int]
    ) -> tuple[RosterEntry, ...] | None:
        """Return the reshaped entries, or None to use the offline fallback."""
        ...
```

- [ ] **Step 2: Write the failing tests**

```python
class TestFallbackFilters:
    """Offline only: groups, places, forms of address."""

    def test_a_group_that_never_speaks_is_removed(self) -> None:
        signals = signals_with({"Fremen": 100, "Stilgar": 80})
        signals.the_det["Fremen"] = 60
        signals.speech["Stilgar"] = 30
        names = {
            e.display_name
            for e in spacy_roster._base_entries(signals, "", fallback=True)
        }  # noqa: SLF001
        assert names == {"Stilgar"}

    def test_an_epithet_character_who_speaks_is_kept(self) -> None:
        signals = signals_with({"Dragon": 368})
        signals.the_det["Dragon"] = 360
        signals.speech["Dragon"] = 102
        names = {
            e.display_name
            for e in spacy_roster._base_entries(signals, "", fallback=True)
        }  # noqa: SLF001
        assert names == {"Dragon"}

    def test_a_place_is_removed_but_of_does_not_count(self) -> None:
        signals = signals_with({"Caladan": 52, "Muad'Dib": 165})
        signals.prep["Caladan"].update({"on": 20, "from": 5})
        signals.prep["Muad'Dib"].update({"of": 60})
        names = {
            e.display_name
            for e in spacy_roster._base_entries(signals, "", fallback=True)
        }  # noqa: SLF001
        assert names == {"Muad'Dib"}

    def test_a_form_of_address_is_removed_but_a_nickname_is_kept(self) -> None:
        signals = signals_with({"Sire": 47, "Nieshka": 39})
        signals.vocative.update({"Sire": 44, "Nieshka": 39})
        text = "Yes, sire. No, sire. Nieshka laughed."
        names = {
            e.display_name
            for e in spacy_roster._base_entries(signals, text, fallback=True)
        }  # noqa: SLF001
        assert names == {"Nieshka"}

    def test_the_identity_path_skips_the_filters(self) -> None:
        signals = signals_with({"Fremen": 100})
        signals.the_det["Fremen"] = 60
        names = {
            e.display_name
            for e in spacy_roster._base_entries(signals, "", fallback=False)
        }  # noqa: SLF001
        assert names == {"Fremen"}
```

- [ ] **Step 3: Run to verify failure**

Run: `uv run pytest tests/test_spacy_roster.py -q --no-cov -k TestFallbackFilters`
Expected: FAIL — `AttributeError: ... '_base_entries'`.

- [ ] **Step 4: Implement**

In `spacy_roster.py` imports: `import re` and `from kenkui._characters.entries import RosterEntry, most_mentioned`. Constants:

```python
# Offline fallback filters. Each asks the book, not a list.
_GROUP_THE_SHARE = 0.3
_PLACE_SHARE = 0.25
_PERSONHOOD_FLOOR = 0.05
_ADDRESS_SHARE = 0.8
# Not "of": book titles abuse it ("the Manual of Muad'Dib").
_LOCATIVE = frozenset({"in", "on", "at", "from", "into", "onto", "across", "upon"})
```

Functions:

```python
def _personhood(name: str, signals: _Signals) -> float:
    """Share of mentions where the name speaks or owns a body part or kin."""
    return (signals.speech[name] + signals.animate[name]) / signals.mentions[name]


def _group_or_place(name: str, signals: _Signals) -> str | None:
    """Why a name is not a person, or None. Epithet characters speak."""
    if _personhood(name, signals) >= _PERSONHOOD_FLOOR:
        return None
    mentions = signals.mentions[name]
    if signals.the_det[name] / mentions >= _GROUP_THE_SHARE:
        return "group"
    governed = sum(
        count for word, count in signals.prep.get(name, {}).items() if word in _LOCATIVE
    )
    return "place" if governed / mentions >= _PLACE_SHARE else None


def _lowercase_attested(name: str, text: str) -> bool:
    """Whether the word also occurs as an ordinary lowercase word: "my son"."""
    word = re.escape(name.lower()).replace("'", "['’]")
    return len(re.findall(rf"(?<![\w'’]){word}(?![\w'’])", text)) >= 2  # noqa: PLR2004


def _is_address(entry: RosterEntry, signals: _Signals, text: str) -> bool:
    """A word used only to address people: never acts, never a multi-word name."""
    if any(len(alias.split()) > 1 for alias in entry.aliases):
        return False
    acts = sum(signals.speech[a] for a in entry.aliases) > 1 or any(
        signals.animate[a] for a in entry.aliases
    )
    vocative = sum(signals.vocative[a] for a in entry.aliases)
    return (
        not acts
        and vocative / max(1, entry.mentions) >= _ADDRESS_SHARE
        and all(_lowercase_attested(alias, text) for alias in entry.aliases)
    )


def _base_entries(signals: _Signals, text: str, *, fallback: bool) -> list[RosterEntry]:
    """Kept names, folded, filtered in the fallback, ranked and capped."""
    kept = {
        name
        for name, count in signals.mentions.items()
        if count >= MIN_MENTIONS and signals.evidence(name) >= MIN_EVIDENCE
    }
    if fallback:
        kept = {name for name in kept if _group_or_place(name, signals) is None}
    canonical, _ = _fold_names(signals, kept, fallback=fallback)
    grouped: dict[str, set[str]] = defaultdict(set)
    for name, target in canonical.items():
        grouped[target].add(name)
    entries = [
        RosterEntry(
            display_name=target,
            aliases=frozenset(aliases),
            mentions=sum(signals.mentions[a] for a in aliases),
            evidence=sum(signals.evidence(a) for a in aliases),
        )
        for target, aliases in sorted(grouped.items())
    ]
    if fallback:
        entries = [e for e in entries if not _is_address(e, signals, text)]
    return sorted(entries, key=lambda e: (-e.evidence, e.display_name))[:MAX_CHARACTERS]


def _pooled(table: Mapping[str, Counter[str]], aliases: frozenset[str]) -> Counter[str]:
    pooled: Counter[str] = Counter()
    for alias in aliases:
        pooled.update(table.get(alias, {}))
    return pooled


def _profiles(
    entries: Sequence[RosterEntry],
    signals: _Signals,
    chapters: Sequence[ChapterInspection],
    dialogue: Mapping[str, Sequence[TextSpan]],
    pipeline: str,
) -> tuple[tuple[CharacterProfile, ...], str | None]:
    """Characters from entries, plus the first-person narrator if any."""
    roster = tuple(
        sorted(
            (
                CharacterProfile(
                    id=slugify(entry.display_name),
                    display_name=entry.display_name,
                    gender=_gender_of(
                        _pooled(signals.gender, entry.aliases),
                        _pooled(signals.title_gender, entry.aliases),
                    ),
                    spoken_characters=0,
                    chapter_ids=tuple(
                        sorted(
                            set().union(*(signals.chapters[a] for a in entry.aliases))
                        )
                    ),
                    aliases=tuple(sorted(entry.aliases)),
                )
                for entry in entries
                if slugify(entry.display_name)
            ),
            key=lambda character: character.id,
        )
    )
    canonical = {
        alias: entry.display_name for entry in entries for alias in entry.aliases
    }
    narrator = _narrator_of(
        chapters, dialogue, signals, canonical, frozenset(c.id for c in roster)
    )
    log_event(
        _LOGGER,
        "spacy_roster_derived",
        context={
            "boundary": "characters",
            "pipeline": pipeline,
            "candidates": len(signals.mentions),
            "characters": len(roster),
            "first_person": narrator is not None,
        },
    )
    return roster, narrator
```

Replace the body of `infer_roster` after its docstring with:

```python
    signals = collect_signals(chapters, dialogue, pipeline=pipeline)
    text = "\n".join(chapter.text for chapter in chapters)
    entries = _base_entries(signals, text, fallback=True)
    return _profiles(entries, signals, chapters, dialogue, pipeline)
```

- [ ] **Step 5: Run to verify pass**

Run: `uv run pytest tests/test_spacy_roster.py -q --no-cov`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add src/kenkui/_characters/entries.py src/kenkui/_characters/spacy_roster.py tests/test_spacy_roster.py
git commit -m "feat: roster entries, and offline filters for groups, places and address"
```

---

### Task 6: Reasoning per client

**Files:**
- Modify: `src/kenkui/_characters/llm.py` (`_LiteLLMClient` ~:55)
- Test: `tests/test_character_llm.py`

**Interfaces:**
- Produces: `_LiteLLMClient(reasoning_effort: str = "none")`; `reasoning_client(effort: str) -> Client`.

- [ ] **Step 1: Write the failing tests**

```python
def _fake_completion(seen: dict[str, object]):  # noqa: ANN202
    class _Message:
        content = "{}"

    class _Choice:
        message = _Message()

    class _Response:
        choices = (_Choice(),)

    def completion(**kwargs: object) -> _Response:
        seen.update(kwargs)
        return _Response()

    return completion


def test_the_default_client_keeps_reasoning_off(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import litellm

    seen: dict[str, object] = {}
    monkeypatch.setattr(litellm, "completion", _fake_completion(seen))
    llm._LiteLLMClient().complete("m", "p")  # noqa: SLF001
    assert seen["reasoning_effort"] == "none"


def test_a_reasoning_client_sends_its_effort(monkeypatch: pytest.MonkeyPatch) -> None:
    import litellm

    seen: dict[str, object] = {}
    monkeypatch.setattr(litellm, "completion", _fake_completion(seen))
    llm.reasoning_client("high").complete("m", "p")
    assert seen["reasoning_effort"] == "high"
```

Import `from kenkui._characters import llm` at the top of the test file if absent.

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/test_character_llm.py -q --no-cov`
Expected: FAIL — `AttributeError: ... 'reasoning_client'`.

- [ ] **Step 3: Implement**

```python
class _LiteLLMClient:
    """Direct LiteLLM call at fixed parameters."""

    def __init__(self, reasoning_effort: str = "none") -> None:
        self._reasoning_effort = reasoning_effort

    def complete(self, model: str, prompt: str) -> str:
        """Return one completion, imported lazily to keep import cheap."""
        import litellm  # noqa: PLC0415 - heavy, and unused unless casting runs

        response = litellm.completion(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            timeout=_REQUEST_TIMEOUT_SECONDS,
            # Off for attribution: reasoning tokens dominated its latency, and
            # those calls are extraction, not deliberation.
            reasoning_effort=self._reasoning_effort,
        )
        return str(response.choices[0].message.content)


def reasoning_client(effort: str) -> Client:
    """Return the provider client with an explicit reasoning effort.

    Deciding whether two names are one person is deliberation, not extraction:
    every model tested with reasoning off merged different people,
    deepseek-v4-pro included.
    """
    return _LiteLLMClient(reasoning_effort=effort)
```

- [ ] **Step 4: Run to verify pass**

Run: `uv run pytest tests/test_character_llm.py -q --no-cov`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/kenkui/_characters/llm.py tests/test_character_llm.py
git commit -m "feat: reasoning effort chosen per client, off by default"
```

---

### Task 7: The identity pass

**Files:**
- Create: `src/kenkui/_characters/identity_pass.py`
- Modify: `src/kenkui/_characters/prompts.py`
- Test: `tests/test_identity_pass.py`

**Interfaces:**
- Consumes: `RosterEntry`, `most_mentioned` (Task 5); `complete_json`, `reasoning_client`, `DEFAULT_BACKOFF_BASE` (llm); `IDENTITY_PROMPT` (prompts).
- Produces: `IDENTITY_REASONING = "high"`; `excerpts(name: str, text: str) -> list[str]`; `build_prompt(entries: Sequence[RosterEntry], text: str, mentions: Mapping[str, int]) -> str`; `Verdict(groups: tuple[tuple[int, ...], ...], excluded: frozenset[int])`; `parse(payload: Mapping[str, Any], count: int) -> Verdict`; `Decision(pairs: frozenset[frozenset[int]], excluded: frozenset[int])`; `agree(first: Verdict, second: Verdict) -> Decision`; `apply(entries, decision) -> tuple[RosterEntry, ...]`; `class IdentityPass(model_id: str, *, client: Client | None = None, cancel: CancellationToken | None = None, backoff_base: float = DEFAULT_BACKOFF_BASE)` implementing `IdentityResolver`.

- [ ] **Step 1: Add the prompt**

In `prompts.py`, set `PROMPT_VERSION = "characters-v6"` and append:

```python
IDENTITY_PROMPT = """\
Below is the cast list extracted from a novel. Each numbered entry is one
character as found so far: the names it appears under, how often it is
mentioned, and a few short excerpts from the book.

Some entries may be the same individual as another entry under a different
name: a nickname, a title or epithet used for that one person, a name given to
them later in the story, or a formal and an informal name.

Return ONLY JSON: {"same_person": [[1, 7], [4, 12, 30]], "not_individuals": [5, 9]}

- Each inner list holds the numbers of entries that are all one individual.
- List only entries that belong to a group of two or more. Leave everything
  else out.
- People who share a surname or a title are different individuals unless the
  excerpts show otherwise: husband and wife, parent and child, two sisters,
  two people with the same first name.
- If you are not sure, leave them out. Merging two different people is much
  worse than missing a match.

- not_individuals: numbers of entries that are not one individual character at
  all: a place, a group or a people, an organisation, an object or concept, the
  title of a book, or a word used to address many different people ("Sire",
  "my Lord"). A title that names different people at different points in the
  story ("the Duke") also goes here. Someone known mainly by an epithet or a
  title held by one person ("the Dragon", "the Emperor", "the Mayor") IS an
  individual: do not list them. If unsure, do not list it.

ENTRIES
"""
```

This prompt is concatenated, never passed through `str.format`, so its braces are literal.

- [ ] **Step 2: Write the failing tests**

Create `tests/test_identity_pass.py`:

```python
"""The identity pass: which roster entries are one person, and which are not people."""
# ruff: noqa: PLR2004

from __future__ import annotations

import json

from kenkui._characters.entries import RosterEntry
from kenkui._characters.identity_pass import (
    Decision,
    IdentityPass,
    Verdict,
    agree,
    apply,
    build_prompt,
    excerpts,
    parse,
)
from kenkui._characters.prompts import IDENTITY_PROMPT

TEXT = " ".join(f"Paul spoke to Usul number {i}." for i in range(8))
ENTRIES = (
    RosterEntry("Paul", frozenset({"Paul", "Paul Atreides"}), 100, 90),
    RosterEntry("Usul", frozenset({"Usul"}), 40, 30),
    RosterEntry("Arrakis", frozenset({"Arrakis"}), 20, 5),
)
MENTIONS = {"Paul": 90, "Paul Atreides": 10, "Usul": 40, "Arrakis": 20}


class ScriptedClient:
    def __init__(self, *answers: str) -> None:
        self.answers = list(answers)
        self.prompts: list[str] = []

    def complete(self, model: str, prompt: str) -> str:
        assert model
        self.prompts.append(prompt)
        return self.answers.pop(0)


def test_excerpts_are_three_windows_at_the_quartiles() -> None:
    found = excerpts("Paul", TEXT)
    assert len(found) == 3
    assert all("Paul" in window for window in found)


def test_prompt_numbers_entries_with_other_names_and_excerpts() -> None:
    prompt = build_prompt(ENTRIES, TEXT, MENTIONS)
    assert prompt.startswith(IDENTITY_PROMPT)
    assert '1. Paul (100 mentions; also "Paul Atreides"): "' in prompt
    assert "2. Usul (40 mentions):" in prompt


def test_parse_discards_what_it_cannot_use() -> None:
    verdict = parse(
        {
            "same_person": [[1, 2], [3, 3], [9, 1], "x"],
            "not_individuals": [3, 0, True, "2"],
        },
        3,
    )
    assert verdict.groups == ((0, 1),)
    assert verdict.excluded == frozenset({1, 2})


def test_agreement_keeps_only_what_both_runs_say() -> None:
    first = Verdict(((0, 1),), frozenset({2}))
    second = Verdict(((0, 1), (1, 2)), frozenset())
    decision = agree(first, second)
    assert decision.pairs == frozenset({frozenset({0, 1})})
    assert decision.excluded == frozenset()


def test_a_pair_touching_an_excluded_entry_is_dropped() -> None:
    both = Verdict(((0, 2),), frozenset({2}))
    decision = agree(both, both)
    assert decision.pairs == frozenset()
    assert decision.excluded == frozenset({2})


def test_apply_merges_and_excludes() -> None:
    merged = apply(ENTRIES, Decision(frozenset({frozenset({0, 1})}), frozenset({2})))
    assert len(merged) == 1
    assert merged[0].display_name == "Paul"
    assert merged[0].aliases == frozenset({"Paul", "Paul Atreides", "Usul"})
    assert merged[0].mentions == 140


def test_resolve_applies_what_two_runs_agree_on() -> None:
    answer = json.dumps({"same_person": [[1, 2]], "not_individuals": [3]})
    client = ScriptedClient(answer, answer)
    result = IdentityPass("m", client=client, backoff_base=0).resolve(
        ENTRIES, TEXT, MENTIONS
    )
    assert result is not None
    assert {e.display_name for e in result} == {"Paul"}
    assert len(client.prompts) == 2


class BrokenClient:
    """Every answer is unusable, so every run exhausts its retries."""

    def complete(self, model: str, prompt: str) -> str:  # noqa: ARG002
        return "not json"


def test_resolve_returns_none_when_the_runs_fail() -> None:
    assert (
        IdentityPass("m", client=BrokenClient(), backoff_base=0).resolve(
            ENTRIES, TEXT, MENTIONS
        )
        is None
    )
```

- [ ] **Step 3: Run to verify failure**

Run: `uv run pytest tests/test_identity_pass.py -q --no-cov`
Expected: FAIL — `ModuleNotFoundError: No module named 'kenkui._characters.identity_pass'`.

- [ ] **Step 4: Implement `identity_pass.py`**

```python
"""The identity pass: a reasoning model reshapes the rule-built cast list.

Rules prevent wrong merges; this decides the merges no rule can reach -- Usul
and Muad'Dib are Paul, "the Dragon" is Sarkan -- and which entries are not
people at all. The model sees numbered entries and answers with numbers only,
so it can group or exclude what the rules produced but never invent a name.

Two independent runs, and only what both say is applied. One run can be wildly
wrong: a cheap model once returned twelve Persuasion characters as one person.
Reasoning is required: with it off, every model tested merged different people.
"""

from __future__ import annotations

import re
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from itertools import combinations
from typing import TYPE_CHECKING, Any

from kenkui._characters import store
from kenkui._characters.entries import RosterEntry, most_mentioned
from kenkui._characters.llm import DEFAULT_BACKOFF_BASE, complete_json, reasoning_client
from kenkui._characters.prompts import IDENTITY_PROMPT, PROMPT_VERSION
from kenkui.errors import ModelError
from kenkui.observability import get_logger, log_event

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from kenkui._characters.llm import Client
    from kenkui.cancellation import CancellationToken

_LOGGER = get_logger(__name__)

IDENTITY_REASONING = "high"
_RUNS = 2
_RADIUS = 110
_SCHEMA: Mapping[str, type] = {"same_person": list}


@dataclass(frozen=True, slots=True)
class Verdict:
    """One run's answer, as 0-based entry indices."""

    groups: tuple[tuple[int, ...], ...]
    excluded: frozenset[int]


@dataclass(frozen=True, slots=True)
class Decision:
    """What both runs agree on."""

    pairs: frozenset[frozenset[int]]
    excluded: frozenset[int]


def excerpts(name: str, text: str) -> list[str]:
    """Three windows around the name at its quartiles, not its first and last.

    The ends of a book are front and back matter.
    """
    pattern = re.escape(name).replace("'", "['’‘ʼ]")
    hits = [m.start() for m in re.finditer(rf"(?<!\w){pattern}(?!\w)", text)]
    if not hits:
        return []
    count = len(hits)
    picks = sorted({hits[count // 4], hits[count // 2], hits[(3 * count) // 4]})
    return [
        " ".join(text[max(0, i - _RADIUS) : i + len(name) + _RADIUS].split())
        for i in picks
    ]


def build_prompt(
    entries: Sequence[RosterEntry], text: str, mentions: Mapping[str, int]
) -> str:
    """The numbered cast list, one line per entry, in the order given."""
    lines = []
    for number, entry in enumerate(entries, start=1):
        head = most_mentioned(entry.aliases, mentions)
        others = sorted(entry.aliases - {head})
        also = f'; also "{", ".join(others)}"' if others else ""
        quoted = " | ".join(f'"{window}"' for window in excerpts(head, text))
        lines.append(f"{number}. {head} ({entry.mentions} mentions{also}): {quoted}")
    return IDENTITY_PROMPT + "\n".join(lines) + "\n"


def _number(value: object, count: int) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int | str):
        return None
    text = str(value).strip()
    if not text.isdigit() or not 1 <= int(text) <= count:
        return None
    return int(text) - 1


def parse(payload: Mapping[str, Any], count: int) -> Verdict:
    """Groups of two or more valid entries, and the valid exclusions."""
    groups = []
    for raw in payload.get("same_person") or []:
        if not isinstance(raw, list):
            continue
        members = sorted({i for i in (_number(x, count) for x in raw) if i is not None})
        if len(members) >= 2:  # noqa: PLR2004 - a group is two or more
            groups.append(tuple(members))
    excluded = frozenset(
        i
        for i in (_number(x, count) for x in payload.get("not_individuals") or [])
        if i is not None
    )
    return Verdict(tuple(groups), excluded)


def _pairs(verdict: Verdict) -> frozenset[frozenset[int]]:
    return frozenset(
        frozenset(pair) for group in verdict.groups for pair in combinations(group, 2)
    )


def agree(first: Verdict, second: Verdict) -> Decision:
    """Pairs and exclusions both runs returned; no pair touches an exclusion."""
    excluded = first.excluded & second.excluded
    pairs = frozenset(p for p in _pairs(first) & _pairs(second) if not p & excluded)
    return Decision(pairs, excluded)


def apply(
    entries: Sequence[RosterEntry], decision: Decision
) -> tuple[RosterEntry, ...]:
    """Merge agreed pairs, drop agreed exclusions; the biggest entry names a group."""
    parent = list(range(len(entries)))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    for pair in sorted(decision.pairs, key=sorted):
        first, second = sorted(pair)
        parent[find(first)] = find(second)
    groups: dict[int, list[int]] = defaultdict(list)
    for index in range(len(entries)):
        if index not in decision.excluded:
            groups[find(index)].append(index)
    merged = []
    for members in groups.values():
        biggest = max(
            sorted(members, key=lambda i: entries[i].display_name),
            key=lambda i: entries[i].mentions,
        )
        merged.append(
            RosterEntry(
                display_name=entries[biggest].display_name,
                aliases=frozenset().union(*(entries[i].aliases for i in members)),
                mentions=sum(entries[i].mentions for i in members),
                evidence=sum(entries[i].evidence for i in members),
            )
        )
    return tuple(sorted(merged, key=lambda e: (-e.evidence, e.display_name)))


class IdentityPass:
    """Two reasoning runs over the cast list; applies only what both agree on."""

    def __init__(
        self,
        model_id: str,
        *,
        client: Client | None = None,
        cancel: CancellationToken | None = None,
        backoff_base: float = DEFAULT_BACKOFF_BASE,
    ) -> None:
        self._model_id = model_id
        self._client = client
        self._cancel = cancel
        self._backoff_base = backoff_base

    def resolve(
        self, entries: Sequence[RosterEntry], text: str, mentions: Mapping[str, int]
    ) -> tuple[RosterEntry, ...] | None:
        """Reshaped entries, or None when either run fails."""
        ordered = sorted(entries, key=lambda e: (-e.mentions, e.display_name))
        prompt = build_prompt(ordered, text, mentions)
        decision = self._decide(prompt, len(ordered))
        return None if decision is None else apply(ordered, decision)

    def _decide(self, prompt: str, count: int) -> Decision | None:
        caller = (
            self._client
            if self._client is not None
            else reasoning_client(IDENTITY_REASONING)
        )
        with ThreadPoolExecutor(max_workers=_RUNS) as pool:
            futures = [
                pool.submit(
                    complete_json,
                    self._model_id,
                    prompt,
                    _SCHEMA,
                    client=caller,
                    cancel=self._cancel,
                    backoff_base=self._backoff_base,
                )
                for _ in range(_RUNS)
            ]
            try:
                payloads = [future.result() for future in futures]
            except ModelError:
                log_event(
                    _LOGGER,
                    "identity_pass_failed",
                    context={"boundary": "characters", "model": self._model_id},
                )
                return None
        return agree(parse(payloads[0], count), parse(payloads[1], count))
```

- [ ] **Step 5: Run to verify pass**

Run: `uv run pytest tests/test_identity_pass.py -q --no-cov`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add src/kenkui/_characters/identity_pass.py src/kenkui/_characters/prompts.py tests/test_identity_pass.py
git commit -m "feat: identity pass -- two reasoning runs merge and exclude roster entries"
```

---

### Task 8: Cache the identity decision; identity in the attribution key

**Files:**
- Modify: `src/kenkui/_characters/store.py` (`_SCHEMA`, `attribution_key`, new functions)
- Modify: `src/kenkui/_characters/identity_pass.py` (`IdentityPass._decide`)
- Test: `tests/test_identity_pass.py`, `tests/test_character_llm.py` or the existing store tests

**Interfaces:**
- Produces: `store.identity_key(prompt: str, model_id: str, reasoning: str, prompt_version: str) -> str`; `store.read_identity(key: str, path: Path | None = None) -> dict[str, Any] | None`; `store.write_identity(key: str, model_id: str, decision: Mapping[str, Any], path: Path | None = None) -> None`; `attribution_key(..., identity_model_id: str = "", identity_reasoning: str = "")`.

Cached because `_with_current_roster` re-derives the roster on every cache-hit render; without the cache each render would repeat two model calls and could get a different answer.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_identity_pass.py`:

```python
def test_a_decision_is_reused_without_calling_the_model() -> None:
    answer = json.dumps({"same_person": [[1, 2]], "not_individuals": [3]})
    first = ScriptedClient(answer, answer)
    IdentityPass("m", client=first, backoff_base=0).resolve(ENTRIES, TEXT, MENTIONS)
    second = ScriptedClient()  # would raise IndexError if called
    result = IdentityPass("m", client=second, backoff_base=0).resolve(
        ENTRIES, TEXT, MENTIONS
    )
    assert result is not None
    assert {e.display_name for e in result} == {"Paul"}
    assert second.prompts == []


def test_a_different_model_is_a_different_decision() -> None:
    from kenkui._characters import store
    from kenkui._characters.prompts import PROMPT_VERSION

    assert store.identity_key("p", "a", "high", PROMPT_VERSION) != store.identity_key(
        "p", "b", "high", PROMPT_VERSION
    )
```

```python
def test_the_identity_model_enters_the_attribution_key() -> None:
    from kenkui._characters import store

    params = {"temperature": 0.0}
    base = store.attribution_key("book", "m", "v", params)
    assert store.attribution_key("book", "m", "v", params, identity_model_id="") == base
    assert (
        store.attribution_key(
            "book", "m", "v", params, identity_model_id="glm", identity_reasoning="high"
        )
        != base
    )
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/test_identity_pass.py -q --no-cov`
Expected: FAIL — second resolve calls the empty client (`IndexError` surfaces as a failed run) and `identity_key` is missing.

- [ ] **Step 3: Implement**

Append to `_SCHEMA` in `store.py`:

```sql
CREATE TABLE IF NOT EXISTS identity_passes(
    identity_key TEXT PRIMARY KEY,
    model_id TEXT NOT NULL,
    decision_json TEXT NOT NULL);
```

Add:

```python
def identity_key(
    prompt: str, model_id: str, reasoning: str, prompt_version: str
) -> str:
    """Key an identity decision by everything that determines it.

    The prompt holds the whole cast list and its excerpts, so its digest stands
    in for the book and the rules that built the list.
    """
    material = {
        "model_id": model_id,
        "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
        "prompt_version": prompt_version,
        "reasoning": reasoning,
    }
    return hashlib.sha256(_canonical(material).encode()).hexdigest()


def read_identity(key: str, path: Path | None = None) -> dict[str, Any] | None:
    """Return a stored identity decision, or None on a miss or an unusable store."""
    with _reading(path) as connection:
        if connection is None:
            return None
        row = connection.execute(
            "SELECT decision_json FROM identity_passes WHERE identity_key = ?", (key,)
        ).fetchone()
    return None if row is None else json.loads(row["decision_json"])


def write_identity(
    key: str, model_id: str, decision: Mapping[str, Any], path: Path | None = None
) -> None:
    """Persist one identity decision. Best effort: a failed write only costs a recompute."""
    try:
        with _connect(path) as connection, connection:
            connection.execute(
                "INSERT OR REPLACE INTO identity_passes(identity_key, model_id, decision_json) "
                "VALUES (?, ?, ?)",
                (key, model_id, _canonical(decision)),
            )
    except (sqlite3.Error, OSError):
        return
```

`store.py` already imports `hashlib`, `json`, `sqlite3` and `Any`; add `Mapping` under its `TYPE_CHECKING` block if absent. The `with _connect(path) as connection, connection:` form is `write_attribution`'s: the second `connection` is the transaction.

In `attribution_key`, add keyword parameters `identity_model_id: str = ""` and `identity_reasoning: str = ""`, and before hashing:

```python
    if identity_model_id:
        # The identity pass decides which entries exist, so which ids every
        # stored span refers to.
        material["identity_model_id"] = identity_model_id
        material["identity_reasoning"] = identity_reasoning
```

In `IdentityPass._decide`, before the pool:

```python
        key = store.identity_key(prompt, self._model_id, IDENTITY_REASONING, PROMPT_VERSION)
        cached = store.read_identity(key)
        if cached is not None:
            return Decision(
                frozenset(frozenset(pair) for pair in cached["pairs"]),
                frozenset(cached["excluded"]),
            )
```

and after computing the decision:

```python
decision = agree(parse(payloads[0], count), parse(payloads[1], count))
store.write_identity(
    key,
    self._model_id,
    {
        "pairs": sorted(sorted(p) for p in decision.pairs),
        "excluded": sorted(decision.excluded),
    },
)
return decision
```

`identity_pass` imports `store` at module top (Task 7's import block already
does). There is no cycle: `kenkui._characters.__init__` imports `store` before it
imports `identity_pass`, and `store` imports nothing from `identity_pass`.

- [ ] **Step 4: Run to verify pass**

Run: `uv run pytest tests/test_identity_pass.py -q --no-cov` and the store tests.
Expected: PASS. The autouse `isolated_cache_root` fixture keeps these writes out of the real cache.

- [ ] **Step 5: Commit**

```bash
git add src/kenkui/_characters/store.py src/kenkui/_characters/identity_pass.py tests/test_identity_pass.py
git commit -m "feat: cache identity decisions; the identity model keys attribution"
```

---

### Task 9: Wire the identity pass into roster derivation

**Files:**
- Modify: `src/kenkui/_characters/spacy_roster.py` (`infer_roster`)
- Modify: `src/kenkui/_characters/__init__.py` (`_discover_roster`, `discover_characters`, `resolve_attribution`, `_with_current_roster`, `_attribute_chapters` call)
- Test: `tests/test_spacy_roster.py`, `tests/test_attribution.py`

**Interfaces:**
- Consumes: `IdentityResolver` (Task 5), `IdentityPass`, `IDENTITY_REASONING` (Tasks 7–8).
- Produces: `infer_roster(chapters, dialogue, *, pipeline=DEFAULT_PIPELINE, identity: IdentityResolver | None = None)`; `identity_model_id: str | None = None` on `discover_characters`, `resolve_attribution` and `_discover_roster`.

- [ ] **Step 1: Write the failing tests**

```python
class _MergeAll:
    """Stands in for the identity pass: merges every entry into the first."""

    def resolve(self, entries, text, mentions):  # noqa: ANN001, ANN201, ARG002
        head = max(entries, key=lambda e: e.mentions)
        return (
            RosterEntry(
                head.display_name,
                frozenset().union(*(e.aliases for e in entries)),
                sum(e.mentions for e in entries),
                sum(e.evidence for e in entries),
            ),
        )


class _Fails:
    def resolve(self, entries, text, mentions):  # noqa: ANN001, ANN201, ARG002
        return None


class TestIdentityWiring:
    def test_the_resolver_reshapes_the_roster(self) -> None:
        chapter = kk.ChapterInspection(
            id="ch-1",
            index=0,
            title="One",
            speech_characters=len(PASSAGE),
            text=PASSAGE,
        )
        characters, _ = spacy_roster.infer_roster(
            (chapter,), {"ch-1": extract_spans("ch-1", PASSAGE)}, identity=_MergeAll()
        )
        assert len(characters) == 1

    def test_a_failed_pass_falls_back_to_the_rules(self) -> None:
        chapter = kk.ChapterInspection(
            id="ch-1",
            index=0,
            title="One",
            speech_characters=len(PASSAGE),
            text=PASSAGE,
        )
        spans = {"ch-1": extract_spans("ch-1", PASSAGE)}
        with_failure, _ = spacy_roster.infer_roster(
            (chapter,), spans, identity=_Fails()
        )
        offline, _ = spacy_roster.infer_roster((chapter,), spans)
        assert with_failure == offline
```

Import `RosterEntry` from `kenkui._characters.entries` in the test file.

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/test_spacy_roster.py -q --no-cov -k TestIdentityWiring`
Expected: FAIL — `TypeError: infer_roster() got an unexpected keyword argument 'identity'`.

- [ ] **Step 3: Implement**

`infer_roster`:

```python
def infer_roster(
    chapters: Sequence[ChapterInspection],
    dialogue: Mapping[str, Sequence[TextSpan]],
    *,
    pipeline: str = DEFAULT_PIPELINE,
    identity: IdentityResolver | None = None,
) -> tuple[tuple[CharacterProfile, ...], str | None]:
    """...existing docstring, plus:

    With ``identity``, the safety rules alone build the list and the resolver
    reshapes it; without one, or when it fails, the offline rules do.
    """
    signals = collect_signals(chapters, dialogue, pipeline=pipeline)
    text = "\n".join(chapter.text for chapter in chapters)
    entries: Sequence[RosterEntry] | None = None
    if identity is not None:
        entries = identity.resolve(
            _base_entries(signals, text, fallback=False), text, signals.mentions
        )
    if entries is None:
        entries = _base_entries(signals, text, fallback=True)
    return _profiles(entries, signals, chapters, dialogue, pipeline)
```

(`IdentityResolver` imported under `TYPE_CHECKING` from `kenkui._characters.entries`.)

`__init__.py`:

- `from kenkui._characters.identity_pass import IDENTITY_REASONING, IdentityPass`.
- `_discover_roster(..., *, on_progress, identity_model_id: str | None = None)`: in the spaCy branch pass
  `identity=IdentityPass(identity_model_id, client=client, cancel=cancel) if identity_model_id else None`
  to `spacy_roster.infer_roster`.
- `discover_characters(..., identity_model_id: str | None = None)` forwards it.
- `resolve_attribution(..., identity_model_id: str | None = None)`: pass
  `identity_model_id=identity_model_id or ""` and
  `identity_reasoning=IDENTITY_REASONING if identity_model_id else ""` to `store.attribution_key`;
  forward to `_discover_roster`; and call
  `_with_current_roster(cached, inspection, roster_model, identity_model_id, client)`.
- `_with_current_roster(record, inspection, roster_model, identity_model_id=None, client=None)`: pass the same `identity=` to its `spacy_roster.infer_roster` call. The cached decision (Task 8) makes this a store read, not a model call.
- In the `_attribute_chapters(...)` call, `include_aliases=True`. The attribution model now always sees the names it may answer with; the prompt change is covered by the `PROMPT_VERSION` bump.

Update the `resolve_attribution` docstring with one sentence on `identity_model_id`.

- [ ] **Step 4: Run to verify pass**

Run: `uv run pytest tests/test_spacy_roster.py tests/test_attribution.py tests/test_identity_pass.py -q --no-cov`
Expected: PASS. Tests that asserted the alias block is absent for an automatic roster encoded the old `include_aliases=supplied_roster`; update them.

- [ ] **Step 5: Commit**

```bash
git add src/kenkui/_characters/spacy_roster.py src/kenkui/_characters/__init__.py tests/test_spacy_roster.py tests/test_attribution.py
git commit -m "feat: identity pass reshapes the spaCy roster, with an offline fallback"
```

---

### Task 10: Resolve an answer by unique alias

**Files:**
- Modify: `src/kenkui/_characters/attribution.py` (`_resolve` ~:77, `attribute_chapter`, `_answers`)
- Test: `tests/test_attribution.py`

**Interfaces:**
- Produces: `_alias_ids(characters: Sequence[CharacterProfile]) -> dict[str, str]`; `_resolve(..., aliases: Mapping[str, str] | None = None)`.

- [ ] **Step 1: Write the failing tests**

```python
from kenkui._characters.attribution import _alias_ids, _resolve


def _profile(cid: str, *aliases: str) -> CharacterProfile:
    return CharacterProfile(
        id=cid,
        display_name=aliases[0],
        gender=None,
        spoken_characters=0,
        chapter_ids=(),
        aliases=aliases,
    )


def test_an_answer_matching_one_alias_resolves_to_that_character() -> None:
    aliases = _alias_ids([_profile("paul-atreides", "Paul Atreides", "Paul", "Usul")])
    known = frozenset({"paul-atreides"})
    assert (
        _resolve("paul", known, chapter_id="ch-1", aliases=aliases) == "paul-atreides"
    )
    assert (
        _resolve("Usul", known, chapter_id="ch-1", aliases=aliases) == "paul-atreides"
    )


def test_an_alias_two_characters_share_resolves_to_neither() -> None:
    aliases = _alias_ids(
        [
            _profile("charles-hayter", "Charles Hayter", "Charles"),
            _profile("charles-musgrove", "Charles Musgrove", "Charles"),
        ]
    )
    known = frozenset({"charles-hayter", "charles-musgrove"})
    assert "charles" not in aliases
    assert (
        _resolve("charles", known, chapter_id="ch-1", aliases=aliases)
        == "role:charles@ch-1"
    )
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/test_attribution.py -q --no-cov -k alias`
Expected: FAIL — `ImportError: cannot import name '_alias_ids'`.

- [ ] **Step 3: Implement**

```python
def _alias_ids(characters: Sequence[CharacterProfile]) -> dict[str, str]:
    """Alias slug to character id, for aliases exactly one character claims.

    The model answers with the name the book uses ("paul") more often than the
    id it was given ("paul-atreides"); on Dune that turned 33,990 characters of
    Paul's speech into seventeen chapter-scoped roles. An alias two characters
    share names neither.
    """
    owners: dict[str, set[str]] = {}
    for character in characters:
        for alias in (*character.aliases, character.display_name):
            slug = slugify(alias)
            if slug:
                owners.setdefault(slug, set()).add(character.id)
    return {slug: next(iter(ids)) for slug, ids in owners.items() if len(ids) == 1}
```

In `_resolve`, add `aliases: Mapping[str, str] | None = None` to the keyword-only parameters, and after `if candidate in known: return candidate`:

```python
    if aliases:
        owner = aliases.get(slugify(candidate))
        if owner is not None:
            return owner
```

In `attribute_chapter`, compute `aliases = _alias_ids(characters)` and pass it to `_answers(..., aliases=aliases)`, which passes it to `_resolve(..., aliases=aliases)`.

- [ ] **Step 4: Run to verify pass**

Run: `uv run pytest tests/test_attribution.py -q --no-cov`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/kenkui/_characters/attribution.py tests/test_attribution.py
git commit -m "feat: resolve an answer that matches exactly one character's alias"
```

---

### Task 11: Public `identity=` option

**Files:**
- Modify: `src/kenkui/_domain/operations.py` (`InferCharacters` ~:109), `src/kenkui/pipeline.py` (`infer_characters` ~:455), `src/kenkui/_domain/summary.py` (~:221), `src/kenkui/_resolution.py` (`resolve_attribution` call ~:140, `discover_characters` call ~:352)
- Modify: `docs/usage.md` (the `infer_characters` section)
- Test: `tests/test_pipeline.py`, `tests/test_introspection.py`

**Interfaces:**
- Produces: `DEFAULT_IDENTITY_MODEL = "openrouter/z-ai/glm-5.3-flash"` in `_domain/operations.py`; `InferCharacters(model_id: str, identity_model_id: str | None = DEFAULT_IDENTITY_MODEL)`; `Pipeline.infer_characters(model: str, *, identity: str | None = DEFAULT_IDENTITY_MODEL)`.

- [ ] **Step 1: Write the failing tests**

```python
def test_infer_characters_defaults_to_the_identity_model(tmp_path: Path) -> None:
    from kenkui._domain.operations import DEFAULT_IDENTITY_MODEL, InferCharacters

    book = kk.book(tmp_path / "x.epub").infer_characters("spacy")
    op = next(o for o in book.operations if isinstance(o, InferCharacters))
    assert op.identity_model_id == DEFAULT_IDENTITY_MODEL


def test_identity_none_keeps_the_roster_offline(tmp_path: Path) -> None:
    from kenkui._domain.operations import InferCharacters

    book = kk.book(tmp_path / "x.epub").infer_characters("spacy", identity=None)
    op = next(o for o in book.operations if isinstance(o, InferCharacters))
    assert op.identity_model_id is None
    assert "identity=None" in repr(book.style)
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/test_pipeline.py -q --no-cov -k identity`
Expected: FAIL — `TypeError: ... unexpected keyword argument 'identity'`.

- [ ] **Step 3: Implement**

`_domain/operations.py`:

```python
# The reasoning model that, after the rules, merges roster entries naming one
# person and removes entries that are not people. Defined in the domain layer
# because the pipeline default lives here and _domain may not import
# _characters.
DEFAULT_IDENTITY_MODEL = "openrouter/z-ai/glm-5.3-flash"


@dataclass(frozen=True, slots=True)
class InferCharacters:
    """Derive a character roster with the named model."""

    model_id: str
    identity_model_id: str | None = DEFAULT_IDENTITY_MODEL
```

`pipeline.py`:

```python
    def infer_characters(
        self, model: str, *, identity: str | None = DEFAULT_IDENTITY_MODEL
    ) -> Pipeline:
        """Return a branch that will derive a character roster.

        ``identity`` names the reasoning model that merges roster entries
        naming one person (Paul, Usul, Muad'Dib) and removes entries that are
        not people. It applies to the spaCy roster. ``None`` keeps roster
        derivation offline, using rule-based filters instead.
        """
        return self._replace(
            InferCharacters(
                _model_id(model), None if identity is None else _model_id(identity)
            )
        )
```

`summary.py`:

```python
    if isinstance(operation, InferCharacters):
        if operation.identity_model_id == DEFAULT_IDENTITY_MODEL:
            return f"infer_characters({operation.model_id!r})"
        return (
            f"infer_characters({operation.model_id!r}, "
            f"identity={operation.identity_model_id!r})"
        )
```

`_resolution.py`: add `identity_model_id=inferring.identity_model_id if inferring is not None else None` to the `resolve_attribution(...)` call, and `identity_model_id=inferring.identity_model_id` to the `discover_characters(...)` call.

`docs/usage.md`: under `infer_characters`, document `identity=`, the default model, that it needs the same OpenRouter credentials as `attribute_quotes`, that its reasoning costs about two cents a book, and that `identity=None` keeps roster derivation offline.

- [ ] **Step 4: Run to verify pass**

Run: `uv run pytest -q --no-cov`
Expected: PASS. A pipeline-level test that resolves a spaCy roster through a fake client that cannot answer the identity prompt must pass `identity=None` — the fake would otherwise exhaust retries and fall back, slowly.

- [ ] **Step 5: Commit**

```bash
git add src/kenkui/_domain/operations.py src/kenkui/pipeline.py src/kenkui/_domain/summary.py src/kenkui/_resolution.py docs/usage.md tests/test_pipeline.py tests/test_introspection.py
git commit -m "feat: infer_characters(identity=...) chooses the identity model or offline"
```

---

### Task 12: Six-book acceptance test

**Files:**
- Create: `tests/data/roster_identity/{dune,eotw,persuasion,citycity,uprooted,foundation}.json`
- Create: `tests/test_roster_corpus.py`

**Interfaces:**
- Consumes: `spacy_roster.infer_roster`, `spacy_roster.collect_signals`, `IdentityPass`.

The corpus test replays the recorded glm-5.3-flash responses through the real `IdentityPass`. A replay client maps the recorded entry numbers onto the production prompt by alias set, and fails loudly on any entry whose alias set the production base roster does not reproduce — so it is also the base-roster parity check.

- [ ] **Step 1: Export the fixtures from the untracked harness**

From `evals/attribution/` (the harness is not committed; only its recorded answers are):

```bash
cd evals/attribution && ../../.venv/bin/python - <<'PY'
import json
from pathlib import Path
import roster_lab as L, roster_merge as M
out = Path("../../tests/data/roster_identity"); out.mkdir(parents=True, exist_ok=True)
MODEL = "openrouter/z-ai/glm-5.3-flash"
for book in L.BOOKS:
    _, roster = M.base_roster(book, 10)
    responses = [json.loads(M.out_path(book, MODEL, r, 10, "classify", "high").read_text())["response"]
                 for r in (1, 2)]
    (out / f"{book}.json").write_text(json.dumps(
        {"model": MODEL, "entries": [sorted(r["aliases"]) for r in roster], "responses": responses},
        ensure_ascii=False, indent=1) + "\n")
    print(book, len(roster))
PY
```

Expected: six files; entry counts 163, 374, 79, 107, 77, 93.

- [ ] **Step 2: Write the corpus test**

`tests/test_roster_corpus.py` — opt-in like `tests/test_grid_exactness.py`:

```python
"""Six books: the roster keeps people apart, and the identity pass merges nicknames."""
# ruff: noqa: PLR2004

from __future__ import annotations

import json
import os
import re
import zipfile
from pathlib import Path

import pytest

import kenkui as kk
from kenkui._characters import spacy_roster
from kenkui._characters.identity_pass import IdentityPass
from kenkui._characters.quotes import extract_spans

pytestmark = pytest.mark.skipif(
    not os.environ.get("KENKUI_RUN_CORPUS"), reason="set KENKUI_RUN_CORPUS=1 to run"
)

LIBRARY = Path("/Users/dizzler/Projects/Calibre Library")
FIXTURES = Path(__file__).parent / "data" / "roster_identity"
BOOKS = {
    "dune": "Frank Herbert/Dune (466)/Dune - Frank Herbert.epub",
    "eotw": "Robert Jordan/The Eye of the World (322)/The Eye of the World - Robert Jordan.epub",
    "persuasion": "Jane Austen/Persuasion (244)/Persuasion - Jane Austen.epub",
    "citycity": "China Mieville/The City & the City (463)/The City & the City - China Mieville.epub",
    "uprooted": (
        "Naomi Novik/Uprooted_ A Lush Fairytale-Inspired Fantasy by the Author of the "
        "Bestselling a Deadly Education (125)/Uprooted_ A Lush Fairytale-Inspired Fantas - "
        "Naomi Novik.epub"
    ),
    "foundation": "Isaac Asimov/Foundation (453)/Foundation - Isaac Asimov.epub",
}
T = lambda *words: set(words)  # noqa: E731
PRINCIPALS = {  # copied from evals/attribution/roster_lab.py
    "dune": {
        "Paul": T("paul"),
        "Jessica": T("jessica"),
        "Leto": T("leto"),
        "Baron": T("vladimir"),
        "Stilgar": T("stilgar"),
        "Chani": T("chani"),
        "Thufir": T("thufir", "hawat"),
        "Gurney": T("gurney", "halleck"),
        "Duncan": T("duncan", "idaho"),
        "Yueh": T("yueh", "wellington"),
        "Piter": T("piter"),
        "Feyd": T("feyd", "feyd-rautha"),
        "Kynes": T("kynes", "liet"),
        "Alia": T("alia"),
        "Irulan": T("irulan"),
        "Nefud": T("nefud"),
        "Rabban": T("rabban"),
        "Mapes": T("mapes"),
        "Harah": T("harah"),
        "Jamis": T("jamis"),
        "Fenring": T("fenring"),
        "Emperor": T("emperor", "shaddam"),
    },
    "eotw": {
        n: T(n.lower())
        for n in (
            "Rand",
            "Mat",
            "Thom",
            "Moiraine",
            "Elayne",
            "Gawyn",
            "Perrin",
            "Lan",
            "Agelmar",
            "Mordeth",
            "Egwene",
            "Nynaeve",
            "Elaida",
            "Morgase",
            "Bartim",
            "Paitr",
            "Loial",
            "Min",
            "Tam",
        )
    }
    | {"Ba'alzamon": T("ba'alzamon")},
    "persuasion": {
        "Anne": T("anne"),
        "Wentworth": T("wentworth", "frederick"),
        "Russell": T("russell"),
        "Harville": T("harville"),
        "Benwick": T("benwick"),
        "Clay": T("clay"),
        "Smith": T("smith"),
        "Louisa": T("louisa"),
        "Henrietta": T("henrietta"),
        "Mary": T("mary"),
        "Elizabeth": T("elizabeth"),
        "Walter": T("walter"),
        "Hayter": T("hayter"),
    },
    "citycity": {
        "Borlu": T("borlú", "borlu", "tyador"),
        "Corwi": T("corwi", "lizbyet"),
        "Dhatt": T("dhatt", "qussim"),
        "Bowden": T("bowden"),
        "Nancy": T("nancy"),
        "Yolanda": T("yolanda"),
        "Gadlem": T("gadlem"),
        "Syedr": T("syedr"),
        "Buric": T("buric"),
        "Aikam": T("aikam"),
    },
    "uprooted": {
        "Agnieszka": T("agnieszka", "nieshka"),
        "Sarkan": T("sarkan", "dragon"),
        "Kasia": T("kasia"),
        "Marek": T("marek"),
        "Solya": T("solya", "falcon"),
        "Alosha": T("alosha"),
        "Ballo": T("ballo"),
        "Danka": T("danka"),
    },
    "foundation": {
        "Seldon": T("seldon", "hari"),
        "Gaal": T("gaal", "dornick"),
        "Hardin": T("hardin", "salvor"),
        "Pirenne": T("pirenne"),
        "Wienis": T("wienis"),
        "Lepold": T("lepold"),
        "Verisof": T("verisof"),
        "Mallow": T("mallow", "hober"),
        "Sutt": T("sutt", "jorane"),
        "Aporat": T("aporat"),
        "Chen": T("chen", "linge"),
    },
}
APART = {
    "dune": [
        ("Paul Atreides", "Leto Atreides"),
        ("Vladimir Harkonnen", "Beast Rabban"),
        ("Count Fenring", "Lady Fenring"),
    ],
    "eotw": [
        ("Rand al'Thor", "Tam al'Thor"),
        ("Rand", "Tam"),
        ("Master Luhhan", "Mistress Luhhan"),
        ("Master al'Vere", "Mistress al'Vere"),
        ("Master Cauthon", "Mistress Cauthon"),
        ("Master Aybara", "Mistress Aybara"),
        ("Master Grinwell", "Mistress Grinwell"),
    ],
    "persuasion": [
        ("Charles Hayter", "Charles Musgrove"),
        ("Anne Elliot", "Walter Elliot"),
        ("Mr Elliot", "Anne Elliot"),
        ("Admiral Croft", "Mrs Croft"),
        ("Mrs Musgrove", "Louisa Musgrove"),
        ("Mr Musgrove", "Charles Musgrove"),
        ("Mr Elliot", "Walter Elliot"),
    ],
    "citycity": [("Mr Geary", "Mrs Geary")],
    "uprooted": [],
    "foundation": [],
}
SAME = {
    "dune": [
        ("Paul", "Usul"),
        ("Paul", "Muad'Dib"),
        ("Liet", "Kynes"),
        ("Paul", "Paul Atreides"),
    ],
    "eotw": [
        ("Mat", "Matrim"),
        ("Mat", "Matrim Cauthon"),
        ("Bran", "Brandelwyn al'Vere"),
    ],
    "uprooted": [("Agnieszka", "Nieshka"), ("Sarkan", "Dragon"), ("Solya", "Falcon")],
    "persuasion": [("Frederick", "Wentworth")],
    "foundation": [("Seldon", "Raven Seldon"), ("Seldon", "Hari Seldon")],
    "citycity": [("Tyador", "Borlú"), ("Tye", "Borlú")],
}
_DOCTYPE = re.compile(rb"<!DOCTYPE[^>\[]*(\[[^\]]*\])?[^>]*>", re.IGNORECASE)
_LINE = re.compile(
    r'^(\d+)\. (.+?) \((\d+) mentions(?:; also "(.*?)")?\): ', re.MULTILINE
)


def _norm(text: str) -> str:
    return text.casefold().replace("’", "'").replace("‘", "'")


def _inspection(book: str, tmp_path: Path) -> kk.BookInspection:
    """Parse a copy with DOCTYPE removed: the parser rejects EPUB2 declarations."""
    source = LIBRARY / BOOKS[book]
    copy = tmp_path / f"{book}.epub"
    with (
        zipfile.ZipFile(source) as archive,
        zipfile.ZipFile(copy, "w", zipfile.ZIP_DEFLATED) as out,
    ):
        if "mimetype" in archive.namelist():
            out.writestr(
                zipfile.ZipInfo("mimetype"),
                archive.read("mimetype"),
                zipfile.ZIP_STORED,
            )
        for item in archive.infolist():
            if item.filename == "mimetype":
                continue
            data = archive.read(item.filename)
            if item.filename.lower().endswith(
                (".xhtml", ".html", ".htm", ".opf", ".ncx", ".xml")
            ):
                data = _DOCTYPE.sub(b"", data)
            out.writestr(item, data)
    return kk.epub(str(copy)).inspect()


class _Replay:
    """Returns the recorded answers, renumbered onto this prompt by alias set."""

    def __init__(self, fixture: dict) -> None:
        self.entries = [frozenset(e) for e in fixture["entries"]]
        self.responses = list(fixture["responses"])

    def complete(self, model: str, prompt: str) -> str:  # noqa: ARG002
        here: dict[frozenset[str], int] = {}
        for match in _LINE.finditer(prompt):
            aliases = {match.group(2)} | (
                set(match.group(4).split(", ")) if match.group(4) else set()
            )
            here[frozenset(aliases)] = int(match.group(1))
        missing = [sorted(e) for e in self.entries if e not in here]
        assert not missing, f"base roster differs from the harness: {missing[:5]}"
        raw = self.responses.pop(0)
        payload = json.loads(raw[raw.find("{") : raw.rfind("}") + 1])
        renumber = {old: here[entry] for old, entry in enumerate(self.entries, start=1)}
        payload["same_person"] = [
            [renumber[int(n)] for n in g if str(n).isdigit() and int(n) in renumber]
            for g in payload.get("same_person", [])
        ]
        payload["not_individuals"] = [
            renumber[int(n)]
            for n in payload.get("not_individuals", [])
            if str(n).isdigit() and int(n) in renumber
        ]
        return json.dumps(payload)


def _score(book: str, roster, mentions) -> tuple[list, list, list, int]:  # noqa: ANN001
    def words(character) -> set[str]:  # noqa: ANN001
        return {
            _norm(w)
            for a in (*character.aliases, character.display_name)
            for w in a.split()
        }

    kept = {a for c in roster for a in c.aliases}
    lost = []
    for label, toks in PRINCIPALS[book].items():
        mine = [n for n in mentions if {_norm(w) for w in n.split()} & toks]
        total = sum(mentions[n] for n in mine)
        if not total or sum(mentions[n] for n in mine if n in kept) / total < 0.5:
            lost.append(label)
    merged = [
        c.display_name
        for c in roster
        if len([p for p, toks in PRINCIPALS[book].items() if toks & words(c)]) > 1
    ]
    apart = [
        f"{a}+{b}"
        for a, b in APART[book]
        for c in roster
        if {_norm(a), _norm(b)} <= {_norm(x) for x in (*c.aliases, c.display_name)}
    ]
    same = sum(
        1
        for a, b in SAME[book]
        if any(a in c.aliases and b in c.aliases for c in roster)
    )
    return lost, merged, apart, same


@pytest.fixture(scope="module")
def results(tmp_path_factory: pytest.TempPathFactory) -> dict[str, dict]:
    out = {}
    for book in BOOKS:
        inspection = _inspection(book, tmp_path_factory.mktemp(book))
        spans = {c.id: extract_spans(c.id, c.text) for c in inspection.chapters}
        mentions = spacy_roster.collect_signals(inspection.chapters, spans).mentions
        offline, _ = spacy_roster.infer_roster(inspection.chapters, spans)
        fixture = json.loads((FIXTURES / f"{book}.json").read_text())
        identity = IdentityPass(
            fixture["model"], client=_Replay(fixture), backoff_base=0
        )
        online, _ = spacy_roster.infer_roster(
            inspection.chapters, spans, identity=identity
        )
        out[book] = {
            "offline": _score(book, offline, mentions),
            "online": _score(book, online, mentions),
        }
    return out


@pytest.mark.parametrize("path", ["offline", "online"])
def test_no_person_lost_merged_or_joined(results: dict, path: str) -> None:
    for book, scored in results.items():
        lost, merged, apart, _ = scored[path]
        assert (lost, merged, apart) == ([], [], []), f"{book} {path}"


def test_the_identity_pass_merges_nicknames(results: dict) -> None:
    assert sum(scored["online"][3] for scored in results.values()) >= 14


def test_the_fallback_matches_its_measured_baseline(results: dict) -> None:
    assert sum(scored["offline"][3] for scored in results.values()) >= 3
```

The replay client returns the two recorded answers in either order; agreement is symmetric, so the thread order of the two runs does not matter.

- [ ] **Step 3: Run it**

Run: `KENKUI_RUN_CORPUS=1 uv run pytest tests/test_roster_corpus.py -q --no-cov`
Expected: PASS, a few minutes. A `base roster differs from the harness` assertion means Tasks 2–5 diverged from `evals/attribution/roster_lab.py` stage 10 — compare the listed alias sets against the harness for that book, fix the code, rerun.

- [ ] **Step 4: Commit**

```bash
git add tests/test_roster_corpus.py tests/data/roster_identity/
git commit -m "test: six-book roster acceptance, replaying recorded identity answers"
```

---

### Task 13: Project gate and the Dune comparison

**Files:**
- none new; operational

- [ ] **Step 1: Run the project gate**

```bash
uv run ruff format --check . && uv run ruff check . && uv run mypy . && uv run pytest -q
```

Expected: all pass, coverage ≥ 90%. Close coverage gaps with tests, not by lowering the gate.

- [ ] **Step 2: Confirm spend before the paid comparison**

Re-attributing Dune makes real calls: two identity runs (about 2¢) plus a full attribution pass with `openrouter/deepseek/deepseek-v4-flash`. Ask the user before running it.

- [ ] **Step 3: Re-attribute Dune and compare with the stored render**

Resolve Dune with `.infer_characters("spacy").attribute_quotes("openrouter/deepseek/deepseek-v4-flash")` (audio not needed; resolution stops after casting). Then, in `casting.sqlite3`, compare the new attribution against `3d156cdd…`:

- Paul: roster entries and `role:` fragments containing "paul", "muad", "usul"; distinct voices per cast;
- total `role:` volume and its ungendered share (baseline 74,991 characters, 100% ungendered);
- speech on non-character entries (baseline: bene-gesserit 10,572, fremen 7,725);
- overall ungendered share (baseline 34.1%; 8.3% from the gender refresh alone).

Use the queries from the investigation (grouping `characters` by `character_id LIKE 'role:%'`, joining `cast_assignments` through `casts`).

- [ ] **Step 4: Record the result**

Append the before/after numbers to the spec's Evidence section in a short "Dune re-attribution" paragraph, and commit it:

```bash
git add docs/superpowers/specs/2026-09-11-roster-identity-design.md
git commit -m "docs: record the Dune re-attribution against the stored render"
```

---

## Self-review notes

- **Spec coverage:** normalisation (T2), title-residue rule (T1, T4), spouse split and gender veto (T3, T4), identity pass input/question/output/agreement/model/failure (T6, T7), caching (T8), offline fallback — bare titles with the union title rule, tie rule, groups, places without "of", address (T4, T5), answer resolution and unconditional aliases (T9, T10), gender pooling (T5), per-call reasoning (T6), cache and key changes with the `PROMPT_VERSION` bump (T7, T8, T9), public option (T11), acceptance gates (T12, T13).
- **Deliberate deviation from the harness:** display names of merged groups come from the biggest constituent entry rather than the most-mentioned alias; ids may differ from the harness, alias sets do not, and the corpus test compares alias sets.
