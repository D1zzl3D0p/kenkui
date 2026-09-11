# Roster Identity — Design

**Status:** Approved for implementation planning on 2026-09-11.

**Supersedes:** `2026-09-10-coreference-and-gender-design.md` and
`../plans/2026-09-10-coreference-and-gender-measurement.md`. Their premise — that
Paul's voices came from unmergeable aliases and needed a coreference evaluation
to decide — was overtaken by measurement: the dominant defects were bugs in the
roster itself.

**Depends on:** `2026-09-10-grid-folds-design.md`, which relocates
`PREFIX_TITLES` out of `_characters/identity.py`. This work lands after it.

## Goal

One character, one voice; the right voice for the right person; and nothing on
the cast that is not a person. Concretely, on Dune: Paul, Usul and Muad'Dib in
one voice; Captain Nefud, Kynes, Rabban and Mapes each in their own; Fremen,
Arrakis and "Sire" in none.

## What the investigation found

Measured against the stored Dune render (attribution `3d156cdd…`, spaCy roster,
deepseek-v4-flash attribution) and six books from the local library.

**Paul was 26 entries, not 4.** Three roster entries plus 23 per-chapter
`role:paul@<chapter>` fragments, 70,280 spoken characters, 21–27 distinct voices
in each of four casts. The attribution model answered "paul"; the roster held
`paul-atreides`; `_resolve` found no exact id and minted a chapter-scoped role.

**Nefud, Kynes, Rabban and Mapes were swallowed into "Bene Gesserit".** That
entry held 37 aliases and 10,572 spoken characters. With those four absent from
the ballot under their own names, every line of theirs became a chapter-scoped
role with no gender — the direct cause of the Nefud misgendering report.

**The cause is two over-merge bugs in shipped code.**

1. *A name made only of titles matches everyone.* `detect_titles` learns
   "great" and "house" as titles (each leads three or more names), so "Great
   House" has an empty residue, and `same_person` treats an empty set as a
   subset of every name. In `group_full_names` it became a wildcard catching
   every lower-ranked full name without an earlier match. The same bug put Rand,
   Mat, Perrin, Thom, Moiraine and Elaida into "Aes Sedai" in The Eye of the
   World.
2. *A title plus a surname matches the whole family.* "Mr Elliot" reduces to
   "Elliot", which nests inside "Anne Elliot" and "Walter Elliot": Persuasion's
   heroine, her father and her cousin shared one entry.

The production roster fails three of six test books on these bugs alone.

**A third merge happens earlier, in title stripping.** `_clean` strips titles
before identity ever runs, so "Count Fenring" and "Lady Fenring" both become
"Fenring" — a husband and wife in one entry — and the honorific vote ("Lady")
outranks nine masculine pronouns. The Eye of the World loses five couples this
way (Master/Mistress Luhhan, al'Vere, Cauthon, Aybara, Grinwell).

**The loose-roster rule was only half right.** "A spurious roster entry is
cheap" (eval README) did not hold: in the Dune render, bare titles ("Baron",
"Duke", "Count": 56,713 characters) split three men from their named entries,
and non-characters ("Bene Gesserit", "Fremen", "Frank Herbert") carried a
further ~22,000.

**The gender defect on Dune is a stale cache, not a code bug.**
`dialogue_tags` shipped in `06c3649` after `characters-v5` without bumping
`PROMPT_VERSION`, so the stored record predates it. A dry run of
`_with_current_roster` with current code regenders 34 characters — Jessica,
Chani, Alia to feminine; Irulan and Bene Gesserit corrected from masculine —
taking ungendered speech from 34.1% to 8.3%. Re-rendering fixes it.

**Reasoning decides whether an LLM can be trusted to merge.** Every model
tested with reasoning off merged different people — deepseek-v4-pro included,
which put about twenty Eye of the World characters into one "person". With
reasoning on, every model that returned parseable output had zero safety
failures. Production sets `reasoning_effort="none"` for all calls
(`_characters/llm.py:70`), correctly for attribution.

## Design

```
spaCy candidate names           (unchanged: spacy_roster._scan)
  │
  ├─ spelling normalisation     apostrophe variants folded; hyphenated names whole
  │
  ├─ safety rules (always)      title-residue rule; spouse split + gender veto
  │
  ├─ identity pass (default)    LLM, two runs, reasoning on:
  │                             which entries are one person; which are not people
  │     └─ act only where both runs agree
  │
  └─ offline fallback           the five merge/filter rules, used only when no
                                identity model is configured or both runs fail
  │
attribution                     _resolve maps an answer that matches exactly one
                                entry's alias to that entry
```

The split of responsibility is the finding the design rests on: **rules prevent
wrong merges; the model makes merges and judges personhood.** A model can merge
but never split, so everything that must keep two people apart is a
deterministic rule, and everything that needs reading the story is the model.

### 1. Spelling normalisation

- Fold `‘ ’ ʼ` to `'` in `spacy_roster._clean` before any other cleaning.
  "Muad‘Dib" and "Muad’Dib" are one name.
- `spacy_roster._proper_noun_spans` joins proper-noun tokens across an unspaced
  hyphen: "Feyd-Rautha" is one span, not "Feyd" and "Rautha Harkonnen".

### 2. Safety rules (always on)

**Title-residue rule.** A name whose residue — its tokens minus titles — is
empty matches no one: `identity.same_person` returns False when either residue
is empty (a one-line fix; it also protects the model-roster path through
`merge_rosters`). In `spacy_roster._canonical_names`, a name with one residual
token or fewer is not a full name: it goes through the unique-claimant rule
rather than `group_full_names`. One claimant owns it; several claimants and a
title keep it as its own entry ("Mrs Musgrove", "Mr Elliot"); several claimants
and no title keep it as its own entry for the identity pass to place (in the
fallback: see the tie rule). A title used alone ("the Baron", "the Duke") also
stays as its own entry on the identity-pass path, for the model to merge into
its holder or exclude.

**Spouse split.** A two-pass collection: the first pass records which title
each cleaned name was attached to. A cleaned name held by a feminine title
*and* any other title ("Lady" and "Count" Fenring; "Mistress" and "Master"
Luhhan) keeps its title on those mentions, so the couple are two candidates. A
name held only by non-feminine titles ("Corporal", then "Captain" Nefud) is one
person and is left alone. The existing `_FEMININE_TITLES` list decides
femininity; no list is added.

**Gender veto.** A titled name never folds into a host whose pronoun majority
contradicts the title's gender. Hosts are read after every untitled name is
placed, so "Perrin" counts toward "Perrin Aybara". This keeps "Mistress Aybara"
out of Perrin.

### 3. Identity pass

**Input.** The safety-rule roster as numbered entries, most-mentioned first.
Each line: the entry's most-mentioned name, its other names, its mention count,
and three ~220-character excerpts taken at the 25th, 50th and 75th percentile of
its occurrences (the ends of a book are front and back matter).

**Question.** Which entries are the same individual under another name (a
nickname, an epithet, a title held by that one person, a name given later), and
which entries are not an individual character at all (a place, a people, an
organisation, a concept, a book title, a form of address used for many people,
a title that names different people at different times). The prompt states the
asymmetry: merging two people is much worse than missing a match; people
sharing a surname or title are different unless the excerpts show otherwise;
someone known mainly by an epithet or a title held by one person ("the Dragon",
"the Emperor", "the Mayor") *is* an individual. Full text in Appendix A.

**Output.** `{"same_person": [[1, 7], [4, 12, 30]], "not_individuals": [5, 9]}`
— entry numbers only. The vocabulary is closed: the model can only group or
exclude entries the rules produced, so it cannot invent aliases ("my Lord", "the
old man", pronouns) the way per-chapter alias lists did. Numbers out of range,
duplicates and malformed groups are discarded.

**Two runs, act on agreement.** Two independent calls. A pair of entries merges
only when both runs group them; an entry is removed only when both runs list it.
A single run can be wildly wrong — deepseek-v4-flash once returned twelve
Persuasion characters as one person — and agreement removed that class entirely
in testing.

**Model.** Default `openrouter/z-ai/glm-5.3-flash` with reasoning on (it cannot
be turned off: the endpoint rejects `reasoning_effort="none"`). Configurable;
`deepseek/deepseek-v4-pro` with reasoning is the measured alternative. The model
must reason: the prompt path sets `reasoning_effort` explicitly and never
inherits the attribution call's `"none"`.

**Failure.** Retry on transport error or unparseable output (observed: one API
error and one response truncated at 204 tokens on Dune). If fewer than two runs
succeed after retries, use the offline fallback and log it. Never act on a
single run.

**Caching.** The pass result is stored per book, keyed by the candidate roster
fingerprint, model, prompt version and reasoning setting. It is part of the
roster, so it is part of attribution's identity (see Identity and cache).

### 4. Offline fallback

Used only when no identity model is configured or the pass fails. These are
the five rules the identity pass replaces; together with the safety rules they
pass every check on the six books.

- **Bare titles.** A name made only of words from either existing title list
  (`identity.PREFIX_TITLES` or `spacy_roster._TITLES` — the first lacks baron,
  duke and count, which the second strips) is a bare title. It folds into its
  only holder ("Baron" → Vladimir Harkonnen, "Admiral" → Croft); with no holder
  it stays as a role ("the Mayor"); with several it is dropped ("the Duke", held
  by Leto and later Paul; "the Count", Fenring and Rabban), leaving the
  attribution model to resolve it per passage. Holders come from both the titles
  `_clean` strips and those it keeps ("Inspector Borlú").
- **Tie rule.** For an untitled bare name with several claimants, claimants
  with under a tenth of the bare name's mentions do not compete ("Paul" beside
  "Paul Atreides" and "Paul Muad'Dib"; "Seldon" beside "Raven Seldon"). Only
  used to break ties; a single claimant is simply the owner. A tie that
  survives drops the bare name ("Charles", between Hayter and Musgrove), as
  today.
- **Groups.** At least 30% of mentions follow "the" (not counting mentions
  that open with a title), and personhood evidence under 5%: the name is almost
  never the subject of speech and never owns a body part or kin. The second
  condition protects epithet characters ("the Dragon", "the Emperor").
- **Places.** At least 25% of mentions governed by in, on, at, from, into,
  onto, across or upon — not "of", which book titles abuse ("the Manual of
  Muad'Dib") — and the same personhood condition.
- **Forms of address.** At least 80% of mentions are vocatives, the name never
  speaks or acts, and the word also occurs as an ordinary lowercase word in the
  book ("my son", "sire"). The last condition protects nicknames used only in
  address ("Nieshka").

No rule adds a word list. The signals come from the book; the only lists used
are the title and gendered-title lists already in `spacy_roster`, and closed
grammar words ("the", prepositions).

### 5. Answer resolution in attribution

`attribution._resolve`, before minting a role: an answer whose slug equals the
slug of exactly one roster entry's alias resolves to that entry. Measured on the
stored Dune render with the new roster, 78–79% of the 74,991 characters of
fragmented `role:` speech lands on the right character. On the old roster the
same rule would have routed 25,339 characters *into* Bene Gesserit, which is why
the roster changes come first. `include_aliases` becomes unconditional so the
model sees the names it can answer with.

### 6. Gender

Unchanged in mechanism: roster honorifics and pronouns, then `dialogue_tags`
after attribution. Two consequences of the design: merged entries pool their
signals, and the spouse split stops one spouse's honorific outvoting the
other's pronouns. The Dune fix itself needs only a re-render.

### 7. Reasoning configuration

`_characters/llm.py` gains a per-call reasoning setting. Attribution and the
model roster keep `"none"`; the identity pass uses reasoning. The reason goes in
the code comment: attribution is extraction, identity is deliberation, and the
second failed without reasoning on every model tested.

## Identity and cache

The roster determines every id a stored attribution refers to, so a changed
roster invalidates stored attributions. This work bumps `PROMPT_VERSION` (new
roster and identity prompt) and adds the identity pass's model and reasoning
setting to `attribution_key` material, beside `roster_model_id`. Consequences to
plan for:

- Every stored attribution is re-bought once, Dune's included.
- The arm-0 baseline for any before/after comparison must be captured before
  the bump (the stored Dune render `3d156cdd…` is already that baseline).
- Grid-folds changes segment identity (`grid-v1`) independently; comparisons
  are made after it lands.

## Cost

Measured on the six books, identity pass only (attribution is separate), from
OpenRouter's reported charges:

| Model | Per book, two runs, chosen design | Per run, all-rules base |
|---|---|---|
| glm-5.3-flash, reasoning | **2.4¢ mean** (0.8¢ Persuasion – 6.4¢ The Eye of the World) | $0.0066 |
| deepseek-v4-pro, reasoning | not run on the safety-only base | $0.048–0.053 |

The chosen design costs more than the all-rules base (about 1.3¢ per book)
because the safety-only cast list is longer, so the prompt is larger.

`z-ai/glm-5.3-flash:batch` lists at half the price for offline renders.

## Evidence

Six books: Dune, The Eye of the World, Persuasion, The City & the City, Uprooted,
Foundation. Checks: **lost** (under half a principal's mentions survive),
**merged** (two principals in one entry), **apart** (named pairs that must stay
separate: Tam/Rand al'Thor, Charles Hayter/Charles Musgrove, Count/Lady Fenring,
five Two Rivers couples, …), **nickname voices** (fifteen pairs that must share a
voice: Paul/Usul, Paul/Muad'Dib, Liet/Kynes, Mat/Matrim, Bran/Brandelwyn,
Agnieszka/Nieshka, Sarkan/the Dragon, Solya/the Falcon, Tye/Borlú,
Seldon/Raven Seldon, …).

| Configuration | Safety failures | Nickname voices | Voices per principal |
|---|---|---|---|
| Production roster | fails 3 of 6 books | — | — |
| All eight rules, no model (fallback) | 0 | 3 / 15 | 1.22 |
| All eight rules + identity pass merges | 0 | 15 / 15 | 1.04 |
| **Safety rules + identity pass merges and exclusions** | **0** | **15 / 15** | **1.02** |
| Any model, reasoning off | 6–20 | — | — |

The chosen configuration removed every entry the fallback filters remove except
234 mentions' worth, and additionally removed peoples, organisations, families,
places and multi-holder roles the filters kept (Fremen, Aes Sedai, Ogier, "the
Warder", the Crofts, Anacreon). Every such removal was read; none is an
individual character.

The harness that produced every number is under `evals/attribution/`
(`roster_signals.py`, `roster_lab.py`, `roster_llm.py`, `roster_merge.py`,
`roster_rules.py`, `roster_reasoning.py`, `roster_simplify.py`,
`roster_prices.py`), reproducible from saved responses without new calls.

## Acceptance gates

- **Parity.** The production roster on the six books equals the harness's for
  the same configuration, on both the identity-pass path (with the saved model
  responses injected through the `Client` seam) and the fallback.
- **Safety.** Zero lost, merged and apart on all six books, on both paths.
- **Nicknames.** At least 14 of 15 nickname voices on the identity-pass path.
- **Fallback.** The fallback path reproduces the all-eight-rules results: zero
  safety failures, 3 of 15 nickname voices.
- **Dune re-attribution** against the stored render: Paul's voices, `role:`
  volume and its ungendered share, speech on non-character entries.
- **Project gate:** Ruff format and check, strict mypy, full pytest with at
  least 90% coverage, unit tests per rule built from the real examples above.

## Known limitations

- **Names that mean different people by scene.** "Miss Elliot" in Persuasion is
  Anne when Elizabeth is absent and Elizabeth otherwise; "the Duke" is Leto, then
  Paul. No fixed roster is right; the attribution model resolves them per
  passage.
- **Disagreeing runs leave an entry as it is.** On Dune the two identity runs
  disagreed about "the Duke" (one merged it with Leto, one excluded it), so it
  stays its own entry and voice on the identity path. Safe, but not fixed.
- **Small wrong folds in the fallback.** "Master Aybara" into Perrin (father into
  son, same gender); "Mrs Charles Musgrove" into Charles (a wife named by her
  husband's name). The identity pass maps the latter correctly to Mary.
- **The model's own answer spellings** ("paul-muaddib", "margot-lady-fenring")
  still miss answer resolution.
- **Model-roster path.** `merge_rosters` gets the `same_person` fix; the identity
  pass was tested only on the spaCy roster path and is not applied to per-chapter
  model rosters in this work.

## Out of scope

- Merging `role:` speakers across chapters (a recurring speaker is a roster
  miss, to be fixed by naming them).
- The `_with_current_roster` early return for model rosters (a latent bug; the
  Dune gender defect does not go through it).
- Replacing `dialogue_tags` or the attribution prompt.
- Consolidating `identity.PREFIX_TITLES` and `spacy_roster._TITLES`, which
  disagree; grid-folds moves the former.

## Approved decisions

- Rules prevent wrong merges; the model makes merges and judges personhood.
- Identity pass: glm-5.3-flash with reasoning, two runs, act only on agreement;
  model configurable.
- The five merge/filter rules stay as an offline-only fallback.
- No new word lists.
- Reasoning set per call; attribution stays `"none"`.
- Answer resolution by unique alias, after the roster fixes.
- `PROMPT_VERSION` bump; identity model and reasoning in the attribution key.
- Sequenced after grid-folds.

## Appendix A — identity pass prompt

```text
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
1. Paul (1678 mentions): "…" | "…" | "…"
…
```
