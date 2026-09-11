# Coreference and Gender — Design

**Status:** Superseded on 2026-09-11 by `2026-09-11-roster-identity-design.md`.
Measurement showed the dominant defects were over-merge bugs in the roster
(`identity.same_person`, title stripping), not unmergeable aliases. Kept for
the record of how the investigation started.

**Relates to:** `2026-08-27-character-identification-design.md` (defect 3,
"coreference under-merges"), `2026-09-05-chunking-and-gender-fixes.md`
(the current gender signal), **depends on** `2026-09-10-grid-folds-design.md`
(see Sequencing).

## Goal

Give one character one voice, and give that voice the right gender. Both are
currently unmeasured, and neither can be fixed honestly until they are.

Two observed defects on Dune, both audible in a completed render:

- **Paul has four voices.** `Paul`, `Paul Atreides`, `Usul`, and `Muad'Dib`
  become separate cast entries because `identity.py` merges names only by
  token nesting after title-stripping. Names sharing no tokens never fold.
- **Captain Nefud is misgendered.** A named character (`Iakin Nefud`, the
  Baron's guard captain after Piter) is cast against his gender, despite
  unambiguous pronoun evidence in the text. Whether he reached the roster at
  all is unverified and is the first thing to check: if he did not, his lines
  went to a `role:` id or to unknown, and the diagnosis changes entirely.

## Context: what is measured today, and what is not

`evals/attribution/score.py` scores quote attribution against consensus gold:
precision, coverage, `wrong_per_1k`. That is the only quality number the
project has.

`score_gender.py` compares a cast voice against the character's *inferred*
gender. It cannot observe a wrong inference — a confidently wrong gender
matched to a matching voice scores clean. This is why Nefud was inaudible to
the harness and audible in the render.

There is no metric at all for whether one person got one voice.

The under-merge behaviour is deliberate and documented
(`evals/attribution/identity.py`, "Under-merge stays"), recording four
rejected routes: complementary distribution, concentration, a direct model
ask, and a risk gate. Under-merging was judged the cheaper error. Dune is the
case that makes it not cheap, so the decision is reopened — but only against
a lever none of those four used.

## Preliminary measurement

A crude approximation of the shipped nearest-following-pronoun vote
(`spacy_roster.py`, `_GENDER_LOOKAHEAD = 12`, stop at an intervening proper
noun, `_GENDER_MINIMUM = 3`, `_GENDER_MARGIN = 2`) run over Dune's extracted
text:

| name | fem | masc | stopped at proper noun | verdict |
|---|---|---|---|---|
| Nefud | 0 | 7 | 44 | masculine |
| Piter | 1 | 17 | 119 | masculine |
| Feyd | 0 | 1 | 230 | **abstain** |
| Paul | 106 | 489 | 1016 | masculine |
| Jessica | 245 | 75 | 532 | feminine |
| Usul | 4 | 6 | 45 | **abstain** |
| Chani | 50 | 36 | 161 | **abstain** |

**This is an approximation, not a result.** It uses capitalisation as a proxy
for proper-noun detection and a regex tokenizer, not spaCy. The real pipeline
must be run before any of it is trusted. It is recorded because it generates a
sharper hypothesis than the one this design started from.

The hypothesis: **the dominant gender defect is abstention on named
characters, not misinference.** The proper-noun stop rule fires on the
overwhelming majority of mentions (Feyd-Rautha is almost always followed by a
capitalised token; 230 stops against 1 vote). An abstaining character reaches
`casting.candidates`, which answers `gender is None` with the *whole pool*
(`_domain/casting.py:118`), and load-balancing hands out a coin flip. Chani
fails on margin rather than volume — 50/36 does not clear 2:1.

Three consequences for the design:

1. Nefud's own votes look adequate, so his defect may lie in vote *splitting*
   across surface forms (`Nefud`, `Iakin Nefud`, `Captain Nefud`) rather than
   in the signal. That makes it partly a coreference defect, not purely a
   gender one — which is the strongest available argument for treating these
   two problems as one project.
2. The escalation gender pass prototyped in `evals/attribution/escalate.py`
   covers **role speakers only** — its prompt says "Omit it for `roster`". It
   would not have fixed Nefud. Named-character gender needs its own arm.
3. `identity.py`'s prefix-title handling already folds `Captain Nefud` into
   `Nefud`. Whether the *gender vote* is keyed on the folded entity or on the
   raw surface name is the thing to check first, and it is free to check.

## Sequencing

**This work starts only after grid-folds lands.** Decided on 2026-09-10.

`2026-09-10-grid-folds-design.md` is an atomic migration that moves prefix-title
data out of `_characters/identity.py` into `_domain`, makes the grid own quote
boundaries, migrates attribution to grid-provided dialogue ranges, and deletes
the legacy quote module. This design modifies `identity.py`, `merge_rosters`,
both prompts, and the attribution store key. The overlap is direct, and running
the two in parallel means hand-resolving a file move inside the component whose
correctness this work is trying to measure.

Sequencing has three concrete consequences for the plan:

1. **Prefix-title constants will have moved.** `PREFIX_TITLES` and
   `detect_titles` are relocating to `_domain`. Cluster composition must import
   them from wherever grid-folds leaves them, not from `_characters.identity`.
2. **Span offsets change identity.** Grid-folds ships `grid-v1` and
   deliberately changes segment identities. Any baseline captured before it
   lands is not comparable afterwards, so **arm 0 must be captured on
   post-grid-folds `main`**, not before. This overrides the ordering note in
   section 4, which concerns only the `PROMPT_VERSION` bump within this work.
3. **`attribute_chapter` will read grid dialogue ranges** rather than rescanning
   quotes. The eval's `identity_run.py` must follow the same path, or it
   measures a code path production no longer takes.

The one part not blocked: gold-standard construction (section 1) reads chapter
text and needs no pipeline internals. It can proceed in parallel and is the
long pole, since it involves hand adjudication.

## Scope

In scope:

- a gold standard of character clusters and per-cluster gender, on Dune;
- volume-weighted cluster metrics reporting splits and merges separately;
- an eval runner that emits a partition rather than a ballot;
- measured arms for coreference and for gender, including cost per arm;
- prompt-layout and task-packing arms, to measure passes over the text;
- shipping whatever clears the acceptance gates below.

Out of scope:

- merging `role:` speakers across chapters (see Approved decisions);
- replacing `spacy_roster`, `dialogue_tags`, or `attribution` wholesale with
  BookNLP, even if BookNLP scores well — that is a larger question;
- changes to casting, which needs none;
- the human-review path, which already exists and only needs documenting.

## 1. Gold standard and metrics

**Score the partition, not the labels.** Gold is a set of clusters of surface
names: `{Paul, Paul Atreides, Usul, Muad'Dib}` is one cluster. Not ids.

This is what makes models comparable without forcing a shared ballot. The
existing harness had to build one spaCy roster because "two models scored
against two different id vocabularies cannot be compared" — one says
`elizabeth-bennet`, the other `lizzy`. Scoring a partition dissolves that: two
models that group the same names score identically whatever they call the
group. The `roster_fingerprint` discipline in `run.py` stays where it is and
continues to guard the attribution eval.

**Weight by spoken volume.** Standard coreference metrics treat every mention
alike. A split costing Paul 40% of his lines and a split on a two-line servant
are not the same defect. `CharacterProfile.spoken_characters` carries the
weight. Report unweighted B-cubed for comparability with the coreference
literature, and volume-weighted as the number that predicts what a listener
hears.

**Report the two error directions separately.**

- **splits** — one person, several voices. Currently unmeasured.
- **merges** — several people, one voice. The failure the whole attribution
  design is organised against.

A single headline number would hide a method that fixes splits by regressing
merges, which is a loss however good the aggregate looks.

**Construction.** Same discipline as `gold.py`: consensus across strong
models, disagreements written out for hand adjudication, nothing guessed. Two
departures. Seed the set with the known-broken cases so the harness is proven
to catch them. And require whole-cluster agreement rather than pairwise votes,
or the gold set can be intransitive.

Scope of the labelling: speaking characters only, and only those with more
than one surface form. On Dune this is tens of clusters, not hundreds.

**Gender gold rides along** — `{cluster -> masculine | feminine |
genuinely-unstated}` from the same pass. `score_gender.py` then gains the
question it cannot currently ask: was the inference *right*.

## 2. Passage-passes as a measured resource

Production spends two full passes over every dialogue-bearing chapter:
`_roster_for` (`__init__.py:112`) and `attribute_chapter`
(`attribution.py:167`). Chapters without dialogue are already skipped. Every
proposal is quoted as a delta against that baseline.

**Not every task needs a pass.** Sorting tasks into these classes matters more
than any caching question:

- **Class A — needs the whole passage.** Attribution, roster discovery.
  Irreducible.
- **Class B — needs local windows only.** Gender from the text around a
  character's own quotes. `dialogue_tags.py` already demonstrates the signal is
  there in +/-120 characters, using a regex.
- **Class C — needs no book text.** Alias reconciliation, once per-chapter
  passes have emitted candidate pairs *with citation sentences*. Order 2k
  tokens for a whole book.

If Class C holds, coreference costs approximately nothing per book. That is a
hypothesis to be measured, not assumed.

**Prompt layout blocks the caching question.** Both prompts place the passage
*after* the instructions — `ROSTER_PROMPT` ends with it (`prompts.py:58`), and
`ATTRIBUTION_PROMPT` puts the per-chapter roster before it (`prompts.py:70`).
Prefix caching pays only when shared content sits at the front, so as written
the roster call and the attribution call for one chapter share no cacheable
prefix on any provider. Moving the passage to the front makes the question
measurable. It is a `PROMPT_VERSION` bump; see section 4.

Two things that will mislead if unmeasured: cache TTLs are minutes on most
providers, and `_attribute_chapters` runs chapters concurrently, so whether a
chapter's roster call lands before its attribution call is a scheduling
question rather than a given.

**The harness cannot currently see a cache hit.** `client.py:179` records
`prompt_tokens` and `completion_tokens` but not
`prompt_tokens_details.cached_tokens`. Adding it is a prerequisite, or caching
is inferred from cost and conflated with price changes.

**Axes.** Task packing (separate / gender folded into attribution / aliases
folded into roster / all-in-one), layout (passage-last, passage-first), and
model. Reported per arm: dollars per book, passes per book, cached-token
share, latency, alongside the section 1 quality numbers.

Full cross-product is waste. Fix one cheap workhorse model, sweep packing and
layout, then sweep models across only the packings that survive.

The failure this matrix exists to catch: packing several tasks into one call
degrading attribution precision. A combined call that saves 40% and costs two
points of precision is a bad trade, and only a matrix shows it.

## 3. Arms

### Harness shape

```
prepare_book.py   EPUB ──► work/chapters.json, quotes.json      free
identity_run.py   arm ───► runs/<arm>/clusters.json             $ varies
gold_clusters.py  runs ──► work/gold-clusters.json              $ once
score_identity.py ───────► splits / merges, weighted and raw    free
```

`score.py` and the attribution path are untouched. One stored attribution run
is held fixed beneath every coreference and gender arm, so an arm cannot
flatter itself by shifting attribution underneath the volume weights.

### Coreference

0. **Baseline, production.** `_model_roster` through `merge_rosters` and
   `identity.py`. This is what gives Paul four voices; without it there is no
   improvement to claim.
0b. **Baseline, spaCy.** `spacy_roster.infer_roster` whole-book. A different
   path with real `chapter_ids`, so a separate arm.
1. **Cited-alias.** Per-chapter roster call emits alias candidates with a
   verbatim citation span; one Class-C reconciliation call over pairs and
   citations produces clusters.
2. **Uncited ask.** The already-rejected route, run deliberately as a control.
   It produced the `dai-shan` into `agelmar` merge. If arm 1 does not beat it,
   the citation gate is ceremony and should not be built.
3. **Coreference model.** BookNLP, plus one OntoNotes-trained option
   (fastcoref or maverick) for contrast. Requires a projection from mention
   clusters to name clusters. `torch 2.13` is already a hard transitive
   dependency via `pocket-tts` (`uv.lock:3176`), so the dependency objection
   recorded in the eval README no longer binds.
4. **Whole-book single call.** One long-context model, entire book, clusters
   out. Almost certainly not shippable. Run once as a **ceiling**: it measures
   what the cheap arms leave on the table.

**Modifier, not an arm:** the chapter-disjoint gate, applied over arms 1-4 as
a second scored variant of each. It cannot help Paul, who shares chapters with
Muad'Dib throughout; it is the safety net for low-confidence merges elsewhere.

### The citation gate

The untried lever, specified precisely enough to fail honestly. The model
returns `{alias, canonical, evidence}` where `evidence` is claimed verbatim
book text.

- **Tier 1 — both names in one verified span.** `evidence` occurs literally in
  the chapter after whitespace normalisation, and contains both surface forms.
  `Paul-Muad'Dib` lands here.
- **Tier 2 — verified naming event.** `evidence` occurs literally, contains
  the alias, and the canonical name is the nearest preceding named entity
  within a proximity window. That window is a swept parameter, not a constant
  to be guessed; the plan states the values tried. This is the tier that
  catches *"You shall be known
  among us as Usul"*, which contains no "Paul" and fails Tier 1. **Tier 2 is
  doing the work for the actual case, and is the tier most likely to leak.**
- **Tier 3 — unverifiable.** No literal match. Falls to the chapter-disjoint
  gate: merge only where being wrong is inaudible, otherwise refuse.

Score each tier's precision separately. Track the **hallucinated-citation
rate** per model — the share of proposed spans that do not occur in the text.
It falls out for free and is a sharp model-quality discriminator the harness
does not currently have.

Expected shape: Tier 1 near-perfect and rare, Tier 2 decisive, Tier 3 mostly
refusals. If Tier 2 precision is poor, arm 1 collapses into arm 2 and
"under-merge stays" is re-confirmed — a legitimate outcome, reached cheaply.

### Gender

Ordered by the corrected hypothesis: named-character abstention first.

0. **Baseline.** `spacy_roster` pronoun and honorific votes, then
   `dialogue_tags.apply` override, then `ROLE_GENDERS` for role speakers.
0b. **Baseline diagnostic, free.** Did Nefud, Feyd and Chani reach the roster
   at all, and under which surface forms? Is the vote keyed on the folded
   entity or the raw surface name? Does relaxing the proper-noun stop or the 2:1 margin
   resolve Feyd and Chani without breaking the two-hander case the current
   values were tuned for? Costs nothing and may resolve much of the defect.
1. **Named-character resolution, Class B or C.** After attribution, for each
   rostered character with `gender is None` and spoken volume above a
   threshold (swept, not guessed), gather their quote windows and ask once. Small, and it is the arm aimed at
   Nefud, Feyd and Chani.
2. **Gender folded into the attribution prompt.** Zero extra passes.
3. **Role speakers via the escalation pass**, ported from `escalate.py`
   largely as-is. Covers roles only. Nearest thing here to already-done work:
   31 of 32 role speakers gendered for $0.046 on one book.
4. **BookNLP referential gender**, if coreference arm 3 is running anyway.

**Gender is conditional on coreference.** Split Paul four ways and each
fragment is gendered independently from different evidence. Gender is
therefore scored per gold cluster on the shared substrate, and a coreference
arm may improve gender scores without any gender arm running.

**Attribution confounds both.** `dialogue_tags.apply` can turn a correct
gender wrong when a quote is misattributed — it logs
`dialogue_tag_gender_conflict` for exactly this — and volume weighting depends
on attribution. Hence the fixed attribution run above.

## 4. What ships

**Clusters compose onto `identity.py`, never replace it.** `identity.py` is
pure, cheap, tested, and its merges are the safe ones. The rule: a cluster may
only *add* a fold, never dissolve one `identity.py` made. Arm 0 therefore
remains a strict lower bound, and a bad reconciliation degrades to today's
output rather than to something worse. The confidence tier travels with the
merge.

**`aliases` stops being decorative.** `include_aliases=supplied_roster`
(`__init__.py:514`) becomes unconditional. `_roster_block` already formats
them (`attribution.py:69`), and `series.py` already matches on
`(*character.aliases, character.display_name)` (lines 47-53, 293-319), so
cross-book continuity improves at no cost. This is plausibly a win independent
of everything else and is worth its own arm.

**Cache invalidation, stated before it bites.** `attribution_key`
(`store.py:170`) includes `prompt_version` deliberately. `characters-v5` to
`v6` discards every stored attribution, Dune's included. So: capture the arm-0
baseline *before* the bump or it is not reproducible. And `roster_model_id`
enters the key only when it differs from `model_id` (`store.py:201`) — if
alias reconciliation uses a third model, that is new material determining
content and must enter the key, or stale clusters are reused silently.

**The warning that would have caught Nefud.** `ungendered_pool_characters`
deliberately excludes `gender is None`: "the whole pool is the designed answer
for those, not a degradation." That is correct while inference is weak and
`None` is honest. It inverts once inference is meant to be good. Leave that
function's contract alone and add a separate reporter for **ungendered spoken
volume** — the number `score_gender.py` prints and production never has.

**Role speakers** take model-supplied gender first, `ROLE_GENDERS` as
fallback, `None` as the floor.

**Casting needs no change**, and improves anyway: `solve` is greedy colouring
over co-occurrence, so merging clusters reduces the character count, loosens
the constraint graph, and spreads a small pool further.

**The human escape hatch already exists.** `review.validate_roster` accepts a
caller-supplied roster with aliases and `resolve_attribution` honours
`reviewed=True`. A wrong cluster is correctable without building anything, and
correctable *after* the fact — which sidesteps the spoiler objection that
killed the ask-the-reader route. Document it; do not build it.

## Acceptance gates

Written before any number exists to rationalise against.

- **Merges must not regress.** Over-merge is the failure the design is
  organised around. Any increase fails an arm outright, whatever it does for
  splits.
- **Splits improve on the volume-weighted score**, with Paul specifically
  resolved to one cluster.
- **Attribution precision unchanged**, held fixed underneath — a check that
  the harness is honest, not a target.
- **Tier 2 citation precision clears a threshold stated in the plan**, chosen
  before the run.
- **Cost per book within a stated budget**, reported as passes and dollars.
- **Project gate:** Ruff format and check, strict mypy, full pytest at 90%
  coverage, and a Dune smoke test.

**Fallback, agreed in advance.** The gender arms are near-certain wins; the
coreference arms may not pay. Ship gender and aliases-to-attribution
regardless. Coreference ships only if Tier 2 clears. "Under-merge stays"
remains a legitimate outcome.

## Approved decisions

- Score partitions, not ballots; keep `roster_fingerprint` for attribution.
- Volume-weighted metrics, splits and merges reported separately.
- Gold by strong-model consensus plus hand adjudication, seeded with the known
  defects, requiring whole-cluster agreement.
- Clusters compose onto `identity.py` and may only add folds.
- Hold one attribution run fixed beneath all coreference and gender arms.
- Stage the model sweep rather than running the full cross-product.
- **`role:` speakers stay chapter-scoped and are never merged across
  chapters.** A recurring unnamed speaker is a roster failure to be fixed by
  naming the character, not a coreference problem to be fixed by merging role
  ids. This confirms the existing behaviour at `attribution.py:117-121`.
- Do not replace existing components with BookNLP within this work, whatever
  it scores.
- **Sequence after grid-folds**, with gold-standard construction the one
  workstream allowed to start in parallel. See Sequencing.

## Risks and open questions

**Consensus gold has the usual blind spot.** An error every adjudicator shares
is invisible, so measured cluster accuracy is an upper bound. Spot-check the
agreed labels as well as the disagreements.

**BookNLP numbers on this corpus are unknown.** No benchmark figures are
quoted in this document deliberately; the arm exists to produce them.

**The preliminary gender table is an approximation** and must be reproduced
with the real `spacy_roster` pipeline before any of it is relied on.

**Dune parses only after `strip_doctype`**, in common with 32 of 376 books in
the local library. The underlying parser issue (`_epub/parser.py:132`,
`forbid_dtd=True`) is a known blocker recorded in the eval README and is not
addressed here.
