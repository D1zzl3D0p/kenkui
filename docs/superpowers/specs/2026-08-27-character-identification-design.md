# Character identification overhaul

Status: draft for review
Date: 2026-08-27

## Problem

Chapter 13 of *The City and the City* renders half of every conversation in
the narrator's voice. The chapter's null-attribution rate is 21.4%; the other
five chapters in the same run are 0.0%, 0.0%, 0.0%, 0.0% and 3.2%.

The nulls alternate with `narrator` in a two-hander:

```
 48 narrator  "I'm interested in archaeology."
 49 NULL      "The fuck you are. Who are you?"
 50 narrator  "I'm …"
 51 NULL      "Tye Adder Borlo."
 52 narrator  "More or less."
 53 NULL      "Police?"
```

It is a first-person novel, so unknown falls back to the narrator's voice and
the whole exchange is read by one person.

Four independent defects produce this.

### 1. The roster is a hard ceiling, and it is capped

`attribute_chapter` builds `known = frozenset(character.id for character in
characters)` (`_characters/attribution.py:135`) and `_resolve` discards any
speaker not in it. A character absent from the roster cannot be attributed by
any model at any price.

Professor Rochambeaux speaks six lines in chapter 13 and is not in the roster.
The slots went to entities that never speak: `Copula Hall` (a building),
`Ul Qomans` (a demonym), `English`, `God`, `Fuck`, `Orciny` and `Besźel`
(cities), `Breach` (an organisation).

Measured across 372 EPUBs of the user's Calibre library:

| Metric | Result |
| --- | --- |
| Books losing >=1 tagged speaker to the cap | 321 / 367 (87.5%) |
| Tagged speakers excluded per book | median 8 |
| Top-50 slots holding no speaker at all | median 31 of 50 |

Oathbringer has 218 characters with explicit dialogue tags. The cap admits 36
of them, and wastes 14 of its 50 slots on non-speakers. The median of 8 is a
**lower bound**: speakers with no name tag at all (see below) are not counted.

### 2. Roles come from a closed 34-word list

The other unknown speaker in chapter 13 is *"a militsya **officer**"*, later
*"his **colleague** said"*. `ROLE_WORDS` (`_characters/prompts.py:21`) contains
`guard`, `soldier`, `watchman`, `captain` — but not `officer`, `policeman` or
`colleague`. The model had no legal token and correctly answered unknown.

### 3. Gender is decided by proximity, not reference

`roster_spacy.py:262-268` tallies gendered pronouns in a +/-40-token window
around each name mention. Proximity is not reference. Results on this book:

```
corwi            gender=None       speech_acts=27   <- #2 speaker, no gender
inspector-borlu  gender=None       speech_acts=7    <- the narrator, no gender
mr-geary         gender=feminine   speech_acts=11   <- wrong
thacker          gender=feminine   speech_acts=7    <- wrong
```

Mr. Geary is feminine because he is never on the page without Mrs. Geary and
Corwi:

> "No, thank you, Inspector," **Mr. Geary** said. I glanced at Corwi, but
> **she** was following what was said... **her** comprehension was good.

A `None` gender falls back to the whole voice pool (`_domain/casting.py:97`),
so the two highest-volume speakers after Dhatt get a coin flip.

### 4. "Unknown" and "dropped" are indistinguishable

`_characters/attribution.py:140-142`:

```python
speakers = [answers.get(index) for index in range(len(dialogue))]
# "A model that skipped an id leaves None, which is unknown: gaps need no
# special case."
```

A model that answered "unknown" and a model that never returned the quote id
produce the same value. These are different failures with different fixes, and
nothing can currently tell them apart — which is why this needed a manual text
dive rather than a metric.

## Rejected: gating roster slots on speech evidence

The obvious fix — require a character to have a dialogue tag to hold a slot —
was tested and does not work.

Bowden has **43 attributed quotes and zero name tags**, under a 50-verb pattern
with adverbs interposed and aliases expanded. He is introduced as *"I'm David
Bowden."* and thereafter tagged only by pronouns. A tag gate deletes him.

Widening the gate to quote-adjacency recovers him and every other speaker
(13/13) but admits `Canada`, `Paris`, `Berlin`, `Copula Hall`, `English`,
`God`, `Fuck`, `Orciny`:

```
real speakers kept    : 13/13
real speakers dropped : 0
non-speakers kept     : 32      <- slots still consumed
non-speakers dropped  : 5       <- of 50 entries
```

In a dialogue-dense novel everything is near a quote. The gate is either
precise enough to free slots and drops Bowden, or inclusive enough to keep
Bowden and frees nothing. There is no working filter.

The asymmetry that settles it, at `_characters/__init__.py:121`:

```python
for character in characters
# A character nobody attributed anything to cannot be cast, and would
# otherwise consume a voice from a pool that is not deep enough to waste one.
if character.id in volume
```

A roster entry that is never attributed a quote is already dropped before
casting and consumes no voice. **A junk entry costs prompt tokens; a missing
entry costs unfixable wrong audio.** Be inclusive.

## Design

### A. Per-chapter rosters, uncapped

Attribution already runs per chapter. Send each chapter's prompt only the
characters that chapter's roster pass found, instead of the merged book-wide
roster. This removes the reason the cap exists: prompt size no longer scales
with the size of the book.

`merge_rosters` still runs book-wide, because casting and identity need one
entry per person. Only the *prompt* becomes local.

- `infer_characters` keeps its per-chapter output rather than discarding it
  after the merge.
- `attribute_chapter` receives the chapter-local roster for the prompt, and
  the book-wide merged roster for `known`, so a model that answers with a name
  from a neighbouring chapter is still accepted rather than discarded.
- Roster inference therefore remains a full pass over every chapter that
  completes before attribution begins, since `known` needs the merged result.
- With B in place, `known` no longer decides whether an answer is *usable* —
  only whether it names an established character or mints a role. An answer in
  `known` resolves to that character; an answer outside it becomes a role.
- The cap is removed. Speech-tag counts become a **ranking** signal used only
  to order the roster block within a prompt, never a filter.

### B. Roles minted from unrecognised answers

`_resolve` currently discards any answer not in `known`. Instead, mint it as a
chapter-scoped role: `role:rochambeaux@ch13`, `role:officer@ch13`.

This retires `ROLE_WORDS` as a closed list. No hand-maintained vocabulary has
to anticipate "officer"; any name the attributor returns becomes a speaker.

Two guards survive unchanged:

- The `PRONOUNS` rejection (`_characters/infer.py:15`) still applies. Minting
  `role:he@ch13` would collapse every male speaker into one voice, which is the
  exact failure that check exists to prevent.
- Slugification still applies; ids are never trusted as returned.

Cost: a role is chapter-scoped, so a character who recurs across chapters
without ever reaching a roster gets a different voice per chapter. This is
accepted. These are overwhelmingly one-scene speakers, and the failure mode is
preferable to the current one — a wrong-but-distinct voice beats collapsing
into the narrator.

### C. Gender by precedence, strongest signal first

Gender inference moves **after** attribution, because the decisive signal is
only bindable once a quote has a speaker.

Because gender is decided after attribution, `merge_rosters` no longer needs
to fold gender across chapters at all: the first-non-null rule at
`_characters/infer.py:148`, which freezes whatever the earliest chapter
guessed, is deleted rather than replaced by a vote. The model's own per-chapter
answer becomes one more signal below the tag vote, not the source of truth.

1. **Dialogue-tag pronoun vote.** For each attributed quote, read the adjacent
   tag; `"...," she said` binds a pronoun to that speaker referentially. Vote
   per character, require a 2x margin.
2. **Honorific on the full display name.** `Mr.` / `Mrs.` / `Professor`,
   matched against the full name only.
3. **Gendered role words.** `role:woman@ch3`, `old-man`, `girl` state their
   own gender and currently synthesise as `None`
   (`_characters/__init__.py:134`).
4. **spaCy pronoun window**, last resort, with a stricter gate than today.
5. `None`.

Measured on chapter 13's book, signal 1 alone:

```
character       tag votes      tag says    spaCy says   verdict
dhatt           f=0  m=17      masculine   masculine    agrees
corwi           f=15 m=0       feminine    None         FILLS gap
bowden          f=1  m=9       masculine   masculine    agrees
mr-khurusch     f=0  m=8       masculine   None         FILLS gap
drodin          f=0  m=4       masculine   None         FILLS gap
mrs-geary       f=3  m=0       feminine    feminine     agrees
mr-geary        f=0  m=1       masculine   feminine     FIXES spaCy
nancy           f=1  m=0       feminine    None         FILLS gap
```

Ten characters decided, zero wrong, five gaps filled, one outright fix. Only
61 of 983 quotes carry a pronoun tag, but the votes concentrate on the
high-volume speakers whose voice is most audible.

**Honorifics must match full names only.** Matched against bare surnames they
tag Corwi — a woman — as masculine, off a line where another character
misaddresses her:

> "Oh yes, you are, you're **Mr. Corwi** are you, is that—"

### D. Coreference across the book

Per-chapter rosters introduce the risk that one person appears under different
surface names in different chapters and is cast twice. `_characters/identity.py`
already handles most of this and is kept: `same_person` merges
`Inspector Borlú` with `Tyador Borlú` by title-aware token nesting, and
`detect_titles` finds a book's invented honorifics.

**Intent: these must resolve to one character with one voice.** `Borlú`,
`Inspector Borlú`, and the first-person `I` are the same man and must not be
cast separately. (Flagging explicitly for review — the request could be read
the other way.)

Remaining gaps:

- **Variant forms that are not token-nested.** `Tyad` against `Tyador`,
  `Lizzy` against `Elizabeth`. `resolve_short_forms` requires an exact token
  match, so these become separate entities. Needs conservative prefix matching,
  keeping the module's existing bias: under-merge rather than over-merge.
- **Alias matching.** The roster records `Mr. Khurusch` while the text says
  `Khurusch said`. Both the honorific check and the tag-binding in C break on
  this. Aliases must be first-class, not derived ad hoc at each call site.
- **First-person narrator linkage.** `narration.py` attributes first-person
  dialogue to the narrator; this must be bound to the narrator's *named*
  roster entry so "I" and "Borlú" share one voice.

### E. Separate unknown from dropped

`attribute_chapter` returns a value distinguishing "model answered unknown"
from "model never returned this quote id". Unknown is a legitimate answer;
dropped is a defect that should be visible and, where cheap, retried.

### F. spaCy as an optional extra

spaCy and `en_core_web_*` are installed in `.venv` but declared nowhere in
`pyproject.toml`. Declare as `kenkui[characters]`, not a hard dependency —
the models are large downloads. `infer_characters()` raises a clear error when
the extra is absent.

## Testing

- Unit tests per component, TDD, following existing test layout.
- The eval harness gains a gender scorer so the precedence chain in C produces
  a measured before/after rather than an assertion.
- Chapter 13 of *The City and the City* becomes a regression fixture: its
  null rate must drop and Rochambeaux must be attributable.
- The Calibre library sweep is repeatable for the roster-coverage numbers.

## Out of scope

Text normalization flags, and series continuity. Series depends on D and gets
its own spec.
