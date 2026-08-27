# Character identification and gendered casting

Status: draft for review (rewritten 2026-08-27 against production data)
Date: 2026-08-27

## How this spec was derived

An earlier draft was written from `evals/attribution/` artifacts. Those are
produced by the eval harness, which **substitutes its own spaCy roster capped
at 50 characters** for kenkui's LLM roster pass (`evals/attribution/run.py:4`,
`run.py:88`). Conclusions drawn from them did not describe kenkui.

Everything below is measured against the real production run in
`~/Library/Caches/kenkui/v1/casting.sqlite3`, attribution
`fe2257386971a24c30298d9e054ceb66253633b1a2aaa8bb0d936bf160db5cd3`, model
`openrouter/deepseek/deepseek-v4-flash`, cast method `gendered`.

Findings from the eval harness that do **not** apply to kenkui, recorded so
they are not re-derived: the roster cap and the missing-Rochambeaux ceiling
(kenkui's roster holds 72 characters and includes `bernard-rochambeaux`); the
"87.5% of the library loses speakers" and "31 of 50 slots" figures; and the
proximity-window gender errors (`mr-geary=feminine`). kenkui genders 56 of its
72 characters and gets Corwi, Borlú, Nancy and Bowden right.

## Problem

### Defect 1: gendered casting is inoperative (root cause of the 30% complaint)

`_domain/casting.py:98-102`:

```python
matched = tuple(voice for voice in pool if voice.perceived_gender == character.gender)
return matched or pool
```

96 of the 98 voices in the installed manifest carry no `perceived_gender`, so
`matched` is always empty and every character falls back to the entire pool.
`method="gendered"` has been behaving exactly as `method="random"`.

The cause is an **id-space mismatch**, not missing data:

| Source | Count | Id form | Gender |
| --- | --- | --- | --- |
| `kenkui-voices/voices.json` | 95 | `alasdair-m-vctk-p246-scottish` | 50 Male / 45 Female |
| bundled `src/kenkui/voices/pack.json` | 95 | identical ids, identical set | 50 Male / 45 Female |
| installed manifest | 98 | `alasdair`, `variety: "built-in"` | absent on 96 |

`registry.py:149` maps `_GENDERS.get(voice["gender"])` correctly, but it never
runs against these entries because the manifest's short ids never join the
catalog's long ids. The assets are the same files: **95 of 98 manifest entries
join `kenkui-voices` on `asset_sha256`**, recovering the full 50/45 split.

The user-visible result, from the real cast:

```
corwi          (feminine) -> aoife    = Female  ok
lizbyet-corwi  (feminine) -> declan   = Male    wrong
```

No improvement to gender *inference* can affect this. The voice side has no
gender to match against.

### Defect 2: the fallback is silent

`return matched or pool` degrades a gendered cast to a random one with no
warning, no event, and no validation failure. This defect is why Defect 1 went
unnoticed; it must be fixed even after Defect 1 is repaired.

### Defect 3: coreference under-merges, giving one person several voices

From the real cast:

```
inspector-borlu -> alasdair      tye                    -> mateo
corwi           -> aoife         lizbyet-corwi          -> declan
dhatt           -> alf           senior-detective-dhatt -> garrett
```

Three people, six voices. `_characters/identity.py` is sound in design —
`same_person` merges `Inspector Borlú` with `Tyador Borlú` by title-aware token
nesting, and `detect_titles` finds a book's invented honorifics — but these
pairs still escaped it.

The split copy also loses gender: `tye` carries 6,703 spoken characters and no
gender at all, the single largest ungendered entry in the book. So under-merge
feeds Defect 1 as well: a duplicate with `gender=None` takes the whole-pool
path at `casting.py:97` even after Defect 1 is fixed.

**Intent: `Borlú`, `Inspector Borlú`, and the first-person `I` are one
character with one voice.** Confirmed at spec review.

### Defect 4: gendered role words are synthesised as ungendered

`_characters/__init__.py:134` sets `gender=None` on every minted role. Seven of
the fifteen largest ungendered entries in the production run state their own
gender:

```
role:woman@ch-v1-719dd9675fc30dab9576f4fa        112 spoken characters
role:woman@ch-v1-aa1188b5e6b7a16af226ce90        107
role:man@ch-v1-95ccb0ce684321ce1e8fc1d7          128
role:woman@ch-v1-4e7a46a2e3ae5a4c57210dc1         73
role:man@ch-v1-f0666bc616848243a9fece48           50
role:young-woman@ch-v1-95ccb0ce684321ce1e8fc1d7   47
role:young-man@ch-v1-f0666bc616848243a9fece48     18
```

### Defect 5: roles come from a closed 34-word list

`ROLE_WORDS` (`_characters/prompts.py:21`) contains `guard`, `soldier`,
`watchman`, `captain` — but not `officer`, `policeman` or `colleague`. In
chapter 13 the speaker is *"a militsya **officer**"*, later *"his **colleague**
said"*. The model had no legal token and correctly answered unknown.

### Defect 6: "unknown" and "dropped" are indistinguishable

`_characters/attribution.py:140-142`:

```python
speakers = [answers.get(index) for index in range(len(dialogue))]
# "A model that skipped an id leaves None, which is unknown: gaps need no
# special case."
```

A model that answered "unknown" and a model that never returned the quote id
produce the same value. Different failures, different fixes, currently
indistinguishable.

### Defect 7: unknown falls back to the narrator's voice

`pipeline.py:225`: `unknown_voice_id=_voice_id(unknown) if unknown else
narrator_id`. The production cast has `unknown_voice_id = eponine =
narrator_voice_id`.

Dialogue-only unattributed rates in the real run, over the four chapters whose
text is available for span re-extraction:

| Chapter | Dialogue spans | Unattributed | Rate |
| --- | --- | --- | --- |
| `ch-v1-eed37b04` | 138 | 0 | 0.0% |
| `ch-v1-c5832b06` | 164 | 0 | 0.0% |
| `ch-v1-b2f9a779` | 156 | 6 | 3.8% |
| `ch-v1-6e416f89` | 168 | 8 | 4.8% |
| `ch-v1-e3439709` (ch. 13) | 201 | 22 | **10.9%** |
| `ch-v1-494f3c7a` (ch. 7) | 156 | 62 | **39.7%** |

In a first-person novel this reads as the narrator speaking both halves of a
conversation.

## Design

### A. Recover voice gender, and stop degrading silently

1. **Backfill by content hash.** A migration joins existing manifest entries to
   the bundled catalog on `asset_sha256` and writes `perceived_gender` where it
   resolves. 95 of 98 entries recover; the remainder stay `None`.
2. **Fix provisioning.** Voices loaded from `kenkui-voices` through
   `load_voice` must retain the catalog id, or carry the catalog's rights and
   `perceived_gender` when registered under a short alias. A voice that is
   byte-identical to a catalog asset must never land in the manifest without
   its catalog metadata.
3. **Refuse to degrade quietly.** When `method="gendered"` and a character with
   a known gender has no gender-matching voice in the pool, emit a `Warning`
   event naming the character and the empty pool. `validate()` reports it as a
   `ValidationIssue` before any model call is made or any audio is rendered.
   `matched or pool` stays as the runtime behaviour — dropping the speech
   would be worse — but it stops being invisible.

This is the whole of complaint 1. Items B-F are independently real and worth
fixing, but none of them changes a voice assignment until A lands.

### B. Coreference: close the under-merge gaps

Keep `identity.py`; its conservative bias (under-merge over over-merge) is
correct and the fix must not invert it.

- **Non-nested variants.** `Tyad`/`Tyador`, `Tye`/`Tyador`. `resolve_short_forms`
  requires an exact token match, so these become separate entities. Needs
  conservative prefix matching against a single unambiguous host.
- **Occupational prefixes.** `Senior Detective Dhatt` against `Dhatt`.
  `detect_titles` uses `threshold=3`, so a title appearing on one name is never
  learned. Lower the threshold or seed from a chapter-local pass.
- **First-class aliases.** `CharacterProfile` gains an `aliases` field. The
  roster records `Mr. Khurusch` while the text says `Khurusch said`; both the
  honorific check in D and any tag binding break on this today, and each call
  site currently re-derives surface forms ad hoc.
- **Narrator linkage.** `narration.py` attributes first-person dialogue to the
  narrator; bind that to the narrator's named roster entry so `I` and `Borlú`
  share one voice and one gender.

### C. Roles minted from unrecognised answers

`_resolve` discards any answer not in `known`. Instead mint it as a
chapter-scoped role: `role:rochambeaux@ch13`, `role:officer@ch13`. This retires
`ROLE_WORDS` as a closed list; no hand-maintained vocabulary has to anticipate
"officer".

Guards that survive unchanged: the `PRONOUNS` rejection (`infer.py:15`), or
`role:he@ch13` would collapse every male speaker into one voice; and
slugification, so ids are never trusted as returned.

Accepted cost: a role is chapter-scoped, so a character recurring across
chapters without reaching a roster gets a different voice per chapter. These
are overwhelmingly one-scene speakers, and a wrong-but-distinct voice beats
collapsing into the narrator.

### D. Gender for entries that still lack it

After A and B, the remaining ungendered entries are minted roles and genuinely
unnamed speakers. In precedence order:

1. **Gendered role words.** `woman`, `man`, `old-man`, `young-woman`, `girl`,
   `boy` state their gender; map directly rather than synthesising `None`.
2. **Honorific on the full display name.** `Mr.`, `Mrs.`, `Professor`. Matched
   against the **full name only** — against bare surnames it tags Corwi, a
   woman, as masculine off a line where another character misaddresses her:
   *"you're **Mr. Corwi** are you"*.
3. **Dialogue-tag pronoun vote.** For each attributed quote, read the adjacent
   tag; `"...," she said` binds a pronoun to that speaker referentially. Vote
   per character, require a 2x margin.

Because gender is settled after attribution, the first-non-null fold at
`infer.py:148` — which freezes whatever the earliest chapter guessed — is
deleted rather than replaced by a vote.

spaCy is **not** required for any of the above. It is deferred until measurement
shows 1-3 leaving a material gap, and would then be an optional
`kenkui[characters]` extra rather than a hard dependency.

### E. Separate unknown from dropped

`attribute_chapter` returns a value distinguishing "model answered unknown"
from "model never returned this quote id". Unknown is a legitimate answer;
dropped is a defect that should be visible and, where cheap, retried. Chapter
7's 39.7% is unexplained until these are separable.

## Testing

- Unit tests per component, TDD, following the existing test layout.
- A regression test asserting that a gendered cast over the bundled pack
  assigns no masculine voice to a feminine character — the assertion that
  would have caught Defect 1.
- Fixtures drawn from the production run: `corwi`/`lizbyet-corwi` must merge to
  one character with one voice; `role:woman@...` must carry `feminine`.
- Chapter 13's unattributed rate must fall; chapter 7 is investigated once E
  makes its failure mode legible.

## Out of scope

Text normalization flags. Series continuity, which depends on B and gets its
own spec.
