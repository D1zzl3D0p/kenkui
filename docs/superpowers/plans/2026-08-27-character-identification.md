# Character Identification and Gendered Casting Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make gendered casting actually run, stop one character being cast as several, and give every speaker the text identifies a voice of their own.

**Architecture:** Five independent repairs against one subsystem. Task 1 restores the gender the voice catalog already holds but `list_voices` discards — the whole of the "gender is wrong 30% of the time" complaint. Tasks 2-3 stop coreference splitting one person into several roster entries with several voices. Tasks 4-6 give unattributed and role-named speakers correct identities and genders. Task 7 makes attribution failures legible.

**Tech Stack:** Python 3.11-3.13, pytest, ruff, mypy. No new runtime dependencies.

**Spec:** `docs/superpowers/specs/2026-08-27-character-identification-design.md`

## Global Constraints

- Python `>=3.11,<3.14`. No new runtime dependencies; spaCy is explicitly out of scope.
- Voice ids stay **short** (`alasdair`, not `alasdair-m-vctk-p246-scottish`). No renaming: stored `casts` and `cast_assignments` rows must stay valid.
- `PROMPT_VERSION` in `src/kenkui/_characters/prompts.py` must be bumped whenever any prompt text changes, or stored results from the old wording are reused silently under the new one.
- `identity.py` keeps its stated bias: under-merge (one person, two voices) is preferred to over-merge (two people, one voice). No change may invert this.
- Run `.venv/bin/pytest` from the repo root. Lint with `.venv/bin/ruff check src tests` and `.venv/bin/mypy src`.

## Status (2026-08-27, complete)

| Task | Outcome |
| --- | --- |
| 1 Restore catalog gender | done, `bccc634` |
| 2 Persist catalog gender | done, `7c07477` — premise refined, see below |
| 3 Report an unhonourable cast | done, `bb0dcf8` — logged, not validated, see below |
| 4-5 Coreference merging | **replaced**, `f95ac14` — see below |
| 6 Role-word gender | done, `87a2863` |
| 7 Mint roles | done, `5eb1a0a` |
| 8 Dropped vs unknown | done, `d080486` |
| 9 Verify | done — result below |
| 10-12 Aliases, honorific, tag vote | **closed unimplemented** by Task 9's gate |

**Result, re-solving the stored production cast against the fixed pool
(pure, no model spend):**

```
before : matched 29   opposite 27   ungendered 16
after  : matched 56   opposite  0   ungendered 16
```

**Deviations.** Task 2's premise was wrong: `_registered_from_catalog`
already stamps the trait on a first registration and `add_voice` rejects
catalog ids, so the real gap was re-loading an existing ungendered record.
Task 3 dropped `ErrorCode.GENDERED_POOL_EMPTY` and the `validate()` issue,
because `validate()` runs before any model call and has no character list;
it follows `_log_collisions` instead. Tasks 4 and 5 were built on a false
premise — `Tye` is not a prefix of `Tyador`, and `merge_rosters` already
handled the rank-word case — so both were replaced by one task fixing the
actual cause, the `contested` rule in `merge_rosters`.

**Why 10-12 are closed.** Task 9 Step 5 measured the residual after the
committed fixes: 7 characters, 7,426 spoken characters, 3.34% of all cast
speech. `tye` alone is 6,703 of that — 90% — and it is Borlú under a second
id, a coreference failure that neither an honorific nor a dialogue-tag vote
repairs. Excluding it the residual is 0.33%. Implementing two speculative
signals for that is not worth the surface area.

**Known and unfixed.** `tye` still splits the narrator into two voices. No
deterministic name rule reaches it: `Tyador` and `Tye` share only `Ty`, and
a rule loose enough to merge them would merge genuinely different names. It
needs the roster prompt to return aliases, or an explicit merge pass, and
is a separate decision.

**Note.** `evals/` is gitignored, so `evals/attribution/score_gender.py`
and the `attribute_chapter` arity fix in `evals/attribution/run.py` exist
locally but are not committed.

---

### Task 1: Restore catalog gender to enumerated voices

`registry.CATALOG` holds 121 entries (50 masculine, 45 feminine, 26 unsourced) keyed by the same short ids the manifest uses, and already knows the gender of 95 of 98 installed voices. `list_voices` seeds `known` from `CATALOG` and then overwrites each manifest entry with a view that reads `perceived_gender` from the manifest record alone, discarding it.

**Files:**
- Modify: `src/kenkui/voices/provision.py:474-490` (`_registered_view`, `_loaded_view`)
- Test: `tests/test_voice_traits.py`

**Interfaces:**
- Consumes: `kenkui.voices.registry.CATALOG: dict[str, CatalogEntry]`, whose entries carry `perceived_gender: PerceivedGender`.
- Produces: `list_voices()` returns `Voice` objects whose `perceived_gender` is the manifest value when set, else the catalog value, else `None`.

- [ ] **Step 1: Write the failing test**

```python
def test_manifest_voice_inherits_catalog_gender(tmp_path, monkeypatch):
    """A manifest entry with no gender still enumerates with the catalog's."""
    from kenkui.voices.registry import CATALOG
    from kenkui.voices.provision import list_voices

    voice_id = next(k for k, v in CATALOG.items() if v.perceived_gender == "feminine")
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "engines": {},
                "voices": {
                    voice_id: {
                        "name": "Test",
                        "language": "english",
                        "variety": "built-in",
                        "state": "registered",
                        "engine_id": "english",
                        "provenance": "test",
                        "license_id": "CC-BY-4.0",
                        "commercial_use_allowed": False,
                        "voice_rights": "test",
                    }
                },
            }
        )
    )

    found = next(v for v in list_voices(manifest=manifest) if v.id == voice_id)
    assert found.perceived_gender == "feminine"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_voice_traits.py::test_manifest_voice_inherits_catalog_gender -v`
Expected: FAIL — `assert None == 'feminine'`

- [ ] **Step 3: Write minimal implementation**

In `src/kenkui/voices/provision.py`, add a resolver and use it in both views:

```python
def _gender_for(record: VoiceRecord) -> PerceivedGender:
    """Prefer the manifest's own trait, else the catalog's for this id.

    A manifest written before the field existed carries no gender, but the
    catalog has always known it for the voices it ships. Reading the record
    alone discards that and silently degrades every gendered cast to a random
    one, so the catalog is consulted as a fallback rather than ignored.
    """
    if record.perceived_gender is not None:
        return record.perceived_gender
    entry = CATALOG.get(record.id)
    return entry.perceived_gender if entry is not None else None
```

Then replace `perceived_gender=record.perceived_gender` with
`perceived_gender=_gender_for(record)` in both `_registered_view` and
`_loaded_view`.

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/pytest tests/test_voice_traits.py -v`
Expected: PASS

- [ ] **Step 5: Add the regression test that would have caught the original bug**

```python
def test_gendered_cast_admits_no_opposite_gender_voice():
    """A feminine character is never offered a masculine voice."""
    from kenkui._domain.casting import CharacterProfile, candidates
    from kenkui.voices.provision import list_voices

    pool = tuple(list_voices())
    character = CharacterProfile(
        id="x",
        display_name="X",
        gender="feminine",
        spoken_characters=100,
        chapter_ids=(),
    )
    admitted = candidates("gendered", character, pool)
    assert admitted, "pool must contain at least one feminine voice"
    assert not [v for v in admitted if v.perceived_gender == "masculine"]
```

- [ ] **Step 6: Run the full voice suite**

Run: `.venv/bin/pytest tests/test_voice_traits.py tests/test_voice_pack.py tests/test_pack_resolution.py tests/test_resolve.py -v`
Expected: PASS

- [ ] **Step 7: Commit**

```bash
git add src/kenkui/voices/provision.py tests/test_voice_traits.py
git commit -m "fix: keep the catalog's perceived gender when enumerating voices"
```

---

### Task 2: Persist catalog gender when a voice is registered

Task 1 repairs enumeration. This makes newly written manifest entries correct at rest, so the fallback is a safety net rather than the only thing holding gendered casting up.

**Files:**
- Modify: `src/kenkui/voices/provision.py:87-160` (`add_voice`), `:443-470` (`load_voice`)
- Test: `tests/test_voice_traits.py`

**Interfaces:**
- Consumes: `_gender_for(record)` from Task 1.
- Produces: `VoiceRecord.perceived_gender` populated from `CATALOG` at write time for catalog-known ids.

- [ ] **Step 1: Write the failing test**

```python
def test_registering_a_catalog_voice_persists_its_gender(tmp_path):
    """The written manifest carries the gender, not just the in-memory view."""
    from kenkui.voices.registry import CATALOG
    from kenkui.voices.provision import load_voice

    voice_id = next(k for k, v in CATALOG.items() if v.perceived_gender == "masculine")
    manifest = tmp_path / "manifest.json"
    load_voice(voice_id, manifest=manifest)

    written = json.loads(manifest.read_text())["voices"][voice_id]
    assert written["perceived_gender"] == "masculine"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_voice_traits.py::test_registering_a_catalog_voice_persists_its_gender -v`
Expected: FAIL — `KeyError: 'perceived_gender'`

- [ ] **Step 3: Write minimal implementation**

Where `load_voice` and `add_voice` build a `VoiceRecord` for a catalog-known id, pass the catalog trait:

```python
perceived_gender = (
    (
        perceived_gender
        if perceived_gender is not None
        else (CATALOG[voice_id].perceived_gender if voice_id in CATALOG else None)
    ),
)
```

`manifest._voice_payload` already writes the field when it is not `None`, so no serialisation change is needed.

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/pytest tests/test_voice_traits.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/kenkui/voices/provision.py tests/test_voice_traits.py
git commit -m "fix: record a catalog voice's perceived gender when registering it"
```

---

### Task 3: Report a gendered cast that cannot be honoured

`candidates()` returns `matched or pool`, degrading a gendered cast to a random one with no signal. That silence is why Task 1's defect shipped.

**Files:**
- Modify: `src/kenkui/_domain/casting.py:85-102`
- Modify: `src/kenkui/pipeline.py` (validation path, near `_issue`)
- Modify: `src/kenkui/errors.py` (new `ErrorCode` member)
- Test: `tests/test_casting_solver.py`

**Interfaces:**
- Produces: `ErrorCode.GENDERED_POOL_EMPTY`; `candidates()` unchanged in return type; a new pure helper `ungendered_pool_characters(method, characters, pool) -> tuple[str, ...]` returning the ids that would silently degrade.

- [ ] **Step 1: Write the failing test**

```python
def test_ungendered_pool_is_reported():
    """A gendered cast with no matching voice names the affected characters."""
    pool = (_voice("m1", "masculine"),)
    characters = (
        _character("her", gender="feminine"),
        _character("him", gender="masculine"),
    )
    assert ungendered_pool_characters("gendered", characters, pool) == ("her",)


def test_random_method_reports_nothing() -> None:
    pool = (_voice("m1", "masculine"),)
    characters = (_character("her", gender="feminine"),)
    assert ungendered_pool_characters("random", characters, pool) == ()
```

`_voice(voice_id, gender)` and `_character(...)` are the existing helpers at
`tests/test_casting_solver.py:22` and `:36`. Import
`ungendered_pool_characters` beside the module's other casting imports.

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_casting_solver.py -k ungendered -v`
Expected: FAIL — `ImportError: cannot import name 'ungendered_pool_characters'`

- [ ] **Step 3: Write minimal implementation**

```python
def ungendered_pool_characters(
    method: str, characters: Sequence[CharacterProfile], pool: Sequence[Voice]
) -> tuple[str, ...]:
    """Return the ids a gendered cast cannot honour, in roster order.

    `candidates` falls back to the whole pool rather than dropping the
    speech, which is the right runtime behaviour and the wrong thing to do
    quietly: it turns a gendered cast into a random one with no signal.
    """
    if method != "gendered":
        return ()
    return tuple(
        character.id
        for character in characters
        if character.gender is not None
        and not any(voice.perceived_gender == character.gender for voice in pool)
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/pytest tests/test_casting_solver.py -k ungendered -v`
Expected: PASS

- [ ] **Step 5: Surface it as a validation issue and a warning event**

Add to `src/kenkui/errors.py`:

```python
GENDERED_POOL_EMPTY = "GENDERED_POOL_EMPTY"
```

In the pipeline's validation path, append an issue when the helper returns a
non-empty tuple, with a message naming the first few ids and the total count.
Emit a `Warning` event carrying the same message during resolution.

- [ ] **Step 6: Test the validation surface**

```python
def test_validate_flags_a_gendered_cast_with_no_matching_voices() -> None:
    """The issue is raised before any model call or render."""
    issues = _issues_for(
        method="gendered",
        characters=(_character("her", gender="feminine"),),
        pool=(_voice("m1", "masculine"),),
    )
    assert any(i.code is ErrorCode.GENDERED_POOL_EMPTY for i in issues)
```

Add `_issues_for` as a thin local helper that calls whichever validation entry
point `tests/test_pipeline.py` already exercises for casting issues, rather
than constructing a full EPUB pipeline — this rule is pure and needs no source.

- [ ] **Step 7: Run the suite and commit**

Run: `.venv/bin/pytest tests/test_casting_solver.py tests/test_pipeline.py -v`

```bash
git add src/kenkui/_domain/casting.py src/kenkui/errors.py src/kenkui/pipeline.py tests/
git commit -m "feat: report a gendered cast the voice pool cannot honour"
```

---

### Task 4: Merge non-nested name variants

`resolve_short_forms` matches a bare name to a host only on an exact token match, so `Tye` never reaches `Tyador Borlú`. In the production run this split the narrator into `inspector-borlu` (alasdair) and `tye` (mateo), and the split copy carried no gender at all.

**Files:**
- Modify: `src/kenkui/_characters/identity.py:135-154` (`resolve_short_forms`)
- Test: `tests/test_identity.py`

**Interfaces:**
- Consumes: `ShortForms(assigned: dict[str, str], ambiguous: dict[str, list[str]])`, unchanged.
- Produces: `resolve_short_forms` additionally attaches a bare name that is a **prefix** of exactly one host token of at least 4 characters.

- [ ] **Step 1: Write the failing test**

```python
def test_prefix_short_form_attaches_to_its_only_host():
    """ "Tye" is Tyador when no other name could claim it."""
    entity = {"Tyador Borlu": "Tyador Borlu"}
    assert resolve_short_forms(["Tye"], entity).assigned["Tye"] == "Tyador Borlu"


def test_prefix_short_form_with_two_hosts_is_refused():
    """Ambiguity still refuses rather than guessing: under-merge is the bias."""
    entity = {"Tyador Borlu": "Tyador Borlu", "Tyene Sand": "Tyene Sand"}
    assert "Tye" in resolve_short_forms(["Tye"], entity).ambiguous


def test_short_prefix_is_not_enough():
    """Three characters could prefix half the cast; require four."""
    entity = {"Tyador Borlu": "Tyador Borlu"}
    assert resolve_short_forms(["Ty"], entity).assigned["Ty"] == "Ty"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_identity.py -k prefix -v`
Expected: FAIL — first test gets `"Tye"`, not `"Tyador Borlu"`

- [ ] **Step 3: Write minimal implementation**

In `resolve_short_forms`, after the exact-token lookup finds no host:

```python
_MIN_PREFIX = 4

# A bare name that opens exactly one host token is that host: "Tye" is
# Tyador. Four characters minimum, because a two- or three-letter opening
# could prefix half a cast, and one unambiguous host only, because the
# module's bias is to under-merge rather than guess.
if not hosts and len(token) >= _MIN_PREFIX:
    hosts = {
        entity[full]
        for full in entity
        if any(part.startswith(token) for part in _tokens(full))
    }
```

Leave the existing `len(hosts) == 1` / `elif hosts` / `else` branches to
classify the result, so ambiguity is refused exactly as before.

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/pytest tests/test_identity.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/kenkui/_characters/identity.py tests/test_identity.py
git commit -m "fix: attach a bare name that opens exactly one full name"
```

---

### Task 5: Learn occupational titles from a single name

`detect_titles` requires a leading token to open `threshold=3` distinct names before it counts as a title. `Senior Detective Dhatt` appears once, so `senior` is never learned and the name does not merge with `Dhatt` — in the production run they were cast `alf` and `garrett`.

**Files:**
- Modify: `src/kenkui/_characters/identity.py:102-117` (`detect_titles`)
- Test: `tests/test_identity.py`

**Interfaces:**
- Produces: `detect_titles(names, threshold=3)` additionally returns any leading token that is a known rank or occupation word, regardless of how many names it opens.

- [ ] **Step 1: Write the failing test**

```python
def test_rank_prefix_merges_without_repetition():
    """One "Senior Detective Dhatt" is still Dhatt."""
    names = ["Senior Detective Dhatt", "Dhatt"]
    titles = detect_titles(names)
    assert same_person("Senior Detective Dhatt", "Dhatt", titles)


def test_unknown_single_prefix_still_separates():
    """A word that is neither a known rank nor repeated stays significant."""
    names = ["Red Sonja", "Sonja"]
    titles = detect_titles(names)
    assert "red" not in titles
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_identity.py -k rank_prefix -v`
Expected: FAIL — `same_person` returns `False`

- [ ] **Step 3: Write minimal implementation**

Add beside `PREFIX_TITLES` in `identity.py`:

```python
# Ranks and occupations that open a name without individuating it. Unlike
# PREFIX_TITLES these need no repetition to count: one "Senior Detective
# Dhatt" is still Dhatt, and waiting for a third occurrence splits him in
# two for the whole book.
RANK_WORDS: frozenset[str] = frozenset(
    {
        "senior",
        "junior",
        "chief",
        "deputy",
        "assistant",
        "acting",
        "detective",
        "constable",
        "officer",
        "commissar",
        "commissioner",
        "professor",
        "warden",
        "marshal",
        "brigadier",
        "corporal",
        "lieutenant",
        "ensign",
        "governor",
        "ambassador",
        "secretary",
    }
)
```

Then union it into the returned set in `detect_titles`:

```python
return (
    PREFIX_TITLES
    | RANK_WORDS
    | {token for token, count in leading.items() if count >= threshold}
)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/pytest tests/test_identity.py -v`
Expected: PASS

- [ ] **Step 5: Verify the merge end to end against the production pair**

```python
def test_production_splits_now_merge():
    """The three pairs that were cast as six voices resolve to three."""
    names = [
        "Inspector Borlu",
        "Tyador Borlu",
        "Senior Detective Dhatt",
        "Dhatt",
        "Lizbyet Corwi",
    ]
    entity = group_full_names(names)
    assert entity["Inspector Borlu"] == entity["Tyador Borlu"]
    assert entity["Senior Detective Dhatt"] == entity["Dhatt"]
    assert (
        resolve_short_forms(["Corwi", "Tye"], entity).assigned["Corwi"]
        == entity["Lizbyet Corwi"]
    )
```

Run: `.venv/bin/pytest tests/test_identity.py::test_production_splits_now_merge -v`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add src/kenkui/_characters/identity.py tests/test_identity.py
git commit -m "fix: treat a rank or occupation as a title without repetition"
```

---

### Task 6: Give minted roles the gender their own words state

`_characters/__init__.py:134` synthesises every role with `gender=None`. Seven of the fifteen largest ungendered entries in the production run are `role:woman@...`, `role:man@...`, `role:young-woman@...` — words that state their gender outright.

**Files:**
- Modify: `src/kenkui/_characters/prompts.py` (add the mapping)
- Modify: `src/kenkui/_characters/__init__.py:128-140`
- Test: `tests/test_narration.py`

**Interfaces:**
- Produces: `ROLE_GENDERS: Mapping[str, str]` in `prompts.py`; synthesised role profiles carry `gender` from it.

- [ ] **Step 1: Write the failing test**

```python
def test_gendered_role_words_carry_their_gender():
    """ "role:woman@ch3" is feminine; the text said so."""
    from kenkui._characters import _measured

    spans = (
        SpeakerSpan("ch3", 0, 10, "role:woman@ch3"),
        SpeakerSpan("ch3", 10, 20, "role:old-man@ch3"),
        SpeakerSpan("ch3", 20, 30, "role:innkeeper@ch3"),
    )
    profiles = {c.id: c for c in _measured((), spans)}
    assert profiles["role:woman@ch3"].gender == "feminine"
    assert profiles["role:old-man@ch3"].gender == "masculine"
    assert profiles["role:innkeeper@ch3"].gender is None
```

`_measured` is defined at `src/kenkui/_characters/__init__.py:87`; the code to
change is the `roles = {` block at `:128`. Import `SpeakerSpan` from
`kenkui._domain.planning`.

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_narration.py -k role_words_carry -v`
Expected: FAIL — `assert None == 'feminine'`

- [ ] **Step 3: Write minimal implementation**

In `prompts.py`:

```python
# The role words that state a gender. An innkeeper or a guard may be anyone,
# and casting them from a gendered pool would be a guess; a woman is a woman.
ROLE_GENDERS: Mapping[str, str] = {
    "woman": "feminine",
    "girl": "feminine",
    "old-woman": "feminine",
    "young-woman": "feminine",
    "first-woman": "feminine",
    "second-woman": "feminine",
    "third-woman": "feminine",
    "man": "masculine",
    "boy": "masculine",
    "old-man": "masculine",
    "young-man": "masculine",
    "first-man": "masculine",
    "second-man": "masculine",
    "third-man": "masculine",
}
```

In `__init__.py`, replace `gender=None` in the synthesised profile with a
lookup on the role's bare word:

```python
gender = (ROLE_GENDERS.get(role.removeprefix("role:").split("@")[0]),)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/pytest tests/test_narration.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/kenkui/_characters/prompts.py src/kenkui/_characters/__init__.py tests/test_narration.py
git commit -m "feat: cast gendered role words from the matching voice pool"
```

---

### Task 7: Mint a role for any unrecognised speaker

`_resolve` discards any answer not in `known`, so a speaker the model named correctly becomes unknown and is read in the narrator's voice. `ROLE_WORDS` is a closed 34-word list that lacks `officer`, `policeman` and `colleague`.

**Files:**
- Modify: `src/kenkui/_characters/attribution.py:53-95` (`_resolve`)
- Modify: `src/kenkui/_characters/prompts.py` (bump `PROMPT_VERSION`, relax the role instruction)
- Test: `tests/test_attribution.py`

**Interfaces:**
- Produces: `_resolve` returns `role:<slug>@<chapter_id>` for any non-pronoun answer outside `known`, instead of `None`.

- [ ] **Step 1: Write the failing test**

```python
def test_unknown_name_becomes_a_chapter_scoped_role():
    """A speaker the roster missed still gets a voice of their own."""
    assert (
        _resolve("Rochambeaux", frozenset({"dhatt"}), chapter_id="ch13")
        == "role:rochambeaux@ch13"
    )


def test_role_word_outside_the_old_list_is_accepted():
    assert _resolve("officer", frozenset(), chapter_id="ch13") == "role:officer@ch13"


def test_a_pronoun_is_still_refused():
    """Minting role:he@ch13 would collapse every male speaker into one voice."""
    assert _resolve("he", frozenset(), chapter_id="ch13") is None
    assert _resolve("She", frozenset(), chapter_id="ch13") is None


def test_unknown_is_still_unknown():
    assert _resolve("unknown", frozenset(), chapter_id="ch13") is None


def test_no_chapter_id_means_no_role():
    assert _resolve("Rochambeaux", frozenset(), chapter_id=None) is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_attribution.py -k role -v`
Expected: FAIL — first two return `None`

- [ ] **Step 3: Write minimal implementation**

Replace the closed-list branch at the end of `_resolve`:

```python
    if chapter_id is not None:
        # Any name the model returns that is not on the roster is a speaker
        # the roster missed, not a non-answer. Scoping to the chapter keeps
        # chapter 40's officer distinct from chapter 12's; a recurring
        # speaker therefore gets a voice per chapter, which is accepted —
        # these are overwhelmingly one-scene parts, and a wrong-but-distinct
        # voice beats collapsing into the narrator. Pronouns are refused
        # above, so no role can merge unrelated speakers.
        slug = slugify(candidate)
        return f"role:{slug}@{chapter_id}" if slug else None
    return None
```

Import `slugify` from `kenkui._characters.infer`. `ROLE_WORDS` stays exported
for `ROLE_GENDERS` in Task 6 but no longer gates resolution.

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/pytest tests/test_attribution.py -v`
Expected: PASS

- [ ] **Step 5: Relax the prompt and bump its version**

In `ATTRIBUTION_PROMPT`, replace the closed word list with an instruction to
answer with a short descriptive noun phrase when the text identifies a speaker
without naming them. Bump `PROMPT_VERSION` from `characters-v3` to
`characters-v4`.

- [ ] **Step 6: Run the full suite and commit**

Run: `.venv/bin/pytest -q`

```bash
git add src/kenkui/_characters/attribution.py src/kenkui/_characters/prompts.py tests/test_attribution.py
git commit -m "feat: mint a role for any speaker the roster did not list"
```

---

### Task 8: Distinguish an unknown answer from a dropped one

`speakers = [answers.get(index) for index in range(len(dialogue))]` gives the same `None` whether the model answered "unknown" or never returned the id. Chapter 7's 39.7% unattributed rate is unexplained until these are separable.

**Files:**
- Modify: `src/kenkui/_characters/attribution.py:98-160` (`attribute_chapter`, `_answers`)
- Modify: `src/kenkui/observability.py` usage in `_characters/__init__.py`
- Test: `tests/test_attribution.py`

**Interfaces:**
- Produces: `attribute_chapter` returns `(spans, trailing, coverage)` where `coverage: AttributionCoverage` is a frozen dataclass with `quotes: int`, `answered: int`, `unknown: int`, `dropped: int`.

- [ ] **Step 1: Write the failing test**

```python
def test_coverage_separates_unknown_from_dropped():
    """Two quotes: one answered unknown, one never returned."""
    client = ScriptedClient(
        [
            json.dumps(
                {
                    "attributions": [
                        {"quote_id": 0, "speaker": "unknown"},
                    ]
                }
            )
        ]
    )
    chapter = _inspection('"One," he said. "Two," she said.').chapters[0]
    _, _, coverage = attribute_chapter(
        chapter, (_character("dhatt"),), "m/x", client=client
    )
    assert coverage.quotes == 2
    assert coverage.unknown == 1
    assert coverage.dropped == 1
```

`ScriptedClient` is at `tests/test_attribution.py:24` and `_inspection` at
`:52`. Reuse `_character` from `tests/test_casting_solver.py:36` or build a
`CharacterProfile` inline — only `id` is read.

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_attribution.py -k coverage -v`
Expected: FAIL — `ValueError: not enough values to unpack`

- [ ] **Step 3: Write minimal implementation**

```python
@dataclass(frozen=True, slots=True)
class AttributionCoverage:
    """How one chapter's quotes were accounted for.

    "The model said unknown" and "the model never mentioned this quote" are
    different failures with different fixes, and collapsing both to None
    hides a truncated or malformed response behind ordinary model caution.
    """

    quotes: int
    answered: int
    unknown: int
    dropped: int
```

Have `_answers` return the set of ids the model actually returned alongside
the resolved mapping, then compute the counts in `attribute_chapter` and
return the third element.

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/pytest tests/test_attribution.py -k coverage -v`
Expected: PASS

- [ ] **Step 5: Log it and update the one call site**

In `_characters/__init__.py`, unpack the third value and `log_event` per
chapter with the four counts. Emit a `Warning` event when `dropped` exceeds
5% of `quotes`.

- [ ] **Step 6: Run the full suite and commit**

Run: `.venv/bin/pytest -q && .venv/bin/ruff check src tests && .venv/bin/mypy src`

```bash
git add src/kenkui/_characters/ tests/test_attribution.py
git commit -m "feat: count dropped attributions separately from unknown ones"
```

---

### Task 9: Verify against the production book

**Files:**
- Create: `evals/attribution/score_gender.py`

**Interfaces:**
- Consumes: the attribution store written by a real run; `list_voices()`.
- Produces: printed per-character rows and three totals — matched, opposite, ungendered.

- [ ] **Step 1: Write the scorer**

```python
"""Score gendered casting against a stored production run.

Answers one question: how often does a character whose gender was inferred
speak in a voice of the other gender? That is the defect a listener hears,
and nothing in the unit suite can observe it, because it only appears once
a real roster meets a real voice pool.
"""

from __future__ import annotations

import argparse
import sqlite3
from pathlib import Path

from kenkui._tts.production import default_cache_root
from kenkui.voices.provision import list_voices


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("attribution_id")
    args = parser.parse_args()

    gender_of = {v.id: v.perceived_gender for v in list_voices()}
    db = sqlite3.connect(default_cache_root() / "casting.sqlite3")
    db.row_factory = sqlite3.Row
    rows = db.execute(
        "SELECT c.character_id, c.display_name, c.gender, c.spoken_characters,"
        " a.voice_id FROM characters c"
        " JOIN cast_assignments a ON a.character_id = c.character_id"
        " WHERE c.attribution_id = ?"
        " ORDER BY c.spoken_characters DESC",
        (args.attribution_id,),
    ).fetchall()

    matched = opposite = ungendered = 0
    print(f"{'character':28} {'inferred':10} {'voice':12} {'voice is':10} verdict")
    print("-" * 76)
    for row in rows:
        voice_gender = gender_of.get(row["voice_id"])
        if not row["gender"]:
            ungendered += 1
            verdict = "no gender inferred"
        elif voice_gender == row["gender"]:
            matched += 1
            verdict = ""
        elif voice_gender is None:
            verdict = "voice has no gender"
        else:
            opposite += 1
            verdict = "WRONG GENDER"
        print(
            f"{row['character_id'][:28]:28} {str(row['gender'])[:10]:10} "
            f"{row['voice_id'][:12]:12} {str(voice_gender)[:10]:10} {verdict}"
        )

    print(f"\nmatched: {matched}   opposite: {opposite}   ungendered: {ungendered}")
    return 1 if opposite else 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 2: Run it against the stored run for a baseline**

Run: `.venv/bin/python evals/attribution/score_gender.py fe2257386971a24c30298d9e054ceb66253633b1a2aaa8bb0d936bf160db5cd3`
Expected: a non-zero `opposite` count, including `lizbyet-corwi -> declan`.

- [ ] **Step 3: Commit the scorer**

```bash
git add evals/attribution/score_gender.py
git commit -m "test: score gendered casting against a stored production run"
```

- [ ] **Step 4: Re-render and compare**

Re-render *The City and the City* with the same model, then re-run the scorer
against the new attribution id. Expected after Tasks 1-8: `opposite` is 0, and
`inspector-borlu`/`tye`, `corwi`/`lizbyet-corwi`, `dhatt`/`senior-detective-dhatt`
each appear once rather than twice.

- [ ] **Step 5: Record the remaining ungendered characters**

Note which characters still report `no gender inferred` and their
`spoken_characters`. **This number decides whether Tasks 11 and 12 are worth
doing.** If the remaining ungendered volume is negligible, stop here and close
them out in the plan rather than implementing speculative signals.

---

### Task 10: Carry aliases on a character profile

Spec section B. The roster records `Mr. Khurusch` while the text says
`Khurusch said`. Every call site that needs a surface form derives one ad hoc,
and Task 11's honorific check cannot work without them.

**Files:**
- Modify: `src/kenkui/_domain/casting.py:28-42` (`CharacterProfile`)
- Modify: `src/kenkui/_characters/infer.py` (`normalise_roster`, `merge_rosters`)
- Modify: `src/kenkui/_characters/store.py` (persist and read the field)
- Test: `tests/test_identity.py`, `tests/test_casting_store.py`

**Interfaces:**
- Produces: `CharacterProfile.aliases: tuple[str, ...] = ()`, sorted and deduplicated, holding every display name folded into this character by `merge_rosters`.

- [ ] **Step 1: Write the failing test**

```python
def test_merged_names_are_kept_as_aliases() -> None:
    """Folding Lizbyet Corwi into Corwi must not lose the other surface form."""
    rosters = (
        (CharacterProfile("corwi", "Corwi", None, 0, ()),),
        (CharacterProfile("lizbyet-corwi", "Lizbyet Corwi", "feminine", 0, ()),),
    )
    merged = merge_rosters(rosters)
    assert len(merged) == 1
    assert set(merged[0].aliases) == {"Corwi", "Lizbyet Corwi"}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_identity.py -k aliases -v`
Expected: FAIL — `AttributeError: 'CharacterProfile' object has no attribute 'aliases'`

- [ ] **Step 3: Write minimal implementation**

Add to `CharacterProfile`:

```python
    # Every surface form folded into this character. The display name alone
    # is not enough to find them in the text: the roster says "Mr. Khurusch"
    # and the page says "Khurusch said".
    aliases: tuple[str, ...] = ()
```

In `merge_rosters`, accumulate each folded `display_name` into a set per head
id and emit `aliases=tuple(sorted(names))`. Add the column to the `characters`
table in `store.py` and read it back in `_cast_from_row`.

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/pytest tests/test_identity.py tests/test_casting_store.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/kenkui/_domain/casting.py src/kenkui/_characters/ tests/
git commit -m "feat: keep every folded name as a character alias"
```

---

### Task 11: Infer gender from an honorific (gated on Task 9 Step 5)

Spec section D2. **Skip if Task 9 Step 5 shows negligible ungendered volume.**

**Files:**
- Create: `src/kenkui/_characters/gender.py`
- Test: `tests/test_gender.py`

**Interfaces:**
- Produces: `from_honorific(profile: CharacterProfile) -> PerceivedGender`.

- [ ] **Step 1: Write the failing test**

```python
def test_honorific_on_a_full_name_settles_gender() -> None:
    assert from_honorific(_character("g", display_name="Mrs. Geary")) == "feminine"
    assert from_honorific(_character("g", display_name="Mr. Geary")) == "masculine"


def test_a_bare_surname_is_never_enough() -> None:
    """ "you're Mr. Corwi are you" is a character misspeaking, not evidence."""
    assert from_honorific(_character("c", display_name="Corwi")) is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_gender.py -v`
Expected: FAIL — module does not exist

- [ ] **Step 3: Write minimal implementation**

```python
_HONORIFICS: Mapping[str, PerceivedGender] = {
    "mr": "masculine",
    "mister": "masculine",
    "sir": "masculine",
    "lord": "masculine",
    "master": "masculine",
    "mrs": "feminine",
    "ms": "feminine",
    "miss": "feminine",
    "lady": "feminine",
    "madam": "feminine",
    "dame": "feminine",
}


def from_honorific(profile: CharacterProfile) -> PerceivedGender:
    """Read gender off a leading honorific, full names only.

    Matched against a bare surname this is actively wrong: a character who
    misaddresses Lizbyet Corwi as "Mr. Corwi" would make her masculine. A
    name is only evidence about itself when the honorific is part of it.
    """
    parts = profile.display_name.split()
    if len(parts) < 2:
        return None
    return _HONORIFICS.get(parts[0].lower().strip("."))
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/pytest tests/test_gender.py -v`
Expected: PASS

- [ ] **Step 5: Apply it after attribution and commit**

Call `from_honorific` in `_measured` for any profile whose `gender` is `None`,
leaving an existing gender untouched.

```bash
git add src/kenkui/_characters/gender.py src/kenkui/_characters/__init__.py tests/test_gender.py
git commit -m "feat: read a character's gender from their honorific"
```

---

### Task 12: Infer gender from dialogue tags (gated on Task 9 Step 5)

Spec section D3. **Skip if Task 9 Step 5 shows negligible ungendered volume.**

**Files:**
- Modify: `src/kenkui/_characters/gender.py`
- Test: `tests/test_gender.py`

**Interfaces:**
- Produces: `from_dialogue_tags(spans, chapter_text_by_id) -> dict[str, PerceivedGender]`.

- [ ] **Step 1: Write the failing test**

```python
def test_a_trailing_tag_pronoun_genders_its_speaker() -> None:
    text = '"Hello," she said. "Goodbye," she added.'
    spans = (SpeakerSpan("ch1", 0, 8, "x"), SpeakerSpan("ch1", 19, 28, "x"))
    assert from_dialogue_tags(spans, {"ch1": text}) == {"x": "feminine"}


def test_a_single_contrary_tag_does_not_flip_a_majority() -> None:
    """Nine "he said" against one "she said" is still masculine."""
    unit = '"Hi," he said. '
    text = unit * 9 + '"Hi," she said. '
    spans = tuple(
        SpeakerSpan("ch1", i * len(unit), i * len(unit) + 6, "x") for i in range(10)
    )
    assert from_dialogue_tags(spans, {"ch1": text}) == {"x": "masculine"}


def test_an_even_split_decides_nothing() -> None:
    """One each way is not a majority, so the character stays ungendered."""
    text = '"Hi," he said. "Bye," she said. '
    spans = (SpeakerSpan("ch1", 0, 6, "x"), SpeakerSpan("ch1", 15, 22, "x"))
    assert from_dialogue_tags(spans, {"ch1": text}) == {}
```

Offsets must land on the closing quote of each span so the tag scan starts at
`end`; adjust the literals if `extract_spans` disagrees, and assert the span
text first if a test fails for that reason.

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_gender.py -k dialogue_tag -v`
Expected: FAIL — `ImportError`

- [ ] **Step 3: Write minimal implementation**

Read up to 40 characters after each span's `end`, stopping at the next quote
character. Match `^[\s,.—-]*(he|she)\s+(said|asked|replied|...)` and tally per
character id. Emit a gender only when the winner has at least twice the
runner-up, so one stray tag cannot flip a character.

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/pytest tests/test_gender.py -v`
Expected: PASS

- [ ] **Step 5: Wire it in below the honorific, and commit**

Precedence, highest first: an existing inferred gender, the role word (Task 6),
the honorific (Task 11), then the tag vote. Delete the first-non-null fold at
`infer.py:148`, which freezes whatever the earliest chapter guessed.

```bash
git add src/kenkui/_characters/ tests/test_gender.py
git commit -m "feat: gender a speaker from the pronoun in their dialogue tags"
```

- [ ] **Step 6: Re-run the scorer**

Run: `.venv/bin/python evals/attribution/score_gender.py <new attribution id>`
Expected: `ungendered` lower than the Task 9 baseline, `opposite` still 0.
