# Speakers Without Names Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let attribution name two speakers it currently cannot — a character whose aliases the roster splits or merges wrongly, and a first-person narrator who says "I said" — and optionally a speaker the text identifies by role rather than name.

**Architecture:** Three additions, staged by how well each is validated. A new pure module `_characters/identity.py` decides when two names are one person, and `merge_rosters` composes it. First-person detection is a regex over dialogue tags plus one extra field in the roster prompt; the narrator id is threaded through derivation only, so nothing is stored and no schema changes. Roles reuse `CharacterProfile` unchanged and are last, because a role read in the narrator's voice is an acceptable outcome and the feature is an increment rather than a requirement.

**Tech Stack:** Python 3.11–3.13, uv, pytest, mypy strict, ruff. No new dependencies in this plan — identity is pure string work and first-person detection is `re`. A spaCy dependency is acceptable to the project and would open two further changes measured in `evals/attribution/` — a deterministic dialogue-tag pass that attributes a fifth of quotes at 0.976 precision for nothing, and identifying the narrator from vocative-versus-speech counts instead of asking the model. Both are separate work: this plan asks the roster pass, which is already reading the chapter, and adds no stage.

**Spec:** `docs/superpowers/specs/2026-08-24-unnamed-speakers-design.md`

## Global Constraints

- Supported Python is `>=3.11,<3.14`; mypy runs at `python_version = 3.12`, strict.
- Every command in CONTRIBUTING must pass: `uv run ruff format --check .`, `uv run ruff check .`, `uv run mypy`, `uv run pytest`.
- Coverage gate is `--cov-fail-under=90`, branch coverage, and it is enforced on every `pytest` run.
- Tests are deterministic and offline. The provider boundary is the existing `Client` protocol in `_characters/llm.py`; no test may open a socket.
- `PROMPT_VERSION` in `_characters/prompts.py` keys the attribution store. Any change to prompt text in this plan bumps it, in the same commit as the text.
- A book with neither a narrator nor a role must produce byte-identical segment identities and an unchanged plan fingerprint. Nothing in this plan may add a key to a segment identity.
- No prompt text, book text, or model response may reach a log record or an exception message.
- Commits are signed off: `git commit --signoff`.
- Add or update tests before the implementation change, per CONTRIBUTING.

---

### Task 1: Alias identity rules

Pure functions deciding when two names denote one person. No I/O, no model, importable without cost.

**Files:**
- Create: `src/kenkui/_characters/identity.py`
- Test: `tests/test_identity.py`

**Interfaces:**
- Consumes: nothing.
- Produces: `same_person(first: str, second: str, titles: frozenset[str] = PREFIX_TITLES) -> bool`, `detect_titles(names: Sequence[str], threshold: int = 3) -> frozenset[str]`, `group_full_names(names: Sequence[str]) -> dict[str, str]`, `resolve_short_forms(shorts: Sequence[str], entity: Mapping[str, str]) -> ShortForms`, and `ShortForms(assigned: dict[str, str], ambiguous: dict[str, list[str]])`.

- [ ] **Step 1: Write the failing test**

```python
"""One person or two, decided from the names alone."""

from __future__ import annotations

import pytest

from kenkui._characters.identity import (
    detect_titles,
    group_full_names,
    resolve_short_forms,
    same_person,
)


@pytest.mark.parametrize(
    ("first", "second", "expected"),
    [
        # Two differing prefix honorifics are two people.
        ("Mr Elliot", "Miss Elliot", False),
        ("Mr Geary", "Mrs Geary", False),
        ("Admiral Brand", "Admiral Croft", False),
        # One honorific against none is one person.
        ("Dr. Pelorat", "Janov Pelorat", True),
        ("Captain Han Pritcher", "Han Pritcher", True),
        # A trailing title attaches to a single person.
        ("Moiraine Sedai", "Moiraine Aes Sedai", True),
        # Anything else differing is a surname.
        ("Charles Hayter", "Charles Musgrove", False),
        ("Balwen Ironhand", "Balwen Mayel", False),
    ],
)
def test_same_person(first: str, second: str, expected: bool) -> None:
    assert same_person(first, second) is expected


def test_detect_titles_learns_invented_honorifics() -> None:
    names = [
        "Brightlord Dalinar",
        "Brightlord Sadeas",
        "Brightlord Roshone",
        "Dalinar Kholin",
    ]
    assert "brightlord" in detect_titles(names)


def test_invented_honorific_does_not_split_one_person() -> None:
    names = ["Brightlord Dalinar", "Brightlord Sadeas", "Brightlord Roshone",
             "Dalinar Kholin"]
    titles = detect_titles(names)
    assert same_person("Brightlord Dalinar", "Dalinar Kholin", titles) is True


def test_group_full_names_folds_one_person() -> None:
    entity = group_full_names(["Moiraine Sedai", "Moiraine Aes Sedai"])
    assert len(set(entity.values())) == 1


def test_short_form_with_one_host_is_assigned() -> None:
    entity = group_full_names(["Tam al'Thor"])
    resolved = resolve_short_forms(["Tam"], entity)
    assert resolved.assigned["Tam"] == "Tam al'Thor"
    assert resolved.ambiguous == {}


def test_short_form_with_two_hosts_is_dropped() -> None:
    entity = group_full_names(["Charles Hayter", "Charles Musgrove"])
    resolved = resolve_short_forms(["Charles"], entity)
    assert "Charles" not in resolved.assigned
    assert sorted(resolved.ambiguous["Charles"]) == ["Charles Hayter", "Charles Musgrove"]


def test_short_form_with_no_host_stands_alone() -> None:
    resolved = resolve_short_forms(["Egwene"], {})
    assert resolved.assigned["Egwene"] == "Egwene"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_identity.py -v --no-cov`
Expected: FAIL, `ModuleNotFoundError: No module named 'kenkui._characters.identity'`

- [ ] **Step 3: Write the implementation**

```python
"""Deciding when two names are one character, and when they are two.

Pure functions over names: no model, no I/O. `merge_rosters` composes them.

Two failure directions with different costs. Over-merging puts two people in
one voice, which is the failure attribution is organised against. Under-merging
gives one person two voices, audible but locally consistent within any stretch
where a single name is used. Under-merging is therefore the default here, and
every rule below either prevents an over-merge or refuses to guess.
"""

from __future__ import annotations

from collections import Counter
from typing import TYPE_CHECKING, NamedTuple

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

# Honorifics that PRECEDE a name separate individuals: "Mr Elliot" and "Miss
# Elliot" are two Elliots, "Mr Geary" and "Mrs Geary" a husband and wife who
# both speak. Honorifics that FOLLOW attach to one person: "Moiraine Sedai"
# and "Moiraine Aes Sedai" are one Moiraine.
PREFIX_TITLES: frozenset[str] = frozenset({
    "mr", "mrs", "miss", "ms", "master", "mistress", "lord", "lady", "sir",
    "dame", "dr", "doctor", "captain", "admiral", "colonel", "major",
    "general", "inspector", "sergeant", "king", "queen", "prince", "princess",
    "goodman", "goodwife", "mother", "father", "elder", "mayor",
})
SUFFIX_TITLES: frozenset[str] = frozenset({
    "sedai", "aes", "gaidin", "jr", "sr", "ii", "iii",
})


class ShortForms(NamedTuple):
    """Where each bare name went, and which were too ambiguous to place."""

    assigned: dict[str, str]
    ambiguous: dict[str, list[str]]


def _tokens(name: str) -> set[str]:
    return {token.lower().strip(".") for token in name.split()}


def same_person(
    first: str, second: str, titles: frozenset[str] = PREFIX_TITLES
) -> bool:
    """Report whether two full names denote one person.

    Two differing prefix titles separate people. One title against none does
    not: "Brightlord Dalinar" and "Dalinar Kholin" are one Dalinar. With titles
    set aside, names denote one person when what remains is equal or nested;
    two different residues are two surnames, hence two people.
    """
    tokens_a, tokens_b = _tokens(first), _tokens(second)
    if tokens_a == tokens_b:
        return True
    prefix_a, prefix_b = tokens_a & titles, tokens_b & titles
    if prefix_a and prefix_b and prefix_a != prefix_b:
        return False
    rest_a = tokens_a - titles - SUFFIX_TITLES
    rest_b = tokens_b - titles - SUFFIX_TITLES
    return rest_a == rest_b or rest_a < rest_b or rest_b < rest_a


def detect_titles(
    names: Sequence[str], threshold: int = 3
) -> frozenset[str]:
    """Find a book's own honorifics: leading tokens shared by many names.

    No fixed list holds every invented honorific, and a missed one splits a
    character in two. A token opening `threshold` or more distinct names is
    doing a title's job whatever the book calls it.
    """
    leading: Counter[str] = Counter()
    for name in names:
        parts = name.split()
        if len(parts) > 1:
            leading[parts[0].lower().strip(".")] += 1
    return PREFIX_TITLES | {
        token for token, count in leading.items() if count >= threshold
    }


def group_full_names(names: Sequence[str]) -> dict[str, str]:
    """Map each multi-token name to the entity that owns it."""
    titles = detect_titles(names)
    entity: dict[str, str] = {}
    for name in names:
        entity[name] = next(
            (
                entity[other]
                for other in names
                if other in entity and same_person(other, name, titles)
            ),
            name,
        )
    return entity


def resolve_short_forms(
    shorts: Sequence[str], entity: Mapping[str, str]
) -> ShortForms:
    """Attach each bare name to its entity, or refuse when two could claim it.

    A refused short form is dropped rather than kept. Keeping it as its own
    entry would split one person into two voices; assigning it would merge two
    people into one. Dropping it leaves the model to name a full form from the
    passage, which it is better placed to do than any rule here.
    """
    assigned: dict[str, str] = {}
    ambiguous: dict[str, list[str]] = {}
    for short in shorts:
        token = short.lower().strip(".")
        hosts = {entity[full] for full in entity if token in _tokens(full)}
        if len(hosts) == 1:
            assigned[short] = next(iter(hosts))
        elif hosts:
            ambiguous[short] = sorted(hosts)
        else:
            assigned[short] = short
    return ShortForms(assigned, ambiguous)
```

- [ ] **Step 4: Run tests and the full gate**

Run: `uv run pytest tests/test_identity.py -v --no-cov && uv run ruff check . && uv run ruff format --check . && uv run mypy`
Expected: all PASS

- [ ] **Step 5: Commit**

```bash
git add src/kenkui/_characters/identity.py tests/test_identity.py
git commit --signoff -m "feat: decide name identity by honorific position"
```

---

### Task 2: Compose identity into roster merging

`merge_rosters` currently deduplicates by exact id, so a model returning both `charles` and `charles-hayter` yields two characters and two voices for one person — or worse, folds a second Charles into the first.

**Files:**
- Modify: `src/kenkui/_characters/infer.py` (`merge_rosters`)
- Test: `tests/test_identity.py` (append)

**Interfaces:**
- Consumes: `same_person`, `detect_titles`, `group_full_names`, `resolve_short_forms` from Task 1.
- Produces: `merge_rosters` unchanged in signature — `(rosters: tuple[tuple[CharacterProfile, ...], ...]) -> tuple[CharacterProfile, ...]` — with alias folding applied.

- [ ] **Step 1: Write the failing test**

```python
from kenkui._characters.infer import merge_rosters
from kenkui._domain.casting import CharacterProfile


def _profile(character_id: str, display: str) -> CharacterProfile:
    return CharacterProfile(
        id=character_id,
        display_name=display,
        gender=None,
        spoken_characters=0,
        chapter_ids=(),
    )


def test_merge_rosters_folds_one_person_under_two_names() -> None:
    merged = merge_rosters((
        (_profile("moiraine-sedai", "Moiraine Sedai"),),
        (_profile("moiraine-aes-sedai", "Moiraine Aes Sedai"),),
    ))
    assert len(merged) == 1


def test_merge_rosters_keeps_two_people_apart() -> None:
    merged = merge_rosters((
        (_profile("mr-elliot", "Mr Elliot"),),
        (_profile("miss-elliot", "Miss Elliot"),),
    ))
    assert len(merged) == 2


def test_merge_rosters_drops_an_ambiguous_short_form() -> None:
    merged = merge_rosters((
        (_profile("charles-hayter", "Charles Hayter"),),
        (_profile("charles-musgrove", "Charles Musgrove"),),
        (_profile("charles", "Charles"),),
    ))
    assert sorted(character.id for character in merged) == [
        "charles-hayter",
        "charles-musgrove",
    ]


def test_merge_rosters_attaches_an_unambiguous_short_form() -> None:
    merged = merge_rosters((
        (_profile("tam-althor", "Tam al'Thor"),),
        (_profile("tam", "Tam"),),
    ))
    assert len(merged) == 1
    assert merged[0].id == "tam-althor"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_identity.py -k merge_rosters -v --no-cov`
Expected: FAIL — `test_merge_rosters_folds_one_person_under_two_names` asserts 1, gets 2.

- [ ] **Step 3: Write the implementation**

Replace the body of `merge_rosters` in `src/kenkui/_characters/infer.py`:

```python
def merge_rosters(
    rosters: tuple[tuple[CharacterProfile, ...], ...],
) -> tuple[CharacterProfile, ...]:
    """Combine per-chapter rosters, folding the names that are one person.

    A character named in ten chapters must be one entry, or casting would
    assign them ten voices. Exact-id matching is not enough for that: a model
    asked about one chapter answers "Tam" and about another "Tam al'Thor", and
    the two are one man. `identity` decides which pairs fold, and refuses the
    short forms that two people could claim.
    """
    from kenkui._characters.identity import (  # noqa: PLC0415 - avoids a cycle
        group_full_names,
        resolve_short_forms,
    )

    by_display: dict[str, CharacterProfile] = {}
    for roster in rosters:
        for character in roster:
            existing = by_display.get(character.display_name)
            if existing is None:
                by_display[character.display_name] = character
            elif existing.gender is None and character.gender is not None:
                by_display[character.display_name] = existing.__class__(
                    id=existing.id,
                    display_name=existing.display_name,
                    gender=character.gender,
                    spoken_characters=existing.spoken_characters,
                    chapter_ids=existing.chapter_ids,
                )

    names = sorted(by_display)
    entity = group_full_names([n for n in names if len(n.split()) > 1])
    resolved = resolve_short_forms([n for n in names if len(n.split()) == 1], entity)
    canonical = {**entity, **resolved.assigned}

    merged: dict[str, CharacterProfile] = {}
    for name, character in by_display.items():
        target = canonical.get(name)
        if target is None:
            # Ambiguous short form: two people could claim it, so it names
            # neither. See identity.resolve_short_forms.
            continue
        head = by_display[target]
        gender = head.gender or character.gender
        merged[head.id] = head.__class__(
            id=head.id,
            display_name=head.display_name,
            gender=gender,
            spoken_characters=head.spoken_characters,
            chapter_ids=head.chapter_ids,
        )
    return tuple(sorted(merged.values(), key=lambda character: character.id))
```

- [ ] **Step 4: Run tests and the full gate**

Run: `uv run pytest tests/test_identity.py tests/test_attribution.py -v --no-cov && uv run ruff check . && uv run mypy`
Expected: all PASS. If `tests/test_attribution.py` fails, the roster fixture there relies on exact-id merging — read the failure before changing anything.

- [ ] **Step 5: Commit**

```bash
git add src/kenkui/_characters/infer.py tests/test_identity.py
git commit --signoff -m "fix: fold roster aliases that name one character"
```

---

### Task 3: Detect first-person dialogue tags

Pure regex. A book qualifies as first-person only if it carries these; the check is what stops a place being nominated as a narrator later.

**Files:**
- Create: `src/kenkui/_characters/narration.py`
- Test: `tests/test_narration.py`

**Interfaces:**
- Consumes: nothing.
- Produces: `first_person_tags(text: str, quote_ends: Sequence[int], window: int = 46) -> int` and `is_first_person(text: str, quote_ends: Sequence[int], minimum: int = 3) -> bool`.

- [ ] **Step 1: Write the failing test**

```python
"""First-person narration is found by its dialogue tags, or not at all."""

from __future__ import annotations

from kenkui._characters.narration import first_person_tags, is_first_person
from kenkui._characters.quotes import extract_spans

FIRST = '"I will not," I said. "You know that." She turned away.'
THIRD = '"I will not," Anne said. "You know that." She turned away.'


def _ends(text: str) -> list[int]:
    return [s.end for s in extract_spans("ch", text) if s.is_dialogue]


def test_first_person_tag_is_counted() -> None:
    assert first_person_tags(FIRST, _ends(FIRST)) == 1


def test_third_person_tag_is_not_counted() -> None:
    assert first_person_tags(THIRD, _ends(THIRD)) == 0


def test_inverted_tag_is_counted() -> None:
    text = '"I will not," said I. "You know that."'
    assert first_person_tags(text, _ends(text)) == 1


def test_a_book_needs_several_tags_to_qualify() -> None:
    assert is_first_person(FIRST, _ends(FIRST)) is False
    doubled = FIRST * 3
    assert is_first_person(doubled, _ends(doubled)) is True
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_narration.py -v --no-cov`
Expected: FAIL, `ModuleNotFoundError: No module named 'kenkui._characters.narration'`

- [ ] **Step 3: Write the implementation**

```python
"""Recognising narration written in the first person.

A narrator who says `"..." I said` is unattributable twice over: the roster is
built from names, and "I" is not one, while the prompt forbids answering with a
pronoun. This finds the tags that say a chapter is narrated that way, so the
roster pass can ask who is speaking them.

Detection is deliberately the cheap half. Which character narrates is a
question for the model, which is already reading the chapter; whether the
question is worth asking is decided here, for nothing.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence

_VERBS = (
    r"said|says|asked|asks|replied|answered|told|murmured|muttered|whispered"
    r"|shouted|called|cried|repeated|added|agreed|demanded|admitted|observed"
    r"|managed|offered|insisted|protested|snapped|breathed"
)

# "I said" and "said I", the two orders English puts a first-person tag in.
# Bounded so the match cannot run past the tag into the next sentence.
_FIRST_PERSON_TAG = re.compile(
    rf'^[^.!?"“”]{{0,14}}?\bI\b[^.!?"“”]{{0,14}}?\b(?:{_VERBS})\b'
    rf"|^\s*,?\s*(?:{_VERBS})\s+I\b",
    re.IGNORECASE,
)

DEFAULT_WINDOW = 46
DEFAULT_MINIMUM = 3


def first_person_tags(
    text: str, quote_ends: Sequence[int], window: int = DEFAULT_WINDOW
) -> int:
    """Count quotes in this chapter tagged with a first-person speech verb."""
    return sum(
        1 for end in quote_ends if _FIRST_PERSON_TAG.match(text[end : end + window])
    )


def is_first_person(
    text: str, quote_ends: Sequence[int], minimum: int = DEFAULT_MINIMUM
) -> bool:
    """Whether this chapter is narrated in the first person.

    Several tags are required rather than one. A third-person book quoting a
    character who says "I told him" produces the occasional false match, and
    one match must not turn a book first-person.
    """
    return first_person_tags(text, quote_ends) >= minimum
```

- [ ] **Step 4: Run tests and the full gate**

Run: `uv run pytest tests/test_narration.py -v --no-cov && uv run ruff check . && uv run mypy`
Expected: all PASS

- [ ] **Step 5: Commit**

```bash
git add src/kenkui/_characters/narration.py tests/test_narration.py
git commit --signoff -m "feat: recognise first-person dialogue tags"
```

---

### Task 4: Ask the roster pass who narrates

The roster prompt already reads the chapter. When the chapter is first-person, it also returns which listed character is the narrator.

**Files:**
- Modify: `src/kenkui/_characters/prompts.py` (`ROSTER_PROMPT`, `PROMPT_VERSION`)
- Modify: `src/kenkui/_characters/__init__.py` (`_roster_for`, `resolve_attribution`)
- Test: `tests/test_narration.py` (append)

**Interfaces:**
- Consumes: `is_first_person` from Task 3; `slugify` from `_characters/infer.py`.
- Produces: `_roster_for(chapter_text: str, model_id: str, client: Client | None, *, first_person: bool) -> tuple[tuple[CharacterProfile, ...], str | None]` — the roster and the narrator's id or `None`. `resolve_attribution` gains no public parameter; it computes the narrator internally.

- [ ] **Step 1: Write the failing test**

```python
import json

import kenkui as kk
from kenkui._characters import resolve_attribution


class NarratedClient:
    """A model that names a narrator when asked, and attributes to them."""

    def __init__(self) -> None:
        self.roster_prompts: list[str] = []

    def complete(self, model: str, prompt: str) -> str:
        assert model
        if "List the speaking characters" in prompt:
            self.roster_prompts.append(prompt)
            return json.dumps({
                "characters": [
                    {"id": "nieshka", "name": "Nieshka", "gender": "feminine"}
                ],
                "narrator": "nieshka",
            })
        return json.dumps({"attributions": [{"quote_id": 0, "speaker": "narrator"}]})


def test_roster_prompt_asks_for_a_narrator_when_first_person(tmp_path) -> None:
    text = '"I will not," I said. "You know that." ' * 3
    book = kk.epub(_epub_with(tmp_path, text))
    client = NarratedClient()
    record = resolve_attribution(
        book.inspect(), "b" * 64, "m/x", client=client, roster_model_id="m/x"
    )
    assert any("narrator" in prompt for prompt in client.roster_prompts)
    assert any(span.character_id == "nieshka" for span in record.spans)
```

Add this helper at the top of `tests/test_narration.py`, since the test needs a real EPUB and the suite has no shared factory for one:

```python
import zipfile
from pathlib import Path


def _epub_with(tmp_path: Path, body: str) -> Path:
    """Write a one-chapter EPUB containing *body*."""
    path = tmp_path / "book.epub"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("mimetype", "application/epub+zip")
        archive.writestr(
            "META-INF/container.xml",
            '<?xml version="1.0"?><container version="1.0" '
            'xmlns="urn:oasis:names:tc:opendocument:xmlns:container">'
            '<rootfiles><rootfile full-path="c.opf" '
            'media-type="application/oebps-package+xml"/></rootfiles></container>',
        )
        archive.writestr(
            "c.opf",
            '<?xml version="1.0"?><package xmlns="http://www.idpf.org/2007/opf" '
            'version="3.0" unique-identifier="i"><metadata '
            'xmlns:dc="http://purl.org/dc/elements/1.1/"><dc:identifier '
            'id="i">x</dc:identifier><dc:title>T</dc:title>'
            "<dc:language>en</dc:language></metadata><manifest>"
            '<item id="a" href="a.xhtml" media-type="application/xhtml+xml"/>'
            '</manifest><spine><itemref idref="a"/></spine></package>',
        )
        archive.writestr(
            "a.xhtml",
            '<html xmlns="http://www.w3.org/1999/xhtml"><body><p>'
            + body
            + "</p></body></html>",
        )
    return path
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_narration.py -k narrator -v --no-cov`
Expected: FAIL — the roster prompt contains no "narrator" wording.

- [ ] **Step 3: Add the prompt field**

In `src/kenkui/_characters/prompts.py`, bump the version and extend the roster prompt:

```python
PROMPT_VERSION = "characters-v2"
```

Append to `ROSTER_PROMPT`, before the `Passage:` block:

```
- "narrator": when the passage is written in the first person, the "id" of the
  character narrating it, taken from the list you are returning. Omit this key
  or return null when the passage is written in the third person, or when the
  narrator is never named.
```

and change the JSON shape line to:

```
{{"characters": [{{"id": "...", "name": "...", "gender": "..."}}], "narrator": "..."}}
```

- [ ] **Step 4: Thread the narrator through derivation**

In `src/kenkui/_characters/__init__.py`, change `_roster_for` to return the pair and `resolve_attribution` to collect it. The narrator is derivation-local: it shapes the attribution prompt and is never written to the store, so no schema changes and `AttributionRecord` is untouched.

```python
def _roster_for(
    chapter_text: str,
    model_id: str,
    client: Client | None,
    *,
    first_person: bool,
) -> tuple[tuple[CharacterProfile, ...], str | None]:
    """Infer one chapter's characters, and who narrates it when asked."""
    escaped = chapter_text.replace("{", "{{").replace("}", "}}")
    try:
        payload = complete_json(
            model_id,
            ROSTER_PROMPT.format(passage=escaped),
            _ROSTER_SCHEMA,
            client=client,
        )
    except ModelError:
        return (), None
    roster = normalise_roster(payload["characters"])
    narrator = None
    if first_person:
        claimed = payload.get("narrator")
        if isinstance(claimed, str) and claimed.strip():
            candidate = slugify(claimed)
            # Only a character the model also listed. A narrator who is not on
            # the roster cannot be cast, and inventing an entry here would put
            # an unvouched id into the plan fingerprint.
            if any(character.id == candidate for character in roster):
                narrator = candidate
    return roster, narrator
```

`_ROSTER_SCHEMA` stays `{"characters": list}`: `narrator` is optional, and adding it would make a third-person answer invalid.

In `resolve_attribution`, replace the roster loop:

```python
    rosters: list[tuple[CharacterProfile, ...]] = []
    narrators: Counter[str] = Counter()
    for chapter in inspection.chapters:
        if cancel is not None:
            cancel.raise_if_cancelled()
        spans = extracted[chapter.id]
        if not any(span.is_dialogue for span in spans):
            continue
        ends = [span.end for span in spans if span.is_dialogue]
        roster, narrator = _roster_for(
            chapter.text,
            roster_model,
            client,
            first_person=is_first_person(chapter.text, ends),
        )
        rosters.append(roster)
        if narrator is not None:
            narrators[narrator] += 1
    characters = merge_rosters(tuple(rosters))
    # One narrator per book: chapters that disagree are outvoted rather than
    # producing a second narrating character.
    narrator_id = narrators.most_common(1)[0][0] if narrators else None
```

Add `from collections import Counter` and `from kenkui._characters.narration import is_first_person` to the imports, and `from kenkui._characters.infer import slugify` alongside the existing `infer` import.

Pass `narrator_id` into each `attribute_chapter` call:

```python
        chapter_spans, recent = attribute_chapter(
            chapter,
            characters,
            model_id,
            client=client,
            recent=recent,
            spans=extracted[chapter.id],
            narrator_id=narrator_id,
        )
```

- [ ] **Step 5: Run the test — it still fails on `attribute_chapter`**

Run: `uv run pytest tests/test_narration.py -k narrator -v --no-cov`
Expected: FAIL, `TypeError: attribute_chapter() got an unexpected keyword argument 'narrator_id'`. Task 5 adds it; leave this red and commit the prompt work only after Task 5 turns it green.

- [ ] **Step 6: Commit after Task 5 is green**

This task and Task 5 land together, because neither compiles without the other.

---

### Task 5: Attribute first-person lines to the narrator

**Files:**
- Modify: `src/kenkui/_characters/prompts.py` (`ATTRIBUTION_PROMPT`)
- Modify: `src/kenkui/_characters/attribution.py` (`_roster_block`, `_resolve`, `_answers`, `attribute_chapter`)
- Test: `tests/test_narration.py` (append)

**Interfaces:**
- Consumes: `narrator_id` from Task 4.
- Produces: `attribute_chapter(..., narrator_id: str | None = None)`; `_resolve(speaker: object, known: frozenset[str], narrator_id: str | None = None) -> str | None`.

- [ ] **Step 1: Write the failing test**

```python
from kenkui._characters.attribution import _resolve


def test_bare_narrator_resolves_to_the_narrating_character() -> None:
    assert _resolve("narrator", frozenset({"nieshka"}), "nieshka") == "nieshka"


def test_bare_narrator_is_unknown_without_a_narrator() -> None:
    assert _resolve("narrator", frozenset({"nieshka"}), None) is None


def test_a_pronoun_is_still_refused() -> None:
    assert _resolve("she", frozenset({"nieshka"}), "nieshka") is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_narration.py -k resolve -v --no-cov`
Expected: FAIL, `TypeError: _resolve() takes 2 positional arguments but 3 were given`

- [ ] **Step 3: Write the implementation**

In `src/kenkui/_characters/attribution.py`:

```python
NARRATOR = "narrator"


def _resolve(
    speaker: object, known: frozenset[str], narrator_id: str | None = None
) -> str | None:
    """Return a known character id, or None meaning unknown.

    A returned pronoun is rejected outright: the prompt forbids it, and a
    pronoun that slipped through would merge unrelated speakers into one voice.

    The bare word "narrator" is the one exception, and only when the book has
    one. Models shorten an id they are asked to reproduce, and a namespaced
    narrator id loses the answers it shortens; accepting the short word keeps
    them while leaving one id per person, so a narrating character's dialogue
    and their narration stay one voice.
    """
    if not isinstance(speaker, str):
        return None
    candidate = speaker.strip().lower()
    if not candidate or candidate == UNKNOWN:
        return None
    if candidate == NARRATOR:
        return narrator_id
    if candidate in PRONOUNS:
        return None
    return candidate if candidate in known else None
```

Mark the narrator in the roster block:

```python
def _roster_block(
    characters: Sequence[CharacterProfile], narrator_id: str | None = None
) -> str:
    return "\n".join(
        f'- {character.id}  (appears as "{character.display_name}")'
        + ("  [narrates this book]" if character.id == narrator_id else "")
        for character in characters
    )
```

Thread it through `_answers` and `attribute_chapter`:

```python
def _answers(
    model_id: str,
    prompt: str,
    client: Client | None,
    known: frozenset[str],
    narrator_id: str | None = None,
) -> dict[int, str | None]:
    ...
        if isinstance(quote_id, int) and not isinstance(quote_id, bool):
            answers[quote_id] = _resolve(item.get("speaker"), known, narrator_id)
```

and in `attribute_chapter`, add the keyword-only parameter `narrator_id: str | None = None`, pass it to `_roster_block(characters, narrator_id)` and to `_answers(model_id, prompt, client, known, narrator_id)`.

Extend `ATTRIBUTION_PROMPT` in `prompts.py`, after the existing pronoun rule:

```
- A character marked "narrates this book" tells it in the first person. A quote
  tagged "I said", "I asked" or "said I" is spoken by them: answer with their
  id. Do not answer "unknown" for those, and never answer with the pronoun.
```

- [ ] **Step 4: Run the whole suite**

Run: `uv run pytest -v` (the coverage gate applies)
Expected: PASS, including Task 4's test. If `tests/test_attribution.py` fails on `_roster_block` arity, it calls the helper directly — update the call, not the helper.

- [ ] **Step 5: Update the documentation**

`docs/architecture.md`, "Attribution as a resolved input", gains one paragraph:

```
A book narrated in the first person names its narrator during roster
inference, and attribution marks them in the roster it sends. Their id is an
ordinary character id, so their spoken lines and their narration differ only in
which voice casting gives them: `_castable` withholds the narrator voice from
every character, so a narrating character cannot be given the voice their own
narration uses.
```

- [ ] **Step 6: Commit both tasks**

```bash
git add src/kenkui/_characters/prompts.py src/kenkui/_characters/attribution.py \
        src/kenkui/_characters/__init__.py tests/test_narration.py docs/architecture.md
git commit --signoff -m "feat: attribute first-person dialogue to the narrator"
```

---

### Task 6: Unnamed role speakers (optional)

Ship only if the increment is wanted. A role read in the narrator's voice is an acceptable outcome — this is roughly 3% of a book's dialogue, and every earlier task stands without it.

Role ids are not stable across derivations, and that is fine: they are consumed by casting in the same run that produced them and are never written to a character list a reader sees.

**Files:**
- Modify: `src/kenkui/_characters/prompts.py` (`ATTRIBUTION_PROMPT`, `PROMPT_VERSION` → `characters-v3`)
- Modify: `src/kenkui/_characters/attribution.py` (`_resolve`)
- Modify: `src/kenkui/_characters/__init__.py` (`_measured`)
- Test: `tests/test_narration.py` (append)

**Interfaces:**
- Consumes: `_resolve` from Task 5.
- Produces: `_resolve(..., chapter_id: str | None = None)` returning `role:<slug>@<chapter_id>` for a role answer; `_measured` synthesising a `CharacterProfile` for each role id found in the spans.

- [ ] **Step 1: Write the failing test**

```python
from kenkui._characters.attribution import _resolve


def test_a_role_answer_is_scoped_to_its_chapter() -> None:
    resolved = _resolve("guard", frozenset({"rand"}), None, chapter_id="ch-1")
    assert resolved == "role:guard@ch-1"


def test_the_same_role_in_two_chapters_is_two_characters() -> None:
    first = _resolve("guard", frozenset(), None, chapter_id="ch-1")
    second = _resolve("guard", frozenset(), None, chapter_id="ch-2")
    assert first != second


def test_a_roster_character_still_wins_over_a_role() -> None:
    assert _resolve("rand", frozenset({"rand"}), None, chapter_id="ch-1") == "rand"

```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_narration.py -k role -v --no-cov`
Expected: FAIL, `TypeError: _resolve() got an unexpected keyword argument 'chapter_id'`

- [ ] **Step 3: Write the implementation**

In `attribution.py`, extend `_resolve` with a keyword-only `chapter_id: str | None = None`, and replace the final line:

```python
    if candidate in known:
        return candidate
    if chapter_id is not None and _ROLE.fullmatch(candidate):
        # A role names a speaker the text identifies without naming: "the
        # lookout", "the first man". It is scoped to the chapter because
        # chapter 40's guard is not chapter 12's, and the model answers the
        # bare word because an id it must reproduce gets shortened and lost.
        return f"role:{candidate}@{chapter_id}"
    return None
```

with, near the other module constants:

```python
# A role slug: lowercase words joined by hyphens, short enough to be a role
# rather than a sentence.
_ROLE = re.compile(r"[a-z]+(?:-[a-z]+){0,2}")
```

Pass `chapter_id=chapter.id` from `attribute_chapter` through `_answers`.

In `_characters/__init__.py`, `_measured` currently keeps only characters already on the roster. Roles are discovered during attribution, so synthesise them:

```python
    roles = {
        span.character_id
        for span in spans
        if span.character_id is not None and span.character_id.startswith("role:")
    }
    synthesised = tuple(
        CharacterProfile(
            id=role,
            display_name=role.removeprefix("role:").split("@")[0].replace("-", " "),
            gender=None,
            spoken_characters=volume.get(role, 0),
            chapter_ids=tuple(chapters.get(role, ())),
        )
        for role in sorted(roles)
    )
```

and return `(*named, *synthesised)` where the function currently returns its
tuple. Add `CharacterProfile` to the runtime imports — it is currently only
imported under `TYPE_CHECKING`.

Gender stays `None` for a role. The text rarely says, asking for it on every
attribution costs prompt space on the 97% of answers that are named characters,
and asking only sometimes is unreliable. Gender narrows the pool only under the
`gendered` method, so a null narrows nothing that was not already open.

Extend `ATTRIBUTION_PROMPT`:

```
- When the text identifies a speaker without naming them -- "the lookout in the
  bow", "the first man", "the innkeeper" -- answer with a short lowercase role
  word taken from the passage: "lookout", "first-man", "innkeeper". Prefer this
  over "unknown" whenever the text says who is speaking at all.
```

and bump `PROMPT_VERSION = "characters-v3"`.

- [ ] **Step 4: Run the whole suite**

Run: `uv run pytest -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/kenkui/_characters/prompts.py src/kenkui/_characters/attribution.py \
        src/kenkui/_characters/__init__.py tests/test_narration.py
git commit --signoff -m "feat: cast speakers the text identifies by role"
```

---

## Verification

After the last task shipped:

```bash
uv run ruff format --check .
uv run ruff check .
uv run mypy
uv run pytest
uv run mkdocs build --strict
```

All must pass. Coverage is enforced at 90% branch by `pytest` itself.

One check the suite cannot make: a book with no narrator and no roles must render byte-identically to before. Confirm by planning any existing single-voice fixture before and after and comparing the plan fingerprint — `tests/test_identity_stability.py` holds the pattern for this.
