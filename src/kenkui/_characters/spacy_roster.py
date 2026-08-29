"""Derive a character roster from a dependency parse rather than a model.

Selected by roster model id: ``infer_characters("spacy")``, or
``"spacy:en_core_web_sm"`` to name a pipeline. Everything else is left to the
model boundary in `llm.py`, so this module is inert unless a caller asks for it.

Why this exists next to the model roster, from `evals/attribution/README.md`:

* **It is model-independent.** The LLM roster asks each model to invent
  character ids, and `attribution._resolve` then rejects any speaker not in
  that model's own roster. Two models cannot be compared against two different
  id vocabularies, and a book re-read by a different model re-casts the series.
* **It halves the spend.** The roster pass sends the whole book through a model
  before attribution sends it again. This pass is free and runs offline.
* **It is repeatable.** Deterministic over the same text and pipeline, which is
  what the plan fingerprint requires.

**Entity labels are not used, and NER is not loaded.** Measured on a fantasy
novel, `en_core_web_sm` tagged the protagonist as ORG 459 times against PERSON
6, and never tagged two other leads as PERSON at all -- invented proper nouns
sit far outside NER's training data, and filtering on the label deleted the
three main characters. The dependency parser answers the real question
directly: a speaker is the ``nsubj`` of a speech verb.

**Recall is preferred over precision, deliberately.** A spurious roster entry
costs a little prompt space and simply never gets chosen. A missing character
forces every line they speak to "unknown" and no model can recover it, so
candidates are admitted on any of several animacy signals, not only on speaking.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import TYPE_CHECKING, Any

from kenkui._characters.identity import group_full_names, resolve_short_forms
from kenkui._characters.infer import PRONOUNS, slugify
from kenkui._characters.narration import is_first_person
from kenkui._domain.casting import CharacterProfile
from kenkui.errors import ErrorCode, ModelError
from kenkui.observability import get_logger, log_event

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from kenkui._characters.quotes import TextSpan
    from kenkui.inspection import ChapterInspection

_LOGGER = get_logger(__name__)

SPACY_SCHEME = "spacy"
# The large pipeline: its parse is what the accuracy above was measured on.
# `spacy:en_core_web_sm` trades accuracy for size, `en_core_web_trf` the
# reverse.
DEFAULT_PIPELINE = "en_core_web_lg"

# A name must be mentioned this often before any evidence is weighed. Below it,
# a parse error and a walk-on are indistinguishable.
MIN_MENTIONS = 3
MIN_EVIDENCE = 1
# Effectively non-binding for a novel, and deliberately so. The old cap of 120
# was sized for a prompt that carried the whole book's roster into every
# chapter; `_chapter_roster` now hands each chapter only the characters it
# contains -- a median of 17 to 23 on the books this was measured against --
# so the book-wide total no longer sets the prompt size. Ranking still decides
# what a book over the cap would lose, and a bound is kept rather than removed
# because an unbounded roster is a bound on nothing at all: a pathological
# source should degrade, not exhaust memory.
MAX_CHARACTERS = 2000
# spaCy's default cap is a document length no novel chapter approaches, but a
# concatenated book does.
MAX_PARSE_CHARACTERS = 5_000_000
# How far either side of a name to look for a gendering pronoun.
_GENDER_WINDOW = 40
# Votes required before a gender is claimed at all, and the margin the winner
# must hold over the loser. A character seen twice near "she" is not evidence.
_GENDER_MINIMUM = 3
_GENDER_MARGIN = 2

# Speech verbs by lemma. The parser resolves inflection, so "said"/"says"/
# "saying" all reduce to "say". Far wider than the obvious set: novels carry
# dialogue on verbs of volume, manner, emotion, and continuation, and every
# verb missing here is a character who may go undiscovered.
_SPEECH_LEMMAS = frozenset(
    {
        # plain
        "say",
        "speak",
        "tell",
        "ask",
        "answer",
        "reply",
        "respond",
        "state",
        # volume
        "shout",
        "yell",
        "roar",
        "bellow",
        "call",
        "cry",
        "scream",
        "shriek",
        "whisper",
        "murmur",
        "mutter",
        "mumble",
        "breathe",
        "hiss",
        # manner
        "drawl",
        "snap",
        "bark",
        "growl",
        "grunt",
        "snort",
        "sniff",
        "purr",
        "stammer",
        "stutter",
        "blurt",
        "gasp",
        "pant",
        "wheeze",
        "croak",
        # emotion
        "laugh",
        "chuckle",
        "giggle",
        "grin",
        "smile",
        "sigh",
        "sob",
        "wail",
        "moan",
        "groan",
        "complain",
        "grumble",
        "protest",
        "object",
        "snarl",
        # discourse
        "add",
        "begin",
        "continue",
        "go",
        "put",
        "offer",
        "observe",
        "remark",
        "note",
        "comment",
        "declare",
        "announce",
        "explain",
        "insist",
        "repeat",
        "agree",
        "admit",
        "confess",
        "concede",
        "counter",
        "argue",
        "urge",
        "demand",
        "order",
        "command",
        "beg",
        "plead",
        "promise",
        "warn",
        "suggest",
        "wonder",
        "muse",
        "reflect",
        "conclude",
        "finish",
        "interrupt",
        "correct",
        "prompt",
        "press",
        "venture",
        "allow",
        "acknowledge",
        "assure",
        "remind",
        "echo",
        "quote",
        "recite",
        "read",
        "sing",
        "chant",
    }
)

# Nouns that, possessed by a name, imply a person rather than a place.
_ANIMATE_NOUNS = frozenset(
    {
        "face",
        "voice",
        "eye",
        "eyes",
        "hand",
        "hands",
        "head",
        "hair",
        "mouth",
        "arm",
        "arms",
        "shoulder",
        "shoulders",
        "back",
        "chest",
        "finger",
        "fingers",
        "foot",
        "feet",
        "leg",
        "legs",
        "smile",
        "frown",
        "glare",
        "gaze",
        "stare",
        "expression",
        "mind",
        "thought",
        "thoughts",
        "heart",
        "stomach",
        "throat",
        "breath",
        "cheek",
        "jaw",
        "brow",
        "neck",
        "skin",
        "father",
        "mother",
        "brother",
        "sister",
        "son",
        "daughter",
        "wife",
        "husband",
        "horse",
        "cloak",
        "coat",
        "sword",
        "knife",
        "bow",
        "purse",
    }
)

# Honorifics stripped from either end, so "Master Lan" and "Lan" are one id.
_TITLES = frozenset(
    {
        "lord",
        "lady",
        "master",
        "mistress",
        "captain",
        "lieutenant",
        "sergeant",
        "colonel",
        "major",
        "general",
        "admiral",
        "queen",
        "king",
        "prince",
        "princess",
        "duke",
        "duchess",
        "baron",
        "count",
        "countess",
        "sir",
        "dame",
        "elder",
        "mayor",
        "doctor",
        "professor",
        "father",
        "mother",
        "brother",
        "sister",
        "mister",
        "madam",
        "madame",
        "monsieur",
    }
)

# Common nouns a parse reports as PROPN at the start of a sentence, and which
# are never a character id on their own. Deliberately generic: a list tuned to
# one book's invented vocabulary would be wrong for every other book, and the
# mention and evidence thresholds are what filter book-specific noise.
_NOT_A_NAME = frozenset(
    {
        "man",
        "men",
        "woman",
        "women",
        "girl",
        "boy",
        "child",
        "children",
        "people",
        "person",
        "one",
        "two",
        "three",
        "four",
        "five",
        "sir",
        "madam",
        "lord",
        "lady",
        "master",
        "mistress",
        "captain",
        "king",
        "queen",
        "prince",
        "princess",
        "father",
        "mother",
        "brother",
        "sister",
        "god",
        "lord god",
        "light",
        "shadow",
        "death",
        "life",
        "time",
        "day",
        "night",
        "morning",
        "evening",
        "yes",
        "no",
        "well",
        "oh",
        "ah",
    }
)

_PHRASE_STARTERS = frozenset(
    {
        "the",
        "a",
        "an",
        "this",
        "that",
        "these",
        "those",
        "his",
        "her",
        "their",
        "my",
        "your",
        "our",
        "its",
        "some",
        "any",
        "no",
        "every",
        "all",
    }
)

_FEMININE = frozenset({"she", "her", "hers", "herself"})
_MASCULINE = frozenset({"he", "him", "his", "himself"})
_MIN_NAME_CHARACTERS = 3
_MAX_NAME_TOKENS = 4


def pipeline_for(model_id: str) -> str | None:
    """Return the spaCy pipeline a roster model id names, or None.

    Matched case-insensitively: "spaCy" is the spelling a caller reaches for
    first, and refusing it on capitalisation alone would be a puzzle rather
    than a guardrail. Anything not using the scheme returns None and is left
    to the model boundary.
    """
    head, separator, tail = model_id.strip().partition(":")
    if head.strip().lower() != SPACY_SCHEME:
        return None
    if not separator:
        return DEFAULT_PIPELINE
    return tail.strip() or None


def is_spacy(model_id: str) -> bool:
    """Whether this roster model id selects the spaCy pass."""
    return model_id.strip().lower().partition(":")[0].strip() == SPACY_SCHEME


def _load(pipeline: str) -> Any:  # noqa: ANN401 - spaCy ships no stubs
    """Load one spaCy pipeline, turning both unavailabilities into ModelError.

    NER and the text categoriser are excluded: neither is read, and both are
    the expensive components. The lemmatizer is REQUIRED -- excluding it
    silently empties ``token.lemma_``, every speech verb stops matching, and
    the result is an empty roster with no error at all.
    """
    try:
        import spacy  # noqa: PLC0415 - heavy, and optional
    except ImportError as error:
        raise ModelError(ErrorCode.SPACY_PACKAGE_MISSING) from error
    try:
        loaded = spacy.load(pipeline, exclude=["ner", "textcat"])
    except (OSError, ValueError) as error:
        # spaCy raises OSError for a pipeline that is not installed. Nothing
        # from the caller's text reaches the message.
        raise ModelError(ErrorCode.SPACY_PIPELINE_MISSING) from error
    loaded.max_length = MAX_PARSE_CHARACTERS
    return loaded


def _strip_titles(name: str) -> str:
    """Remove honorifics from both ends so one person is one id.

    Adjacent proper nouns give "Master Lan" and "Lady Moiraine", which slug to
    different ids than "Lan" and "Moiraine" and split one character into three
    roster entries competing for the same voice.
    """
    tokens = name.split()
    while tokens and tokens[0].lower().strip(".") in _TITLES:
        tokens = tokens[1:]
    while tokens and tokens[-1].lower().strip(".") in _TITLES:
        tokens = tokens[:-1]
    return " ".join(tokens) if tokens else name


def _clean(name: str) -> str:
    """Strip surrounding punctuation, the possessive, and honorifics."""
    cleaned = name.strip().strip(" ,.;:!?-\u2014\u2019'").replace("\u2019s", "")
    return _strip_titles(cleaned)


def _plausible(name: str) -> bool:
    """Whether a cleaned proper-noun span could be somebody's name."""
    if not name or len(name) < _MIN_NAME_CHARACTERS or name.lower() in _NOT_A_NAME:
        return False
    tokens = name.split()
    if len(tokens) > _MAX_NAME_TOKENS or tokens[0].lower() in _PHRASE_STARTERS:
        return False
    return all(
        token[:1].isupper() and not token.isupper()
        for token in tokens
        if token.isalpha()
    )


def _addressed(token: Any) -> bool:  # noqa: ANN401 - a spaCy Token
    """Report whether this name is being spoken *to* rather than mentioned.

    Direct address is set off by punctuation ("Rand, look out", "run, Mat!") or
    sits at the very start or end of the quoted run. Counting every proper noun
    inside a quote instead let places people merely discuss score as though
    they had been addressed.
    """
    before = token.nbor(-1) if token.i else None
    after = token.nbor(1) if token.i + 1 < len(token.doc) else None
    if token.dep_ == "npadvmod" and token.head.pos_ == "VERB":
        return True
    return bool(
        (before is not None and before.text in {",", "“", '"', "!", "?"})
        or (after is not None and after.text in {",", "”", '"', "!", "?"})
    )


def _proper_noun_spans(doc: Any) -> list[Any]:  # noqa: ANN401 - a spaCy Doc
    """Group adjacent proper nouns into one name.

    Without this, "Padan Fain" becomes two candidates, neither merges, and the
    fullest form -- which is what attribution asks the model to answer with --
    never exists at all.
    """
    spans = []
    start: int | None = None
    for token in doc:
        if token.pos_ == "PROPN":
            start = token.i if start is None else start
            continue
        if start is not None:
            spans.append(doc[start : token.i])
            start = None
    if start is not None:
        spans.append(doc[start:])
    return spans


class _Signals:
    """Per-name tallies, accumulated across the whole book."""

    def __init__(self) -> None:
        self.mentions: Counter[str] = Counter()
        self.speech: Counter[str] = Counter()  # nsubj of a speech verb: speaks
        self.agency: Counter[str] = Counter()  # nsubj of any verb: acts
        self.vocative: Counter[str] = Counter()  # named in dialogue: addressed
        self.animate: Counter[str] = Counter()  # possesses a body part or kin
        self.titled: Counter[str] = Counter()  # preceded by Lord/Lady/Master
        self.chapters: dict[str, set[str]] = defaultdict(set)
        self.gender: dict[str, Counter[str]] = defaultdict(Counter)
        # Per chapter, for first-person detection: a narrator is addressed
        # without ever speaking, and that shows up only chapter by chapter.
        self.addressed_in: dict[str, Counter[str]] = defaultdict(Counter)
        self.speaking_in: dict[str, Counter[str]] = defaultdict(Counter)

    def evidence(self, name: str) -> int:
        """Weighted animacy evidence: how strongly this name denotes a person.

        Weighted because the signals differ in how much they prove. Speaking
        is decisive, being addressed inside dialogue is nearly as good, and
        acting or owning a face is suggestive.
        """
        return (
            3 * self.speech[name]
            + 2 * self.vocative[name]
            + 2 * self.titled[name]
            + self.animate[name]
            + self.agency[name]
        )


def _scan(  # noqa: C901 - one branch per signal, and they are independent
    doc: Any,  # noqa: ANN401 - a spaCy Doc; the package ships no stubs
    chapter_id: str,
    quote_bounds: Sequence[tuple[int, int]],
    into: _Signals,
) -> None:
    """Accumulate every signal one parsed chapter carries."""
    for span in _proper_noun_spans(doc):
        token = span[0]
        name = _clean(span.text)
        if not _plausible(name) or name.lower() in PRONOUNS:
            continue
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
        window = doc[max(0, token.i - _GENDER_WINDOW) : token.i + _GENDER_WINDOW]
        for other in window:
            if other.lower_ in _FEMININE:
                into.gender[name]["feminine"] += 1
            elif other.lower_ in _MASCULINE:
                into.gender[name]["masculine"] += 1


def _vote_gender(votes: Counter[str]) -> str | None:
    """Pick a gender only on a clear majority, else leave it unsourced.

    An unsourced gender is an admission of ignorance, and `casting.candidates`
    answers it by offering the whole pool. A wrong guess is worse: it silently
    restricts a character to voices that sound wrong for them.
    """
    top = votes.most_common(1)
    if not top or top[0][1] < _GENDER_MINIMUM:
        return None
    winner, count = top[0]
    other = votes["masculine" if winner == "feminine" else "feminine"]
    return winner if count >= _GENDER_MARGIN * other else None


def _canonical_names(signals: _Signals, kept: set[str]) -> dict[str, str]:
    """Fold the surface forms that denote one person onto a single name.

    Full names first, so "Moiraine Sedai" and "Moiraine Aes Sedai" become one
    entity before any short form is offered to them. A bare form then belongs
    to a full name only when exactly one entity claims it: "Tam" reaches "Tam
    al'Thor" unopposed, while "Charles" is claimed by both Charles Hayter and
    Charles Musgrove, and merging it into either would put the other's lines
    in the wrong voice. An ambiguous short form is dropped rather than kept as
    its own entry, which would split one person across two voices.
    """
    fulls = sorted(
        (name for name in kept if len(name.split()) > 1),
        key=lambda name: (-signals.mentions[name], name),
    )
    entity = group_full_names(fulls)
    shorts = sorted(name for name in kept if len(name.split()) == 1)
    resolved = resolve_short_forms(shorts, entity)
    canonical = {name: entity[name] for name in fulls}
    canonical.update(resolved.assigned)
    return canonical


def _narrator_of(
    chapters: Sequence[ChapterInspection],
    dialogue: Mapping[str, Sequence[TextSpan]],
    signals: _Signals,
    canonical: Mapping[str, str],
    roster_ids: frozenset[str],
) -> str | None:
    """Name the first-person narrator, when there is one.

    Invisible to everything above: the narration says "I said", so the narrator
    governs no speech verb. What identifies them is the mismatch -- across
    first-person chapters they are addressed by name far more often than they
    are seen to speak, because their own speech carries no name at all.

    Returned only when they are already on the roster. A narrator who is never
    named cannot be cast, and inventing an entry would put an unvouched id
    into the plan fingerprint.
    """
    first_person = [
        chapter.id
        for chapter in chapters
        if is_first_person(
            chapter.text,
            [span.end for span in dialogue.get(chapter.id, ()) if span.is_dialogue],
        )
    ]
    if not first_person:
        return None
    addressed: Counter[str] = Counter()
    spoke: Counter[str] = Counter()
    for chapter_id in first_person:
        for name, count in signals.addressed_in[chapter_id].items():
            if name in canonical:
                addressed[slugify(canonical[name])] += count
        for name, count in signals.speaking_in[chapter_id].items():
            if name in canonical:
                spoke[slugify(canonical[name])] += count
    candidates = [
        (count - spoke[character_id], character_id)
        for character_id, count in addressed.items()
        if character_id in roster_ids and count > spoke[character_id]
    ]
    if not candidates:
        return None
    # Sorted by id as well as margin, so a tie cannot depend on dict order.
    margin, character_id = max(candidates, key=lambda item: (item[0], item[1]))
    return character_id if margin > 0 else None


def infer_roster(
    chapters: Sequence[ChapterInspection],
    dialogue: Mapping[str, Sequence[TextSpan]],
    *,
    pipeline: str = DEFAULT_PIPELINE,
) -> tuple[tuple[CharacterProfile, ...], str | None]:
    """Return the whole book's roster and its first-person narrator, if any.

    One pass over the book rather than one per chapter: a name is a person on
    the evidence of everywhere they appear, and the per-chapter rosters the
    model pass produces exist only because a model cannot hold a book at once.

    ``spoken_characters`` is left at zero here, exactly as the model roster
    leaves it. It is filled in later, once spans have been attributed.
    """
    nlp = _load(pipeline)
    signals = _Signals()
    for chapter in chapters:
        bounds = [
            (span.start, span.end)
            for span in dialogue.get(chapter.id, ())
            if span.is_dialogue
        ]
        _scan(nlp(chapter.text), chapter.id, bounds, signals)

    kept = {
        name
        for name, count in signals.mentions.items()
        if count >= MIN_MENTIONS and signals.evidence(name) >= MIN_EVIDENCE
    }
    canonical = _canonical_names(signals, kept)

    merged: dict[str, dict[str, Any]] = {}
    for name in sorted(canonical):
        target = canonical[name]
        row = merged.setdefault(
            target,
            {
                "display_name": target,
                "aliases": set(),
                "evidence": 0,
                "chapter_ids": set(),
                "gender": Counter(),
            },
        )
        row["aliases"].add(name)
        row["evidence"] += signals.evidence(name)
        row["chapter_ids"] |= signals.chapters[name]
        row["gender"].update(signals.gender[name])

    # Ranked by evidence to apply the cap, then returned sorted by id: the
    # plan fingerprint requires a total order that content alone decides.
    ranked = sorted(
        merged.values(), key=lambda row: (-row["evidence"], row["display_name"])
    )[:MAX_CHARACTERS]
    roster = tuple(
        sorted(
            (
                CharacterProfile(
                    id=slugify(row["display_name"]),
                    display_name=row["display_name"],
                    gender=_vote_gender(row["gender"]),
                    spoken_characters=0,
                    chapter_ids=tuple(sorted(row["chapter_ids"])),
                    aliases=tuple(sorted(row["aliases"])),
                )
                for row in ranked
                if slugify(row["display_name"])
            ),
            key=lambda character: character.id,
        )
    )
    narrator = _narrator_of(
        chapters,
        dialogue,
        signals,
        canonical,
        frozenset(character.id for character in roster),
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
