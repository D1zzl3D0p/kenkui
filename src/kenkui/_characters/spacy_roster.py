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

from kenkui._characters.identity import (
    detect_titles,
    group_full_names,
    name_tokens,
    residue,
)
from kenkui._characters.infer import PRONOUNS, slugify
from kenkui._characters.narration import is_first_person
from kenkui._domain.casting import CharacterProfile
from kenkui.errors import ErrorCode, ModelError
from kenkui.observability import get_logger, log_event

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from kenkui._domain.grid import DialogueRange
    from kenkui.inspection import ChapterInspection

_LOGGER = get_logger(__name__)

SPACY_SCHEME = "spacy"
# The large pipeline: its parse is what the accuracy above was measured on.
# `spacy:en_core_web_sm` trades accuracy for size, `en_core_web_trf` the
# reverse.
DEFAULT_PIPELINE = "en_core_web_lg"

# Curly and modifier apostrophes are one character to a reader but three to a
# string comparison.
_APOSTROPHES = str.maketrans({"‘": "'", "’": "'", "ʼ": "'"})

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
# How far past a name to look for the pronoun that refers to it. The scan stops
# at the first gendered pronoun, and at any intervening proper noun -- once
# another name has come between, the pronoun is more likely theirs.
#
# This replaces a symmetric +/-40-token window that counted every gendered
# pronoun near a name. That measured how many men and women were in the scene
# rather than who the name was, so two characters who share their scenes
# cancelled each other out: on one novel the female lead scored feminine 2500
# to masculine 1269, a ratio of 1.97 against the 2.0 margin below, and so
# abstained -- which sends her to the whole voice pool and can hand her a
# masculine voice. The nearest-pronoun signal scores her 339 to 87, and also
# resolves the male lead, whom the window could not resolve at any width.
_GENDER_LOOKAHEAD = 12
# Votes required before a gender is claimed at all, and the margin the winner
# must hold over the loser. A character seen twice near "she" is not evidence.
_GENDER_MINIMUM = 3
_GENDER_MARGIN = 2

# Fallback only: claimants below this share of a bare name do not compete for
# it when breaking a tie between several possible full names.
_TIE_SHARE = 0.1

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
# Honorifics and kinship terms that state a gender outright. Read from the name
# span itself and the token before it, never from the stripped name: this is the
# strongest evidence a text offers about how a character should sound, and it
# costs nothing to collect.
#
# Kept deliberately apart from _TITLES, which _strip_titles consumes. Adding
# these there would rename "Aunt Vera" to "Vera" and "Mr. Neeson" to "Neeson",
# changing character ids and what the cast is keyed on -- a different change
# with a much wider blast radius than gendering them correctly.
_FEMININE_TITLES = frozenset(
    {
        "mrs",
        "ms",
        "miss",
        "madam",
        "madame",
        "mistress",
        "lady",
        "dame",
        "queen",
        "princess",
        "duchess",
        "countess",
        "baroness",
        "sister",
        "mother",
        "mom",
        "mum",
        "mama",
        "aunt",
        "auntie",
        "grandma",
        "grandmother",
        "granny",
        "widow",
    }
)
_MASCULINE_TITLES = frozenset(
    {
        "mr",
        "mister",
        "sir",
        "lord",
        "master",
        "king",
        "prince",
        "duke",
        "baron",
        "earl",
        "brother",
        "father",
        "dad",
        "papa",
        "uncle",
        "grandpa",
        "grandfather",
        "monsieur",
        "herr",
    }
)
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
    """Fold apostrophes, strip punctuation, the possessive, and honorifics."""
    name = name.translate(_APOSTROPHES)
    cleaned = name.strip().strip(" ,.;:!?-—'").replace("'s", "")
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
    """Group adjacent proper nouns into one name, across an unspaced hyphen.

    Without this, "Padan Fain" becomes two candidates, neither merges, and the
    fullest form -- which is what attribution asks the model to answer with --
    never exists at all.
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
        # Honorific evidence, kept apart from the pronoun tally because it
        # outranks it rather than adding to it.
        self.title_gender: dict[str, Counter[str]] = defaultdict(Counter)
        # Per chapter, for first-person detection: a narrator is addressed
        # without ever speaking, and that shows up only chapter by chapter.
        self.addressed_in: dict[str, Counter[str]] = defaultdict(Counter)
        self.speaking_in: dict[str, Counter[str]] = defaultdict(Counter)
        self.flag_hosts: dict[str, set[str]] = defaultdict(set)
        self.titled_forms: dict[str, str] = {}
        self.title_hosts: dict[str, Counter[str]] = defaultdict(Counter)
        self.the_det: Counter[str] = Counter()
        self.prep: dict[str, Counter[str]] = defaultdict(Counter)

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

    def rename(self, source: str, target: str | None) -> None:
        """Move every tally for ``source`` onto ``target``, or drop it."""
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


def _usable(name: str) -> bool:
    return _plausible(name) and name.lower() not in PRONOUNS


def _only_titles(name: str) -> bool:
    return all(token.lower().strip(".") in _TITLES for token in name.split())


def _attached_title(raw: str, before: str) -> str | None:
    """Return the title opening a mention, or standing just before it."""
    words = raw.split()
    leading = words[0].lower().strip(".") if words else ""
    if leading in _TITLES:
        return leading
    preceding = before.strip(".")
    return preceding if preceding in _TITLES else None


def _leading_title(raw: str, cleaned: str) -> str | None:
    """Return the title word a mention opens with, preserving its spelling."""
    words = raw.strip().split()
    if not words:
        return None
    head = words[0].strip(".,")
    if head.lower() in _TITLES and head.lower() != cleaned.lower():
        return head
    return None


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
    """Keep couples apart and fold every other titled form into its bare name."""
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
    dialogue: Mapping[str, Sequence[DialogueRange]],
    *,
    pipeline: str = DEFAULT_PIPELINE,
) -> _Signals:
    """Parse the book once and return every tally the roster reads."""
    nlp = _load(pipeline)
    signals = _Signals()
    for chapter in chapters:
        bounds = [(span.start, span.end) for span in dialogue.get(chapter.id, ())]
        _scan(nlp(chapter.text), chapter.id, bounds, signals)
    _fold_titles(signals)
    return signals


def _gender_signals(
    doc: Any,  # noqa: ANN401 - a spaCy Doc; the package ships no stubs
    span: Any,  # noqa: ANN401 - a spaCy Span
    name: str,
    into: _Signals,
) -> None:
    """Record the honorific attached to one mention, and the pronoun after it."""
    words = span.text.split()
    leading = words[0].lower().strip(".") if words else ""
    preceding = doc[span.start - 1].lower_.strip(".") if span.start else ""
    for word in (leading, preceding):
        if word in _FEMININE_TITLES:
            into.title_gender[name]["feminine"] += 1
            break
        if word in _MASCULINE_TITLES:
            into.title_gender[name]["masculine"] += 1
            break
    for other in doc[span.end : span.end + _GENDER_LOOKAHEAD]:
        if other.pos_ == "PROPN":
            break  # another name came between; the pronoun is likely theirs
        if other.lower_ in _FEMININE:
            into.gender[name]["feminine"] += 1
            break
        if other.lower_ in _MASCULINE:
            into.gender[name]["masculine"] += 1
            break


def _majority(votes: Counter[str]) -> str | None:
    """Pick a gender only on a clear majority, else leave it unsourced."""
    top = votes.most_common(1)
    if not top or top[0][1] < _GENDER_MINIMUM:
        return None
    winner, count = top[0]
    other = votes["masculine" if winner == "feminine" else "feminine"]
    return winner if count >= _GENDER_MARGIN * other else None


def _gender_of(pronouns: Counter[str], titles: Counter[str]) -> str | None:
    """Decide from honorifics first, then from the pronouns that follow the name.

    An honorific states the gender outright, so it wins wherever it clears the
    threshold. The pronouns near "Aunt Vera" are frequently about whoever she
    is speaking to, and must not be allowed to overturn the word "Aunt".

    An unsourced gender is an admission of ignorance, and `casting.candidates`
    answers it by offering the whole pool. A wrong guess is worse: it silently
    restricts a character to voices that sound wrong for them.
    """
    return _majority(titles) or _majority(pronouns)


def _contradicts(
    name: str, host: str, canonical: Mapping[str, str], signals: _Signals
) -> bool:
    """Report whether a titled name's gender contradicts its host's pronouns."""
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
    """Return who a bare title belongs to, across stripped and kept forms."""
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
    """Fold surface forms conservatively and report deliberately removed names."""
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
    for name in sorted(
        set(kept) - set(fulls), key=lambda item: (bool(name_tokens(item) & titles), item)
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
                name if titled and _contradicts(name, host, canonical, signals) else host
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
            canonical[name] = name
        else:
            removed[name] = "title held by several"
    return canonical, removed


def _narrator_of(
    chapters: Sequence[ChapterInspection],
    dialogue: Mapping[str, Sequence[DialogueRange]],
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
            [span.end for span in dialogue.get(chapter.id, ())],
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
    dialogue: Mapping[str, Sequence[DialogueRange]],
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
    signals = collect_signals(chapters, dialogue, pipeline=pipeline)

    kept = {
        name
        for name, count in signals.mentions.items()
        if count >= MIN_MENTIONS and signals.evidence(name) >= MIN_EVIDENCE
    }
    canonical, _ = _fold_names(signals, kept, fallback=True)

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
                "title_gender": Counter(),
            },
        )
        row["aliases"].add(name)
        row["evidence"] += signals.evidence(name)
        row["chapter_ids"] |= signals.chapters[name]
        row["gender"].update(signals.gender[name])
        row["title_gender"].update(signals.title_gender[name])

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
                    gender=_gender_of(row["gender"], row["title_gender"]),
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
