"""Stage 2: Named-entity extraction and deterministic alias clustering.

spaCy's ``en_core_web_sm`` model finds PERSON entity strings.  A pure-Python
word-overlap heuristic then groups aliases under a single canonical name —
no LLM call needed for this stage, which makes it faster and fully
reproducible.

Alias clustering heuristic
--------------------------
Names are sorted longest-first (by word count, then character length).
For each unclaimed name N, every other unclaimed name A is merged into N's
group when ALL *significant* words of A (i.e. non-title, non-particle words
of length > 1) appear in the significant words of N.

Examples that merge correctly:
    "Harry" + "Harry Potter"  → canonical "Harry Potter"
    "Mr. Potter"              → canonical "Harry Potter"  (sig word: "potter")
    "Hermione" + "Hermione Granger" → canonical "Hermione Granger"
    "Ron" + "Ron Weasley"     → canonical "Ron Weasley"
    "Albus Dumbledore" + "Dumbledore" → canonical "Albus Dumbledore"

Known limitation
----------------
If two distinct characters share a first name (e.g. "Tom Sawyer" and
"Tom Robinson" in the same book) the bare form "Tom" will be merged into
whichever full name appears first in the sorted list.  This is an acceptable
trade-off: it is better to unify a name than to split one character into
two separate voice tracks.

Public API
----------
extract_person_names(text, nlp)                        → list[str]
build_roster(text, nlp)                                → CharacterRoster
infer_gender_pronouns(canonical, aliases, text)        → str
"""

from __future__ import annotations

import logging
import re
from typing import TYPE_CHECKING

from .models import AliasGroup, CharacterRoster

if TYPE_CHECKING:
    from .llm import LLMClient

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Word filtering constants
# ---------------------------------------------------------------------------

_TITLES: frozenset[str] = frozenset({
    "mr", "mrs", "ms", "miss", "dr", "prof", "lord", "lady",
    "sir", "captain", "capt", "gen", "sgt", "cpl", "pvt",
    "rev", "fr", "br", "sr", "jr", "ii", "iii", "iv",
})

_PARTICLES: frozenset[str] = frozenset({
    "the", "a", "an", "of", "von", "de", "le", "la", "du", "van",
    "den", "der", "des", "di", "da", "del", "bin", "bint",
})


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _significant_words(name: str) -> list[str]:
    """Return lowercase significant words from *name*, stripping titles/particles.

    Punctuation attached to the word token (period, comma, apostrophe) is
    stripped before comparison.
    """
    words: list[str] = []
    for raw in name.lower().split():
        w = raw.strip(".,'-\u2019\u2018")
        if len(w) > 1 and w not in _TITLES and w not in _PARTICLES:
            words.append(w)
    return words


def _is_alias_of(candidate: str, canonical: str) -> bool:
    """Return True when *candidate* is plausibly a short form of *canonical*.

    Both names must have at least one significant word.  Every significant
    word of the *shorter* name must appear in the significant word set of
    the *longer* canonical name.
    """
    cand_words = _significant_words(candidate)
    canon_words = set(_significant_words(canonical))
    if not cand_words or not canon_words:
        return False
    # Guard: candidate must be strictly shorter (by sig-word count).
    if len(cand_words) >= len(canon_words):
        return False
    return all(w in canon_words for w in cand_words)


def _cluster_by_heuristic(names: list[str]) -> list[AliasGroup]:
    """Group *names* into alias clusters using word-overlap heuristics.

    Returns a list of ``AliasGroup`` objects sorted by canonical name length
    descending (most specific characters first).
    """
    # Deduplicate and sort: most words first, then longest string first.
    unique = sorted(
        {n.strip() for n in names if n.strip()},
        key=lambda n: (len(n.split()), len(n)),
        reverse=True,
    )

    groups: list[AliasGroup] = []
    claimed: set[str] = set()

    for name in unique:
        if name in claimed:
            continue
        aliases: list[str] = [name]
        for other in unique:
            if other == name or other in claimed:
                continue
            if _is_alias_of(other, name):
                aliases.append(other)
                claimed.add(other)
        claimed.add(name)
        groups.append(AliasGroup(canonical=name, aliases=aliases))

    return groups


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def extract_person_names(text: str, nlp) -> list[str]:
    """Run spaCy over *text* and return deduplicated PERSON entity strings."""
    doc = nlp(text)
    seen: set[str] = set()
    names: list[str] = []
    for ent in doc.ents:
        if ent.label_ == "PERSON" and ent.text not in seen:
            seen.add(ent.text)
            names.append(ent.text)
    return names


def build_roster(text: str, nlp) -> CharacterRoster:
    """Extract PERSON names from *text* and cluster aliases into a roster.

    Uses spaCy for NER and a deterministic word-overlap heuristic for
    clustering — no LLM call is made at this stage.

    Returns an empty ``CharacterRoster`` when no names are found.
    """
    names = extract_person_names(text, nlp)
    if not names:
        logger.debug("build_roster: no PERSON entities found")
        return CharacterRoster(characters=[])

    logger.debug("build_roster: %d raw names → clustering", len(names))
    groups = _cluster_by_heuristic(names)
    logger.debug(
        "build_roster: %d canonical characters (from %d raw names)",
        len(groups), len(names),
    )
    return CharacterRoster(characters=groups)


# ---------------------------------------------------------------------------
# Gender inference
# ---------------------------------------------------------------------------

_HE_RE = re.compile(r"\b(he|him|his)\b", re.IGNORECASE)
_SHE_RE = re.compile(r"\b(she|her|hers)\b", re.IGNORECASE)
_THEY_RE = re.compile(r"\b(they|them|their|theirs)\b", re.IGNORECASE)

# Window (in characters) around each name mention to scan for pronouns.
# 300 chars covers roughly the preceding and following 1-2 sentences.
_PRONOUN_WINDOW = 300


def infer_gender_pronouns(
    canonical: str,
    aliases: list[str],
    text: str,
) -> str:
    """Infer the most likely pronoun set for a character from surrounding text.

    For each mention of the character's name (canonical or any alias), counts
    gendered pronouns in a ±300-character window.  Returns the majority set as
    ``"he/him"``, ``"she/her"``, ``"they/them"``, or ``""`` (no data or tie).

    Args:
        canonical: The character's canonical name.
        aliases:   All known name forms including the canonical.
        text:      The full book text to search.

    Returns:
        Pronoun string suitable for ``CharacterInfo.gender_pronoun``.
    """
    name_forms = {canonical} | set(aliases)
    escaped = [re.escape(n) for n in name_forms if n]
    if not escaped:
        return ""
    name_re = re.compile(r"\b(?:" + "|".join(escaped) + r")\b", re.IGNORECASE)

    he_count = she_count = they_count = 0

    for m in name_re.finditer(text):
        start = max(0, m.start() - _PRONOUN_WINDOW)
        end = min(len(text), m.end() + _PRONOUN_WINDOW)
        window = text[start:end]
        he_count += len(_HE_RE.findall(window))
        she_count += len(_SHE_RE.findall(window))
        they_count += len(_THEY_RE.findall(window))

    if she_count > he_count and she_count > they_count:
        return "she/her"
    if he_count > she_count and he_count > they_count:
        return "he/him"
    if they_count > 0 and they_count >= he_count and they_count >= she_count:
        return "they/them"
    return ""


# ---------------------------------------------------------------------------
# LLM roster cleanup passes
# ---------------------------------------------------------------------------

_DEDUP_PROMPT = """\
The following character names were extracted from a novel. Some entries may
be different surface forms of the same character (nickname, shortened form,
contraction, or partial name).

Names:
{name_lines}

TASK: Identify groups of names that refer to the same character.
For each group with more than one entry, specify:
- "canonical": the most complete/formal name to keep
- "duplicates": the other names in the group that should be merged into it

RULES:
- Only merge when you are highly confident they are the same person.
- Do NOT merge characters who merely share a first name (e.g. two different "Johns").
- If a name stands alone with no apparent duplicate, omit it entirely.
- Use the EXACT name strings from the list above.

Return ONLY the JSON.
"""


def deduplicate_roster_with_llm(
    roster: "CharacterRoster",
    llm: "LLMClient",
) -> "CharacterRoster":
    """Merge canonical entries that refer to the same character via a single LLM call.

    Catches nickname contractions and other alias forms that word-overlap
    clustering misses (e.g. "Mat" → "Matrim Cauthon").  Each merged group's
    aliases are combined under the surviving canonical; gender is taken from
    whichever entry has a non-empty value.

    Returns *roster* unchanged on any LLM error or empty result.
    """
    from .models import CanonicalMergeResult

    if len(roster.characters) < 2:
        return roster

    name_lines = "\n".join(f"- {g.canonical}" for g in roster.characters)
    try:
        result: CanonicalMergeResult = llm.generate(
            _DEDUP_PROMPT.format(name_lines=name_lines),
            CanonicalMergeResult,
        )

        if not result.merges:
            return roster

        canonical_set = {g.canonical for g in roster.characters}
        absorb_into: dict[str, str] = {}
        for entry in result.merges:
            if entry.canonical not in canonical_set:
                continue
            for dup in entry.duplicates:
                if dup in canonical_set and dup != entry.canonical:
                    absorb_into[dup] = entry.canonical

        if not absorb_into:
            return roster

        group_by_canonical = {g.canonical: g for g in roster.characters}
        merged = 0
        for dup_canonical, survivor_canonical in absorb_into.items():
            dup_group = group_by_canonical.get(dup_canonical)
            survivor_group = group_by_canonical.get(survivor_canonical)
            if not dup_group or not survivor_group:
                continue
            for alias in [dup_canonical] + dup_group.aliases:
                if alias not in survivor_group.aliases:
                    survivor_group.aliases.append(alias)
            if not survivor_group.gender and dup_group.gender:
                survivor_group.gender = dup_group.gender
            merged += 1

        remaining = [g for g in roster.characters if g.canonical not in absorb_into]
        logger.info(
            "deduplicate_roster_with_llm: merged %d duplicate(s), %d → %d characters",
            merged, len(roster.characters), len(remaining),
        )
        return CharacterRoster(characters=remaining)
    except Exception as exc:
        logger.warning("deduplicate_roster_with_llm: LLM call failed (%s) — skipping", exc)
        return roster


_EPITHET_PROMPT = """\
CHARACTER ROSTER (proper names — do not modify or invent new entries):
{roster_lines}

COMMON PHRASES found in the same book (may include epithets, titles, roles):
{phrase_lines}

TASK: For each common phrase that is clearly an epithet or alternate title
for EXACTLY ONE named character above, return a mapping.

RULES:
- Only include phrases you are highly confident refer to exactly one character.
- Skip generic roles: "the innkeeper", "the woman", "the soldier", etc.
- Skip phrases that could apply to multiple characters.
- Use the EXACT canonical name string from the roster — no variations.

Return ONLY the JSON.
"""


def resolve_epithets_with_llm(
    roster: "CharacterRoster",
    common_phrases: list[str],
    llm: "LLMClient",
) -> "CharacterRoster":
    """Add epithet aliases to *roster* characters via a single LLM call.

    Passes canonical names and high-frequency common-noun phrases extracted
    by BookNLP to the LLM, which maps phrases to characters.  Matched phrases
    are appended to the relevant ``AliasGroup.aliases``.

    Returns *roster* unchanged on any LLM error or when *common_phrases* is empty.
    """
    from .models import EpithetResolutionResult

    if not common_phrases:
        return roster

    canonical_set = {g.canonical for g in roster.characters}
    roster_lines = "\n".join(f"- {g.canonical}" for g in roster.characters)
    phrase_lines = "\n".join(f"- {p}" for p in common_phrases)

    try:
        result: EpithetResolutionResult = llm.generate(
            _EPITHET_PROMPT.format(roster_lines=roster_lines, phrase_lines=phrase_lines),
            EpithetResolutionResult,
        )

        canonical_to_group = {g.canonical: g for g in roster.characters}
        added = 0
        for mapping in result.mappings:
            epithet = mapping.epithet.strip()
            target = mapping.canonical_name.strip()
            if not epithet or target not in canonical_set:
                continue
            group = canonical_to_group[target]
            if epithet not in group.aliases:
                group.aliases.append(epithet)
                added += 1

        logger.info("resolve_epithets_with_llm: added %d epithet alias(es)", added)
        return roster
    except Exception as exc:
        logger.warning("resolve_epithets_with_llm: LLM call failed (%s) — skipping", exc)
        return roster


_NORMALIZE_PROMPT = """\
The following are character canonical names extracted from a novel.
Some may contain trailing appositive phrases or non-name suffixes that
should be stripped to leave only the character's actual name.

For each name, return the simplified form. If no change is needed,
return the name unchanged.

Names:
{name_lines}

Return ONLY the JSON.
"""


def normalize_canonical_names_with_llm(
    roster: "CharacterRoster",
    llm: "LLMClient",
) -> "CharacterRoster":
    """Strip trailing descriptors from canonical names via a single LLM call.

    e.g. "Rand al'Thor, Dragon Reborn" → "Rand al'Thor"

    Keeps the original as an alias.  Returns *roster* unchanged on any LLM error.
    """
    from .models import NameNormalizationResult

    if not roster.characters:
        return roster

    name_lines = "\n".join(f"- {g.canonical}" for g in roster.characters)
    try:
        result: NameNormalizationResult = llm.generate(
            _NORMALIZE_PROMPT.format(name_lines=name_lines),
            NameNormalizationResult,
        )

        orig_to_simplified = {e.original.strip(): e.simplified.strip() for e in result.names}
        changed = 0
        for group in roster.characters:
            simplified = orig_to_simplified.get(group.canonical, group.canonical)
            if simplified and simplified != group.canonical:
                if group.canonical not in group.aliases:
                    group.aliases.append(group.canonical)
                group.canonical = simplified
                changed += 1

        logger.info(
            "normalize_canonical_names_with_llm: simplified %d canonical name(s)", changed
        )
        return roster
    except Exception as exc:
        logger.warning(
            "normalize_canonical_names_with_llm: LLM call failed (%s) — skipping", exc
        )
        return roster


# ---------------------------------------------------------------------------
# LLM-augmented roster building
# ---------------------------------------------------------------------------

_ROSTER_PROMPT = """\
TASK: Identify every named character in the following fiction excerpt.

SEED NAMES (from NER pass — may be incomplete or fragmented):
{seed_names}

EXCERPT:
---
{sample_text}
---

INSTRUCTIONS:
1. List every human character who speaks, acts, or is directly addressed.
   Use name strings EXACTLY as they appear in the excerpt — do not invent
   variants not present in the text.
2. Choose the most complete name form as "canonical" (e.g. "Harry Potter"
   over "Harry").
3. List ALL name forms from the excerpt for this character under "aliases",
   including the canonical itself.
4. Exclude place names, organisations, and non-human entities.
5. Seed names are hints only — include additional characters you find,
   and discard seeds that are not characters.

Return ONLY the JSON — no explanation.
"""

# Number of equally-spaced buckets to sample from the book.
_SAMPLE_BUCKETS = 5


def _sample_text_for_roster(full_text: str, target_words: int = 4000) -> str:
    """Return a representative ~*target_words*-word excerpt of *full_text*.

    Splits the text into *_SAMPLE_BUCKETS* equally-spaced windows and takes
    the leading paragraphs from each window until the per-bucket word budget
    is met.  Windows are joined with ``[...]`` separators so the LLM knows
    the fragments are non-contiguous.

    If *full_text* is already at or under *target_words*, it is returned
    unchanged.
    """
    paragraphs = [p for p in full_text.split("\n\n") if p.strip()]
    if not paragraphs:
        return full_text

    total_words = sum(len(p.split()) for p in paragraphs)
    if total_words <= target_words:
        return full_text

    budget_per_bucket = max(1, target_words // _SAMPLE_BUCKETS)
    bucket_size = max(1, len(paragraphs) // _SAMPLE_BUCKETS)

    samples: list[str] = []
    for i in range(_SAMPLE_BUCKETS):
        start = i * bucket_size
        bucket_words = 0
        bucket_paras: list[str] = []
        for para in paragraphs[start:start + bucket_size]:
            bucket_paras.append(para)
            bucket_words += len(para.split())
            if bucket_words >= budget_per_bucket:
                break
        if bucket_paras:
            samples.append("\n\n".join(bucket_paras))

    return "\n\n[...]\n\n".join(samples)


def _filter_roster_hallucinations(
    roster: CharacterRoster,
    full_text: str,
) -> CharacterRoster:
    """Remove aliases from *roster* that do not appear verbatim in *full_text*.

    Per character:
    - Keeps aliases whose lowercased form is a substring of lowercased
      *full_text* and whose stripped length is >= 2.
    - If the canonical was dropped, promotes the longest surviving alias.
    - Drops the entire entry when no aliases survive.

    After individual-entry filtering, re-runs ``_cluster_by_heuristic`` on all
    surviving aliases to deduplicate any entries the LLM split incorrectly.
    """
    text_lower = full_text.lower()
    surviving_names: list[str] = []

    for group in roster.characters:
        kept = [
            a for a in group.aliases
            if len(a.strip()) >= 2 and a.lower() in text_lower
        ]
        if not kept:
            logger.debug("filter_hallucinations: dropped entire entry %r", group.canonical)
            continue

        # Ensure canonical is among kept aliases; if not, promote longest.
        if group.canonical not in kept:
            promoted = max(kept, key=len)
            logger.debug(
                "filter_hallucinations: canonical %r hallucinated; promoting %r",
                group.canonical, promoted,
            )
        surviving_names.extend(kept)

    if not surviving_names:
        return CharacterRoster(characters=[])

    groups = _cluster_by_heuristic(surviving_names)
    return CharacterRoster(characters=groups)


def build_roster_with_llm(
    text: str,
    nlp,
    llm: "LLMClient",
    sample_words: int = 4000,
) -> CharacterRoster:
    """Build a character roster using a three-tier fallback strategy.

    Tier 1 — BookNLP (best quality):
        Literary-fiction-specific NER + neural coreference resolution.
        Returns immediately if ``booknlp`` is installed.

    Tier 2 — LLM sample extraction (medium quality):
        The LLM reads a representative text sample and returns a structured
        roster. Useful when BookNLP is unavailable. Quality is model-dependent;
        small models (≤7B) may miss secondary characters.

    Tier 3 — spaCy + heuristic (always available):
        Fast, deterministic fallback. Misses name forms that spaCy's general
        NER model doesn't detect on literary prose.

    Args:
        text:         Full book text (used for sampling and hallucination guard).
        nlp:          Loaded spaCy model.
        llm:          ``LLMClient`` instance pointing at the configured model.
        sample_words: Approximate word budget for the text sample sent to the LLM.

    Returns:
        ``CharacterRoster`` with alias-grouped characters.
    """
    # ── Tier 1: BookNLP ──────────────────────────────────────────────────────
    from .booknlp_roster import build_roster_from_booknlp

    bnlp_data = build_roster_from_booknlp(text)
    if bnlp_data is not None:
        roster = bnlp_data.roster
        common_phrases = bnlp_data.common_phrases
        logger.info(
            "build_roster: BookNLP path — %d canonical characters",
            len(roster.characters),
        )
        roster = deduplicate_roster_with_llm(roster, llm)
        roster = resolve_epithets_with_llm(roster, common_phrases, llm)
        roster = normalize_canonical_names_with_llm(roster, llm)
        return roster

    logger.info("build_roster: BookNLP unavailable, trying LLM")

    # ── Tier 2: LLM ──────────────────────────────────────────────────────────
    try:
        seed_names = extract_person_names(text, nlp)
        logger.debug("build_roster: %d spaCy seed names", len(seed_names))

        sample = _sample_text_for_roster(text, sample_words)
        prompt = _ROSTER_PROMPT.format(
            seed_names=", ".join(seed_names) if seed_names else "(none)",
            sample_text=sample,
        )

        raw: CharacterRoster = llm.generate(prompt, CharacterRoster)
        logger.info("build_roster: LLM returned %d entries", len(raw.characters))

        filtered = _filter_roster_hallucinations(raw, text)
        all_names = [name for g in filtered.characters for name in g.aliases]

        if not all_names:
            logger.info("build_roster: no names survived hallucination filter; using heuristic")
            return build_roster(text, nlp)

        groups = _cluster_by_heuristic(all_names)
        logger.info(
            "build_roster: LLM path — %d canonical characters after clustering",
            len(groups),
        )
        roster_obj = CharacterRoster(characters=groups)
        roster_obj = deduplicate_roster_with_llm(roster_obj, llm)
        roster_obj = normalize_canonical_names_with_llm(roster_obj, llm)
        return roster_obj

    except Exception as exc:
        logger.warning(
            "build_roster: LLM call failed (%s); falling back to heuristic", exc,
        )

    # ── Tier 3: spaCy + heuristic ────────────────────────────────────────────
    roster = build_roster(text, nlp)
    logger.info(
        "build_roster: heuristic path — %d canonical characters",
        len(roster.characters),
    )
    roster = deduplicate_roster_with_llm(roster, llm)
    roster = normalize_canonical_names_with_llm(roster, llm)
    return roster
