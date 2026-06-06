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
import os
import re
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from ._filters import _is_proper_name
from .models import (
    CharacterRecord,
    CharacterRoster,
    CharacterRosterFullWire,
    CharacterRosterWire,
    roster_wire_to_full,
    slugify,
)

if TYPE_CHECKING:
    from kenkui.models import Chapter

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


def _escape_format_braces(text: str) -> str:
    """Escape { and } so text can safely be used in str.format() calls."""
    return text.replace("{", "{{").replace("}", "}}")


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


def _cluster_by_heuristic(names: list[str]) -> list[CharacterRecord]:
    """Group *names* into alias clusters using word-overlap heuristics.

    Returns a list of ``CharacterRecord`` objects sorted by canonical name length
    descending (most specific characters first).
    """
    # Deduplicate and sort: most words first, then longest string first.
    unique = sorted(
        {n.strip() for n in names if n.strip()},
        key=lambda n: (len(n.split()), len(n)),
        reverse=True,
    )

    groups: list[CharacterRecord] = []
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
        groups.append(CharacterRecord(slug=slugify(name), canonical_name=name, aliases=aliases))

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
        if ent.label_ == "PERSON" and ent.text not in seen and _is_proper_name(ent.text):
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
    roster: CharacterRoster,
    llm: LLMClient,
) -> CharacterRoster:
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

    name_lines = "\n".join(
        f"- {_escape_format_braces(g.canonical_name)}" for g in roster.characters
    )
    try:
        result: CanonicalMergeResult = llm.generate(
            _DEDUP_PROMPT.format(name_lines=name_lines),
            CanonicalMergeResult,
        )

        if not result.merges:
            return roster

        canonical_set = {g.canonical_name for g in roster.characters}
        absorb_into: dict[str, str] = {}
        for entry in result.merges:
            if entry.canonical not in canonical_set:
                continue
            for dup in entry.duplicates:
                if dup in canonical_set and dup != entry.canonical:
                    absorb_into[dup] = entry.canonical

        if not absorb_into:
            return roster

        group_by_canonical = {g.canonical_name: g for g in roster.characters}
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

        remaining = [g for g in roster.characters if g.canonical_name not in absorb_into]
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
    roster: CharacterRoster,
    common_phrases: list[str],
    llm: LLMClient,
) -> CharacterRoster:
    """Add epithet aliases to *roster* characters via a single LLM call.

    Passes canonical names and high-frequency common-noun phrases extracted
    by BookNLP to the LLM, which maps phrases to characters.  Matched phrases
    are appended to the relevant ``CharacterRecord.aliases``.

    Returns *roster* unchanged on any LLM error or when *common_phrases* is empty.
    """
    from .models import EpithetResolutionResult

    if not common_phrases:
        return roster

    canonical_set = {g.canonical_name for g in roster.characters}
    roster_lines = "\n".join(
        f"- {_escape_format_braces(g.canonical_name)}" for g in roster.characters
    )
    phrase_lines = "\n".join(f"- {_escape_format_braces(p)}" for p in common_phrases)

    try:
        result: EpithetResolutionResult = llm.generate(
            _EPITHET_PROMPT.format(roster_lines=roster_lines, phrase_lines=phrase_lines),
            EpithetResolutionResult,
        )

        canonical_to_group = {g.canonical_name: g for g in roster.characters}
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
    roster: CharacterRoster,
    llm: LLMClient,
) -> CharacterRoster:
    """Strip trailing descriptors from canonical names via a single LLM call.

    e.g. "Rand al'Thor, Dragon Reborn" → "Rand al'Thor"

    Keeps the original as an alias.  Returns *roster* unchanged on any LLM error.
    """
    from .models import NameNormalizationResult

    if not roster.characters:
        return roster

    name_lines = "\n".join(
        f"- {_escape_format_braces(g.canonical_name)}" for g in roster.characters
    )
    try:
        result: NameNormalizationResult = llm.generate(
            _NORMALIZE_PROMPT.format(name_lines=name_lines),
            NameNormalizationResult,
        )

        orig_to_simplified = {e.original.strip(): e.simplified.strip() for e in result.names}
        changed = 0
        for group in roster.characters:
            simplified = orig_to_simplified.get(group.canonical_name, group.canonical_name)
            if simplified and simplified != group.canonical_name:
                if group.canonical_name not in group.aliases:
                    group.aliases.append(group.canonical_name)
                group.canonical_name = simplified
                group.slug = slugify(simplified)
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
1. List EVERY named human character who appears in the excerpt in any capacity —
   speaking, acting, being addressed, or merely mentioned by name.
   Use name strings EXACTLY as they appear in the excerpt — do not invent
   variants not present in the text.
2. Choose the most complete name form as "canonical" (e.g. "Harry Potter"
   over "Harry").
3. List ALL name forms from the excerpt for this character under "aliases",
   including the canonical itself.
4. Exclude place names, organisations, and non-human entities.
5. Seed names are hints only — include additional characters you find,
   and discard seeds that are not characters.
6. Do NOT include personal pronouns (I, me, my, we, us, you, he, she, they,
   etc.) as character names or aliases, even in first-person narratives.

Return ONLY the JSON — no explanation.
"""

_ROSTER_SECTION_PROMPT_VERSION = "roster-section-v1"
_DEFAULT_ROSTER_SECTION_TARGET_TOKENS = 6000
_DEFAULT_REMOTE_CONTEXT_TOKENS = 32768
_DEFAULT_REMOTE_OUTPUT_TOKENS = 8192
_DEFAULT_MAX_SECTION_DEPTH = 12

# Number of equally-spaced buckets to sample from the book.
_SAMPLE_BUCKETS = 8


@dataclass(frozen=True)
class _LLMCallLimits:
    provider: str
    context_tokens: int
    output_tokens: int
    target_prompt_tokens: int
    max_depth: int

    @property
    def profile(self) -> str:
        return (
            f"ctx{self.context_tokens}-out{self.output_tokens}-"
            f"target{self.target_prompt_tokens}-depth{self.max_depth}"
        )


@dataclass(frozen=True)
class _RosterSection:
    chapter_index: int
    chapter_title: str
    paragraphs: tuple[str, ...]
    para_start: int
    para_end: int
    depth: int = 0
    word_start: int | None = None
    word_end: int | None = None

    @property
    def label(self) -> str:
        title = self.chapter_title or f"Chapter {self.chapter_index}"
        if self.word_start is None:
            return f"{title} paragraphs {self.para_start}:{self.para_end}"
        return (
            f"{title} paragraph {self.para_start} "
            f"words {self.word_start}:{self.word_end}"
        )

    def text(self) -> str:
        if self.word_start is not None:
            words = self.paragraphs[0].split()
            return " ".join(words[self.word_start:self.word_end])
        return "\n\n".join(self.paragraphs)

    def cache_bounds(self) -> dict[str, int | None]:
        return {
            "chapter_index": self.chapter_index,
            "para_start": self.para_start,
            "para_end": self.para_end,
            "word_start": self.word_start,
            "word_end": self.word_end,
        }


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        value = int(raw)
    except ValueError:
        logger.warning("Ignoring invalid %s=%r; expected integer", name, raw)
        return default
    return value if value > 0 else default


def _limits_for_provider(provider: str) -> _LLMCallLimits:
    provider_norm = (provider or "ollama").lower()
    if provider_norm == "ollama":
        context_tokens = _env_int("KENKUI_NLP_OLLAMA_NUM_CTX", 65536)
        output_tokens = _env_int("KENKUI_NLP_OLLAMA_NUM_PREDICT", 32768)
    else:
        context_tokens = _env_int("KENKUI_NLP_REMOTE_CONTEXT_TOKENS", _DEFAULT_REMOTE_CONTEXT_TOKENS)
        output_tokens = _env_int("KENKUI_NLP_REMOTE_OUTPUT_TOKENS", _DEFAULT_REMOTE_OUTPUT_TOKENS)

    available_prompt = max(1024, context_tokens - output_tokens)
    configured_target = _env_int(
        "KENKUI_NLP_ROSTER_SECTION_TARGET_TOKENS",
        _DEFAULT_ROSTER_SECTION_TARGET_TOKENS,
    )
    target_prompt_tokens = min(configured_target, max(1024, int(available_prompt * 0.75)))
    max_depth = _env_int("KENKUI_NLP_ROSTER_SECTION_MAX_DEPTH", _DEFAULT_MAX_SECTION_DEPTH)
    return _LLMCallLimits(
        provider=provider_norm,
        context_tokens=context_tokens,
        output_tokens=output_tokens,
        target_prompt_tokens=target_prompt_tokens,
        max_depth=max_depth,
    )


def _estimate_tokens(text: str) -> int:
    return max(1, len(text) // 4)


def _build_roster_prompt(section_text: str, seed_names: list[str]) -> str:
    return _ROSTER_PROMPT.format(
        seed_names=", ".join(_escape_format_braces(s) for s in seed_names)
        if seed_names else "(none)",
        sample_text=section_text,
    )


def _split_section(section: _RosterSection) -> list[_RosterSection]:
    if len(section.paragraphs) > 1:
        midpoint = len(section.paragraphs) // 2
        left_paras = section.paragraphs[:midpoint]
        right_paras = section.paragraphs[midpoint:]
        split_at = section.para_start + midpoint
        return [
            _RosterSection(
                chapter_index=section.chapter_index,
                chapter_title=section.chapter_title,
                paragraphs=left_paras,
                para_start=section.para_start,
                para_end=split_at,
                depth=section.depth + 1,
            ),
            _RosterSection(
                chapter_index=section.chapter_index,
                chapter_title=section.chapter_title,
                paragraphs=right_paras,
                para_start=split_at,
                para_end=section.para_end,
                depth=section.depth + 1,
            ),
        ]

    words = section.text().split()
    if len(words) <= 1:
        return []

    base_start = section.word_start or 0
    midpoint = len(words) // 2
    word_mid = base_start + midpoint
    word_end = section.word_end if section.word_end is not None else base_start + len(words)
    para_index = section.para_start
    return [
        _RosterSection(
            chapter_index=section.chapter_index,
            chapter_title=section.chapter_title,
            paragraphs=section.paragraphs,
            para_start=para_index,
            para_end=para_index + 1,
            depth=section.depth + 1,
            word_start=base_start,
            word_end=word_mid,
        ),
        _RosterSection(
            chapter_index=section.chapter_index,
            chapter_title=section.chapter_title,
            paragraphs=section.paragraphs,
            para_start=para_index,
            para_end=para_index + 1,
            depth=section.depth + 1,
            word_start=word_mid,
            word_end=word_end,
        ),
    ]


def _heuristic_section_roster(section_text: str, nlp) -> CharacterRoster:
    if nlp is None:
        return CharacterRoster(characters=[])
    return _build_roster_spacy_chunked(section_text, nlp)


def _load_cached_section(
    section: _RosterSection,
    *,
    book_path: Path | None,
    provider: str,
    model: str,
    method: str,
    limits: _LLMCallLimits,
) -> CharacterRoster | None:
    if book_path is None:
        return None
    from kenkui.nlp import get_cached_roster_section

    return get_cached_roster_section(
        book_path,
        provider=provider,
        model=model,
        method=method,
        prompt_version=_ROSTER_SECTION_PROMPT_VERSION,
        limit_profile=limits.profile,
        **section.cache_bounds(),
    )


def _cache_section(
    roster: CharacterRoster,
    section: _RosterSection,
    *,
    book_path: Path | None,
    provider: str,
    model: str,
    method: str,
    limits: _LLMCallLimits,
) -> None:
    if book_path is None:
        return
    from kenkui.nlp import cache_roster_section

    try:
        cache_roster_section(
            roster,
            book_path,
            provider=provider,
            model=model,
            method=method,
            prompt_version=_ROSTER_SECTION_PROMPT_VERSION,
            limit_profile=limits.profile,
            **section.cache_bounds(),
        )
    except Exception as exc:
        logger.warning("build_roster: failed to write roster section cache (%s)", exc)


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

    # When the entire text is one unbroken block (no \n\n separators — common
    # when full_text was built by joining paragraph strings with spaces), the
    # paragraph-level loop would append the whole block before the budget check
    # fires, returning the entire book.  Fall back to word-level slicing instead.
    if len(paragraphs) == 1:
        words = paragraphs[0].split()
        word_chunk = max(1, len(words) // _SAMPLE_BUCKETS)
        samples: list[str] = []
        for i in range(_SAMPLE_BUCKETS):
            start = i * word_chunk
            chunk = " ".join(words[start:start + budget_per_bucket])
            if chunk:
                samples.append(chunk)
        return "\n\n[...]\n\n".join(samples)

    bucket_size = max(1, len(paragraphs) // _SAMPLE_BUCKETS)

    samples = []
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


def _chunk_text_for_roster(full_text: str, target_words: int = 8000) -> list[str]:
    """Split *full_text* into paragraph-preserving word-budget chunks."""
    paragraphs = [p for p in full_text.split("\n\n") if p.strip()]
    if not paragraphs:
        return [full_text] if full_text.strip() else []

    chunks: list[str] = []
    current: list[str] = []
    current_words = 0
    budget = max(1, target_words)

    for para in paragraphs:
        words = para.split()
        if not words:
            continue
        if len(words) > budget:
            if current:
                chunks.append("\n\n".join(current))
                current = []
                current_words = 0
            for start in range(0, len(words), budget):
                chunks.append(" ".join(words[start:start + budget]))
            continue
        if current and current_words + len(words) > budget:
            chunks.append("\n\n".join(current))
            current = []
            current_words = 0
        current.append(para)
        current_words += len(words)

    if current:
        chunks.append("\n\n".join(current))

    return chunks


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
    kept_count = 0
    dropped_count = 0
    dropped_entries = 0

    for group in roster.characters:
        kept = [
            a for a in group.aliases
            if len(a.strip()) >= 2
            and bool(re.search(r'(?<!\w)' + re.escape(a.lower()) + r'(?!\w)', text_lower))
        ]
        dropped_aliases = max(0, len(group.aliases) - len(kept))
        dropped_count += dropped_aliases
        kept_count += len(kept)
        if not kept:
            dropped_entries += 1
            logger.info(
                "filter_hallucinations: dropped hallucinated entry %r aliases=%d",
                group.canonical_name,
                len(group.aliases),
            )
            continue

        # Ensure canonical is among kept aliases; if not, promote longest.
        if group.canonical_name not in kept:
            promoted = max(kept, key=len)
            logger.info(
                "filter_hallucinations: canonical %r hallucinated; promoting %r",
                group.canonical_name, promoted,
            )
        surviving_names.extend(kept)

    if not surviving_names:
        logger.warning(
            "filter_hallucinations: no aliases survived (kept=%d dropped=%d entries_dropped=%d)",
            kept_count,
            dropped_count,
            dropped_entries,
        )
        return CharacterRoster(characters=[])

    groups = _cluster_by_heuristic(surviving_names)
    logger.info(
        "filter_hallucinations: kept %d alias(es), dropped %d alias(es), dropped %d entry(s), clustered to %d character(s)",
        kept_count,
        dropped_count,
        dropped_entries,
        len(groups),
    )
    return CharacterRoster(characters=groups)


def _build_roster_spacy_chunked(
    text: str,
    nlp,
    step_callback: Callable[[str], None] | None = None,
) -> CharacterRoster:
    """Run spaCy NER on *text* in max_length-safe chunks and merge results.

    Used when the full text exceeds ``nlp.max_length``.  Each chunk is
    processed independently; PERSON entities are deduplicated then clustered
    together via the word-overlap heuristic.
    """
    try:
        max_len = int(nlp.max_length)
    except (TypeError, ValueError, AttributeError):
        return build_roster(text, nlp)
    if len(text) <= max_len:
        return build_roster(text, nlp)

    # Pre-count chunks so we can report accurate block progress.
    total_chunks = 0
    _pos = 0
    while _pos < len(text):
        _end = min(_pos + max_len, len(text))
        if _end < len(text):
            _b = text.rfind(" ", _pos, _end)
            if _b > _pos:
                _end = _b
        total_chunks += 1
        _pos = _end

    all_names: list[str] = []
    seen: set[str] = set()
    start = 0
    chunk_idx = 0
    t0 = time.monotonic()
    while start < len(text):
        end = min(start + max_len, len(text))
        if end < len(text):
            boundary = text.rfind(" ", start, end)
            if boundary > start:
                end = boundary
        for name in extract_person_names(text[start:end], nlp):
            if name not in seen:
                seen.add(name)
                all_names.append(name)
        chunk_idx += 1
        if step_callback:
            elapsed = time.monotonic() - t0
            step_callback(f"Block {chunk_idx}/{total_chunks} — {elapsed:.0f}s")
        start = end

    if not all_names:
        return CharacterRoster(characters=[])

    return CharacterRoster(characters=_cluster_by_heuristic(all_names))


def _coerce_llm_roster_response(response: object) -> CharacterRoster:
    """Accept real wire responses and older test doubles returning full rosters."""
    if isinstance(response, CharacterRoster):
        return response
    if isinstance(response, CharacterRosterWire | CharacterRosterFullWire):
        return roster_wire_to_full(response)
    raise TypeError(f"unexpected roster response type: {type(response).__name__}")


def _extract_roster_section(
    section: _RosterSection,
    *,
    nlp,
    llm: LLMClient,
    explicit_llm: bool,
    book_path: Path | None,
    provider: str,
    model: str,
    method: str,
    limits: _LLMCallLimits,
) -> tuple[CharacterRoster, int]:
    """Extract one section, recursively splitting when the call is too large or fails."""
    section_text = section.text()
    if not section_text.strip():
        return CharacterRoster(characters=[]), 0

    if section.depth > limits.max_depth:
        logger.warning(
            "build_roster: section %s exceeded max split depth=%d; using heuristic fallback",
            section.label,
            limits.max_depth,
        )
        return _heuristic_section_roster(section_text, nlp), 0

    seed_names = [] if explicit_llm or nlp is None else extract_person_names(section_text, nlp)
    prompt = _build_roster_prompt(section_text, seed_names)
    est_prompt_tokens = _estimate_tokens(prompt)

    if est_prompt_tokens > limits.target_prompt_tokens:
        children = _split_section(section)
        if children:
            logger.info(
                "build_roster: section %s estimated %d prompt tokens above target %d; "
                "bisecting into %d section(s)",
                section.label,
                est_prompt_tokens,
                limits.target_prompt_tokens,
                len(children),
            )
            characters: list[CharacterRecord] = []
            failures = 0
            for child in children:
                child_roster, child_failures = _extract_roster_section(
                    child,
                    nlp=nlp,
                    llm=llm,
                    explicit_llm=explicit_llm,
                    book_path=book_path,
                    provider=provider,
                    model=model,
                    method=method,
                    limits=limits,
                )
                characters.extend(child_roster.characters)
                failures += child_failures
            return CharacterRoster(characters=characters), failures

    cached = _load_cached_section(
        section,
        book_path=book_path,
        provider=provider,
        model=model,
        method=method,
        limits=limits,
    )
    if cached is not None:
        logger.debug(
            "build_roster: roster section cache hit for %s (%d character(s))",
            section.label,
            len(cached.characters),
        )
        return cached, 0

    try:
        raw_response = llm.generate(prompt, CharacterRosterWire)
        roster = _coerce_llm_roster_response(raw_response)
        logger.info(
            "build_roster: LLM roster section %s returned %d character(s) "
            "(prompt~%d tokens)",
            section.label,
            len(roster.characters),
            est_prompt_tokens,
        )
        _cache_section(
            roster,
            section,
            book_path=book_path,
            provider=provider,
            model=model,
            method=method,
            limits=limits,
        )
        return roster, 0
    except Exception as exc:
        children = _split_section(section)
        if children:
            logger.warning(
                "build_roster: LLM roster section %s failed (%s); bisecting into %d section(s)",
                section.label,
                exc,
                len(children),
            )
            characters = []
            failures = 0
            for child in children:
                child_roster, child_failures = _extract_roster_section(
                    child,
                    nlp=nlp,
                    llm=llm,
                    explicit_llm=explicit_llm,
                    book_path=book_path,
                    provider=provider,
                    model=model,
                    method=method,
                    limits=limits,
                )
                characters.extend(child_roster.characters)
                failures += child_failures
            return CharacterRoster(characters=characters), failures

        logger.warning(
            "build_roster: LLM roster leaf section %s failed (%s); using heuristic fallback",
            section.label,
            exc,
        )
        fallback = _heuristic_section_roster(section_text, nlp)
        if fallback.characters:
            return fallback, 1
        return CharacterRoster(characters=[]), 1


def _finalize_llm_roster(
    raw: CharacterRoster,
    *,
    full_text: str,
    nlp,
    llm: LLMClient,
    explicit_llm: bool,
    step_callback: Callable[[str], None] | None,
) -> CharacterRoster:
    filtered = _filter_roster_hallucinations(raw, full_text)
    all_names = [name for group in filtered.characters for name in group.aliases]

    if not all_names:
        logger.info("build_roster: no names survived hallucination filter; using heuristic")
        if explicit_llm or nlp is None:
            return CharacterRoster(characters=[])
        return _build_roster_spacy_chunked(full_text, nlp, step_callback=step_callback)

    groups = _cluster_by_heuristic(all_names)
    logger.info(
        "build_roster: LLM path — %d canonical characters after clustering",
        len(groups),
    )
    roster_obj = CharacterRoster(characters=groups)
    roster_obj = deduplicate_roster_with_llm(roster_obj, llm)
    if step_callback:
        step_callback("Deduplicated roster")
    roster_obj = normalize_canonical_names_with_llm(roster_obj, llm)
    if step_callback:
        step_callback("Normalized names")
    return roster_obj


def build_roster_from_chapters_with_llm(
    chapters: list[Chapter],
    nlp=None,
    llm: LLMClient | None = None,
    method: str = "auto",
    step_callback: Callable[[str], None] | None = None,
    book_path: Path | None = None,
    provider: str = "ollama",
    model: str = "",
) -> CharacterRoster:
    """Build a roster from chapter-bounded LLM calls with recursive bisection."""
    from .booknlp_roster import build_roster_from_booknlp

    full_text = "\n\n".join("\n\n".join(ch.paragraphs) for ch in chapters)

    if method == "spacy":
        if nlp is None:
            raise ValueError("nlp model is required for method='spacy'")
        logger.info("build_roster: spaCy-only path")
        roster = _build_roster_spacy_chunked(full_text, nlp, step_callback=step_callback)
        if step_callback:
            step_callback("Extracted characters")
        return roster

    if method == "booknlp":
        bnlp_data = build_roster_from_booknlp(full_text)
        if bnlp_data is None:
            raise RuntimeError(
                "BookNLP is not installed or failed to process the text. "
                "Install it with: pip install booknlp"
            )
        roster = bnlp_data.roster
        if step_callback:
            step_callback("Extracted characters")
        roster = deduplicate_roster_with_llm(roster, llm)
        if step_callback:
            step_callback("Deduplicated roster")
        roster = resolve_epithets_with_llm(roster, bnlp_data.common_phrases, llm)
        if step_callback:
            step_callback("Resolved epithets")
        roster = normalize_canonical_names_with_llm(roster, llm)
        if step_callback:
            step_callback("Normalized names")
        return roster

    if llm is None:
        raise ValueError("llm client is required for LLM roster extraction")

    if method == "auto":
        bnlp_data = build_roster_from_booknlp(full_text)
        if bnlp_data is not None:
            roster = bnlp_data.roster
            logger.info(
                "build_roster: BookNLP path — %d canonical characters",
                len(roster.characters),
            )
            if step_callback:
                step_callback("Extracted characters")
            roster = deduplicate_roster_with_llm(roster, llm)
            if step_callback:
                step_callback("Deduplicated roster")
            roster = resolve_epithets_with_llm(roster, bnlp_data.common_phrases, llm)
            if step_callback:
                step_callback("Resolved epithets")
            roster = normalize_canonical_names_with_llm(roster, llm)
            if step_callback:
                step_callback("Normalized names")
            return roster
        logger.info("build_roster: BookNLP unavailable, trying chapter-bounded LLM")

    explicit_llm = method not in ("auto", "booknlp", "spacy")
    limits = _limits_for_provider(provider)
    sections = [
        _RosterSection(
            chapter_index=getattr(chapter, "index", idx),
            chapter_title=getattr(chapter, "title", "") or f"Chapter {getattr(chapter, 'index', idx)}",
            paragraphs=tuple(p for p in chapter.paragraphs if p.strip()),
            para_start=0,
            para_end=len([p for p in chapter.paragraphs if p.strip()]),
        )
        for idx, chapter in enumerate(chapters)
        if any(p.strip() for p in chapter.paragraphs)
    ]
    if not sections:
        return CharacterRoster(characters=[])

    logger.info(
        "build_roster: chapter-bounded LLM extraction starting sections=%d provider=%s "
        "model=%s target_prompt_tokens=%d",
        len(sections),
        provider,
        model,
        limits.target_prompt_tokens,
    )

    raw_characters: list[CharacterRecord] = []
    failed_sections = 0
    for section_idx, section in enumerate(sections, start=1):
        roster, failures = _extract_roster_section(
            section,
            nlp=nlp,
            llm=llm,
            explicit_llm=explicit_llm,
            book_path=book_path,
            provider=provider,
            model=model,
            method=method,
            limits=limits,
        )
        raw_characters.extend(roster.characters)
        failed_sections += failures
        if step_callback:
            step_callback(f"Block {section_idx}/{len(sections)}")

    if not raw_characters and failed_sections:
        if explicit_llm:
            logger.warning(
                "build_roster: all chapter-bounded LLM sections failed with no heuristic results"
            )
            return CharacterRoster(characters=[])
        logger.warning(
            "build_roster: no LLM section results; falling back to full heuristic extraction"
        )
        return _build_roster_spacy_chunked(full_text, nlp, step_callback=step_callback)

    raw = CharacterRoster(characters=raw_characters)
    logger.info(
        "build_roster: LLM returned %d total entries across %d chapter section(s), "
        "leaf_failures=%d",
        len(raw.characters),
        len(sections),
        failed_sections,
    )
    if step_callback:
        step_callback("Extracted characters")

    return _finalize_llm_roster(
        raw,
        full_text=full_text,
        nlp=nlp,
        llm=llm,
        explicit_llm=explicit_llm,
        step_callback=step_callback,
    )


def build_roster_with_llm(
    text: str,
    nlp=None,
    llm: LLMClient | None = None,
    sample_words: int = 8000,
    method: str = "auto",
    step_callback: Callable[[str], None] | None = None,
) -> CharacterRoster:
    """Build a character roster using a configurable strategy.

    ``method`` controls which discovery path is taken:

    ``"auto"`` (default) — Three-tier fallback:
        Tier 1 — BookNLP (best quality, if installed)
        Tier 2 — LLM sample extraction with optional spaCy seeds
        Tier 3 — spaCy + heuristic (chunked for large texts)

    ``"booknlp"`` — Force BookNLP.  Raises ``RuntimeError`` if not installed.

    ``"spacy"`` — spaCy NER + heuristic only, no LLM calls.

    Any other value (``"llm"``, ``"ollama"``, ``"litellm"``, …) — Skip
    BookNLP, use LLM directly with no spaCy seed pass.  Raises on LLM
    failure instead of silently degrading.

    Args:
        text:         Full book text (used for sampling and hallucination guard).
        nlp:          Loaded spaCy model.  Required for ``"auto"`` and ``"spacy"``
                      methods; unused (and may be ``None``) for explicit LLM methods.
        llm:          ``LLMClient`` instance.  Required for all non-spaCy paths.
        sample_words: Approximate word budget for the text sample sent to the LLM.
        method:       Discovery strategy (see above).

    Returns:
        ``CharacterRoster`` with alias-grouped characters.
    """
    from .booknlp_roster import build_roster_from_booknlp

    # ── spaCy-only mode (no LLM) ──────────────────────────────────────────────
    if method == "spacy":
        if nlp is None:
            raise ValueError("nlp model is required for method='spacy'")
        logger.info("build_roster: spaCy-only path")
        roster = _build_roster_spacy_chunked(text, nlp, step_callback=step_callback)
        if step_callback:
            step_callback("Extracted characters")
        return roster

    # ── Explicit BookNLP mode ─────────────────────────────────────────────────
    if method == "booknlp":
        bnlp_data = build_roster_from_booknlp(text)
        if bnlp_data is None:
            raise RuntimeError(
                "BookNLP is not installed or failed to process the text. "
                "Install it with: pip install booknlp"
            )
        roster = bnlp_data.roster
        common_phrases = bnlp_data.common_phrases
        logger.info(
            "build_roster: BookNLP (explicit) — %d canonical characters",
            len(roster.characters),
        )
        if step_callback:
            step_callback("Extracted characters")
        roster = deduplicate_roster_with_llm(roster, llm)
        if step_callback:
            step_callback("Deduplicated roster")
        roster = resolve_epithets_with_llm(roster, common_phrases, llm)
        if step_callback:
            step_callback("Resolved epithets")
        roster = normalize_canonical_names_with_llm(roster, llm)
        if step_callback:
            step_callback("Normalized names")
        return roster

    # ── Auto mode: try BookNLP first ─────────────────────────────────────────
    if method == "auto":
        bnlp_data = build_roster_from_booknlp(text)
        if bnlp_data is not None:
            roster = bnlp_data.roster
            common_phrases = bnlp_data.common_phrases
            logger.info(
                "build_roster: BookNLP path — %d canonical characters",
                len(roster.characters),
            )
            if step_callback:
                step_callback("Extracted characters")
            roster = deduplicate_roster_with_llm(roster, llm)
            if step_callback:
                step_callback("Deduplicated roster")
            roster = resolve_epithets_with_llm(roster, common_phrases, llm)
            if step_callback:
                step_callback("Resolved epithets")
            roster = normalize_canonical_names_with_llm(roster, llm)
            if step_callback:
                step_callback("Normalized names")
            return roster
        logger.info("build_roster: BookNLP unavailable, trying LLM")

    # ── LLM path ──────────────────────────────────────────────────────────────
    # Explicit LLM methods (anything other than "auto"/"booknlp"/"spacy"):
    #   - no spaCy seed pass (nlp may be None; avoids max_length failures)
    #   - raises on failure instead of silently degrading to spaCy
    # Auto fallback path:
    #   - seeds from the same sample sent to the LLM (well under max_length)
    #   - falls through to Tier 3 on failure
    _explicit_llm = method not in ("auto", "booknlp", "spacy")

    try:
        chunks = _chunk_text_for_roster(text, sample_words)
        if not chunks:
            return CharacterRoster(characters=[])
        logger.info(
            "build_roster: LLM chunk extraction starting chunks=%d target_words=%d explicit=%s",
            len(chunks),
            sample_words,
            _explicit_llm,
        )

        raw_characters: list[CharacterRecord] = []
        failed_chunks = 0
        for chunk_idx, chunk_text in enumerate(chunks, start=1):
            if not _explicit_llm and nlp is not None:
                seed_names = extract_person_names(chunk_text, nlp)
                logger.debug(
                    "build_roster: chunk %d/%d spaCy seed names=%d words=%d chars=%d",
                    chunk_idx,
                    len(chunks),
                    len(seed_names),
                    len(chunk_text.split()),
                    len(chunk_text),
                )
            else:
                seed_names = []

            prompt = _ROSTER_PROMPT.format(
                seed_names=", ".join(_escape_format_braces(s) for s in seed_names) if seed_names else "(none)",
                sample_text=chunk_text,
            )

            try:
                raw_response = llm.generate(prompt, CharacterRosterWire)
                chunk_roster = _coerce_llm_roster_response(raw_response)
            except Exception as chunk_exc:
                failed_chunks += 1
                logger.warning(
                    "build_roster: LLM roster chunk %d/%d failed (%s)",
                    chunk_idx,
                    len(chunks),
                    chunk_exc,
                )
                continue

            logger.info(
                "build_roster: LLM roster chunk %d/%d returned %d character(s)",
                chunk_idx,
                len(chunks),
                len(chunk_roster.characters),
            )
            raw_characters.extend(chunk_roster.characters)

        if failed_chunks == len(chunks):
            raise RuntimeError(f"all {len(chunks)} roster extraction chunk(s) failed")

        raw = CharacterRoster(characters=raw_characters)
        logger.info(
            "build_roster: LLM returned %d total entries across %d chunk(s), failures=%d",
            len(raw.characters),
            len(chunks),
            failed_chunks,
        )
        if step_callback:
            step_callback("Extracted characters")

        filtered = _filter_roster_hallucinations(raw, text)
        all_names = [name for g in filtered.characters for name in g.aliases]

        if not all_names:
            logger.info("build_roster: no names survived hallucination filter; using heuristic")
            if _explicit_llm:
                return CharacterRoster(characters=[])
            return _build_roster_spacy_chunked(text, nlp, step_callback=step_callback)

        groups = _cluster_by_heuristic(all_names)
        logger.info(
            "build_roster: LLM path — %d canonical characters after clustering",
            len(groups),
        )
        roster_obj = CharacterRoster(characters=groups)
        roster_obj = deduplicate_roster_with_llm(roster_obj, llm)
        if step_callback:
            step_callback("Deduplicated roster")
        roster_obj = normalize_canonical_names_with_llm(roster_obj, llm)
        if step_callback:
            step_callback("Normalized names")
        return roster_obj

    except Exception as exc:
        if _explicit_llm:
            raise
        logger.warning(
            "build_roster: roster extraction failed (%s); falling back to heuristic", exc,
        )

    # ── Tier 3: spaCy + heuristic (auto mode only) ───────────────────────────
    roster = _build_roster_spacy_chunked(text, nlp, step_callback=step_callback)
    logger.info(
        "build_roster: heuristic path — %d canonical characters",
        len(roster.characters),
    )
    if step_callback:
        step_callback("Extracted characters")
    roster = deduplicate_roster_with_llm(roster, llm)
    if step_callback:
        step_callback("Deduplicated roster")
    roster = normalize_canonical_names_with_llm(roster, llm)
    if step_callback:
        step_callback("Normalized names")
    return roster
