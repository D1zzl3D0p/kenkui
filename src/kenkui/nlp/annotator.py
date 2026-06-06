"""Stage 1.5: Annotated-chapter builder.

Produces a pre-labelled version of the chapter text where every extracted
quote is replaced with a ``[QUOTE:N …]`` tag and surrounding narration is
wrapped in ``[NARRATOR]`` tags.  The result is fed to the LLM attribution
step as a fill-in-the-blanks task rather than free-form analysis.

Public API
----------
annotate_chapter(paragraphs, quotes, alias_to_slug, slug_to_pronoun) → str
"""

from __future__ import annotations

import logging
import re
from collections import defaultdict

from .models import CharacterRoster, Quote

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Attribution-verb patterns
# ---------------------------------------------------------------------------

_ATTR_VERBS = (
    "said", "asked", "replied", "whispered", "shouted", "muttered",
    "cried", "answered", "added", "snapped", "growled", "murmured",
    "exclaimed",
)
_VERB_GROUP = "(?:" + "|".join(_ATTR_VERBS) + ")"

# Pronoun cluster patterns
_PRONOUN_HE = re.compile(r"\b(he|him|his)\b", re.IGNORECASE)
_PRONOUN_SHE = re.compile(r"\b(she|her|hers)\b", re.IGNORECASE)
_PRONOUN_THEY = re.compile(r"\b(they|them|their)\b", re.IGNORECASE)

# Closing-quote characters (straight and curly)
_CLOSE_QUOTE_RE = re.compile(r'["\u201d]')


def _build_alias_pattern(
    alias_to_slug: dict[str, str],
) -> tuple[re.Pattern, str] | tuple[None, None]:
    """Compile a single OR-pattern from all known aliases (longest first).

    Returns:
        A tuple ``(compiled_pattern, inner_group_str)`` where *inner_group_str*
        is the raw ``"alias1|alias2|..."`` alternation string (before wrapping
        in ``\\b(...)\\b``), or ``(None, None)`` when *alias_to_slug* is empty.
    """
    if not alias_to_slug:
        return None, None
    # Sort by length descending so longer aliases match before substrings.
    aliases = sorted(alias_to_slug.keys(), key=len, reverse=True)
    escaped = [re.escape(a) for a in aliases]
    inner_group_str = "|".join(escaped)
    return re.compile(r"\b(" + inner_group_str + r")\b", re.IGNORECASE), inner_group_str


def _extract_hints(
    para: str,
    quote: Quote,
    alias_to_slug: dict[str, str],
    alias_pat: re.Pattern | None,
    alias_inner: str | None,
    para_start_offset: int,
) -> str:
    """Return the hint/pronoun suffix for a QUOTE tag.

    Three independent tiers:

    1. Strong hint (``hint=``): attribution verb + alias within ±120 chars.
    2. Weak guess (``guess=``): alias immediately after the closing quote mark
       (dialogue only — italic spans have no quote-mark boundary).
    3. Pronoun (``pronoun=``): gendered pronoun within ±60 chars.

    All three can coexist and are concatenated in hint → guess → pronoun order.

    Args:
        para:              Full paragraph text.
        quote:             The quote being annotated.
        alias_to_slug:     Mapping of ``alias.lower() → character_slug``.
        alias_pat:         Pre-compiled alias regex (or ``None`` if no aliases).
        alias_inner:       Raw ``"alias1|alias2|..."`` alternation string
                           matching ``alias_pat``'s inner group (or ``None``).
        para_start_offset: Character offset of *para* in the joined chapter.
    """
    suffix = ""

    # Local position of the quote within *para*.
    local_start = quote.char_offset - para_start_offset
    # Italic quote.text is plain content (markers stripped); raw span is +2 bytes.
    local_end = local_start + len(quote.text) + (2 if quote.kind == "italic" else 0)

    # Guard against bad offsets (e.g. italic spans with stripped markers).
    local_start = max(0, local_start)
    local_end = min(len(para), local_end)

    # -----------------------------------------------------------------------
    # Tier 1 — Strong hint: attribution verb near alias
    # -----------------------------------------------------------------------
    if alias_pat is not None:
        window_start = max(0, local_start - 120)
        window_end = min(len(para), local_end + 120)
        window = para[window_start:window_end]

        # Pattern: verb said alias  OR  alias said verb
        strong_pat = re.compile(
            r"\b" + _VERB_GROUP + r"\s+(" + alias_inner + r")\b"
            r"|"
            r"\b(" + alias_inner + r")\s+" + _VERB_GROUP + r"\b",
            re.IGNORECASE,
        )
        m = strong_pat.search(window)
        if m:
            # Group 1: verb + alias form; group 2: alias + verb form
            matched_alias = (m.group(1) or m.group(2) or "").lower()
            slug = alias_to_slug.get(matched_alias)
            if slug:
                suffix += f' hint="{slug}"'

    # -----------------------------------------------------------------------
    # Tier 2 — Weak guess: alias immediately after closing quote
    #
    # The closing-quote character is the *last* character of quote.text, so
    # we start the scan one position back (local_end - 1) to include it.
    # Only meaningful for dialogue — italic spans have no enclosing quote marks.
    # -----------------------------------------------------------------------
    if alias_pat is not None and quote.kind == "dialogue":
        after_start = max(0, local_end - 1)
        after_end = min(len(para), after_start + 80)
        after_text = para[after_start:after_end]

        guess_pat = re.compile(
            r'^["\u201d\u2019][,.]?\s+(' + alias_inner + r")\b",
            re.IGNORECASE,
        )
        gm = guess_pat.search(after_text)
        if gm:
            matched_alias = gm.group(1).lower()
            slug = alias_to_slug.get(matched_alias)
            if slug:
                suffix += f' guess="{slug}"'

    # -----------------------------------------------------------------------
    # Tier 3 — Pronoun: gendered pronoun near attribution verb
    # -----------------------------------------------------------------------
    prn_start = max(0, local_start - 60)
    prn_end = min(len(para), local_end + 60)
    prn_window = para[prn_start:prn_end]

    # Only count pronouns that appear near an attribution verb.
    has_verb_nearby = bool(re.search(_VERB_GROUP, prn_window, re.IGNORECASE))
    if has_verb_nearby:
        he_count = len(_PRONOUN_HE.findall(prn_window))
        she_count = len(_PRONOUN_SHE.findall(prn_window))
        they_count = len(_PRONOUN_THEY.findall(prn_window))
        counts = [("he/him", he_count), ("she/her", she_count), ("they/them", they_count)]
        best_label, best_count = max(counts, key=lambda t: t[1])
        if best_count > 0:
            suffix += f' pronoun="{best_label}"'

    return suffix


def annotate_chapter(
    paragraphs: list[str],
    quotes: list[Quote],
    alias_to_slug: dict[str, str],
    slug_to_pronoun: dict[str, str],
) -> str:
    """Build an annotated chapter string from *paragraphs* and pre-extracted *quotes*.

    Each paragraph is split at quote boundaries.  Surrounding narration gets a
    ``[NARRATOR]`` prefix; each quote gets a ``[QUOTE:N …]`` tag.

    Args:
        paragraphs:      Raw paragraph strings (chapter text split by blank lines).
        quotes:          Quotes extracted by ``extract_quotes()``.
        alias_to_slug:   Mapping of ``alias.lower() → character_slug``.
        slug_to_pronoun: Mapping of ``slug → "he/him" | "she/her" | "they/them" | ""``.
                         Currently unused — pronoun hints are detected from raw text
                         rather than roster lookup.

    Returns:
        A single string with ``\\n\\n``-joined annotated segments.
    """
    if not paragraphs:
        return ""

    # Compile alias pattern once for the whole chapter (not per-quote).
    alias_pat, alias_inner = _build_alias_pattern(alias_to_slug)

    # Build para_index → quotes map.
    para_quotes: dict[int, list[Quote]] = defaultdict(list)
    for q in quotes:
        para_quotes[q.para_index].append(q)

    # Compute per-paragraph start offsets in the joined chapter text
    # (paragraphs joined by "\n\n", so each separator is 2 chars).
    para_start_offsets: list[int] = []
    offset = 0
    for para in paragraphs:
        para_start_offsets.append(offset)
        offset += len(para) + 2  # +2 for "\n\n"

    segments: list[str] = []

    for para_idx, para in enumerate(paragraphs):
        pq = sorted(para_quotes.get(para_idx, []), key=lambda q: q.char_offset)
        para_start = para_start_offsets[para_idx]

        if not pq:
            # No quotes — entire paragraph is narration.
            segments.append(f"[NARRATOR] {para}")
            continue

        # Split the paragraph at quote boundaries.
        cursor = 0
        for quote in pq:
            local_start = quote.char_offset - para_start
            # Italic quote.text is plain content (markers stripped), but the span
            # in the raw paragraph includes \x02 + content + \x03 (+2 bytes).
            local_end = local_start + len(quote.text) + (2 if quote.kind == "italic" else 0)

            # Guard against bad offsets.
            local_start = max(cursor, min(local_start, len(para)))
            local_end = max(local_start, min(local_end, len(para)))

            # Leading / between-quote narration.
            before = para[cursor:local_start]
            if before.strip():
                segments.append(f"[NARRATOR] {before}")

            # Build the QUOTE tag.
            hint_suffix = _extract_hints(
                para, quote, alias_to_slug, alias_pat, alias_inner, para_start
            )
            segments.append(f"[QUOTE:{quote.id}{hint_suffix}] {quote.text}")

            cursor = local_end

        # Trailing narration after the last quote.
        trailing = para[cursor:]
        if trailing.strip():
            segments.append(f"[NARRATOR] {trailing}")

    return "\n\n".join(segments)


# ---------------------------------------------------------------------------
# Shared attribution prompt builders (used by both OllamaProvider and CloudProvider)
# ---------------------------------------------------------------------------


def _build_alias_to_slug(roster: CharacterRoster) -> dict[str, str]:
    """Build a case-folded alias → slug mapping from *roster*.

    First-writer-wins on collision: if two characters share an alias key, the
    first character in the roster claims it and subsequent duplicates are ignored.
    """
    alias_to_slug: dict[str, str] = {}
    for c in roster.characters:
        for alias in [c.canonical_name] + c.aliases:
            key = alias.lower()
            if key not in alias_to_slug:
                alias_to_slug[key] = c.slug
            elif alias_to_slug[key] != c.slug:
                logger.debug(
                    "alias_to_slug collision: %r claimed by %r, ignoring %r",
                    key, alias_to_slug[key], c.slug,
                )
    return alias_to_slug


def _build_attribution_static_block(roster: CharacterRoster) -> str:
    """Build the cacheable part of the attribution prompt: instructions + slug roster.

    Identical for every chapter in a book, making it safe to cache via Anthropic's
    ephemeral prompt caching on the cloud path.
    """
    roster_lines = ["SLUG | CANONICAL NAME | ALIASES | PRONOUNS"]
    for c in roster.characters:
        aliases = ", ".join(c.aliases) if c.aliases else ""
        roster_lines.append(f"{c.slug} | {c.canonical_name} | {aliases} | {c.gender}")
    roster_block = "\n".join(roster_lines)

    return f"""You are a literary analyst performing speaker attribution.

CHARACTER ROSTER (use the slug field as the s value):
{roster_block}

For each [QUOTE:N] tag in the annotated chapter, return one object with:
- q: the N from [QUOTE:N]
- s: character slug, "NARRATOR", or "Unknown"

Rules:
- Every [QUOTE:N] present MUST appear in your response — no exceptions, no skipping
- hint= is strong guidance (~90% accurate) — override only if context clearly contradicts it
- guess= is a weaker signal (~50% accurate) — use as a tiebreaker, not a determination
- pronoun= filters the roster to characters with matching pronouns
- "NARRATOR" for scare quotes, titles, labels, non-spoken text
- "Unknown" only when you have genuinely no basis for any guess
- Read [NARRATOR] passages for context — they are pre-labeled, do not return them"""


def _build_attribution_dynamic_block(annotated_text: str) -> str:
    """Build the per-chapter part of the attribution prompt: annotated chapter text."""
    return f"ANNOTATED CHAPTER:\n{annotated_text}"
