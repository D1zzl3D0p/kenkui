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

import re
from collections import defaultdict

from .models import Quote

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


def _build_alias_pattern(alias_to_slug: dict[str, str]) -> re.Pattern | None:
    """Compile a single OR-pattern from all known aliases (longest first)."""
    if not alias_to_slug:
        return None
    # Sort by length descending so longer aliases match before substrings.
    aliases = sorted(alias_to_slug.keys(), key=len, reverse=True)
    escaped = [re.escape(a) for a in aliases]
    return re.compile(r"\b(" + "|".join(escaped) + r")\b", re.IGNORECASE)


def _extract_hints(
    para: str,
    quote: Quote,
    alias_to_slug: dict[str, str],
    slug_to_pronoun: dict[str, str],
    para_start_offset: int,
) -> str:
    """Return the hint/pronoun suffix for a QUOTE tag.

    Three independent tiers:

    1. Strong hint (``hint=``): attribution verb + alias within ±120 chars.
    2. Weak guess (``guess=``): alias immediately after the closing quote mark.
    3. Pronoun (``pronoun=``): gendered pronoun within ±60 chars.

    All three can coexist and are concatenated in hint → guess → pronoun order.
    """
    alias_pat = _build_alias_pattern(alias_to_slug)
    suffix = ""

    # Local position of the quote within *para*.
    local_start = quote.char_offset - para_start_offset
    local_end = local_start + len(quote.text)

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
            r"\b" + _VERB_GROUP + r"\s+(" + alias_pat.pattern[3:-3] + r")\b"
            r"|"
            r"\b(" + alias_pat.pattern[3:-3] + r")\s+" + _VERB_GROUP + r"\b",
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
    # -----------------------------------------------------------------------
    if alias_pat is not None:
        after_start = max(0, local_end - 1)
        after_end = min(len(para), after_start + 80)
        after_text = para[after_start:after_end]

        guess_pat = re.compile(
            r'^["\u201d][,.]?\s+(' + alias_pat.pattern[3:-3] + r")\b",
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

    Returns:
        A single string with ``\\n\\n``-joined annotated segments.
    """
    if not paragraphs:
        return ""

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
            local_end = local_start + len(quote.text)

            # Guard against bad offsets.
            local_start = max(cursor, min(local_start, len(para)))
            local_end = max(local_start, min(local_end, len(para)))

            # Leading / between-quote narration.
            before = para[cursor:local_start]
            if before.strip():
                segments.append(f"[NARRATOR] {before}")

            # Build the QUOTE tag.
            hint_suffix = _extract_hints(
                para, quote, alias_to_slug, slug_to_pronoun, para_start
            )
            segments.append(f"[QUOTE:{quote.id}{hint_suffix}] {quote.text}")

            cursor = local_end

        # Trailing narration after the last quote.
        trailing = para[cursor:]
        if trailing.strip():
            segments.append(f"[NARRATOR] {trailing}")

    return "\n\n".join(segments)
