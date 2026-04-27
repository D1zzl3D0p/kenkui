"""CloudProvider — two-pass large-context NLP via LiteLLM + instructor.

Pass 1 (build_roster):  Send the full book text (chunked at chapter
    boundaries if needed) to the LLM and extract a rich CharacterRoster.
Pass 2 (attribute_chapter): Send each chapter + the full roster and get
    back speaker attributions with slug-keyed speakers.
"""

from __future__ import annotations

import logging
import time
import os
from collections.abc import Callable

logger = logging.getLogger(__name__)

# Rate-limit retry settings
_RATE_LIMIT_WAIT_S = 60   # seconds to wait between rate-limit retries
_RATE_LIMIT_MAX_RETRIES = 3

import instructor
import litellm

from kenkui.models import AppConfig, Chapter
from kenkui.nlp.models import (
    AttributionResult,
    AttributionResultWire,
    CharacterRecord,
    CharacterRoster,
    CharacterRosterFullWire,
    CharacterRosterWire,
    attribution_wire_to_full,
    roster_wire_to_full,
    slugify,
)

# ---------------------------------------------------------------------------
# Token / budget utilities
# ---------------------------------------------------------------------------

_CHARS_PER_TOKEN = 4          # rough heuristic (conservative)
_SYSTEM_PROMPT_TOKENS = 2_000  # headroom for system/instruction prompt
_TOKENS_PER_CHARACTER_WIRE = 80   # compact wire CharacterRecordWire (slug+name+aliases+role+gender)
_TOKENS_PER_CHARACTER_FULL = 600  # full CharacterRecord with chapters/appearances/titles
_TOKENS_PER_ATTRIBUTION_ITEM = 20 # slim AttributionItemWire (quote_id + speaker + confidence)
_SERIES_ROSTER_CAP = 100          # max characters injected from a series roster


def estimate_tokens(text: str) -> int:
    """Estimate token count for *text* using a 4-chars-per-token heuristic."""
    return max(1, len(text) // _CHARS_PER_TOKEN)


def compute_roster_output_budget(estimated_chars: int, compact: bool) -> int:
    """Compute max_tokens for a roster extraction call.

    Uses a compact per-character estimate when the wire model is in compact mode
    (no chapters/appearances), and a fuller estimate otherwise.  Always allows
    at least 2048 tokens for very small rosters.
    """
    per_char = _TOKENS_PER_CHARACTER_WIRE if compact else _TOKENS_PER_CHARACTER_FULL
    return max(2048, estimated_chars * per_char)


def compute_attribution_output_budget(num_quotes: int) -> int:
    """Compute max_tokens for a chapter attribution call.

    Sized for the slim AttributionItemWire schema with a 4x safety margin.
    """
    return max(512, num_quotes * _TOKENS_PER_ATTRIBUTION_ITEM * 4)


def needs_chunking(
    input_tokens: int,
    output_budget: int,
    system_tokens: int,
    context_limit: int,
    rate_limit_tpm: int | None = None,
) -> bool:
    """Return True if chunking is needed — context overflow or rate-limit excess."""
    if (input_tokens + output_budget + system_tokens) > context_limit:
        return True
    if rate_limit_tpm is not None:
        if input_tokens > int(rate_limit_tpm * _RATE_LIMIT_CHUNK_SAFETY):
            return True
    return False


# ---------------------------------------------------------------------------
# Model context limit registry
# ---------------------------------------------------------------------------

_CONTEXT_LIMITS: dict[str, int] = {
    "claude-opus-4-6": 200_000,
    "claude-sonnet-4-6": 200_000,
    "claude-haiku-4-5-20251001": 200_000,
    "claude-3-5-sonnet-20241022": 200_000,
    "gpt-4o": 128_000,
    "gpt-4o-mini": 128_000,
    "gemini/gemini-2.0-flash": 1_000_000,
    "gemini/gemini-1.5-pro": 2_000_000,
}
_DEFAULT_CONTEXT_LIMIT = 128_000


def _context_limit_for(model: str) -> int:
    return _CONTEXT_LIMITS.get(model, _DEFAULT_CONTEXT_LIMIT)


# ---------------------------------------------------------------------------
# Rate-limit registry  (input tokens per minute, keyed by model name prefix)
# ---------------------------------------------------------------------------

_RATE_LIMITS_TPM: dict[str, int] = {
    "claude-haiku-": 100_000,  # Haiku has higher TPM than Sonnet/Opus
    "claude-": 30_000,          # Sonnet, Opus, and other Claude models
}
_RATE_LIMIT_CHUNK_SAFETY = 0.75      # use 75% of TPM as max chunk input tokens
_RATE_LIMIT_THROTTLE_BUFFER = 1.10   # sleep 10% longer than the calculated minimum


def _rate_limit_tpm_for(model: str) -> int | None:
    """Return tokens-per-minute input limit for *model*, or None if unconstrained."""
    for prefix, tpm in _RATE_LIMITS_TPM.items():
        if model.startswith(prefix):
            return tpm
    return None


# ---------------------------------------------------------------------------
# Prompt builders
# ---------------------------------------------------------------------------


def _build_roster_prompt(
    book_text: str,
    series_roster: "CharacterRoster | None",
    compact_roster: bool = False,
    descriptions_protagonists_only: bool = False,
) -> str:
    """Build the prompt for whole-book character extraction."""
    known_chars = ""
    if series_roster and series_roster.characters:
        # Cap to the most-mentioned characters to avoid huge prompts for long series.
        chars_to_inject = sorted(
            series_roster.characters, key=lambda c: c.mention_count, reverse=True
        )[:_SERIES_ROSTER_CAP]
        lines = []
        for c in chars_to_inject:
            aliases = ", ".join(c.aliases) if c.aliases else ""
            line = f"  {c.slug} | {c.canonical_name}"
            if aliases:
                line += f" | {aliases}"
            lines.append(line)
        known_chars = (
            "\n\nKNOWN CHARACTERS FROM SERIES (reuse these slugs exactly):\n"
            + "\n".join(lines)
        )

    if descriptions_protagonists_only:
        description_instruction = (
            "- description: one sentence summarising the character — "
            "only for protagonist and antagonist roles; leave empty for all others"
        )
    else:
        description_instruction = (
            '- description: one sentence summarising the character (or "" for very minor characters)'
        )

    if compact_roster:
        position_fields = ""
    else:
        position_fields = (
            "\n- chapters: list of chapter indices this character appears in"
            "\n- first_appearance: [null, chapter_index] tuple (book_slug is set by the caller)"
            "\n- last_appearance: [null, chapter_index] tuple"
        )

    return f"""You are a literary analyst. Extract every named character from the book text below.{known_chars}

For each character return:
- slug: stable lowercase underscore identifier derived from canonical_name (e.g. "elizabeth_bennet")
- canonical_name: the most complete formal name used in the text
- aliases: all other name forms (nicknames, last-name-only references, epithets)
- titles: list of positional titles or honorifics (e.g. "Mr.", "Queen of Andor") — title strings only
- gender: pronouns used ("he/him", "she/her", "they/them", or "" if unclear)
- role: one of protagonist, antagonist, supporting, minor
- {description_instruction}{position_fields}

BOOK TEXT:
{book_text}"""


def _build_attribution_static_block(
    roster: "CharacterRoster",
) -> str:
    """Build the static (cacheable) part of the attribution prompt: instructions + roster.

    This block is identical for every chapter in a book, so it can be cached once
    and reused across all chapter calls.
    """
    roster_lines = ["SLUG | CANONICAL NAME | ALIASES | PRONOUNS"]
    for c in roster.characters:
        aliases = ", ".join(c.aliases) if c.aliases else ""
        line = f"{c.slug} | {c.canonical_name} | {aliases} | {c.gender}"
        roster_lines.append(line)
    roster_block = "\n".join(roster_lines)

    return f"""You are a literary analyst performing speaker attribution.

CHARACTER ROSTER (use the slug field as the speaker value):
{roster_block}

For each [QUOTE:N] tag in the annotated chapter, return:
- quote_id: the N from [QUOTE:N]
- speaker: character slug, "NARRATOR", or "Unknown"
- confidence: 1–5

Rules:
- Every [QUOTE:N] present MUST appear in your response — no exceptions, no skipping
- hint= is strong guidance (~90% accurate) — override only if context clearly contradicts it
- guess= is a weaker signal (~50% accurate) — use as a tiebreaker, not a determination
- pronoun= filters the roster to characters with matching pronouns
- "NARRATOR" for scare quotes, titles, labels, non-spoken text
- "Unknown" only when you have genuinely no basis for any guess
- Read [NARRATOR] passages for context — they are pre-labeled, do not return them"""


def _build_attribution_dynamic_block(annotated_text: str) -> str:
    """Build the dynamic (per-chapter) part: annotated chapter text."""
    return f"ANNOTATED CHAPTER:\n{annotated_text}"


# ---------------------------------------------------------------------------
# Chunking and merge utilities
# ---------------------------------------------------------------------------


def split_chapters_into_segments(
    chapters: "list[Chapter]",
    max_tokens_per_segment: int,
    overlap: int = 2,
) -> "list[list[Chapter]]":
    """Split *chapters* into overlapping segments capped at *max_tokens_per_segment*.

    Boundaries are always at chapter edges. Adjacent segments share *overlap* chapters.
    """
    segments: list[list] = []
    current: list = []
    current_tokens = 0

    for ch in chapters:
        ch_tokens = estimate_tokens("\n".join(ch.paragraphs))
        if current and (current_tokens + ch_tokens) > max_tokens_per_segment:
            segments.append(current)
            current = current[-overlap:] if len(current) >= overlap else list(current)
            current_tokens = sum(estimate_tokens("\n".join(c.paragraphs)) for c in current)
        current.append(ch)
        current_tokens += ch_tokens

    if current:
        segments.append(current)

    return segments


def merge_rosters(rosters: "list[CharacterRoster]", book_slug: str) -> "CharacterRoster":
    """Merge multiple partial rosters into a single deduplicated CharacterRoster."""
    merged: list[CharacterRecord] = []

    def _find_match(record: CharacterRecord) -> int | None:
        record_names = {record.slug, record.canonical_name.lower()} | {a.lower() for a in record.aliases}
        for i, existing in enumerate(merged):
            existing_names = {existing.slug, existing.canonical_name.lower()} | {a.lower() for a in existing.aliases}
            if record.slug == existing.slug or record_names & existing_names:
                return i
        return None

    for roster in rosters:
        for record in roster.characters:
            idx = _find_match(record)
            if idx is None:
                r = record.model_copy(deep=True)
                if r.first_appearance and r.first_appearance[0] is None:
                    r.first_appearance = (book_slug, r.first_appearance[1])
                if r.last_appearance and r.last_appearance[0] is None:
                    r.last_appearance = (book_slug, r.last_appearance[1])
                merged.append(r)
            else:
                existing = merged[idx]
                all_aliases = list(dict.fromkeys(existing.aliases + record.aliases))
                all_chapters = sorted(set(existing.chapters) | set(record.chapters))

                def _chapter_key(app: "tuple[str, int] | None") -> int:
                    return app[1] if app else 999999

                first = existing.first_appearance if _chapter_key(existing.first_appearance) <= _chapter_key(record.first_appearance) else record.first_appearance
                last = existing.last_appearance if _chapter_key(existing.last_appearance) >= _chapter_key(record.last_appearance) else record.last_appearance
                if first and first[0] is None:
                    first = (book_slug, first[1])
                if last and last[0] is None:
                    last = (book_slug, last[1])
                description = existing.description if len(existing.description) >= len(record.description) else record.description
                role = existing.role or record.role
                canonical = existing.canonical_name if existing.mention_count >= record.mention_count else record.canonical_name
                slug = slugify(canonical)
                merged[idx] = CharacterRecord(
                    slug=slug,
                    canonical_name=canonical,
                    aliases=all_aliases,
                    titles=existing.titles + record.titles,
                    gender=existing.gender or record.gender,
                    role=role,
                    description=description,
                    chapters=all_chapters,
                    first_appearance=first,
                    last_appearance=last,
                    mention_count=existing.mention_count + record.mention_count,
                    quote_count=existing.quote_count + record.quote_count,
                )

    return CharacterRoster(characters=merged)


# ---------------------------------------------------------------------------
# Rate-limit retry helper
# ---------------------------------------------------------------------------


def _call_with_rate_limit_retry(fn, *args, progress_callback=None, **kwargs):
    """Call *fn(*args, **kwargs)*, retrying on RateLimitError after a delay.

    Waits *_RATE_LIMIT_WAIT_S* seconds between attempts and retries up to
    *_RATE_LIMIT_MAX_RETRIES* additional times before re-raising.

    Also retries on IncompleteOutputException (instructor raises this when the
    LLM hits max_tokens mid-response) by doubling max_tokens on each retry.
    """
    for attempt in range(_RATE_LIMIT_MAX_RETRIES + 1):
        try:
            return fn(*args, **kwargs)
        except Exception as exc:
            exc_type = type(exc).__name__
            is_rate_limit = (
                "RateLimitError" in exc_type
                or "rate_limit" in str(exc).lower()
                or "rate limit" in str(exc).lower()
            )
            is_incomplete = "IncompleteOutputException" in exc_type
            if is_rate_limit and attempt < _RATE_LIMIT_MAX_RETRIES:
                wait = _RATE_LIMIT_WAIT_S
                msg = f"Rate limit hit — waiting {wait}s before retry {attempt + 1}/{_RATE_LIMIT_MAX_RETRIES}"
                logger.warning(msg)
                if progress_callback:
                    progress_callback(msg)
                time.sleep(wait)
            elif is_incomplete and attempt < _RATE_LIMIT_MAX_RETRIES:
                old_budget = kwargs.get("max_tokens", 0)
                new_budget = old_budget * 2
                kwargs["max_tokens"] = new_budget
                msg = (
                    f"Output truncated at max_tokens={old_budget} — "
                    f"retrying with max_tokens={new_budget} (attempt {attempt + 1}/{_RATE_LIMIT_MAX_RETRIES})"
                )
                logger.warning(msg)
                if progress_callback:
                    progress_callback(msg)
            else:
                raise


# ---------------------------------------------------------------------------
# Compact roster helpers
# ---------------------------------------------------------------------------


def _fill_roster_positions(
    roster: "CharacterRoster",
    chapter_indices: "list[int]",
) -> "CharacterRoster":
    """Back-fill chapters/first_appearance/last_appearance when nlp_compact_roster is on.

    The LLM was not asked to produce these fields, so they default to [] / None.
    We conservatively assign all chapter indices in the segment to every character
    found in that segment, and derive first/last appearance from the min/max index.
    """
    if not chapter_indices:
        return roster
    first_idx = min(chapter_indices)
    last_idx = max(chapter_indices)
    updated = []
    for rec in roster.characters:
        chapters = rec.chapters if rec.chapters else list(chapter_indices)
        first = rec.first_appearance or (None, first_idx)
        last = rec.last_appearance or (None, last_idx)
        updated.append(rec.model_copy(update={
            "chapters": chapters,
            "first_appearance": first,
            "last_appearance": last,
        }))
    return CharacterRoster(characters=updated)


# ---------------------------------------------------------------------------
# CloudProvider
# ---------------------------------------------------------------------------


class CloudProvider:
    """NLP provider using LiteLLM + instructor for cloud model access."""

    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self._client = instructor.from_litellm(litellm.completion, mode=instructor.Mode.MD_JSON)
        self._inject_credentials()

    def _inject_credentials(self) -> None:
        """Load credentials from credentials.toml and set env vars for LiteLLM."""
        from kenkui.config import inject_provider_env_vars, load_provider_credentials
        inject_provider_env_vars(load_provider_credentials())

    def _resolved_model(self) -> str:
        """Return the LiteLLM model string to use for attribution.

        Falls back to provider default from credentials.toml when nlp_model is empty.
        """
        if self.config.nlp_model:
            return self.config.nlp_model
        from kenkui.config import load_provider_credentials
        creds = load_provider_credentials()
        provider_creds = creds.get(self.config.nlp_provider)
        if provider_creds and provider_creds.default_model:
            return provider_creds.default_model
        # Last resort defaults
        defaults = {
            "anthropic": "claude-haiku-4-5-20251001",
            "openai": "gpt-4o",
            "google": "gemini/gemini-2.0-flash",
        }
        return defaults.get(self.config.nlp_provider, "gpt-4o")

    def _resolved_roster_model(self) -> str:
        """Return the model to use for character discovery (roster extraction).

        Uses nlp_roster_model when set, otherwise falls back to _resolved_model().
        This allows using a cheaper model (e.g. haiku) for roster extraction while
        using a more capable model for per-chapter attribution.
        """
        if self.config.nlp_roster_model:
            return self.config.nlp_roster_model
        return self._resolved_model()

    def build_roster(
        self,
        chapters: list[Chapter],
        series_roster: CharacterRoster | None = None,
        progress_callback: Callable[[str], None] | None = None,
        book_path: "Path | None" = None,
    ) -> CharacterRoster:
        """Extract character roster — whole-book pass, chunked if needed."""
        model = self._resolved_roster_model()
        context_limit = _context_limit_for(model)
        rate_limit_tpm = _rate_limit_tpm_for(model)

        chapter_texts = [
            f"[Chapter {ch.index}]\n" + "\n".join(ch.paragraphs)
            for ch in chapters
        ]
        full_text = "\n\n".join(chapter_texts)

        compact_roster = self.config.nlp_compact_roster
        estimated_unique = max(20, len(full_text) // 5000)
        output_budget = compute_roster_output_budget(estimated_unique, compact=compact_roster)
        input_tokens = estimate_tokens(full_text)

        if needs_chunking(input_tokens, output_budget, _SYSTEM_PROMPT_TOKENS, context_limit,
                          rate_limit_tpm=rate_limit_tpm):
            reason = (
                "rate limit"
                if rate_limit_tpm and input_tokens > int(rate_limit_tpm * _RATE_LIMIT_CHUNK_SAFETY)
                else "context limit"
            )
            if progress_callback:
                progress_callback(f"Book requires chunking ({reason}) — splitting for roster extraction")
            return self._build_roster_chunked(
                chapters, series_roster, model, output_budget, progress_callback,
                book_path=book_path, rate_limit_tpm=rate_limit_tpm,
            )

        if progress_callback:
            progress_callback("Extracting character roster (single pass)")

        desc_leads_only = self.config.nlp_descriptions_protagonists_only
        prompt = _build_roster_prompt(
            full_text, series_roster,
            compact_roster=compact_roster,
            descriptions_protagonists_only=desc_leads_only,
        )
        wire_model = CharacterRosterWire if compact_roster else CharacterRosterFullWire
        wire = _call_with_rate_limit_retry(
            self._client.chat.completions.create,
            model=model,
            response_model=wire_model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=output_budget,
            max_retries=1,
            num_retries=0,
            progress_callback=progress_callback,
        )
        roster = roster_wire_to_full(wire)
        if compact_roster:
            all_indices = [ch.index for ch in chapters]
            roster = _fill_roster_positions(roster, all_indices)
        return roster

    def _build_roster_chunked(
        self,
        chapters: list[Chapter],
        series_roster: "CharacterRoster | None",
        model: str,
        output_budget: int,
        progress_callback: "Callable[[str], None] | None",
        book_path: "Path | None" = None,
        rate_limit_tpm: "int | None" = None,
    ) -> "CharacterRoster":
        from kenkui.nlp import cache_chunk_roster, get_cached_chunk_roster

        compact_roster = self.config.nlp_compact_roster
        # Re-derive output_budget using the correct per-character estimate for the wire model
        # used in chunked mode (the caller passes a budget sized for the full book; per-segment
        # rosters are smaller so a tighter budget reduces wasted context).
        seg_estimated_unique = max(10, len(chapters) // 2)
        output_budget = compute_roster_output_budget(seg_estimated_unique, compact=compact_roster)

        context_limit = _context_limit_for(model)
        max_input = context_limit - output_budget - _SYSTEM_PROMPT_TOKENS
        if rate_limit_tpm is not None:
            max_input = min(max_input, int(rate_limit_tpm * _RATE_LIMIT_CHUNK_SAFETY))

        segments = split_chapters_into_segments(chapters, max_tokens_per_segment=max_input)

        if progress_callback:
            progress_callback(f"Chunked into {len(segments)} segments for roster extraction")

        partial_rosters: list[CharacterRoster] = []
        for i, segment in enumerate(segments):
            chapter_indices = [ch.index for ch in segment]

            # Checkpoint resume — skip LLM call if this chunk is already cached
            if book_path is not None:
                cached = get_cached_chunk_roster(book_path, chapter_indices)
                if cached is not None:
                    if progress_callback:
                        progress_callback(
                            f"Resuming: segment {i + 1}/{len(segments)} loaded from cache"
                        )
                    partial_rosters.append(cached)
                    continue  # no tokens sent — skip throttle

            if progress_callback:
                progress_callback(f"Extracting characters from segment {i + 1}/{len(segments)}")

            segment_text = "\n\n".join(
                f"[Chapter {ch.index}]\n" + "\n".join(ch.paragraphs)
                for ch in segment
            )
            segment_tokens = estimate_tokens(segment_text)
            desc_leads_only = self.config.nlp_descriptions_protagonists_only
            prompt = _build_roster_prompt(
                segment_text, series_roster,
                compact_roster=compact_roster,
                descriptions_protagonists_only=desc_leads_only,
            )

            wire_model = CharacterRosterWire if compact_roster else CharacterRosterFullWire
            wire = _call_with_rate_limit_retry(
                self._client.chat.completions.create,
                model=model,
                response_model=wire_model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=output_budget,
                max_retries=1,
                num_retries=0,
                progress_callback=progress_callback,
            )
            partial = roster_wire_to_full(wire)
            if compact_roster:
                partial = _fill_roster_positions(partial, chapter_indices)
            partial_rosters.append(partial)

            # Checkpoint write — persisted before throttle sleep so a SIGINT still saves progress
            if book_path is not None:
                cache_chunk_roster(partial, book_path, chapter_indices)

            # Proactive throttle — pace requests to stay under the per-minute token rate limit
            if rate_limit_tpm is not None and i < len(segments) - 1:
                sleep_s = (segment_tokens / rate_limit_tpm) * 60 * _RATE_LIMIT_THROTTLE_BUFFER
                msg = f"Throttling {sleep_s:.1f}s to respect rate limit before next segment"
                logger.debug(msg)
                if progress_callback:
                    progress_callback(msg)
                time.sleep(sleep_s)

        book_slug = getattr(self.config, "_book_slug", "unknown")
        return merge_rosters(partial_rosters, book_slug=book_slug)

    def attribute_chapter(
        self,
        chapter: Chapter,
        roster: CharacterRoster,
        progress_callback: Callable[[str], None] | None = None,
    ) -> AttributionResult:
        """Attribute quotes in *chapter* to speakers — single chapter pass."""
        from kenkui.nlp.quotes import strip_scare_quotes, extract_quotes
        from kenkui.nlp.annotator import annotate_chapter

        model = self._resolved_model()

        if progress_callback:
            progress_callback(f"Attributing chapter {chapter.index}")

        clean_paragraphs = strip_scare_quotes(chapter.paragraphs)
        quotes = extract_quotes(clean_paragraphs)
        if not quotes:
            return AttributionResult(attributions=[])

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
        slug_to_pronoun = {c.slug: c.gender for c in roster.characters}
        # slug_to_pronoun is currently unused by annotate_chapter (pronoun hints are
        # detected from raw text); kept in the API for future roster-based pronoun lookup.
        annotated_text = annotate_chapter(clean_paragraphs, quotes, alias_to_slug, slug_to_pronoun)

        output_budget = compute_attribution_output_budget(len(quotes))

        static_block = _build_attribution_static_block(roster)
        dynamic_block = _build_attribution_dynamic_block(annotated_text)

        # Prompt caching: cache only the static block (instructions + roster) which is
        # identical for every chapter in this book.  The annotated chapter text is
        # dynamic and must NOT be included in the cached prefix.
        # Only Anthropic models support cache_control; other providers get a plain string.
        # cache_control is only supported for Anthropic/Claude models; non-None TPM = Claude.
        is_anthropic = _rate_limit_tpm_for(model) is not None
        if is_anthropic and estimate_tokens(static_block) >= 1024:
            messages: list[dict] = [{
                "role": "user",
                "content": [
                    {"type": "text", "text": static_block, "cache_control": {"type": "ephemeral"}},
                    {"type": "text", "text": dynamic_block},
                ],
            }]
        else:
            messages = [{"role": "user", "content": f"{static_block}\n\n{dynamic_block}"}]

        # Use slim wire model (AttributionResultWire) which has only quote_id, speaker,
        # confidence — exactly what the new annotated format asks for.
        attr_model = AttributionResultWire
        result = _call_with_rate_limit_retry(
            self._client.chat.completions.create,
            model=model,
            response_model=attr_model,
            messages=messages,
            max_tokens=output_budget,
            max_retries=1,
            num_retries=0,
            progress_callback=progress_callback,
        )
        return attribution_wire_to_full(result)
