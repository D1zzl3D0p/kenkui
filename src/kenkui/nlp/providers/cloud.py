"""CloudProvider — two-pass large-context NLP via LiteLLM + instructor.

Pass 1 (build_roster):  Send the full book text (chunked at chapter
    boundaries if needed) to the LLM and extract a rich CharacterRoster.
Pass 2 (attribute_chapter): Send each chapter + the full roster and get
    back speaker attributions with slug-keyed speakers.
"""

from __future__ import annotations

import os
from collections.abc import Callable

import instructor
import litellm

from kenkui.models import AppConfig, Chapter
from kenkui.nlp.models import (
    AttributionResult,
    CharacterRecord,
    CharacterRoster,
    slugify,
)

# ---------------------------------------------------------------------------
# Token / budget utilities
# ---------------------------------------------------------------------------

_CHARS_PER_TOKEN = 4          # rough heuristic (conservative)
_SYSTEM_PROMPT_TOKENS = 2_000  # headroom for system/instruction prompt
_MIN_OUTPUT_RESERVE = 0.20     # always reserve at least 20% of context for output
_TOKENS_PER_CHARACTER = 600    # estimated tokens per CharacterRecord in JSON output


def estimate_tokens(text: str) -> int:
    """Estimate token count for *text* using a 4-chars-per-token heuristic."""
    return max(1, len(text) // _CHARS_PER_TOKEN)


def compute_output_budget(context_limit: int, estimated_chars: int) -> int:
    """Compute how many tokens to reserve for LLM output.

    Takes the larger of: 20% of context_limit, or tokens needed for
    *estimated_chars* CharacterRecords.
    """
    floor = int(context_limit * _MIN_OUTPUT_RESERVE)
    character_estimate = estimated_chars * _TOKENS_PER_CHARACTER
    return max(floor, character_estimate)


def needs_chunking(
    input_tokens: int,
    output_budget: int,
    system_tokens: int,
    context_limit: int,
) -> bool:
    """Return True if the combined token count exceeds the model's context limit."""
    return (input_tokens + output_budget + system_tokens) > context_limit


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
# Prompt builders
# ---------------------------------------------------------------------------


def _build_roster_prompt(book_text: str, series_roster: "CharacterRoster | None") -> str:
    """Build the prompt for whole-book character extraction."""
    known_chars = ""
    if series_roster and series_roster.characters:
        lines = []
        for c in series_roster.characters:
            aliases = ", ".join(c.aliases) if c.aliases else "none"
            lines.append(f"  - slug={c.slug!r}, name={c.canonical_name!r}, aliases=[{aliases}]")
        known_chars = (
            "\n\nKNOWN CHARACTERS FROM SERIES (use their established slugs exactly):\n"
            + "\n".join(lines)
        )

    return f"""You are a literary analyst. Extract every named character from the book text below.{known_chars}

For each character return:
- slug: stable lowercase underscore identifier derived from canonical_name (e.g. "elizabeth_bennet")
- canonical_name: the most complete formal name used in the text
- aliases: all other name forms (nicknames, last-name-only references, epithets)
- titles: positional titles with chapter indices and book_slug=null for this book
- gender: pronouns used ("he/him", "she/her", "they/them", or "" if unclear)
- role: one of protagonist, antagonist, supporting, minor
- description: one sentence summarising the character (or "" for very minor characters)
- chapters: list of chapter indices this character appears in
- first_appearance: [null, chapter_index] tuple (book_slug is set by the caller)
- last_appearance: [null, chapter_index] tuple

BOOK TEXT:
{book_text}"""


def _build_attribution_prompt(chapter_text: str, roster: "CharacterRoster") -> str:
    """Build the prompt for per-chapter quote attribution."""
    roster_lines = []
    for c in roster.characters:
        aliases = ", ".join(c.aliases) if c.aliases else "none"
        roster_lines.append(f"  slug={c.slug!r}  name={c.canonical_name!r}  aliases=[{aliases}]")
    roster_block = "\n".join(roster_lines)

    return f"""You are a literary analyst performing speaker attribution.

CHARACTER ROSTER (use the slug field as the speaker value):
{roster_block}

For each dialogue quote or inner monologue in the chapter text, return:
- quote_id: sequential integer starting at 1
- speaker: the character's slug from the roster above, or "NARRATOR", or "Unknown"
- emotion: one of neutral, happy, sad, angry, fearful, surprised, disgusted
- confidence: 1 (very uncertain) to 5 (very confident)

Rules:
- Use the exact slug from the roster — never a canonical name or alias
- "NARRATOR" for narration, scene descriptions, or when no clear speaker
- "Unknown" only when the speaker cannot be identified with any confidence
- Scare quotes and titles (e.g. "The King") are not spoken dialogue — mark as NARRATOR

CHAPTER TEXT:
{chapter_text}"""


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
# CloudProvider
# ---------------------------------------------------------------------------


class CloudProvider:
    """NLP provider using LiteLLM + instructor for cloud model access."""

    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self._client = instructor.from_litellm(litellm.completion)
        self._inject_credentials()

    def _inject_credentials(self) -> None:
        """Load credentials from credentials.toml and set env vars for LiteLLM."""
        from kenkui.config import inject_provider_env_vars, load_provider_credentials
        inject_provider_env_vars(load_provider_credentials())

    def _resolved_model(self) -> str:
        """Return the LiteLLM model string to use.

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
            "anthropic": "claude-sonnet-4-6",
            "openai": "gpt-4o",
            "google": "gemini/gemini-2.0-flash",
        }
        return defaults.get(self.config.nlp_provider, "gpt-4o")

    def build_roster(
        self,
        chapters: list[Chapter],
        series_roster: CharacterRoster | None = None,
        progress_callback: Callable[[str], None] | None = None,
    ) -> CharacterRoster:
        """Extract character roster — whole-book pass, chunked if needed."""
        model = self._resolved_model()
        context_limit = _context_limit_for(model)

        chapter_texts = [
            f"[Chapter {ch.index}]\n" + "\n".join(ch.paragraphs)
            for ch in chapters
        ]
        full_text = "\n\n".join(chapter_texts)

        estimated_unique = max(20, len(full_text) // 5000)
        output_budget = compute_output_budget(context_limit, estimated_unique)
        input_tokens = estimate_tokens(full_text)

        if needs_chunking(input_tokens, output_budget, _SYSTEM_PROMPT_TOKENS, context_limit):
            if progress_callback:
                progress_callback("Book exceeds context limit — chunking for roster extraction")
            return self._build_roster_chunked(chapters, series_roster, model, output_budget, progress_callback)

        if progress_callback:
            progress_callback("Extracting character roster (single pass)")

        prompt = _build_roster_prompt(full_text, series_roster)
        return self._client.chat.completions.create(
            model=model,
            response_model=CharacterRoster,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=output_budget,
        )

    def _build_roster_chunked(
        self,
        chapters: list[Chapter],
        series_roster: "CharacterRoster | None",
        model: str,
        output_budget: int,
        progress_callback: "Callable[[str], None] | None",
    ) -> "CharacterRoster":
        context_limit = _context_limit_for(model)
        max_input = context_limit - output_budget - _SYSTEM_PROMPT_TOKENS
        segments = split_chapters_into_segments(chapters, max_tokens_per_segment=max_input)

        if progress_callback:
            progress_callback(f"Chunked into {len(segments)} segments for roster extraction")

        partial_rosters: list[CharacterRoster] = []
        for i, segment in enumerate(segments):
            if progress_callback:
                progress_callback(f"Extracting characters from segment {i + 1}/{len(segments)}")
            segment_text = "\n\n".join(
                f"[Chapter {ch.index}]\n" + "\n".join(ch.paragraphs)
                for ch in segment
            )
            prompt = _build_roster_prompt(segment_text, series_roster)
            partial = self._client.chat.completions.create(
                model=model,
                response_model=CharacterRoster,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=output_budget,
            )
            partial_rosters.append(partial)

        book_slug = getattr(self.config, "_book_slug", "unknown")
        return merge_rosters(partial_rosters, book_slug=book_slug)

    def attribute_chapter(
        self,
        chapter: Chapter,
        roster: CharacterRoster,
        progress_callback: Callable[[str], None] | None = None,
    ) -> AttributionResult:
        """Attribute quotes in *chapter* to speakers — single chapter pass."""
        model = self._resolved_model()

        if progress_callback:
            progress_callback(f"Attributing chapter {chapter.index}")

        chapter_text = "\n".join(chapter.paragraphs)
        prompt = _build_attribution_prompt(chapter_text, roster)
        context_limit = _context_limit_for(model)
        output_budget = compute_output_budget(context_limit, len(roster.characters))

        return self._client.chat.completions.create(
            model=model,
            response_model=AttributionResult,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=output_budget,
        )
