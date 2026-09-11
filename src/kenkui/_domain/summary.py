"""REPL-facing summaries of a pipeline's three intent tiers.

Each summary is a tuple scan over ``operations`` with no parse and no I/O:
cheap enough to be a property, unlike ``inspect()`` or ``script()``. Rules
and operations render in declaration order and are never sorted -- that
order is the precedence tiebreaker the resolver actually uses, so a sorted
view would misrepresent it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from kenkui._domain.operations import (
    DEFAULT_IDENTITY_MODEL,
    Annotations,
    AssignVoices,
    AttributeQuotes,
    Attributions,
    InferCharacters,
    MetadataIntent,
    Pauses,
    Pronunciations,
    Select,
    SelectChapterRange,
    SelectChapters,
    Series,
    Silences,
    SpokenForm,
    SynthesizeSpeech,
)
from kenkui._domain.paths import Any, Exact, Last, OneOf, Span, parse_path, render_path
from kenkui._domain.tiers import Tier, tier_of

if TYPE_CHECKING:
    from collections.abc import Iterator, Mapping

    from kenkui._domain.operations import Operation
    from kenkui._domain.paths import Pattern, Selector
    from kenkui._domain.tuning import Rule

# Lists longer than this per tuning kind truncate for display; iteration
# still yields every rule regardless.
_MAX_SHOWN = 3
_LEVEL_PREFIXES: dict[str, str] = {
    "chapter": "",
    "paragraph": "¶",
    "line": "l",
    "sentence": "s",
    "phrase": "p",
}


@dataclass(frozen=True, slots=True)
class TierSummary:
    """A read-only view over one tier's declared operations, for the REPL."""

    label: str
    operations: tuple[Operation, ...]

    def __iter__(self) -> Iterator[Operation]:
        """Yield this tier's operations in declaration order."""
        return iter(self.operations)

    def __repr__(self) -> str:
        """Render one line per declared operation."""
        if not self.operations:
            return f"<{self.label}: nothing set>"
        lines = "\n".join(f"  {_format_operation(op)}" for op in self.operations)
        return f"<{self.label}\n{lines}\n>"


@dataclass(frozen=True, slots=True)
class RuleGroup:
    """One tuning operation kind's rules, and where its loaded prefix ends."""

    label: str
    rules: tuple[Rule, ...]
    loaded: int


@dataclass(frozen=True, slots=True)
class TuningSummary:
    """A read-only view over this book's tuning rules, for the REPL."""

    groups: tuple[RuleGroup, ...]

    def __iter__(self) -> Iterator[Rule]:
        """Yield every rule of every kind, in declaration order."""
        return iter(rule for group in self.groups for rule in group.rules)

    def __repr__(self) -> str:
        """Render each kind's rules, truncated for display, never for iteration."""
        populated = tuple(group for group in self.groups if group.rules)
        if not populated:
            return "<tuning: no rules>"
        sections = "\n".join(_render_group(group) for group in populated)
        return f"<tuning\n{sections}\n>"


def summarize_identity(operations: tuple[Operation, ...]) -> TierSummary:
    """Collect this pipeline's identity-tier operations, as declared."""
    return TierSummary("identity", _tier_operations(operations, "identity"))


def summarize_style(operations: tuple[Operation, ...]) -> TierSummary:
    """Collect this pipeline's style-tier operations.

    Style settings replace rather than accumulate, so each entry already
    holds its effective value -- there is nothing further to resolve here.
    """
    return TierSummary("style", _tier_operations(operations, "style"))


def summarize_tuning(operations: tuple[Operation, ...]) -> TuningSummary:
    """Collect this pipeline's tuning rules by kind, marking unsaved ones.

    A rule is unsaved when its index is at or beyond the count of rules the
    ``Annotations`` operation loaded for that kind -- loaded rules form a
    prefix of the rule tuple, so that boundary is exact. Absent an
    ``Annotations`` operation, nothing has been loaded and every rule
    qualifies.
    """
    loaded = _loaded_counts(operations)
    groups: list[RuleGroup] = []
    for operation in operations:
        if isinstance(operation, Attributions):
            label = "attributions"
            groups.append(RuleGroup(label, operation.rules, loaded.get(label, 0)))
        elif isinstance(operation, Silences):
            label = "silences"
            groups.append(RuleGroup(label, operation.rules, loaded.get(label, 0)))
        elif isinstance(operation, Pronunciations):
            label = "pronunciations"
            groups.append(RuleGroup(label, operation.rules, loaded.get(label, 0)))
    return TuningSummary(tuple(groups))


def _loaded_counts(operations: tuple[Operation, ...]) -> Mapping[str, int]:
    annotations = next((op for op in operations if isinstance(op, Annotations)), None)
    return annotations.loaded if annotations is not None else {}


def _tier_operations(
    operations: tuple[Operation, ...], tier: Tier
) -> tuple[Operation, ...]:
    return tuple(op for op in operations if tier_of(type(op)) == tier)


def _render_group(group: RuleGroup) -> str:
    shown = group.rules[:_MAX_SHOWN]
    lines = [f"  {group.label} ({len(group.rules)}):"]
    lines.extend(f"    {_render_rule(rule, group.loaded)}" for rule in shown)
    remaining = len(group.rules) - len(shown)
    if remaining:
        lines.append(f"    … {remaining} more")
    return "\n".join(lines)


def _render_rule(rule: Rule, loaded: int) -> str:
    marker = "  [unsaved]" if rule.index >= loaded else ""
    return f"{render_pattern(rule.where)} -> {rule.value!r}{marker}"


def render_pattern(pattern: Pattern) -> str:
    """Render a rule's match set for display.

    Exact coordinates go through ``render_path``, which elides line one as a
    degenerate level; the stored pattern itself keeps it. Set-valued
    selectors (wildcards, spans, lists, "last") render alongside as extra
    components, since they carry information ``render_path`` cannot express.
    """
    if not pattern:
        return "*"
    exact = {
        level: selector.value
        for level, selector in pattern.items()
        if isinstance(selector, Exact)
    }
    base = render_path(parse_path(exact)) if exact else "*"
    extras = tuple(
        _render_selector(level, selector)
        for level, selector in pattern.items()
        if not isinstance(selector, (Exact, Any))
    )
    if not extras:
        return base
    joined = "  ".join(extras)
    return joined if base == "*" else f"{base}  {joined}"


def _render_selector(level: str, selector: Selector) -> str:
    if isinstance(selector, Last):
        value = "last"
    elif isinstance(selector, OneOf):
        value = "{" + ",".join(str(item) for item in sorted(selector.values)) + "}"
    elif isinstance(selector, Span):
        value = f"{selector.lo}..{selector.hi}"
    else:  # pragma: no cover - callers exclude Exact and Any before calling.
        msg = f"unexpected selector: {selector!r}"
        raise TypeError(msg)
    return f"{_LEVEL_PREFIXES[level]}{value}"


def _format_operation(operation: Operation) -> str:  # noqa: C901, PLR0911
    if isinstance(operation, MetadataIntent):
        return _format_metadata(operation)
    if isinstance(operation, Series):
        return _format_series(operation)
    if isinstance(operation, SelectChapters):
        return f"select_chapters({', '.join(operation.chapter_ids)})"
    if isinstance(operation, SelectChapterRange):
        return f"select_chapter_range({operation.start_id}, {operation.end_id})"
    if isinstance(operation, Select):
        patterns = ", ".join(render_pattern(p) for p in operation.patterns)
        return f"select({patterns})"
    if isinstance(operation, Pauses):
        return _format_pauses(operation)
    if isinstance(operation, SpokenForm):
        return _format_spoken_form(operation)
    if isinstance(operation, InferCharacters):
        if operation.identity_model_id == DEFAULT_IDENTITY_MODEL:
            return f"infer_characters({operation.model_id!r})"
        return (
            f"infer_characters({operation.model_id!r}, "
            f"identity={operation.identity_model_id!r})"
        )
    if isinstance(operation, AttributeQuotes):
        return f"attribute_quotes({operation.model_id!r})"
    if isinstance(operation, AssignVoices):
        return _format_assign_voices(operation)
    if isinstance(operation, SynthesizeSpeech):
        return "tts()"
    msg = f"unexpected operation: {operation!r}"  # pragma: no cover
    raise TypeError(msg)  # pragma: no cover - identity/style are exhaustive above.


def _format_metadata(operation: MetadataIntent) -> str:
    parts: list[str] = []
    if operation.title is not None:
        parts.append(f"title={operation.title!r}")
    if operation.author is not None:
        parts.append(f"author={operation.author!r}")
    if operation.cover != "source":
        parts.append(f"cover={operation.cover}")
    return f"metadata({', '.join(parts)})"


def _format_series(operation: Series) -> str:
    parts = [f"series_id={operation.series_id!r}"]
    if operation.book is not None:
        parts.append(f"book={operation.book}")
    if operation.allow_recast:
        parts.append("allow_recast=True")
    if operation.allow_narrator_change:
        parts.append("allow_narrator_change=True")
    return f"series({', '.join(parts)})"


def _format_pauses(operation: Pauses) -> str:
    fields = (
        ("chapter_ms", operation.chapter_ms),
        ("heading_before_ms", operation.heading_before_ms),
        ("heading_after_ms", operation.heading_after_ms),
        ("paragraph_ms", operation.paragraph_ms),
        ("line_ms", operation.line_ms),
    )
    parts = [f"{name}={value}ms" for name, value in fields if value]
    return f"pauses({', '.join(parts)})" if parts else "pauses(off)"


def _format_spoken_form(operation: SpokenForm) -> str:
    parts = [f"numbers={operation.numbers!r}"]
    if not operation.builtin_lexicon:
        parts.append("builtin=False")
    if operation.lexicon:
        parts.append(f"lexicon({len(operation.lexicon)} entries)")
    parts.extend(f"{name}={value}" for name, value in operation.features)
    return f"pronounce({', '.join(parts)})"


def _format_assign_voices(operation: AssignVoices) -> str:
    parts = [f"narrator={operation.narrator_voice_id!r}"]
    if operation.unknown_voice_id != operation.narrator_voice_id:
        parts.append(f"unknown={operation.unknown_voice_id!r}")
    if operation.cast:
        parts.append(f"cast({len(operation.cast)})")
    parts.append(f"method={operation.method!r}")
    return f"assign_voices({', '.join(parts)})"
