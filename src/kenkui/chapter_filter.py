"""
Chapter filtering system with presets and custom regex filters.
Applied sequentially based on command-line argument order.
"""

import logging
import re
from dataclasses import dataclass
from typing import Callable, Literal, TYPE_CHECKING

if TYPE_CHECKING:
    from .helpers import Chapter
    from .chapter_classifier import ChapterTags

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class FilterOperation:
    """A single filter operation to be applied to chapters."""

    type: Literal["preset", "include", "exclude"]
    value: str


@dataclass
class FilterPreset:
    """A named filter preset with description."""

    name: str
    description: str
    filter_fn: Callable[["ChapterTags"], bool]

    def apply(self, tags: "ChapterTags") -> bool:
        """Apply this filter to chapter tags."""
        return self.filter_fn(tags)


class ChapterFilter:
    """Manages filter presets and applies them to chapters."""

    DEFAULT_PRESET = "content-only"

    PRESETS: dict[str, FilterPreset] = {
        "none": FilterPreset(
            name="None",
            description="Include no chapters (use with -i/-e to build custom selection)",
            filter_fn=lambda t: False,
        ),
        "all": FilterPreset(
            name="All Chapters",
            description="Include all chapters including front/back matter",
            filter_fn=lambda t: True,
        ),
        "content-only": FilterPreset(
            name="Content Only",
            description="Exclude front matter, back matter, and title pages (default)",
            filter_fn=lambda t: not (
                t.is_front_matter or t.is_back_matter or t.is_title_page
            ),
        ),
        "chapters-only": FilterPreset(
            name="Main Chapters",
            description="Only main chapters (exclude all dividers)",
            filter_fn=lambda t: t.is_chapter and not t.is_part_divider,
        ),
        "with-parts": FilterPreset(
            name="With Part Dividers",
            description="Main chapters plus Part/Book dividers",
            filter_fn=lambda t: t.is_chapter,
        ),
    }

    def __init__(self, operations: list[FilterOperation]):
        """
        Initialize filter with a sequence of operations.

        Operations are applied in order, with later operations taking precedence.
        """
        self.operations = operations
        self._validate_operations()

    def _validate_operations(self) -> None:
        """Validate all operations have valid presets and regex patterns."""
        for op in self.operations:
            if op.type == "preset":
                if op.value not in self.PRESETS:
                    raise ValueError(
                        f"Unknown preset '{op.value}'. "
                        f"Available: {', '.join(self.PRESETS.keys())}"
                    )
            elif op.type in ("include", "exclude"):
                try:
                    re.compile(op.value)
                except re.error as e:
                    raise ValueError(f"Invalid regex pattern '{op.value}': {e}") from e

    def apply(self, chapters: list["Chapter"]) -> list["Chapter"]:
        """
        Apply all filter operations sequentially.

        Rules:
        - First operation: if preset, use it; if include, start empty then add;
          if exclude, start with all then remove
        - Subsequent operations modify the current selection
        - Later operations override earlier ones
        """
        if not self.operations:
            logger.debug("No filter operations specified, returning all chapters")
            return chapters

        all_chapters = {ch.index: ch for ch in chapters}

        # Determine initial selection based on first operation type
        # - include: start empty (will add matching)
        # - exclude: start with all (will remove matching)
        # - preset: will replace selection entirely
        selected: set[int]
        if self.operations[0].type == "exclude":
            selected = set(all_chapters.keys())
            logger.debug("First operation is exclude, starting with all chapters")
        else:
            selected = set()
            logger.debug(
                "First operation is include/preset, starting with empty selection"
            )

        for i, op in enumerate(self.operations):
            logger.debug(
                f"Applying filter operation {i + 1}/{len(self.operations)}: {op.type}={op.value}"
            )

            if op.type == "preset":
                preset = self.PRESETS[op.value]
                new_selected = {ch.index for ch in chapters if preset.apply(ch.tags)}
                removed = selected - new_selected
                added = new_selected - selected
                selected = new_selected
                logger.debug(
                    f"Preset '{op.value}': {len(added)} chapters added, "
                    f"{len(removed)} chapters removed, {len(selected)} total"
                )

            elif op.type == "include":
                pattern = re.compile(op.value)
                matched = {ch.index for ch in chapters if pattern.search(ch.title)}
                added = matched - selected
                selected.update(matched)
                logger.debug(
                    f"Include pattern '{op.value}': {len(added)} chapters added, "
                    f"{len(selected)} total"
                )

            elif op.type == "exclude":
                pattern = re.compile(op.value)
                matched = {ch.index for ch in chapters if pattern.search(ch.title)}
                removed = matched & selected
                selected -= matched
                logger.debug(
                    f"Exclude pattern '{op.value}': {len(removed)} chapters removed, "
                    f"{len(selected)} remaining"
                )

        # Log summary of filtered chapters
        included = [all_chapters[idx] for idx in sorted(selected)]
        excluded = [ch for ch in chapters if ch.index not in selected]

        logger.info(
            f"Chapter filtering complete: {len(included)} included, "
            f"{len(excluded)} excluded out of {len(chapters)} total"
        )

        for ch in excluded:
            logger.debug(f"Excluded chapter: #{ch.index} '{ch.title}'")

        return included

    @classmethod
    def register_preset(cls, name: str, preset: FilterPreset) -> None:
        """
        Register a new preset for custom filtering.

        This allows extensions and plugins to add custom presets.
        """
        cls.PRESETS[name] = preset
        logger.debug(f"Registered new preset '{name}': {preset.description}")

    @classmethod
    def get_preset(cls, name: str) -> FilterPreset | None:
        """Get a preset by name."""
        return cls.PRESETS.get(name)

    @classmethod
    def apply_preset(
        cls, chapters: list["Chapter"], preset_name: str
    ) -> list["Chapter"]:
        """Apply a preset filter to chapters (legacy method)."""
        preset = cls.get_preset(preset_name)
        if not preset:
            logger.warning(f"Unknown preset '{preset_name}', returning all chapters")
            return chapters

        return [ch for ch in chapters if preset.apply(ch.tags)]

    @classmethod
    def list_presets(cls) -> list[tuple[str, str]]:
        """List available presets with descriptions."""
        return [(name, p.description) for name, p in cls.PRESETS.items()]

    @classmethod
    def get_preset_help_text(cls) -> str:
        """Generate help text for CLI argument."""
        lines = ["Available filter presets:"]
        for name, desc in cls.list_presets():
            lines.append(f"  {name:15} - {desc}")
        return "\n".join(lines)


__all__ = ["FilterOperation", "FilterPreset", "ChapterFilter"]
