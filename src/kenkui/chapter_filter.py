"""
Chapter filtering system with presets and custom filters.
Applied BEFORE interactive selection to provide sensible defaults.
"""

from dataclasses import dataclass
from typing import Callable, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from .helpers import Chapter
    from .chapter_classifier import ChapterTags


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

    # Built-in presets
    PRESETS = {
        "all": FilterPreset(
            name="All Chapters",
            description="Include all chapters including front/back matter",
            filter_fn=lambda t: True,
        ),
        "content-only": FilterPreset(
            name="Content Only",
            description="Exclude front matter, back matter, and title pages",
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

    @classmethod
    def get_preset(cls, name: str) -> Optional[FilterPreset]:
        """Get a preset by name."""
        return cls.PRESETS.get(name)

    @classmethod
    def apply_preset(
        cls, chapters: list["Chapter"], preset_name: str
    ) -> list["Chapter"]:
        """Apply a preset filter to chapters."""
        preset = cls.get_preset(preset_name)
        if not preset:
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
