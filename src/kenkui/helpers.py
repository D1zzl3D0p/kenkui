from __future__ import annotations

import logging
import re
import importlib.resources
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from rich.console import Console
from rich.table import Table
from rich import box
from rich.prompt import Prompt
from rich.panel import Panel
from rich.markup import escape

from .chapter_classifier import ChapterTags
from .chapter_filter import FilterOperation
from .utils import DEFAULT_VOICES, VOICE_DESCRIPTIONS


# Module-level logger
logger = logging.getLogger(__name__)


def _natural_sort_key(text: str) -> list:
    """Generate a sort key for natural (human-friendly) sorting.

    Splits text into alternating non-numeric and numeric parts.
    Non-numeric parts are lowercased for case-insensitive sorting.
    Numeric parts are converted to integers for proper numeric comparison.

    Example: "Chapter 10" -> ['chapter ', 10]
             "Chapter 2" -> ['chapter ', 2]

    This ensures "Chapter 2" comes before "Chapter 10".
    """
    parts = []
    for part in re.split(r"(\d+)", text.lower()):
        if part.isdigit():
            parts.append(int(part))
        else:
            parts.append(part)
    return parts


def _count_chapters_with_reader(book_path: Path) -> int | None:
    """Count chapters using the reader interface (supports all formats)."""
    from .readers import get_reader

    try:
        reader = get_reader(book_path, verbose=False)
        return reader.count_chapters()
    except Exception:
        return None


def quick_count_chapters(ebook_path: Path) -> int | None:
    """Quickly count chapters in an ebook by parsing TOC without full extraction.

    This lightweight version uses the reader interface to count TOC entries
    without extracting full text content.

    Supports all ebook formats: EPUB, MOBI, AZW, AZW3, AZW4

    Returns None if counting fails.
    """
    logger.debug(f"Counting chapters in {ebook_path}")
    return _count_chapters_with_reader(ebook_path)


def select_books_interactive(
    book_files: list[Path],
    console: Console,
) -> list[Path]:
    """Interactive book selection when processing a directory.

    Displays a table of books with chapter counts (sorted alphabetically)
    and lets user select which ones to process using comma-separated numbers
    or ranges.

    Returns filtered list of selected book files.
    """
    logger.info(f"Starting interactive book selection for {len(book_files)} books")

    if len(book_files) <= 1:
        logger.debug("Only one book, skipping selection")
        return book_files

    # Count chapters for each book
    console.print("[dim]Scanning books for chapter counts...[/dim]")
    book_info: list[tuple[Path, int | None]] = []
    for book_path in book_files:
        chapter_count = quick_count_chapters(book_path)
        book_info.append((book_path, chapter_count))

    # Sort books by filename using natural sort (case-insensitive)
    book_info.sort(key=lambda x: _natural_sort_key(x[0].name))

    # Create selection table
    table = Table(
        title="[bold magenta]Book Selection[/bold magenta]",
        show_header=True,
        header_style="bold magenta",
        box=box.ROUNDED,
        expand=True,
    )
    table.add_column("#", style="cyan", width=6, justify="center")
    table.add_column("Filename", style="white")
    table.add_column("Chapters", style="green", width=12, justify="center")

    for i, (book_path, chapter_count) in enumerate(book_info, 1):
        chapter_text = str(chapter_count) if chapter_count is not None else "?"
        table.add_row(f"{i}", escape(book_path.name), chapter_text)

    console.print()
    console.print(table)
    console.print()
    console.print(
        Panel(
            "[dim]Enter book numbers to process (e.g., '1,3,5-10')[/dim]\n"
            "[dim]Press Enter to select ALL books[/dim]",
            title="Selection Options",
            border_style="blue",
        )
    )

    while True:
        selection = Prompt.ask("[bold cyan]Select books[/bold cyan]")

        if not selection.strip():
            # Empty selection = all books
            return book_files

        indices = parse_range_string(selection, len(book_info))

        if not indices:
            console.print(
                "[yellow]No books selected. Try again or press Enter for all.[/yellow]"
            )
            continue

        selected_books = [book_info[i][0] for i in indices]

        # Show selected summary
        summary_table = Table(
            title=f"[green]Selected {len(selected_books)} of {len(book_info)} books[/green]",
            show_header=True,
            header_style="bold green",
            box=box.SIMPLE,
        )
        summary_table.add_column("#", style="cyan", width=6)
        summary_table.add_column("Filename", style="white")
        summary_table.add_column("Chapters", style="green", width=12)

        for i, book_path in enumerate(selected_books[:10], 1):
            chapter_count = book_info[book_files.index(book_path)][1]
            chapter_text = str(chapter_count) if chapter_count is not None else "?"
            summary_table.add_row(f"{i}", escape(book_path.name), chapter_text)

        if len(selected_books) > 10:
            summary_table.add_row(
                "...", f"[dim]and {len(selected_books) - 10} more...[/dim]", ""
            )

        console.print()
        console.print(summary_table)
        console.print()

        logger.info(
            f"User selected {len(selected_books)} books: {[b.name for b in selected_books]}"
        )
        return selected_books


# Helper Classes


@dataclass
class Config:
    voice: str
    ebook_path: Path  # Supports any ebook format (epub, mobi, azw, etc.)
    output_path: Path
    pause_line_ms: int
    pause_chapter_ms: int
    workers: int
    m4b_bitrate: str
    keep_temp: bool
    debug_html: bool
    chapter_filters: list[FilterOperation]  # Chapter filtering operations
    preview: bool = False  # Preview mode (show what would be converted)
    verbose: bool = False
    # TTS configuration attributes (voice field supports: built-in name, local path, or hf:// URL)
    tts_model: str = "kyutai/pocket-tts"
    tts_provider: str = "huggingface"
    model_name: str = "pocket-tts"
    elevenlabs_key: str = ""
    elevenlabs_turbo: bool = False

    # Backwards compatibility - provide epub_path as an alias
    @property
    def epub_path(self) -> Path:
        """Backwards compatibility alias for ebook_path."""
        return self.ebook_path

    @epub_path.setter
    def epub_path(self, value: Path):
        """Backwards compatibility setter for ebook_path."""
        self.ebook_path = value


@dataclass
class Chapter:
    index: int
    title: str
    paragraphs: list[str]
    tags: ChapterTags = field(default_factory=lambda: ChapterTags(is_chapter=True))
    toc_index: int = 0


@dataclass
class AudioResult:
    chapter_index: int
    title: str
    file_path: Path
    duration_ms: int


# --- HELPER: SELECTION LOGIC ---


def parse_range_string(selection_str: str, max_val: int) -> list[int]:
    """Parses '1, 2, 4-6' into [0, 1, 3, 4, 5]. Returns 0-based indices."""
    selection_str = selection_str.strip()
    if not selection_str or selection_str.lower() == "all":
        return list(range(max_val))

    selected_indices = set()
    parts = selection_str.split(",")

    for part in parts:
        part = part.strip()
        if "-" in part:
            try:
                start, end = map(int, part.split("-"))
                # Handle standard human input (1-based)
                start = max(1, start)
                end = min(max_val, end)
                if start <= end:
                    for i in range(start, end + 1):
                        selected_indices.add(i - 1)
            except ValueError:
                continue
        else:
            try:
                val = int(part)
                if 1 <= val <= max_val:
                    selected_indices.add(val - 1)
            except ValueError:
                continue

    return sorted(list(selected_indices))


def interactive_select[T](
    items: list[T],
    title: str,
    console: Console,
    item_formatter: Callable[[T], str] = str,
) -> list[T]:
    """Generic TUI selection menu using Rich components."""
    if not items:
        return []

    # Create styled table matching --list-voices format
    table = Table(
        title=f"[bold magenta]{title}[/bold magenta]",
        show_header=True,
        header_style="bold magenta",
        box=box.ROUNDED,
        expand=True,
    )
    table.add_column("#", style="cyan", width=6, justify="center")
    table.add_column("Item", style="white", overflow="fold")

    for i, item in enumerate(items, 1):
        # Escape Rich markup to prevent parsing errors from special characters
        item_text = escape(item_formatter(item))
        table.add_row(f"{i}", item_text)

    console.print()
    console.print(table)
    console.print()
    console.print(
        Panel(
            "[dim]Enter chapter numbers to convert (e.g., '1,3,5-10')[/dim]\n"
            "[dim]Press Enter to select ALL chapters[/dim]",
            title="Selection Options",
            border_style="blue",
        )
    )

    while True:
        selection = Prompt.ask("[bold cyan]Select chapters[/bold cyan]")
        indices = parse_range_string(selection, len(items))

        if not indices:
            console.print(
                "[yellow]No items selected. Try again or type 'all' to select all.[/yellow]"
            )
            continue

        selected_items = [items[i] for i in indices]

        # Show selected chapters summary
        summary_table = Table(
            title=f"[green]Selected {len(selected_items)} of {len(items)} chapters[/green]",
            show_header=True,
            header_style="bold green",
            box=box.SIMPLE,
        )
        summary_table.add_column("#", style="cyan", width=6)
        summary_table.add_column("Chapter Title", style="white")

        for i, item in enumerate(selected_items[:10], 1):  # Show first 10
            summary_table.add_row(f"{i}", escape(item_formatter(item)))

        if len(selected_items) > 10:
            summary_table.add_row(
                "...", f"[dim]and {len(selected_items) - 10} more[/dim]"
            )

        console.print()
        console.print(summary_table)
        console.print()

        return selected_items


def get_bundled_voices():
    """
    Scans the 'voices' directory inside the package for custom voice files.
    Returns a list of filenames.
    """
    custom_voices = []
    try:
        # Determine package name. If run directly, __package__ might be None.
        # We assume the package name 'kenkui' if installed, or we check local dir.
        pkg_name = __package__

        if pkg_name:
            # 1. INSTALLED MODE: Use importlib to find files inside the package
            # We assume 'voices' is a subdirectory in the same package
            # We need to target the specific sub-resource
            voices_path = importlib.resources.files(pkg_name) / "voices"
            if voices_path.is_dir():
                # Iterate over files
                for entry in voices_path.iterdir():
                    if entry.is_file() and not entry.name.startswith("__"):
                        custom_voices.append(entry.name)
        else:
            # 2. LOCAL DEV MODE: Fallback to filesystem check relative to this script
            local_voices_path = Path(__file__).parent / "voices"
            if local_voices_path.exists():
                custom_voices = [
                    f.name
                    for f in local_voices_path.iterdir()
                    if f.is_file() and not f.name.startswith("__")
                ]

    except Exception:
        # Fail silently or log if needed, return empty list if path not found
        pass

    return sorted(custom_voices)


def print_available_voices(console: Console):
    """Prints a styled table of all available voices."""
    table = Table(
        title="Available Voices", show_header=True, header_style="bold magenta"
    )
    table.add_column("Type", style="dim", width=12)
    table.add_column("Voice Name", style="bold cyan")
    table.add_column("Description", style="white")

    # Add Built-in Defaults
    for voice in DEFAULT_VOICES:
        desc = VOICE_DESCRIPTIONS.get(voice, "Standard Voice")
        table.add_row("Built-in", voice, desc)

    # Add Custom Voices found in package
    custom_files = get_bundled_voices()
    if custom_files:
        table.add_section()
        for filename in custom_files:
            # Strip extension for display if you want, or keep it to be precise
            clean_name = Path(filename).stem
            table.add_row("Custom/Local", clean_name, f"File: {filename}")

    # Add Remote Voices section (hf:// URLs)
    table.add_section()
    table.add_row(
        "Remote/HF",
        "hf://user/repo/voice.wav",
        "HuggingFace Hub voice file",
    )
    table.add_row(
        "Remote/HF",
        "/path/to/local.wav",
        "Local voice file path",
    )

    console.print(table)
    console.print(
        Panel(
            "[dim]To use a voice, run:[/dim]\n"
            "[green]kenkui input.epub --voice [bold]voice_name[/bold][/green]\n"
            "[green]kenkui input.epub --voice [bold]hf://user/repo/voice.wav[/bold][/green]\n"
            "[green]kenkui input.epub --voice [bold]/path/to/custom.wav[/bold][/green]",
            title="Usage Hint",
            expand=False,
        )
    )


def print_chapter_presets(console: Console):
    """Prints a styled table of all available chapter filter presets."""
    from .chapter_filter import ChapterFilter

    table = Table(
        title="Chapter Filter Presets", show_header=True, header_style="bold magenta"
    )
    table.add_column("Preset Name", style="bold cyan", width=18)
    table.add_column("Description", style="white")

    for name, preset in ChapterFilter.PRESETS.items():
        # Mark the default preset
        display_name = (
            f"{name} [green](default)[/green]"
            if name == ChapterFilter.DEFAULT_PRESET
            else name
        )
        table.add_row(display_name, preset.description)

    console.print(table)
    console.print(
        Panel(
            "[dim]To use a preset, run:[/dim]\n"
            "[green]kenkui input.epub --chapter-preset [bold]preset_name[/bold][/green]\n"
            "[green]kenkui input.epub --chapter-preset [bold]content-only[/bold][/green]",
            title="Usage Hint",
            expand=False,
        )
    )


__all__ = [
    "Config",
    "Chapter",
    "AudioResult",
    "parse_range_string",
    "interactive_select",
    "get_bundled_voices",
    "print_available_voices",
    "print_chapter_presets",
    "select_books_interactive",
    "quick_count_chapters",
]
