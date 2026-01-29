from dataclasses import dataclass
from pathlib import Path
from typing import List, Any

from rich.console import Console
from rich.table import Table
from rich import box
from rich.prompt import Prompt

# Helper Classes


@dataclass
class Config:
    voice: str
    epub_path: Path
    output_path: Path
    pause_line_ms: int
    pause_chapter_ms: int
    workers: int
    m4b_bitrate: str
    keep_temp: bool
    debug_html: bool
    interactive_chapters: bool  # New flag


@dataclass
class Chapter:
    index: int
    title: str
    paragraphs: List[str]


@dataclass
class AudioResult:
    chapter_index: int
    title: str
    file_path: Path
    duration_ms: int


# --- HELPER: SELECTION LOGIC ---


def parse_range_string(selection_str: str, max_val: int) -> List[int]:
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


def interactive_select(
    items: List[Any], title: str, console: Console, item_formatter=str
) -> List[Any]:
    """Generic TUI selection menu."""
    if not items:
        return []

    # Display Table
    table = Table(
        title=title, show_header=True, header_style="bold magenta", box=box.SIMPLE
    )
    table.add_column("#", style="cyan", width=4, justify="right")
    table.add_column("Item", style="white")

    for i, item in enumerate(items, 1):
        table.add_row(str(i), item_formatter(item))

    console.print(table)
    console.print("[dim]Enter numbers (e.g. '1,3,5-10') or press Enter for ALL[/dim]")

    while True:
        selection = Prompt.ask("Select")
        indices = parse_range_string(selection, len(items))

        if not indices:
            console.print(
                "[yellow]No items selected. Try again or type 'all'.[/yellow]"
            )
            continue

        selected_items = [items[i] for i in indices]
        console.print(f"[green]Selected {len(selected_items)} items.[/green]")
        return selected_items
