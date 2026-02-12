import logging
import re
import sys
import importlib.resources
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable
from xml.etree import ElementTree as ET

from rich.console import Console
from rich.table import Table
from rich import box
from rich.prompt import Prompt
from rich.panel import Panel
from rich.markup import escape

from huggingface_hub import HfApi, login
from huggingface_hub.errors import (
    GatedRepoError,
    RepositoryNotFoundError,
)

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


def quick_count_chapters(epub_path: Path) -> int | None:
    """Quickly count chapters in an EPUB by parsing TOC without full extraction.

    This lightweight version:
    1. Opens the EPUB as a ZIP file
    2. Finds the TOC file (NCX or NAV)
    3. Counts TOC entries
    4. Falls back to counting HTML/XHTML files if no TOC found

    Much faster than full chapter extraction since it avoids:
    - BeautifulSoup HTML parsing
    - Full text extraction
    - Chapter object creation

    Returns None if counting fails.
    """
    logger.debug(f"Counting chapters in {epub_path}")

    try:
        with zipfile.ZipFile(epub_path, "r") as epub_zip:
            namelist = epub_zip.namelist()

            # Find the container.xml to locate the OPF file
            container_path = "META-INF/container.xml"
            if container_path not in namelist:
                return _count_html_files(namelist)

            try:
                container_xml = epub_zip.read(container_path)
                container_tree = ET.fromstring(container_xml)

                ns = {"container": "urn:oasis:names:tc:opendocument:xmlns:container"}
                rootfile = container_tree.find(".//container:rootfile", ns)

                if rootfile is None:
                    return _count_html_files(namelist)

                opf_path = rootfile.get("full-path")
                if opf_path is None or opf_path not in namelist:
                    return _count_html_files(namelist)

                # Read OPF to find TOC reference
                opf_xml = epub_zip.read(opf_path)
                opf_tree = ET.fromstring(opf_xml)
                opf_ns = {"opf": "http://www.idpf.org/2007/opf"}

                # Look for NCX file (EPUB2)
                ncx_item = opf_tree.find(
                    ".//opf:item[@media-type='application/x-dtbncx+xml']", opf_ns
                )

                if ncx_item is not None:
                    ncx_href = ncx_item.get("href")
                    if ncx_href:
                        opf_dir = str(Path(opf_path).parent)
                        if opf_dir == ".":
                            ncx_path = ncx_href
                        else:
                            ncx_path = str(Path(opf_dir) / ncx_href)

                        if ncx_path in namelist:
                            count = _count_ncx_chapters(epub_zip, ncx_path)
                            if count > 0:
                                logger.debug(
                                    f"Found {count} chapters via NCX in {epub_path}"
                                )
                                return count

                # Look for NAV file (EPUB3)
                nav_item = opf_tree.find(".//opf:item[@properties='nav']", opf_ns)

                if nav_item is not None:
                    nav_href = nav_item.get("href")
                    if nav_href:
                        opf_dir = str(Path(opf_path).parent)
                        if opf_dir == ".":
                            nav_path = nav_href
                        else:
                            nav_path = str(Path(opf_dir) / nav_href)

                        if nav_path in namelist:
                            count = _count_nav_chapters(epub_zip, nav_path)
                            if count > 0:
                                logger.debug(
                                    f"Found {count} chapters via NAV in {epub_path}"
                                )
                                return count

                # Fallback: count HTML files
                return _count_html_files(namelist)

            except ET.ParseError:
                return _count_html_files(namelist)

    except (zipfile.BadZipFile, OSError, PermissionError) as e:
        logger.warning(f"Failed to read EPUB {epub_path}: {e}")
        return None


def _count_ncx_chapters(epub_zip: zipfile.ZipFile, ncx_path: str) -> int:
    """Count chapters in NCX TOC file."""
    try:
        ncx_content = epub_zip.read(ncx_path).decode("utf-8", errors="ignore")
        ncx_tree = ET.fromstring(ncx_content)

        ns = {"ncx": "http://www.daisy.org/z3986/2005/ncx/"}
        navpoints = ncx_tree.findall(".//ncx:navPoint", ns)

        return len(navpoints)
    except Exception:
        return 0


def _count_nav_chapters(epub_zip: zipfile.ZipFile, nav_path: str) -> int:
    """Count chapters in EPUB3 NAV file."""
    try:
        nav_content = epub_zip.read(nav_path).decode("utf-8", errors="ignore")
        nav_tree = ET.fromstring(nav_content)

        ns = {"xhtml": "http://www.w3.org/1999/xhtml"}

        # Look for toc nav element
        toc_nav = nav_tree.find(".//xhtml:nav[@epub:type='toc']", ns)
        if toc_nav is None:
            toc_nav = nav_tree.find(".//nav[@epub:type='toc']")

        if toc_nav is not None:
            # Count links in the toc
            links = toc_nav.findall(".//xhtml:a", ns)
            if not links:
                links = toc_nav.findall(".//a")
            return len(links)

        return 0
    except Exception:
        return 0


def _count_html_files(namelist: list[str]) -> int:
    """Fallback: count HTML/XHTML files as approximate chapter count."""
    count = 0
    for name in namelist:
        if name.lower().endswith((".html", ".htm", ".xhtml")):
            # Skip common non-chapter files
            lower_name = name.lower()
            if any(skip in lower_name for skip in ["toc", "nav", "cover", "copyright"]):
                continue
            count += 1
    return count


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
    epub_path: Path
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


# --- AUTH CHECKER (Run Once at Startup) ---


def check_huggingface_access(model_id: str = "kyutai/pocket-tts"):
    """
    Ensures the user is logged in and has accepted the ToS for the gated model.
    """
    console = Console()
    api = HfApi()

    try:
        # Check repo access without downloading model files
        api.model_info(model_id)
        # If successful, return silently
        return
    except GatedRepoError:
        # Handle gated repository - requires authentication and terms acceptance
        console.rule("[bold red]Authentication Required")
        console.print(
            f"[yellow]The model '{model_id}' requires Hugging Face authentication.[/yellow]"
        )

        # 1. Attempt Login
        print("\nAttempting to log in...")
        login()  # This handles the token prompt securely

        # 2. Re-check for Gate Acceptance
        try:
            api.model_info(model_id)
            console.print("[green]Authentication successful![/green]")
            return
        except GatedRepoError:
            console.print("\n" + "!" * 60, style="bold red")
            console.print(
                "[bold red]ACCESS DENIED: TERMS OF USE NOT ACCEPTED[/bold red]"
            )
            console.print(
                f"The model '{model_id}' is gated and requires you to accept the terms on Hugging Face."
            )
            console.print(
                f"Please visit [blue underline]https://huggingface.co/{model_id}[/blue underline]"
            )
            console.print("Read the license and click 'Agree and access repository'.")
            console.print("After accepting, run this command again.\n")
            console.print("!" * 60, style="bold red")
            sys.exit(1)
    except RepositoryNotFoundError:
        # Handle repository not found (not gated)
        console.rule("[bold red]Model Not Found")
        console.print(
            f"[yellow]The model '{model_id}' could not be found on Hugging Face.[/yellow]"
        )
        console.print(
            "Check the model ID or update the project to use an available model."
        )
        console.print(
            f"[blue underline]https://huggingface.co/{model_id}[/blue underline]"
        )
        sys.exit(1)


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
    "check_huggingface_access",
    "get_bundled_voices",
    "print_available_voices",
    "print_chapter_presets",
    "select_books_interactive",
    "quick_count_chapters",
]
