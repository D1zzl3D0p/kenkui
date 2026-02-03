import sys
import importlib.resources
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Any

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

# --- CONSTANTS ---

DEFAULT_VOICES = [
    "alba",
    "marius",
    "javert",
    "jean",
    "fantine",
    "cosette",
    "eponine",
    "azelma",
]

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
    verbose: bool = False
    # TTS configuration attributes (voice field supports: built-in name, local path, or hf:// URL)
    tts_model: str = "kyutai/pocket-tts"
    tts_provider: str = "huggingface"
    model_name: str = "pocket-tts"
    temperature: float = 0.7
    eos_threshold: float = -4.0
    lsd_decode_steps: int = 1
    elevenlabs_key: str = ""
    elevenlabs_turbo: bool = False


@dataclass
class Chapter:
    index: int
    title: str
    paragraphs: list[str]


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
                f"[bold red]ACCESS DENIED: TERMS OF USE NOT ACCEPTED[/bold red]"
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

    except Exception as e:
        # Fail silently or log if needed, return empty list if path not found
        pass

    return sorted(custom_voices)


def print_available_voices(console: Console):
    """Prints a styled table of all available voices."""

    # --- CONSTANTS ---
    # Adjust this list to match the actual defaults provided by your underlying library
    DEFAULT_VOICES = [
        "alba",
        "marius",
        "javert",
        "jean",
        "fantine",
        "cosette",
        "eponine",
        "azelma",
    ]

    table = Table(
        title="Available Voices", show_header=True, header_style="bold magenta"
    )
    table.add_column("Type", style="dim", width=12)
    table.add_column("Voice Name", style="bold cyan")
    table.add_column("Description", style="white")

    # Add Built-in Defaults
    for voice in DEFAULT_VOICES:
        # Determine rough description based on prefix conventions (af=American Female, etc)
        desc = "Standard Voice"
        if voice.startswith("alba"):
            desc = "American Male"
        elif voice.startswith("marius"):
            desc = "American Male"
        elif voice.startswith("javert"):
            desc = "American Male"
        elif voice.startswith("jean"):
            desc = "American Male"
        elif voice.startswith("fantine"):
            desc = "British Female"
        elif voice.startswith("cosette"):
            desc = "American Female"
        elif voice.startswith("eponine"):
            desc = "British Female"
        elif voice.startswith("azelma"):
            desc = "American Female"

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
