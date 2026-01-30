import sys
import importlib.resources
from dataclasses import dataclass
from pathlib import Path
from typing import List, Any

from rich.console import Console
from rich.table import Table
from rich import box
from rich.prompt import Prompt
from rich.panel import Panel

from huggingface_hub import login, hf_hub_download
from huggingface_hub.errors import GatedRepoError, RepositoryNotFoundError

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


# --- AUTH CHECKER (Run Once at Startup) ---


def check_huggingface_access(model_id: str = "kyutai/pocket-tts"):
    """
    Ensures the user is logged in and has accepted the ToS for the gated model.
    """
    console = Console()

    try:
        # Check if we can access the config without downloading the whole thing
        hf_hub_download(
            model_id,
            filename="config.json",
            force_download=False,
            local_files_only=False,
        )
        # If successful, return silently
        return

    except (GatedRepoError, RepositoryNotFoundError, Exception) as e:
        # If it failed, it might be auth or gate issues.
        console.rule("[bold red]Authentication Required")
        console.print(
            f"[yellow]The model '{model_id}' requires Hugging Face authentication.[/yellow]"
        )

        # 1. Attempt Login
        print("\nAttempting to log in...")
        login()  # This handles the token prompt securely

        # 2. Re-check for Gate Acceptance
        try:
            hf_hub_download(model_id, filename="config.json", force_download=False)
            console.print("[green]Authentication successful![/green]")
            return
        except GatedRepoError:
            console.print("\n" + "!" * 60, style="bold red")
            console.print(
                f"[bold red]ACCESS DENIED: TERMS OF USE NOT ACCEPTED[/bold red]"
            )
            console.print("!" * 60)
            console.print(
                f"\nThis model ({model_id}) requires you to agree to a license"
            )
            console.print("specifically regarding voice cloning and safety.")
            console.print(
                f"\n[blue underline]https://huggingface.co/{model_id}[/blue underline]\n"
            )
            console.print("1. Log in to the website.")
            console.print("2. Click 'Agree' on the model card.")
            console.print("3. Return here and press Enter.")

            input("\nPress Enter once you have accepted the terms...")

            # Final Check
            try:
                hf_hub_download(model_id, filename="config.json", force_download=False)
                console.print("[green]Success! Proceeding...[/green]")
            except Exception:
                console.print(
                    "[bold red]Still unable to access model. Exiting.[/bold red]"
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
    else:
        # Optional: Add a row indicating no custom voices were found
        pass

    console.print(table)
    console.print(
        Panel(
            "[dim]To use a voice, run:[/dim]\n[green]kenkui input.epub --voice [bold]voice_name[/bold][/green]",
            title="Usage Hint",
            expand=False,
        )
    )
