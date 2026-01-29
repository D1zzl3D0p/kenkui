"""
EPUB to Audiobook Converter
Batch Processing + Auto-Naming + Interactive Selection + TUI
"""

import os

# Performance tuning
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import argparse
import shutil
import sys
import warnings
import importlib.resources
from pathlib import Path
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

# Local imports
# Note: When installed as a package, these imports work relative to the package
from parsing import AudioBuilder
from helpers import Config, interactive_select

warnings.filterwarnings("ignore")

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


def main():
    parser = argparse.ArgumentParser(description="Batch EPUB to Audiobook Converter")

    # Arguments
    parser.add_argument(
        "input",
        nargs="?",
        type=Path,
        default=None,
        help="Input file or Directory containing EPUBs",
    )
    parser.add_argument("--voice", default="alba", help="Voice to use for TTS")
    parser.add_argument(
        "-o", "--output", type=Path, default=None, help="Output folder (optional)"
    )
    parser.add_argument("-j", "--workers", type=int, default=os.cpu_count())
    parser.add_argument("--keep", action="store_true")
    parser.add_argument("--debug", action="store_true")

    # Selection Flags
    parser.add_argument(
        "--select-books",
        action="store_true",
        help="Interactively select which books to process from directory",
    )
    parser.add_argument(
        "--select-chapters",
        action="store_true",
        help="Interactively select chapters for each book",
    )

    # New Flag: List Voices
    parser.add_argument(
        "--list-voices",
        action="store_true",
        help="List all available built-in and custom voices",
    )

    args = parser.parse_args()
    console = Console()

    # --- 0. Handle Voice Listing ---
    if args.list_voices:
        print_available_voices(console)
        sys.exit(0)

    # --- Validation: Input is required if not listing voices ---
    if not args.input:
        parser.print_help()
        console.print(
            "\n[red]Error: input argument is required unless listing voices.[/red]"
        )
        sys.exit(1)

    if not shutil.which("ffmpeg"):
        console.print("[red]Error: ffmpeg not found.[/red]")
        sys.exit(1)

    # 1. Build Queue
    queue_files = []

    if args.input.is_file():
        if args.input.suffix.lower() == ".epub":
            queue_files.append(args.input)
    elif args.input.is_dir():
        console.print(f"[blue]Scanning directory: {args.input}[/blue]")
        queue_files = sorted(list(args.input.rglob("*.epub")))

    if not queue_files:
        console.print("[red]No EPUB files found![/red]")
        sys.exit(1)

    # 2. Interactive Book Selection
    if args.select_books and len(queue_files) > 1:
        queue_files = interactive_select(
            queue_files, "Detected Books", console, lambda f: f.name
        )
        if not queue_files:
            sys.exit(0)

    console.print(f"[bold green]Queue:[/bold green] {len(queue_files)} books.")

    # 3. Process Queue
    for idx, epub_file in enumerate(queue_files, 1):
        console.rule(f"[bold magenta]Processing Book {idx}/{len(queue_files)}")

        cfg = Config(
            voice=args.voice,
            epub_path=epub_file,
            output_path=args.output,
            pause_line_ms=400,
            pause_chapter_ms=2000,
            workers=args.workers,
            m4b_bitrate="64k",
            keep_temp=args.keep,
            debug_html=args.debug,
            interactive_chapters=args.select_chapters,
        )

        builder = AudioBuilder(cfg)
        try:
            builder.run()
        except KeyboardInterrupt:
            console.print("\n[bold red]Batch Cancelled.[/bold red]")
            sys.exit(130)
        except Exception as e:
            console.print(f"[red]Error processing {epub_file.name}: {e}[/red]")
            # Optional: Print traceback if debug is on
            if args.debug:
                import traceback

                traceback.print_exc()


if __name__ == "__main__":
    main()
