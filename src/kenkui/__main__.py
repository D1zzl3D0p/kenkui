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
from pathlib import Path
from rich.console import Console

# Local imports
# Note: When installed as a package, these imports work relative to the package
from .parsing import AudioBuilder
from .helpers import (
    Config,
    check_huggingface_access,
    interactive_select,
    print_available_voices,
)

warnings.filterwarnings("ignore")


def main():
    parser = argparse.ArgumentParser(
        description="Batch EPUB to Audiobook Converter",
        formatter_class=argparse.RawTextHelpFormatter,
        epilog=(
            "Examples:\n"
            "  kenkui book.epub\n"
            "  kenkui library/ --select-books --voice alba\n"
            "  kenkui book.epub -o output/ -j 4\n"
            "\n"
            "Notes:\n"
            "  - Use --list-voices to see available voices.\n"
            "  - ffmpeg is required and must be on PATH.\n"
            "  - --select-books only applies when input is a directory.\n"
            "  - --select-chapters runs for each processed book."
        ),
    )

    # Arguments
    parser.add_argument(
        "input",
        nargs="?",
        type=Path,
        default=None,
        help="Input EPUB file or directory containing EPUBs",
    )
    parser.add_argument(
        "--voice",
        default="alba",
        help="TTS voice name (see --list-voices)",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help=(
            "Output directory or full output file path (default: input file directory)"
        ),
    )
    parser.add_argument(
        "-j",
        "--workers",
        type=int,
        default=os.cpu_count(),
        help="Parallel workers (default: CPU count)",
    )
    parser.add_argument(
        "--keep",
        action="store_true",
        help="Keep temporary files after conversion",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Print detailed errors and debug output",
    )

    # Selection Flags
    parser.add_argument(
        "--select-books",
        action="store_true",
        help="Pick books interactively when input is a directory",
    )
    parser.add_argument(
        "--select-chapters",
        action="store_true",
        help="Pick chapters interactively for each selected book",
    )

    # New Flag: List Voices
    parser.add_argument(
        "--list-voices",
        action="store_true",
        help="List built-in and custom voice names",
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

    check_huggingface_access()

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
