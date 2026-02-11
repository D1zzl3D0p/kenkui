"""
EPUB to Audiobook Converter
Batch Processing + Auto-Naming + Chapter Filtering
"""

import os

# Performance tuning
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import argparse
import multiprocessing
import shutil
import sys
import warnings
from pathlib import Path
from contextlib import contextmanager
from rich.console import Console

# Local imports
from . import __version__
from .chapter_filter import FilterOperation
from .parsing import AudioBuilder
from .helpers import (
    Config,
    check_huggingface_access,
    print_available_voices,
)

warnings.filterwarnings("ignore")


@contextmanager
def suppress_c_stderr():
    """Context manager to suppress C-level stderr output (e.g., from lxml/libxml2)."""
    # Save original stderr file descriptor
    original_stderr_fd = os.dup(2)
    try:
        # Redirect stderr to /dev/null at OS level
        with open(os.devnull, "w") as devnull:
            os.dup2(devnull.fileno(), 2)
        yield
    finally:
        # Restore original stderr
        os.dup2(original_stderr_fd, 2)
        os.close(original_stderr_fd)


def print_abbreviated_help():
    """Print abbreviated help with examples when no arguments provided."""
    help_text = """
[bold cyan]Kenkui - EPUB to Audiobook Converter[/bold cyan]

[bold]Usage:[/bold]
  kenkui <epub_file_or_directory> [options]

    [bold]Quick Examples:[/bold]
  # Convert a single book with default voice
  kenkui book.epub

  # Use a specific voice
  kenkui book.epub --voice AlbusDumbledore

  # Process all books in a directory
  kenkui ~/books/

  # Filter chapters with regex patterns
  kenkui book.epub --include-chapter "Chapter.*" --exclude-chapter "Appendix"

  # Use chapter filter preset
  kenkui book.epub --chapter-preset content-only

  # Preview what would be converted
  kenkui book.epub --preview

[bold]Common Options:[/bold]
  --voice NAME          TTS voice (default: alba)
  -o, --output DIR      Output directory
  --chapter-preset      Filter: all|content-only|chapters-only|with-parts
  --include-chapter     Include chapters matching regex (repeatable)
  --exclude-chapter     Exclude chapters matching regex (repeatable)
  --preview             Preview what would be converted
  --list-voices         Show available voices
  -v, --verbose         Show detailed logs

[dim]Use --help for full options[/dim]
"""
    console = Console()
    console.print(help_text)


def main():
    parser = argparse.ArgumentParser(
        description="Convert EPUB files to audiobooks with custom voice support."
    )
    parser.add_argument(
        "input",
        type=Path,
        nargs="?",  # Make input optional for --fix-audiobook mode
        default=None,
        help="EPUB file or directory",
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
        help="Output directory (default: same as input)",
    )
    parser.add_argument(
        "-w",
        "--workers",
        type=int,
        default=max(1, multiprocessing.cpu_count() - 1),
        help="Number of parallel workers (default: number of CPU cores - 1)",
    )
    parser.add_argument(
        "--fix-audiobook",
        nargs=2,  # Require 2 arguments for --fix-audiobook
        metavar=("EBOOK", "AUDIOBOOK"),
        help="Fix audiobook metadata and missing chapters (requires: ebook_path audiobook_path)",
    )

    # Chapter filtering arguments
    chapter_filter_group = parser.add_argument_group("chapter filtering")
    chapter_filter_group.add_argument(
        "--chapter-preset",
        choices=["all", "content-only", "chapters-only", "with-parts"],
        action="append",
        dest="chapter_presets",
        help="Chapter filter preset (repeatable, later overrides earlier)",
    )
    chapter_filter_group.add_argument(
        "-I",
        "--include-chapter",
        action="append",
        dest="chapter_includes",
        help="Regex pattern to include chapters by title (repeatable)",
    )
    chapter_filter_group.add_argument(
        "-X",
        "--exclude-chapter",
        action="append",
        dest="chapter_excludes",
        help="Regex pattern to exclude chapters by title (repeatable)",
    )

    parser.add_argument(
        "--preview",
        action="store_true",
        help="Preview what would be converted without processing",
    )
    parser.add_argument(
        "--list-voices",
        action="store_true",
        help="Display all available voices",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Show all worker logs and TTS output (not sent to /dev/null)",
    )
    parser.add_argument(
        "-d",
        "--debug",
        action="store_true",
        help="Show full exception tracebacks",
    )
    parser.add_argument(
        "--keep",
        action="store_true",
        help="Keep temporary files (for debugging)",
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {__version__}",
        help="Show version information and exit",
    )

    args = parser.parse_args()
    console = Console()

    def build_chapter_operations(
        presets: list[str] | None,
        includes: list[str] | None,
        excludes: list[str] | None,
    ) -> list[FilterOperation]:
        """Build filter operations from CLI arguments."""
        operations: list[FilterOperation] = []

        # Track which arguments were provided to apply default if needed
        has_any_filter = bool(presets or includes or excludes)

        if not has_any_filter:
            # Default to content-only preset if no filters specified
            return [FilterOperation("preset", "content-only")]

        # Build operations in the order: presets, includes, excludes
        # (presets should generally come first to establish base selection)
        if presets:
            for preset in presets:
                operations.append(FilterOperation("preset", preset))

        if includes:
            for pattern in includes:
                operations.append(FilterOperation("include", pattern))

        if excludes:
            for pattern in excludes:
                operations.append(FilterOperation("exclude", pattern))

        return operations

    # Handle --fix-audiobook
    if args.fix_audiobook:
        ebook_path = Path(args.fix_audiobook[0])
        audiobook_path = Path(args.fix_audiobook[1])

        if not ebook_path.exists():
            console.print(f"[red]Error: EPUB file not found: {ebook_path}[/red]")
            sys.exit(1)

        if not audiobook_path.exists():
            console.print(
                f"[red]Error: Audiobook file not found: {audiobook_path}[/red]"
            )
            sys.exit(1)

        # Import fix_audiobook module
        from .fix_audiobook import fix_audiobook

        cfg = Config(
            voice=args.voice,
            epub_path=ebook_path,
            output_path=audiobook_path.parent,
            pause_line_ms=400,
            pause_chapter_ms=2000,
            workers=args.workers,
            m4b_bitrate="64k",
            keep_temp=args.keep,
            debug_html=args.debug,
            chapter_filters=build_chapter_operations(
                args.chapter_presets,
                args.chapter_includes,
                args.chapter_excludes,
            ),
            preview=args.preview,
            verbose=args.verbose,
        )

        success = fix_audiobook(ebook_path, audiobook_path, cfg, console)
        sys.exit(0 if success else 1)

    # Handle --list-voices
    if args.list_voices:
        print_available_voices(console)
        return

    # Show abbreviated help if no input provided
    if args.input is None:
        print_abbreviated_help()
        return

    # Validate input
    if not args.input.exists():
        console.print(f"[red]Error: Path '{args.input}' does not exist.[/red]")
        sys.exit(1)

    # Build queue
    if args.input.is_dir():
        queue_files = sorted(args.input.glob("*.epub"))
    else:
        queue_files = [args.input]

    if not queue_files:
        console.print("[red]No EPUB files found in input path.[/red]")
        sys.exit(1)

    console.print(f"[bold green]Queue:[/bold green] {len(queue_files)} books.")

    check_huggingface_access()

    # Process Queue
    for idx, epub_file in enumerate(queue_files, 1):
        console.rule(f"[bold magenta]Processing Book {idx}/{len(queue_files)}")

        # Validate voice parameter
        voice = args.voice.strip() if args.voice else "alba"
        if not voice or voice.lower() == "voice":
            console.print(
                "[red]Error: Invalid voice name. Please specify a valid voice using --voice[/red]"
            )
            console.print("[dim]Use --list-voices to see available options[/dim]")
            sys.exit(1)

        cfg = Config(
            voice=voice,
            epub_path=epub_file,
            output_path=args.output,
            pause_line_ms=400,
            pause_chapter_ms=2000,
            workers=args.workers,
            m4b_bitrate="64k",
            keep_temp=args.keep,
            debug_html=args.debug,
            chapter_filters=build_chapter_operations(
                args.chapter_presets,
                args.chapter_includes,
                args.chapter_excludes,
            ),
            preview=args.preview,
            verbose=args.verbose,
        )

        builder = AudioBuilder(cfg)
        try:
            # Suppress C-level stderr unless in verbose mode
            if not args.verbose:
                with suppress_c_stderr():
                    builder.run()
            else:
                builder.run()
        except KeyboardInterrupt:
            console.print("\n[bold red]Batch Cancelled.[/bold red]")
            sys.exit(130)
        except Exception as e:
            # Sanitize error message to prevent Rich markup errors
            error_msg = str(e)
            error_msg = error_msg.encode("utf-8", errors="replace").decode("utf-8")
            # Escape Rich markup characters to prevent parsing errors
            error_msg = error_msg.replace("[", "\\[").replace("]", "\\]")
            console.print(f"[red]Error processing {epub_file.name}: {error_msg}[/red]")
            if args.debug:
                import traceback

                console.print(traceback.format_exc())


if __name__ == "__main__":
    main()
