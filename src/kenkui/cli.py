"""
EPUB to Audiobook Converter
Batch Processing + Auto-Naming + Chapter Filtering
"""

import argparse
import logging
import multiprocessing
import os
import sys
import warnings
from pathlib import Path
from contextlib import contextmanager
from rich.console import Console

# Performance tuning (must be before other imports)
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

# Local imports  # noqa: E402
from . import __version__  # noqa: E402
from .chapter_filter import FilterOperation  # noqa: E402
from .parsing import AudioBuilder  # noqa: E402
from .helpers import (  # noqa: E402
    Config,
    print_available_voices,
    print_chapter_presets,
    select_books_interactive,
)
from .file_finder import find_epub_files  # noqa: E402
from .huggingface_auth import ensure_huggingface_access, is_custom_voice  # noqa: E402

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
    help_text = """[bold cyan]Kenkui - EPUB to Audiobook Converter[/bold cyan]

[bold]Usage:[/bold]
  kenkui [epub_file_or_directory] [options]

[bold]Quick Examples:[/bold]
  # Convert a single book with default voice
  kenkui book.epub

  # Use a specific voice
  kenkui book.epub -v AlbusDumbledore

  # Process all books in current directory (recursive search)
  kenkui

  # Process all books in a specific directory
  kenkui ~/books/

  # Include hidden directories in search
  kenkui ~/books/ --search-hidden-dirs

  # Filter chapters with regex patterns
  kenkui book.epub -i "Chapter.*" -e "Appendix"

  # Use chapter filter preset
  kenkui book.epub --chapter-preset content-only

  # Preview what would be converted
  kenkui book.epub --preview

[bold]Common Options:[/bold]
  -v, --voice NAME      TTS voice (default: alba)
  -o, --output DIR      Output directory
  --chapter-preset      Filter: none|all|content-only|chapters-only|with-parts
  -i, --include-chapter Include chapters matching regex (repeatable)
  -e, --exclude-chapter Exclude chapters matching regex (repeatable)
  --select-books        Interactively select books from directory (default)
  --no-select-books     Process all books without prompting
  --preview             Preview what would be converted
  --list-voices         Show available voices
  --list-chapter-presets  Show available chapter filter presets
  --log FILE            Log detailed output to file
  --verbose             Show detailed logs

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
        nargs="?",
        default=Path.cwd(),  # Default to current directory
        help="EPUB file or directory (default: current directory)",
    )
    parser.add_argument(
        "-v",
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
        choices=["none", "all", "content-only", "chapters-only", "with-parts"],
        action="append",
        dest="chapter_presets",
        help="Chapter filter preset (repeatable, later overrides earlier)",
    )
    chapter_filter_group.add_argument(
        "-i",
        "--include-chapter",
        action="append",
        dest="chapter_includes",
        help="Regex pattern to include chapters by title (repeatable)",
    )
    chapter_filter_group.add_argument(
        "-e",
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
        "--list-chapter-presets",
        action="store_true",
        help="Display all available chapter filter presets",
    )
    parser.add_argument(
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
        "--log",
        type=Path,
        default=None,
        help="Log file path for detailed output logging",
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {__version__}",
        help="Show version information and exit",
    )
    parser.add_argument(
        "--select-books",
        action="store_true",
        default=True,
        help="Interactively select books when processing a directory (default: True)",
    )
    parser.add_argument(
        "--no-select-books",
        action="store_false",
        dest="select_books",
        help="Process all books in directory without selection prompt",
    )
    parser.add_argument(
        "--search-hidden-dirs",
        action="store_true",
        default=False,
        help="Search hidden directories when looking for EPUB files (default: False)",
    )

    args = parser.parse_args()
    console = Console()

    # Setup logging if log file path provided
    if args.log:
        # Ensure log directory exists
        args.log.parent.mkdir(parents=True, exist_ok=True)

        # Configure logging
        log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        logging.basicConfig(
            level=logging.DEBUG if args.verbose else logging.INFO,
            format=log_format,
            handlers=[
                logging.FileHandler(args.log, mode="w"),
            ],
        )
        console.print(f"[dim]Logging to: {args.log}[/dim]")

    def build_chapter_operations(
        presets: list[str] | None,
        includes: list[str] | None,
        excludes: list[str] | None,
    ) -> list[FilterOperation]:
        """Build filter operations from CLI arguments."""
        operations: list[FilterOperation] = []

        # Track which arguments were provided
        has_preset = bool(presets)
        has_include = bool(includes)
        has_exclude = bool(excludes)

        # Default to content-only preset if no filters specified at all
        if not has_preset and not has_include and not has_exclude:
            return [FilterOperation("preset", "content-only")]

        # If includes are provided but no preset, default to the default preset first
        # This allows -i to add to the default selection instead of starting empty
        if has_include and not has_preset:
            operations.append(FilterOperation("preset", "content-only"))

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

    # Handle --list-chapter-presets
    if args.list_chapter_presets:
        print_chapter_presets(console)
        return

    # Handle case when input is explicitly None (shouldn't happen with default)
    if args.input is None:
        args.input = Path.cwd()

    # Validate input
    if not args.input.exists():
        error_msg = f"Path '{args.input}' does not exist."
        logging.error(error_msg)
        console.print(f"[red]Error: {error_msg}[/red]")
        sys.exit(1)

    logging.info(f"Processing input: {args.input}")

    # Build queue - uses fast recursive search for directories
    if args.input.is_dir():
        queue_files = sorted(find_epub_files(args.input, args.search_hidden_dirs))
        logging.info(f"Found {len(queue_files)} EPUB files in directory (recursive)")
    else:
        queue_files = [args.input]
        logging.info(f"Processing single file: {args.input}")

    if not queue_files:
        logging.error("No EPUB files found in input path")
        console.print("[red]No EPUB files found in input path.[/red]")
        sys.exit(1)

    # Interactive book selection for directories with multiple books
    is_multi_book = len(queue_files) > 1
    if is_multi_book and args.select_books:
        if args.preview:
            # In preview mode, show selector first
            queue_files = select_books_interactive(queue_files, console)
            if not queue_files:
                console.print("[yellow]No books selected. Exiting.[/yellow]")
                sys.exit(0)
        else:
            # In normal mode, show selector
            queue_files = select_books_interactive(queue_files, console)
            if not queue_files:
                console.print("[yellow]No books selected. Exiting.[/yellow]")
                sys.exit(0)

    console.print(f"[bold green]Queue:[/bold green] {len(queue_files)} books.")
    logging.info(f"Queue: {len(queue_files)} books to process")

    # Check if using custom voice and ensure HuggingFace access if needed
    voice = args.voice.strip() if args.voice else "alba"
    from .helpers import get_bundled_voices

    bundled_voices = get_bundled_voices()

    if is_custom_voice(voice, bundled_voices):
        logging.info("Custom voice detected, checking HuggingFace access")
        if not ensure_huggingface_access(console=console):
            console.print(
                "[yellow]Custom voice authentication failed or was cancelled.[/yellow]"
            )
            console.print("[yellow]Falling back to built-in voice 'alba'.[/yellow]")
            voice = "alba"
    logging.info("Voice access check completed")

    # Create base output directory for multi-book processing
    base_output_path = args.output
    if is_multi_book and base_output_path is None:
        # Default to input directory if not specified
        base_output_path = args.input if args.input.is_dir() else args.input.parent

    # Process Queue
    for idx, epub_file in enumerate(queue_files, 1):
        console.rule(f"[bold magenta]Processing Book {idx}/{len(queue_files)}")
        logging.info(f"Processing book {idx}/{len(queue_files)}: {epub_file}")

        # Validate voice parameter (voice already set earlier, just validate here)
        if not voice or voice.lower() == "voice":
            error_msg = "Invalid voice name"
            logging.error(error_msg)
            console.print(
                f"[red]Error: {error_msg}. Please specify a valid voice using --voice[/red]"
            )
            console.print("[dim]Use --list-voices to see available options[/dim]")
            sys.exit(1)

        logging.info(f"Using voice: {voice}")

        # Determine output path - file will be placed next to the source EPUB
        book_output_path = args.output

        cfg = Config(
            voice=voice,
            epub_path=epub_file,
            output_path=book_output_path,
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
            logging.info(f"Successfully processed: {epub_file}")
        except KeyboardInterrupt:
            logging.warning("Batch cancelled by user")
            console.print("\n[bold red]Batch Cancelled.[/bold red]")
            sys.exit(130)
        except Exception as e:
            # Sanitize error message to prevent Rich markup errors
            error_msg = str(e)
            error_msg = error_msg.encode("utf-8", errors="replace").decode("utf-8")
            # Escape Rich markup characters to prevent parsing errors
            error_msg_safe = error_msg.replace("[", "\\[").replace("]", "\\]")
            logging.error(
                f"Error processing {epub_file}: {error_msg}", exc_info=args.debug
            )
            console.print(
                f"[red]Error processing {epub_file.name}: {error_msg_safe}[/red]"
            )
            if args.debug:
                import traceback

                console.print(traceback.format_exc())


if __name__ == "__main__":
    main()
