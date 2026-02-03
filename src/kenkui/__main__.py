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
from contextlib import contextmanager
from rich.console import Console

# Local imports
from .parsing import AudioBuilder
from .helpers import (
    Config,
    check_huggingface_access,
    interactive_select,
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
        default=4,
        help="Number of parallel workers",
    )
    parser.add_argument(
        "--fix-audiobook",
        nargs=2,  # Require 2 arguments for --fix-audiobook
        metavar=("EBOOK", "AUDIOBOOK"),
        help="Fix audiobook metadata and missing chapters (requires: ebook_path audiobook_path)",
    )
    parser.add_argument(
        "--select-chapters",
        action="store_true",
        help="Pick chapters interactively",
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
        "--temperature",
        type=float,
        default=0.7,
        help="Adjust TTS model expressiveness (0.1-2.0, default: 0.7)",
    )
    parser.add_argument(
        "--eos-threshold",
        type=float,
        default=-4.0,
        help="Adjust when TTS finishes (default: -4.0, smaller = earlier)",
    )
    parser.add_argument(
        "--lsd-decode-steps",
        type=int,
        default=1,
        help="Increase TTS quality by running more generations (default: 1)",
    )
    parser.add_argument(
        "--keep",
        action="store_true",
        help="Keep temporary files (for debugging)",
    )

    args = parser.parse_args()
    console = Console()

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
            interactive_chapters=args.select_chapters,
            verbose=args.verbose,
            temperature=args.temperature,
            eos_threshold=args.eos_threshold,
            lsd_decode_steps=args.lsd_decode_steps,
        )

        success = fix_audiobook(ebook_path, audiobook_path, cfg, console)
        sys.exit(0 if success else 1)

    # Handle --list-voices
    if args.list_voices:
        print_available_voices(console)
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
            interactive_chapters=args.select_chapters,
            verbose=args.verbose,
            temperature=args.temperature,
            eos_threshold=args.eos_threshold,
            lsd_decode_steps=args.lsd_decode_steps,
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
