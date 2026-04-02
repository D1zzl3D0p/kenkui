"""
Kenkui — Ebook to Audiobook Converter
Entry point / sub-command dispatcher.

Sub-commands
------------
  kenkui book.epub                    Interactive wizard → auto-start queue → live dashboard
  kenkui book.epub -c config.toml     Headless: queue → start → Rich progress poll → exit 0/1

  kenkui add book.epub                Interactive wizard → queue only (prints hint)
  kenkui add book.epub -c config.toml Headless: queue only → print hint (no auto-start)

  kenkui queue                        Snapshot: Rich table of all jobs, exits immediately
  kenkui queue --live                 Live-refreshing Rich dashboard (Ctrl+C to exit)
  kenkui queue start                  Start processing next pending job
  kenkui queue start --live           Start processing + enter live dashboard
  kenkui queue stop                   Stop current job

  kenkui config path/to/config.toml   Create/edit a config at the given path
"""

from __future__ import annotations

import logging
import multiprocessing
import subprocess
import sys
import time
from pathlib import Path

# MUST set multiprocessing start method BEFORE any multiprocessing usage.
if __name__ == "__main__":
    try:
        multiprocessing.set_start_method("spawn", force=True)
    except RuntimeError:
        pass

import argparse

import httpx

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 45365

# Recognised ebook extensions that trigger the bare-path shorthand.
EBOOK_EXTENSIONS = {".epub", ".mobi", ".fb2", ".azw", ".azw3", ".azw4"}


# ---------------------------------------------------------------------------
# Server lifecycle helpers  (kept verbatim from the original implementation)
# ---------------------------------------------------------------------------


def wait_for_server(host: str, port: int, timeout: int = 30) -> bool:
    """Poll until the server responds healthy or timeout expires."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            r = httpx.get(f"http://{host}:{port}/health", timeout=2.0)
            if r.status_code == 200:
                return True
        except (httpx.ConnectError, httpx.ReadTimeout):
            pass
        time.sleep(0.5)
    return False


def start_server(host: str, port: int) -> subprocess.Popen:
    """Launch the worker server as a subprocess and wait until it is ready."""
    print(f"Starting kenkui worker server on {host}:{port}…", file=sys.stderr)
    proc = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "kenkui.server.server",
            "--host",
            host,
            "--port",
            str(port),
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    if not wait_for_server(host, port):
        proc.terminate()
        print("Error: server failed to start within timeout.", file=sys.stderr)
        sys.exit(1)
    print("Server started.", file=sys.stderr)
    return proc


def ensure_server(args) -> subprocess.Popen | None:
    """Start the server if it is not already running."""
    if getattr(args, "no_auto_server", False):
        return None
    try:
        httpx.get(f"http://{args.server_host}:{args.server_port}/health", timeout=2.0)
        return None  # already running
    except (httpx.ConnectError, httpx.ReadTimeout):
        return start_server(args.server_host, args.server_port)


def shutdown_server(proc: subprocess.Popen | None) -> None:
    if proc is None:
        return
    print("Shutting down worker server…", file=sys.stderr)
    proc.terminate()
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------


def _add_server_flags(parser: argparse.ArgumentParser) -> None:
    """Attach the common server-connection flags to *parser*."""
    parser.add_argument(
        "--server-host",
        default=DEFAULT_HOST,
        help=f"Worker server host (default: {DEFAULT_HOST})",
    )
    parser.add_argument(
        "--server-port",
        type=int,
        default=DEFAULT_PORT,
        help=f"Worker server port (default: {DEFAULT_PORT})",
    )
    parser.add_argument(
        "--no-auto-server",
        action="store_true",
        help="Do not auto-start the worker server (assume it is already running).",
    )


def _build_bare_parser() -> argparse.ArgumentParser:
    """Parser for the bare shorthand: kenkui book.epub [-c config] [-o dir]."""
    from . import __version__

    parser = argparse.ArgumentParser(
        prog="kenkui",
        description=(
            "Kenkui — Ebook to Audiobook Converter.\n\n"
            "Pass an ebook path directly to run the interactive wizard then\n"
            "auto-start the queue with a live dashboard.  Add -c config.toml\n"
            "to skip the wizard and run headless instead.\n\n"
            "Sub-commands: add, queue, config, voices"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    parser.add_argument("--verbose", action="store_true", help="Enable DEBUG logging.")
    parser.add_argument("--log-file", default=None, metavar="PATH")
    _add_server_flags(parser)
    parser.add_argument("book", type=Path, help="Ebook file path.")
    parser.add_argument("-c", "--config", default=None, metavar="PATH_OR_NAME")
    parser.add_argument("-o", "--output", default=None, metavar="DIR")
    return parser


def _build_parser() -> argparse.ArgumentParser:
    from . import __version__

    parser = argparse.ArgumentParser(
        prog="kenkui",
        description=(
            "Kenkui — Ebook to Audiobook Converter.\n\n"
            "Pass an ebook path directly to run the interactive wizard then\n"
            "auto-start the queue with a live dashboard.  Add -c config.toml\n"
            "to skip the wizard and run headless instead.\n\n"
            "Sub-commands: add, queue, config, voices"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    parser.add_argument("--verbose", action="store_true", help="Enable DEBUG logging.")
    parser.add_argument(
        "--log-file",
        default=None,
        metavar="PATH",
        help="Write log output to this file.",
    )
    _add_server_flags(parser)
    parser.add_argument(
        "-c",
        "--config",
        default=None,
        metavar="PATH_OR_NAME",
        help=(
            "Config file path or bare name to look up in XDG config dir. "
            "Triggers headless mode when combined with a book path."
        ),
    )
    parser.add_argument(
        "-o",
        "--output",
        default=None,
        metavar="DIR",
        help="Output directory (headless only).",
    )

    sub = parser.add_subparsers(dest="command")

    # ---- kenkui add --------------------------------------------------------
    add_p = sub.add_parser("add", help="Add a book to the queue (interactive wizard).")
    add_p.add_argument("book", type=Path, help="Path to the ebook file.")
    add_p.add_argument(
        "-c",
        "--config",
        default=None,
        metavar="PATH_OR_NAME",
        help="Config file/name. Triggers headless mode (queue only, no auto-start).",
    )
    add_p.add_argument("-o", "--output", default=None, metavar="DIR")
    _add_server_flags(add_p)

    # ---- kenkui queue ------------------------------------------------------
    queue_p = sub.add_parser("queue", help="View or control the job queue.")
    queue_p.add_argument("--live", action="store_true", help="Show a live-refreshing dashboard.")
    _add_server_flags(queue_p)

    queue_sub = queue_p.add_subparsers(dest="queue_command")

    queue_start = queue_sub.add_parser("start", help="Start processing.")
    queue_start.add_argument(
        "--live", action="store_true", help="Enter live dashboard after starting."
    )
    _add_server_flags(queue_start)

    queue_sub.add_parser("stop", help="Stop current job.")

    # ---- kenkui config -----------------------------------------------------
    cfg_p = sub.add_parser("config", help="Create or edit a config file.")
    cfg_p.add_argument(
        "path",
        metavar="PATH_OR_NAME",
        help=(
            "Destination file path (e.g. ~/my-config.toml) or bare name to "
            "save in the XDG config directory."
        ),
    )
    _add_server_flags(cfg_p)

    # ---- kenkui voices -----------------------------------------------------
    voices_p = sub.add_parser("voices", help="List and manage available voices.")
    voices_sub = voices_p.add_subparsers(dest="voices_command")

    voices_list_p = voices_sub.add_parser("list", help="List available voices.")
    voices_list_p.add_argument("--gender", default=None, help="Filter by gender (Male/Female).")
    voices_list_p.add_argument("--accent", default=None, help="Filter by accent (e.g. Scottish).")
    voices_list_p.add_argument("--dataset", default=None, help="Filter by dataset (VCTK/EARS).")
    voices_list_p.add_argument(
        "--source",
        default=None,
        choices=["compiled", "builtin", "uncompiled"],
        help="Filter by source type.",
    )

    voices_fetch_p = voices_sub.add_parser(
        "fetch", help="Download custom uncompiled voices from a HuggingFace repo."
    )
    voices_fetch_p.add_argument(
        "--repo",
        default=None,
        metavar="USER/REPO",
        help="HuggingFace repo containing .wav voice files. Overrides KENKUI_VOICES_REPO env var.",
    )

    voices_download_p = voices_sub.add_parser(
        "download", help="Download compiled voices from HuggingFace."
    )
    voices_download_p.add_argument(
        "--force",
        action="store_true",
        help="Re-download voices even if already present.",
    )

    voices_exclude_p = voices_sub.add_parser(
        "exclude", help="Exclude a voice from the auto-assignment pool."
    )
    voices_exclude_p.add_argument("voice", help="Voice name to exclude.")

    voices_include_p = voices_sub.add_parser(
        "include", help="Re-add a previously excluded voice to the pool."
    )
    voices_include_p.add_argument("voice", help="Voice name to re-include.")

    voices_cast_p = voices_sub.add_parser(
        "cast", help="Show character→voice cast for a completed book."
    )
    voices_cast_p.add_argument("title", help="Book title (fuzzy matched).")

    voices_audition_p = voices_sub.add_parser(
        "audition", help="Synthesize a voice preview and open it in the system player."
    )
    voices_audition_p.add_argument("voice", help="Voice name to audition.")
    voices_audition_p.add_argument(
        "--text", default=None, metavar="TEXT",
        help="Text to synthesize (default: built-in sample phrase).",
    )

    return parser


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def _ensure_voices() -> None:
    """Download voices on first run if not present."""
    from rich.console import Console
    from rich.panel import Panel
    from .voices.download import voices_are_present, download_voices

    console = Console()
    try:
        if not voices_are_present():
            console.print(Panel(
                "[bold]Voice files not found.[/bold]\n"
                "Downloading compiled voices from HuggingFace (~440 MB)…\n"
                "[dim]This only happens once. Use 'kenkui voices download' to re-download.[/dim]",
                title="First Run Setup",
                border_style="cyan",
            ))
            download_voices()
            console.print("[green]✓ Voices downloaded successfully.[/green]")
    except Exception as exc:
        console.print(f"[yellow]Warning: Could not download voices: {exc}[/yellow]")
        console.print("[dim]8 built-in voices are still available. Run 'kenkui voices download' later.[/dim]")


def _is_bare_book_invocation() -> bool:
    """Return True when sys.argv looks like 'kenkui book.epub [flags]'.

    We detect this by checking whether the first non-flag argument in sys.argv
    (after the program name) looks like an ebook path rather than a sub-command
    name.  This lets us route to _build_bare_parser() before argparse gets a
    chance to confuse the path with a sub-command.
    """
    for arg in sys.argv[1:]:
        if arg.startswith("-"):
            # Skip flags and their values (best-effort; doesn't need to be
            # perfect because _build_bare_parser will catch any real errors).
            continue
        # First non-flag arg: if it has an ebook extension it's a bare book path.
        return Path(arg).suffix.lower() in EBOOK_EXTENSIONS
    return False


def main() -> None:
    # Pre-detect bare-book invocation BEFORE handing off to the subparser,
    # because argparse subparsers clash with positional paths that contain
    # spaces or dots when they're not listed as valid sub-command names.
    if _is_bare_book_invocation():
        bare_parser = _build_bare_parser()
        args = bare_parser.parse_args()
    else:
        parser = _build_parser()
        args = parser.parse_args()
        # Attach a dummy bare_parser reference so the else-branch below can
        # call parser.print_help() safely.
        bare_parser = parser  # type: ignore[assignment]

    # Logging setup — headless / CLI only; TUI-style stripping not needed.
    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stderr)]
    if getattr(args, "log_file", None):
        handlers.append(logging.FileHandler(args.log_file))
    logging.basicConfig(
        level=logging.DEBUG if getattr(args, "verbose", False) else logging.WARNING,
        format="%(levelname)s [%(name)s] %(message)s",
        handlers=handlers,
    )

    command = getattr(args, "command", None)

    # First-run voice check — skip when the user is already running voices download
    # to avoid a redundant download before the explicit one starts.
    voices_cmd = getattr(args, "voices_command", None)
    if not (command == "voices" and voices_cmd == "download"):
        _ensure_voices()

    # ---- Sub-command dispatch ----------------------------------------------
    if command == "add":
        from .cli.add import cmd_add

        if args.book is None:
            _build_parser().error("kenkui add: book path is required.")
        server_proc = ensure_server(args)
        try:
            sys.exit(cmd_add(args))
        finally:
            # 'add' queues only; server stays alive for subsequent commands.
            shutdown_server(server_proc)

    elif command == "queue":
        from .cli.queue import cmd_queue

        server_proc = ensure_server(args)
        try:
            sys.exit(cmd_queue(args))
        finally:
            shutdown_server(server_proc)

    elif command == "config":
        from .cli.config import cmd_config

        sys.exit(cmd_config(args))

    elif command == "voices":
        from .cli.voices import (
            cmd_voices_list,
            cmd_voices_fetch,
            cmd_voices_download,
            cmd_voices_exclude,
            cmd_voices_include,
            cmd_voices_cast,
            cmd_voices_audition,
            cmd_voices_tui,
        )

        voices_command = getattr(args, "voices_command", None)
        if voices_command == "list":
            cmd_voices_list(args)
        elif voices_command == "fetch":
            cmd_voices_fetch(args)
        elif voices_command == "download":
            sys.exit(cmd_voices_download(args))
        elif voices_command == "exclude":
            cmd_voices_exclude(args)
        elif voices_command == "include":
            cmd_voices_include(args)
        elif voices_command == "cast":
            cmd_voices_cast(args)
        elif voices_command == "audition":
            cmd_voices_audition(args)
        else:
            # No subcommand: launch interactive TUI
            cmd_voices_tui(args)
        sys.exit(0)

    else:
        # ---- Bare shorthand: kenkui book.epub [-c config] ------------------
        book_path: Path | None = getattr(args, "book", None)
        if book_path is None:
            bare_parser.print_help()
            sys.exit(0)

        if not book_path.exists():
            print(f"Error: file not found: {book_path}", file=sys.stderr)
            sys.exit(1)
        if book_path.suffix.lower() not in EBOOK_EXTENSIONS:
            print(
                f"Error: unrecognised ebook format: {book_path.suffix}",
                file=sys.stderr,
            )
            sys.exit(1)

        from .cli.add import cmd_bare

        server_proc = ensure_server(args)
        try:
            sys.exit(cmd_bare(args))
        finally:
            shutdown_server(server_proc)


if __name__ == "__main__":
    main()
