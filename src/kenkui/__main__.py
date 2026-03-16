"""
Kenkui — Ebook to Audiobook Converter
Entry point for both TUI and headless (CLI) modes.

Usage:
  kenkui                          # open TUI, browse for a book
  kenkui /path/to/book.epub       # headless: convert using saved config
  kenkui /path/to/dir             # open TUI pre-navigated to that directory
  kenkui --list-voices            # print available voices and exit
  kenkui --list-presets           # print chapter filter presets and exit
  kenkui /path/to/book.epub -c myprofile          # use named config
  kenkui /path/to/book.epub --output ~/Audiobooks # override output dir
"""

from __future__ import annotations

import logging
import multiprocessing
import subprocess
import sys
import time
from pathlib import Path

# MUST set multiprocessing start method BEFORE any multiprocessing usage
if __name__ == "__main__":
    try:
        multiprocessing.set_start_method("spawn", force=True)
    except RuntimeError:
        pass

import argparse

import httpx

from .app import run_app

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 45365

# Recognised ebook extensions that trigger headless mode
EBOOK_EXTENSIONS = {".epub", ".mobi", ".fb2", ".azw", ".azw3", ".azw4"}


# ---------------------------------------------------------------------------
# Server lifecycle helpers
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
    if args.no_auto_server:
        return None
    try:
        httpx.get(f"http://{args.server_host}:{args.server_port}/health", timeout=2.0)
        return None  # already running
    except (httpx.ConnectError, httpx.ReadTimeout):
        return start_server(args.server_host, args.server_port)


def shutdown_server(proc: subprocess.Popen | None):
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
# Informational commands (no server needed)
# ---------------------------------------------------------------------------


def _list_voices() -> int:
    from .helpers import get_bundled_voices
    from .utils import DEFAULT_VOICES, VOICE_DESCRIPTIONS

    print("=== Built-in Voices ===")
    for v in DEFAULT_VOICES:
        desc = VOICE_DESCRIPTIONS.get(v, "")
        print(f"  {v:<20} {desc}")
    bundled = [b for b in get_bundled_voices() if b.lower() != "default.txt"]
    if bundled:
        print("\n=== Bundled Voices (requires HuggingFace login) ===")
        for b in bundled:
            print(f"  {b.replace('.wav', '')}")
    print("\nCustom voices: pass a local file path or hf://user/repo/voice.wav")
    return 0


def _list_presets() -> int:
    from .chapter_filter import ChapterFilter

    print("=== Chapter Filter Presets ===")
    for name, desc in ChapterFilter.list_presets():
        print(f"  {name:<20} {desc}")
    return 0


# ---------------------------------------------------------------------------
# Headless processing
# ---------------------------------------------------------------------------


def _run_headless(args) -> int:
    """Submit a job to the server and poll until it completes."""
    from .api_client import APIClient
    from .config import get_config_manager
    from .models import AppConfig, ChapterPreset, ChapterSelection

    cfg_mgr = get_config_manager()
    app_config: AppConfig = cfg_mgr.load_app_config(args.config)

    input_path: Path = args.input
    if args.output:
        output_dir = Path(args.output).expanduser().resolve()
    elif app_config.default_output_dir:
        output_dir = Path(app_config.default_output_dir).expanduser().resolve()
    else:
        output_dir = input_path.parent

    voice = app_config.default_voice
    preset_str = app_config.default_chapter_preset

    try:
        preset_enum = ChapterPreset(preset_str)
    except ValueError:
        preset_enum = ChapterPreset.CONTENT_ONLY
    chapter_selection = ChapterSelection(preset=preset_enum).to_dict()

    print(f"Book:    {input_path}")
    print(f"Voice:   {voice}")
    print(f"Preset:  {preset_str}")
    print(f"Output:  {output_dir}")
    print(f"Config:  {app_config.name}")
    print()

    server_proc = ensure_server(args)
    client = APIClient(host=args.server_host, port=args.server_port)

    try:
        client.update_config(app_config.to_dict())
        job_info = client.add_job(
            ebook_path=str(input_path),
            voice=voice,
            chapter_selection=chapter_selection,
            output_path=str(output_dir),
        )
        job_id = job_info.id
        print(f"Job queued: {job_id}")
        client.start_processing()

        last_chapter = ""
        while True:
            time.sleep(2)
            try:
                item = client.get_job(job_id)
            except Exception:
                continue

            if item is None:
                print("Error: job disappeared from queue.", file=sys.stderr)
                return 1

            progress = item.progress
            chapter = item.current_chapter or ""
            filled = int(progress / 5)
            bar = "█" * filled + "░" * (20 - filled)
            line = f"\r  [{bar}] {progress:5.1f}%"
            if chapter and chapter != last_chapter:
                line += f"  {chapter[:50]}"
                last_chapter = chapter
            print(line, end="", flush=True)

            if item.status == "completed":
                print()
                print(f"\nDone! Output: {item.output_path or output_dir}")
                return 0
            elif item.status == "failed":
                print()
                print(f"\nFailed: {item.error_message}", file=sys.stderr)
                return 1
            elif item.status == "cancelled":
                print()
                print("\nCancelled.", file=sys.stderr)
                return 1

    finally:
        client.close()
        shutdown_server(server_proc)

    return 0


# ---------------------------------------------------------------------------
# TUI mode
# ---------------------------------------------------------------------------


def _run_tui(args) -> int:
    server_proc = ensure_server(args)
    try:
        run_app(
            config_name=args.config,
            initial_path=args.input,
            server_host=args.server_host,
            server_port=args.server_port,
        )
        return 0
    finally:
        shutdown_server(server_proc)


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    from . import __version__

    parser = argparse.ArgumentParser(
        prog="kenkui",
        description=(
            "Kenkui — Ebook to Audiobook Converter.\n\n"
            "Passing an ebook file (.epub/.mobi/.fb2) runs headless (no TUI).\n"
            "Passing a directory or no argument opens the interactive TUI.\n\n"
            "Create named configs in the TUI (Config screen → Save Config…)\n"
            "then reuse them with:  kenkui book.epub -c myconfig"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "input",
        type=Path,
        nargs="?",
        default=None,
        help="Ebook file (headless) or directory (TUI). Omit to open TUI.",
    )
    parser.add_argument(
        "-c",
        "--config",
        default=None,
        metavar="NAME",
        help=(
            "Named config from ~/.config/kenkui/configs/<NAME>.yaml. "
            "Defaults to 'default'."
        ),
    )
    parser.add_argument(
        "-o",
        "--output",
        default=None,
        metavar="DIR",
        help="Output directory (headless only). Overrides config default_output_dir.",
    )
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
    parser.add_argument(
        "--list-voices",
        action="store_true",
        help="Print all available voices and exit.",
    )
    parser.add_argument(
        "--list-presets",
        action="store_true",
        help="Print all chapter filter presets and exit.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose (DEBUG) logging to stderr.",
    )
    parser.add_argument(
        "--log-file",
        default=None,
        metavar="PATH",
        help="Write log output to this file in addition to stderr.",
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {__version__}",
    )
    return parser


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = _build_parser()
    args = parser.parse_args()

    log_level = logging.DEBUG if args.verbose else logging.WARNING
    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stderr)]
    if args.log_file:
        handlers.append(logging.FileHandler(args.log_file))
    logging.basicConfig(
        level=log_level,
        format="%(levelname)s [%(name)s] %(message)s",
        handlers=handlers,
    )

    if args.list_voices:
        sys.exit(_list_voices())
    if args.list_presets:
        sys.exit(_list_presets())

    if (
        args.input is not None
        and args.input.is_file()
        and args.input.suffix.lower() in EBOOK_EXTENSIONS
    ):
        if not args.input.exists():
            print(f"Error: file not found: {args.input}", file=sys.stderr)
            sys.exit(1)
        sys.exit(_run_headless(args))
    else:
        if args.input is not None and not args.input.exists():
            print(f"Error: path not found: {args.input}", file=sys.stderr)
            sys.exit(1)
        sys.exit(_run_tui(args))


if __name__ == "__main__":
    main()
