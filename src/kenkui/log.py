"""Centralised logging configuration for kenkui.

Each process (TUI, server, worker) calls ``setup_logging()`` once at startup.

Default behaviour (12-factor XI): logs go to stdout.
Opt-in file logging: set the ``KENKUI_LOG_FILE`` environment variable to any
non-empty value to write rotating files to ``~/.local/state/kenkui/`` instead::

    KENKUI_LOG_FILE=1 kenkui run          # writes kenkui-tui.log etc.

Usage::

    from kenkui.log import setup_logging
    setup_logging("tui")          # in app.py on_mount
    setup_logging("server")       # in server/server.py main()
    setup_logging("workers")      # in workers.py worker_process_chapter (lazy)
"""

from __future__ import annotations

import logging
import logging.handlers
import os
import sys
from pathlib import Path

from .config import STATE_DIR

LOG_DIR: Path = STATE_DIR
LOG_FORMAT = "%(asctime)s [%(process)d] %(name)s %(levelname)s %(message)s"
LOG_MAX_BYTES = 5 * 1024 * 1024  # 5 MB per file
LOG_BACKUP_COUNT = 2  # keep 2 rotated backups

# Third-party libraries that produce high-volume DEBUG output irrelevant to
# kenkui development.  Setting them to WARNING keeps the log files focused on
# kenkui's own code.  The most egregious offender is httpcore/httpx, which
# emits 13 trace lines per HTTP request — the poll timer makes 2 requests per
# second, flooding kenkui-tui.log with 26 lines/second of noise.
_THIRD_PARTY_WARNING_LOGGERS: tuple[str, ...] = (
    # HTTP clients (poll timer fires every second)
    "httpcore",
    "httpcore.connection",
    "httpcore.http11",
    "httpcore.proxy",
    "httpx",
    "urllib3",
    "urllib3.connection",
    "urllib3.connectionpool",
    "urllib3.poolmanager",
    "urllib3.response",
    "urllib3.util.retry",
    "requests",
    # ASGI / web server
    "uvicorn",
    "uvicorn.access",
    "uvicorn.error",
    "fastapi",
    # ML / NLP stack
    "spacy",
    "torch",
    "torch.__trace",
    "transformers",
    "huggingface_hub",
    "tokenizers",
    "pocket_tts",
    # Async / concurrency
    "asyncio",
    "concurrent.futures",
    "multiprocessing",
    # Misc noisy libraries
    "filelock",
    "charset_normalizer",
    "packaging",
    "rich",
    "tqdm",
    "tqdm.cli",
    "weasel",
    "h5py",
)

_configured_processes: set[str] = set()


def setup_logging(process_name: str, level: int = logging.DEBUG) -> Path | None:
    """Configure the root logger for one kenkui process.

    By default logs go to stdout (12-factor XI). Set the ``KENKUI_LOG_FILE``
    environment variable to any non-empty value to write rotating files to
    ``~/.local/state/kenkui/kenkui-<process_name>.log`` instead.

    Args:
        process_name: One of ``"tui"``, ``"server"``, or ``"workers"``.
        level:        Logging level. Defaults to ``logging.DEBUG``.

    Returns:
        The log ``Path`` when file logging is active, otherwise ``None``.

    Safe to call multiple times for the same process_name; subsequent calls
    are no-ops.
    """
    global _configured_processes

    use_file = bool(os.environ.get("KENKUI_LOG_FILE"))

    if process_name in _configured_processes:
        return LOG_DIR / f"kenkui-{process_name}.log" if use_file else None

    root = logging.getLogger()

    # Remove every existing handler so nothing leaks to stderr / stdout
    for handler in list(root.handlers):
        try:
            handler.close()
        except Exception:
            pass
        root.removeHandler(handler)

    log_path: Path | None = None
    if use_file:
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        log_path = LOG_DIR / f"kenkui-{process_name}.log"
        try:
            fh = logging.handlers.RotatingFileHandler(
                log_path,
                mode="a",
                maxBytes=LOG_MAX_BYTES,
                backupCount=LOG_BACKUP_COUNT,
                encoding="utf-8",
            )
            fh.setLevel(level)
            fh.setFormatter(logging.Formatter(LOG_FORMAT))
            root.addHandler(fh)
        except Exception:
            root.addHandler(logging.NullHandler())
    else:
        sh = logging.StreamHandler(sys.stdout)
        sh.setLevel(level)
        sh.setFormatter(logging.Formatter(LOG_FORMAT))
        root.addHandler(sh)

    root.setLevel(level)

    # Silence third-party loggers that would otherwise inherit DEBUG from root
    # and flood output with irrelevant trace output.
    for _name in _THIRD_PARTY_WARNING_LOGGERS:
        logging.getLogger(_name).setLevel(logging.WARNING)

    _configured_processes.add(process_name)
    return log_path


__all__ = ["setup_logging", "LOG_DIR"]
