"""Centralised logging configuration for kenkui.

Each process (TUI, server, worker) calls ``setup_logging()`` once at startup
to configure the root logger with a per-process rotating file handler.

Log files are written to ``~/.config/kenkui/``:
    kenkui-tui.log     — the main Textual TUI process
    kenkui-server.log  — the uvicorn/FastAPI worker server subprocess
    kenkui-workers.log — ProcessPoolExecutor TTS worker subprocesses

All three files use the same format so they can be concatenated or tailed
together for debugging.  Each message includes the PID so log lines from
different workers in the same pool are distinguishable.

Usage::

    from kenkui.log import setup_logging
    setup_logging("tui")          # in app.py on_mount
    setup_logging("server")       # in server/server.py main()
    setup_logging("workers")      # in workers.py worker_process_chapter (lazy)
"""

from __future__ import annotations

import logging
import logging.handlers
from pathlib import Path

from .config import CONFIG_DIR

LOG_DIR: Path = CONFIG_DIR
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


def setup_logging(process_name: str, level: int = logging.DEBUG) -> Path:
    """Configure the root logger for one kenkui process.

    Removes any pre-existing handlers on the root logger (eliminating bleed
    from StreamHandlers added by ``logging.basicConfig`` or third-party
    libraries such as uvicorn), then attaches a single
    ``RotatingFileHandler`` writing to::

        ~/.config/kenkui/kenkui-<process_name>.log

    Args:
        process_name: One of ``"tui"``, ``"server"``, or ``"workers"``.
                      Used as the filename suffix and echoed in log records
                      via the PID field.
        level:        Logging level for the file handler and root logger.
                      Defaults to ``logging.DEBUG``.

    Returns:
        The ``Path`` of the log file that was configured.

    Safe to call multiple times for the same process_name; subsequent calls
    are no-ops so worker processes that call this function on every chapter
    only pay the setup cost once.
    """
    global _configured_processes

    if process_name in _configured_processes:
        return LOG_DIR / f"kenkui-{process_name}.log"

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_path = LOG_DIR / f"kenkui-{process_name}.log"

    root = logging.getLogger()

    # Remove every existing handler so nothing leaks to stderr / stdout
    for handler in list(root.handlers):
        try:
            handler.close()
        except Exception:
            pass
        root.removeHandler(handler)

    # Attach the rotating file handler
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
        root.setLevel(level)
    except Exception:
        # If we cannot open the log file, fall back to NullHandler
        root.addHandler(logging.NullHandler())

    # Silence third-party loggers that would otherwise inherit DEBUG from root
    # and flood the log with irrelevant trace output.
    for _name in _THIRD_PARTY_WARNING_LOGGERS:
        logging.getLogger(_name).setLevel(logging.WARNING)

    _configured_processes.add(process_name)
    return log_path


__all__ = ["setup_logging", "LOG_DIR"]
