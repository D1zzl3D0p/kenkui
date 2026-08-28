"""Structured library logging that leaves application handler configuration alone."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Mapping

LogContext = str | int | bool


def get_logger(name: str) -> logging.Logger:
    """Return a namespaced library logger without configuring any handlers."""
    return logging.getLogger(name)


def log_event(
    logger: logging.Logger,
    event: str,
    *,
    level: int = logging.INFO,
    context: Mapping[str, LogContext],
) -> None:
    """Emit an event name and safe structured fields through the caller's logger."""
    logger.log(level, event, extra={"event": event, **context})
