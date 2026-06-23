"""Errors raised by optional Modal runtime providers."""
from __future__ import annotations

from kenkui.errors import KenkuiError


class ModalRuntimeError(KenkuiError):
    """Base class for Modal runtime failures."""


class ModalRuntimeUnavailableError(ModalRuntimeError):
    """Raised when Modal execution is selected but no client is configured."""


__all__ = ["ModalRuntimeError", "ModalRuntimeUnavailableError"]
