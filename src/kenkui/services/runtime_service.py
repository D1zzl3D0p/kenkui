"""Runtime extension registration helpers.

This module keeps optional execution backends (Modal today) out of import-time
core paths. Registration is driven by AppConfig/environment, preserving local as
kenkui's default execution mode.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

from kenkui.errors import KenkuiError

if TYPE_CHECKING:
    from kenkui.models import AppConfig


class RuntimeRegistrationError(KenkuiError):
    """Raised when a configured optional runtime cannot be registered."""


def register_configured_runtimes(app_config: AppConfig) -> None:
    """Register optional execution runtimes selected by configuration.

    Modal is intentionally imported lazily so bare/local installs do not need
    the optional ``kenkui[modal]`` dependencies or Modal credentials.
    """
    if not getattr(app_config, "modal_enabled", False):
        return

    try:
        from kenkui.modal_runtime import register_modal_runtime
    except ImportError as exc:  # pragma: no cover - exact import failure is environment-specific
        raise RuntimeRegistrationError(
            "Modal runtime is enabled but unavailable. Install kenkui[modal] and configure "
            "Modal credentials before selecting modal execution modes."
        ) from exc

    register_modal_runtime(app_config)


__all__ = ["RuntimeRegistrationError", "register_configured_runtimes"]
