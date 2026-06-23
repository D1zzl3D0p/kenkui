"""Runtime extension registration helpers.

This module keeps optional execution backends out of import-time core paths.
Registration is driven by AppConfig/environment, preserving local as kenkui's
default execution mode.
"""
from __future__ import annotations

from importlib.metadata import entry_points
from typing import TYPE_CHECKING

from kenkui.errors import KenkuiError

if TYPE_CHECKING:
    from kenkui.models import AppConfig


class RuntimeRegistrationError(KenkuiError):
    """Raised when a configured optional runtime cannot be registered."""


RUNTIME_PROVIDER_ENTRY_POINT_GROUP = "kenkui.runtime_providers"


def _runtime_provider_entry_points():
    return entry_points().select(group=RUNTIME_PROVIDER_ENTRY_POINT_GROUP)


def register_configured_runtimes(app_config: AppConfig) -> None:
    """Register optional execution runtimes selected by configuration.

    Modal is discovered through the ``kenkui.runtime_providers`` entry point so
    bare/local installs do not need cloud-provider dependencies or credentials.
    """
    if not getattr(app_config, "modal_enabled", False):
        return

    for entry_point in _runtime_provider_entry_points():
        if entry_point.name != "modal":
            continue
        try:
            register_runtime = entry_point.load()
        except Exception as exc:  # pragma: no cover - exact plugin failure is environment-specific
            raise RuntimeRegistrationError(
                "Modal runtime is enabled but the kenkui-modal plugin failed to load. "
                "Check the kenkui-modal installation and Modal configuration."
            ) from exc
        register_runtime(app_config)
        return

    raise RuntimeRegistrationError(
        "Modal runtime is enabled but no 'modal' runtime provider plugin was found. "
        "Install kenkui-modal before selecting modal execution modes."
    )


__all__ = [
    "RUNTIME_PROVIDER_ENTRY_POINT_GROUP",
    "RuntimeRegistrationError",
    "register_configured_runtimes",
]
