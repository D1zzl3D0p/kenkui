"""Config management for kenkui.

Configs are arbitrary TOML files.  The default config lives at:

    $XDG_CONFIG_HOME/kenkui/default-config.toml
      (falls back to ~/.config/kenkui/default-config.toml)

When a name (no path separators) is supplied instead of a file path, this
module searches for  $XDG_CONFIG_HOME/kenkui/<name>.toml  automatically.
"""

from __future__ import annotations

import os
import tomllib
from pathlib import Path

import tomli_w

from .models import AppConfig

# ---------------------------------------------------------------------------
# XDG helpers
# ---------------------------------------------------------------------------


def _xdg_config_home() -> Path:
    """Return the XDG config home directory, defaulting to ~/.config."""
    xdg = os.environ.get("XDG_CONFIG_HOME", "")
    if xdg:
        return Path(xdg)
    return Path.home() / ".config"


def _kenkui_config_dir() -> Path:
    """Return (and create if needed) the kenkui config directory."""
    d = _xdg_config_home() / "kenkui"
    d.mkdir(parents=True, exist_ok=True)
    return d


# Public constant so server/worker.py can locate the queue file without
# importing the full ConfigManager.
CONFIG_DIR = _kenkui_config_dir()

DEFAULT_CONFIG_PATH = CONFIG_DIR / "default-config.toml"

# ---------------------------------------------------------------------------
# Resolution helper
# ---------------------------------------------------------------------------


def resolve_config_path(path_or_name: str | None) -> Path:
    """Resolve a config specifier to an absolute Path.

    Resolution order:
    1. ``None``  →  default-config.toml in XDG dir.
    2. Existing file path  →  use directly.
    3. Bare name (no path separators)  →  search XDG dir for <name>.toml.
    4. Anything else  →  treat as a literal path (may not exist yet).
    """
    if path_or_name is None:
        return DEFAULT_CONFIG_PATH

    candidate = Path(path_or_name)

    # If it looks like an explicit path (has directory components or extension)
    # and the file exists, use it directly.
    if candidate.exists() and candidate.is_file():
        return candidate.resolve()

    # Bare name → search XDG dir
    if os.sep not in path_or_name and "/" not in path_or_name:
        xdg_candidate = _kenkui_config_dir() / f"{path_or_name}.toml"
        if xdg_candidate.exists():
            return xdg_candidate
        # Name given but not found — still return the XDG path so callers can
        # create it there if they want.
        return xdg_candidate

    # Explicit path that doesn't exist yet — return as-is so callers can write it.
    return candidate.resolve()


# ---------------------------------------------------------------------------
# Read / write
# ---------------------------------------------------------------------------


def load_app_config(path_or_name: str | None = None) -> AppConfig:
    """Load an AppConfig from a TOML file.

    Creates and persists a default config if the resolved path does not exist.
    """
    path = resolve_config_path(path_or_name)

    if path.exists():
        try:
            data = tomllib.loads(path.read_text(encoding="utf-8"))
            return AppConfig.from_dict(data)
        except Exception:
            pass  # Fall through to defaults

    # No file (or unparseable) — return and immediately persist defaults so the
    # file exists on the next invocation.
    config = AppConfig()
    if path == DEFAULT_CONFIG_PATH:
        _write_toml(config, path)
    return config


def save_app_config(config: AppConfig, path: Path | str) -> Path:
    """Save an AppConfig to *path* as TOML.  Creates parent dirs as needed."""
    dest = Path(path).resolve()
    _write_toml(config, dest)
    return dest


def _strip_none(obj: object) -> object:
    """Recursively remove None values from dicts/lists — TOML has no null type."""
    if isinstance(obj, dict):
        return {k: _strip_none(v) for k, v in obj.items() if v is not None}
    if isinstance(obj, list):
        return [_strip_none(v) for v in obj if v is not None]
    return obj


def _write_toml(config: AppConfig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = _strip_none(config.to_dict())
    assert isinstance(data, dict)
    path.write_bytes(tomli_w.dumps(data).encode("utf-8"))


# ---------------------------------------------------------------------------
# Backwards-compatible ConfigManager shim
# ---------------------------------------------------------------------------
# Parts of the codebase (server, tests) import ConfigManager / get_config_manager.
# We keep a thin shim so those call-sites don't need to change.


class ConfigManager:
    """Thin compatibility wrapper around the module-level helpers."""

    def load_app_config(self, name: str | None = None) -> AppConfig:
        return load_app_config(name)

    def save_app_config(self, config: AppConfig, path: Path | str | None = None) -> Path:
        dest = path if path is not None else DEFAULT_CONFIG_PATH
        return save_app_config(config, dest)


_config_manager: ConfigManager | None = None


def get_config_manager() -> ConfigManager:
    global _config_manager
    if _config_manager is None:
        _config_manager = ConfigManager()
    return _config_manager
