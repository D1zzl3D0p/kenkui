"""Config management for kenkui.

Configs are arbitrary TOML files.  The default config lives at:

    $XDG_CONFIG_HOME/kenkui/default-config.toml
      (falls back to ~/.config/kenkui/default-config.toml)

When a name (no path separators) is supplied instead of a file path, this
module searches for  $XDG_CONFIG_HOME/kenkui/<name>.toml  automatically.
"""

from __future__ import annotations

import os
import shutil
import tomllib
from dataclasses import dataclass
from pathlib import Path

import tomli_w
from pydantic_settings import TomlConfigSettingsSource

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


def _xdg_cache_home() -> Path:
    """Return the XDG cache home directory, defaulting to ~/.cache."""
    xdg = os.environ.get("XDG_CACHE_HOME", "")
    return Path(xdg) if xdg else Path.home() / ".cache"


def _xdg_state_home() -> Path:
    """Return the XDG state home directory, defaulting to ~/.local/state."""
    xdg = os.environ.get("XDG_STATE_HOME", "")
    return Path(xdg) if xdg else Path.home() / ".local" / "state"


def _kenkui_config_dir() -> Path:
    """Return (and create if needed) the kenkui config directory."""
    d = _xdg_config_home() / "kenkui"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _kenkui_cache_dir() -> Path:
    """Return (and create if needed) the kenkui XDG cache directory."""
    d = _xdg_cache_home() / "kenkui"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _kenkui_state_dir() -> Path:
    """Return (and create if needed) the kenkui XDG state directory."""
    d = _xdg_state_home() / "kenkui"
    d.mkdir(parents=True, exist_ok=True)
    return d


# Public constants — callers import these directly.
# CONFIG_DIR: user configuration files (config.toml, credentials.toml, series/)
# CACHE_DIR:  regeneratable cache data (nlp_cache/, book_cache.json, booknlp_cache/)
# STATE_DIR:  persistent runtime state (logs, analytics.jsonl)
CONFIG_DIR = _kenkui_config_dir()
CACHE_DIR = _kenkui_cache_dir()
STATE_DIR = _kenkui_state_dir()

DEFAULT_CONFIG_PATH = CONFIG_DIR / "default-config.toml"


def _migrate_caches_if_needed() -> None:
    """One-time silent migration: move cache items from CONFIG_DIR → CACHE_DIR.

    Runs at import time so existing installs transparently pick up the new
    XDG-compliant layout without requiring a manual migration step.
    """
    for name in ("nlp_cache", "book_cache.json", "booknlp_cache"):
        src = CONFIG_DIR / name
        dst = CACHE_DIR / name
        if src.exists() and not dst.exists():
            try:
                shutil.move(str(src), str(dst))
            except Exception:
                pass  # never block startup


_migrate_caches_if_needed()

# ---------------------------------------------------------------------------
# Provider credentials
# ---------------------------------------------------------------------------


@dataclass
class ProviderCredentials:
    api_key: str
    default_model: str = ""


_PROVIDER_ENV_VARS = {
    "anthropic": "ANTHROPIC_API_KEY",
    "openai": "OPENAI_API_KEY",
    "google": "GEMINI_API_KEY",
    "openrouter": "OPENROUTER_API_KEY",
}

CREDENTIALS_PATH = CONFIG_DIR / "credentials.toml"


def load_provider_credentials(
    path: Path | None = None,
) -> dict[str, ProviderCredentials]:
    """Load provider credentials from *path* (defaults to CONFIG_DIR/credentials.toml).

    Returns an empty dict if the file does not exist.
    """
    target = path or CREDENTIALS_PATH
    if not target.exists():
        return {}
    try:
        data = tomllib.loads(target.read_text(encoding="utf-8"))
        providers = data.get("providers", {})
        return {
            name: ProviderCredentials(
                api_key=cfg.get("api_key", ""),
                default_model=cfg.get("default_model", ""),
            )
            for name, cfg in providers.items()
        }
    except Exception:
        return {}


def save_provider_credentials(
    credentials: dict[str, ProviderCredentials],
    path: Path | None = None,
) -> Path:
    """Write provider credentials to *path* with 0o600 permissions."""
    target = path or CREDENTIALS_PATH
    target.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "providers": {
            name: {"api_key": creds.api_key, "default_model": creds.default_model}
            for name, creds in credentials.items()
        }
    }
    target.write_bytes(tomli_w.dumps(data).encode("utf-8"))
    target.chmod(0o600)
    return target


_KENKUI_PROVIDER_ENV_VARS = {
    "anthropic": "KENKUI_ANTHROPIC_API_KEY",
    "openai": "KENKUI_OPENAI_API_KEY",
    "google": "KENKUI_GOOGLE_API_KEY",
    "openrouter": "KENKUI_OPENROUTER_API_KEY",
}


def inject_provider_env_vars(credentials: dict[str, ProviderCredentials]) -> None:
    """Set provider API keys as environment variables for LiteLLM.

    KENKUI_*_API_KEY env vars take precedence over credentials.toml so the
    app is fully 12-factor: secrets can live entirely in the environment.
    """
    for provider_name, standard_var in _PROVIDER_ENV_VARS.items():
        kenkui_var = _KENKUI_PROVIDER_ENV_VARS.get(provider_name, "")
        # Prefer KENKUI_* env var, fall back to credentials.toml
        key = os.environ.get(kenkui_var, "")
        if not key and provider_name in credentials:
            key = credentials[provider_name].api_key
        if key:
            os.environ[standard_var] = key


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
    """Load an AppConfig using the 12-factor priority stack.

    Priority (highest to lowest):
      1. CLI flags  — applied by callers via AppConfig.model_copy(update={...})
      2. KENKUI_*   — pydantic-settings env layer
      3. TOML file  — resolved from path_or_name / KENKUI_CONFIG / XDG default
      4. Defaults   — Field default= / default_factory= values

    Creates and persists the default config file on first run.
    """
    # KENKUI_CONFIG env var overrides when no explicit path is given
    if path_or_name is None:
        path_or_name = os.environ.get("KENKUI_CONFIG")

    path = resolve_config_path(path_or_name)
    toml_path: Path | None = path if path.exists() else None

    class _AppConfigForPath(AppConfig):
        @classmethod
        def settings_customise_sources(
            cls,
            settings_cls,
            init_settings,
            env_settings,
            dotenv_settings,
            file_secret_settings,
        ):
            sources: list = [init_settings, env_settings]
            if toml_path is not None:
                sources.append(TomlConfigSettingsSource(settings_cls, toml_file=toml_path))
            return tuple(sources)

    config = _AppConfigForPath()

    # Auto-persist defaults on first run
    if toml_path is None and path == DEFAULT_CONFIG_PATH:
        _write_toml(config, path)

    return config


def save_app_config(config: AppConfig, path: Path | str) -> Path:
    """Save an AppConfig to *path* as TOML.  Creates parent dirs as needed."""
    dest = Path(path).resolve()
    _write_toml(config, dest)
    return dest


def _write_toml(config: AppConfig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = config.model_dump(mode="json", exclude_none=True)
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
