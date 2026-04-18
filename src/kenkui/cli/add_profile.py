"""Profile persistence for the job confirmation screen."""
from __future__ import annotations
from pathlib import Path


def _profile_path() -> Path:
    import os
    config_home = Path(os.environ.get("XDG_CONFIG_HOME", Path.home() / ".config"))
    return config_home / "kenkui" / "last_job_profile.toml"


def load_last_profile() -> dict:
    """Load last-used job profile. Returns {} on first run or error."""
    path = _profile_path()
    if not path.exists():
        return {}
    try:
        try:
            import tomllib
        except ImportError:
            import tomli as tomllib  # type: ignore[no-redef]
        return tomllib.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def save_last_profile(profile: dict) -> None:
    """Persist profile dict to disk as TOML."""
    path = _profile_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        import tomli_w
        raw = tomli_w.dumps(profile)
        if isinstance(raw, bytes):
            path.write_bytes(raw)
        else:
            path.write_text(raw, encoding="utf-8")
    except ImportError:
        # Minimal TOML writer for flat string/number/bool dicts
        lines = []
        for k, v in profile.items():
            if isinstance(v, str):
                lines.append(f'{k} = "{v}"')
            elif isinstance(v, bool):
                lines.append(f"{k} = {'true' if v else 'false'}")
            elif isinstance(v, (int, float)) and not isinstance(v, bool):
                lines.append(f"{k} = {v}")
            # Skip None and nested dicts for the fallback
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _quality_from_profile(profile: dict, app_config) -> dict:
    """Extract quality overrides from the profile dict.

    Returns the ``quality_overrides`` sub-dict if present, or ``{}`` on first run.
    ``app_config`` is accepted for future use (e.g. filtering keys that match
    server defaults) but is not used by the current implementation.
    """
    from ..services.confirmation_service import quality_overrides_from_profile

    return quality_overrides_from_profile(profile)
