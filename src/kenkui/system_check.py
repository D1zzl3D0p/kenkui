"""system_check — VRAM detection and API key env-var checks.

Used by the post-submission requirement validation to warn the user when
a job's dependencies are not satisfied at queue time.
"""

from __future__ import annotations

import os

_API_KEY_VARS: dict[str, str] = {
    "anthropic": "ANTHROPIC_API_KEY",
    "openai": "OPENAI_API_KEY",
    "google": "GOOGLE_API_KEY",
}

# Rough VRAM estimates (GB) by common model size keywords
_VRAM_ESTIMATES: list[tuple[list[str], float]] = [
    (["70b", "65b"], 40.0),
    (["34b", "30b"], 20.0),
    (["13b", "14b"], 10.0),
    (["7b", "8b"], 6.0),
    (["3b"], 3.0),
    (["phi", "mini", "1b"], 2.0),
]


def get_api_key_var(provider: str) -> str | None:
    """Return the env-var name for a cloud NLP provider, or None if unknown."""
    return _API_KEY_VARS.get(provider.lower())


def check_api_key(provider: str) -> bool:
    """Return True if the required API key for the provider is available.

    Checks the environment variable first, then falls back to credentials.toml
    (written by ``kenkui configure-provider``).
    """
    var = get_api_key_var(provider)
    if not var:
        return True  # Unknown provider — no env var to check
    if os.environ.get(var):
        return True
    # Also check credentials.toml so keys saved via configure-provider are detected
    try:
        from kenkui.config import load_provider_credentials
        creds = load_provider_credentials()
        c = creds.get(provider.lower())
        return bool(c and c.api_key)
    except Exception:
        return False


def get_available_vram_gb() -> float | None:
    """Return available GPU VRAM in GB, or None if it cannot be determined.

    Tries torch.cuda first (more accurate), falls back to nvidia-smi.
    Returns None silently when neither is available.
    """
    # Try torch.cuda
    try:
        import torch

        if torch.cuda.is_available():
            free, _total = torch.cuda.mem_get_info()
            return free / (1024**3)
    except Exception:
        pass

    # Try nvidia-smi
    try:
        import subprocess

        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            lines = result.stdout.strip().split("\n")
            if lines and lines[0].strip().isdigit():
                return int(lines[0].strip()) / 1024.0
    except Exception:
        pass

    return None


def get_ollama_model_vram_gb(model: str) -> float | None:
    """Estimate VRAM requirement for an Ollama model from its name.

    Returns None for unknown/unrecognised model names.
    """
    model_lower = (model or "").lower()
    for keywords, vram_gb in _VRAM_ESTIMATES:
        if any(kw in model_lower for kw in keywords):
            return vram_gb
    return None


__all__ = [
    "get_api_key_var",
    "check_api_key",
    "get_available_vram_gb",
    "get_ollama_model_vram_gb",
]
