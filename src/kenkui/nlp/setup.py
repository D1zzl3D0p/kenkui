"""Ollama model availability and setup helpers.

Public API
----------
check_llm_available(config)   -> bool
list_recommended_models()     -> list[dict]
pull_ollama_model(model)      -> bool
"""

from __future__ import annotations

import logging
import subprocess
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..models import AppConfig

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Curated model list — update this list as the ecosystem evolves.
# Fields:
#   name        Ollama pull name
#   size_gb     Approximate download / disk size
#   min_ram_gb  Minimum system RAM for comfortable CPU inference
#   desc        One-line description shown in the picker
# ---------------------------------------------------------------------------
RECOMMENDED_MODELS: list[dict] = [
    {
        "name": "gemma2:2b",
        "size_gb": 1.6,
        "min_ram_gb": 4,
        "desc": "Smallest viable; surprisingly strong comprehension per parameter",
    },
    {
        "name": "llama3.2",
        "size_gb": 2.0,
        "min_ram_gb": 6,
        "desc": "Fast, excellent language comprehension (recommended default)",
    },
    {
        "name": "phi3:mini",
        "size_gb": 2.3,
        "min_ram_gb": 6,
        "desc": "Microsoft Phi-3 Mini — punches above its weight for dialogue",
    },
    {
        "name": "mistral",
        "size_gb": 4.1,
        "min_ram_gb": 8,
        "desc": "Strong reasoning and instruction following",
    },
    {
        "name": "llama3.1:8b",
        "size_gb": 4.7,
        "min_ram_gb": 10,
        "desc": "Balanced quality and speed",
    },
    {
        "name": "phi3:medium",
        "size_gb": 7.9,
        "min_ram_gb": 16,
        "desc": "Phi-3 Medium — excellent quality for complex attribution",
    },
    {
        "name": "llama3.3:70b",
        "size_gb": 43.0,
        "min_ram_gb": 48,
        "desc": "Maximum quality — requires high-end hardware",
    },
]


# ---------------------------------------------------------------------------
# System capability helpers
# ---------------------------------------------------------------------------


def _get_ram_gb() -> float:
    try:
        import psutil
        return psutil.virtual_memory().total / (1024 ** 3)
    except Exception:
        return 16.0  # safe default


def _get_vram_gb() -> float | None:
    """Return first GPU's VRAM in GB, or None if undetectable."""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5,
        )
        if out.returncode == 0:
            return int(out.stdout.strip().split("\n")[0]) / 1024
    except Exception:
        pass
    return None


def _get_installed_models() -> set[str]:
    try:
        import ollama
        return {m.model.split(":")[0] + (":" + m.model.split(":")[1] if ":" in m.model else "")
                for m in ollama.list().models}
    except Exception:
        return set()


def check_ollama_running() -> bool:
    try:
        import ollama
        ollama.list()
        return True
    except Exception:
        return False


def check_llm_available(config: AppConfig) -> bool:
    """Return True only when Ollama is running AND the configured model is installed."""
    if not config.nlp_model:
        return False
    if not check_ollama_running():
        return False
    installed = _get_installed_models()
    # Accept both "llama3.2" and "llama3.2:latest" as matching "llama3.2".
    base = config.nlp_model.split(":")[0]
    return any(m.split(":")[0] == base for m in installed)


# ---------------------------------------------------------------------------
# Noninteractive setup helpers
# ---------------------------------------------------------------------------


def list_recommended_models() -> list[dict]:
    """Return recommended Ollama models annotated with local availability."""
    installed = _get_installed_models()
    capacity_gb = max(_get_ram_gb(), _get_vram_gb() or 0)

    results: list[dict] = []
    for model in RECOMMENDED_MODELS:
        name = model["name"]
        base = name.split(":")[0]
        results.append(
            {
                **model,
                "installed": any(item.split(":")[0] == base for item in installed),
                "fits_local_memory": model["min_ram_gb"] <= capacity_gb,
            }
        )
    return results


def pull_ollama_model(model: str, progress_callback=None) -> bool:
    """Pull *model* through Ollama.

    ``progress_callback`` receives the raw Ollama streaming update object when
    supplied. Returns False when Ollama is unavailable or the pull fails.
    """
    if not model:
        return False

    try:
        import ollama

        for update in ollama.pull(model, stream=True):
            if progress_callback is not None:
                progress_callback(update)
        return True
    except Exception as exc:
        logger.warning("Ollama pull failed for %s: %s", model, exc)
        return False


def run_setup_dialogue(config: AppConfig) -> AppConfig | None:
    """Deprecated noninteractive compatibility shim.

    Core no longer owns model-selection prompts. Clients should call
    :func:`list_recommended_models`, choose a model in their own UI, call
    :func:`pull_ollama_model` if needed, and persist the updated config.
    """
    _ = config
    logger.info("run_setup_dialogue is deprecated; clients own NLP setup UI")
    return None
