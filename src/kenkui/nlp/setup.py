"""Ollama model availability check and interactive setup dialogue.

Called when the user chooses multi-voice mode and either no model is
configured or they explicitly ask to reconfigure.

Public API
----------
check_llm_available(config)   → bool
run_setup_dialogue(config)    → AppConfig | None   (None = user cancelled)
"""

from __future__ import annotations

import subprocess
import logging
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


def check_llm_available(config: "AppConfig") -> bool:
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
# Interactive setup dialogue
# ---------------------------------------------------------------------------


def run_setup_dialogue(config: "AppConfig") -> "AppConfig | None":
    """Walk the user through selecting and (if needed) pulling an Ollama model.

    Returns an updated ``AppConfig`` with ``nlp_model`` set, or ``None`` if
    the user cancels.  The caller is responsible for persisting the config.
    """
    from InquirerPy import inquirer
    from rich.console import Console
    from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn
    import ollama

    console = Console()

    console.print()
    console.print("[bold cyan]Multi-voice NLP model setup[/bold cyan]")
    console.print()

    # ---- Check Ollama is reachable ----------------------------------------
    if not check_ollama_running():
        console.print("[red]Ollama is not running (or not installed).[/red]")
        console.print(
            "Install Ollama from [link=https://ollama.com]ollama.com[/link], "
            "start it with [bold]ollama serve[/bold], then try again."
        )
        return None

    # ---- Gather system info for smart recommendations ---------------------
    ram_gb = _get_ram_gb()
    vram_gb = _get_vram_gb()
    installed = _get_installed_models()

    capacity_gb = max(ram_gb, vram_gb or 0)
    logger.debug("RAM %.1f GB, VRAM %s GB", ram_gb, vram_gb)

    # ---- Build picker choices ---------------------------------------------
    choices: list[dict] = []
    for m in RECOMMENDED_MODELS:
        fits = m["min_ram_gb"] <= capacity_gb
        is_installed = any(m["name"].split(":")[0] == i.split(":")[0] for i in installed)
        tag = "[green]✓ installed[/green]" if is_installed else f"~{m['size_gb']:.0f} GB"
        dim = "" if fits else " [dim](may be slow)[/dim]"
        label = f"{m['name']:<20} {tag}  {m['desc']}{dim}"
        choices.append({"name": label, "value": m["name"]})

    choices.append({"name": "Enter model name manually…", "value": "__custom__"})

    # Put installed models first so the cursor lands on them.
    choices.sort(
        key=lambda c: (
            0 if any(
                (c["value"] or "").split(":")[0] == i.split(":")[0] for i in installed
            ) else 1
        )
    )

    # ---- Prompt -----------------------------------------------------------
    current = config.nlp_model
    if current:
        console.print(f"Current model: [bold]{current}[/bold]")
        console.print()

    selected = inquirer.select(
        message="Select Ollama model for speaker inference:",
        choices=choices,
        max_height="50%",
    ).execute()

    if selected is None:
        return None

    if selected == "__custom__":
        selected = inquirer.text(
            message="Enter Ollama model name (e.g. llama3.2 or mistral:7b):"
        ).execute().strip()
        if not selected:
            return None

    # ---- Pull if not installed --------------------------------------------
    base_selected = selected.split(":")[0]
    already_installed = any(i.split(":")[0] == base_selected for i in installed)

    if not already_installed:
        console.print(f"\n[cyan]Pulling [bold]{selected}[/bold] from Ollama library…[/cyan]")
        try:
            with Progress(
                SpinnerColumn(),
                TextColumn("{task.description}"),
                BarColumn(),
                console=console,
            ) as prog:
                task = prog.add_task(f"Downloading {selected}…", total=None)
                for update in ollama.pull(selected, stream=True):
                    if update.total and update.completed:
                        prog.update(
                            task,
                            total=update.total,
                            completed=update.completed,
                            description=update.status or f"Downloading {selected}…",
                        )
                prog.update(task, description="Done!")
            console.print(f"[green]✓ {selected} ready[/green]\n")
        except Exception as exc:
            console.print(f"[red]Pull failed: {exc}[/red]")
            console.print("You can pull it manually with: [bold]ollama pull {selected}[/bold]")
            return None

    # ---- Return updated config --------------------------------------------
    from dataclasses import replace
    return replace(config, nlp_model=selected)
