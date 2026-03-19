"""kenkui config — interactive config creator / editor.

Usage
-----
    kenkui config path/to/config.toml
    kenkui config my-profile          # searches XDG dir for my-profile.toml

If the resolved path already exists the wizard pre-fills every field with the
current values.  If it does not exist the wizard starts from defaults.

Saves to the specified path on confirmation.
"""

from __future__ import annotations

import multiprocessing
from pathlib import Path

from rich.console import Console
from rich.table import Table

console = Console()


def cmd_config(args) -> int:
    """Handle 'kenkui config path/to/config.toml'."""
    from InquirerPy import inquirer
    from InquirerPy.validator import NumberValidator

    from ..config import load_app_config, resolve_config_path, save_app_config
    from ..models import AppConfig

    path_spec: str = args.path
    dest_path = resolve_config_path(path_spec)

    # Load existing or use defaults.
    if dest_path.exists():
        console.print(f"Editing existing config: [bold]{dest_path}[/bold]")
        cfg = load_app_config(path_spec)
    else:
        console.print(f"Creating new config: [bold]{dest_path}[/bold]")
        cfg = AppConfig()

    console.print()

    # ---- Voice choices ------------------------------------------------
    from ..helpers import get_bundled_voices
    from ..utils import DEFAULT_VOICES, VOICE_DESCRIPTIONS

    voice_choices = []
    for v in DEFAULT_VOICES:
        desc = VOICE_DESCRIPTIONS.get(v, "")
        voice_choices.append({"name": f"{v:<20} {desc}", "value": v})
    for wav in get_bundled_voices():
        if wav.lower() == "default.txt":
            continue
        name = wav.replace(".wav", "")
        voice_choices.append({"name": f"{name:<20} (bundled)", "value": name})

    # ---- Prompt each field --------------------------------------------

    workers = inquirer.number(
        message="Parallel TTS workers:",
        default=cfg.workers,
        min_allowed=1,
        max_allowed=multiprocessing.cpu_count(),
        validate=NumberValidator(),
    ).execute()

    default_output_dir = (
        inquirer.text(
            message="Default output directory (blank = same as ebook):",
            default=str(cfg.default_output_dir) if cfg.default_output_dir else "",
        )
        .execute()
        .strip()
        or None
    )

    default_voice = inquirer.fuzzy(
        message="Default voice:",
        choices=voice_choices,
        default=cfg.default_voice,
        max_height="40%",
    ).execute()

    preset_choices = [
        {"name": "Content Only  (body chapters, skip front/back matter)", "value": "content-only"},
        {"name": "Main Chapters  (titled chapters only)", "value": "chapters-only"},
        {"name": "With Parts  (chapters + part headings)", "value": "with-parts"},
        {"name": "All  (every item in the ebook)", "value": "all"},
        {"name": "None  (skip all chapters)", "value": "none"},
    ]
    default_chapter_preset = inquirer.select(
        message="Default chapter preset:",
        choices=preset_choices,
        default=cfg.default_chapter_preset,
    ).execute()

    bitrate_choices = [
        {"name": "64k  (small files, lower quality)", "value": "64k"},
        {"name": "96k  (default)", "value": "96k"},
        {"name": "128k", "value": "128k"},
        {"name": "192k", "value": "192k"},
        {"name": "256k  (large files, high quality)", "value": "256k"},
    ]
    m4b_bitrate = inquirer.select(
        message="M4B output bitrate:",
        choices=bitrate_choices,
        default=cfg.m4b_bitrate,
    ).execute()

    pause_line_ms = inquirer.number(
        message="Pause between lines (ms):",
        default=cfg.pause_line_ms,
        min_allowed=0,
        validate=NumberValidator(),
    ).execute()

    pause_chapter_ms = inquirer.number(
        message="Pause between chapters (ms):",
        default=cfg.pause_chapter_ms,
        min_allowed=0,
        validate=NumberValidator(),
    ).execute()

    temp = inquirer.number(
        message="Sampling temperature (lower=stable, higher=expressive) [0.0–1.5]:",
        default=cfg.temp,
        min_allowed=0.0,
        max_allowed=1.5,
        float_allowed=True,
    ).execute()

    lsd_decode_steps = inquirer.number(
        message="LSD decode steps (higher=better quality, slower) [1–50]:",
        default=cfg.lsd_decode_steps,
        min_allowed=1,
        max_allowed=50,
        validate=NumberValidator(),
    ).execute()

    noise_clamp = inquirer.number(
        message="Noise clamp (0=off, ~3.0 reduces glitches) [0.0–10.0]:",
        default=cfg.noise_clamp if cfg.noise_clamp is not None else 0.0,
        min_allowed=0.0,
        max_allowed=10.0,
        float_allowed=True,
    ).execute()
    noise_clamp_val: float | None = None if noise_clamp == 0.0 else noise_clamp

    # ---- Confirmation summary -----------------------------------------
    console.print()
    tbl = Table(title="Config Summary", show_header=False, box=None)
    tbl.add_column("Field", style="bold", width=28)
    tbl.add_column("Value")
    tbl.add_row("Workers", str(workers))
    tbl.add_row("Default output dir", default_output_dir or "(same as ebook)")
    tbl.add_row("Default voice", default_voice)
    tbl.add_row("Default chapter preset", default_chapter_preset)
    tbl.add_row("M4B bitrate", m4b_bitrate)
    tbl.add_row("Pause between lines", f"{pause_line_ms} ms")
    tbl.add_row("Pause between chapters", f"{pause_chapter_ms} ms")
    tbl.add_row("Sampling temperature", str(temp))
    tbl.add_row("LSD decode steps", str(lsd_decode_steps))
    tbl.add_row("Noise clamp", str(noise_clamp_val) if noise_clamp_val else "off")
    tbl.add_row("Save to", str(dest_path))
    console.print(tbl)
    console.print()

    confirmed = inquirer.confirm(message="Save this config?", default=True).execute()
    if not confirmed:
        console.print("Cancelled.")
        return 0

    # ---- Build and save -----------------------------------------------
    updated = AppConfig(
        name=dest_path.stem,
        workers=int(workers),
        verbose=cfg.verbose,
        log_path=cfg.log_path,
        keep_temp=cfg.keep_temp,
        m4b_bitrate=m4b_bitrate,
        pause_line_ms=int(pause_line_ms),
        pause_chapter_ms=int(pause_chapter_ms),
        temp=float(temp),
        lsd_decode_steps=int(lsd_decode_steps),
        noise_clamp=float(noise_clamp_val) if noise_clamp_val is not None else None,
        default_voice=default_voice,
        default_chapter_preset=default_chapter_preset,
        default_output_dir=Path(default_output_dir).expanduser() if default_output_dir else None,
        booknlp_model=cfg.booknlp_model,
    )

    saved_path = save_app_config(updated, dest_path)
    console.print(f"[green]Config saved to {saved_path}[/green]")
    return 0
