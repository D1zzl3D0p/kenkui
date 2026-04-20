"""kenkui config — interactive config editor (thin-client rewrite).

Usage
-----
    kenkui config
    kenkui config my-profile   # profile name stored on server (informational)

The wizard reads the current config from the kenkui server, prompts the user
to edit every field, then sends the updated config back via PUT /config.
"""

from __future__ import annotations

import multiprocessing

import httpx
from rich.console import Console
from rich.table import Table

from ..api_client import APIClient

console = Console()


def cmd_config(args) -> int:
    """Handle 'kenkui config [profile]'."""
    from InquirerPy import inquirer

    class _RangeValidator:
        """Validate numeric text input within optional min/max bounds.

        Duck-typed prompt_toolkit validator — no InquirerPy base class needed.
        """

        def __init__(self, min_val=None, max_val=None, float_ok=False):
            self._min = min_val
            self._max = max_val
            self._float_ok = float_ok

        def validate(self, document) -> None:
            from prompt_toolkit.validation import ValidationError as _PTKValidationError
            text = document.text.strip()
            try:
                val = float(text) if self._float_ok else int(text)
            except ValueError:
                raise _PTKValidationError(
                    message=f"Enter a {'decimal' if self._float_ok else 'whole'} number.",
                    cursor_position=len(document.text),
                )
            if self._min is not None and val < self._min:
                raise _PTKValidationError(
                    message=f"Minimum value is {self._min}.",
                    cursor_position=len(document.text),
                )
            if self._max is not None and val > self._max:
                raise _PTKValidationError(
                    message=f"Maximum value is {self._max}.",
                    cursor_position=len(document.text),
                )

    # ---- Fetch current config and available voices from server -----------
    try:
        with APIClient() as client:
            cfg = client.get_config()
            voices_data = client.list_voices()
    except httpx.ConnectError:
        console.print("[red]Cannot connect to kenkui server. Is it running?[/red]")
        return 1
    except Exception as exc:
        console.print(f"[red]Error fetching config: {exc}[/red]")
        return 1

    # ---- Voice choices ------------------------------------------------
    voice_choices = []
    for v in voices_data.get("voices") or []:
        label = v.get("description") or v["name"]
        voice_choices.append({"name": label, "value": v["name"]})

    # ---- Prompt each field --------------------------------------------

    workers = inquirer.text(
        message="Parallel TTS workers:",
        default=str(cfg.get("workers", max(2, multiprocessing.cpu_count() - 2))),
        validate=_RangeValidator(min_val=1, max_val=multiprocessing.cpu_count()),
        filter=lambda x: int(x.strip()),
    ).execute()

    default_output_dir = (
        inquirer.text(
            message="Default output directory (blank = same as ebook):",
            default=cfg.get("default_output_dir") or "",
        )
        .execute()
        .strip()
        or None
    )

    default_voice = inquirer.fuzzy(
        message="Default voice:",
        choices=voice_choices,
        default=cfg.get("default_voice", "alba"),
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
        default=cfg.get("default_chapter_preset", "content-only"),
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
        default=cfg.get("m4b_bitrate", "96k"),
    ).execute()

    pause_line_ms = inquirer.text(
        message="Pause between lines (ms):",
        default=str(cfg.get("pause_line_ms", 800)),
        validate=_RangeValidator(min_val=0),
        filter=lambda x: int(x.strip()),
    ).execute()

    pause_chapter_ms = inquirer.text(
        message="Pause between chapters (ms):",
        default=str(cfg.get("pause_chapter_ms", 2000)),
        validate=_RangeValidator(min_val=0),
        filter=lambda x: int(x.strip()),
    ).execute()

    temp = inquirer.text(
        message="Temperature: 0.3=conservative/robotic, 0.7=balanced, 0.9=expressive/unstable [0.0–1.5]:",
        default=str(cfg.get("temp", 0.7)),
        validate=_RangeValidator(min_val=0.0, max_val=1.5, float_ok=True),
        filter=lambda x: float(x.strip()),
    ).execute()

    lsd_decode_steps = inquirer.text(
        message="Generation steps — higher = better quality, prosody & clarity (slower) [1–50]:",
        default=str(cfg.get("lsd_decode_steps", 1)),
        validate=_RangeValidator(min_val=1, max_val=50),
        filter=lambda x: int(x.strip()),
    ).execute()

    eos_threshold = inquirer.text(
        message=(
            "EOS threshold — cut-off sensitivity. "
            "Higher (e.g. -2.0) = later cut-off, good if audio truncates. "
            "Lower (e.g. -6.0) = earlier cut-off, use if there is babbling/silence. "
            "Default -4.0 [-10.0–0.0]:"
        ),
        default=str(cfg.get("eos_threshold", -4.0)),
        validate=_RangeValidator(min_val=-10.0, max_val=0.0, float_ok=True),
        filter=lambda x: float(x.strip()),
    ).execute()

    frames_after_eos_default = cfg.get("frames_after_eos") or 0
    frames_after_eos_raw = inquirer.text(
        message="Frames after EOS (0=auto ~3–20 frames, each frame=80ms; recommended: leave as 0=auto):",
        default=str(frames_after_eos_default),
        validate=_RangeValidator(min_val=0, max_val=50),
        filter=lambda x: int(x.strip()),
    ).execute()
    frames_after_eos: int | None = None if int(frames_after_eos_raw) == 0 else int(frames_after_eos_raw)

    # Keep noise_clamp value from existing config (not exposed in wizard)
    noise_clamp_val = cfg.get("noise_clamp")

    # ---- NLP / autoprocessing -----------------------------------------
    console.print()
    console.print("[bold]NLP / Autoprocessing[/bold]  (multi-voice speaker attribution)")
    nlp_model = (
        inquirer.text(
            message="Ollama model for speaker attribution (e.g. llama3.2, phi3:mini):",
            default=cfg.get("nlp_model", "llama3.2"),
        )
        .execute()
        .strip()
        or cfg.get("nlp_model", "llama3.2")
    )
    nlp_roster_model = (
        inquirer.text(
            message="Model for character discovery / roster extraction (leave blank to use attribution model):",
            default=cfg.get("nlp_roster_model", ""),
        )
        .execute()
        .strip()
    )

    # ---- Post-processing effects chain --------------------------------
    console.print()
    console.print("[bold]Audio Post-Processing[/bold]  (broadcast-quality effects chain)")
    pp = cfg.get("post_processing") or {}

    enable_pp = inquirer.confirm(
        message="Enable audio post-processing?",
        default=pp.get("enabled", True),
    ).execute()

    if enable_pp:
        noise_reduce = inquirer.confirm(
            message="Enable noise reduction?",
            default=pp.get("noise_reduce", True),
        ).execute()

        noise_prop = pp.get("noise_reduce_prop_decrease", 0.8)
        if noise_reduce:
            noise_prop = inquirer.text(
                message="Noise reduction strength (0.0–1.0):",
                default=str(pp.get("noise_reduce_prop_decrease", 0.8)),
                validate=_RangeValidator(min_val=0.0, max_val=1.0, float_ok=True),
                filter=lambda x: float(x.strip()),
            ).execute()

        highpass_hz = inquirer.text(
            message="High-pass cutoff Hz (removes low-end rumble) [0–500]:",
            default=str(pp.get("highpass_hz", 80)),
            validate=_RangeValidator(min_val=0, max_val=500),
            filter=lambda x: int(x.strip()),
        ).execute()

        lowshelf_hz = inquirer.text(
            message="Low shelf Hz (boominess control) [50–1000]:",
            default=str(pp.get("lowshelf_hz", 250)),
            validate=_RangeValidator(min_val=50, max_val=1000),
            filter=lambda x: int(x.strip()),
        ).execute()
        lowshelf_db = inquirer.text(
            message="Low shelf gain dB (negative = cut) [-12.0–6.0]:",
            default=str(pp.get("lowshelf_db", -3.0)),
            validate=_RangeValidator(min_val=-12.0, max_val=6.0, float_ok=True),
            filter=lambda x: float(x.strip()),
        ).execute()

        presence_hz = inquirer.text(
            message="Presence boost Hz (speech clarity) [0–10000]:",
            default=str(pp.get("presence_hz", 3500)),
            validate=_RangeValidator(min_val=0, max_val=10000),
            filter=lambda x: int(x.strip()),
        ).execute()
        presence_db = inquirer.text(
            message="Presence boost dB [-6.0–9.0]:",
            default=str(pp.get("presence_db", 2.0)),
            validate=_RangeValidator(min_val=-6.0, max_val=9.0, float_ok=True),
            filter=lambda x: float(x.strip()),
        ).execute()

        deesser = inquirer.confirm(
            message="Enable de-esser (reduces harsh 'S' sounds)?",
            default=pp.get("deesser", True),
        ).execute()

        autogain = inquirer.confirm(
            message="Enable autogain (normalize each clip to a common volume level)?",
            default=pp.get("autogain", True),
        ).execute()

        autogain_target_lufs = pp.get("autogain_target_lufs", -23.0)
        if autogain:
            autogain_target_lufs = inquirer.text(
                message="Autogain target level (EBU R128 LUFS, e.g. -23.0) [-40.0 – -6.0]:",
                default=str(pp.get("autogain_target_lufs", -23.0)),
                validate=_RangeValidator(min_val=-40.0, max_val=-6.0, float_ok=True),
                filter=lambda x: float(x.strip()),
            ).execute()

        comp_thresh = inquirer.text(
            message="Compressor threshold dB [-40.0–0.0]:",
            default=str(pp.get("compressor_threshold_db", -18.0)),
            validate=_RangeValidator(min_val=-40.0, max_val=0.0, float_ok=True),
            filter=lambda x: float(x.strip()),
        ).execute()
        comp_ratio = inquirer.text(
            message="Compressor ratio (e.g. 3.0 = 3:1) [1.0–20.0]:",
            default=str(pp.get("compressor_ratio", 3.0)),
            validate=_RangeValidator(min_val=1.0, max_val=20.0, float_ok=True),
            filter=lambda x: float(x.strip()),
        ).execute()

        limiter_thresh = inquirer.text(
            message="Limiter threshold dB (safety ceiling) [-12.0–0.0]:",
            default=str(pp.get("limiter_threshold_db", -1.0)),
            validate=_RangeValidator(min_val=-12.0, max_val=0.0, float_ok=True),
            filter=lambda x: float(x.strip()),
        ).execute()

        normalize = inquirer.confirm(
            message="Normalize final output loudness? (Audible/ACX publishing)",
            default=pp.get("normalize", False),
        ).execute()

        normalize_target = pp.get("normalize_target_db", -3.0)
        normalize_lufs = pp.get("normalize_lufs")
        if normalize:
            norm_mode = inquirer.select(
                message="Normalization mode:",
                choices=[
                    {"name": "Peak  (-3.0 dBFS ceiling — good for M4B)", "value": "peak"},
                    {"name": "EBU R128  (-23 LUFS broadcast standard — Audible recommended)", "value": "ebu"},
                ],
                default="peak" if pp.get("normalize_lufs") is None else "ebu",
            ).execute()
            if norm_mode == "peak":
                normalize_target = inquirer.text(
                    message="Peak target dBFS (e.g. -3.0) [-12.0–0.0]:",
                    default=str(pp.get("normalize_target_db", -3.0)),
                    validate=_RangeValidator(min_val=-12.0, max_val=0.0, float_ok=True),
                    filter=lambda x: float(x.strip()),
                ).execute()
                normalize_lufs = None
            else:
                normalize_lufs = inquirer.text(
                    message="LUFS target (e.g. -23.0 for Audible) [-40.0 – -5.0]:",
                    default=str(pp.get("normalize_lufs") if pp.get("normalize_lufs") is not None else -23.0),
                    validate=_RangeValidator(min_val=-40.0, max_val=-5.0, float_ok=True),
                    filter=lambda x: float(x.strip()),
                ).execute()

        post_processing = {
            "enabled": True,
            "noise_reduce": noise_reduce,
            "noise_reduce_prop_decrease": float(noise_prop),
            "highpass_hz": int(highpass_hz),
            "lowshelf_hz": int(lowshelf_hz),
            "lowshelf_db": float(lowshelf_db),
            "presence_hz": int(presence_hz),
            "presence_db": float(presence_db),
            "deesser": deesser,
            "deesser_hz": pp.get("deesser_hz", 6500),
            "deesser_db": pp.get("deesser_db", -4.0),
            "autogain": autogain,
            "autogain_target_lufs": float(autogain_target_lufs),
            "compressor_threshold_db": float(comp_thresh),
            "compressor_ratio": float(comp_ratio),
            "compressor_attack_ms": pp.get("compressor_attack_ms", 5.0),
            "compressor_release_ms": pp.get("compressor_release_ms", 50.0),
            "limiter_threshold_db": float(limiter_thresh),
            "normalize": normalize,
            "normalize_target_db": normalize_target,
            "normalize_lufs": normalize_lufs,
        }
    else:
        post_processing = {**pp, "enabled": False}

    # ---- Credits chapter -----------------------------------------------
    console.print()
    console.print("[bold]Credits Chapter[/bold]  (synthesized audio appended to each audiobook)")

    credits_enabled = inquirer.confirm(
        message="Append a credits chapter at the end of each audiobook?",
        default=cfg.get("credits_enabled", True),
    ).execute()

    credits_acknowledgements = ""
    credits_license = ""
    if credits_enabled:
        credits_acknowledgements = (
            inquirer.text(
                message="Acknowledgements text (blank = none):",
                default=cfg.get("credits_acknowledgements", ""),
            )
            .execute()
            .strip()
        )
        credits_license = (
            inquirer.text(
                message="License text (blank = none):",
                default=cfg.get("credits_license", ""),
            )
            .execute()
            .strip()
        )

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
    tbl.add_row("Temperature", str(temp))
    tbl.add_row("Generation steps", str(lsd_decode_steps))
    tbl.add_row("EOS threshold", str(eos_threshold))
    tbl.add_row("Frames after EOS", "auto" if frames_after_eos is None else str(frames_after_eos))
    tbl.add_row("NLP model", nlp_model)
    tbl.add_row("NLP roster model", nlp_roster_model or f"(same as NLP model: {nlp_model})")
    tbl.add_row("Credits chapter", "on" if credits_enabled else "off")
    tbl.add_row("Post-processing", "on" if post_processing["enabled"] else "off")
    if post_processing["enabled"]:
        tbl.add_row("  Noise reduction", "on" if post_processing["noise_reduce"] else "off")
        tbl.add_row("  De-esser", "on" if post_processing["deesser"] else "off")
        tbl.add_row("  Autogain", "on" if post_processing["autogain"] else "off")
        tbl.add_row("  Normalize output", "on" if post_processing["normalize"] else "off")
    console.print(tbl)
    console.print()

    confirmed = inquirer.confirm(message="Save this config?", default=True).execute()
    if not confirmed:
        console.print("Cancelled.")
        return 0

    # ---- Build updated config dict and send to server -----------------
    updated = {
        **cfg,
        "workers": int(workers),
        "m4b_bitrate": m4b_bitrate,
        "pause_line_ms": int(pause_line_ms),
        "pause_chapter_ms": int(pause_chapter_ms),
        "temp": float(temp),
        "lsd_decode_steps": int(lsd_decode_steps),
        "noise_clamp": float(noise_clamp_val) if noise_clamp_val is not None else None,
        "eos_threshold": float(eos_threshold),
        "frames_after_eos": frames_after_eos,
        "default_voice": default_voice,
        "default_chapter_preset": default_chapter_preset,
        "default_output_dir": default_output_dir,
        "nlp_model": nlp_model,
        "nlp_roster_model": nlp_roster_model,
        "credits_enabled": credits_enabled,
        "credits_acknowledgements": credits_acknowledgements,
        "credits_license": credits_license,
        "post_processing": post_processing,
    }

    try:
        with APIClient() as client:
            client.update_config(updated)
    except httpx.ConnectError:
        console.print("[red]Cannot connect to server to save config.[/red]")
        return 1
    except Exception as exc:
        console.print(f"[red]Error saving config: {exc}[/red]")
        return 1

    console.print("[green]Config updated on server.[/green]")
    return 0
