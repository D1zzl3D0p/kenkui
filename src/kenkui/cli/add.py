"""kenkui add — interactive wizard and headless job submission.

Entry points
------------
cmd_add(args)   kenkui add book.epub [-c config]
cmd_bare(args)  kenkui book.epub [-c config]   (bare shorthand)

Modes
-----
Interactive  book given, no -c:
    Wizard walks through chapters → narration mode → voice → output dir,
    then queues the job.

Headless  book given AND -c given:
    Loads config, submits job using defaults.
    cmd_add  → queue only, prints hint.
    cmd_bare → queue + start + live Rich progress poll.
"""

from __future__ import annotations

import random
import sys
import time
from pathlib import Path

from rich.console import Console
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn, TimeElapsedColumn
from rich.table import Table

console = Console()

# ---------------------------------------------------------------------------
# Helpers shared between wizard paths
# ---------------------------------------------------------------------------


def _get_client(args):
    from ..api_client import APIClient

    return APIClient(host=args.server_host, port=args.server_port)


def _load_config(args):
    from ..config import load_app_config

    return load_app_config(getattr(args, "config", None))


def _gender_pool(gender_str: str | None) -> str:
    """Return 'male', 'female', or 'they' from a gender_pronoun string.

    Handles values like "she/her", "he/him/his", "they/them/their".
    Splits on '/' and checks each segment to handle all three formats.
    """
    raw = (gender_str or "").strip().lower()
    if not raw:
        return "they"
    for segment in raw.split("/"):
        segment = segment.strip()
        if segment in ("he", "him", "his", "male"):
            return "male"
        if segment in ("she", "her", "hers", "female"):
            return "female"
    return "they"


def _build_voice_choices() -> list[dict]:
    """Return InquirerPy-compatible choice list for voice selection.

    Groups voices by source:
    1. Compiled voices (metadata-rich .safetensors, no HF auth needed)
    2. Built-in pocket-tts voices
    3. Custom/uncompiled voices (optional; only shown if installed)
    4. Escape hatch for raw file paths and hf:// URLs
    """
    from ..voice_registry import get_registry

    registry = get_registry()
    choices: list[dict] = []

    compiled = registry.filter(source="compiled")
    if compiled:
        choices.append({"name": "── Compiled voices ──────────────────────", "value": "__sep__", "disabled": True})
        for v in compiled:
            choices.append({"name": v.display_label, "value": v.name})

    builtins = registry.filter(source="builtin")
    if builtins:
        choices.append({"name": "── Built-in voices ──────────────────────", "value": "__sep__", "disabled": True})
        for v in builtins:
            choices.append({"name": v.display_label, "value": v.name})

    uncompiled = registry.filter(source="uncompiled")
    if uncompiled:
        choices.append({"name": "── Custom voices ────────────────────────", "value": "__sep__", "disabled": True})
        for v in uncompiled:
            choices.append({"name": v.display_label, "value": v.name})

    choices.append({"name": "Custom path or hf:// URL…", "value": "__custom__"})
    return choices


def _prompt_voice(default: str = "alba", message: str = "Select voice:") -> str:
    """Prompt the user to select a voice; returns voice string."""
    from InquirerPy import inquirer

    choices = _build_voice_choices()
    voice = inquirer.fuzzy(
        message=message,
        choices=choices,
        default=default,
        max_height="40%",
    ).execute()

    if voice == "__custom__":
        voice = (
            inquirer.text(
                message="Enter file path or hf:// URL:",
            )
            .execute()
            .strip()
        )

    return voice


def _check_hf_auth(voice: str) -> None:
    """If the voice requires HuggingFace auth, prompt for token if needed."""
    from ..helpers import get_bundled_voices
    from ..huggingface_auth import (
        AuthStatus,
        HF_TOKEN_URL,
        check_auth_status,
        do_login,
        is_custom_voice,
        open_token_page,
    )
    from InquirerPy import inquirer

    bundled = get_bundled_voices()
    if not is_custom_voice(voice, bundled):
        return

    status = check_auth_status("kyutai/pocket-tts")
    if status == AuthStatus.OK:
        return

    console.print()
    console.print("[yellow]This voice requires a free HuggingFace account.[/yellow]")
    console.print(f"  Token page: [link={HF_TOKEN_URL}]{HF_TOKEN_URL}[/link]")
    console.print()

    for attempt in range(3):
        token = inquirer.secret(message="Paste your HuggingFace token (hf_…):").execute().strip()
        ok, msg = do_login(token)
        if ok:
            console.print(f"[green]{msg}[/green]")
            return
        console.print(f"[red]{msg}[/red]")
        if attempt < 2:
            console.print("Please try again.")

    console.print("[red]Could not authenticate. Custom voices may not work.[/red]")


def _prompt_chapter_preset_and_selection(book_path: Path) -> dict:
    """Return a ChapterSelection.to_dict() based on user input."""
    from InquirerPy import inquirer

    from ..chapter_filter import ChapterFilter
    from ..models import ChapterPreset, ChapterSelection

    preset_choices = [
        {"name": "Content Only  (body chapters, skip front/back matter)", "value": "content-only"},
        {"name": "Main Chapters  (titled chapters only)", "value": "chapters-only"},
        {"name": "With Parts  (chapters + part headings)", "value": "with-parts"},
        {"name": "All  (every item in the ebook)", "value": "all"},
        {"name": "None  (skip all chapters)", "value": "none"},
    ]

    preset_val = inquirer.select(
        message="Chapter selection:",
        choices=preset_choices,
    ).execute()

    try:
        preset_enum = ChapterPreset(preset_val)
    except ValueError:
        preset_enum = ChapterPreset.CONTENT_ONLY

    # Load chapters from the ebook for finetuning.
    console.print("Loading chapters…", end=" ")
    try:
        from ..readers import get_reader

        reader = get_reader(book_path, verbose=False)
        chapters = reader.get_chapters()
        console.print(f"[green]{len(chapters)} found[/green]")
    except Exception as exc:
        console.print(f"[red]Failed to load chapters: {exc}[/red]")
        return ChapterSelection(preset=preset_enum).to_dict()

    # Determine which chapters are included by the preset (for checkbox defaults).
    if preset_val == "none":
        default_included: set[int] = set()
    else:
        filtered = ChapterFilter.apply_preset(chapters, preset_val)
        default_included = {ch.index for ch in filtered}

    # Choice values use the reader's chapter index so the returned 'included'
    # list is compatible with ChapterFilter and the worker.
    # InquirerPy checkbox uses per-choice 'enabled' for pre-selection in multiselect
    # mode — the 'default=' kwarg is silently ignored for checkboxes.
    chapter_choices = [
        {
            "name": f"[{ch.index:>3}]  {ch.title or '(untitled)'}",
            "value": ch.index,
            "enabled": ch.index in default_included,
        }
        for ch in chapters
    ]

    included = inquirer.checkbox(
        message=f"Select chapters to include (preset: {preset_val}):",
        choices=chapter_choices,
        instruction="(Space to toggle, Enter to confirm)",
    ).execute()

    return ChapterSelection(
        preset=ChapterPreset.MANUAL if set(included) != default_included else preset_enum,
        included=included,
    ).to_dict()


def _pip_install(package_spec: str) -> None:
    """Install *package_spec* using the first available package manager.

    Tries, in order:
      1. ``python -m pip``          — standard pip in any virtualenv
      2. ``uv pip install``         — uv-managed envs where pip is absent
      3. ``pip install``            — pip on PATH (pipx, conda, system)
      4. ``pip3 install``           — alternate pip entry point

    Raises ``RuntimeError`` if every strategy fails.
    """
    import shutil
    import subprocess
    import sys

    strategies: list[list[str]] = [
        [sys.executable, "-m", "pip", "install", package_spec],
    ]
    if shutil.which("uv"):
        strategies.append(["uv", "pip", "install", "--python", sys.executable, package_spec])
    if shutil.which("pip"):
        strategies.append(["pip", "install", package_spec])
    if shutil.which("pip3"):
        strategies.append(["pip3", "install", package_spec])

    last_exc: Exception | None = None
    for cmd in strategies:
        try:
            subprocess.run(cmd, check=True, capture_output=True)
            return
        except (FileNotFoundError, subprocess.CalledProcessError) as exc:
            last_exc = exc

    raise RuntimeError(
        f"Could not install '{package_spec}' — tried pip, uv, pip3. "
        f"Last error: {last_exc}"
    )


def _ensure_spacy() -> bool:
    """Download spaCy model if missing. Returns True when ready."""
    import spacy  # type: ignore[import]

    if spacy.util.is_package("en_core_web_sm"):
        return True

    console.print("[cyan]spaCy model (en_core_web_sm) not found. Downloading (~12 MB)…[/cyan]")
    _whl = (
        "https://github.com/explosion/spacy-models/releases/download/"
        "en_core_web_sm-3.8.0/en_core_web_sm-3.8.0-py3-none-any.whl"
    )
    try:
        _pip_install(_whl)
        return True
    except Exception as exc:
        console.print(f"[red]spaCy model download failed: {exc}[/red]")
        console.print("[red]Cannot proceed without spaCy model.[/red]")
        return False


def _run_nlp_analysis(book_path: Path, nlp_model: str, use_cache: bool = True):
    """Run the NLP attribution pipeline with a spinner. Returns NLPResult or None."""
    from ..nlp import cache_result, get_cached_result, run_analysis
    from ..readers import get_reader

    if use_cache:
        cached = get_cached_result(book_path)
        if cached is not None:
            console.print("[green]Using cached NLP analysis.[/green]")
            return cached
    else:
        console.print("[yellow]Cache skipped — running fresh NLP analysis…[/yellow]")

    console.print(f"[cyan]Running NLP analysis on '{book_path.name}'…[/cyan]")

    try:
        reader = get_reader(book_path, verbose=False)
        chapters = reader.get_chapters()
    except Exception as exc:
        console.print(f"[red]Could not read ebook: {exc}[/red]")
        return None

    last_msg: list[str] = []

    def _cb(msg: str) -> None:
        last_msg.append(msg)

    result = None
    with Progress(
        SpinnerColumn(),
        TextColumn("{task.description}"),
        TimeElapsedColumn(),
        console=console,
    ) as prog:
        task = prog.add_task("Analysing…", total=None)
        try:
            result = run_analysis(
                chapters=chapters,
                book_path=book_path,
                nlp_model=nlp_model,
                progress_callback=lambda msg: prog.update(task, description=msg),
            )
            prog.update(task, description="Analysis complete")
        except Exception as exc:
            console.print(f"[red]NLP analysis failed: {exc}[/red]")
            return None

    if result:
        cache_result(result, book_path)

    return result


def _top_gender_matched_voice(characters, default_voice: str) -> str:
    """Return a voice that matches the gender dominant in the top roles.

    The narrator should sound like a main character. If the top female
    character has more quotes than the top male character, default to a female
    voice; otherwise default to a male voice. Falls back to default_voice
    if no character's gender is known.
    """
    by_quotes = sorted(characters, key=lambda c: c.quote_count, reverse=True)

    top_male_quotes = 0
    top_female_quotes = 0
    for ch in by_quotes:
        g = _gender_pool(ch.gender_pronoun)
        if g == "male":
            top_male_quotes = ch.quote_count
            break
        if g == "female":
            top_female_quotes = ch.quote_count
            break

    from ..voice_registry import get_registry
    registry = get_registry()
    male_voices = [v.name for v in registry.filter(gender="Male")]
    female_voices = [v.name for v in registry.filter(gender="Female")]

    if top_female_quotes > top_male_quotes and female_voices:
        return female_voices[0]
    if male_voices:
        return male_voices[0]
    return default_voice


def _auto_assign_character_voices(characters, narrator_voice: str) -> dict[str, str]:
    """Auto-assign voices to characters.

    Phase 1: Assign known-gender characters in true round-robin order.
      - Male chars (he/him/his) → male pool (round-robin), male_idx++
      - Female chars (she/her)  → female pool (round-robin), female_idx++

    Phase 2: Assign they/them chars to the pool with fewer total quotes,
      then round-robin within that pool.
    """
    from ..voice_registry import get_registry
    registry = get_registry()
    male_voices = [v.name for v in registry.filter(gender="Male")]
    female_voices = [v.name for v in registry.filter(gender="Female")]

    # Exclude narrator voice from character pools so it stays unique to the narrator.
    male_pool = [v for v in male_voices if v != narrator_voice] or male_voices
    female_pool = [v for v in female_voices if v != narrator_voice] or female_voices

    male_idx, female_idx = 0, 0
    male_quotes, female_quotes = 0, 0
    speaker_voices: dict[str, str] = {}

    for ch in sorted(characters, key=lambda c: c.quote_count, reverse=True):
        gender = _gender_pool(ch.gender_pronoun)

        if gender == "male":
            voice = male_pool[male_idx % len(male_pool)]
            male_idx += 1
            male_quotes += ch.quote_count
        elif gender == "female":
            voice = female_pool[female_idx % len(female_pool)]
            female_idx += 1
            female_quotes += ch.quote_count
        else:
            if male_quotes <= female_quotes:
                voice = male_pool[male_idx % len(male_pool)]
                male_idx += 1
                male_quotes += ch.quote_count
            else:
                voice = female_pool[female_idx % len(female_pool)]
                female_idx += 1
                female_quotes += ch.quote_count

        speaker_voices[ch.character_id] = voice

    speaker_voices["NARRATOR"] = narrator_voice
    return speaker_voices


def _prompt_character_voice_review(
    speaker_voices: dict[str, str], characters, narrator_voice: str
) -> dict[str, str]:
    """Show a reference table, then review each character via an InquirerPy list."""
    from InquirerPy import inquirer

    # Exclude narrator voice from per-character assignment choices
    all_voice_choices = _build_voice_choices()
    voice_choices = [c for c in all_voice_choices if c.get("value") != narrator_voice]
    if not voice_choices:  # Safety: if all voices are narrator, allow all
        voice_choices = all_voice_choices

    # Print reference table — show full character IDs, no truncation.
    tbl = Table(title="Characters", show_header=True)
    tbl.add_column("Character ID", style="dim")
    tbl.add_column("Name")
    tbl.add_column("Quotes", justify="right")
    tbl.add_column("Gender")
    tbl.add_column("Voice", style="bold cyan")

    # Build list of review choices — one per character + Done.
    review_choices = []
    for ch in sorted(characters, key=lambda c: c.quote_count, reverse=True):
        current = speaker_voices.get(ch.character_id, narrator_voice)
        gender = ch.gender_pronoun or "?"
        review_choices.append(
            {
                "name": f"{ch.display_name} ({ch.quote_count} quotes, {gender})  →  {current}",
                "value": ch.character_id,
            }
        )
    review_choices.append({"name": "Done — accept assignments", "value": "__done__"})

    while True:
        chosen = inquirer.select(
            message="Select a character to re-assign (or Done to accept):",
            choices=review_choices,
            max_height="40%",
        ).execute()

        if chosen == "__done__":
            break

        new_voice = inquirer.fuzzy(
            message=f"Voice for {chosen}:",
            choices=voice_choices,
            default=speaker_voices.get(chosen, narrator_voice),
            max_height="40%",
        ).execute()

        if new_voice == "__custom__":
            new_voice = inquirer.text(message="Enter path or hf:// URL:").execute().strip()

        speaker_voices[chosen] = new_voice

        # Update the display name of the choice in the list.
        for rc in review_choices:
            if rc["value"] == chosen:
                ch = next(c for c in characters if c.character_id == chosen)
                gender = ch.gender_pronoun or "?"
                rc["name"] = (
                    f"{ch.display_name} ({ch.quote_count} quotes, {gender})  →  {new_voice}"
                )
                break

    return speaker_voices


def _prompt_multivoice_character_voices(nlp_result, default_voice: str) -> dict[str, str]:
    """Run the full multi-voice character assignment flow.

    Returns a dict mapping character_id (including "NARRATOR") to voice name.
    """
    from InquirerPy import inquirer

    characters = nlp_result.characters
    if not characters:
        console.print(
            "[yellow]No characters found by the NLP pipeline. Using narrator voice for all.[/yellow]"
        )
        narrator_voice = _prompt_voice(
            default=default_voice, message="Fallback voice for NARRATOR:"
        )
        return {"NARRATOR": narrator_voice}

    # Step A — pick NARRATOR fallback voice (defaults to top gender-matched lead).
    console.print("[bold]Select fallback voice for NARRATOR:[/bold]")
    narrator_default = _top_gender_matched_voice(characters, default_voice)
    narrator_voice = _prompt_voice(
        default=narrator_default,
        message="NARRATOR fallback voice:",
    )
    _check_hf_auth(narrator_voice)

    # Step B — choose simple or advanced assignment mode.
    assignment_mode = inquirer.select(
        message="Character voice assignment mode:",
        choices=[
            {
                "name": "Simple   — all males → one voice, all females → another",
                "value": "simple",
            },
            {
                "name": "Advanced — individual voice per character",
                "value": "advanced",
            },
        ],
    ).execute()

    if assignment_mode == "simple":
        return _prompt_simple_voice_assignment(characters, narrator_voice)

    # Advanced mode: auto-assign then review.
    # Step C — auto-assign character voices.
    speaker_voices = _auto_assign_character_voices(characters, narrator_voice)

    # Step D — review / adjust via list.
    speaker_voices = _prompt_character_voice_review(speaker_voices, characters, narrator_voice)

    return speaker_voices


# ---------------------------------------------------------------------------
# Multi-voice requirements check
# ---------------------------------------------------------------------------


def _check_multivoice_requirements(app_config) -> bool:
    """Show a requirements status table for multi-voice mode.

    Returns True if all requirements are met or the user chooses to continue
    anyway (wizard will attempt fixes inline).
    """
    from InquirerPy import inquirer
    from rich.table import Table
    from rich.text import Text

    checks: list[tuple[str, bool, str]] = []

    # Check 1: spaCy model
    try:
        import spacy
        spacy_ok = spacy.util.is_package("en_core_web_sm")
    except Exception:
        spacy_ok = False
    checks.append((
        "spaCy model (en_core_web_sm)",
        spacy_ok,
        "python -m spacy download en_core_web_sm" if not spacy_ok else "",
    ))

    # Check 2: Ollama server running
    try:
        import ollama
        ollama.list()
        ollama_ok = True
    except Exception:
        ollama_ok = False
    checks.append((
        "Ollama server",
        ollama_ok,
        "Start with: ollama serve" if not ollama_ok else "",
    ))

    # Check 3: NLP model pulled (only if Ollama is up)
    if ollama_ok:
        try:
            import ollama
            ollama.show(app_config.nlp_model)
            model_ok = True
        except Exception:
            model_ok = False
        checks.append((
            f"NLP model ({app_config.nlp_model})",
            model_ok,
            f"ollama pull {app_config.nlp_model}" if not model_ok else "",
        ))

    tbl = Table(title="Multi-Voice Requirements", show_header=True, header_style="bold")
    tbl.add_column("Requirement", min_width=30)
    tbl.add_column("Status", width=10)
    tbl.add_column("Fix", overflow="fold")
    for name, ok, fix in checks:
        status = Text("✓ Ready", style="green") if ok else Text("✗ Missing", style="red bold")
        tbl.add_row(name, status, fix)
    console.print(tbl)

    all_ok = all(ok for _, ok, _ in checks)
    if all_ok:
        return True

    proceed = inquirer.confirm(
        message="Continue anyway? (the wizard will attempt to install missing requirements)",
        default=False,
    ).execute()
    return proceed


def _prompt_simple_voice_assignment(characters, narrator_voice: str) -> dict[str, str]:
    """Simple mode: all males → one voice, all females → another.

    Returns speaker_voices dict (including NARRATOR).
    """
    from ..voice_registry import get_registry
    registry = get_registry()
    male_voices = [v.name for v in registry.filter(gender="Male")]
    female_voices = [v.name for v in registry.filter(gender="Female")]

    male_pool = [v for v in male_voices if v != narrator_voice] or male_voices
    female_pool = [v for v in female_voices if v != narrator_voice] or female_voices

    male_voice = _prompt_voice(
        default=male_pool[0] if male_pool else "alba",
        message="Voice for all male characters:",
    )
    female_voice = _prompt_voice(
        default=female_pool[0] if female_pool else "alba",
        message="Voice for all female characters:",
    )

    speaker_voices: dict[str, str] = {"NARRATOR": narrator_voice}
    for ch in characters:
        g = _gender_pool(ch.gender_pronoun)
        if g == "male":
            speaker_voices[ch.character_id] = male_voice
        elif g == "female":
            speaker_voices[ch.character_id] = female_voice
        else:
            speaker_voices[ch.character_id] = narrator_voice  # ambiguous → narrator

    return speaker_voices


def _prompt_chapter_voices(chapters, default_voice: str) -> dict[str, str]:
    """Chapter-voice mode: assign a voice per chapter.

    Returns {str(chapter_index): voice_name}.
    """
    from rich.rule import Rule

    console.print()
    console.print(Rule("[bold]Chapter Voice Assignment[/bold]"))
    console.print("[dim]Assign a voice for each chapter. Press Enter to accept the default.[/dim]")
    console.print()

    chapter_voices: dict[str, str] = {}
    for ch in chapters:
        voice = _prompt_voice(
            default=default_voice,
            message=f"Voice for '{ch.title[:50]}':",
        )
        chapter_voices[str(ch.index)] = voice

    return chapter_voices


def _prompt_quality_overrides(app_config) -> dict:
    """Optionally override TTS quality settings for this specific job.

    Returns a dict of job_* override fields (only keys where user changed the value).
    """
    from InquirerPy import inquirer

    want = inquirer.confirm(
        message="Customize audio quality for this job? (defaults come from your config)",
        default=False,
    ).execute()
    if not want:
        return {}

    overrides: dict = {}

    temp = inquirer.number(
        message=f"Temperature (0.0–1.5, current default {app_config.temp}):",
        default=app_config.temp,
        float_allowed=True,
        min_allowed=0.0,
        max_allowed=1.5,
    ).execute()
    if float(temp) != app_config.temp:
        overrides["job_temp"] = float(temp)

    lsd = inquirer.number(
        message=f"LSD decode steps (1–50, current default {app_config.lsd_decode_steps}):",
        default=app_config.lsd_decode_steps,
        min_allowed=1,
        max_allowed=50,
    ).execute()
    if int(lsd) != app_config.lsd_decode_steps:
        overrides["job_lsd_decode_steps"] = int(lsd)

    noise_default = app_config.noise_clamp or 0.0
    noise = inquirer.number(
        message=f"Noise clamp (0=off, ~3.0=reduce glitches, current default {noise_default}):",
        default=noise_default,
        float_allowed=True,
        min_allowed=0.0,
        max_allowed=10.0,
    ).execute()
    noise_val = float(noise)
    if noise_val != noise_default:
        overrides["job_noise_clamp"] = noise_val if noise_val > 0 else None

    fae = inquirer.number(
        message=f"Frames after EoS cutoff (0=suppress noise, current default {app_config.frames_after_eos}):",
        default=app_config.frames_after_eos,
        min_allowed=0,
        max_allowed=50,
    ).execute()
    if int(fae) != app_config.frames_after_eos:
        overrides["job_frames_after_eos"] = int(fae)

    bitrate_choices = [
        {"name": "64k  (small file, lower quality)", "value": "64k"},
        {"name": "96k  (default)", "value": "96k"},
        {"name": "128k", "value": "128k"},
        {"name": "192k", "value": "192k"},
        {"name": "256k  (large file, higher quality)", "value": "256k"},
    ]
    bitrate = inquirer.select(
        message=f"Output bitrate (current default {app_config.m4b_bitrate}):",
        choices=bitrate_choices,
        default=app_config.m4b_bitrate,
    ).execute()
    if bitrate != app_config.m4b_bitrate:
        overrides["job_m4b_bitrate"] = bitrate

    pause_line = inquirer.number(
        message=f"Silence between lines in ms (current default {app_config.pause_line_ms}):",
        default=app_config.pause_line_ms,
        min_allowed=0,
        max_allowed=5000,
    ).execute()
    if int(pause_line) != app_config.pause_line_ms:
        overrides["job_pause_line_ms"] = int(pause_line)

    pause_chapter = inquirer.number(
        message=f"Silence between chapters in ms (current default {app_config.pause_chapter_ms}):",
        default=app_config.pause_chapter_ms,
        min_allowed=0,
        max_allowed=30000,
    ).execute()
    if int(pause_chapter) != app_config.pause_chapter_ms:
        overrides["job_pause_chapter_ms"] = int(pause_chapter)

    return overrides


# ---------------------------------------------------------------------------
# Interactive wizard core
# ---------------------------------------------------------------------------


def _run_wizard(book_path: Path, app_config) -> dict | None:
    """Run the interactive wizard.

    Returns a kwargs dict suitable for client.add_job(), or None if the user
    cancels.
    """
    from InquirerPy import inquirer

    from ..models import NarrationMode

    console.rule(f"[bold]kenkui — {book_path.name}[/bold]")

    # Step 1 — Chapter preset / custom selection.
    chapter_selection = _prompt_chapter_preset_and_selection(book_path)

    # Step 2 — Narration mode.
    mode_val = inquirer.select(
        message="Narration mode:",
        choices=[
            {"name": "Single Voice", "value": "single"},
            {"name": "Multi-Voice  (per-character, requires local LLM)", "value": "multi"},
            {"name": "Chapter Voice  (assign a voice per chapter)", "value": "chapter"},
        ],
    ).execute()

    speaker_voices: dict[str, str] = {}
    chapter_voices: dict[str, str] = {}
    annotated_chapters_path: str | None = None
    narration_mode = mode_val

    if mode_val == "multi":
        from ..config import save_app_config, DEFAULT_CONFIG_PATH
        from ..nlp.setup import check_llm_available, run_setup_dialogue
        from ..nlp import CACHE_DIR, book_hash

        # Step 2a — Show requirements status and confirm readiness.
        console.print()
        if not _check_multivoice_requirements(app_config):
            console.print("[yellow]Falling back to single-voice mode.[/yellow]")
            narration_mode = "single"
            mode_val = "single"

        if mode_val == "multi":
            # Step 2b — Ensure spaCy model is available.
            if not _ensure_spacy():
                console.print("[red]Cannot proceed without spaCy model.[/red]")
                return None

            # Step 2c — NLP model: show current model and offer reconfigure.
            console.print()
            console.print(
                f"NLP model for speaker inference: [bold]{app_config.nlp_model}[/bold]"
            )
            reconfigure_action = inquirer.select(
                message="Continue with this model or reconfigure?",
                choices=[
                    {"name": f"Continue with {app_config.nlp_model}", "value": "continue"},
                    {"name": "Reconfigure NLP model…", "value": "reconfigure"},
                ],
            ).execute()

            if reconfigure_action == "reconfigure" or not check_llm_available(app_config):
                updated = run_setup_dialogue(app_config)
                if updated is None:
                    console.print("[yellow]Falling back to single-voice mode.[/yellow]")
                    narration_mode = "single"
                else:
                    app_config = updated
                    save_app_config(app_config, DEFAULT_CONFIG_PATH)
                    console.print(
                        f"[green]NLP model set to [bold]{app_config.nlp_model}[/bold][/green]"
                    )

        if narration_mode == "multi":
            # Step 2d — Check for a cached analysis and let the user decide.
            from ..nlp import get_cached_result

            use_cache = True
            if get_cached_result(book_path) is not None:
                use_cache = inquirer.select(
                    message="Cached NLP analysis found for this book.",
                    choices=[
                        {"name": "Use cached analysis  (fast)", "value": True},
                        {
                            "name": "Regenerate  (re-runs spaCy + LLM speaker attribution"
                                    " — takes several minutes)",
                            "value": False,
                        },
                    ],
                ).execute()

            result = _run_nlp_analysis(book_path, app_config.nlp_model, use_cache=use_cache)
            if result is None:
                console.print("[yellow]Falling back to single-voice mode.[/yellow]")
                narration_mode = "single"
            else:
                # Step 2e — Character voice assignment (simple or advanced).
                speaker_voices = _prompt_multivoice_character_voices(
                    result, app_config.default_voice
                )
                annotated_chapters_path = str(CACHE_DIR / f"{book_hash(book_path)}.json")

    elif mode_val == "chapter":
        # Chapter-voice mode: load chapters already fetched in step 1, assign per chapter.
        narration_mode = "single"  # Workers use single-voice path + chapter_voices override
        console.print()
        console.print("[bold]Voice for narrator / unassigned chapters:[/bold]")
        voice = _prompt_voice(default=app_config.default_voice, message="Default voice:")
        _check_hf_auth(voice)
        # Load chapters for assignment
        try:
            from ..readers import get_reader
            reader = get_reader(book_path, verbose=False)
            all_chapters = reader.get_chapters()
            chapter_voices = _prompt_chapter_voices(all_chapters, voice)
        except Exception as exc:
            console.print(f"[red]Could not load chapters for assignment: {exc}[/red]")
            chapter_voices = {}

    # Step 3 — Voice.
    # In single-voice mode: prompt for the narrator voice.
    # In multi-voice mode: narrator voice was already selected inside
    # _prompt_multivoice_character_voices; skip this prompt.
    # In chapter-voice mode: voice was selected above.
    if narration_mode == "multi":
        voice = speaker_voices.get("NARRATOR", app_config.default_voice)
    elif mode_val != "chapter":
        console.print()
        console.print("[bold]Voice:[/bold]")
        voice = _prompt_voice(default=app_config.default_voice, message="Select voice:")
        _check_hf_auth(voice)

    # Step 4 — Optional per-job quality overrides.
    quality_overrides = _prompt_quality_overrides(app_config)

    # Step 5 — Output directory.
    default_out = str(app_config.default_output_dir or book_path.parent)
    output_dir = (
        inquirer.text(
            message="Output directory:",
            default=default_out,
        )
        .execute()
        .strip()
        or default_out
    )

    # Step 6 — Confirmation table.
    console.print()
    summary = Table(title="Job Summary", show_header=False, box=None)
    summary.add_column("Field", style="bold", width=20)
    summary.add_column("Value")
    summary.add_row("Book", str(book_path))
    preset_label = chapter_selection.get("preset", "content-only")
    included = chapter_selection.get("included", [])
    if included:
        summary.add_row("Chapters", f"{preset_label} ({len(included)} selected)")
    else:
        summary.add_row("Chapters", preset_label)
    display_mode = mode_val if mode_val != "chapter" else "chapter-voice"
    summary.add_row("Mode", display_mode)
    summary.add_row("Narrator voice", voice)
    if narration_mode == "multi" and speaker_voices:
        non_narrator = {k: v for k, v in speaker_voices.items() if k != "NARRATOR"}
        summary.add_row("Character voices", f"{len(non_narrator)} assigned")
    if chapter_voices:
        summary.add_row("Chapter voices", f"{len(chapter_voices)} chapters assigned")
    if quality_overrides:
        summary.add_row("Quality overrides", ", ".join(
            f"{k.replace('job_', '')}={v}" for k, v in quality_overrides.items()
        ))
    summary.add_row("Output", output_dir)
    console.print(summary)
    console.print()

    confirmed = inquirer.confirm(message="Queue this job?", default=True).execute()
    if not confirmed:
        console.print("Cancelled.")
        return None

    job_kwargs = dict(
        ebook_path=str(book_path),
        voice=voice,
        chapter_selection=chapter_selection,
        output_path=output_dir,
        narration_mode=narration_mode,
        speaker_voices=speaker_voices or None,
        annotated_chapters_path=annotated_chapters_path,
        chapter_voices=chapter_voices or None,
    )
    job_kwargs.update(quality_overrides)
    return job_kwargs


# ---------------------------------------------------------------------------
# Headless submission
# ---------------------------------------------------------------------------


def _headless_submit(args, client) -> str:
    """Submit a job using config defaults. Returns the job ID."""
    from ..config import load_app_config
    from ..models import ChapterPreset, ChapterSelection

    app_config = load_app_config(args.config)

    try:
        preset_enum = ChapterPreset(app_config.default_chapter_preset)
    except ValueError:
        preset_enum = ChapterPreset.CONTENT_ONLY

    chapter_selection = ChapterSelection(preset=preset_enum).to_dict()
    output_dir = (
        str(Path(args.output).expanduser().resolve())
        if getattr(args, "output", None)
        else (
            str(app_config.default_output_dir)
            if app_config.default_output_dir
            else str(args.book.parent)
        )
    )

    client.update_config(app_config.to_dict())
    job_info = client.add_job(
        ebook_path=str(args.book),
        voice=app_config.default_voice,
        chapter_selection=chapter_selection,
        output_path=output_dir,
    )
    return job_info.id


# ---------------------------------------------------------------------------
# Rich progress poll loop (used by cmd_bare headless path)
# ---------------------------------------------------------------------------


def _poll_until_done(client, job_id: str) -> int:
    """Poll job status with a Rich progress bar. Returns exit code."""
    with Progress(
        SpinnerColumn(),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>5.1f}%"),
        TextColumn("{task.description}"),
        console=console,
    ) as prog:
        task = prog.add_task("Processing…", total=100)
        last_chapter = ""

        while True:
            time.sleep(2)
            try:
                item = client.get_job(job_id)
            except Exception:
                continue

            if item is None:
                console.print("[red]Error: job disappeared from queue.[/red]")
                return 1

            chapter = item.current_chapter or ""
            if chapter and chapter != last_chapter:
                prog.update(task, description=chapter[:60])
                last_chapter = chapter

            prog.update(task, completed=item.progress)

            if item.status == "completed":
                prog.update(task, completed=100, description="Done!")
                console.print(
                    f"\n[green]Done! Output: {item.output_path or '(see output dir)'}[/green]"
                )
                return 0
            elif item.status == "failed":
                console.print(f"\n[red]Failed: {item.error_message}[/red]")
                return 1
            elif item.status == "cancelled":
                console.print("\n[yellow]Cancelled.[/yellow]")
                return 1


# ---------------------------------------------------------------------------
# Public entry points
# ---------------------------------------------------------------------------


def cmd_add(args) -> int:
    """Handle 'kenkui add book.epub [-c config]'."""
    book_path: Path = args.book
    if not book_path.exists():
        console.print(f"[red]Error: file not found: {book_path}[/red]")
        return 1
    if book_path.suffix.lower() not in {".epub", ".mobi", ".fb2", ".azw", ".azw3", ".azw4"}:
        console.print(f"[red]Error: unrecognised ebook format: {book_path.suffix}[/red]")
        return 1

    client = _get_client(args)
    try:
        if getattr(args, "config", None):
            # Headless: queue only.
            job_id = _headless_submit(args, client)
            console.print(f"[green]Job queued: {job_id}[/green]")
            console.print("Run [bold]kenkui queue start --live[/bold] to watch progress.")
            return 0

        # Interactive wizard.
        app_config = _load_config(args)
        job_kwargs = _run_wizard(book_path, app_config)
        if job_kwargs is None:
            return 0  # User cancelled

        job_info = client.add_job(**job_kwargs)
        console.print(f"\n[green]Job queued: {job_info.id}[/green]")
        console.print("Run [bold]kenkui queue start --live[/bold] to watch progress.")
        return 0

    finally:
        client.close()


def cmd_bare(args) -> int:
    """Handle 'kenkui book.epub [-c config]' (bare shorthand).

    Interactive  (no -c):  wizard → queue → start → live dashboard.
    Headless     (-c set):  queue → start → Rich progress poll → exit 0/1.
    """
    book_path: Path = args.book
    client = _get_client(args)

    try:
        if getattr(args, "config", None):
            # Headless path.
            console.print(f"[bold]Book:[/bold]    {book_path}")
            job_id = _headless_submit(args, client)
            console.print(f"[green]Job queued: {job_id}[/green]")
            client.start_processing()
            console.print("[cyan]Processing started.[/cyan]")
            return _poll_until_done(client, job_id)

        # Interactive path.
        app_config = _load_config(args)
        job_kwargs = _run_wizard(book_path, app_config)
        if job_kwargs is None:
            return 0  # User cancelled

        job_info = client.add_job(**job_kwargs)
        console.print(f"\n[green]Job queued: {job_info.id}[/green]")
        client.start_processing()
        console.print("[cyan]Processing started. Entering live dashboard…[/cyan]\n")

        # Enter live dashboard (import here to avoid circular).
        from .queue import _live_dashboard

        return _live_dashboard(client)

    finally:
        client.close()
