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
    """Return 'male', 'female', or 'they' from a BookNLP gender_pronoun value.

    BookNLP returns values like "she/her", "he/him/his", "they/them/their".
    We split on '/' and check each segment as a word to handle all three formats.
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
    """Return InquirerPy-compatible choice list for voice selection."""
    from ..helpers import get_bundled_voices
    from ..utils import DEFAULT_VOICES, VOICE_DESCRIPTIONS

    choices = []
    for v in DEFAULT_VOICES:
        desc = VOICE_DESCRIPTIONS.get(v, "")
        choices.append({"name": f"{v:<20} {desc}", "value": v})

    for wav in get_bundled_voices():
        if wav.lower() == "default.txt":
            continue
        name = wav.replace(".wav", "")
        choices.append({"name": f"{name:<20} (bundled)", "value": name})

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


def _ensure_spacy(booknlp_model: str = "small") -> bool:
    """Download spaCy model if missing. Returns True when ready."""
    import spacy  # type: ignore[import]

    if spacy.util.is_package("en_core_web_sm"):
        return True

    console.print("[cyan]spaCy model (en_core_web_sm) not found. Downloading (~12 MB)…[/cyan]")
    from ..booknlp_processor import ensure_spacy_model

    success = False

    def _cb(msg: str) -> None:
        console.print(f"  {msg}")

    with Progress(SpinnerColumn(), TextColumn("{task.description}"), console=console) as prog:
        prog.add_task("Downloading spaCy model…", total=None)
        try:
            ensure_spacy_model(progress_callback=_cb)
            success = True
        except Exception as exc:
            console.print(f"[red]Download failed: {exc}[/red]")

    return success


def _run_booknlp_analysis(book_path: Path, booknlp_model: str = "small"):
    """Run BookNLP analysis with a spinner. Returns BookNLPResult or None."""
    from ..booknlp_processor import BOOKNLP_AVAILABLE, cache_result, get_cached_result, run_analysis
    from ..readers import get_reader

    if not BOOKNLP_AVAILABLE():
        console.print("[red]BookNLP is not available.[/red]")
        return None

    # Check cache first.
    cached = get_cached_result(book_path)
    if cached is not None:
        console.print("[green]Using cached BookNLP analysis.[/green]")
        return cached

    console.print(f"[cyan]Running BookNLP analysis on '{book_path.name}'…[/cyan]")

    try:
        reader = get_reader(book_path, verbose=False)
        chapters = reader.get_chapters()
    except Exception as exc:
        console.print(f"[red]Could not read ebook: {exc}[/red]")
        return None

    result = None
    status_msgs: list[str] = []

    def _cb(msg: str) -> None:
        status_msgs.append(msg)

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
                model_size=booknlp_model,
                progress_callback=_cb,
            )
            prog.update(task, description="Analysis complete")
        except Exception as exc:
            console.print(f"[red]BookNLP analysis failed: {exc}[/red]")
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
    from ..utils import MALE_VOICES, FEMALE_VOICES

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

    if top_female_quotes > top_male_quotes and FEMALE_VOICES:
        return FEMALE_VOICES[0]
    if MALE_VOICES:
        return MALE_VOICES[0]
    return default_voice


def _auto_assign_character_voices(characters, narrator_voice: str) -> dict[str, str]:
    """Auto-assign voices to characters.

    Phase 1: Assign known-gender characters in true round-robin order.
      - Male chars (he/him/his) → MALE_VOICES[male_idx % len(MALE_VOICES)], male_idx++
      - Female chars (she/her) → FEMALE_VOICES[female_idx % len(FEMALE_VOICES)], female_idx++

    Phase 2: Assign they/them chars to the pool with fewer total quotes,
      then round-robin within that pool.
    """
    from ..utils import MALE_VOICES, FEMALE_VOICES

    male_idx, female_idx = 0, 0
    male_quotes, female_quotes = 0, 0
    speaker_voices: dict[str, str] = {}

    for ch in sorted(characters, key=lambda c: c.quote_count, reverse=True):
        gender = _gender_pool(ch.gender_pronoun)

        if gender == "male":
            voice = MALE_VOICES[male_idx % len(MALE_VOICES)]
            male_idx += 1
            male_quotes += ch.quote_count
        elif gender == "female":
            voice = FEMALE_VOICES[female_idx % len(FEMALE_VOICES)]
            female_idx += 1
            female_quotes += ch.quote_count
        else:
            if male_quotes <= female_quotes:
                voice = MALE_VOICES[male_idx % len(MALE_VOICES)]
                male_idx += 1
                male_quotes += ch.quote_count
            else:
                voice = FEMALE_VOICES[female_idx % len(FEMALE_VOICES)]
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

    voice_choices = _build_voice_choices()

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


def _prompt_multivoice_character_voices(booknlp_result, default_voice: str) -> dict[str, str]:
    """Run the full multi-voice character assignment flow.

    Returns a dict mapping character_id (including "NARRATOR") to voice name.
    """
    from InquirerPy import inquirer

    characters = booknlp_result.characters
    if not characters:
        console.print(
            "[yellow]No characters found by BookNLP. Using narrator voice for all.[/yellow]"
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

    # Step B — auto-assign character voices.
    speaker_voices = _auto_assign_character_voices(characters, narrator_voice)

    # Step C — review / adjust via list.
    speaker_voices = _prompt_character_voice_review(speaker_voices, characters, narrator_voice)

    return speaker_voices


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
            {"name": "Multi-Voice  (BookNLP — per-character voices)", "value": "multi"},
        ],
    ).execute()

    speaker_voices: dict[str, str] = {}
    annotated_chapters_path: str | None = None
    narration_mode = mode_val

    if mode_val == "multi":
        # Step 2a — Ensure spaCy.
        if not _ensure_spacy(app_config.booknlp_model):
            console.print("[red]Cannot proceed without spaCy model.[/red]")
            return None

        # Step 2b — BookNLP analysis.
        result = _run_booknlp_analysis(book_path, app_config.booknlp_model)
        if result is None:
            console.print("[yellow]Falling back to single-voice mode.[/yellow]")
            narration_mode = "single"
        else:
            # Step 2c — Character voice assignment.
            speaker_voices = _prompt_multivoice_character_voices(result, app_config.default_voice)
            # Cache path for the worker (derive from CACHE_DIR + book hash).
            from ..booknlp_processor import CACHE_DIR, _book_hash

            annotated_chapters_path = str(CACHE_DIR / f"{_book_hash(book_path)}.json")

    # Step 3 — Voice.
    # In single-voice mode: prompt for the narrator voice.
    # In multi-voice mode: narrator voice was already selected inside
    # _prompt_multivoice_character_voices; skip this prompt.
    if narration_mode == "multi":
        voice = speaker_voices.get("NARRATOR", app_config.default_voice)
    else:
        console.print()
        console.print("[bold]Voice:[/bold]")
        voice = _prompt_voice(default=app_config.default_voice, message="Select voice:")
        _check_hf_auth(voice)

    # Step 4 — Output directory.
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

    # Step 5 — Confirmation table.
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
    summary.add_row("Mode", narration_mode)
    summary.add_row("Narrator voice", voice)
    if narration_mode == "multi" and speaker_voices:
        non_narrator = {k: v for k, v in speaker_voices.items() if k != "NARRATOR"}
        summary.add_row("Character voices", f"{len(non_narrator)} assigned")
    summary.add_row("Output", output_dir)
    console.print(summary)
    console.print()

    confirmed = inquirer.confirm(message="Queue this job?", default=True).execute()
    if not confirmed:
        console.print("Cancelled.")
        return None

    return dict(
        ebook_path=str(book_path),
        voice=voice,
        chapter_selection=chapter_selection,
        output_path=output_dir,
        narration_mode=narration_mode,
        speaker_voices=speaker_voices or None,
        annotated_chapters_path=annotated_chapters_path,
    )


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
