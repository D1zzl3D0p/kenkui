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


# ---------------------------------------------------------------------------
# Back-navigation support
# ---------------------------------------------------------------------------


def _wizard_execute(prompt):
    """Execute an InquirerPy prompt and return its value."""
    return prompt.execute()

from rich.console import Console
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn, TimeElapsedColumn
from rich.table import Table

console = Console()


# ---------------------------------------------------------------------------
# Validator (replaces InquirerPy NumberValidator to fix cursor-positioning bug)
# ---------------------------------------------------------------------------


class _RangeValidator:
    """Validate numeric text input within optional min/max bounds.

    Duck-typed prompt_toolkit validator — no InquirerPy base class needed.
    The prompt_toolkit validate= kwarg accepts any object with .validate(document).
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

# ---------------------------------------------------------------------------
# Helpers shared between wizard paths
# ---------------------------------------------------------------------------


def _get_client(args):
    from ..api_client import APIClient

    return APIClient(host=args.server_host, port=args.server_port)


def _load_config(args):
    from ..config import load_app_config

    return load_app_config(getattr(args, "config", None))


def _build_voice_choices(client=None) -> list[dict]:
    """Return InquirerPy-compatible choice list for voice selection.

    Groups voices by source:
    1. Compiled voices (metadata-rich .safetensors, no HF auth needed)
    2. Built-in pocket-tts voices
    3. Custom/uncompiled voices (optional; only shown if installed)
    4. Escape hatch for raw file paths and hf:// URLs

    When ``client`` is provided, fetches voices via the API.
    Falls back to the local registry if ``client`` is None.
    """
    choices: list[dict] = []

    if client is not None:
        for source_key, sep_label in [
            ("compiled", "── Compiled voices ──────────────────────"),
            ("builtin", "── Built-in voices ──────────────────────"),
            ("uncompiled", "── Custom voices ────────────────────────"),
        ]:
            try:
                data = client.list_voices(source=source_key)
                voices = data.get("voices") or []
            except Exception:
                voices = []
            if voices:
                choices.append({"name": sep_label, "value": "__sep__", "disabled": True})
                for v in voices:
                    label = v.get("display_label") or v.get("name", "")
                    choices.append({"name": label, "value": v["name"]})
    else:
        from ..voice_registry import get_registry

        registry = get_registry()

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


def _prompt_voice(default: str = "alba", message: str = "Select voice:", client=None) -> str:
    """Prompt the user to select a voice; returns voice string."""
    from InquirerPy import inquirer

    choices = _build_voice_choices(client=client)
    voice = _wizard_execute(inquirer.fuzzy(
        message=message,
        choices=choices,
        default=default,
        max_height="40%",
    ))

    if voice == "__custom__":
        voice = _wizard_execute(
            inquirer.text(
                message="Enter file path or hf:// URL:",
            )
        ).strip()

    return voice


def _check_hf_auth(voice: str, args=None) -> None:
    """If the voice requires HuggingFace auth, prompt for token if needed."""
    from ..huggingface_auth import is_custom_voice
    from InquirerPy import inquirer

    if not is_custom_voice(voice):
        return

    if args is None:
        console.print("[yellow]HuggingFace auth check skipped (no server args available).[/yellow]")
        return

    HF_TOKEN_URL = "https://huggingface.co/settings/tokens"

    try:
        with _get_client(args) as client:
            status = client.get_hf_status()
            if status.get("authenticated"):
                return

            console.print()
            console.print("[yellow]This voice requires a free HuggingFace account.[/yellow]")
            console.print(f"  Token page: [link={HF_TOKEN_URL}]{HF_TOKEN_URL}[/link]")
            console.print()

            for attempt in range(3):
                token = _wizard_execute(inquirer.secret(message="Paste your HuggingFace token (hf_…):")).strip()
                result = client.login_hf(token)
                if result.get("authenticated"):
                    username = result.get("username", "")
                    console.print(f"[green]Logged in as {username}[/green]")
                    return
                err = result.get("error", "Authentication failed.")
                console.print(f"[red]{err}[/red]")
                if attempt < 2:
                    console.print("Please try again.")

        console.print("[red]Could not authenticate. Custom voices may not work.[/red]")
    except Exception as exc:
        console.print(f"[yellow]HuggingFace auth check failed ({exc}). Continuing without auth.[/yellow]")


def _prompt_chapter_preset_and_selection(book_path: Path, client=None) -> dict:
    """Return a ChapterSelection.to_dict() based on user input.

    When ``client`` is provided, parses the book and filters chapters via the
    server API.  Falls back to local ``get_reader`` / ``ChapterFilter`` when
    ``client`` is None (e.g. in tests that don't spin up a server).
    """
    from InquirerPy import inquirer

    from ..models import ChapterPreset, ChapterSelection

    preset_choices = [
        {"name": "Content Only  (body chapters, skip front/back matter)", "value": "content-only"},
        {"name": "Main Chapters  (titled chapters only)", "value": "chapters-only"},
        {"name": "With Parts  (chapters + part headings)", "value": "with-parts"},
        {"name": "All  (every item in the ebook)", "value": "all"},
        {"name": "None  (skip all chapters)", "value": "none"},
    ]

    preset_val = _wizard_execute(inquirer.select(
        message="Chapter selection:",
        choices=preset_choices,
    ))

    try:
        preset_enum = ChapterPreset(preset_val)
    except ValueError:
        preset_enum = ChapterPreset.CONTENT_ONLY

    if client is not None:
        # --- Server path ---
        console.print("Loading chapters…", end=" ")
        try:
            parse_result = client.parse_book(str(book_path))
            book_hash = parse_result.get("book_hash", "")
            chapters_raw = parse_result.get("chapters", [])
            console.print(f"[green]{len(chapters_raw)} found[/green]")
        except Exception as exc:
            import httpx as _httpx
            detail = str(exc)
            if isinstance(exc, _httpx.HTTPStatusError):
                try:
                    detail = exc.response.json().get("detail", detail)
                except Exception:
                    pass
            console.print(f"[red]Failed to load chapters: {detail}[/red]")
            return ChapterSelection(preset=preset_enum).to_dict()

        if preset_val == "none":
            default_included: set[int] = set()
        else:
            try:
                filter_result = client.filter_chapters(book_hash, {
                        "preset": preset_val,
                        "included": [],
                        "excluded": [],
                    })
                default_included = set(filter_result.get("included_indices", []))
            except Exception:
                default_included = set()

        chapter_choices = [
            {
                "name": f"[{ch.get('index', i):>3}]  {ch.get('title') or '(untitled)'}",
                "value": ch.get("index", i),
                "enabled": ch.get("index", i) in default_included,
            }
            for i, ch in enumerate(chapters_raw)
        ]

        included = _wizard_execute(inquirer.checkbox(
            message=f"Select chapters to include (preset: {preset_val}):",
            choices=chapter_choices,
            instruction="(Space to toggle, Enter to confirm)",
        ))

        return ChapterSelection(
            preset=ChapterPreset.MANUAL if set(included) != default_included else preset_enum,
            included=included,
        ).to_dict()

    else:
        # --- Local fallback path ---
        from ..chapter_filter import ChapterFilter

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
            default_included = set()
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

        included = _wizard_execute(inquirer.checkbox(
            message=f"Select chapters to include (preset: {preset_val}):",
            choices=chapter_choices,
            instruction="(Space to toggle, Enter to confirm)",
        ))

        return ChapterSelection(
            preset=ChapterPreset.MANUAL if set(included) != default_included else preset_enum,
            included=included,
        ).to_dict()




def _run_nlp_analysis(client, book_path: Path, nlp_model: str):
    """Run the NLP attribution pipeline via server task. Returns result dict or None."""
    console.print(f"[cyan]Running NLP analysis on '{book_path.name}'…[/cyan]")

    try:
        task_info = client.scan_book(str(book_path), nlp_model=nlp_model)
        task_id = task_info.get("task_id")
        if not task_id:
            console.print(f"[red]Server returned no task ID: {task_info}[/red]")
            return None
    except Exception as exc:
        console.print(f"[red]Could not start NLP analysis: {exc}[/red]")
        return None

    with Progress(
        SpinnerColumn(),
        TextColumn("{task.description}"),
        TimeElapsedColumn(),
        console=console,
    ) as prog:
        prog_task = prog.add_task("Analysing…", total=None)
        try:
            result = client.poll_task(
                task_id,
                timeout=600.0,
                progress_callback=lambda pct, msg: prog.update(prog_task, description=msg or "Analysing…"),
            )
            prog.update(prog_task, description="Analysis complete")
        except TimeoutError:
            console.print("[red]NLP analysis timed out.[/red]")
            return None
        except Exception as exc:
            console.print(f"[red]NLP analysis failed: {exc}[/red]")
            return None

    if result.get("status") == "failed":
        console.print(f"[red]NLP analysis failed: {result.get('error')}[/red]")
        return None

    return result.get("result")


def _run_fast_scan_wizard(
    client,
    book_path: Path,
    nlp_model: str,
):
    """Run Stage 1-2 fast scan via server task. Returns result dict or None."""
    console.print(f"[cyan]Scanning characters in '{book_path.name}'…[/cyan]")

    try:
        task_info = client.scan_book(str(book_path), nlp_model=nlp_model)
        task_id = task_info.get("task_id")
        if not task_id:
            console.print(f"[red]Server returned no task ID: {task_info}[/red]")
            return None
    except Exception as exc:
        console.print(f"[red]Could not start character scan: {exc}[/red]")
        return None

    with Progress(
        SpinnerColumn(),
        TextColumn("{task.description}"),
        TimeElapsedColumn(),
        console=console,
    ) as prog:
        prog_task = prog.add_task("Scanning characters…", total=None)
        try:
            result = client.poll_task(
                task_id,
                progress_callback=lambda pct, msg: prog.update(prog_task, description=msg or "Scanning characters…"),
            )
            prog.update(prog_task, description="Character scan complete")
        except Exception as exc:
            console.print(f"[red]Character scan failed: {exc}[/red]")
            return None

    if result.get("status") == "failed":
        console.print(f"[red]Character scan failed: {result.get('error')}[/red]")
        return None

    return result.get("result")


def _top_gender_matched_voice(characters, default_voice: str, client=None) -> str:
    """Return a voice that matches the gender dominant in the top roles.

    The narrator should sound like a main character. If the top female
    character has more quotes than the top male character, default to a female
    voice; otherwise default to a male voice. Falls back to default_voice
    if no character's gender is known.

    When ``client`` is provided, fetches voice lists via the API.
    """
    from ..services.voice_service import (
        build_roster_payload,
        top_gender_matched_voice as _recommend_local,
    )

    roster_payload = build_roster_payload(characters)

    if client is not None:
        try:
            result = client.recommend_narrator_voice(
                roster=roster_payload,
                default_voice=default_voice,
                excluded_voices=[],
            )
            return result.get("voice_name", default_voice)
        except Exception:
            pass

    return _recommend_local(characters, excluded=[], default_voice=default_voice)



def _auto_assign_voices(
    client,
    characters,
    narrator_voice: str,
    excluded_voices: "list[str] | None" = None,
) -> "tuple[dict[str, str], list[tuple[str, str]]]":
    """Auto-assign voices via the server suggest-cast API.

    Returns (speaker_voices, unresolved_conflicts).
    Returns ({}, []) on failure so the caller falls through to manual review.
    """
    from ..services.voice_service import build_roster_payload, suggest_cast

    roster_payload = build_roster_payload(characters)
    try:
        if client is not None:
            result = client.suggest_cast(
                roster=roster_payload,
                excluded_voices=excluded_voices or [],
                default_voice=narrator_voice,
            )
            speaker_voices = result.get("speaker_voices", {})
            warnings = result.get("warnings", [])
        else:
            local_result = suggest_cast(
                roster=characters,
                excluded_voices=excluded_voices or [],
                default_voice=narrator_voice,
            )
            speaker_voices = local_result.speaker_voices
            warnings = local_result.warnings
        for w in warnings:
            console.print(f"[yellow]Warning:[/yellow] {w}")
        return speaker_voices, []
    except Exception as exc:
        console.print(f"[red]Could not auto-assign voices: {exc}[/red]")
        return {}, []   # return empty, caller will prompt manually


def _make_character_review_label(
    ch,
    voice: str,
    pinned: "set[str]",
    series_name: "str | None" = None,
) -> str:
    """Build a display label for the character review list."""
    from ..services.voice_service import format_character_review_label

    return format_character_review_label(ch, voice, pinned=pinned, series_name=series_name)


def _prompt_character_voice_review(
    speaker_voices: dict[str, str],
    characters,
    narrator_voice: str,
    pinned: "set[str] | None" = None,
    series_name: "str | None" = None,
    unresolved_conflicts: "list[tuple[str, str]] | None" = None,
    client=None,
) -> dict[str, str]:
    """Show a reference table, then review each character via an InquirerPy list."""
    from InquirerPy import inquirer
    from ..services.voice_service import (
        annotate_voice_choices,
        build_character_review_choices,
        format_character_review_label,
        build_voice_users,
        format_unresolved_conflict_warnings,
    )

    _pinned = pinned or set()
    for warning in format_unresolved_conflict_warnings(unresolved_conflicts, _pinned):
        console.print(f"[yellow]⚠ {warning}[/yellow]")
    if unresolved_conflicts:
        console.print()

    # Exclude narrator voice from per-character assignment choices
    all_voice_choices = _build_voice_choices(client=client)
    voice_choices = [c for c in all_voice_choices if c.get("value") != narrator_voice]
    if not voice_choices:  # Safety: if all voices are narrator, allow all
        voice_choices = all_voice_choices

    # Print reference table — show full character IDs, no truncation.
    tbl = Table(title="Characters", show_header=True)
    tbl.add_column("Character ID", style="dim")
    tbl.add_column("Name")
    tbl.add_column("Mentions", justify="right")
    tbl.add_column("Gender")
    tbl.add_column("Voice", style="bold cyan")

    def _annotated_voice_choices(exclude_char_name: str | None = None) -> list[dict]:
        """Return voice choices annotated with which other chars already use each voice."""
        voice_users = build_voice_users(speaker_voices, characters)
        return annotate_voice_choices(
            voice_choices,
            voice_users,
            exclude_char_name=exclude_char_name,
        )

    # Build list of review choices — one per character (voice-first format).
    review_choices = build_character_review_choices(
        characters,
        speaker_voices,
        narrator_voice,
        pinned=_pinned,
        series_name=series_name,
    )

    while True:
        # Task 3B: confirm-based loop instead of "Done" sentinel.
        accept = _wizard_execute(inquirer.confirm(
            message="Accept these voice assignments?",
            default=True,
        ))
        if accept:
            break

        # User said No — let them pick a character to re-assign.
        chosen = _wizard_execute(inquirer.select(
            message="Select a character to re-assign:",
            choices=review_choices,
            max_height="40%",
        ))

        ch = next((c for c in characters if c.character_id == chosen), None)
        exclude_name = ch.display_name if ch else chosen

        new_voice = _wizard_execute(inquirer.fuzzy(
            message=f"Voice for {chosen}:",
            choices=_annotated_voice_choices(exclude_char_name=exclude_name),
            default=speaker_voices.get(chosen, narrator_voice),
            max_height="40%",
        ))

        if new_voice == "__custom__":
            new_voice = _wizard_execute(
                inquirer.text(message="Enter path or hf:// URL:")
            ).strip()

        speaker_voices[chosen] = new_voice

        # Update the display label in the list (voice-first format).
        for rc in review_choices:
            if rc["value"] == chosen:
                if ch is not None:
                    rc["name"] = format_character_review_label(
                        ch,
                        new_voice,
                        pinned=_pinned,
                        series_name=series_name,
                    )
                break

    return speaker_voices


def _prompt_multivoice_character_voices(
    scan_result,
    default_voice: str,
    args=None,
    client=None,
    inherited_voices: "dict[str, str] | None" = None,
    pinned: "set[str] | None" = None,
    series_name: "str | None" = None,
    excluded_voices: "list[str] | None" = None,
) -> dict[str, str]:
    """Run the full multi-voice character assignment flow.

    Returns a dict mapping character_id (including "NARRATOR") to voice name.

    When ``client`` is provided, auto-assignment is delegated to the server
    via ``suggest_cast``.  ``args`` is kept for HF auth checks.
    """
    from InquirerPy import inquirer
    from ..services.voice_service import merge_speaker_voices

    characters = scan_result.characters
    if not characters:
        console.print(
            "[yellow]No characters found by the NLP pipeline. Using narrator voice for all.[/yellow]"
        )
        narrator_voice = _prompt_voice(
            default=default_voice, message="Fallback voice for NARRATOR:", client=client
        )
        return {"NARRATOR": narrator_voice}

    # Step A — pick NARRATOR fallback voice (defaults to top gender-matched lead).
    console.print("[bold]Select fallback voice for NARRATOR:[/bold]")
    narrator_default = _top_gender_matched_voice(characters, default_voice, client=client)
    narrator_voice = _prompt_voice(
        default=narrator_default,
        message="NARRATOR fallback voice:",
        client=client,
    )
    _check_hf_auth(narrator_voice, args)

    # Step B — choose simple or advanced assignment mode.
    assignment_mode = _wizard_execute(inquirer.select(
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
    ))

    if assignment_mode == "simple":
        return _prompt_simple_voice_assignment(characters, narrator_voice, client=client)

    # Advanced mode: auto-assign then review.
    # Step C — auto-assign character voices via server suggest-cast.
    speaker_voices, unresolved_conflicts = _auto_assign_voices(
        client, characters, narrator_voice, excluded_voices=excluded_voices
    )

    # Pre-populate with inherited series voices (override auto-assigned ones)
    speaker_voices = merge_speaker_voices(speaker_voices, inherited_voices)

    # Step D — review / adjust via list.
    speaker_voices = _prompt_character_voice_review(
        speaker_voices, characters, narrator_voice,
        pinned=pinned or set(),
        series_name=series_name,
        unresolved_conflicts=unresolved_conflicts,
        client=client,
    )

    return speaker_voices


def _run_series_setup(
    fast_result=None,
    mode: str = "multi",
    prompts: "list | None" = None,
    _roster_candidates: "list | None" = None,
    client=None,
) -> "tuple":
    """Prompt for series selection and compute inherited voice assignments.

    Returns (manifest, inherited_voices, pinned):
        manifest        — loaded/created SeriesManifest, or None if skipped
        inherited_voices — char_id → voice for characters matched to series
        pinned          — set of char_ids with inherited voices

    ``prompts`` and ``_roster_candidates`` are test seams (leave as None in production).
    When ``client`` is provided the production path uses API calls; otherwise
    it falls back to local series functions.
    """
    from ..series import (
        SeriesCharacter,
        SeriesManifest,
        build_manifest_from_predecessor,
        list_roster_candidates,
        load_series,
        match_characters,
        save_series,
        slugify,
    )

    if mode != "multi":
        return None, {}, set()

    if prompts is None:
        # --- Production path: live InquirerPy prompts ---
        from InquirerPy import inquirer

        console.print()
        wants_series = _wizard_execute(inquirer.confirm(
            message="Is this book part of a series?",
            default=False,
        ))
        if not wants_series:
            return None, {}, set()

        # Fetch the series list via API if client is available, else local.
        if client is not None:
            try:
                series_data = client.list_series()
                series_entries = series_data.get("series", [])
                series_choices = [{"name": s["name"], "value": s["slug"]} for s in series_entries]
            except Exception as exc:
                console.print(f"[yellow]Could not load series list: {exc}[/yellow]")
                series_choices = []
        else:
            from ..series import list_series as _local_list_series
            series_choices = [{"name": s.name, "value": s.slug} for s in _local_list_series()]

        series_choices.append({"name": "[ + New series ]", "value": "__new__"})

        chosen_slug = _wizard_execute(inquirer.select(
            message="Select series:",
            choices=series_choices,
        ))

        if chosen_slug != "__new__":
            if client is not None:
                try:
                    series_dict = client.get_series(chosen_slug)
                    characters = [
                        SeriesCharacter(
                            canonical=c["canonical"],
                            aliases=c.get("aliases", []),
                            voice=c.get("voice", ""),
                            gender=c.get("gender", ""),
                        )
                        for c in series_dict.get("characters", [])
                    ]
                    manifest = SeriesManifest(
                        name=series_dict.get("name", chosen_slug),
                        slug=series_dict.get("slug", chosen_slug),
                        updated_at=series_dict.get("updated_at", ""),
                        characters=characters,
                    )
                except Exception as exc:
                    console.print(f"[red]Could not load series '{chosen_slug}': {exc}[/red]")
                    return None, {}, set()
            else:
                manifest = load_series(chosen_slug)
        else:
            if client is not None:
                try:
                    candidate_data = client.list_series_roster_candidates()
                    candidates = candidate_data.get("candidates", [])
                except Exception as exc:
                    console.print(f"[yellow]Could not load roster candidates: {exc}[/yellow]")
                    candidates = []
            else:
                candidates = list_roster_candidates()
            seed_idx: int | None = None
            if candidates:
                candidate_choices = [
                    {"name": "[ Fresh start — no predecessor ]", "value": -1},
                ] + [
                    {"name": c["title"] or c["hash"][:8], "value": i}
                    for i, c in enumerate(candidates)
                ]
                seed_idx = _wizard_execute(inquirer.select(
                    message="Seed from a previously processed book? (optional)",
                    choices=candidate_choices,
                ))

            series_name = _wizard_execute(inquirer.text(
                message="Series name:",
                default=candidates[seed_idx]["title"] if seed_idx is not None and seed_idx >= 0 else "",
            )).strip()

            if seed_idx is not None and seed_idx >= 0:
                if client is not None:
                    created = client.create_series_from_candidate(
                        series_name,
                        candidates[seed_idx]["roster_path"],
                    )
                    characters = [
                        SeriesCharacter(
                            canonical=c["canonical"],
                            aliases=c.get("aliases", []),
                            voice=c.get("voice", ""),
                            gender=c.get("gender", ""),
                        )
                        for c in created.get("characters", [])
                    ]
                    manifest = SeriesManifest(
                        name=created.get("name", series_name),
                        slug=created.get("slug", slugify(series_name)),
                        updated_at=created.get("updated_at", ""),
                        characters=characters,
                    )
                else:
                    manifest = build_manifest_from_predecessor(candidates[seed_idx], series_name)
            else:
                if client is not None:
                    created = client.create_empty_series(series_name)
                    manifest = SeriesManifest(
                        name=created.get("name", series_name),
                        slug=created.get("slug", slugify(series_name)),
                        updated_at=created.get("updated_at", ""),
                        characters=[],
                    )
                else:
                    manifest = SeriesManifest(
                        name=series_name,
                        slug=slugify(series_name),
                        updated_at="",
                        characters=[],
                    )

        if client is None:
            save_series(manifest)

    else:
        # --- Test seam: replay scripted prompt answers ---
        _p = iter(prompts)
        wants_series_val = next(_p, None)
        if wants_series_val != "yes":
            return None, {}, set()

        chosen = next(_p, None)
        if chosen == "new":
            candidates = _roster_candidates or list_roster_candidates()
            idx = next(_p, -1)
            series_name = next(_p, "Test Series")
            if candidates and idx >= 0:
                manifest = build_manifest_from_predecessor(candidates[idx], series_name)
            else:
                manifest = SeriesManifest(
                    name=series_name,
                    slug=slugify(series_name),
                    updated_at="",
                    characters=[],
                )
        else:
            manifest = load_series(chosen)

        if manifest is None:
            return None, {}, set()
        save_series(manifest)

    if fast_result is not None:
        if client is not None and manifest is not None and prompts is None:
            try:
                match_result = client.match_series_characters(manifest.slug, fast_result.to_dict())
                inherited_voices = match_result.get("inherited_voices", {})
                pinned = set(match_result.get("pinned", []))
            except Exception as exc:
                console.print(f"[yellow]Could not match series characters: {exc}[/yellow]")
                inherited_voices, pinned = {}, set()
        else:
            inherited_voices, pinned = match_characters(fast_result.characters, fast_result, manifest)
    else:
        inherited_voices, pinned = {}, set()
    return manifest, inherited_voices, pinned


# ---------------------------------------------------------------------------
# Multi-voice requirements check
# ---------------------------------------------------------------------------


def _check_multivoice_requirements(client) -> bool:
    """Check multivoice readiness via server and show status table.

    Returns True if all requirements are met or the user chooses to continue
    anyway.
    """
    from InquirerPy import inquirer
    from rich.table import Table
    from rich.text import Text

    try:
        status = client.get_multivoice_status()
    except Exception as exc:
        console.print(f"[red]Could not check multivoice requirements: {exc}[/red]")
        return False

    spacy_ok = status.get("spacy_ok", False)
    ollama_ok = status.get("ollama_ok", False)
    message = status.get("message", "")

    checks: list[tuple[str, bool, str]] = [
        (
            "spaCy model (en_core_web_sm)",
            spacy_ok,
            "python -m spacy download en_core_web_sm" if not spacy_ok else "",
        ),
        (
            "Ollama server",
            ollama_ok,
            "Start with: ollama serve" if not ollama_ok else "",
        ),
    ]

    tbl = Table(title="Multi-Voice Requirements", show_header=True, header_style="bold")
    tbl.add_column("Requirement", min_width=30)
    tbl.add_column("Status", width=10)
    tbl.add_column("Fix", overflow="fold")
    for name, ok, fix in checks:
        row_status = Text("✓ Ready", style="green") if ok else Text("✗ Missing", style="red bold")
        tbl.add_row(name, row_status, fix)
    console.print(tbl)

    if message:
        console.print(f"[dim]{message}[/dim]")

    all_ok = all(ok for _, ok, _ in checks)
    if all_ok:
        return True

    proceed = _wizard_execute(inquirer.confirm(
        message="Continue anyway? (the server will attempt to satisfy missing requirements)",
        default=False,
    ))
    return proceed


def _prompt_simple_voice_assignment(characters, narrator_voice: str, client=None) -> dict[str, str]:
    """Simple mode: all males → one voice, all females → another.

    Returns speaker_voices dict (including NARRATOR).

    When ``client`` is provided, fetches voice defaults via the API.
    """
    from ..services.voice_service import build_roster_payload

    if client is not None:
        try:
            male_voices = [v["name"] for v in (client.list_voices(gender="Male").get("voices") or [])]
            female_voices = [v["name"] for v in (client.list_voices(gender="Female").get("voices") or [])]
        except Exception:
            male_voices = []
            female_voices = []
    else:
        from ..voice_registry import get_registry
        registry = get_registry()
        male_voices = [v.name for v in registry.filter(gender="Male")]
        female_voices = [v.name for v in registry.filter(gender="Female")]

    male_pool = [v for v in male_voices if v != narrator_voice] or male_voices
    female_pool = [v for v in female_voices if v != narrator_voice] or female_voices

    male_voice = _prompt_voice(
        default=male_pool[0] if male_pool else "alba",
        message="Voice for all male characters:",
        client=client,
    )
    female_voice = _prompt_voice(
        default=female_pool[0] if female_pool else "alba",
        message="Voice for all female characters:",
        client=client,
    )

    roster_payload = build_roster_payload(characters)

    if client is not None:
        try:
            result = client.assign_simple_cast(
                roster=roster_payload,
                narrator_voice=narrator_voice,
                male_voice=male_voice,
                female_voice=female_voice,
            )
            return result.get("speaker_voices", {})
        except Exception as exc:
            console.print(f"[yellow]Could not assign simple cast via server: {exc}[/yellow]")

    from ..services.voice_service import assign_simple_cast as _assign_simple_cast

    return _assign_simple_cast(
        roster=characters,
        narrator_voice=narrator_voice,
        male_voice=male_voice,
        female_voice=female_voice,
    )


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

    want = _wizard_execute(inquirer.confirm(
        message="Customize audio quality for this job? (defaults come from your config)",
        default=False,
    ))
    if not want:
        return {}

    overrides: dict = {}

    temp = _wizard_execute(inquirer.text(
        message=f"Temperature (0.0–1.5, current default {app_config.temp}):",
        default=str(app_config.temp),
        validate=_RangeValidator(min_val=0.0, max_val=1.5, float_ok=True),
        filter=lambda x: float(x.strip()),
    ))
    if float(temp) != app_config.temp:
        overrides["job_temp"] = float(temp)

    lsd = _wizard_execute(inquirer.text(
        message=f"LSD decode steps (1–50, current default {app_config.lsd_decode_steps}):",
        default=str(app_config.lsd_decode_steps),
        validate=_RangeValidator(min_val=1, max_val=50),
        filter=lambda x: int(x.strip()),
    ))
    if int(lsd) != app_config.lsd_decode_steps:
        overrides["job_lsd_decode_steps"] = int(lsd)

    noise_default = app_config.noise_clamp or 0.0
    noise = _wizard_execute(inquirer.text(
        message=f"Noise clamp (0=off, ~3.0=reduce glitches, current default {noise_default}):",
        default=str(noise_default),
        validate=_RangeValidator(min_val=0.0, max_val=10.0, float_ok=True),
        filter=lambda x: float(x.strip()),
    ))
    noise_val = float(noise)
    if noise_val != noise_default:
        overrides["job_noise_clamp"] = noise_val if noise_val > 0 else None

    fae = _wizard_execute(inquirer.text(
        message=f"Frames after EoS cutoff (0=suppress noise, current default {app_config.frames_after_eos}):",
        default=str(app_config.frames_after_eos),
        validate=_RangeValidator(min_val=0, max_val=50),
        filter=lambda x: int(x.strip()),
    ))
    if int(fae) != app_config.frames_after_eos:
        overrides["job_frames_after_eos"] = int(fae)

    bitrate_choices = [
        {"name": "64k  (small file, lower quality)", "value": "64k"},
        {"name": "96k  (default)", "value": "96k"},
        {"name": "128k", "value": "128k"},
        {"name": "192k", "value": "192k"},
        {"name": "256k  (large file, higher quality)", "value": "256k"},
    ]
    bitrate = _wizard_execute(inquirer.select(
        message=f"Output bitrate (current default {app_config.m4b_bitrate}):",
        choices=bitrate_choices,
        default=app_config.m4b_bitrate,
    ))
    if bitrate != app_config.m4b_bitrate:
        overrides["job_m4b_bitrate"] = bitrate

    pause_line = _wizard_execute(inquirer.text(
        message=f"Silence between lines in ms (current default {app_config.pause_line_ms}):",
        default=str(app_config.pause_line_ms),
        validate=_RangeValidator(min_val=0, max_val=5000),
        filter=lambda x: int(x.strip()),
    ))
    if int(pause_line) != app_config.pause_line_ms:
        overrides["job_pause_line_ms"] = int(pause_line)

    pause_chapter = _wizard_execute(inquirer.text(
        message=f"Silence between chapters in ms (current default {app_config.pause_chapter_ms}):",
        default=str(app_config.pause_chapter_ms),
        validate=_RangeValidator(min_val=0, max_val=30000),
        filter=lambda x: int(x.strip()),
    ))
    if int(pause_chapter) != app_config.pause_chapter_ms:
        overrides["job_pause_chapter_ms"] = int(pause_chapter)

    return overrides


# ---------------------------------------------------------------------------
# Hub-and-spoke confirmation screen
# ---------------------------------------------------------------------------


def _init_state_from_profile(book_path, app_config, profile: dict) -> dict:
    """Build initial wizard state from last profile, falling back to app_config defaults."""
    from ..services.confirmation_service import init_confirmation_state

    return init_confirmation_state(book_path, app_config, profile)


def _state_to_profile(state: dict) -> dict:
    """Extract saveable profile keys from wizard state."""
    from ..services.confirmation_service import confirmation_state_to_profile

    return confirmation_state_to_profile(state)


def _build_confirmation_choices(state: dict, app_config) -> list:
    """Build InquirerPy select choices for the confirmation screen."""
    from InquirerPy.base.control import Choice, Separator
    from ..services.confirmation_service import summarize_confirmation_state

    summary = summarize_confirmation_state(state, app_config)

    return [
        Separator(f"  Book:      {summary['book_name']}"),
        Separator(f"  Chapters:  {summary['chapter_line']}"),
        Separator(f"  Voice/NLP: {summary['voice_line']}"),
        Separator(f"  Quality:   {summary['quality_line']}"),
        Separator(f"  Output:    {summary['output_line']}"),
        Separator(""),
        Choice(value="submit",         name="  Submit Job"),
        Choice(value="chapters",       name="  Chapters \u2192"),
        Choice(value="voice",          name="  Voice / NLP \u2192"),
        Choice(value="quality",        name="  Audio Quality \u2192"),
        Choice(value="postprocessing", name="  Post-Processing \u2192"),
        Choice(value="voices",         name="  Manage Voices \u2192"),
        Choice(value="cancel",         name="  Cancel"),
    ]


def _submenu_chapters(state: dict, app_config, client) -> dict:
    """Chapter selection submenu. Returns updated state."""
    from InquirerPy import inquirer
    from ..services.workflow_service import reset_chapter_selection

    action = _wizard_execute(inquirer.select(
        message="Chapters",
        choices=[
            {"name": f"Keep current ({state['chapter_selection'].get('preset', 'content-only')})", "value": "keep"},
            {"name": "Select chapters...", "value": "select"},
            {"name": "Reset to defaults", "value": "reset"},
            {"name": "Back", "value": "back"},
        ],
    ))
    if action == "reset":
        state = reset_chapter_selection(state, app_config)
    elif action == "select":
        book_path = state["_book_path"]
        try:
            chapter_selection = _prompt_chapter_preset_and_selection(book_path, client=client)
            state = {**state, "chapter_selection": chapter_selection}
        except (KeyboardInterrupt, EOFError):
            raise
        except Exception as exc:
            console.print(f"[yellow]Could not load chapter list: {exc}[/yellow]")
    return state


def _submenu_voice(state: dict, app_config, client) -> dict:
    """Voice/NLP submenu. Returns updated state."""
    from InquirerPy import inquirer
    from ..services.workflow_service import describe_voice_mode, reset_voice_mode

    mode_label = describe_voice_mode(state)

    action = _wizard_execute(inquirer.select(
        message="Voice / NLP",
        choices=[
            {"name": f"Keep current ({state.get('voice', '')} / {mode_label})", "value": "keep"},
            {"name": "Change narrator voice...", "value": "select"},
            {"name": "Change mode...", "value": "mode"},
            {"name": "Reset to defaults (single narrator)", "value": "reset"},
            {"name": "Back", "value": "back"},
        ],
    ))
    if action == "reset":
        state = reset_voice_mode(state, app_config)
    elif action == "select":
        voice = _prompt_voice(default=state.get("voice", app_config.default_voice),
                              message="Select narrator voice:", client=client)
        state = {**state, "voice": voice}
    elif action == "mode":
        state = _submenu_voice_mode(state, app_config, client)
    return state


def _submenu_voice_mode(state: dict, app_config, client) -> dict:
    """Mode selection for Voice / NLP submenu."""
    from InquirerPy import inquirer
    from ..services.workflow_service import set_single_voice_mode

    mode = _wizard_execute(inquirer.select(
        message="Narration mode:",
        choices=[
            {"name": "Single narrator  — one voice reads everything", "value": "single"},
            {"name": "Multi-voice (NLP) — voice per character (requires spaCy + Ollama)", "value": "multi"},
            {"name": "Chapter-voice    — assign a voice per chapter", "value": "chapter"},
        ],
    ))
    if mode == "single":
        state = set_single_voice_mode(state)
    elif mode == "multi":
        state = _setup_multi_voice(state, app_config, client)
    elif mode == "chapter":
        state = _setup_chapter_voice(state, client)
    return state


def _setup_multi_voice(state: dict, app_config, client) -> dict:
    """Run the full multi-voice NLP setup flow. Returns updated state."""
    from ..services.setup_service import cache_roster_result, parse_fast_scan_result
    from ..services.workflow_service import apply_multi_voice_setup

    if not _check_multivoice_requirements(client):
        return state

    book_path = state["_book_path"]
    nlp_model = getattr(app_config, "nlp_model", None)

    scan_raw = _run_fast_scan_wizard(client, book_path, nlp_model)
    if scan_raw is None:
        console.print("[yellow]Character scan failed; keeping current mode.[/yellow]")
        return state

    scan_result = parse_fast_scan_result(scan_raw)

    manifest, inherited_voices, pinned = _run_series_setup(
        fast_result=scan_result, mode="multi", client=client
    )

    voice = state.get("voice", getattr(app_config, "default_voice", "alba"))
    args = state.get("_args")
    series_name = manifest.name if manifest else None

    speaker_voices = _prompt_multivoice_character_voices(
        scan_result,
        default_voice=voice,
        args=args,
        client=client,
        inherited_voices=inherited_voices,
        pinned=pinned,
        series_name=series_name,
    )

    roster_cache_path = cache_roster_result(scan_result, book_path)
    if roster_cache_path is None:
        console.print("[dim]Could not cache roster[/dim]")

    return apply_multi_voice_setup(
        state,
        speaker_voices=speaker_voices,
        roster_cache_path=roster_cache_path,
        manifest=manifest,
    )


def _setup_chapter_voice(state: dict, client) -> dict:
    """Run chapter-voice assignment flow. Returns updated state."""
    from ..services.setup_service import build_chapter_prompt_items
    from ..services.workflow_service import apply_chapter_voice_setup

    book_path = state["_book_path"]
    voice = state.get("voice", "alba")

    try:
        parsed = client.parse_book(str(book_path))
    except Exception as exc:
        console.print(f"[red]Could not load chapter list: {exc}[/red]")
        return state

    chapters = build_chapter_prompt_items(parsed)
    if not chapters:
        console.print("[yellow]No chapters found.[/yellow]")
        return state

    chapter_voices = _prompt_chapter_voices(chapters, default_voice=voice)

    return apply_chapter_voice_setup(state, chapter_voices)


def _submenu_audio_quality(state: dict, app_config) -> dict:
    """Per-job audio quality overrides submenu."""
    from InquirerPy import inquirer
    from ..services.workflow_service import reset_quality_overrides

    action = _wizard_execute(inquirer.select(
        message="Audio Quality",
        choices=[
            {"name": f"Keep current ({'custom' if state.get('quality_overrides') else 'defaults'})", "value": "keep"},
            {"name": "Edit quality overrides...", "value": "edit"},
            {"name": "Reset to defaults (clear overrides)", "value": "reset"},
            {"name": "Back", "value": "back"},
        ],
    ))
    if action == "reset":
        state = reset_quality_overrides(state)
    elif action == "edit":
        overrides = dict(state.get("quality_overrides") or {})
        temp_str = _wizard_execute(inquirer.text(
            message="Temperature [0.0-1.5] (blank=inherit from config):",
            default=str(overrides.get("temp", "")),
            validate=_RangeValidator(min_val=0.0, max_val=1.5, float_ok=True),
        )).strip()
        if temp_str:
            try:
                overrides["temp"] = float(temp_str)
            except ValueError:
                pass
        steps_str = _wizard_execute(inquirer.text(
            message="Generation steps [1-50] (blank=inherit):",
            default=str(overrides.get("lsd_decode_steps", "")),
            validate=_RangeValidator(min_val=1, max_val=50, float_ok=False),
        )).strip()
        if steps_str:
            try:
                overrides["lsd_decode_steps"] = int(steps_str)
            except ValueError:
                pass
        state = {**state, "quality_overrides": overrides}
    return state


def _submenu_post_processing(state: dict, app_config) -> dict:
    """Post-processing per-job override submenu."""
    from InquirerPy import inquirer
    from ..services.workflow_service import reset_post_processing_overrides

    action = _wizard_execute(inquirer.select(
        message="Post-Processing",
        choices=[
            {"name": f"Keep current ({'custom' if state.get('pp_overrides') else 'inherit from config'})", "value": "keep"},
            {"name": "Reset to defaults (inherit from config)", "value": "reset"},
            {"name": "Back", "value": "back"},
        ],
    ))
    if action == "reset":
        state = reset_post_processing_overrides(state)
    return state


def _submenu_manage_voices(state: dict, app_config, client) -> None:
    """Voice management submenu (list/filter voices). Does not modify job state."""
    from InquirerPy import inquirer
    from .voices import _voices_interactive

    _voices_interactive(client)
    _wizard_execute(inquirer.text(message="Press Enter to return to job setup..."))


def _state_to_job_kwargs(state: dict) -> dict:
    """Convert hub-and-spoke state dict to the job kwargs expected by client.add_job()."""
    from ..services.job_service import build_job_kwargs_from_state

    job_kwargs = build_job_kwargs_from_state(state)
    # pp_overrides are tracked in state for future API support but not yet forwarded
    # to add_job (the API does not yet accept per-job post-processing overrides).
    return job_kwargs


def _run_confirmation_screen(book_path, app_config, args, client=None):
    """Hub-and-spoke confirmation screen. Returns job kwargs or None if cancelled."""
    from InquirerPy import inquirer
    from .add_profile import load_last_profile, save_last_profile

    _owns_client = client is None
    if _owns_client:
        client = _get_client(args)
    profile = load_last_profile()
    state = _init_state_from_profile(book_path, app_config, profile)

    try:
        while True:
            choices = _build_confirmation_choices(state, app_config)
            action = _wizard_execute(inquirer.select(
                message=f"kenkui \u2014 {book_path.name}",
                choices=choices,
                max_height="80%",
            ))

            if action == "submit":
                save_last_profile(_state_to_profile(state))
                return _state_to_job_kwargs(state)
            elif action == "chapters":
                state = _submenu_chapters(state, app_config, client)
            elif action == "voice":
                state = _submenu_voice(state, app_config, client)
            elif action == "quality":
                state = _submenu_audio_quality(state, app_config)
            elif action == "postprocessing":
                state = _submenu_post_processing(state, app_config)
            elif action == "voices":
                _submenu_manage_voices(state, app_config, client)
            elif action is None or action == "cancel":
                return None
    finally:
        if _owns_client:
            client.close()


# ---------------------------------------------------------------------------
# Headless submission
# ---------------------------------------------------------------------------


def _headless_submit(args, client) -> str:
    """Submit a job using config defaults. Returns the job ID."""
    from ..config import load_app_config
    from ..services.job_service import build_headless_job_kwargs

    app_config = load_app_config(args.config)

    client.update_config(app_config.to_dict())
    job_info = client.add_job(**build_headless_job_kwargs(args, app_config))
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

        # Interactive: hub-and-spoke confirmation screen.
        app_config = _load_config(args)
        job_kwargs = _run_confirmation_screen(book_path, app_config, args, client=client)
        if job_kwargs is None:
            return 0  # User cancelled

        job_info = client.add_job(**job_kwargs)
        console.print(f"\n[green]Job queued: {job_info.id}[/green]")
        console.print("Run [bold]kenkui queue start --live[/bold] to watch progress.")
        return 0

    except (KeyboardInterrupt, EOFError):
        console.print("\n[dim]Cancelled.[/dim]")
        return 0

    finally:
        client.close()


def cmd_bare(args) -> int:
    """Handle 'kenkui book.epub [-c config]' (bare shorthand).

    Interactive  (no -c):  confirmation screen → queue → start → live dashboard.
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

        # Interactive: hub-and-spoke confirmation screen.
        app_config = _load_config(args)
        job_kwargs = _run_confirmation_screen(book_path, app_config, args, client=client)
        if job_kwargs is None:
            return 0  # User cancelled

        job_info = client.add_job(**job_kwargs)
        console.print(f"\n[green]Job queued: {job_info.id}[/green]")
        client.start_processing()
        console.print("[cyan]Processing started. Entering live dashboard…[/cyan]\n")

        # Enter live dashboard (import here to avoid circular).
        from .queue import _live_dashboard

        return _live_dashboard(client)

    except (KeyboardInterrupt, EOFError):
        console.print("\n[dim]Cancelled.[/dim]")
        return 0

    finally:
        client.close()
