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
    by_quotes = sorted(characters, key=lambda c: c.prominence, reverse=True)

    top_male_quotes = 0
    top_female_quotes = 0
    for ch in by_quotes:
        g = _gender_pool(ch.gender_pronoun)
        if g == "male":
            top_male_quotes = ch.prominence
            break
        if g == "female":
            top_female_quotes = ch.prominence
            break

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

    if top_female_quotes > top_male_quotes and female_voices:
        return female_voices[0]
    if male_voices:
        return male_voices[0]
    return default_voice



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
    roster_payload = [
        {
            "name": c.character_id,
            "pronoun": c.gender_pronoun or None,
            "quote_count": c.quote_count,
            "mention_count": c.mention_count,
        }
        for c in characters
    ]
    try:
        result = client.suggest_cast(
            roster=roster_payload,
            excluded_voices=excluded_voices or [],
            default_voice=narrator_voice,
        )
        for w in result.get("warnings", []):
            console.print(f"[yellow]Warning:[/yellow] {w}")
        return result.get("speaker_voices", {}), []
    except Exception as exc:
        console.print(f"[red]Could not auto-assign voices: {exc}[/red]")
        return {}, []   # return empty, caller will prompt manually


def _make_character_review_label(
    ch,
    voice: str,
    pinned: "set[str]",
    series_name: "str | None" = None,
) -> str:
    """Build a display label for the character review list.

    Format: voice (20 chars)  CharName  (N mentions, gender)  [series: Name]
    """
    gender = ch.gender_pronoun or "?"
    base = f"{voice:<20}  {ch.display_name}  ({ch.prominence} mentions, {gender})"
    if ch.character_id in pinned and series_name:
        base += f"  [series: {series_name}]"
    return base


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
    from collections import defaultdict

    from InquirerPy import inquirer

    if unresolved_conflicts:
        _pinned = pinned or set()
        for char_a, char_b in unresolved_conflicts:
            if char_a in _pinned:
                inherited, other = char_a, char_b
            elif char_b in _pinned:
                inherited, other = char_b, char_a
            else:
                inherited = other = None
            if inherited:
                console.print(
                    f"[yellow]⚠ {char_a!r} and {char_b!r} share a chapter with the same voice. "
                    f"{inherited!r} is inherited from the series — change the other if needed.[/yellow]"
                )
            else:
                console.print(
                    f"[yellow]⚠ {char_a!r} and {char_b!r} share a chapter with the same voice "
                    f"and no spare voice exists.[/yellow]"
                )
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

    _pinned = pinned or set()

    def _build_voice_users() -> dict[str, list[str]]:
        """Build inverse map: voice_name → list of character display names using it."""
        users: dict[str, list[str]] = defaultdict(list)
        for cid, v in speaker_voices.items():
            if cid == "NARRATOR":
                continue
            char = next((c for c in characters if c.character_id == cid), None)
            label = char.display_name if char else cid
            users[v].append(label)
        return users

    def _annotated_voice_choices(exclude_char_name: str | None = None) -> list[dict]:
        """Return voice choices annotated with which other chars already use each voice."""
        voice_users = _build_voice_users()
        result = []
        for c in voice_choices:
            if c.get("value") == "__custom__":
                result.append(c)
                continue
            v = c["value"]
            users = [u for u in voice_users.get(v, []) if u != exclude_char_name]
            suffix = f"  ← {', '.join(users[:2])}" if users else ""
            result.append({**c, "name": c["name"] + suffix})
        return result

    # Build list of review choices — one per character (voice-first format).
    review_choices = []
    for ch in sorted(characters, key=lambda c: c.prominence, reverse=True):
        current = speaker_voices.get(ch.character_id, narrator_voice)
        review_choices.append(
            {
                "name": _make_character_review_label(ch, current, _pinned, series_name),
                "value": ch.character_id,
            }
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
                    rc["name"] = _make_character_review_label(ch, new_voice, _pinned, series_name)
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
    if inherited_voices:
        speaker_voices.update(inherited_voices)

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
                manifest = build_manifest_from_predecessor(candidates[seed_idx], series_name)
            else:
                manifest = SeriesManifest(
                    name=series_name,
                    slug=slugify(series_name),
                    updated_at="",
                    characters=[],
                )

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

    want = _wizard_execute(inquirer.confirm(
        message="Customize audio quality for this job? (defaults come from your config)",
        default=False,
    ))
    if not want:
        return {}

    overrides: dict = {}

    temp = _wizard_execute(inquirer.number(
        message=f"Temperature (0.0–1.5, current default {app_config.temp}):",
        default=app_config.temp,
        float_allowed=True,
        min_allowed=0.0,
        max_allowed=1.5,
    ))
    if float(temp) != app_config.temp:
        overrides["job_temp"] = float(temp)

    lsd = _wizard_execute(inquirer.number(
        message=f"LSD decode steps (1–50, current default {app_config.lsd_decode_steps}):",
        default=app_config.lsd_decode_steps,
        min_allowed=1,
        max_allowed=50,
    ))
    if int(lsd) != app_config.lsd_decode_steps:
        overrides["job_lsd_decode_steps"] = int(lsd)

    noise_default = app_config.noise_clamp or 0.0
    noise = _wizard_execute(inquirer.number(
        message=f"Noise clamp (0=off, ~3.0=reduce glitches, current default {noise_default}):",
        default=noise_default,
        float_allowed=True,
        min_allowed=0.0,
        max_allowed=10.0,
    ))
    noise_val = float(noise)
    if noise_val != noise_default:
        overrides["job_noise_clamp"] = noise_val if noise_val > 0 else None

    fae = _wizard_execute(inquirer.number(
        message=f"Frames after EoS cutoff (0=suppress noise, current default {app_config.frames_after_eos}):",
        default=app_config.frames_after_eos,
        min_allowed=0,
        max_allowed=50,
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

    pause_line = _wizard_execute(inquirer.number(
        message=f"Silence between lines in ms (current default {app_config.pause_line_ms}):",
        default=app_config.pause_line_ms,
        min_allowed=0,
        max_allowed=5000,
    ))
    if int(pause_line) != app_config.pause_line_ms:
        overrides["job_pause_line_ms"] = int(pause_line)

    pause_chapter = _wizard_execute(inquirer.number(
        message=f"Silence between chapters in ms (current default {app_config.pause_chapter_ms}):",
        default=app_config.pause_chapter_ms,
        min_allowed=0,
        max_allowed=30000,
    ))
    if int(pause_chapter) != app_config.pause_chapter_ms:
        overrides["job_pause_chapter_ms"] = int(pause_chapter)

    return overrides


# ---------------------------------------------------------------------------
# Interactive wizard core
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Wizard step functions (each accepts state dict, returns updated state dict)
# ---------------------------------------------------------------------------


def _step_chapters(state: dict) -> dict:
    """Step 1: chapter preset + selection."""
    book_path: Path = state["_book_path"]
    args = state.get("_args")
    if args is not None:
        with _get_client(args) as client:
            chapter_selection = _prompt_chapter_preset_and_selection(book_path, client=client)
    else:
        chapter_selection = _prompt_chapter_preset_and_selection(book_path)
    return {**state, "chapter_selection": chapter_selection}


def _step_mode(state: dict) -> dict:
    """Step 2: narration mode select."""
    from InquirerPy import inquirer

    mode_val = _wizard_execute(inquirer.select(
        message="Narration mode:",
        choices=[
            {"name": "Single Voice", "value": "single"},
            {"name": "Multi-Voice  (per-character, requires local LLM)", "value": "multi"},
            {"name": "Chapter Voice  (assign a voice per chapter)", "value": "chapter"},
        ],
    ))
    return {**state, "mode": mode_val}


def _step_quality(state: dict) -> dict:
    """Step 3: quality overrides."""
    app_config = state["_app_config"]
    quality_overrides = _prompt_quality_overrides(app_config)
    return {**state, "quality_overrides": quality_overrides}


def _step_output_dir(state: dict) -> dict:
    """Step 4: output directory."""
    from InquirerPy import inquirer

    app_config = state["_app_config"]
    book_path: Path = state["_book_path"]
    default_out = str(app_config.default_output_dir or book_path.parent)
    output_dir = (
        _wizard_execute(inquirer.text(
            message="Output directory:",
            default=default_out,
        )).strip()
        or default_out
    )
    return {**state, "output_dir": output_dir}


def _step_voice_setup(state: dict) -> dict:
    """Step 5: mode-specific voice setup (multi / chapter / single)."""
    from InquirerPy import inquirer

    mode_val: str = state["mode"]
    app_config = state["_app_config"]
    book_path: Path = state["_book_path"]
    chapter_selection: dict = state["chapter_selection"]
    args = state["_args"]

    voice: str = app_config.default_voice
    speaker_voices: dict[str, str] = {}
    chapter_voices: dict[str, str] = {}
    roster_cache_path: str | None = None
    narration_mode = mode_val
    fast_result = None
    series_manifest = None
    inherited_voices: dict[str, str] = {}
    pinned: set[str] = set()

    if mode_val == "multi":
        from ..config import save_app_config, DEFAULT_CONFIG_PATH
        from ..nlp.setup import check_llm_available, run_setup_dialogue

        console.print()
        with _get_client(args) as client:
            if not _check_multivoice_requirements(client):
                console.print("[yellow]Falling back to single-voice mode.[/yellow]")
                narration_mode = "single"
                mode_val = "single"

            if mode_val == "multi":
                console.print()
                console.print(
                    f"NLP model for speaker inference: [bold]{app_config.nlp_model}[/bold]"
                )
                reconfigure_action = _wizard_execute(inquirer.select(
                    message="Continue with this model or reconfigure?",
                    choices=[
                        {"name": f"Continue with {app_config.nlp_model}", "value": "continue"},
                        {"name": "Reconfigure NLP model…", "value": "reconfigure"},
                    ],
                ))

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
                # Pick narrator fallback voice upfront; characters are assigned
                # automatically during processing (deferred cast assignment).
                console.print()
                console.print("[bold]Select fallback voice for NARRATOR:[/bold]")
                narrator_voice = _prompt_voice(
                    default=app_config.default_voice,
                    message="NARRATOR fallback voice:",
                    client=client,
                )
                _check_hf_auth(narrator_voice, args)
                speaker_voices = {"NARRATOR": narrator_voice}

                series_manifest, _, _ = _run_series_setup(
                    fast_result=None,
                    mode="multi",
                    client=client,
                )

    elif mode_val == "chapter":
        narration_mode = "single"
        console.print()
        console.print("[bold]Voice for narrator / unassigned chapters:[/bold]")
        with _get_client(args) as _chap_client:
            voice = _prompt_voice(default=app_config.default_voice, message="Default voice:", client=_chap_client)
            _check_hf_auth(voice, args)
            try:
                parse_result = _chap_client.parse_book(str(book_path))
                from ..models import Chapter
                all_chapters = [
                    Chapter(index=ch.get("index", i), title=ch.get("title", ""), paragraphs=[])
                    for i, ch in enumerate(parse_result.get("chapters", []))
                ]
                chapter_voices = _prompt_chapter_voices(all_chapters, voice)
            except Exception as exc:
                console.print(f"[red]Could not parse book for chapter assignment: {exc}[/red]")
                chapter_voices = {}

    return {
        **state,
        "_app_config": app_config,  # may have been updated during multi setup
        "voice": voice,
        "speaker_voices": speaker_voices,
        "chapter_voices": chapter_voices,
        "roster_cache_path": roster_cache_path,
        "narration_mode": narration_mode,
        "mode": mode_val,
        "series_slug": series_manifest.slug if series_manifest is not None else None,
        "_series_manifest": series_manifest if narration_mode == "multi" else None,
    }


def _step_narrator_voice(state: dict) -> dict:
    """Step 6: narrator voice for single mode (no-op for multi/chapter)."""
    narration_mode: str = state["narration_mode"]
    mode_val: str = state["mode"]
    app_config = state["_app_config"]
    speaker_voices: dict = state["speaker_voices"]
    voice: str = state["voice"]
    args = state["_args"]

    if narration_mode == "multi":
        voice = speaker_voices.get("NARRATOR", app_config.default_voice)
    elif mode_val != "chapter":
        console.print()
        console.print("[bold]Voice:[/bold]")
        with _get_client(args) as _voice_client:
            voice = _prompt_voice(default=app_config.default_voice, message="Select voice:", client=_voice_client)
        _check_hf_auth(voice, args)

    return {**state, "voice": voice}


def _step_confirm(state: dict) -> dict:
    """Step 7: confirmation table + submit."""
    from InquirerPy import inquirer

    book_path: Path = state["_book_path"]
    chapter_selection: dict = state["chapter_selection"]
    mode_val: str = state["mode"]
    narration_mode: str = state["narration_mode"]
    voice: str = state["voice"]
    speaker_voices: dict = state["speaker_voices"]
    chapter_voices: dict = state["chapter_voices"]
    quality_overrides: dict = state["quality_overrides"]
    output_dir: str = state["output_dir"]
    roster_cache_path = state["roster_cache_path"]
    series_slug = state.get("series_slug")

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
    if narration_mode == "multi":
        non_narrator = {k: v for k, v in speaker_voices.items() if k != "NARRATOR"}
        if non_narrator:
            summary.add_row("Character voices", f"{len(non_narrator)} assigned")
        else:
            summary.add_row("Character voices", "Auto-assign during processing")
        if series_slug:
            series_manifest = state.get("_series_manifest")
            series_label = series_manifest.name if series_manifest else series_slug
            summary.add_row("Series", series_label)
    if chapter_voices:
        summary.add_row("Chapter voices", f"{len(chapter_voices)} chapters assigned")
    if quality_overrides:
        summary.add_row("Quality overrides", ", ".join(
            f"{k.replace('job_', '')}={v}" for k, v in quality_overrides.items()
        ))
    summary.add_row("Output", output_dir)
    console.print(summary)
    console.print()

    confirmed = _wizard_execute(inquirer.confirm(message="Queue this job?", default=True))
    if not confirmed:
        console.print("Cancelled.")
        return {**state, "_cancelled": True}

    job_kwargs = dict(
        ebook_path=str(book_path),
        voice=voice,
        chapter_selection=chapter_selection,
        output_path=output_dir,
        narration_mode=narration_mode,
        speaker_voices=speaker_voices or None,
        annotated_chapters_path=None,
        roster_cache_path=roster_cache_path,
        chapter_voices=chapter_voices or None,
        series_slug=series_slug,
        **quality_overrides,
    )
    return {**state, "_job_kwargs": job_kwargs}


def _run_wizard(book_path: Path, app_config, args) -> dict | None:
    """Run the interactive wizard using a step-stack state machine.

    Returns a kwargs dict suitable for client.add_job(), or None if the user
    cancels.
    """
    console.rule(f"[bold]kenkui — {book_path.name}[/bold]")

    state: dict = {
        "_book_path": book_path,
        "_app_config": app_config,
        "_args": args,
        # Defaults for step outputs
        "voice": app_config.default_voice,
        "speaker_voices": {},
        "chapter_voices": {},
        "roster_cache_path": None,
        "quality_overrides": {},
        "series_slug": None,
        "_series_manifest": None,
    }

    steps = [
        _step_chapters,
        _step_mode,
        _step_quality,
        _step_output_dir,
        _step_voice_setup,
        _step_narrator_voice,
        _step_confirm,
    ]

    i = 0
    while i < len(steps):
        state = steps[i](state)
        if state.get("_abort"):
            return None
        if state.get("_cancelled"):
            return None
        if "_job_kwargs" in state:
            return state["_job_kwargs"]
        i += 1

    return None


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
        job_kwargs = _run_wizard(book_path, app_config, args)
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
        job_kwargs = _run_wizard(book_path, app_config, args)
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
