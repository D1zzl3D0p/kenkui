"""CLI subcommand: ``kenkui voices`` — list and manage available voices."""

from __future__ import annotations

import os
import sys
from pathlib import Path

from rich.console import Console
from rich.table import Table

from ..voice_registry import get_registry
from ..config import load_app_config, save_app_config, DEFAULT_CONFIG_PATH
from ..queue import QueueManager
from ..workers import _get_or_load_model, _render_text
from ..voice_loader import load_voice

console = Console()


def cmd_voices_list(args) -> None:
    """List all available voices with metadata.

    Accepts optional filter flags: ``--gender``, ``--accent``, ``--dataset``, ``--source``.
    """
    from ..voice_registry import get_registry

    registry = get_registry()
    voices = registry.filter(
        gender=getattr(args, "gender", None),
        accent=getattr(args, "accent", None),
        dataset=getattr(args, "dataset", None),
        source=getattr(args, "source", None),
    )

    if not voices:
        console.print("[yellow]No voices match the specified filters.[/yellow]")
        return

    table = Table(title="Available Voices", show_header=True, header_style="bold cyan")
    table.add_column("Name", style="bold", min_width=20)
    table.add_column("Source", min_width=12)
    table.add_column("Gender", min_width=8)
    table.add_column("Accent", min_width=22)
    table.add_column("Dataset", min_width=6)

    source_style = {
        "compiled": "green",
        "builtin": "blue",
        "uncompiled": "yellow",
    }

    for v in voices:
        style = source_style.get(v.source, "")
        table.add_row(
            v.name,
            f"[{style}]{v.source}[/{style}]" if style else v.source,
            v.gender or "—",
            v.accent or "—",
            v.dataset or "—",
        )

    console.print(table)
    console.print(f"\n[dim]{len(voices)} voice(s) listed.[/dim]")


def cmd_voices_fetch(args) -> None:
    """Download uncompiled custom voices from a HuggingFace repo to the user data directory."""
    # Guard: check huggingface_hub is available
    try:
        import huggingface_hub  # noqa: F401
    except ImportError:
        console.print(
            "[red]Custom voices extra is not installed.[/red]\n"
            "Run: [bold]pip install kenkui\\[custom-voices][/bold]"
        )
        sys.exit(1)

    from ..huggingface_auth import AuthStatus, check_auth_status, do_login, open_token_page, HF_TOKEN_URL
    from ..voice_registry import get_registry
    from InquirerPy import inquirer
    from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn

    # ── Auth check ────────────────────────────────────────────────────────
    status = check_auth_status("kyutai/pocket-tts")
    if status != AuthStatus.OK:
        console.print()
        console.print("[yellow]HuggingFace authentication required for custom voices.[/yellow]")
        console.print(f"  Token page: [link={HF_TOKEN_URL}]{HF_TOKEN_URL}[/link]")
        console.print()

        for attempt in range(3):
            token = inquirer.secret(message="Paste your HuggingFace token (hf_…):").execute().strip()
            ok, msg = do_login(token)
            console.print(f"  {msg}")
            if ok:
                break
            if attempt < 2:
                console.print("  [red]Try again.[/red]")
        else:
            console.print("[red]Authentication failed — cannot download custom voices.[/red]")
            sys.exit(1)

    # ── Determine HF repo and destination ─────────────────────────────────
    hf_repo = getattr(args, "repo", None) or os.environ.get("KENKUI_VOICES_REPO", "")
    if not hf_repo:
        console.print(
            "[red]No HuggingFace repo specified.[/red]\n"
            "Set KENKUI_VOICES_REPO env var or pass --repo <user/repo-name>."
        )
        sys.exit(1)

    xdg_data = os.environ.get("XDG_DATA_HOME", str(Path.home() / ".local" / "share"))
    dest_dir = Path(xdg_data) / "kenkui" / "voices" / "uncompiled"
    dest_dir.mkdir(parents=True, exist_ok=True)

    # ── Download ──────────────────────────────────────────────────────────
    try:
        from huggingface_hub import HfApi, hf_hub_download

        api = HfApi()
        # list_repo_files returns list[str] (plain filenames) in huggingface_hub >= 0.14
        repo_files = [f for f in api.list_repo_files(hf_repo) if f.endswith(".wav")]
    except Exception as exc:
        console.print(f"[red]Could not list files in repo {hf_repo!r}: {exc}[/red]")
        sys.exit(1)

    if not repo_files:
        console.print(f"[yellow]No .wav files found in {hf_repo!r}.[/yellow]")
        sys.exit(1)

    console.print(f"Downloading {len(repo_files)} voice file(s) to [bold]{dest_dir}[/bold] …")

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Downloading…", total=len(repo_files))
        failed = []
        for filename in repo_files:
            try:
                local_path = hf_hub_download(
                    repo_id=hf_repo,
                    filename=filename,
                    local_dir=str(dest_dir),
                )
                progress.advance(task)
                progress.update(task, description=f"Downloaded {Path(local_path).name}")
            except Exception as exc:
                failed.append((filename, str(exc)))
                progress.advance(task)

    if failed:
        console.print(f"[red]{len(failed)} file(s) failed to download:[/red]")
        for fname, err in failed:
            console.print(f"  [dim]{fname}:[/dim] {err}")

    # Invalidate registry so new voices appear immediately
    get_registry().invalidate()

    downloaded = len(repo_files) - len(failed)
    console.print(f"[green]Done! {downloaded} voice(s) downloaded to {dest_dir}[/green]")
    console.print("Run [bold]kenkui voices list[/bold] to see the new voices.")


def cmd_voices_download(args) -> int:
    """Download compiled voices from HuggingFace to the user data directory."""
    from ..voices.download import download_voices, voices_are_present

    force = getattr(args, "force", False)
    if voices_are_present() and not force:
        console.print("[green]Voices already present. Use --force to re-download.[/green]")
        return 0
    console.print("Downloading voices from HuggingFace…")
    download_voices(force=force)
    console.print("[green]Done.[/green]")
    return 0


def cmd_voices_exclude(args) -> None:
    """Add a voice to the global excluded-from-auto-assignment list."""
    voice_name: str = args.voice
    registry = get_registry()

    if registry.resolve(voice_name) is None:
        console.print(f"[yellow]Warning: '{voice_name}' is not in the voice registry. "
                      f"Excluding anyway.[/yellow]")

    config = load_app_config()
    if voice_name in config.excluded_voices:
        console.print(f"[yellow]'{voice_name}' is already excluded.[/yellow]")
        return

    config.excluded_voices = list(config.excluded_voices) + [voice_name]

    # Warn if excluding this voice empties a gender pool
    male_names = {v.name for v in registry.filter(gender="Male")}
    female_names = {v.name for v in registry.filter(gender="Female")}
    excluded_set = set(config.excluded_voices)
    if male_names and male_names <= excluded_set:
        console.print("[yellow]Warning: all Male voices are now excluded. "
                      "Auto-assignment will fall back to the full pool.[/yellow]")
    if female_names and female_names <= excluded_set:
        console.print("[yellow]Warning: all Female voices are now excluded. "
                      "Auto-assignment will fall back to the full pool.[/yellow]")

    save_app_config(config, DEFAULT_CONFIG_PATH)
    console.print(f"[green]'{voice_name}' excluded from auto-assignment pool.[/green]")


def cmd_voices_include(args) -> None:
    """Remove a voice from the excluded list, restoring it to auto-assignment."""
    voice_name: str = args.voice
    config = load_app_config()

    if voice_name not in config.excluded_voices:
        console.print(f"[yellow]'{voice_name}' is not in the excluded list.[/yellow]")
        return

    config.excluded_voices = [v for v in config.excluded_voices if v != voice_name]
    save_app_config(config, DEFAULT_CONFIG_PATH)
    console.print(f"[green]'{voice_name}' restored to auto-assignment pool.[/green]")


def _sort_cast(speaker_voices: dict) -> list:
    """Sort cast alphabetically with NARRATOR pinned last."""
    return sorted(
        speaker_voices.items(),
        key=lambda kv: ("~" if kv[0] == "NARRATOR" else kv[0].lower()),
    )


def cmd_voices_cast(args) -> None:
    """Display the character→voice cast for a completed multi-voice book."""
    import difflib
    from ..models import NarrationMode

    title_query: str = args.title
    qm = QueueManager()

    candidates = [
        item for item in qm.completed_items
        if item.job.narration_mode == NarrationMode.MULTI
        and item.job.speaker_voices
    ]

    if not candidates:
        console.print("[yellow]No completed multi-voice jobs found in queue.[/yellow]")
        return

    job_names = [item.job.name for item in candidates]

    # Case-insensitive fuzzy match
    job_names_lower = [n.lower() for n in job_names]
    close_lower = difflib.get_close_matches(title_query.lower(), job_names_lower, n=5, cutoff=0.4)
    close = [job_names[job_names_lower.index(n)] for n in close_lower]
    # Substring fallback
    substring = [n for n in job_names if title_query.lower() in n.lower()]
    matched_names = close or substring

    if not matched_names:
        console.print(f"[yellow]No jobs matching '{title_query}'.[/yellow]")
        if job_names:
            console.print("Available: " + ", ".join(job_names[:8]))
        return

    matched_items = [item for item in candidates if item.job.name in matched_names]

    if len(matched_items) > 1:
        console.print(f"[yellow]Multiple jobs match '{title_query}':[/yellow]")
        for item in matched_items:
            console.print(f"  • {item.job.name}  [dim](id: {item.id})[/dim]")
        console.print("Re-run with a more specific title.")
        return

    item = matched_items[0]
    table = Table(title=f"Cast — {item.job.name}", show_header=True, box=None)
    table.add_column("Character", style="bold", min_width=30)
    table.add_column("Voice", min_width=20)

    for char_id, voice_name in _sort_cast(item.job.speaker_voices):
        display = "Narrator" if char_id == "NARRATOR" else char_id
        table.add_row(display, voice_name)

    console.print(table)
    if item.output_path:
        console.print(f"\n[dim]Job ID: {item.id} | Output: {item.output_path}[/dim]")


DEFAULT_AUDITION_TEXT = (
    "The rain in Spain stays mainly in the plain. "
    "How wonderful it is to simply speak and be heard."
)


def _player_command() -> str:
    """Return the system audio player command for the current platform."""
    return "open" if sys.platform == "darwin" else "xdg-open"


def _preview_path(voice_name: str) -> Path:
    """Return the temp path for an audition preview file."""
    return Path(f"/tmp/kenkui-{voice_name}-preview.wav")


def cmd_voices_audition(args) -> None:
    """Synthesize a short voice preview and open it in the system audio player."""
    import subprocess

    voice_name: str = args.voice
    text: str = getattr(args, "text", None) or DEFAULT_AUDITION_TEXT
    out_path = _preview_path(voice_name)

    # Validate voice exists (warn only — don't block)
    if get_registry().resolve(voice_name) is None:
        console.print(
            f"[yellow]Warning: '{voice_name}' not found in registry. Attempting anyway.[/yellow]"
        )

    console.print(
        "[yellow]Loading TTS model — this may take 10–30 seconds on first use…[/yellow]"
    )

    config = load_app_config()
    try:
        model = _get_or_load_model(
            config.temp,
            config.lsd_decode_steps,
            config.noise_clamp,
            config.eos_threshold,
        )
    except Exception as exc:
        console.print(f"[red]Failed to load TTS model: {exc}[/red]")
        sys.exit(1)

    try:
        voice_resolved = load_voice(voice_name)
        voice_state = model.get_state_for_audio_prompt(voice_resolved)
    except Exception as exc:
        console.print(f"[red]Failed to load voice '{voice_name}': {exc}[/red]")
        sys.exit(1)

    preview_text = text[:60] + ("…" if len(text) > 60 else "")
    console.print(
        f"Synthesizing preview for [bold]{voice_name}[/bold]: [dim]\"{preview_text}\"[/dim]"
    )

    seg = _render_text(
        model, voice_state, text,
        log_message=lambda _: None,
        pid=0, batch_idx=0, total_batches=1,
        frames_after_eos=0,
    )
    if seg is None:
        console.print("[red]Synthesis returned no audio.[/red]")
        sys.exit(1)

    seg.export(str(out_path), format="wav")
    console.print(f"[green]Preview saved to {out_path}[/green]")

    try:
        subprocess.Popen([_player_command(), str(out_path)])
    except Exception as exc:
        console.print(
            f"[yellow]Could not open system player ({exc}). Play manually: {out_path}[/yellow]"
        )


__all__ = [
    "cmd_voices_list",
    "cmd_voices_fetch",
    "cmd_voices_download",
    "cmd_voices_exclude",
    "cmd_voices_include",
    "cmd_voices_cast",
    "cmd_voices_audition",
    "cmd_voices_tui",
    "DEFAULT_AUDITION_TEXT",
    "_sort_cast",
    "_player_command",
    "_preview_path",
    "_tui_execute",
]


# ---------------------------------------------------------------------------
# TUI — interactive voice manager
# ---------------------------------------------------------------------------


def _tui_execute(prompt):
    """Execute an InquirerPy prompt. Extracted for test monkeypatching."""
    return prompt.execute()


def _do_tui_audition(voice_name: str, config) -> None:
    """Synthesize a preview from within the TUI. Returns on failure (no sys.exit)."""
    import subprocess

    out_path = _preview_path(voice_name)
    console.print("\n[yellow]Loading TTS model — may take 10–30 s on first use…[/yellow]")

    try:
        model = _get_or_load_model(
            config.temp, config.lsd_decode_steps, config.noise_clamp, config.eos_threshold,
        )
    except Exception as exc:
        console.print(f"[red]Failed to load TTS model: {exc}[/red]")
        return

    try:
        voice_resolved = load_voice(voice_name)
        voice_state = model.get_state_for_audio_prompt(voice_resolved)
    except Exception as exc:
        console.print(f"[red]Failed to load voice '{voice_name}': {exc}[/red]")
        return

    console.print(f"Synthesizing [bold]{voice_name}[/bold]…")
    seg = _render_text(
        model, voice_state, DEFAULT_AUDITION_TEXT,
        log_message=lambda _: None,
        pid=0, batch_idx=0, total_batches=1,
        frames_after_eos=0,
    )
    if seg is None:
        console.print("[red]Synthesis returned no audio.[/red]")
        return

    seg.export(str(out_path), format="wav")
    console.print(f"[green]Saved to {out_path}[/green]")

    try:
        subprocess.Popen([_player_command(), str(out_path)])
    except Exception as exc:
        console.print(f"[yellow]Could not open player: {exc}. Play manually: {out_path}[/yellow]")


def _tui_voice_actions(voice_name: str) -> None:
    """Per-voice action sub-menu. Loops until user chooses Back.

    Audition fires and returns to the browse list (caller re-shows browse).
    Exclude/Include toggles and stays here so the user can see the updated status.
    """
    from InquirerPy import inquirer

    auditioned = False
    while True:
        config = load_app_config()
        excluded = set(config.excluded_voices)
        registry = get_registry()
        meta = registry.resolve(voice_name)

        # Details header
        console.print()
        console.rule(f"[bold]{voice_name}[/bold]", style="dim")
        if meta:
            table = Table(show_header=False, box=None, padding=(0, 2))
            table.add_column("Field", style="dim", width=14)
            table.add_column("Value")
            table.add_row("Source", meta.source)
            table.add_row("Gender", meta.gender or "?")
            table.add_row("Accent", meta.accent or "?")
            if meta.dataset:
                table.add_row("Dataset", meta.dataset)
            if meta.speaker_id:
                table.add_row("Speaker ID", meta.speaker_id)
            pool_status = "[yellow]excluded[/yellow]" if voice_name in excluded else "[green]in pool[/green]"
            table.add_row("Pool status", pool_status)
            console.print(table)
        console.print()

        in_pool = voice_name not in excluded
        pool_label = "Exclude from pool" if in_pool else "Include in pool"

        choices = [
            {"name": "Audition  (play preview)", "value": "audition"},
            {"name": pool_label, "value": "toggle"},
            {"name": "← Back to browse list", "value": "back"},
        ]

        action = _tui_execute(inquirer.select(
            message=f"Action for {voice_name}",
            choices=choices,
            instruction="(↑↓ to move, Enter to confirm)",
        ))

        if action == "audition":
            _do_tui_audition(voice_name, config)
            # Return so browse list re-shows — user picks next voice to compare
            return

        elif action == "toggle":
            if in_pool:
                config.excluded_voices = list(config.excluded_voices) + [voice_name]
                save_app_config(config, DEFAULT_CONFIG_PATH)
                console.print(f"[yellow]'{voice_name}' excluded from pool.[/yellow]")
            else:
                config.excluded_voices = [v for v in config.excluded_voices if v != voice_name]
                save_app_config(config, DEFAULT_CONFIG_PATH)
                console.print(f"[green]'{voice_name}' restored to pool.[/green]")
            # Stay in sub-menu — loop refreshes with updated status

        elif action == "back":
            return


def _tui_browse() -> None:
    """Browse all voices with fuzzy search; drill into action sub-menu on select."""
    from InquirerPy import inquirer

    while True:
        config = load_app_config()
        excluded_set = set(config.excluded_voices)
        registry = get_registry()

        choices = []
        for meta in sorted(registry.voices, key=lambda v: v.name.lower()):
            pool_tag = " [excl]" if meta.name in excluded_set else ""
            label = f"{meta.name:<22} {meta.description}{pool_tag}"
            choices.append({"name": label, "value": meta.name})
        choices.append({"name": "← Back to main menu", "value": "__back__"})

        selected = _tui_execute(inquirer.fuzzy(
            message="Browse voices  (type to filter by name / accent / gender)",
            choices=choices,
            instruction="(type to filter, Enter to select, Ctrl-C to cancel)",
        ))

        if selected == "__back__":
            return

        _tui_voice_actions(selected)
        # Loop: re-show browse list after returning from actions


def _tui_pool() -> None:
    """Show excluded voices and offer bulk-restore via checkbox."""
    from InquirerPy import inquirer

    config = load_app_config()
    excluded = list(config.excluded_voices)

    console.print()
    if not excluded:
        console.print("[green]No voices are currently excluded from the pool.[/green]")
        console.print("[dim]Use Browse & audition to exclude individual voices.[/dim]")
        console.print()
        return

    console.print(f"[bold]Excluded voices ({len(excluded)}):[/bold]")
    for v in excluded:
        console.print(f"  • {v}")
    console.print()

    choices = [{"name": v, "value": v, "enabled": False} for v in excluded]
    to_restore = _tui_execute(inquirer.checkbox(
        message="Select voices to restore to pool (Space to toggle, Enter to confirm):",
        choices=choices,
        instruction="(Space=toggle, Enter=confirm, Ctrl-C=cancel)",
    ))

    if not to_restore:
        return

    restore_set = set(to_restore)
    config.excluded_voices = [v for v in config.excluded_voices if v not in restore_set]
    save_app_config(config, DEFAULT_CONFIG_PATH)
    for v in sorted(restore_set):
        console.print(f"[green]'{v}' restored to pool.[/green]")
    console.print()


def _tui_cast_lookup() -> None:
    """Interactive cast lookup — prompt for title, display cast table."""
    from InquirerPy import inquirer
    from argparse import Namespace

    title_query = _tui_execute(inquirer.text(
        message="Book title (or part of it):",
        instruction="(fuzzy matched against completed multi-voice jobs)",
    )).strip()

    if not title_query:
        return

    console.print()
    cmd_voices_cast(Namespace(title=title_query))
    console.print()


def cmd_voices_tui(args) -> None:
    """Interactive TUI for voice management (launched by `kenkui voices` with no subcommand)."""
    from InquirerPy import inquirer

    console.rule("[bold]kenkui — Voice Manager[/bold]")
    console.print()

    while True:
        action = _tui_execute(inquirer.select(
            message="What would you like to do?",
            choices=[
                {"name": "Browse & audition voices", "value": "browse"},
                {"name": "Manage exclusion pool", "value": "pool"},
                {"name": "Look up book cast", "value": "cast"},
                {"name": "Exit", "value": "exit"},
            ],
            instruction="(↑↓ to move, Enter to select)",
        ))

        if action == "browse":
            _tui_browse()
        elif action == "pool":
            _tui_pool()
        elif action == "cast":
            _tui_cast_lookup()
        elif action == "exit":
            break

    console.print()
    console.print("[dim]Bye.[/dim]")
