"""CLI subcommand: ``kenkui voices`` — list and manage available voices."""

from __future__ import annotations

import sys
from pathlib import Path

from rich.console import Console
from rich.table import Table

from ..api_client import APIClient

console = Console()


def cmd_voices_list(args) -> None:
    """List all available voices with metadata.

    Accepts optional filter flags: ``--gender``, ``--accent``, ``--dataset``, ``--source``.
    """
    with APIClient() as client:
        data = client.list_voices(
            gender=getattr(args, "gender", None),
            accent=getattr(args, "accent", None),
            dataset=getattr(args, "dataset", None),
            source=getattr(args, "source", None),
        )
    voices = data["voices"]

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
        src = v.get("source", "")
        style = source_style.get(src, "")
        table.add_row(
            v.get("name", ""),
            f"[{style}]{src}[/{style}]" if style else src,
            v.get("gender") or "—",
            v.get("accent") or "—",
            v.get("dataset") or "—",
        )

    console.print(table)
    console.print(f"\n[dim]{len(voices)} voice(s) listed.[/dim]")


def cmd_voices_fetch(args) -> None:
    """Download uncompiled custom voices from a HuggingFace repo (via server)."""
    import os

    hf_repo = getattr(args, "repo", None) or os.environ.get("KENKUI_VOICES_REPO", "")
    if not hf_repo:
        console.print(
            "[red]No HuggingFace repo specified.[/red]\n"
            "Set KENKUI_VOICES_REPO env var or pass --repo <user/repo-name>."
        )
        sys.exit(1)

    patterns = getattr(args, "patterns", None)
    with APIClient() as client:
        result = client.fetch_uncompiled_voices(repo_id=hf_repo, patterns=patterns)

    task_id = result.get("task_id")
    if not task_id:
        console.print(f"[red]Server error: {result}[/red]")
        sys.exit(1)

    console.print(f"Fetching voices from [bold]{hf_repo}[/bold] (task {task_id})…")
    with APIClient() as client:
        try:
            final = client.poll_task(task_id, timeout=300.0)
        except TimeoutError:
            console.print("[red]Timed out waiting for voice fetch to complete.[/red]")
            sys.exit(1)
        except Exception as exc:
            console.print(f"[red]Voice fetch error: {exc}[/red]")
            sys.exit(1)

    if final.get("status") == "failed":
        console.print(f"[red]Download failed: {final.get('error')}[/red]")
        sys.exit(1)

    console.print(f"[green]Done! Run [bold]kenkui voices list[/bold] to see the new voices.[/green]")


def cmd_voices_download(args) -> int:
    """Download compiled voices from HuggingFace (via server)."""
    force = getattr(args, "force", False)
    with APIClient() as client:
        result = client.download_compiled_voices(force=force)

    task_id = result.get("task_id")
    if not task_id:
        console.print(f"[red]Server error: {result}[/red]")
        return 1

    console.print("Downloading voices from HuggingFace…")
    with APIClient() as client:
        try:
            final = client.poll_task(task_id, timeout=300.0)
        except TimeoutError:
            console.print("[red]Timed out waiting for voice download to complete.[/red]")
            return 1
        except Exception as exc:
            console.print(f"[red]Voice download error: {exc}[/red]")
            return 1

    if final.get("status") == "failed":
        console.print(f"[red]Download failed: {final.get('error')}[/red]")
        return 1

    console.print("[green]Done.[/green]")
    return 0


def cmd_voices_exclude(args) -> None:
    """Add a voice to the global excluded-from-auto-assignment list."""
    voice_name: str = args.voice
    with APIClient() as client:
        result = client.exclude_voice(voice_name)
    if result.get("warning"):
        console.print(f"[yellow]Warning: {result['warning']}[/yellow]")
    else:
        console.print(f"[green]'{voice_name}' excluded from auto-assignment pool.[/green]")


def cmd_voices_include(args) -> None:
    """Remove a voice from the excluded list, restoring it to auto-assignment."""
    voice_name: str = args.voice
    with APIClient() as client:
        result = client.include_voice(voice_name)
    if result.get("warning"):
        console.print(f"[yellow]Warning: {result['warning']}[/yellow]")
    else:
        console.print(f"[green]'{voice_name}' restored to auto-assignment pool.[/green]")


def cmd_voices_cast(args) -> None:
    """Display the character→voice cast for a completed multi-voice book."""
    job_id: str = args.job_id
    with APIClient() as client:
        cast_data = client.get_queue_cast(job_id)

    cast_entries = cast_data.get("cast", [])
    book_name = cast_data.get("book_name", job_id)

    if not cast_entries:
        console.print(f"[yellow]No cast information found for job '{job_id}'.[/yellow]")
        return

    table = Table(title=f"Cast — {book_name}", show_header=True, box=None)
    table.add_column("Character", style="bold", min_width=30)
    table.add_column("Voice", min_width=20)

    # Sort with NARRATOR last
    sorted_entries = sorted(
        cast_entries,
        key=lambda e: ("~" if e.get("character_id") == "NARRATOR" else (e.get("display_name") or e.get("character_id") or "").lower()),
    )
    for entry in sorted_entries:
        char_id = entry.get("character_id", "")
        display = entry.get("display_name") or char_id
        if char_id == "NARRATOR":
            display = "Narrator"
        table.add_row(display, entry.get("voice_name", ""))

    console.print(table)
    console.print(f"\n[dim]Job ID: {job_id}[/dim]")


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
    import tempfile

    voice_name: str = args.voice
    text: str = getattr(args, "text", None) or DEFAULT_AUDITION_TEXT

    console.print(
        "[yellow]Submitting audition request to server…[/yellow]"
    )

    with APIClient() as client:
        task_data = client.audition_voice(voice_name, text=text)
        task_id = task_data["task_id"]

        preview_text = text[:60] + ("…" if len(text) > 60 else "")
        console.print(
            f"Synthesizing preview for [bold]{voice_name}[/bold]: [dim]\"{preview_text}\"[/dim]"
        )

        try:
            result = client.poll_task(task_id, timeout=120.0)
        except TimeoutError:
            console.print("[red]Timed out waiting for audition to complete.[/red]")
            return
        except Exception as exc:
            console.print(f"[red]Audition error: {exc}[/red]")
            return

        if result["status"] == "failed":
            console.print(f"[red]Audition failed: {result.get('error')}[/red]")
            return

        wav_url = client.get_audition_wav_url(task_id)

    try:
        import httpx as _httpx
        wav_data = _httpx.get(wav_url, timeout=30.0)
        wav_data.raise_for_status()
        wav_bytes = wav_data.content
    except Exception as exc:
        console.print(f"[red]Could not download audio preview: {exc}[/red]")
        return

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        f.write(wav_bytes)
        wav_path = f.name

    console.print(f"[green]Preview saved to {wav_path}[/green]")

    player = _player_command()
    try:
        subprocess.Popen([player, wav_path])
    except Exception as exc:
        console.print(
            f"[yellow]Could not open system player ({exc}). Play manually: {wav_path}[/yellow]"
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


def _do_tui_audition(voice_name: str, client) -> None:
    """Synthesize a preview from within the TUI using an open APIClient."""
    import httpx
    import subprocess
    import tempfile

    console.print("\n[yellow]Submitting audition request to server…[/yellow]")
    try:
        task_data = client.audition_voice(voice_name, text=DEFAULT_AUDITION_TEXT)
        task_id = task_data["task_id"]
        console.print(f"Synthesizing [bold]{voice_name}[/bold]…")
        result = client.poll_task(task_id, timeout=120.0)
    except Exception as exc:
        console.print(f"[red]Audition failed: {exc}[/red]")
        return

    if result.get("status") == "failed":
        console.print(f"[red]Synthesis failed: {result.get('error')}[/red]")
        return

    try:
        wav_url = client.get_audition_wav_url(task_id)
        wav_data = httpx.get(wav_url).content
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            f.write(wav_data)
            wav_path = f.name
        console.print(f"[green]Saved to {wav_path}[/green]")
        subprocess.run([_player_command(), wav_path], check=False)
    except Exception as exc:
        console.print(f"[yellow]Could not play audio: {exc}[/yellow]")


def _tui_voice_actions(voice_name: str, client) -> None:
    """Per-voice action sub-menu. Loops until user chooses Back.

    Audition fires and returns to the browse list (caller re-shows browse).
    Exclude/Include toggles and stays here so the user can see the updated status.
    """
    from InquirerPy import inquirer

    while True:
        try:
            meta = client.get_voice(voice_name)
        except Exception:
            meta = None

        excluded = meta.get("excluded", False) if meta else False

        # Details header
        console.print()
        console.rule(f"[bold]{voice_name}[/bold]", style="dim")
        if meta:
            table = Table(show_header=False, box=None, padding=(0, 2))
            table.add_column("Field", style="dim", width=14)
            table.add_column("Value")
            table.add_row("Source", meta.get("source", "?"))
            table.add_row("Gender", meta.get("gender") or "?")
            table.add_row("Accent", meta.get("accent") or "?")
            if meta.get("dataset"):
                table.add_row("Dataset", meta["dataset"])
            if meta.get("speaker_id"):
                table.add_row("Speaker ID", meta["speaker_id"])
            pool_status = "[yellow]excluded[/yellow]" if excluded else "[green]in pool[/green]"
            table.add_row("Pool status", pool_status)
            console.print(table)
        console.print()

        in_pool = not excluded
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
            _do_tui_audition(voice_name, client)
            # Return so browse list re-shows — user picks next voice to compare
            return

        elif action == "toggle":
            if in_pool:
                try:
                    client.exclude_voice(voice_name)
                    console.print(f"[yellow]'{voice_name}' excluded from pool.[/yellow]")
                except Exception as exc:
                    console.print(f"[red]Failed to exclude: {exc}[/red]")
            else:
                try:
                    client.include_voice(voice_name)
                    console.print(f"[green]'{voice_name}' restored to pool.[/green]")
                except Exception as exc:
                    console.print(f"[red]Failed to include: {exc}[/red]")
            # Stay in sub-menu — loop refreshes with updated status

        elif action == "back":
            return


def _tui_browse(client) -> None:
    """Browse all voices with fuzzy search; drill into action sub-menu on select."""
    from InquirerPy import inquirer

    while True:
        try:
            data = client.list_voices()
        except Exception as exc:
            console.print(f"[red]Failed to load voices: {exc}[/red]")
            return

        voices = data.get("voices", [])
        choices = []
        for v in sorted(voices, key=lambda x: x.get("name", "").lower()):
            pool_tag = " [excl]" if v.get("excluded") else ""
            desc = v.get("description") or f"{v.get('gender', '?')} · {v.get('accent', '?')}"
            label = f"{v.get('name', ''):<22} {desc}{pool_tag}"
            choices.append({"name": label, "value": v.get("name")})
        choices.append({"name": "← Back to main menu", "value": "__back__"})

        selected = _tui_execute(inquirer.fuzzy(
            message="Browse voices  (type to filter by name / accent / gender)",
            choices=choices,
            instruction="(type to filter, Enter to select, Ctrl-C to cancel)",
        ))

        if selected == "__back__":
            return

        _tui_voice_actions(selected, client)
        # Loop: re-show browse list after returning from actions


def _tui_pool(client) -> None:
    """Show excluded voices and offer bulk-restore via checkbox."""
    from InquirerPy import inquirer

    try:
        data = client.list_voices()
    except Exception as exc:
        console.print(f"[red]Failed to load voices: {exc}[/red]")
        return

    excluded = [v["name"] for v in data.get("voices", []) if v.get("excluded")]

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

    for v in sorted(to_restore):
        try:
            client.include_voice(v)
            console.print(f"[green]'{v}' restored to pool.[/green]")
        except Exception as exc:
            console.print(f"[red]Failed to restore '{v}': {exc}[/red]")
    console.print()


def _tui_cast_lookup() -> None:
    """Interactive cast lookup — prompt for job ID, display cast table."""
    from InquirerPy import inquirer
    from argparse import Namespace

    job_id = _tui_execute(inquirer.text(
        message="Job ID:",
        instruction="(enter the job ID for the completed multi-voice book)",
    )).strip()

    if not job_id:
        return

    console.print()
    cmd_voices_cast(Namespace(job_id=job_id))
    console.print()


def cmd_voices_tui(args) -> None:
    """Interactive TUI for voice management (launched by `kenkui voices` with no subcommand)."""
    from InquirerPy import inquirer

    console.rule("[bold]kenkui — Voice Manager[/bold]")
    console.print()

    with APIClient() as client:
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
                _tui_browse(client)
            elif action == "pool":
                _tui_pool(client)
            elif action == "cast":
                _tui_cast_lookup()
            elif action == "exit":
                break

    console.print()
    console.print("[dim]Bye.[/dim]")
