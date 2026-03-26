"""CLI subcommand: ``kenkui voices`` — list and manage available voices."""

from __future__ import annotations

import os
from pathlib import Path

from rich.console import Console
from rich.table import Table

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
    """Download uncompiled custom voices to the user data directory.

    Requires ``kenkui[custom-voices]`` to be installed (``pip install kenkui[custom-voices]``).
    """
    # Guard: check huggingface_hub is available
    try:
        import huggingface_hub  # noqa: F401
    except ImportError:
        console.print(
            "[red]Custom voices extra is not installed.[/red]\n"
            "Run: [bold]pip install kenkui\\[custom-voices][/bold]"
        )
        return

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
            return

    # ── Determine HF repo and destination ─────────────────────────────────
    hf_repo = getattr(args, "repo", None) or os.environ.get("KENKUI_VOICES_REPO", "")
    if not hf_repo:
        console.print(
            "[red]No HuggingFace repo specified.[/red]\n"
            "Set KENKUI_VOICES_REPO env var or pass --repo <user/repo-name>."
        )
        return

    xdg_data = os.environ.get("XDG_DATA_HOME", str(Path.home() / ".local" / "share"))
    dest_dir = Path(xdg_data) / "kenkui" / "voices" / "uncompiled"
    dest_dir.mkdir(parents=True, exist_ok=True)

    # ── Download ──────────────────────────────────────────────────────────
    try:
        from huggingface_hub import HfApi, hf_hub_download

        api = HfApi()
        repo_files = [
            f.rfilename
            for f in api.list_repo_files(hf_repo)  # type: ignore[attr-defined]
            if f.rfilename.endswith(".wav")
        ]
    except Exception as exc:
        console.print(f"[red]Could not list files in repo {hf_repo!r}: {exc}[/red]")
        return

    if not repo_files:
        console.print(f"[yellow]No .wav files found in {hf_repo!r}.[/yellow]")
        return

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


__all__ = ["cmd_voices_list", "cmd_voices_fetch"]
