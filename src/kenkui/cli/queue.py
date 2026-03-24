"""kenkui queue — snapshot, live dashboard, and queue control.

Entry point
-----------
cmd_queue(args)   Dispatches based on args.queue_command and args.live.

Sub-commands
------------
kenkui queue              Snapshot: prints a Rich table, exits.
kenkui queue --live       Live-refreshing dashboard (Ctrl+C to exit).
kenkui queue start        Start processing next pending job, exits.
kenkui queue start --live Start processing then enter live dashboard.
kenkui queue stop         Stop current job, exits.
"""

from __future__ import annotations

import time

from rich.columns import Columns
from rich.console import Console
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn, TimeRemainingColumn
from rich.table import Table
from rich.text import Text

console = Console()

# Status colours for Rich markup.
_STATUS_STYLE: dict[str, str] = {
    "pending": "dim",
    "processing": "cyan bold",
    "completed": "green",
    "failed": "red bold",
    "cancelled": "yellow",
}


# ---------------------------------------------------------------------------
# Rendering helpers
# ---------------------------------------------------------------------------


def _eta_str(eta_seconds: int) -> str:
    if eta_seconds <= 0:
        return "—"
    m, s = divmod(eta_seconds, 60)
    h, m = divmod(m, 60)
    if h:
        return f"{h}h {m}m"
    if m:
        return f"{m}m {s}s"
    return f"{s}s"


def _build_queue_table(queue_info, include_terminal: bool = True) -> Table:
    """Build a Rich Table from a QueueInfo object."""
    tbl = Table(
        title="Job Queue",
        show_header=True,
        header_style="bold",
        expand=True,
    )
    tbl.add_column("ID", style="dim", width=10)
    tbl.add_column("Name", min_width=20)
    tbl.add_column("Status", width=12)
    tbl.add_column("Progress", width=8, justify="right")
    tbl.add_column("Elapsed", width=8, justify="right")
    tbl.add_column("ETA", width=8, justify="right")
    tbl.add_column("Chapter", overflow="fold")

    terminal = {"completed", "failed", "cancelled"}
    for item in queue_info.items:
        if not include_terminal and item.status in terminal:
            continue
        status_style = _STATUS_STYLE.get(item.status, "")
        progress_capped = min(item.progress or 0, 100)
        progress_str = f"{progress_capped:.1f}%"
        if item.status == "processing" and getattr(item, "started_at", 0) > 0:
            elapsed_s = int(time.time() - item.started_at)
            elapsed_str = _eta_str(elapsed_s)
        else:
            elapsed_str = "—"
        eta_str = _eta_str(item.eta_seconds) if item.status == "processing" else "—"
        chapter_str = (item.current_chapter or "")[:60]
        job_name = item.job.get("name", item.id) if isinstance(item.job, dict) else item.id

        tbl.add_row(
            item.id,
            job_name,
            Text(item.status, style=status_style),
            progress_str,
            elapsed_str,
            eta_str,
            chapter_str,
        )

    return tbl


def _build_progress_bar(queue_info) -> Progress | None:
    """Build a Rich Progress bar for the active job, or None if idle."""
    active = queue_info.current_item
    if active is None:
        return None

    prog = Progress(
        SpinnerColumn(),
        BarColumn(bar_width=None),
        TextColumn("[progress.percentage]{task.percentage:>5.1f}%"),
        TimeRemainingColumn(),
        TextColumn("{task.description}"),
        expand=True,
    )
    desc = (active.current_chapter or "")[:60]
    progress_capped = min(active.progress or 0, 100)
    prog.add_task(desc, total=100, completed=progress_capped)
    return prog


def _build_summary_line(queue_info) -> Text:
    parts = [
        f"[cyan]{queue_info.pending_count} pending[/cyan]",
        f"[cyan bold]{1 if queue_info.current_item else 0} processing[/cyan bold]",
        f"[green]{queue_info.completed_count} completed[/green]",
        f"[red]{queue_info.failed_count} failed[/red]",
    ]
    return Text.from_markup("  ·  ".join(parts))


# ---------------------------------------------------------------------------
# Snapshot (non-live) view
# ---------------------------------------------------------------------------


def _snapshot(client) -> int:
    try:
        queue_info = client.get_queue()
    except Exception as exc:
        console.print(f"[red]Could not reach server: {exc}[/red]")
        return 1

    console.print(_build_queue_table(queue_info))
    console.print()

    prog = _build_progress_bar(queue_info)
    if prog:
        active = queue_info.current_item
        job_name = active.job.get("name", active.id) if isinstance(active.job, dict) else active.id
        console.print(Panel(prog, title=f"Active: {job_name}"))
        console.print()

    console.print(_build_summary_line(queue_info))
    return 0


# ---------------------------------------------------------------------------
# Live dashboard
# ---------------------------------------------------------------------------


def _live_dashboard(client) -> int:
    """Run a live-refreshing dashboard.  Returns exit code.

    Can be imported and called from cli/add.py (bare shorthand interactive path).
    """
    notified_ids: set[str] = set()

    def _make_layout(queue_info) -> Layout:
        layout = Layout()
        table = _build_queue_table(queue_info, include_terminal=False)
        prog = _build_progress_bar(queue_info)

        if prog:
            active = queue_info.current_item
            job_name = (
                active.job.get("name", active.id) if isinstance(active.job, dict) else active.id
            )
            progress_panel = Panel(prog, title=f"Active: {job_name}")
        else:
            progress_panel = Panel(
                Text("[dim]No active job[/dim]"), title="Active", style="dim"
            )

        footer = Text.from_markup("[dim]Ctrl+C to exit  ·  [/dim]")
        footer.append_text(_build_summary_line(queue_info))

        layout.split_column(
            Layout(table, name="table"),
            Layout(progress_panel, name="progress", size=5),  # fixed — never shifts
            Layout(footer, name="footer", size=1),
        )
        return layout


    try:
        with Live(console=console, refresh_per_second=1, screen=True, transient=False) as live:
            while True:
                try:
                    queue_info = client.get_queue()
                except Exception:
                    time.sleep(1)
                    continue

                live.update(_make_layout(queue_info))

                # Fire completion notifications.
                for item in queue_info.items:
                    if item.id in notified_ids:
                        continue
                    if item.status in ("completed", "failed", "cancelled"):
                        notified_ids.add(item.id)
                        job_name = (
                            item.job.get("name", item.id) if isinstance(item.job, dict) else item.id
                        )
                        if item.status == "completed":
                            live.console.print(
                                f"[green]✓ Job complete: {job_name}"
                                + (f"  →  {item.output_path}" if item.output_path else "")
                                + "[/green]"
                            )
                        elif item.status == "failed":
                            live.console.print(
                                f"[red]✗ Job failed: {job_name} — {item.error_message}[/red]"
                            )
                        elif item.status == "cancelled":
                            live.console.print(f"[yellow]⊘ Job cancelled: {job_name}[/yellow]")

                time.sleep(1)

    except KeyboardInterrupt:
        console.print("\n[dim]Exiting live dashboard.[/dim]")

    return 0


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def cmd_queue(args) -> int:
    """Handle 'kenkui queue [--live] [start [--live] | stop]'."""
    from ..api_client import APIClient

    # For nested sub-commands (start/stop) the server flags may be on the
    # parent 'queue' namespace; fall back gracefully.
    host = getattr(args, "server_host", "127.0.0.1")
    port = getattr(args, "server_port", 45365)
    client = APIClient(host=host, port=port)

    try:
        queue_command = getattr(args, "queue_command", None)

        if queue_command == "stop":
            try:
                client.stop_processing()
                console.print("[yellow]Processing stopped.[/yellow]")
            except Exception as exc:
                console.print(f"[red]Error: {exc}[/red]")
                return 1
            return 0

        if queue_command == "start":
            try:
                client.start_processing()
                console.print("[green]Processing started.[/green]")
            except Exception as exc:
                console.print(f"[red]Error: {exc}[/red]")
                return 1

            if getattr(args, "live", False):
                return _live_dashboard(client)
            return 0

        # No sub-command — snapshot or live.
        if getattr(args, "live", False):
            return _live_dashboard(client)

        return _snapshot(client)

    finally:
        client.close()
