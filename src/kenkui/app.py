"""Kenkui Textual application entry point."""

import multiprocessing
from pathlib import Path

try:
    multiprocessing.set_start_method("spawn", force=True)
except RuntimeError:
    pass

import logging

from textual.app import App

from .config import get_config_manager

logger = logging.getLogger(__name__)
from .api_client import DEFAULT_HOST, DEFAULT_PORT, APIClient, get_client
from .models import AppConfig, ChapterSelection
from .screens import (
    BookSelectionScreen,
    ChapterSelectionScreen,
    ConfigScreen,
    QueueScreen,
    VoiceSelectionScreen,
)


class KenkuiApp(App):
    CSS = """
    /* ── Layout skeleton ─────────────────────────────────────── */
    Screen {
        background: $surface;
        layers: base overlay;
    }
    #main-content {
        height: 1fr;
        overflow-y: auto;
        padding: 1 2;
    }

    /* ── Uniform bottom action bar ───────────────────────────── */
    #bottom-bar {
        dock: bottom;
        height: auto;
        padding: 1 2;
        background: $panel;
        border-top: solid $primary-darken-2;
        align: left middle;
    }
    #bottom-bar Button {
        margin: 0 1 0 0;
    }

    /* ── Typography ──────────────────────────────────────────── */
    #title-label {
        text-style: bold;
        margin-bottom: 0;
    }
    #screen-hint {
        color: $text-muted;
        margin-bottom: 1;
    }

    /* ── DataTable ───────────────────────────────────────────── */
    DataTable {
        height: 15;
        margin-bottom: 1;
    }
    DataTable > .datatable--cursor {
        background: $primary;
        color: $text;
    }
    DataTable > .datatable--cursor-row {
        background: $primary-darken-1;
    }

    /* ── Loading row ─────────────────────────────────────────── */
    #loading-row {
        height: auto;
        margin-bottom: 1;
    }
    #loading-indicator {
        width: 3;
        margin-right: 1;
    }

    /* ── Chapter selection ───────────────────────────────────── */
    #chapter-list {
        height: 1fr;
        min-height: 10;
        border: solid $panel;
        margin-bottom: 1;
    }
    #chapter-count {
        color: $text-muted;
        margin-bottom: 1;
    }
    #chapter-filter {
        margin-bottom: 1;
    }

    /* ── Config form ─────────────────────────────────────────── */
    .config-form {
        height: auto;
    }
    #config-form-inner {
        height: auto;
    }
    #voice-header {
        text-style: bold;
        margin-top: 1;
    }

    /* ── Queue progress panel ────────────────────────────────── */
    #progress-container {
        height: auto;
        margin-top: 1;
        padding: 1;
        border: solid $primary;
        background: $panel;
    }
    #current-job-info { height: auto; }
    #chapter-info     { height: auto; color: $text-muted; }

    /* ── Inline dialogs (SaveConfigDialog, PathSelectionDialog) ─ */
    #save-config-dialog, #path-dialog {
        width: 50;
        height: auto;
        border: solid $primary;
        padding: 1 2;
        background: $surface;
        margin-top: 1;
    }
    #dialog-title  { text-style: bold; margin-bottom: 1; }
    #dialog-hint   { color: $text-muted; margin-bottom: 1; }
    #dialog-buttons { height: auto; margin-top: 1; }
    #dialog-buttons Button { margin: 0 1 0 0; }
    """

    BINDINGS = [
        ("escape", "pop_screen", "Back"),
        ("ctrl+q", "quit", "Quit"),
    ]

    SCREENS = {
        "book": BookSelectionScreen,
        "chapters": ChapterSelectionScreen,
        "voice": VoiceSelectionScreen,
        "config": ConfigScreen,
        "queue": QueueScreen,
    }

    def __init__(
        self,
        config_name: str | None = None,
        initial_path: Path | None = None,
        server_host: str = DEFAULT_HOST,
        server_port: int = DEFAULT_PORT,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.config_name = config_name
        self.initial_path = initial_path
        self.server_host = server_host
        self.server_port = server_port

        self.current_book: Path | None = None
        self.current_voice: str | None = None
        self.chapter_selection = ChapterSelection()
        self.app_config = AppConfig()

        # Per-session book scan cache shared across all BookSelectionScreen instances
        self.book_cache: dict[Path, list] = {}

        self._client: APIClient | None = None
        self._poll_timer = None
        self._last_notified_job_id: str | None = None

        self._load_config()

    # ------------------------------------------------------------------
    # Config
    # ------------------------------------------------------------------

    def _load_config(self):
        try:
            client = self._get_client()
            server_config = client.get_config()
            self.app_config = AppConfig.from_dict(server_config)
        except Exception as e:
            logger.debug(f"Could not fetch config from server: {e}")
            cfg_mgr = get_config_manager()
            self.app_config = cfg_mgr.load_app_config(self.config_name)

    def _get_client(self) -> APIClient:
        if self._client is None:
            self._client = get_client(self.server_host, self.server_port)
        return self._client

    def sync_config_to_server(self):
        try:
            client = self._get_client()
            client.update_config(self.app_config.to_dict())
            logger.debug("Config synced to server")
        except Exception as e:
            logger.debug(f"Failed to sync config to server: {e}")

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def on_mount(self):
        logger.debug("on_mount — connecting to worker server")
        try:
            health = self._get_client().health_check()
            logger.debug(f"Server healthy: {health}")
            self._load_config()
        except Exception as e:
            logger.debug(f"Could not connect to server: {e}")
            self.notify(
                f"Could not connect to worker server at "
                f"{self.server_host}:{self.server_port}. "
                "Make sure kenkui-server is running.",
                title="Connection Error",
            )
            cfg_mgr = get_config_manager()
            self.app_config = cfg_mgr.load_app_config(self.config_name)

        self._poll_timer = self.set_interval(1.0, self._poll_progress)

        if self.initial_path and self.initial_path.exists():
            if self.initial_path.is_file():
                self.current_book = self.initial_path
                self.push_screen(ChapterSelectionScreen())
            else:
                self.push_screen(BookSelectionScreen(book_path=self.initial_path))
        else:
            self.push_screen(BookSelectionScreen())

    def on_unmount(self):
        if self._poll_timer:
            self._poll_timer.stop()
        if self._client:
            self._client.close()

    # ------------------------------------------------------------------
    # Poll timer — refresh queue screen and fire completion notifications
    # ------------------------------------------------------------------

    def _poll_progress(self):
        try:
            client = self._get_client()
            client.get_status()
            queue_info = client.get_queue()
        except Exception as e:
            logger.debug(f"Polling error: {e}")
            return

        # Refresh any visible QueueScreen
        for screen in self.screen_stack:
            if isinstance(screen, QueueScreen):
                screen._refresh_queue()
                break

        # Completion notifications (fire once per job)
        for item in queue_info.items:
            if item.id == self._last_notified_job_id:
                continue
            if item.status in ("completed", "failed", "cancelled"):
                self._last_notified_job_id = item.id
                if item.status == "completed":
                    job_name = item.job.get("name", "Unknown")
                    msg = f"Job completed: {job_name}"
                    if item.output_path:
                        msg += f"\nSaved to: {item.output_path}"
                    self.notify(msg, title="Complete")
                elif item.status == "failed":
                    self.notify(
                        f"Job failed: {item.error_message or 'Unknown error'}",
                        title="Failed",
                    )
                elif item.status == "cancelled":
                    self.notify(
                        f"Job cancelled: {item.job.get('name', 'Unknown')}",
                        title="Cancelled",
                    )
                # Refresh queue display once more to show final status
                for screen in self.screen_stack:
                    if isinstance(screen, QueueScreen):
                        screen._refresh_queue()
                        break
                break

    # ------------------------------------------------------------------
    # Processing control (called from QueueScreen)
    # ------------------------------------------------------------------

    def start_processing(self):
        logger.debug("start_processing called")
        try:
            client = self._get_client()
            queue_info = client.get_queue()
            if queue_info.pending_count == 0:
                self.notify("No pending jobs in queue", title="Cannot Start")
                return
            if queue_info.current_item is not None:
                self.notify("Processing already in progress", title="Cannot Start")
                return
            self._last_notified_job_id = None
            client.start_processing()
            self._refresh_queue_screen()
        except Exception as e:
            logger.debug(f"start_processing error: {e}")
            self.notify(f"Error: {e}", title="Error")

    def stop_processing(self):
        logger.debug("stop_processing called")
        try:
            client = self._get_client()
            client.stop_processing()
            self.notify("Processing stopped", title="Stopped")
        except Exception as e:
            logger.debug(f"stop_processing error: {e}")
            self.notify(f"Error: {e}", title="Error")

    def _refresh_queue_screen(self):
        for screen in self.screen_stack:
            if isinstance(screen, QueueScreen):
                screen._refresh_queue()
                break


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def run_app(
    config_name: str | None = None,
    initial_path: Path | None = None,
    server_host: str = DEFAULT_HOST,
    server_port: int = DEFAULT_PORT,
):
    KenkuiApp(
        config_name=config_name,
        initial_path=initial_path,
        server_host=server_host,
        server_port=server_port,
    ).run()


if __name__ == "__main__":
    run_app()
