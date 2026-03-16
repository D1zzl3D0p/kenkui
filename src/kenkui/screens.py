"""Textual screens for the Kenkui TUI."""

from __future__ import annotations

import logging
import threading
from pathlib import Path
from typing import TYPE_CHECKING, Protocol

logger = logging.getLogger(__name__)

from textual.app import ComposeResult
from textual.containers import Container, Horizontal, VerticalScroll
from textual.message import Message
from textual.screen import Screen
from textual.widgets import (
    Button,
    DataTable,
    Footer,
    Header,
    Input,
    Label,
    LoadingIndicator,
    SelectionList,
    Static,
)
from textual.widgets.selection_list import Selection
from textual_fspicker import SelectDirectory

from .api_client import get_client
from .chapter_filter import ChapterFilter
from .config import get_config_manager
from .file_finder import find_ebook_files
from .helpers import get_bundled_voices
from .models import AppConfig, BookInfo, ChapterPreset, ChapterSelection
from .widgets import (
    ChapterPresetSelector,
    ConfigForm,
    HuggingFaceAuthModal,
    JobActionsModal,
    LoadConfigDialog,
    SaveConfigDialog,
)

if TYPE_CHECKING:
    pass


# ---------------------------------------------------------------------------
# Protocol so screens can reference typed app attributes without circular imports
# ---------------------------------------------------------------------------


class HasAppAttributes(Protocol):
    current_book: Path | None
    current_voice: str | None
    chapter_selection: ChapterSelection
    app_config: AppConfig

    def push_screen(self, screen, *args, **kwargs): ...
    def pop_screen(self, *args, **kwargs): ...
    def notify(self, message, **kwargs): ...
    def sync_config_to_server(self): ...


def _app(screen: Screen) -> HasAppAttributes:
    return screen.app  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


class SelectionHelpers:
    @staticmethod
    def filter_items(items: list, search_term: str, *fields) -> list:
        if not search_term:
            return items
        term = search_term.lower()
        return [item for item in items if any(f(item) and term in f(item).lower() for f in fields)]


# ---------------------------------------------------------------------------
# Messages
# ---------------------------------------------------------------------------


class BooksLoaded(Message):
    def __init__(self, books: list[BookInfo], scanned_path: Path):
        super().__init__()
        self.books = books
        self.scanned_path = scanned_path


# ---------------------------------------------------------------------------
# TableSelectionMixin
# ---------------------------------------------------------------------------


class TableSelectionMixin:
    """Mixin for screens that use a DataTable for single-item selection."""

    selected_item: object = None

    def get_table(self) -> DataTable:
        raise NotImplementedError

    def get_selection_items(self) -> list:
        raise NotImplementedError

    def get_next_button_id(self) -> str:
        return "btn-next"

    def on_selection_changed(self, item: object):
        pass

    def on_data_table_row_selected(self, event):
        idx = event.cursor_row
        items = self.get_selection_items()
        if idx is not None and 0 <= idx < len(items):
            self.selected_item = items[idx]
            self.query_one(f"#{self.get_next_button_id()}", Button).disabled = False  # type: ignore[attr-defined]
            self.on_selection_changed(self.selected_item)

    def action_next(self):
        pass


# ---------------------------------------------------------------------------
# BookSelectionScreen
# ---------------------------------------------------------------------------

BOOK_HINT = (
    "↑↓ navigate  •  Enter to select  •  Type to search  •  "
    "Browse… to scan another folder  •  Q for queue"
)


class BookSelectionScreen(Screen):
    BINDINGS = [("q", "push_queue", "Queue")]

    def __init__(self, book_path: Path | None = None, **kwargs):
        super().__init__(**kwargs)
        self.book_path = book_path
        self.selected_book: BookInfo | None = None
        self.all_books: list[BookInfo] = []
        self.filtered_books: list[BookInfo] = []
        self._current_scan_dir: Path | None = None

    @property
    def _book_cache(self) -> dict[Path, list[BookInfo]]:
        """Per-session book cache stored on the app so it survives screen pushes."""
        return self.app.book_cache  # type: ignore[attr-defined]

    def compose(self) -> ComposeResult:
        yield Header()
        yield Footer()
        with Container(id="main-content"):
            yield Label("Select a Book to Convert", id="title-label")
            yield Static(BOOK_HINT, id="screen-hint")
            yield Input(placeholder="Search by title or author…", id="search-input")
            yield DataTable(id="book-table")
            with Horizontal(id="loading-row"):
                yield LoadingIndicator(id="loading-indicator")
                yield Static("", id="loading-status")
        with Horizontal(id="bottom-bar"):
            yield Button("Browse…", id="btn-browse", variant="primary")
            yield Button("Queue", id="btn-queue", variant="default")
            yield Button("Next →", id="btn-next", variant="success", disabled=True)

    def on_mount(self):
        table = self.query_one("#book-table", DataTable)
        table.add_columns("#", "Title", "Author", "Format", "Path")
        table.cursor_type = "row"
        table.focus()

        start_path = self.book_path or Path.home()
        self._load_books(start_path)

    # ------------------------------------------------------------------
    # Book loading + cache
    # ------------------------------------------------------------------

    def _load_books(self, path: Path):
        self._current_scan_dir = path if path.is_dir() else path.parent
        self._show_loading(True, path)

        if path.is_file():
            book = self._make_book_info(path)
            self.all_books = [book]
            self.filtered_books = self.all_books
            self._book_cache[path.parent] = self.all_books
            self._update_table()
            self._show_loading(False)
            return

        # Directory — check cache first
        if path in self._book_cache:
            self.all_books = self._book_cache[path]
            self.filtered_books = self.all_books
            self._update_table()
            self._show_loading(False)
            return

        # Scan in background thread
        def _scan():
            from .readers import get_reader

            infos: list[BookInfo] = []
            for bp in find_ebook_files(path):
                try:
                    reader = get_reader(bp, verbose=False)
                    meta = reader.get_metadata()
                    infos.append(
                        BookInfo(
                            path=bp,
                            title=meta.title or bp.stem,
                            author=meta.author,
                            format=bp.suffix.upper(),
                        )
                    )
                except Exception:
                    infos.append(BookInfo(path=bp, title=bp.stem, format=bp.suffix.upper()))
            infos.sort(key=lambda b: b.title.lower())
            self.post_message(BooksLoaded(infos, path))

        threading.Thread(target=_scan, daemon=True).start()

    def _make_book_info(self, path: Path) -> BookInfo:
        from .readers import get_reader

        try:
            reader = get_reader(path, verbose=False)
            meta = reader.get_metadata()
            return BookInfo(
                path=path,
                title=meta.title or path.stem,
                author=meta.author,
                format=path.suffix.upper(),
            )
        except Exception:
            return BookInfo(path=path, title=path.stem, format=path.suffix.upper())

    def _show_loading(self, show: bool, path: Path | None = None):
        ind = self.query_one("#loading-indicator", LoadingIndicator)
        status = self.query_one("#loading-status", Static)
        ind.display = show
        if show and path:
            dir_name = path.name if path.is_dir() else path.parent.name
            status.update(f"Scanning {dir_name!r} for EPUBs, MOBIs, FB2s…")
        elif not show:
            count = len(self.filtered_books)
            total = len(self.all_books)
            dir_name = self._current_scan_dir.name if self._current_scan_dir else "?"
            if count == total:
                status.update(f"Found {count} book(s) in {dir_name!r}")
            else:
                status.update(f"Showing {count} of {total} books in {dir_name!r}")

    def on_books_loaded(self, event: BooksLoaded):
        # Deduplicate
        seen: set[Path] = set()
        unique = []
        for b in event.books:
            if b.path not in seen:
                seen.add(b.path)
                unique.append(b)
        self._book_cache[event.scanned_path] = unique
        self.all_books = unique
        self.filtered_books = unique
        self._update_table()
        self._show_loading(False)

    def _update_table(self):
        table = self.query_one("#book-table", DataTable)
        table.clear()
        home = Path.home()
        for i, book in enumerate(self.filtered_books, 1):
            try:
                rel = book.path.relative_to(home)
                path_str = f"~/{rel}"
            except ValueError:
                path_str = str(book.path)
            # Truncate cells so no single column dominates the layout
            title = book.title[:40] + "…" if len(book.title) > 40 else book.title
            author = (
                (book.author or "—")[:24] + "…"
                if len(book.author or "") > 24
                else (book.author or "—")
            )
            path_d = path_str[:50] + "…" if len(path_str) > 50 else path_str
            table.add_row(str(i), title, author, book.format, path_d)
        if self.filtered_books:
            table.move_cursor(row=0)

    def _filter_books(self, term: str):
        self.filtered_books = SelectionHelpers.filter_items(
            self.all_books,
            term,
            lambda b: b.title,
            lambda b: b.author or "",
        )
        self._update_table()
        self._show_loading(False)

    # ------------------------------------------------------------------
    # Events
    # ------------------------------------------------------------------

    def on_input_changed(self, event: Input.Changed):
        if event.input.id == "search-input":
            self._filter_books(event.value)

    def on_data_table_row_selected(self, event: DataTable.RowSelected):
        idx = event.cursor_row
        if idx is not None and 0 <= idx < len(self.filtered_books):
            self.selected_book = self.filtered_books[idx]
            self.query_one("#btn-next", Button).disabled = False

    def on_button_pressed(self, event: Button.Pressed):
        bid = event.button.id
        if bid == "btn-browse":
            self._action_browse()
        elif bid == "btn-queue":
            self.action_push_queue()
        elif bid == "btn-next":
            self._action_next()

    def _action_browse(self):
        async def _pick():
            picker = SelectDirectory(location=str(Path.home()), title="Select Directory")
            result = await self.app.push_screen(picker, wait_for_dismiss=True)
            if result:
                p = Path(result)
                if p.is_dir():
                    self._load_books(p)

        self.run_worker(_pick)

    def _action_next(self):
        if self.selected_book:
            self.app.current_book = self.selected_book.path  # type: ignore[attr-defined]
            self.app.push_screen(ChapterSelectionScreen())

    def action_push_queue(self):
        self.app.push_screen(QueueScreen())


# ---------------------------------------------------------------------------
# ChapterSelectionScreen
# ---------------------------------------------------------------------------

CHAPTER_HINT = (
    "Choose a filter preset above, or check/uncheck individual chapters below.\n"
    "Space = toggle  •  ↑↓ = navigate  •  Type in the filter box to search  •  Q for queue"
)


class ChapterSelectionScreen(Screen):
    BINDINGS = [("q", "push_queue", "Queue")]

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.all_chapters: list = []
        # Maps display index in SelectionList → original chapter index
        self._display_to_chapter_idx: list[int] = []
        # The authoritative checked set — survives filter rebuilds
        self._checked_orig_indices: set[int] = set()
        self._filter_term: str = ""

    def compose(self) -> ComposeResult:
        yield Header()
        yield Footer()
        with Container(id="main-content"):
            yield Label("Select Chapters", id="title-label")
            yield Static(CHAPTER_HINT, id="screen-hint")
            yield ChapterPresetSelector(id="preset-selector", on_change=self._on_preset_changed)
            yield Input(placeholder="Filter chapters by title…", id="chapter-filter")
            yield Static("", id="chapter-count")
            yield SelectionList(id="chapter-list")
        with Horizontal(id="bottom-bar"):
            yield Button("← Back", id="btn-back", variant="default")
            yield Button("Select All", id="btn-select-all", variant="primary")
            yield Button("Select None", id="btn-select-none", variant="default")
            yield Button("Queue", id="btn-queue", variant="default")
            yield Button("Next →", id="btn-next", variant="success")

    def on_mount(self):
        from .readers import get_reader

        book = self.app.current_book  # type: ignore[attr-defined]
        if book:
            try:
                reader = get_reader(book, verbose=False)
                self.all_chapters = reader.get_chapters()
            except Exception:
                self.all_chapters = []
        self.app.chapter_selection = ChapterSelection()  # type: ignore[attr-defined]
        # Derive initial selection from "content-only" preset
        self._checked_orig_indices = self._preset_indices("content-only")
        self._rebuild_list()

    # ------------------------------------------------------------------
    # Core helpers
    # ------------------------------------------------------------------

    def _chapters_with_content(self) -> list[tuple[int, object]]:
        return [
            (i, ch)
            for i, ch in enumerate(self.all_chapters)
            if hasattr(ch, "paragraphs") and len(ch.paragraphs) > 0
        ]

    def _preset_indices(self, preset_id: str) -> set[int]:
        from .chapter_filter import FilterOperation

        if preset_id == "none":
            return set()
        if preset_id == "manual":
            return {i for i, _ in self._chapters_with_content()}
        try:
            fc = ChapterFilter([FilterOperation("preset", preset_id)])
            filtered = fc.apply(self.all_chapters)
            return {self.all_chapters.index(ch) for ch in filtered}
        except Exception:
            return set()

    def _rebuild_list(self):
        """Repopulate the SelectionList honouring filter term and checked set."""
        sel = self.query_one("#chapter-list", SelectionList)
        sel.clear_options()
        self._display_to_chapter_idx = []

        content_chapters = self._chapters_with_content()
        skipped = len(self.all_chapters) - len(content_chapters)

        term = self._filter_term.lower()
        for orig_idx, ch in content_chapters:
            title: str = getattr(ch, "title", "")
            if term and term not in title.lower():
                continue
            display_idx = len(self._display_to_chapter_idx)
            self._display_to_chapter_idx.append(orig_idx)
            checked = orig_idx in self._checked_orig_indices
            label = f"{orig_idx + 1:>4}.  {title[:60]}"
            sel.add_option(Selection(label, display_idx, checked))

        self._sync_app_selection()
        self._update_count(skipped)

    # ------------------------------------------------------------------
    # Preset / filter handlers
    # ------------------------------------------------------------------

    def _on_preset_changed(self, preset_id: str):
        self._checked_orig_indices = self._preset_indices(preset_id)
        self._rebuild_list()

    def on_radio_set_changed(self, event):
        self._on_preset_changed(event.pressed.id)

    def on_input_changed(self, event: Input.Changed):
        if event.input.id == "chapter-filter":
            self._filter_term = event.value
            self._rebuild_list()

    # ------------------------------------------------------------------
    # SelectionList toggle event
    # ------------------------------------------------------------------

    def on_selection_list_selected_changed(self, event: SelectionList.SelectedChanged):
        # Rebuild _checked_orig_indices from current visible state
        for display_idx, orig_idx in enumerate(self._display_to_chapter_idx):
            if display_idx in event.selection_list.selected:
                self._checked_orig_indices.add(orig_idx)
            else:
                self._checked_orig_indices.discard(orig_idx)
        self._sync_app_selection()
        n = len(self._checked_orig_indices)
        self.query_one("#chapter-count", Static).update(
            f"{n} chapter{'s' if n != 1 else ''} selected"
        )

    # ------------------------------------------------------------------
    # Sync to app state
    # ------------------------------------------------------------------

    def _sync_app_selection(self):
        included = sorted(self._checked_orig_indices)
        all_orig = {orig for orig in self._display_to_chapter_idx}
        excluded = sorted(all_orig - self._checked_orig_indices)
        self.app.chapter_selection.preset = ChapterPreset.CUSTOM  # type: ignore[attr-defined]
        self.app.chapter_selection.included = included  # type: ignore[attr-defined]
        self.app.chapter_selection.excluded = excluded  # type: ignore[attr-defined]

    def _update_count(self, skipped: int):
        n = len(self._checked_orig_indices)
        total = len(self._display_to_chapter_idx)
        text = f"{n} of {total} chapters selected"
        if skipped:
            text += f"  ({skipped} empty chapters hidden)"
        self.query_one("#chapter-count", Static).update(text)

    # ------------------------------------------------------------------
    # Buttons
    # ------------------------------------------------------------------

    def on_button_pressed(self, event: Button.Pressed):
        bid = event.button.id
        if bid == "btn-back":
            self.app.pop_screen()
        elif bid == "btn-select-all":
            self._checked_orig_indices = {
                orig for orig, _ in [(orig, ch) for orig, ch in self._chapters_with_content()]
            }
            self._rebuild_list()
        elif bid == "btn-select-none":
            self._checked_orig_indices = set()
            self._rebuild_list()
        elif bid == "btn-queue":
            self.action_push_queue()
        elif bid == "btn-next":
            self._sync_app_selection()
            self.app.push_screen(VoiceSelectionScreen())

    def action_push_queue(self):
        self.app.push_screen(QueueScreen())


# ---------------------------------------------------------------------------
# VoiceSelectionScreen
# ---------------------------------------------------------------------------

VOICE_HINT = (
    "↑↓ or click to highlight a voice  •  Enter to confirm  •  "
    "Select 'Custom…' to use your own voice file  •  Q for queue"
)


class VoiceSelectionScreen(Screen, TableSelectionMixin):
    BINDINGS = [("q", "push_queue", "Queue")]

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.voices: list[tuple[str, str]] = []
        self.selected_voice: str | None = None

    def compose(self) -> ComposeResult:
        yield Header()
        yield Footer()
        with Container(id="main-content"):
            yield Label("Select a Voice", id="title-label")
            yield Static(VOICE_HINT, id="screen-hint")
            yield DataTable(id="voice-table")
        with Horizontal(id="bottom-bar"):
            yield Button("← Back", id="btn-back", variant="default")
            yield Button("Queue", id="btn-queue", variant="default")
            yield Button("Next →", id="btn-next", variant="success", disabled=True)

    def get_table(self) -> DataTable:
        return self.query_one("#voice-table", DataTable)

    def get_selection_items(self) -> list:
        return self.voices

    def on_mount(self):
        from .utils import DEFAULT_VOICES, VOICE_DESCRIPTIONS

        table = self.query_one("#voice-table", DataTable)
        table.add_columns("#", "Voice", "Description")
        self.voices = []
        for v in DEFAULT_VOICES:
            self.voices.append((v, f"Built-in — {VOICE_DESCRIPTIONS.get(v, 'Default voice')}"))
        for v in get_bundled_voices():
            if v.lower() != "default.txt":
                self.voices.append(
                    (
                        v.replace(".wav", ""),
                        "Bundled voice (requires HuggingFace login)",
                    )
                )
        self.voices.append(("Custom…", "Browse for a voice file or enter a HuggingFace URL"))
        for i, (name, desc) in enumerate(self.voices, 1):
            table.add_row(str(i), name, desc)
        table.cursor_type = "row"
        table.focus()

    def on_selection_changed(self, item):
        voice_id = item[0]
        if voice_id == "Custom…":
            self._show_custom_voice_dialog()
        else:
            self.selected_voice = voice_id
            self.app.current_voice = voice_id  # type: ignore[attr-defined]
            # Check HuggingFace auth for bundled / non-default voices
            from .utils import DEFAULT_VOICES

            if voice_id not in DEFAULT_VOICES:
                self._check_hf_auth(voice_id)

    def _show_custom_voice_dialog(self):
        from .widgets import PathSelectionDialog

        def on_select(path: str):
            if path:
                self.selected_voice = path
                self.app.current_voice = path  # type: ignore[attr-defined]
                self.query_one("#btn-next", Button).disabled = False

        dialog = PathSelectionDialog(
            mode="both",
            title="Select Custom Voice File",
            filetypes={"Audio": [".wav", ".mp3", ".ogg"]},
            on_select=on_select,
        )
        self.mount(dialog)

    def _check_hf_auth(self, voice_id: str):
        """Check HuggingFace auth status for a bundled/custom voice.

        If auth is needed, opens HuggingFaceAuthModal.  The user can still
        proceed to the next screen regardless — auth is advisory, not blocking,
        because the actual download happens later in the worker process.
        """
        from .huggingface_auth import AuthStatus, check_auth_status

        status = check_auth_status("kyutai/pocket-tts")
        if status == AuthStatus.OK:
            # Already authenticated — nothing to do
            return

        # Notify user and offer to open setup flow
        self.app.notify(  # type: ignore[attr-defined]
            f"Voice '{voice_id}' may require HuggingFace access. Starting setup…",
            title="HuggingFace Setup",
        )

        async def _open_modal():
            await self.app.push_screen(  # type: ignore[attr-defined]
                HuggingFaceAuthModal(model_id="kyutai/pocket-tts"),
                wait_for_dismiss=True,
            )
            # Regardless of result, allow the user to continue.
            # The worker will fail gracefully if access is not granted.

        self.run_worker(_open_modal)

    def action_next(self):
        if self.selected_voice:
            self.app.push_screen(ConfigScreen())

    def on_button_pressed(self, event: Button.Pressed):
        bid = event.button.id
        if bid == "btn-back":
            self.app.pop_screen()
        elif bid == "btn-queue":
            self.action_push_queue()
        elif bid == "btn-next":
            self.action_next()

    def action_push_queue(self):
        self.app.push_screen(QueueScreen())


# ---------------------------------------------------------------------------
# ConfigScreen
# ---------------------------------------------------------------------------

CONFIG_HINT = (
    "Adjust processing options and job defaults below.\n"
    "Set 'Default Voice' and 'Default Chapter Preset' to enable headless CLI usage:\n"
    "  kenkui book.epub -c myconfig\n"
    "Load Config… to restore settings  •  Save Config… to save for reuse  •  Q for queue"
)


class ConfigScreen(Screen):
    BINDINGS = [("q", "push_queue", "Queue")]

    def compose(self) -> ComposeResult:
        yield Header()
        yield Footer()
        with VerticalScroll(id="main-content"):
            yield Label("Processing Options", id="title-label")
            yield Static(CONFIG_HINT, id="screen-hint")
            yield ConfigForm(
                id="config-form",
                on_save=self._on_form_save,
                initial_values=self.app.app_config.to_dict(),  # type: ignore[attr-defined]
            )
        with Horizontal(id="bottom-bar"):
            yield Button("← Back", id="btn-back", variant="default")
            yield Button("Load Config…", id="btn-load-config", variant="default")
            yield Button("Save Config…", id="btn-save-config", variant="primary")
            yield Button("Queue", id="btn-queue", variant="default")
            yield Button("Save & Queue →", id="btn-save-queue", variant="success")

    def on_show(self):
        """Refresh form with the current live config whenever this screen becomes visible."""
        try:
            form = self.query_one("#config-form", ConfigForm)
            form.load_values(self.app.app_config.to_dict())  # type: ignore[attr-defined]
        except Exception:
            pass  # Form not yet mounted on first show — compose() handles initial values

    def _on_form_save(self, data: dict):
        self.app.app_config = AppConfig.from_dict(data)  # type: ignore[attr-defined]
        self.app.sync_config_to_server()  # type: ignore[attr-defined]

    # ------------------------------------------------------------------
    # Load config
    # ------------------------------------------------------------------

    def _action_load_config(self):
        cfg_mgr = get_config_manager()
        names = cfg_mgr.list_configs()
        if not names:
            self.app.notify("No saved configs found.", title="Load Config")
            return

        async def _show():
            chosen = await self.app.push_screen(
                LoadConfigDialog(config_names=names),
                wait_for_dismiss=True,
            )
            if chosen:
                config = cfg_mgr.load_app_config(chosen)
                self.app.app_config = config  # type: ignore[attr-defined]
                self.app.sync_config_to_server()  # type: ignore[attr-defined]
                self.query_one("#config-form", ConfigForm).load_values(config.to_dict())
                self.app.notify(f"Loaded config '{chosen}'", title="Loaded")

        self.run_worker(_show)

    # ------------------------------------------------------------------
    # Save config
    # ------------------------------------------------------------------

    def _action_save_config(self):
        def on_save(name: str):
            self.app.app_config.name = name  # type: ignore[attr-defined]
            cfg_mgr = get_config_manager()
            cfg_mgr.save_app_config(self.app.app_config)  # type: ignore[attr-defined]
            self.app.sync_config_to_server()  # type: ignore[attr-defined]
            self.app.notify(f"Config saved as '{name}'", title="Saved")

        dialog = SaveConfigDialog(
            current_name=self.app.app_config.name,  # type: ignore[attr-defined]
            on_save=on_save,
        )
        self.mount(dialog)

    # ------------------------------------------------------------------
    # Save & queue
    # ------------------------------------------------------------------

    def _action_save_and_queue(self):
        # Collect form values first so user doesn't have to press Apply separately
        form = self.query_one("#config-form", ConfigForm)
        self.app.app_config = AppConfig.from_dict(form._collect())  # type: ignore[attr-defined]
        cfg_mgr = get_config_manager()
        cfg_mgr.save_app_config(self.app.app_config)  # type: ignore[attr-defined]
        self.app.sync_config_to_server()  # type: ignore[attr-defined]
        client = get_client()
        chapter_sel = self.app.chapter_selection  # type: ignore[attr-defined]
        current_book = self.app.current_book  # type: ignore[attr-defined]
        client.add_job(
            ebook_path=str(current_book),
            voice=self.app.current_voice or "alba",  # type: ignore[attr-defined]
            chapter_selection=chapter_sel.to_dict() if chapter_sel else None,
            output_path=str(current_book.parent) if current_book else None,
        )
        self.app.push_screen(QueueScreen())

    def on_button_pressed(self, event: Button.Pressed):
        bid = event.button.id
        if bid == "btn-back":
            self.app.pop_screen()
        elif bid == "btn-load-config":
            self._action_load_config()
        elif bid == "btn-save-config":
            self._action_save_config()
        elif bid == "btn-queue":
            self.action_push_queue()
        elif bid == "btn-save-queue":
            self._action_save_and_queue()

    def action_push_queue(self):
        self.app.push_screen(QueueScreen())


# ---------------------------------------------------------------------------
# QueueScreen
# ---------------------------------------------------------------------------

QUEUE_HINT = (
    "↑↓ = navigate jobs  •  Enter = select  •  "
    "Add Job to start a new conversion  •  Actions ▾ for job operations"
)


class QueueScreen(Screen):
    BINDINGS = [("q", "go_back", "Back")]

    def compose(self) -> ComposeResult:
        yield Header()
        yield Footer()
        with Container(id="main-content"):
            yield Label("Job Queue", id="title-label")
            yield Static(QUEUE_HINT, id="screen-hint")
            yield DataTable(id="queue-table")
            with Container(id="progress-container"):
                yield Static("", id="current-job-info")
                yield Static("", id="chapter-info")
        with Horizontal(id="bottom-bar"):
            yield Button("← Back", id="btn-back", variant="default")
            yield Button("Add Job", id="btn-add", variant="primary")
            yield Button("Start / Stop", id="btn-startstop", variant="success")
            yield Button("Actions ▾", id="btn-actions", variant="default", disabled=True)

    def on_mount(self):
        table = self.query_one("#queue-table", DataTable)
        table.add_columns("Status", "Book", "Voice", "Progress")
        table.cursor_type = "row"
        self._refresh_queue()

    # ------------------------------------------------------------------
    # Queue refresh
    # ------------------------------------------------------------------

    def _refresh_queue(self):
        try:
            client = get_client()
            queue_info = client.get_queue()
        except Exception:
            return

        table = self.query_one("#queue-table", DataTable)
        table.clear()
        for item in queue_info.items:
            ebook_path = item.job.get("ebook_path", "")
            name = item.job.get("name") or (Path(ebook_path).stem if ebook_path else "Unknown")
            voice = item.job.get("voice", "alba")
            progress = f"{item.progress:.0f}%"
            is_finalising = (
                item.status == "processing" and item.progress >= 100.0 and item.current_chapter
            )
            if is_finalising:
                status_icon = "⚙ finalising"
            else:
                status_icon = {
                    "pending": "⏳ pending",
                    "processing": "▶ processing",
                    "completed": "✓ completed",
                    "failed": "✗ failed",
                    "cancelled": "— cancelled",
                }.get(item.status, item.status)
            table.add_row(status_icon, name, voice, progress)

        # Enable/disable Actions based on row count
        has_rows = len(queue_info.items) > 0
        self.query_one("#btn-actions", Button).disabled = not has_rows

        # Start/Stop button label
        ss = self.query_one("#btn-startstop", Button)
        if queue_info.current_item is not None:
            ss.label = "Stop"
            ss.variant = "warning"
        else:
            ss.label = "Start Queue"
            ss.variant = "success"

        self._update_progress_display(queue_info)

    def _update_progress_display(self, queue_info):
        current = queue_info.current_item
        job_info = self.query_one("#current-job-info", Static)
        chapter_info = self.query_one("#chapter-info", Static)

        parts = []
        if queue_info.pending_count:
            parts.append(f"{queue_info.pending_count} pending")
        if queue_info.completed_count:
            parts.append(f"{queue_info.completed_count} completed")
        if queue_info.failed_count:
            parts.append(f"{queue_info.failed_count} failed")
        summary = "Jobs: " + ", ".join(parts) if parts else "Queue empty"

        if current:
            ebook_path = current.job.get("ebook_path", "")
            book_name = current.job.get("name") or (
                Path(ebook_path).stem if ebook_path else "Unknown"
            )
            # At 100% the chapter field carries a post-TTS phase message
            # (e.g. "Stitching audio files…"). Make that highly visible.
            is_finalising = (
                current.progress >= 100.0
                and current.current_chapter
                and current.status == "processing"
            )
            if is_finalising:
                job_info.update(
                    f"[bold yellow]⚙ {current.current_chapter}[/bold yellow]  "
                    f"[dim]({book_name} — do not close the app)[/dim]  ({summary})"
                )
                chapter_info.update(
                    "[bold yellow]Processing is still running — please wait…[/bold yellow]"
                )
            else:
                job_info.update(
                    f"[bold]Processing:[/bold] {book_name}  "
                    f"[bold]Progress:[/bold] {current.progress:.1f}%  "
                    f"[bold]ETA:[/bold] {self._fmt_eta(current.eta_seconds)}  "
                    f"({summary})"
                )
                chapter_info.update(
                    f"[dim]Chapter: {current.current_chapter}[/dim]"
                    if current.current_chapter
                    else ""
                )
        else:
            job_info.update(f"[dim]No job processing[/dim]  ({summary})")
            chapter_info.update("")

    def _fmt_eta(self, seconds: int) -> str:
        if seconds <= 0:
            return "--:--"
        h, r = divmod(seconds, 3600)
        m, s = divmod(r, 60)
        return f"{h}:{m:02d}:{s:02d}" if h else f"{m}:{s:02d}"

    # ------------------------------------------------------------------
    # Button / event handlers
    # ------------------------------------------------------------------

    def on_button_pressed(self, event: Button.Pressed):
        bid = event.button.id
        logger.debug("QueueScreen button: %s", bid)
        if bid == "btn-back":
            self.app.pop_screen()
        elif bid == "btn-add":
            self.app.push_screen(BookSelectionScreen())
        elif bid == "btn-startstop":
            self._toggle_start_stop()
        elif bid == "btn-actions":
            self._show_actions()

    def on_data_table_row_selected(self, event: DataTable.RowSelected):
        # Enable Actions whenever a row is highlighted
        self.query_one("#btn-actions", Button).disabled = False

    def _toggle_start_stop(self):
        try:
            client = get_client()
            queue_info = client.get_queue()
            if queue_info.current_item is not None:
                client.stop_processing()
                self.app.notify("Processing stopped", title="Stopped")
            else:
                if queue_info.pending_count == 0:
                    self.app.notify("No pending jobs in queue", title="Cannot Start")
                    return
                client.start_processing()
                self.app.notify("Processing started", title="Started")
            self._refresh_queue()
        except Exception as e:
            self.app.notify(f"Error: {e}", title="Error")

    def _show_actions(self):
        try:
            client = get_client()
            queue_info = client.get_queue()
            table = self.query_one("#queue-table", DataTable)
            cursor = table.cursor_row
            if cursor is None or cursor >= len(queue_info.items):
                self.app.notify("Select a job first", title="No Selection")
                return
            item = queue_info.items[cursor]
            ebook_path = item.job.get("ebook_path", "")
            job_name = item.job.get("name") or (Path(ebook_path).stem if ebook_path else "Unknown")
            can_retry = item.status == "failed"
        except Exception as e:
            self.app.notify(f"Error: {e}", title="Error")
            return

        async def _open_modal():
            action = await self.app.push_screen(
                JobActionsModal(job_name=job_name, can_retry=can_retry),
                wait_for_dismiss=True,
            )
            if action == "edit":
                self._edit_job(item)
            elif action == "remove":
                self._remove_job(item.id)
            elif action == "retry":
                self._retry_job(item)
            elif action == "clear":
                self._clear_completed()

        self.run_worker(_open_modal)

    # ------------------------------------------------------------------
    # Job operations
    # ------------------------------------------------------------------

    def _edit_job(self, item):
        """Re-inject job into the wizard flow and remove from queue."""
        try:
            client = get_client()
            ebook_path = item.job.get("ebook_path", "")
            voice = item.job.get("voice", "alba")
            ch_sel_dict = item.job.get("chapter_selection") or {}
            self.app.current_book = Path(ebook_path) if ebook_path else None  # type: ignore[attr-defined]
            self.app.current_voice = voice  # type: ignore[attr-defined]
            self.app.chapter_selection = ChapterSelection.from_dict(ch_sel_dict)  # type: ignore[attr-defined]
            client.remove_job(item.id)
            self.app.push_screen(ChapterSelectionScreen())
        except Exception as e:
            self.app.notify(f"Error editing job: {e}", title="Error")

    def _remove_job(self, job_id: str):
        try:
            client = get_client()
            client.remove_job(job_id)
            self._refresh_queue()
            self.app.notify("Job removed", title="Removed")
        except Exception as e:
            self.app.notify(f"Error: {e}", title="Error")

    def _retry_job(self, item):
        try:
            client = get_client()
            client.add_job(
                ebook_path=item.job.get("ebook_path", ""),
                voice=item.job.get("voice", "alba"),
                chapter_selection=item.job.get("chapter_selection"),
                output_path=item.job.get("output_path"),
                name=item.job.get("name"),
            )
            client.remove_job(item.id)
            self._refresh_queue()
            self.app.notify("Job re-queued", title="Retry")
        except Exception as e:
            self.app.notify(f"Error: {e}", title="Error")

    def _clear_completed(self):
        try:
            client = get_client()
            queue_info = client.get_queue()
            removed = 0
            for item in queue_info.items:
                if item.status in ("completed", "failed", "cancelled"):
                    client.remove_job(item.id)
                    removed += 1
            self._refresh_queue()
            self.app.notify(f"Cleared {removed} finished job(s)", title="Cleared")
        except Exception as e:
            self.app.notify(f"Error: {e}", title="Error")

    def action_go_back(self):
        self.app.pop_screen()
