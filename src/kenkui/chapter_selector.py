"""
Interactive chapter selection using Textual.
Provides checkbox interface with filter management.
"""

from textual.app import App, ComposeResult
from textual.containers import Horizontal, Vertical
from textual.message import Message
from textual.reactive import reactive
from textual.widgets import (
    Button,
    DataTable,
    Header,
    Footer,
    Input,
    Label,
    Select,
    Static,
)
from textual.binding import Binding

from rich.console import Console

from .helpers import Chapter
from .chapter_classifier import ChapterTags
from .chapter_filter import ChapterFilter, FilterPreset


class FilterPanel(Static):
    """Panel showing active filters and presets."""

    DEFAULT_CSS = """
    FilterPanel {
        width: 100%;
        height: auto;
        padding: 1;
        border: solid $primary;
    }
    FilterPanel .header {
        text-align: center;
        text-style: bold;
        margin-bottom: 1;
    }
    FilterPanel Select {
        width: 100%;
        margin-bottom: 1;
    }
    FilterPanel .preset-info {
        color: $text-muted;
        text-align: center;
        margin-bottom: 1;
    }
    FilterPanel Button {
        width: 100%;
        margin: 1 0;
    }
    """

    current_preset = reactive("content-only")

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.presets = list(ChapterFilter.PRESETS.keys())

    def compose(self) -> ComposeResult:
        yield Label("Filter Preset", classes="header")

        options = [(name, name) for name in self.presets]
        yield Select(options, id="preset-select", value=self.current_preset)

        preset = ChapterFilter.get_preset(self.current_preset)
        yield Static(
            preset.description if preset else "",
            id="preset-desc",
            classes="preset-info",
        )

        yield Label("Quick Actions", classes="header")
        yield Button("Select All", id="btn-select-all", variant="primary")
        yield Button("Select None", id="btn-select-none", variant="primary")
        yield Button("Reset to Preset", id="btn-reset", variant="warning")

    def watch_current_preset(self, preset_name: str) -> None:
        """Update description when preset changes."""
        preset = ChapterFilter.get_preset(preset_name)
        try:
            desc_label = self.query_one("#preset-desc", Static)
            if desc_label and preset:
                desc_label.update(preset.description)
        except Exception:
            pass

    def on_select_changed(self, event: Select.Changed) -> None:
        """Handle preset selection change."""
        if event.value:
            self.current_preset = str(event.value)
            self.post_message(self.PresetChanged(self.current_preset))

    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle button presses."""
        button_id = event.button.id
        if button_id == "btn-select-all":
            self.post_message(self.SelectAll())
        elif button_id == "btn-select-none":
            self.post_message(self.SelectNone())
        elif button_id == "btn-reset":
            self.post_message(self.ResetToPreset())

    class PresetChanged(Message):
        def __init__(self, preset_name: str) -> None:
            self.preset_name = preset_name
            super().__init__()

    class SelectAll(Message):
        pass

    class SelectNone(Message):
        pass

    class ResetToPreset(Message):
        pass


class ChapterSelectorApp(App):
    """Textual app for interactive chapter selection."""

    CSS = """
    Screen { align: center middle; }

    #main-container {
        width: 95%;
        height: 95%;
        border: solid $primary;
    }

    #header-bar {
        height: auto;
        padding: 1;
        background: $surface;
        text-align: center;
        border-bottom: solid $primary;
    }

    #content-area {
        width: 100%;
        height: 1fr;
    }

    #chapter-list {
        width: 70%;
        height: 100%;
        border-right: solid $primary;
    }

    #filter-area {
        width: 30%;
        height: 100%;
        padding: 1;
    }

    #footer-bar {
        height: auto;
        padding: 1;
        background: $surface;
        border-top: solid $primary;
    }

    #search-box {
        margin-bottom: 1;
    }

    DataTable {
        height: 1fr;
        border: none;
    }

    DataTable > .datatable--cursor {
        background: $surface-darken-1;
    }
    """

    BINDINGS = [
        Binding("q", "cancel", "Cancel", show=True),
        Binding("enter", "confirm", "Confirm", show=True),
        Binding("a", "select_all", "Select All", show=True),
        Binding("n", "select_none", "Select None", show=True),
        Binding("r", "reset", "Reset", show=True),
        Binding("f", "focus_search", "Search", show=True),
        Binding("space", "toggle_selection", "Toggle", show=True),
    ]

    def __init__(
        self, chapters: list[Chapter], initial_preset: str = "content-only", **kwargs
    ):
        self.all_chapters = chapters
        self.current_preset = initial_preset
        self.filtered_chapters = ChapterFilter.apply_preset(chapters, initial_preset)
        self.selected_indices: set[int] = set(range(len(self.filtered_chapters)))
        self.search_term: str = ""
        super().__init__(**kwargs)

    def compose(self) -> ComposeResult:
        yield Header(show_clock=False)

        with Vertical(id="main-container"):
            # Header with stats
            total = len(self.all_chapters)
            filtered = len(self.filtered_chapters)
            selected = len(self.selected_indices)
            yield Static(
                f"📚 {total} total chapters | 📋 {filtered} visible | ✅ {selected} selected",
                id="header-bar",
            )

            with Horizontal(id="content-area"):
                # Chapter list with DataTable
                with Vertical(id="chapter-list"):
                    yield Input(
                        placeholder="Search chapters... (press 'f' to focus)",
                        id="search-box",
                    )
                    yield DataTable(id="chapter-table")

                # Filter panel
                with Vertical(id="filter-area"):
                    filter_panel = FilterPanel()
                    filter_panel.current_preset = self.current_preset
                    yield filter_panel

            # Footer with actions
            with Horizontal(id="footer-bar"):
                yield Button("✓ Confirm Selection", id="btn-confirm", variant="success")
                yield Button("✗ Cancel", id="btn-cancel", variant="error")
                yield Static(
                    " | Use arrow keys to navigate, Space to toggle, Enter to confirm"
                )

        yield Footer()

    def on_mount(self) -> None:
        """Populate the data table when the app mounts."""
        table = self.query_one("#chapter-table", DataTable)
        table.add_columns("✓", "#", "Chapter Title", "Tags", "Stats")
        self._populate_table()

    def _populate_table(self) -> None:
        """Populate the table with chapter data."""
        table = self.query_one("#chapter-table", DataTable)
        table.clear()

        for i, chapter in enumerate(self.filtered_chapters):
            tags_str = ", ".join(chapter.tags.get_applied_tags())
            word_count = (
                len(" ".join(chapter.paragraphs).split()) if chapter.paragraphs else 0
            )
            stats_str = f"{len(chapter.paragraphs)}p / ~{word_count}w"

            is_selected = i in self.selected_indices
            checkbox = "☑" if is_selected else "☐"

            table.add_row(
                checkbox,
                str(i + 1),
                chapter.title,
                tags_str,
                stats_str,
                key=str(i),
            )

    def update_header(self) -> None:
        """Update the header bar with current stats."""
        total = len(self.all_chapters)
        filtered = len(self.filtered_chapters)
        selected = len(self.selected_indices)
        self.query_one("#header-bar", Static).update(
            f"📚 {total} total chapters | 📋 {filtered} visible | ✅ {selected} selected"
        )

    def on_data_table_row_selected(self, event: DataTable.RowSelected) -> None:
        """Handle row selection in the table."""
        row_key = event.row_key.value
        if row_key is not None:
            index = int(row_key)
            if index in self.selected_indices:
                self.selected_indices.discard(index)
            else:
                self.selected_indices.add(index)
            self._update_table_row(index)
            self.update_header()

    def _update_table_row(self, index: int) -> None:
        """Update a single row in the table."""
        table = self.query_one("#chapter-table", DataTable)
        chapter = self.filtered_chapters[index]
        tags_str = ", ".join(chapter.tags.get_applied_tags())
        word_count = (
            len(" ".join(chapter.paragraphs).split()) if chapter.paragraphs else 0
        )
        stats_str = f"{len(chapter.paragraphs)}p / ~{word_count}w"

        is_selected = index in self.selected_indices
        checkbox = "☑" if is_selected else "☐"

        table.update_cell(str(index), "✓", checkbox)

    def on_filter_panel_preset_changed(self, event: FilterPanel.PresetChanged) -> None:
        """Handle preset changes."""
        self.current_preset = event.preset_name
        self.filtered_chapters = ChapterFilter.apply_preset(
            self.all_chapters, self.current_preset
        )
        self.selected_indices = set(range(len(self.filtered_chapters)))
        self._populate_table()
        self.update_header()

    def on_filter_panel_select_all(self, event: FilterPanel.SelectAll) -> None:
        """Select all chapters."""
        self.action_select_all()

    def on_filter_panel_select_none(self, event: FilterPanel.SelectNone) -> None:
        """Deselect all chapters."""
        self.action_select_none()

    def on_filter_panel_reset_to_preset(self, event: FilterPanel.ResetToPreset) -> None:
        """Reset to current preset."""
        self.selected_indices = set()
        preset = ChapterFilter.get_preset(self.current_preset)
        if preset:
            for i, chapter in enumerate(self.filtered_chapters):
                if preset.apply(chapter.tags):
                    self.selected_indices.add(i)
        else:
            self.selected_indices = set(range(len(self.filtered_chapters)))
        self._populate_table()
        self.update_header()

    def on_input_changed(self, event: Input.Changed) -> None:
        """Handle search input changes."""
        if event.input.id == "search-box":
            self.search_term = event.value.lower()
            self._apply_search_filter()

    def _apply_search_filter(self) -> None:
        """Filter chapters based on search term."""
        table = self.query_one("#chapter-table", DataTable)

        for i, chapter in enumerate(self.filtered_chapters):
            if not self.search_term or self.search_term in chapter.title.lower():
                # Show row
                table.update_cell(str(i), "#", str(i + 1))
            else:
                # Hide row by clearing content (Textual doesn't support hiding rows)
                table.update_cell(str(i), "#", "")
                table.update_cell(str(i), "Chapter Title", "")
                table.update_cell(str(i), "Tags", "")
                table.update_cell(str(i), "Stats", "")
                table.update_cell(str(i), "✓", "")

    def action_select_all(self) -> None:
        """Select all chapters."""
        self.selected_indices = set(range(len(self.filtered_chapters)))
        self._populate_table()
        self.update_header()

    def action_select_none(self) -> None:
        """Deselect all chapters."""
        self.selected_indices.clear()
        self._populate_table()
        self.update_header()

    def action_toggle_selection(self) -> None:
        """Toggle selection of currently focused row."""
        table = self.query_one("#chapter-table", DataTable)
        cursor_row = table.cursor_row
        if cursor_row is not None and cursor_row < len(self.filtered_chapters):
            if cursor_row in self.selected_indices:
                self.selected_indices.discard(cursor_row)
            else:
                self.selected_indices.add(cursor_row)
            self._update_table_row(cursor_row)
            self.update_header()

    def action_reset(self) -> None:
        """Reset to current preset."""
        self.selected_indices = set()
        preset = ChapterFilter.get_preset(self.current_preset)
        if preset:
            for i, chapter in enumerate(self.filtered_chapters):
                if preset.apply(chapter.tags):
                    self.selected_indices.add(i)
        else:
            self.selected_indices = set(range(len(self.filtered_chapters)))
        self._populate_table()
        self.update_header()

    def action_focus_search(self) -> None:
        """Focus the search box."""
        self.query_one("#search-box", Input).focus()

    def action_confirm(self) -> None:
        """Confirm selection and exit."""
        self.exit(self.selected_indices)

    def action_cancel(self) -> None:
        """Cancel and exit with empty selection."""
        self.exit(set())

    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle button presses."""
        button_id = event.button.id
        if button_id == "btn-confirm":
            self.action_confirm()
        elif button_id == "btn-cancel":
            self.action_cancel()


def interactive_select_chapters(
    chapters: list[Chapter], console: Console, initial_preset: str = "content-only"
) -> list[Chapter]:
    """Launch interactive chapter selector."""
    if not chapters:
        return []

    app = ChapterSelectorApp(chapters, initial_preset)

    try:
        selected_indices = app.run()
    except Exception as e:
        console.print(f"[yellow]Interactive UI failed: {e}[/yellow]")
        console.print("[yellow]Falling back to simple selection...[/yellow]")
        from .helpers import interactive_select

        filtered = ChapterFilter.apply_preset(chapters, initial_preset)
        return interactive_select(
            filtered, "Available Chapters", console, lambda ch: ch.title
        )

    if selected_indices is None:
        return []

    if len(selected_indices) == 0:
        filtered = ChapterFilter.apply_preset(chapters, initial_preset)
        return filtered

    filtered = ChapterFilter.apply_preset(chapters, initial_preset)
    return [filtered[i] for i in sorted(selected_indices)]
