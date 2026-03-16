from __future__ import annotations

import multiprocessing
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical
from textual.reactive import reactive
from textual.screen import ModalScreen
from textual.widgets import (
    Button,
    DataTable,
    Input,
    Label,
    OptionList,
    RadioButton,
    RadioSet,
    Static,
)
from textual.widgets.option_list import Option

from .models import _normalize_bitrate

# ---------------------------------------------------------------------------
# Shared data structures
# ---------------------------------------------------------------------------


@dataclass
class SelectableItem:
    id: str
    label: str
    description: str = ""


# ---------------------------------------------------------------------------
# UnifiedSelector — generic DataTable-backed item picker
# ---------------------------------------------------------------------------


class UnifiedSelector(Static):
    def __init__(
        self,
        items: list[SelectableItem],
        on_select: Callable[[SelectableItem], None],
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.items = items
        self.on_select = on_select
        self._table: DataTable | None = None

    def compose(self) -> ComposeResult:
        yield DataTable(id="selector-table")

    def on_mount(self):
        self._table = self.query_one("#selector-table", DataTable)
        self._table.add_columns("#", "Name", "Description")
        for i, item in enumerate(self.items, 1):
            self._table.add_row(str(i), item.label, item.description)
        self._table.focus()

    def on_data_table_row_selected(self, event):
        idx = event.cursor_row
        if 0 <= idx < len(self.items):
            self.on_select(self.items[idx])


# ---------------------------------------------------------------------------
# ChapterPresetSelector
# ---------------------------------------------------------------------------


class ChapterPresetSelector(Static):
    def __init__(self, on_change: Callable[[str], None], **kwargs):
        super().__init__(**kwargs)
        self.on_change = on_change

    def compose(self) -> ComposeResult:
        with RadioSet(id="preset-select"):
            yield RadioButton("Content Only (default)", id="content-only")
            yield RadioButton("Main Chapters Only", id="chapters-only")
            yield RadioButton("With Part Dividers", id="with-parts")
            yield RadioButton("All Chapters", id="manual")
            yield RadioButton("None", id="none")

    def on_mount(self):
        self.query_one("#content-only", RadioButton).value = True

    def on_radio_set_changed(self, event):
        self.on_change(event.pressed.id)


# ---------------------------------------------------------------------------
# VoicePickerModal
# ---------------------------------------------------------------------------


class VoicePickerModal(ModalScreen):
    """Modal voice picker with inline custom-path input.

    Dismisses with the selected voice name/path string, or ``None`` if cancelled.
    """

    CSS = """
    VoicePickerModal {
        align: center middle;
    }
    #voice-picker-container {
        width: 80;
        height: auto;
        max-height: 40;
        border: solid $primary;
        padding: 1 2;
        background: $surface;
    }
    #voice-picker-title {
        text-style: bold;
        margin-bottom: 1;
    }
    #voice-picker-hint {
        color: $text-muted;
        margin-bottom: 1;
    }
    #voice-picker-table {
        height: 18;
        border: solid $panel;
        margin-bottom: 1;
    }
    #custom-row {
        height: auto;
        display: none;
        margin-bottom: 1;
    }
    #custom-row.visible {
        display: block;
    }
    #custom-input-row {
        height: auto;
    }
    #custom-voice-input {
        width: 1fr;
        margin-right: 1;
    }
    #voice-picker-buttons {
        height: auto;
        margin-top: 1;
        align: right middle;
    }
    #voice-picker-buttons Button {
        margin-left: 1;
    }
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._selected_voice: str | None = None
        self._custom_visible = False

    def compose(self) -> ComposeResult:
        with Vertical(id="voice-picker-container"):
            yield Label("Select a Voice", id="voice-picker-title")
            yield Label(
                "↑↓ navigate  •  Enter / click to select  •  "
                "Choose 'Custom…' to enter a file path or hf:// URL  •  Escape to cancel",
                id="voice-picker-hint",
            )
            yield DataTable(id="voice-picker-table")
            with Vertical(id="custom-row"):
                yield Label("Path or hf:// URL:")
                with Horizontal(id="custom-input-row"):
                    yield Input(
                        placeholder="e.g. /path/to/voice.wav  or  hf://user/repo/voice.wav",
                        id="custom-voice-input",
                    )
                    yield Button("Browse…", id="btn-browse-voice", variant="default")
            with Horizontal(id="voice-picker-buttons"):
                yield Button("Cancel", id="btn-cancel-voice", variant="default")
                yield Button("Select", id="btn-select-voice", variant="success")

    def on_mount(self):
        from .helpers import get_bundled_voices
        from .utils import DEFAULT_VOICES, VOICE_DESCRIPTIONS

        table = self.query_one("#voice-picker-table", DataTable)
        table.add_columns("#", "Voice", "Description")

        voices = []
        for v in DEFAULT_VOICES:
            voices.append((v, f"Built-in — {VOICE_DESCRIPTIONS.get(v, 'Default voice')}"))
        for v in get_bundled_voices():
            if v.lower() not in ("default.txt",):
                voices.append((v.replace(".wav", ""), "Bundled voice"))
        voices.append(("Custom…", "Enter a file path or hf:// URL"))

        self._voices = voices
        for i, (name, desc) in enumerate(voices, 1):
            table.add_row(str(i), name, desc)

        table.cursor_type = "row"
        table.focus()

    def on_data_table_row_selected(self, event: DataTable.RowSelected):
        idx = event.cursor_row
        if idx is None or idx >= len(self._voices):
            return
        name = self._voices[idx][0]
        if name == "Custom…":
            self._show_custom_row()
        else:
            self._selected_voice = name
            self._hide_custom_row()

    def _show_custom_row(self):
        self._custom_visible = True
        self.query_one("#custom-row").add_class("visible")
        self.query_one("#custom-voice-input", Input).focus()

    def _hide_custom_row(self):
        self._custom_visible = False
        self.query_one("#custom-row").remove_class("visible")

    def on_button_pressed(self, event: Button.Pressed):
        bid = event.button.id
        if bid == "btn-cancel-voice":
            self.dismiss(None)
        elif bid == "btn-select-voice":
            self._confirm_selection()
        elif bid == "btn-browse-voice":
            self._browse_for_voice()

    def _confirm_selection(self):
        if self._custom_visible:
            val = self.query_one("#custom-voice-input", Input).value.strip()
            if val:
                self.dismiss(val)
            else:
                self.dismiss(None)
        elif self._selected_voice:
            self.dismiss(self._selected_voice)
        else:
            self.dismiss(None)

    def _browse_for_voice(self):
        """Open a file picker for browsing voice files."""
        from textual_fspicker import FileOpen

        async def _pick():
            result = await self.app.push_screen(
                FileOpen(location="~", title="Select Voice File"),
                wait_for_dismiss=True,
            )
            if result:
                self.query_one("#custom-voice-input", Input).value = str(result)

        self.run_worker(_pick)

    def on_key(self, event):
        if event.key == "escape":
            self.dismiss(None)


# ---------------------------------------------------------------------------
# ConfigForm
# ---------------------------------------------------------------------------


class ConfigForm(Static):
    def __init__(
        self,
        on_save: Callable[[dict[str, Any]], None],
        initial_values: dict[str, Any] | None = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.on_save = on_save
        self.initial_values = initial_values or {}

    def compose(self) -> ComposeResult:
        default_workers = max(2, multiprocessing.cpu_count() - 2)
        with Vertical(id="config-form-inner", classes="config-form"):
            yield Label(
                "Workers  (default: CPU count − 2)\n"
                "Parallel chapters. More = faster, but each worker loads the model into RAM.\n"
                "  1 → lowest RAM   |   2–4 → recommended   |   6+ → fastest (needs RAM)"
            )
            yield Input(
                value=str(self.initial_values.get("workers", default_workers)),
                id="workers",
            )
            yield Label("Verbose output  (logs extra diagnostic information):")
            with RadioSet(id="verbose-select"):
                yield RadioButton("Yes", id="verbose-yes")
                yield RadioButton("No", id="verbose-no")
            yield Label("Log file path (optional — leave blank to disable file logging):")
            yield Input(
                value=str(self.initial_values.get("log_path", "") or ""),
                id="log_path",
            )
            yield Label(
                "M4B audio bitrate  (default 96k — always include the 'k')\n"
                "Controls output file size and fidelity. AAC is efficient;"
                " 96k is near-transparent for speech with modest file sizes.\n"
                "  32k → smallest, audible artifacts   |   64k → good (Audible standard)"
                "   |   96k → excellent (recommended)   |   128k → best, ~33% larger than 96k"
            )
            yield Input(
                value=str(self.initial_values.get("m4b_bitrate", "96k")),
                id="m4b_bitrate",
            )
            yield Label(
                "Pause between lines (ms, default 400)\n"
                "Silence inserted between paragraphs. Increase for a slower, more relaxed"
                " narration pace; decrease to keep things moving."
            )
            yield Input(
                value=str(self.initial_values.get("pause_line_ms", 400)),
                id="pause_line_ms",
            )
            yield Label(
                "Pause between chapters (ms, default 2000)\n"
                "Silence inserted at the end of each chapter."
            )
            yield Input(
                value=str(self.initial_values.get("pause_chapter_ms", 2000)),
                id="pause_chapter_ms",
            )
            yield Label("─── Voice / Audio Quality ───", id="voice-header")
            yield Label(
                "Temperature  (default 0.7)\n"
                "Controls how expressive the voice is. Lower = more stable and consistent"
                " speech; higher = more animated and varied, but can sound less reliable.\n"
                "  0.4–0.6 → calm and even   |   0.7 → default (recommended)"
                "   |   0.9–1.2 → expressive   |   >1.2 → unpredictable",
                id="temp-label",
            )
            yield Input(
                value=str(self.initial_values.get("temp", 0.7)),
                id="temp",
            )
            yield Label(
                "Decode Steps / Quality  (default 1, must be a positive integer)\n"
                "Number of Lagrangian Self Distillation steps. More steps = richer, more"
                " natural audio at the cost of generation speed.\n"
                "  1 → fastest, good quality (default)   |   2–3 → noticeably better"
                "   |   4–5 → best quality, ~2–5× slower",
                id="lsd-label",
            )
            yield Input(
                value=str(self.initial_values.get("lsd_decode_steps", 1)),
                id="lsd_decode_steps",
            )
            yield Label(
                "Noise Clamp  (default: off — leave blank to disable)\n"
                "Caps the noise amplitude during generation. Useful if you hear occasional"
                " audio glitches or crackling. Recommended value if needed: 3.0.\n"
                "  blank / 0 → disabled (default)   |   3.0 → mild clamping"
                "   |   1.5 → aggressive clamping (may reduce expressiveness)",
                id="noise-clamp-label",
            )
            yield Input(
                value=str(self.initial_values.get("noise_clamp", "") or ""),
                placeholder="Leave blank to disable",
                id="noise_clamp",
            )
            yield Label(
                "─── Job Defaults  (used by CLI / headless mode) ───",
                id="job-defaults-header",
            )
            yield Label(
                "Default Voice  — used when running from the command line.\n"
                "Click 'Choose Voice…' to browse the full voice list."
            )
            with Horizontal(id="default-voice-row"):
                yield Input(
                    value=str(self.initial_values.get("default_voice", "alba")),
                    id="default_voice",
                )
                yield Button("Choose Voice…", id="btn-pick-voice", variant="default")
            yield Label("Default Chapter Preset  — chapter filter applied for CLI conversions:")
            with RadioSet(id="default-preset-select"):
                yield RadioButton("Content Only (default)", id="dp-content-only")
                yield RadioButton("Main Chapters Only", id="dp-chapters-only")
                yield RadioButton("With Part Dividers", id="dp-with-parts")
                yield RadioButton("All Chapters", id="dp-all")
                yield RadioButton("None", id="dp-none")
            yield Label("Default Output Directory  (leave blank to use the book's folder):")
            with Horizontal(id="default-output-row"):
                yield Input(
                    value=str(self.initial_values.get("default_output_dir", "") or ""),
                    placeholder="Leave blank to use book's directory",
                    id="default_output_dir",
                )
                yield Button("Browse…", id="btn-browse-output", variant="default")
            yield Button("Apply", id="btn-save", variant="primary")

    def on_mount(self):
        verbose = self.initial_values.get("verbose", False)
        self.query_one("#verbose-yes" if verbose else "#verbose-no", RadioButton).value = True
        # Set default preset radio
        preset = self.initial_values.get("default_chapter_preset", "content-only")
        preset_id_map = {
            "content-only": "dp-content-only",
            "chapters-only": "dp-chapters-only",
            "with-parts": "dp-with-parts",
            "all": "dp-all",
            "none": "dp-none",
        }
        radio_id = preset_id_map.get(preset, "dp-content-only")
        self.query_one(f"#{radio_id}", RadioButton).value = True

    def on_button_pressed(self, event):
        bid = event.button.id
        if bid == "btn-save":
            self._save()
        elif bid == "btn-pick-voice":
            self._open_voice_picker()
        elif bid == "btn-browse-output":
            self._browse_output_dir()

    def _open_voice_picker(self):
        """Open VoicePickerModal and populate the default_voice input on selection."""

        async def _pick():
            result = await self.app.push_screen(VoicePickerModal(), wait_for_dismiss=True)
            if result:
                self.query_one("#default_voice", Input).value = result

        self.run_worker(_pick)

    def _browse_output_dir(self):
        """Open directory picker for the default output directory."""
        from textual_fspicker import SelectDirectory

        async def _pick():
            result = await self.app.push_screen(
                SelectDirectory(location="~", title="Select Output Directory"),
                wait_for_dismiss=True,
            )
            if result:
                self.query_one("#default_output_dir", Input).value = str(result)

        self.run_worker(_pick)

    def _collect(self) -> dict[str, Any]:
        """Collect current form values into a dict (does not call on_save)."""
        try:
            workers = int(self.query_one("#workers", Input).value)
        except ValueError:
            workers = max(2, multiprocessing.cpu_count() - 2)
        try:
            pause_line = int(self.query_one("#pause_line_ms", Input).value)
        except ValueError:
            pause_line = 400
        try:
            pause_ch = int(self.query_one("#pause_chapter_ms", Input).value)
        except ValueError:
            pause_ch = 2000
        try:
            temp = float(self.query_one("#temp", Input).value)
        except ValueError:
            temp = 0.7
        try:
            quality = int(self.query_one("#lsd_decode_steps", Input).value)
            quality = max(1, quality)
        except ValueError:
            quality = 1

        noise_clamp_raw = self.query_one("#noise_clamp", Input).value.strip()
        try:
            noise_clamp: float | None = float(noise_clamp_raw) if noise_clamp_raw else None
            # Treat 0 as disabled
            if noise_clamp is not None and noise_clamp <= 0:
                noise_clamp = None
        except ValueError:
            noise_clamp = None

        # Default chapter preset from the dp-* radio set
        preset_id_to_value = {
            "dp-content-only": "content-only",
            "dp-chapters-only": "chapters-only",
            "dp-with-parts": "with-parts",
            "dp-all": "all",
            "dp-none": "none",
        }
        default_chapter_preset = "content-only"
        for radio_id, preset_value in preset_id_to_value.items():
            try:
                if self.query_one(f"#{radio_id}", RadioButton).value:
                    default_chapter_preset = preset_value
                    break
            except Exception:
                pass

        default_output_raw = self.query_one("#default_output_dir", Input).value.strip()

        return {
            "workers": workers,
            "verbose": self.query_one("#verbose-yes", RadioButton).value,
            "log_path": self.query_one("#log_path", Input).value.strip() or None,
            "m4b_bitrate": _normalize_bitrate(
                self.query_one("#m4b_bitrate", Input).value, default="96k"
            ),
            "pause_line_ms": pause_line,
            "pause_chapter_ms": pause_ch,
            "temp": temp,
            "lsd_decode_steps": quality,
            "noise_clamp": noise_clamp,
            "default_voice": self.query_one("#default_voice", Input).value.strip() or "alba",
            "default_chapter_preset": default_chapter_preset,
            "default_output_dir": default_output_raw or None,
        }

    def _save(self):
        self.on_save(self._collect())

    def load_values(self, data: dict[str, Any]) -> None:
        """Repopulate form fields from a data dict (e.g. after loading a config)."""
        default_workers = max(2, multiprocessing.cpu_count() - 2)

        self.query_one("#workers", Input).value = str(data.get("workers", default_workers))
        verbose = data.get("verbose", False)
        self.query_one("#verbose-yes" if verbose else "#verbose-no", RadioButton).value = True
        self.query_one("#log_path", Input).value = str(data.get("log_path", "") or "")
        self.query_one("#m4b_bitrate", Input).value = _normalize_bitrate(
            data.get("m4b_bitrate"), default="96k"
        )
        self.query_one("#pause_line_ms", Input).value = str(data.get("pause_line_ms", 400))
        self.query_one("#pause_chapter_ms", Input).value = str(data.get("pause_chapter_ms", 2000))
        self.query_one("#temp", Input).value = str(data.get("temp", 0.7))
        self.query_one("#lsd_decode_steps", Input).value = str(data.get("lsd_decode_steps", 1))
        noise_clamp = data.get("noise_clamp")
        self.query_one("#noise_clamp", Input).value = (
            str(noise_clamp) if noise_clamp is not None else ""
        )
        # Job Defaults
        self.query_one("#default_voice", Input).value = str(data.get("default_voice", "alba"))
        preset = data.get("default_chapter_preset", "content-only")
        preset_id_map = {
            "content-only": "dp-content-only",
            "chapters-only": "dp-chapters-only",
            "with-parts": "dp-with-parts",
            "all": "dp-all",
            "none": "dp-none",
        }
        radio_id = preset_id_map.get(preset, "dp-content-only")
        try:
            self.query_one(f"#{radio_id}", RadioButton).value = True
        except Exception:
            pass
        self.query_one("#default_output_dir", Input).value = str(
            data.get("default_output_dir", "") or ""
        )
        # Fire on_save so the app picks up the new values immediately
        self.on_save(self._collect())


# ---------------------------------------------------------------------------
# ProgressDisplay
# ---------------------------------------------------------------------------


class ProgressDisplay(Static):
    elapsed_seconds: reactive[int] = reactive(0)
    eta_seconds: reactive[int] = reactive(0)
    current_chapter: reactive[str] = reactive("")

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._running = False

    def compose(self) -> ComposeResult:
        yield Static("Idle", id="progress-status")

    def watch_elapsed_seconds(self, _: int):
        self._update_display()

    def watch_eta_seconds(self, _: int):
        self._update_display()

    def watch_current_chapter(self, _: str):
        self._update_display()

    def _update_display(self):
        try:
            status = self.query_one("#progress-status", Static)
        except Exception:
            return
        if not self._running:
            status.update("Idle")
            return
        elapsed = self._format_time(self.elapsed_seconds)
        eta = self._format_time(self.eta_seconds)
        chapter = self.current_chapter[:40] if self.current_chapter else "—"
        status.update(f"Elapsed: {elapsed}  |  ETA: {eta}  |  {chapter}")

    def _format_time(self, seconds: int) -> str:
        h = seconds // 3600
        m = (seconds % 3600) // 60
        s = seconds % 60
        return f"{h:02d}:{m:02d}:{s:02d}"

    def start(self, eta: int = 0):
        self._running = True
        self.eta_seconds = eta
        self.elapsed_seconds = 0

    def stop(self):
        self._running = False
        self._update_display()

    def update(self, chapter: str, eta: int):  # type: ignore[override]
        self.current_chapter = chapter
        self.eta_seconds = eta


# ---------------------------------------------------------------------------
# PathSelectionDialog
# ---------------------------------------------------------------------------


class PathSelectionDialog(Static):
    """Inline path selection widget (file / directory / URL input)."""

    def __init__(
        self,
        mode: str = "directory",
        title: str = "Select Path",
        initial_dir: str = "~",
        filetypes: dict[str, list[str]] | None = None,
        on_select: Callable[[str], None] | None = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.mode = mode
        self.title = title
        self.initial_dir = initial_dir
        self.filetypes = filetypes or {}
        self.on_select = on_select

    def compose(self) -> ComposeResult:
        with Vertical(id="path-dialog"):
            yield Label(self.title, id="dialog-title")
            yield Input(placeholder="Enter path or URL...", id="path-input")
            with Horizontal(id="dialog-buttons"):
                if self.mode in ("directory", "both"):
                    yield Button("Browse Directory…", id="btn-browse-dir", variant="primary")
                if self.mode in ("file", "both"):
                    yield Button("Browse File…", id="btn-browse-file", variant="primary")
                yield Button("Cancel", id="btn-cancel", variant="default")
                yield Button("OK", id="btn-ok", variant="success")

    def on_button_pressed(self, event):
        bid = event.button.id
        if bid == "btn-cancel":
            self.remove()
        elif bid == "btn-browse-dir":
            self._browse_directory()
        elif bid == "btn-browse-file":
            self._browse_file()
        elif bid == "btn-ok":
            self._submit()

    def _browse_directory(self):
        import tkinter as tk
        from tkinter import filedialog

        root = tk.Tk()
        root.withdraw()
        path = filedialog.askdirectory(initialdir=self.initial_dir)
        root.destroy()
        if path:
            self.query_one("#path-input", Input).value = path

    def _browse_file(self):
        import tkinter as tk
        from tkinter import filedialog

        filetypes = [(lbl, " ".join(exts)) for lbl, exts in self.filetypes.items()]
        root = tk.Tk()
        root.withdraw()
        path = filedialog.askopenfilename(
            initialdir=self.initial_dir,
            filetypes=filetypes or None,
        )
        root.destroy()
        if path:
            self.query_one("#path-input", Input).value = path

    def _submit(self):
        path = self.query_one("#path-input", Input).value.strip()
        if path and self.on_select:
            self.on_select(path)
        self.remove()


# ---------------------------------------------------------------------------
# SaveConfigDialog
# ---------------------------------------------------------------------------


class SaveConfigDialog(Static):
    """Inline dialog for naming and saving a config."""

    def __init__(
        self,
        current_name: str = "default",
        on_save: Callable[[str], None] | None = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.current_name = current_name
        self.on_save = on_save

    def compose(self) -> ComposeResult:
        with Vertical(id="save-config-dialog", classes="dialog"):
            yield Label("Save Config As:", id="dialog-title")
            yield Input(value=self.current_name, id="config-name-input")
            yield Label("Enter a name for this configuration.", id="dialog-hint")
            with Horizontal(id="dialog-buttons"):
                yield Button("Cancel", id="btn-cancel", variant="default")
                yield Button("Save", id="btn-save", variant="success")

    def on_button_pressed(self, event):
        if event.button.id == "btn-cancel":
            self.remove()
        elif event.button.id == "btn-save":
            self._submit()

    def _submit(self):
        name = self.query_one("#config-name-input", Input).value.strip()
        if name and self.on_save:
            self.on_save(name)
        self.remove()


# ---------------------------------------------------------------------------
# LoadConfigDialog — ModalScreen listing saved configs
# ---------------------------------------------------------------------------


class LoadConfigDialog(ModalScreen):
    """Modal that lists saved configs and returns the chosen name."""

    CSS = """
    LoadConfigDialog {
        align: center middle;
    }
    #load-config-container {
        width: 60;
        height: auto;
        max-height: 30;
        border: solid $primary;
        padding: 1 2;
        background: $surface;
    }
    #load-config-title {
        text-style: bold;
        margin-bottom: 1;
    }
    #load-config-list {
        height: 16;
        border: solid $panel;
        margin-bottom: 1;
    }
    #load-config-hint {
        color: $text-muted;
        margin-bottom: 1;
    }
    """

    def __init__(self, config_names: list[str], **kwargs):
        super().__init__(**kwargs)
        self.config_names = config_names

    def compose(self) -> ComposeResult:
        with Vertical(id="load-config-container"):
            yield Label("Load Saved Config", id="load-config-title")
            yield Label(
                "↑↓ navigate  •  Enter / double-click to load  •  Escape to cancel",
                id="load-config-hint",
            )
            opt_list = OptionList(
                *[Option(name, id=name) for name in self.config_names],
                id="load-config-list",
            )
            yield opt_list
            with Horizontal(id="dialog-buttons"):
                yield Button("Cancel", id="btn-cancel", variant="default")
                yield Button("Load", id="btn-load", variant="success")

    def on_mount(self):
        self.query_one("#load-config-list", OptionList).focus()

    def on_option_list_option_selected(self, event: OptionList.OptionSelected):
        self.dismiss(event.option.id)

    def on_button_pressed(self, event):
        if event.button.id == "btn-cancel":
            self.dismiss(None)
        elif event.button.id == "btn-load":
            opt = self.query_one("#load-config-list", OptionList)
            if opt.highlighted is not None:
                self.dismiss(self.config_names[opt.highlighted])
            else:
                self.dismiss(None)

    def on_key(self, event):
        if event.key == "escape":
            self.dismiss(None)


# ---------------------------------------------------------------------------
# JobActionsModal — contextual action menu for a queue job
# ---------------------------------------------------------------------------


class JobActionsModal(ModalScreen):
    """Small action menu that appears when a user presses 'Actions' on a queue row."""

    CSS = """
    JobActionsModal {
        align: center middle;
    }
    #job-actions-container {
        width: 40;
        height: auto;
        border: solid $primary;
        padding: 1 2;
        background: $surface;
    }
    #job-actions-title {
        text-style: bold;
        margin-bottom: 1;
    }
    #job-actions-container Button {
        width: 100%;
        margin-bottom: 1;
    }
    """

    def __init__(self, job_name: str, can_retry: bool = False, **kwargs):
        super().__init__(**kwargs)
        self.job_name = job_name
        self.can_retry = can_retry

    def compose(self) -> ComposeResult:
        with Vertical(id="job-actions-container"):
            yield Label(f"Actions: {self.job_name[:30]}", id="job-actions-title")
            yield Button("Edit Job", id="action-edit", variant="primary")
            yield Button("Remove Job", id="action-remove", variant="error")
            yield Button(
                "Retry Job",
                id="action-retry",
                variant="warning",
                disabled=not self.can_retry,
            )
            yield Button("Clear Completed", id="action-clear", variant="default")
            yield Button("Cancel", id="action-cancel", variant="default")

    def on_mount(self):
        self.query_one("#action-edit", Button).focus()

    def on_button_pressed(self, event):
        bid = event.button.id
        if bid == "action-cancel":
            self.dismiss(None)
        else:
            # Return the action string to the caller
            action_map = {
                "action-edit": "edit",
                "action-remove": "remove",
                "action-retry": "retry",
                "action-clear": "clear",
            }
            self.dismiss(action_map.get(bid))

    def on_key(self, event):
        if event.key == "escape":
            self.dismiss(None)


# ---------------------------------------------------------------------------
# PreviewPanel (kept for any future use; no longer used by screens)
# ---------------------------------------------------------------------------


class PreviewPanel(Static):
    def __init__(self, data: dict[str, Any], **kwargs):
        super().__init__(**kwargs)
        self.data = data

    def compose(self) -> ComposeResult:
        lines = [
            "=== Job Summary ===",
            f"Book:     {self.data.get('book_name', 'N/A')}",
            f"Path:     {self.data.get('book_path', 'N/A')}",
            f"Chapters: {self.data.get('chapter_count', 0)} selected"
            f"  [{self.data.get('chapter_preset', 'N/A')}]",
            f"Voice:    {self.data.get('voice', 'N/A')}",
            f"Output:   {self.data.get('output_path', 'N/A')}",
            "",
            "=== Options ===",
            f"Workers:  {self.data.get('workers', 4)}",
            f"Verbose:  {self.data.get('verbose', False)}",
            f"Bitrate:  {self.data.get('m4b_bitrate', '64k')}",
        ]
        yield Static("\n".join(lines))
