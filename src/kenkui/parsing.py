import multiprocessing
import re
import shutil
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import contextmanager
from pathlib import Path
from typing import List, Dict

import ebooklib
from ebooklib import epub
from bs4 import BeautifulSoup

# Rich Imports
from rich.progress import (
    Progress,
    SpinnerColumn,
    BarColumn,
    TextColumn,
    TimeRemainingColumn,
    MofNCompleteColumn,
)
from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.layout import Layout
from rich import box

# import other files
from .helpers import Chapter, AudioResult, Config, interactive_select
from .workers import worker_process_chapter


class EpubReader:
    def __init__(self, filepath: Path):
        self.filepath = filepath
        self.book = epub.read_epub(str(filepath))

    def get_book_title(self) -> str:
        try:
            metadata = self.book.get_metadata("DC", "title")
            if metadata and len(metadata) > 0:
                return self._sanitize_filename(metadata[0][0])
        except Exception:
            pass
        return self.filepath.stem

    def extract_chapters(self, min_text_len: int = 50) -> List[Chapter]:
        toc_map = self._build_toc_map()
        chapters = []
        for item in self.book.get_items_of_type(ebooklib.ITEM_DOCUMENT):
            file_id = item.get_name()
            toc_title = toc_map.get(file_id)
            soup = BeautifulSoup(item.get_content(), "html.parser")
            self._clean_soup(soup)

            final_title = toc_title if toc_title else self._scrape_title(soup)
            paragraphs = self._extract_paragraphs(soup, final_title)

            if self._is_skippable(final_title, paragraphs, min_text_len):
                continue
            if not final_title:
                final_title = f"Chapter {len(chapters) + 1}"

            chapters.append(
                Chapter(len(chapters) + 1, final_title, self._batch_text(paragraphs))
            )
        return chapters

    def _build_toc_map(self) -> Dict[str, str]:
        toc_map = {}

        def traverse(nodes):
            for node in nodes:
                if isinstance(node, ebooklib.epub.Link):
                    toc_map[node.href.split("#")[0]] = node.title
                elif isinstance(node, ebooklib.epub.Section):
                    traverse(node.children)
                elif isinstance(node, list):
                    traverse(node)

        traverse(self.book.toc)
        return toc_map

    def _clean_soup(self, soup: BeautifulSoup):
        for t in soup.find_all(["sup", "script", "style"]):
            t.decompose()
        for t in soup.find_all(class_=re.compile(r"page-?number|hidden", re.I)):
            t.decompose()
        for t in soup.find_all(attrs={"epub:type": "pagebreak"}):
            t.decompose()

    def _scrape_title(self, soup: BeautifulSoup) -> str:
        lines = [
            self._clean_text(c.get_text(" "))
            for c in soup.find_all(["h1", "h2", "h3", "p", "div", "b"], limit=10)
        ]
        lines = [x for x in lines if x]
        for i, line in enumerate(lines[:5]):
            if re.match(r"^(chapter|prologue|epilogue)\s*(\d+|[ivxlc]+)?$", line, re.I):
                if i + 1 < len(lines):
                    return f"{line}: {lines[i + 1]}"
                return line
        for h in soup.find_all(["h1", "h2"]):
            t = self._clean_text(h.get_text(" "))
            if 3 < len(t) < 100:
                return t
        return ""

    def _extract_paragraphs(self, soup: BeautifulSoup, title_filter: str) -> List[str]:
        texts = []
        for tag in soup.find_all(["p", "div", "h1", "h2", "h3", "li"]):
            if tag.find_parent(["p", "div", "li"]):
                continue
            text = self._clean_text(tag.get_text(" "))
            if len(text) < 3:
                continue
            if title_filter and (text.lower() in title_filter.lower()):
                continue
            if re.match(r"^CHAPTER\s*\d+$", text, re.I):
                continue
            texts.append(text)
        return texts

    @staticmethod
    def _clean_text(text: str) -> str:
        return re.sub(r"\s+", " ", text).strip()

    @staticmethod
    def _batch_text(paragraphs: List[str], max_chars: int = 500) -> List[str]:
        batched, current, curr_len = [], [], 0
        for p in paragraphs:
            if current and (curr_len + len(p) > max_chars):
                batched.append(" ".join(current))
                current, curr_len = [], 0
            current.append(p)
            curr_len += len(p)
        if current:
            batched.append(" ".join(current))
        return batched

    def _is_skippable(self, title: str, paragraphs: List[str], min_len: int) -> bool:
        content = " ".join(paragraphs).lower()
        if len(content) < min_len:
            return True
        skip = ["copyright", "rights reserved", "isbn", "table of contents"]
        if title and any(x in title.lower() for x in skip):
            return True
        if any(x in content[:300] for x in skip):
            return True
        return False

    @staticmethod
    def _sanitize_filename(name: str) -> str:
        return re.sub(r'[\\/*?:"<>|]', "", name).strip()


# --- UI MANAGER ---


class AudioBuilder:
    def __init__(self, config: Config):
        self.cfg = config
        self.temp_dir = Path("temp_audio_build")
        self.console = Console()

    def run(self):
        # 1. Parse EPUB
        self.console.rule(f"[bold blue]1. Parsing: {self.cfg.epub_path.name}")
        reader = EpubReader(self.cfg.epub_path)

        book_title = reader.get_book_title()
        final_output = self._determine_output_path(book_title)

        if final_output.exists():
            self.console.print(
                f"[yellow]Skipping: {final_output.name} already exists.[/yellow]"
            )
            return

        chapters = reader.extract_chapters()
        if not chapters:
            self.console.print("[red]No valid chapters found.")
            return

        # 2. Interactive Chapter Selection
        if self.cfg.interactive_chapters:
            self.console.print("\n[bold]Select Chapters to Include:[/bold]")
            chapters = interactive_select(
                chapters,
                f"Chapters in '{book_title}'",
                self.console,
                lambda c: f"{c.title} ([dim]{len(c.paragraphs)} blocks[/dim])",
            )
            if not chapters:
                self.console.print("[red]No chapters selected. Skipping book.[/red]")
                return

        total_blocks = sum(len(ch.paragraphs) for ch in chapters)
        self.console.print(f"Target: [bold green]{book_title}[/bold green]")
        self.console.print(
            f"Job: {len(chapters)} chapters | {total_blocks} text blocks"
        )

        if self.cfg.debug_html:
            return

        # 3. Generate
        self.console.rule("[bold blue]2. Generating Audio")
        with self._managed_temp_dir():
            results = self._process_parallel(chapters, total_blocks)

            if not results:
                self.console.print("[red]Generation failed.")
                return

            # 4. Stitch
            self.console.rule("[bold blue]3. Stitching Audiobook")
            self._stitch_files(results, final_output)

        self.console.print(
            f"\n[bold green]Finished![/bold green] Saved to: [underline]{final_output}[/underline]\n"
        )

    def _determine_output_path(self, book_title: str) -> Path:
        safe_title = re.sub(r'[\\/*?:"<>|]', "", book_title)
        safe_voice = re.sub(r'[\\/*?:"<>|]', "", self.cfg.voice)

        if not self.cfg.output_path or self.cfg.output_path.is_dir():
            filename = f"{safe_title} ({safe_voice}).m4b"
            base_dir = (
                self.cfg.output_path
                if self.cfg.output_path
                else self.cfg.epub_path.parent
            )
            return base_dir / filename

        return self.cfg.output_path

    def _process_parallel(
        self, chapters: List[Chapter], total_blocks: int
    ) -> List[AudioResult]:
        results = []
        cfg_dict = {
            "voice": self.cfg.voice,
            "pause_line_ms": self.cfg.pause_line_ms,
            "pause_chapter_ms": self.cfg.pause_chapter_ms,
        }

        manager = multiprocessing.Manager()
        queue = manager.Queue()
        worker_state = {}

        layout = Layout()
        layout.split(Layout(name="upper", size=3), Layout(name="lower"))

        overall_progress = Progress(
            SpinnerColumn(),
            BarColumn(bar_width=None),
            MofNCompleteColumn(),
            TextColumn("•"),
            TimeRemainingColumn(),
            expand=True,
        )
        overall_task = overall_progress.add_task(
            "[bold cyan]Total Progress", total=total_blocks
        )

        with ProcessPoolExecutor(max_workers=self.cfg.workers) as pool:
            futures = {}
            for ch in chapters:
                fut = pool.submit(
                    worker_process_chapter, ch, cfg_dict, self.temp_dir, queue
                )
                futures[fut] = ch

            with Live(layout, refresh_per_second=8, console=self.console) as live:
                while True:
                    while not queue.empty():
                        try:
                            msg = queue.get_nowait()
                            event, pid = msg[0], msg[1]
                            if event == "START":
                                worker_state[pid] = {
                                    "title": msg[2],
                                    "total": msg[3],
                                    "current": 0,
                                }
                            elif event == "UPDATE":
                                overall_progress.advance(overall_task, msg[2])
                                if pid in worker_state:
                                    worker_state[pid]["current"] += msg[2]
                            elif event == "DONE":
                                if pid in worker_state:
                                    del worker_state[pid]
                        except Exception:
                            break

                    if overall_progress.tasks[0].finished:
                        break

                    layout["upper"].update(
                        Panel(
                            overall_progress,
                            title="Overall Progress",
                            border_style="blue",
                        )
                    )

                    worker_table = Table(
                        box=box.SIMPLE,
                        show_header=True,
                        header_style="bold magenta",
                        expand=True,
                    )
                    worker_table.add_column("PID", width=6)
                    worker_table.add_column("Chapter", ratio=2)
                    worker_table.add_column("Progress", ratio=1)

                    for pid in sorted(worker_state.keys()):
                        state = worker_state[pid]
                        pct = (
                            (state["current"] / state["total"]) * 100
                            if state["total"] > 0
                            else 0
                        )
                        bar_len = 20
                        filled = int((pct / 100) * bar_len)
                        bar_str = "█" * filled + "░" * (bar_len - filled)
                        worker_table.add_row(
                            str(pid),
                            state["title"][:40],
                            f"[green]{bar_str}[/green] {pct:.0f}%",
                        )

                    active_count = len(worker_state)
                    if active_count < self.cfg.workers:
                        for _ in range(self.cfg.workers - active_count):
                            worker_table.add_row("-", "[dim]Idle[/dim]", "")

                    layout["lower"].update(
                        Panel(
                            worker_table,
                            title=f"Worker Threads ({self.cfg.workers})",
                            border_style="grey50",
                        )
                    )

            for future in as_completed(futures):
                res = future.result()
                if res:
                    results.append(res)

        return sorted(results, key=lambda x: x.chapter_index)

    def _stitch_files(self, results: List[AudioResult], output_file: Path):
        file_list = self.temp_dir / "files.txt"
        meta_file = self.temp_dir / "metadata.txt"

        with open(file_list, "w", encoding="utf-8") as f:
            for res in results:
                f.write(f"file '{res.file_path.resolve().as_posix()}'\n")

        with open(meta_file, "w", encoding="utf-8") as f:
            f.write(";FFMETADATA1\n")
            t = 0
            for res in results:
                start, end = int(t), int(t + res.duration_ms)
                f.write(
                    f"[CHAPTER]\nTIMEBASE=1/1000\nSTART={start}\nEND={end}\ntitle={res.title}\n"
                )
                t += res.duration_ms

        cmd = [
            "ffmpeg",
            "-y",
            "-v",
            "error",
            "-stats",
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            str(file_list),
            "-i",
            str(meta_file),
            "-map_metadata",
            "1",
            "-c:a",
            "aac" if output_file.suffix == ".m4b" else "libmp3lame",
            "-b:a",
            self.cfg.m4b_bitrate if output_file.suffix == ".m4b" else "128k",
        ]
        if output_file.suffix == ".m4b":
            cmd.extend(["-movflags", "+faststart"])
        cmd.append(str(output_file))
        subprocess.run(cmd, check=True)

    @contextmanager
    def _managed_temp_dir(self):
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)
        self.temp_dir.mkdir(parents=True)
        try:
            yield
        finally:
            if not self.cfg.keep_temp and self.temp_dir.exists():
                shutil.rmtree(self.temp_dir)
