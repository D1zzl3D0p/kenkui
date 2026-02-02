import multiprocessing
import re
import shutil
import subprocess
import warnings
import zipfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import contextmanager
from pathlib import Path
from xml.etree import ElementTree as ET

# Suppress ALL warnings by default (verbose mode will re-enable them)
# This must be before any imports that might issue warnings
warnings.filterwarnings("ignore")

# Specifically catch common EPUB/HTML warnings
warnings.filterwarnings("ignore", message=".*characters could not be decoded.*")
warnings.filterwarnings("ignore", message=".*looks like.*")
warnings.filterwarnings("ignore", message=".*surrogate.*")

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
from .helpers import Chapter, AudioResult, Config
from .workers import worker_process_chapter


class EpubReader:
    def __init__(self, filepath: Path, verbose: bool = False):
        self.filepath = filepath
        self.verbose = verbose
        self.book = epub.read_epub(str(filepath))

    def get_book_title(self) -> str:
        try:
            metadata = self.book.get_metadata("DC", "title")
            if metadata:
                return self._sanitize_filename(metadata[0][0])
        except Exception:
            pass
        return self.filepath.stem

    def find_toc_file(self) -> tuple[str | None, str | None]:
        """
        Find the TOC file (NCX or NAV) in the EPUB.
        Returns tuple of (file_path, toc_type) where toc_type is 'ncx' or 'nav'
        """
        with zipfile.ZipFile(str(self.filepath), "r") as epub_zip:
            # First, find the content.opf file
            namelist = epub_zip.namelist()

            # Look for container.xml to find the OPF file
            container_path = "META-INF/container.xml"
            if container_path in namelist:
                container_xml = epub_zip.read(container_path)
                container_tree = ET.fromstring(container_xml)

                # Find the OPF file path
                ns = {"container": "urn:oasis:names:tc:opendocument:xmlns:container"}
                rootfile = container_tree.find(".//container:rootfile", ns)
                if rootfile is not None:
                    opf_path = rootfile.get("full-path")
                    if opf_path is None:
                        return (None, None)

                    # Read the OPF file
                    opf_xml = epub_zip.read(opf_path)
                    opf_tree = ET.fromstring(opf_xml)

                    # Look for NCX file reference
                    opf_ns = {"opf": "http://www.idpf.org/2007/opf"}
                    ncx_item = opf_tree.find(
                        ".//opf:item[@media-type='application/x-dtbncx+xml']", opf_ns
                    )

                    if ncx_item is not None:
                        ncx_href = ncx_item.get("href")
                        if ncx_href is not None:
                            # Resolve relative path
                            opf_dir = str(Path(opf_path).parent)
                            if opf_dir == ".":
                                ncx_path = ncx_href
                            else:
                                ncx_path = str(Path(opf_dir) / ncx_href)
                            return (ncx_path, "ncx")

                    # Look for NAV file (EPUB3)
                    nav_item = opf_tree.find(".//opf:item[@properties='nav']", opf_ns)
                    if nav_item is not None:
                        nav_href = nav_item.get("href")
                        if nav_href is not None:
                            opf_dir = str(Path(opf_path).parent)
                            if opf_dir == ".":
                                nav_path = nav_href
                            else:
                                nav_path = str(Path(opf_dir) / nav_href)
                            return (nav_path, "nav")

            # Fallback: search for common TOC file names
            for name in namelist:
                if name.endswith(".ncx"):
                    return (name, "ncx")
                if "nav.xhtml" in name.lower() or "toc.xhtml" in name.lower():
                    return (name, "nav")

        return (None, None)

    def extract_raw_toc(self) -> str | None:
        """
        Extract the raw TOC content from the EPUB file.
        Returns the raw XML/XHTML content as a string.
        """
        toc_file, toc_type = self.find_toc_file()

        if toc_file is None:
            return None

        with zipfile.ZipFile(str(self.filepath), "r") as epub_zip:
            toc_content = epub_zip.read(toc_file).decode("utf-8")
            return toc_content

    def get_toc_info(self) -> dict[str, any]:  # type: ignore
        """
        Get information about the TOC file without extracting full content.
        Returns a dict with toc_file path, toc_type, and whether it was found.
        """
        toc_file, toc_type = self.find_toc_file()

        return {
            "found": toc_file is not None,
            "file_path": toc_file,
            "toc_type": toc_type,
            "description": (
                f"{'EPUB3 NAV' if toc_type == 'nav' else 'EPUB2 NCX'} file"
                if toc_file
                else "No TOC file found"
            ),
        }

    def _parse_toc_structure(self) -> list[dict]:
        """Parse the EPUB TOC (NCX or NAV) into a structured list of chapters.

        Returns a list of dicts with keys: title, href, src, is_chapter
        Only returns actual chapters (filters out title pages, volumes, books, etc.)
        """
        toc_file, toc_type = self.find_toc_file()
        chapters = []

        if toc_file is None:
            return chapters

        try:
            with zipfile.ZipFile(str(self.filepath), "r") as epub_zip:
                toc_content = epub_zip.read(toc_file).decode("utf-8")
                toc_tree = ET.fromstring(toc_content)

                if toc_type == "ncx":
                    # Parse NCX format
                    ns = {"ncx": "http://www.daisy.org/z3986/2005/ncx/"}
                    for navpoint in toc_tree.findall(".//ncx:navPoint", ns):
                        navlabel = navpoint.find("ncx:navLabel/ncx:text", ns)
                        title = navlabel.text if navlabel is not None else "Untitled"

                        content = navpoint.find("ncx:content", ns)
                        if content is not None:
                            src = content.get("src", "")
                            href = src.split("#")[0]

                            # Determine if this is an actual chapter
                            is_chapter = self._is_chapter_title(title)

                            chapters.append(
                                {
                                    "title": title,
                                    "href": href,
                                    "src": src,
                                    "is_chapter": is_chapter,
                                }
                            )
                else:
                    # Parse EPUB3 NAV format
                    ns = {"xhtml": "http://www.w3.org/1999/xhtml"}
                    toc_nav = toc_tree.find(".//xhtml:nav[@epub:type='toc']", ns)
                    if toc_nav is None:
                        toc_nav = toc_tree.find(".//nav[@epub:type='toc']")

                    if toc_nav is not None:
                        for link in toc_nav.findall(".//xhtml:a", ns):
                            title = link.text or "Untitled"
                            src = link.get("href", "")
                            href = src.split("#")[0] if src else ""

                            if href:
                                is_chapter = self._is_chapter_title(title)
                                chapters.append(
                                    {
                                        "title": title,
                                        "href": href,
                                        "src": src,
                                        "is_chapter": is_chapter,
                                    }
                                )
        except Exception:
            # If TOC parsing fails, return empty list
            pass

        return chapters

    @staticmethod
    def _is_chapter_title(title: str | None) -> bool:
        """Determine if a TOC entry is an actual chapter (not volume/book/title page).

        Chapters typically start with "CHAPTER" or are numbered sections.
        """
        if title is None:
            return False
        title_upper = title.upper()

        # Skip these types of entries
        # Note: BOOK and VOLUME are intentionally NOT skipped - these headers
        # should appear as separate chapters to help with landmark navigation
        skip_patterns = [
            "CONTENTS",
            "PREFACE",
            "INTRODUCTION",
            "TITLE",
            "COPYRIGHT",
            "TRANSLATED",
            "ILLUSTRATIONS",
            "LES MISÉRABLES",  # Skip duplicate title entries
        ]

        for pattern in skip_patterns:
            if pattern in title_upper:
                return False

        # Check if it looks like a chapter, book, or volume
        # Patterns check for CHAPTER/BOOK/VOLUME at start OR anywhere in title
        chapter_patterns = [
            r"CHAPTER\s+",  # CHAPTER anywhere in title
            r"CHAP\.\s+",
            r"BOOK\s+(FIRST|SECOND|THIRD|FOURTH|FIFTH|SIXTH|SEVENTH|EIGHTH|NINTH|TENTH|\d+|I+|II+|III+|IV+|V+|VI+|VII+|VIII+|IX+|X+)",
            r"VOLUME\s+(FIRST|SECOND|THIRD|FOURTH|FIFTH|SIXTH|SEVENTH|EIGHTH|NINTH|TENTH|\d+|I+|II+|III+|IV+|V+|VI+|VII+|VIII+|IX+|X+)",
            r"^\d+[\s\.—:]+",  # Starts with number followed by punctuation
            r"^I+[\s\.—:]+",  # Roman numerals followed by punctuation
        ]

        for pattern in chapter_patterns:
            if re.search(pattern, title, re.IGNORECASE):
                return True

        # If we have very few chapters, be more permissive
        return False

    def extract_chapters(self, min_text_len: int = 50) -> list[Chapter]:
        """Extract chapters using TOC as ground truth, falling back to regex detection."""
        # Capture warnings during chapter extraction, only show in verbose mode
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            # Try to use TOC as ground truth
            toc_chapters = self._parse_toc_structure()

        if toc_chapters:
            return self._extract_chapters_from_toc(toc_chapters, min_text_len)
        else:
            # Fallback to legacy regex-based extraction
            return self._extract_chapters_legacy(min_text_len)

    def _extract_chapters_from_toc(
        self, toc_chapters: list[dict], min_text_len: int
    ) -> list[Chapter]:
        """Extract chapters using TOC structure as ground truth."""
        chapters = []

        # Build a map of which TOC chapter each file belongs to
        file_to_chapter_idx: dict[str, int] = {}
        for idx, ch in enumerate(toc_chapters):
            href = ch["href"]
            if href not in file_to_chapter_idx:
                file_to_chapter_idx[href] = idx

        # Track paragraphs for each chapter
        chapter_paragraphs: dict[int, list[str]] = {
            i: [] for i in range(len(toc_chapters))
        }
        current_chapter_idx = -1

        # Get all items in reading order
        for item in self.book.get_items():
            if not hasattr(item, "get_content") or not hasattr(item, "get_name"):
                continue

            item_name = item.get_name()

            # Check which chapter this file belongs to
            if item_name in file_to_chapter_idx:
                current_chapter_idx = file_to_chapter_idx[item_name]

            try:
                content = item.get_content()
            except Exception:
                continue

            soup = BeautifulSoup(content, "html.parser")
            self._clean_soup(soup)

            # Extract text from all paragraphs and divs
            for elem in soup.find_all(["p", "div"]):
                # Skip nested elements
                if elem.find_parent(["p", "div"]):
                    continue

                text = self._clean_text(elem.get_text(" "))
                if text and len(text) >= 2:
                    # Add to current chapter if we have one
                    if current_chapter_idx >= 0 and current_chapter_idx < len(
                        toc_chapters
                    ):
                        chapter_paragraphs[current_chapter_idx].append(text)

        # Create Chapter objects from TOC with accumulated paragraphs
        chapter_idx = 1
        for toc_idx, toc_ch in enumerate(toc_chapters):
            paragraphs = chapter_paragraphs[toc_idx]

            # Skip non-chapter entries and empty chapters
            if not toc_ch["is_chapter"]:
                continue

            # Check if this is a book/volume header (should be included even if empty)
            is_book_header = bool(
                re.search(
                    r"^BOOK\s+(FIRST|SECOND|THIRD|FOURTH|FIFTH|SIXTH|SEVENTH|EIGHTH|NINTH|TENTH|I+|II+|III+|IV+|V+|VI+|VII+|VIII+|IX+|X+)",
                    toc_ch["title"],
                    re.IGNORECASE,
                )
            )
            is_volume_header = bool(
                re.search(
                    r"^VOLUME\s+(FIRST|SECOND|THIRD|FOURTH|FIFTH|SIXTH|SEVENTH|EIGHTH|NINTH|TENTH|I+|II+|III+|IV+|V+|VI+|VII+|VIII+|IX+|X+|\d+)",
                    toc_ch["title"],
                    re.IGNORECASE,
                )
            )

            # Include book/volume headers even if they have no paragraphs (they're landmarks)
            if is_book_header or is_volume_header:
                chapters.append(Chapter(chapter_idx, toc_ch["title"], []))
                chapter_idx += 1
            elif paragraphs:
                content = " ".join(paragraphs)
                if len(content) >= min_text_len:
                    # Clean up paragraphs (remove title from first paragraph if repeated)
                    if paragraphs and toc_ch["title"].lower().endswith(
                        paragraphs[0].lower()
                    ):
                        paragraphs = paragraphs[1:]

                    # Store original paragraphs for accurate progress tracking
                    # Batching happens in the worker, not here
                    chapters.append(Chapter(chapter_idx, toc_ch["title"], paragraphs))
                    chapter_idx += 1

        return chapters

    def _extract_chapters_legacy(self, min_text_len: int = 50) -> list[Chapter]:
        """Legacy regex-based chapter extraction (fallback when no TOC)."""
        toc_map = self._build_toc_map()
        chapters = []

        # Track hierarchy for books like Les Mis
        current_vol = ""
        current_book = ""

        for item in self.book.get_items():
            # Only process HTML documents
            if not hasattr(item, "get_content") or not hasattr(item, "get_name"):
                continue
            try:
                content = item.get_content()
            except Exception:
                continue
            soup = BeautifulSoup(content, "html.parser")
            self._clean_soup(soup)

            # We iterate through all block-level elements
            elements = soup.find_all(["h1", "h2", "h3", "h4", "p", "div", "section"])

            current_chapter_title = toc_map.get(item.get_name(), "")
            current_paragraphs = []

            for elem in elements:
                # Avoid processing nested tags twice
                if elem.find_parent(["p", "h1", "h2", "h3", "h4", "section"]):
                    continue

                text = self._clean_text(elem.get_text(" "))
                if not text or len(text) < 2:
                    continue

                # 1. Check for Volume/Book markers to update hierarchy
                if re.match(r"^(volume|part)\s+[ivxlcdm\d]+", text, re.I):
                    current_vol = text
                    continue
                if re.match(r"^book\s+(?:the\s+)?(?:[ivxlcdm\d]+|[a-z]+)", text, re.I):
                    current_book = text
                    continue

                # 2. Check for Chapter markers
                # Regex looks for "Chapter X" or "X." at start of line
                chap_match = re.match(
                    r"^(chapter\s+[ivxlcdm\d]+|(?=[IVXLCDM]+\.)[IVXLCDM]+)([\.\-\—\s:]+)(.*)$",
                    text,
                    re.I,
                )

                if chap_match:
                    # If we had content, save it as the previous chapter
                    if current_paragraphs:
                        self._add_chapter(
                            chapters,
                            current_chapter_title,
                            current_paragraphs,
                            min_text_len,
                        )

                    header_label = chap_match.group(1)  # e.g., "CHAPTER I"
                    remaining_text = chap_match.group(
                        3
                    ).strip()  # Title text + potentially body text

                    # Construct a full title: "Vol 1, Book 1, Chapter I: Title"
                    prefix = f"{current_vol}, " if current_vol else ""
                    prefix += f"{current_book}, " if current_book else ""

                    # Split if the tag contains both title and body
                    # If remaining_text is very long and contains a sentence break, split it.
                    if len(remaining_text) > 200 and "." in remaining_text:
                        # Split at first period followed by space
                        parts = re.split(r"(?<=\.)\s+", remaining_text, maxsplit=1)
                        current_chapter_title = f"{prefix}{header_label}: {parts[0]}"
                        current_paragraphs = [parts[1]] if len(parts) > 1 else []
                    else:
                        current_chapter_title = (
                            f"{prefix}{header_label}: {remaining_text}"
                        )
                        current_paragraphs = []
                else:
                    # It's just normal body text
                    current_paragraphs.append(text)

            # Close the last chapter of the file
            if current_paragraphs:
                self._add_chapter(
                    chapters, current_chapter_title, current_paragraphs, min_text_len
                )

        return chapters

    def _add_chapter(
        self, chapters: list[Chapter], title: str, paragraphs: list[str], min_len: int
    ):
        content = " ".join(paragraphs)
        if len(content) < min_len:
            return

        # Fallback for untitled sections
        final_title = title if title else f"Chapter {len(chapters) + 1}"

        # Clean up cases where title is just a repeat of first paragraph
        if paragraphs and final_title.lower().endswith(paragraphs[0].lower()):
            paragraphs = paragraphs[1:]

        # Store original paragraphs for accurate progress tracking
        # Batching happens in the worker, not here
        chapters.append(Chapter(len(chapters) + 1, final_title, paragraphs))

    def _build_toc_map(self) -> dict[str, str]:
        toc_map = {}

        def traverse(nodes):
            for node in nodes:
                if hasattr(node, "href") and hasattr(node, "title"):  # Link-like object
                    toc_map[node.href.split("#")[0]] = node.title
                elif hasattr(node, "children"):  # Section-like object
                    traverse(node.children)

        if hasattr(self.book, "toc"):
            traverse(self.book.toc)
        return toc_map

    def _clean_soup(self, soup: BeautifulSoup):
        # Remove common non-story elements
        for t in soup.find_all(["sup", "script", "style", "nav", "footer"]):
            t.decompose()
        for t in soup.find_all(
            class_=re.compile(r"page-?number|hidden|metadata|footnote", re.I)
        ):
            t.decompose()

    @staticmethod
    def _clean_text(text: str) -> str:
        # Normalize text and handle encoding issues
        text = text.encode("utf-8", errors="replace").decode("utf-8")
        return re.sub(r"\s+", " ", text).strip()

    @staticmethod
    def _batch_text(paragraphs: list[str], max_chars: int = 1000) -> list[str]:
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

    @staticmethod
    def _sanitize_filename(name: str) -> str:
        return re.sub(r'[\\/*?:"<>|]', "", name).strip()


# --- UI MANAGER ---


class AudioBuilder:
    def __init__(self, config: Config):
        self.cfg = config
        self.temp_dir = Path("temp_audio_build")
        self.console = Console()

    def build(
        self, chapters: list[Chapter], output_file: Path, total_blocks: int
    ) -> bool:
        output_file.parent.mkdir(parents=True, exist_ok=True)

        with self._managed_temp_dir():
            self.console.print(
                f"[bold cyan]Building audiobook:[/bold cyan] {output_file.name}"
            )
            results = self._process_chapters(chapters, total_blocks)
            if not results:
                self.console.print("[red]No results generated. Aborting.[/red]")
                return False

            with self.console.status("[bold green]Stitching audio..."):
                self._stitch_files(results, output_file)

            self.console.print(
                f"[bold green]✓ Audiobook created:[/bold green] {output_file}"
            )
            return True

    def _process_chapters(
        self, chapters: list[Chapter], total_blocks: int
    ) -> list[AudioResult]:
        results = []
        worker_state = {}
        worker_errors = []
        worker_logs = []

        manager = multiprocessing.Manager()
        queue = manager.Queue()  # type: ignore

        # Create configuration dict for workers
        cfg_dict: dict = {}
        cfg_dict["voice"] = self.cfg.voice
        cfg_dict["pause_line_ms"] = self.cfg.pause_line_ms
        cfg_dict["pause_chapter_ms"] = self.cfg.pause_chapter_ms
        cfg_dict["tts_model"] = self.cfg.tts_model
        cfg_dict["tts_provider"] = self.cfg.tts_provider
        cfg_dict["model_name"] = self.cfg.model_name
        cfg_dict["elevenlabs_key"] = self.cfg.elevenlabs_key
        cfg_dict["elevenlabs_turbo"] = self.cfg.elevenlabs_turbo
        cfg_dict["debug_html"] = self.cfg.debug_html
        cfg_dict["verbose"] = self.cfg.verbose

        # Create a layout depending on verbose mode
        if self.cfg.verbose:
            layout = Layout()
            layout.split_column(
                Layout(name="header", size=5),
                Layout(name="logs", size=12),
                Layout(name="footer"),
            )
        else:
            layout = Layout()
            layout.split_column(Layout(name="upper", size=5), Layout(name="lower"))

        overall_progress = Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            TextColumn("•"),
            TimeRemainingColumn(),
            expand=True,
        )
        overall_task = overall_progress.add_task(
            "[bold cyan]Total Progress", total=total_blocks
        )

        try:
            with ProcessPoolExecutor(max_workers=self.cfg.workers) as pool:
                futures = {}
                for ch in chapters:
                    fut = pool.submit(
                        worker_process_chapter,
                        ch,
                        cfg_dict,
                        self.temp_dir,
                        queue,  # type: ignore
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
                                elif event == "ERROR":
                                    worker_errors.append(
                                        {
                                            "pid": pid,
                                            "chapter": msg[2],
                                            "message": msg[3],
                                            "traceback": msg[4],
                                        }
                                    )
                                elif event == "LOG":
                                    # Handle log messages from workers
                                    worker_logs.append(f"[{pid}] {msg[2]}")
                                    # Keep only last 20 log messages to avoid memory issues
                                    if len(worker_logs) > 20:
                                        worker_logs.pop(0)
                            except Exception:
                                break

                        all_done = all(f.done() for f in futures)
                        if overall_progress.tasks[0].finished:
                            break
                        if all_done and not worker_state and queue.empty():
                            break

                        # Update log panel in verbose mode
                        if self.cfg.verbose:
                            if worker_logs:
                                log_text = "\n".join(
                                    worker_logs[-10:]
                                )  # Show last 10 logs
                            else:
                                log_text = "[dim]Waiting for worker logs...[/dim]"
                            logs_panel = Panel(
                                log_text,
                                title="Worker Logs",
                                border_style="green",
                            )
                            layout["logs"].update(logs_panel)

                        # Update progress panel (different layout in verbose mode)
                        progress_panel = Panel(
                            overall_progress,
                            title="Overall Progress",
                            border_style="blue",
                        )
                        if self.cfg.verbose:
                            layout["header"].update(progress_panel)
                        else:
                            layout["upper"].update(progress_panel)

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

                        # Update worker panel (different layout in verbose mode)
                        worker_panel = Panel(
                            worker_table,
                            title=f"Worker Threads ({self.cfg.workers})",
                            border_style="grey50",
                        )
                        if self.cfg.verbose:
                            layout["footer"].update(worker_panel)
                        else:
                            layout["lower"].update(worker_panel)

                for future in as_completed(futures):
                    res = future.result()
                    if res:
                        results.append(res)

        except KeyboardInterrupt:
            self.console.print(
                "\n[yellow]Interrupted by user. Shutting down workers...[/yellow]"
            )
            # Properly shutdown the executor to prevent zombie processes
            if "pool" in locals():
                pool.shutdown(wait=False, cancel_futures=True)
            return []
        finally:
            if worker_errors:
                self.console.print("[red]Worker errors encountered:[/red]")
                for err in worker_errors:
                    self.console.print(
                        f"[red]- PID {err['pid']} {err['chapter']}: {err['message']}[/red]"
                    )
                    if self.cfg.debug_html:
                        self.console.print(f"[dim]{err['traceback']}[/dim]")

        return sorted(results, key=lambda x: x.chapter_index)

    def _stitch_files(self, results: list[AudioResult], output_file: Path):
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

    def run(self):
        """Main entry point for audiobook creation"""
        reader = EpubReader(self.cfg.epub_path, self.cfg.verbose)
        chapters = reader.extract_chapters()

        if not chapters:
            self.console.print("[red]No chapters found in EPUB[/red]")
            return False

        # Interactive chapter selection if requested
        if self.cfg.interactive_chapters:
            from .helpers import interactive_select

            chapters = interactive_select(
                chapters, "Available Chapters", self.console, lambda ch: ch.title
            )
            if not chapters:
                self.console.print("[yellow]No chapters selected[/yellow]")
                return False

        # Calculate total blocks for progress tracking
        total_blocks = sum(len(ch.paragraphs) for ch in chapters)

        # Determine output file path
        if self.cfg.output_path and self.cfg.output_path.suffix:
            output_file = self.cfg.output_path
        else:
            book_title = reader.get_book_title()
            output_dir = (
                self.cfg.output_path
                if self.cfg.output_path
                else self.cfg.epub_path.parent
            )
            output_file = output_dir / f"{book_title}.m4b"

        return self.build(chapters, output_file, total_blocks)

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
