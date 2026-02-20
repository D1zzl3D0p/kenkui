"""
EPUB ebook reader implementation.

Provides EbookReader interface for EPUB files using ebooklib.
"""

from __future__ import annotations

import re
import warnings
import zipfile
from pathlib import Path

from bs4 import BeautifulSoup
from ebooklib import epub

from ..chapter_classifier import ChapterClassifier
from ..helpers import Chapter
from ..utils import extract_epub_cover
from . import EbookMetadata, EbookReader, Registry, TocEntry


@Registry.register
class EpubReader(EbookReader):
    """EPUB ebook reader using ebooklib."""

    SUPPORTED_EXTENSIONS = {".epub"}

    def __init__(self, filepath: Path, verbose: bool = False):
        super().__init__(filepath, verbose)
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            self.book = epub.read_epub(str(filepath))

    def get_metadata(self) -> EbookMetadata:
        """Extract metadata from EPUB."""
        title = self._get_title()
        author = self._get_author()
        language = self._get_language()
        publisher = self._get_publisher()
        description = self._get_description()

        cover_data, cover_mime = extract_epub_cover(self.filepath)

        return EbookMetadata(
            title=title,
            author=author,
            language=language,
            publisher=publisher,
            description=description,
            cover_image=cover_data,
            cover_mime_type=cover_mime,
        )

    def _get_title(self) -> str:
        try:
            metadata = self.book.get_metadata("DC", "title")
            if metadata:
                return self._sanitize_filename(metadata[0][0])
        except Exception:
            pass
        return self.filepath.stem

    def _get_author(self) -> str | None:
        try:
            metadata = self.book.get_metadata("DC", "creator")
            if metadata:
                return metadata[0][0]
        except Exception:
            pass
        return None

    def _get_language(self) -> str | None:
        try:
            metadata = self.book.get_metadata("DC", "language")
            if metadata:
                return metadata[0][0]
        except Exception:
            pass
        return None

    def _get_publisher(self) -> str | None:
        try:
            metadata = self.book.get_metadata("DC", "publisher")
            if metadata:
                return metadata[0][0]
        except Exception:
            pass
        return None

    def _get_description(self) -> str | None:
        try:
            metadata = self.book.get_metadata("DC", "description")
            if metadata:
                return metadata[0][0]
        except Exception:
            pass
        return None

    def _sanitize_filename(self, name: str) -> str:
        return re.sub(r'[\\/*?:"<>|]', "", name).strip()

    def get_toc(self) -> list[TocEntry]:
        """Extract table of contents from EPUB."""
        toc_chapters = self._parse_toc_structure()
        return [
            TocEntry(
                title=ch["title"], href=ch.get("href", ""), level=ch.get("level", 0)
            )
            for ch in toc_chapters
        ]

    def _parse_toc_structure(self) -> list[dict]:
        """Parse the EPUB TOC (NCX or NAV) into a structured list."""
        from xml.etree import ElementTree as ET

        chapters: list[dict] = []

        toc_file, toc_type = self._find_toc_file()
        if toc_file is None:
            return chapters

        try:
            with zipfile.ZipFile(str(self.filepath), "r") as epub_zip:
                toc_content = epub_zip.read(toc_file).decode("utf-8")
                toc_tree = ET.fromstring(toc_content)

                if toc_type == "ncx":
                    ns = {"ncx": "http://www.daisy.org/z3986/2005/ncx/"}
                    for navpoint in toc_tree.findall(".//ncx:navPoint", ns):
                        navlabel = navpoint.find("ncx:navLabel/ncx:text", ns)
                        title = navlabel.text if navlabel is not None else "Untitled"

                        content = navpoint.find("ncx:content", ns)
                        if content is not None:
                            src = content.get("src", "")
                            href = src.split("#")[0]

                            # Determine level from navPoint depth
                            level = 0
                            depth_attr = navpoint.get("depth")
                            if depth_attr:
                                try:
                                    level = int(depth_attr)
                                except (ValueError, TypeError):
                                    pass

                            chapters.append(
                                {
                                    "title": title,
                                    "href": href,
                                    "src": src,
                                    "level": level,
                                }
                            )
                else:
                    # EPUB3 NAV format
                    ns = {"xhtml": "http://www.w3.org/1999/xhtml"}
                    toc_nav = toc_tree.find(".//xhtml:nav[@epub:type='toc']", ns)
                    if toc_nav is None:
                        toc_nav = toc_tree.find(".//nav[@epub:type='toc']")

                    if toc_nav is not None:
                        # Build hierarchical levels from nested lists
                        self._parse_nav_recursive(toc_nav, chapters, level=0)

        except Exception:
            pass

        return chapters

    def _parse_nav_recursive(self, element, chapters: list[dict], level: int):
        """Parse NAV element recursively to extract TOC with levels."""
        from xml.etree import ElementTree as ET

        ns = {"xhtml": "http://www.w3.org/1999/xhtml"}

        # Find direct child ol elements (nested lists)
        ol_elements = element.findall(".//xhtml:ol", ns)
        if not ol_elements:
            ol_elements = element.findall(".//ol")

        for ol in ol_elements:
            for li in (
                ol.findall(".//xhtml:li", ns)
                if ns.get("xhtml")
                else ol.findall(".//li")
            ):
                # Find anchor in this li
                link = li.find(".//xhtml:a", ns) if ns.get("xhtml") else li.find(".//a")
                if link is not None:
                    title = link.text or "Untitled"
                    src = link.get("href", "")
                    href = src.split("#")[0] if src else ""

                    if href:
                        chapters.append(
                            {
                                "title": title,
                                "href": href,
                                "src": src,
                                "level": level,
                            }
                        )

                # Recurse into nested ol
                nested_ol = (
                    li.find(".//xhtml:ol", ns) if ns.get("xhtml") else li.find(".//ol")
                )
                if nested_ol is not None:
                    self._parse_nav_recursive(nested_ol, chapters, level + 1)

    def _find_toc_file(self) -> tuple[str | None, str | None]:
        """Find the TOC file (NCX or NAV) in the EPUB."""
        from xml.etree import ElementTree as ET

        with zipfile.ZipFile(str(self.filepath), "r") as epub_zip:
            namelist = epub_zip.namelist()

            container_path = "META-INF/container.xml"
            if container_path in namelist:
                container_xml = epub_zip.read(container_path)
                container_tree = ET.fromstring(container_xml)

                ns = {"container": "urn:oasis:names:tc:opendocument:xmlns:container"}
                rootfile = container_tree.find(".//container:rootfile", ns)
                if rootfile is not None:
                    opf_path = rootfile.get("full-path")
                    if opf_path is None:
                        return (None, None)

                    opf_xml = epub_zip.read(opf_path)
                    opf_tree = ET.fromstring(opf_xml)

                    opf_ns = {"opf": "http://www.idpf.org/2007/opf"}

                    # Look for NCX
                    ncx_item = opf_tree.find(
                        ".//opf:item[@media-type='application/x-dtbncx+xml']", opf_ns
                    )
                    if ncx_item is not None:
                        ncx_href = ncx_item.get("href")
                        if ncx_href is not None:
                            opf_dir = str(Path(opf_path).parent)
                            ncx_path = (
                                ncx_href
                                if opf_dir == "."
                                else str(Path(opf_dir) / ncx_href)
                            )
                            return (ncx_path, "ncx")

                    # Look for NAV (EPUB3)
                    nav_item = opf_tree.find(".//opf:item[@properties='nav']", opf_ns)
                    if nav_item is not None:
                        nav_href = nav_item.get("href")
                        if nav_href is not None:
                            opf_dir = str(Path(opf_path).parent)
                            nav_path = (
                                nav_href
                                if opf_dir == "."
                                else str(Path(opf_dir) / nav_href)
                            )
                            return (nav_path, "nav")

            # Fallback: search for common TOC file names
            for name in namelist:
                if name.endswith(".ncx"):
                    return (name, "ncx")
                if "nav.xhtml" in name.lower() or "toc.xhtml" in name.lower():
                    return (name, "nav")

        return (None, None)

    def get_chapters(self, min_text_len: int = 50) -> list[Chapter]:
        """Extract chapters from EPUB."""
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            toc_chapters = self._parse_toc_structure()

        if toc_chapters:
            return self._extract_chapters_from_toc(toc_chapters, min_text_len)
        else:
            return self._extract_chapters_fallback(min_text_len)

    def _extract_chapters_from_toc(
        self, toc_chapters: list[dict], min_text_len: int
    ) -> list[Chapter]:
        """Extract chapters using TOC structure as ground truth."""
        chapters = []

        # Build file -> chapters map
        file_to_chapters: dict[str, list[tuple[str | None, int]]] = {}
        for idx, ch in enumerate(toc_chapters):
            src = ch.get("src", ch.get("href", ""))
            if "#" in src:
                file_name, anchor = src.split("#", 1)
            else:
                file_name, anchor = src, None

            if file_name not in file_to_chapters:
                file_to_chapters[file_name] = []
            file_to_chapters[file_name].append((anchor, idx))

        chapter_paragraphs: dict[int, list[str]] = {
            i: [] for i in range(len(toc_chapters))
        }

        # Get all items in reading order
        for item in self.book.get_items():
            if not hasattr(item, "get_content") or not hasattr(item, "get_name"):
                continue

            item_name = item.get_name()

            if item_name not in file_to_chapters:
                continue

            chapter_entries = file_to_chapters[item_name]

            try:
                content = item.get_content()
            except Exception:
                continue

            soup = BeautifulSoup(content, "html.parser")
            self._clean_soup(soup)

            if len(chapter_entries) == 1:
                _, chapter_idx = chapter_entries[0]
                # Use comprehensive paragraph extraction
                chapter_paragraphs[chapter_idx] = self._extract_chapter_paragraphs(soup)
            else:
                # Multiple chapters in file - split by anchor
                sorted_entries = self._get_sorted_entries(soup, chapter_entries)

                for i, (pos, anchor, chapter_idx) in enumerate(sorted_entries):
                    start_elem, end_elem = self._get_chapter_boundaries(
                        soup, anchor, sorted_entries, i
                    )

                    if start_elem and end_elem:
                        # Extract content between anchors
                        chapter_soup = BeautifulSoup("", "html.parser")
                        current = start_elem.find_next_sibling()
                        while current and current != end_elem:
                            chapter_soup.append(current)
                            current = current.find_next_sibling()
                        chapter_paragraphs[chapter_idx] = (
                            self._extract_chapter_paragraphs(chapter_soup)
                        )
                    elif start_elem:
                        # From start element to end of file
                        chapter_soup = BeautifulSoup("", "html.parser")
                        current = start_elem.find_next_sibling()
                        while current:
                            chapter_soup.append(current)
                            current = current.find_next_sibling()
                        chapter_paragraphs[chapter_idx] = (
                            self._extract_chapter_paragraphs(chapter_soup)
                        )
                    else:
                        # No anchor, try to get content from soup directly
                        chapter_paragraphs[chapter_idx] = (
                            self._extract_chapter_paragraphs(soup)
                        )

        # Create Chapter objects
        chapter_idx = 1
        for toc_idx, toc_ch in enumerate(toc_chapters):
            paragraphs = chapter_paragraphs[toc_idx]
            tags = ChapterClassifier.classify(toc_ch["title"])

            if paragraphs:
                content = " ".join(paragraphs)
                if len(content) >= min_text_len:
                    if paragraphs and toc_ch["title"].lower().endswith(
                        paragraphs[0].lower()
                    ):
                        paragraphs = paragraphs[1:]

                    chapters.append(
                        Chapter(
                            index=chapter_idx,
                            title=toc_ch["title"],
                            paragraphs=paragraphs,
                            tags=tags,
                            toc_index=toc_idx,
                        )
                    )
                    chapter_idx += 1
            else:
                chapters.append(
                    Chapter(
                        index=chapter_idx,
                        title=toc_ch["title"],
                        paragraphs=[],
                        tags=tags,
                        toc_index=toc_idx,
                    )
                )
                chapter_idx += 1

        return chapters

    def _get_sorted_entries(self, soup, chapter_entries):
        """Sort chapter entries by position in document."""
        sorted_entries = []
        for anchor, chapter_idx in chapter_entries:
            if anchor:
                elem = soup.find(id=anchor) or soup.find(attrs={"name": anchor})
                if elem:
                    position = len(list(elem.find_all_previous()))
                    sorted_entries.append((position, anchor, chapter_idx))
                else:
                    sorted_entries.append((float("inf"), anchor, chapter_idx))
            else:
                sorted_entries.append((0, anchor, chapter_idx))

        sorted_entries.sort(key=lambda x: x[0])
        return sorted_entries

    def _get_chapter_boundaries(self, soup, anchor, sorted_entries, i):
        """Get start and end elements for a chapter."""
        if anchor:
            start_elem = soup.find(id=anchor) or soup.find(attrs={"name": anchor})
        else:
            start_elem = None

        if i + 1 < len(sorted_entries):
            next_anchor = sorted_entries[i + 1][1]
            if next_anchor:
                end_elem = soup.find(id=next_anchor) or soup.find(
                    attrs={"name": next_anchor}
                )
            else:
                end_elem = None
        else:
            end_elem = None

        return start_elem, end_elem

    def _extract_chapter_paragraphs(self, soup: BeautifulSoup) -> list[str]:
        """Extract paragraphs from a chapter, handling various EPUB structures.

        This method handles:
        - Standard <p> elements
        - Script/dialogue format with <b> tags
        - Content in <div> elements
        - Other block-level elements

        Returns a list of paragraph strings.
        """
        paragraphs = []

        # Strategy 1: Try to find a main content section
        section = soup.find("section")
        if section:
            # Extract from section, looking at direct children first
            for child in section.children:
                if hasattr(child, "name") and child.name:
                    text = self._clean_text(child.get_text(" "))
                    if text and len(text) >= 2:
                        paragraphs.append(text)

            # If no direct children worked, get all text from section
            if not paragraphs:
                text = self._clean_text(section.get_text(separator="\n"))
                if text:
                    # Split by double newlines or multiple spaces
                    lines = [l.strip() for l in text.split("\n\n") if l.strip()]
                    if lines:
                        paragraphs.extend(lines)
                    else:
                        # Fallback: split by single newlines
                        lines = [l.strip() for l in text.split("\n") if l.strip()]
                        paragraphs.extend(lines)

        # Strategy 2: Standard <p> and <div> elements
        if not paragraphs:
            for elem in soup.find_all(["p", "div"]):
                if elem.find_parent(["p", "div"]):
                    continue
                text = self._clean_text(elem.get_text(" "))
                if text and len(text) >= 2:
                    paragraphs.append(text)

        # Strategy 3: Handle script/dialogue format (<b> tags for speakers)
        # This is crucial for books like "Anxious People" where dialogue is in <b> tags
        if not paragraphs or len(" ".join(paragraphs)) < 100:
            # Get all text content and try to segment it
            all_text = soup.get_text(separator="|")

            # Split by speaker names (typically in <b> tags, followed by colon)
            # Pattern: |SPEAKER: dialogue|
            import re

            # Split on patterns like "SPEAKER:" where SPEAKER is in caps
            parts = re.split(r"\|\s*([A-Z][A-Z\s]+):\s*", all_text)

            if len(parts) > 1:
                # We have dialogue format - reconstruct paragraphs
                paragraphs = []
                current_speaker = ""
                current_text = ""

                for i, part in enumerate(parts):
                    if i == 0:
                        # First part is usually intro text
                        if part.strip():
                            paragraphs.append(part.strip())
                    elif i % 2 == 1:
                        # This is a speaker name
                        current_speaker = part.strip()
                    else:
                        # This is the dialogue
                        if current_speaker:
                            # Combine speaker and dialogue
                            dialogue = f"{current_speaker}: {part.strip()}"
                            if len(dialogue) >= 2:
                                paragraphs.append(dialogue)
                        elif part.strip():
                            if len(part.strip()) >= 2:
                                paragraphs.append(part.strip())
                        current_speaker = ""

        # Strategy 4: Last resort - get all text and split intelligently
        if not paragraphs:
            text = soup.get_text(separator="\n")
            if text:
                lines = [
                    l.strip()
                    for l in text.split("\n")
                    if l.strip() and len(l.strip()) >= 2
                ]
                paragraphs.extend(lines)

        return paragraphs

    def _extract_chapters_fallback(self, min_text_len: int = 50) -> list[Chapter]:
        """Fallback regex-based chapter extraction when no TOC is available."""
        toc_map = self._build_toc_map()
        chapters: list[Chapter] = []

        current_vol = ""
        current_book = ""

        for item in self.book.get_items():
            if not hasattr(item, "get_content") or not hasattr(item, "get_name"):
                continue
            try:
                content = item.get_content()
            except Exception:
                continue

            soup = BeautifulSoup(content, "html.parser")
            self._clean_soup(soup)

            elements = soup.find_all(["h1", "h2", "h3", "h4", "p", "div", "section"])

            current_chapter_title = toc_map.get(item.get_name(), "")
            current_paragraphs: list[str] = []

            for elem in elements:
                if elem.find_parent(["p", "h1", "h2", "h3", "h4", "section"]):
                    continue

                text = self._clean_text(elem.get_text(" "))
                if not text or len(text) < 2:
                    continue

                # Check for Volume/Book markers
                if re.match(r"^(volume|part)\s+[ivxlcdm\d]+", text, re.I):
                    current_vol = text
                    continue
                if re.match(r"^book\s+(?:the\s+)?(?:[ivxlcdm\d]+|[a-z]+)", text, re.I):
                    current_book = text
                    continue

                # Check for Chapter markers
                chap_match = re.match(
                    r"^(chapter\s+[ivxlcdm\d]+|(?=[IVXLCDM]+\.)[IVXLCDM]+)([\.\-\—\s:]+)(.*)$",
                    text,
                    re.I,
                )

                if chap_match:
                    if current_paragraphs:
                        self._add_chapter(
                            chapters,
                            current_chapter_title,
                            current_paragraphs,
                            min_text_len,
                        )

                    header_label = chap_match.group(1)
                    remaining_text = chap_match.group(3).strip()

                    prefix = f"{current_vol}, " if current_vol else ""
                    prefix += f"{current_book}, " if current_book else ""

                    if len(remaining_text) > 200 and "." in remaining_text:
                        parts = re.split(r"(?<=\.)\s+", remaining_text, maxsplit=1)
                        current_chapter_title = f"{prefix}{header_label}: {parts[0]}"
                        current_paragraphs = [parts[1]] if len(parts) > 1 else []
                    else:
                        current_chapter_title = (
                            f"{prefix}{header_label}: {remaining_text}"
                        )
                        current_paragraphs = []
                else:
                    current_paragraphs.append(text)

            if current_paragraphs:
                self._add_chapter(
                    chapters, current_chapter_title, current_paragraphs, min_text_len
                )

        return chapters

    def _add_chapter(self, chapters, title, paragraphs, min_len, toc_index: int = 0):
        content = " ".join(paragraphs)
        if len(content) < min_len:
            return

        final_title = title if title else f"Chapter {len(chapters) + 1}"

        if paragraphs and final_title.lower().endswith(paragraphs[0].lower()):
            paragraphs = paragraphs[1:]

        tags = ChapterClassifier.classify(final_title)

        chapters.append(
            Chapter(
                index=len(chapters) + 1,
                title=final_title,
                paragraphs=paragraphs,
                tags=tags,
                toc_index=toc_index,
            )
        )

    def _build_toc_map(self) -> dict[str, str]:
        toc_map = {}

        def traverse(nodes):
            for node in nodes:
                if hasattr(node, "href") and hasattr(node, "title"):
                    toc_map[node.href.split("#")[0]] = node.title
                elif hasattr(node, "children"):
                    traverse(node.children)

        if hasattr(self.book, "toc"):
            traverse(self.book.toc)
        return toc_map

    def _clean_soup(self, soup: BeautifulSoup):
        for t in soup.find_all(["sup", "script", "style", "nav", "footer"]):
            t.decompose()
        for t in soup.find_all(
            class_=re.compile(r"page-?number|hidden|metadata|footnote", re.I)
        ):
            t.decompose()

    @staticmethod
    def _clean_text(text: str) -> str:
        text = text.encode("utf-8", errors="replace").decode("utf-8")
        return re.sub(r"\s+", " ", text).strip()

    def get_cover(self) -> tuple[bytes | None, str | None]:
        """Extract cover image from EPUB."""
        return extract_epub_cover(self.filepath)


__all__ = ["EpubReader"]
