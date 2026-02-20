"""
MOBI/AZW ebook reader implementation.

Provides EbookReader interface for MOBI and AZW formats using the mobi library.
"""

from __future__ import annotations

import os
import re
import shutil
import tempfile
import warnings
from pathlib import Path

from bs4 import BeautifulSoup

from ..chapter_classifier import ChapterClassifier
from ..helpers import Chapter
from . import EbookMetadata, EbookReader, Registry, TocEntry


@Registry.register
class MobiReader(EbookReader):
    """MOBI/AZW ebook reader using mobi library."""

    SUPPORTED_EXTENSIONS = {".mobi", ".azw", ".azw3", ".azw4"}

    def __init__(self, filepath: Path, verbose: bool = False):
        super().__init__(filepath, verbose)
        self._temp_dir: Path | None = None
        self._extracted_html: Path | None = None
        self._extracted_dir: Path | None = None
        self._html_files: list[Path] = []
        self._cover_info: tuple[bytes | None, str | None] = (None, None)

        self._extract()

    def _extract(self):
        """Extract MOBI content to temporary directory."""
        import mobi

        try:
            # Extract to temp directory
            self._temp_dir = Path(tempfile.mkdtemp(prefix="kenkui_mobi_"))

            if self.verbose:
                print(f"Extracting MOBI to: {self._temp_dir}")

            # mobi.extract returns (tempdir, filepath) where filepath is the main HTML
            extracted_dir, main_file = mobi.extract(str(self.filepath), self._temp_dir)

            self._extracted_dir = Path(extracted_dir)

            if main_file and Path(main_file).exists():
                self._extracted_html = Path(main_file)
            else:
                # Try to find the HTML file in the extracted directory
                html_files = list(self._extracted_dir.glob("*.html")) + list(
                    self._extracted_dir.glob("*.xhtml")
                )
                if html_files:
                    # Sort by name to get reading order
                    html_files.sort(key=lambda x: x.name)
                    self._extracted_html = html_files[0]

            # Find all HTML files for chapter extraction
            self._html_files = sorted(
                self._extracted_dir.glob("**/*.html"), key=lambda x: x.name
            )

            # Extract cover image
            self._extract_cover()

        except Exception as e:
            if self.verbose:
                print(f"Error extracting MOBI: {e}")
            raise

    def _extract_cover(self):
        """Extract cover image from unpacked files."""
        if not self._extracted_dir:
            return

        # Look for common cover image names
        cover_names = [
            "cover.jpg",
            "cover.jpeg",
            "cover.png",
            "Cover.jpg",
            "Cover.jpeg",
            "Cover.png",
            "images/cover.jpg",
            "images/cover.jpeg",
            "images/cover.png",
        ]

        for cover_name in cover_names:
            cover_path = self._extracted_dir / cover_name
            if cover_path.exists():
                try:
                    with open(cover_path, "rb") as f:
                        data = f.read()
                    mime = (
                        "image/jpeg"
                        if cover_name.lower().endswith((".jpg", ".jpeg"))
                        else "image/png"
                    )
                    self._cover_info = (data, mime)
                    return
                except Exception:
                    continue

        # Try to find any image in the directory that might be a cover
        for img_path in self._extracted_dir.glob("**/*"):
            if img_path.suffix.lower() in (".jpg", ".jpeg", ".png"):
                # Heuristic: cover is usually one of the first images
                try:
                    with open(img_path, "rb") as f:
                        data = f.read()
                    # Check if it's reasonably sized (not a tiny icon)
                    if len(data) > 5000:
                        mime = (
                            "image/jpeg"
                            if img_path.suffix.lower() in (".jpg", ".jpeg")
                            else "image/png"
                        )
                        self._cover_info = (data, mime)
                        return
                except Exception:
                    continue

    def get_metadata(self) -> EbookMetadata:
        """Extract metadata from MOBI."""
        title = self.filepath.stem
        author = None

        # Try to get metadata from the main HTML file
        if self._extracted_html and self._extracted_html.exists():
            try:
                with open(
                    self._extracted_html, "r", encoding="utf-8", errors="ignore"
                ) as f:
                    html_content = f.read()

                soup = BeautifulSoup(html_content, "html.parser")

                # Look for title in metadata
                title_tag = soup.find("title")
                if title_tag and title_tag.text:
                    title = title_tag.text.strip()

                # Look for author in meta tags
                author_meta = soup.find("meta", {"name": "author"}) or soup.find(
                    "meta", {"name": "Creator"}
                )
                if author_meta and author_meta.get("content"):
                    author = author_meta["content"]

                # Try DC metadata
                dc_creator = soup.find("meta", {"name": "dc.creator"}) or soup.find(
                    "dc:creator"
                )
                if dc_creator:
                    author = dc_creator.get("content", author)

            except Exception:
                pass

        return EbookMetadata(
            title=title,
            author=author,
            cover_image=self._cover_info[0],
            cover_mime_type=self._cover_info[1],
        )

    def get_toc(self) -> list[TocEntry]:
        """Extract table of contents from MOBI.

        MOBI files often don't have a formal TOC, so we try multiple approaches.
        """
        toc_entries = []

        # Try to find a separate TOC file
        if self._extracted_dir:
            toc_files = (
                list(self._extracted_dir.glob("*toc*.html"))
                + list(self._extracted_dir.glob("*toc*.xhtml"))
                + list(self._extracted_dir.glob("*TOC*.html"))
            )

            for toc_file in toc_files:
                try:
                    with open(toc_file, "r", encoding="utf-8", errors="ignore") as f:
                        toc_content = f.read()

                    soup = BeautifulSoup(toc_content, "html.parser")

                    # Look for links (common in MOBI TOCs)
                    for link in soup.find_all("a"):
                        href = link.get("href", "")
                        text = link.get_text(strip=True)
                        if text and href:
                            toc_entries.append(TocEntry(title=text, href=href))

                    if toc_entries:
                        break
                except Exception:
                    continue

        # If no TOC file, build from HTML files
        if not toc_entries and self._html_files:
            for html_file in self._html_files:
                # Use filename as chapter title
                title = html_file.stem
                # Skip common non-chapter files
                lower_name = title.lower()
                if any(
                    skip in lower_name
                    for skip in [
                        "cover",
                        "copyright",
                        "toc",
                        "nav",
                        "title",
                        "dedication",
                        "acknowledg",
                    ]
                ):
                    continue

                # Clean up title
                title = re.sub(r"^[0-9]+\.?", "", title)  # Remove leading numbers
                title = title.replace("_", " ").replace("-", " ").strip()

                if title:
                    toc_entries.append(
                        TocEntry(
                            title=title.title(),
                            href=str(html_file.relative_to(self._extracted_dir))
                            if self._extracted_dir
                            else str(html_file),
                        )
                    )

        return toc_entries

    def get_chapters(self, min_text_len: int = 50) -> list[Chapter]:
        """Extract chapters from MOBI."""
        if not self._html_files:
            return []

        # First, try to use TOC for chapter extraction
        toc = self.get_toc()

        if toc:
            return self._extract_chapters_from_toc(toc, min_text_len)
        else:
            return self._extract_chapters_from_files(min_text_len)

    def _extract_chapters_from_toc(
        self, toc: list[TocEntry], min_text_len: int
    ) -> list[Chapter]:
        """Extract chapters using TOC structure."""
        chapters = []
        chapter_idx = 1

        # Build a map of hrefs to content
        html_content_map: dict[str, str] = {}
        for html_file in self._html_files:
            try:
                with open(html_file, "r", encoding="utf-8", errors="ignore") as f:
                    content = f.read()
                rel_path = (
                    str(html_file.relative_to(self._extracted_dir))
                    if self._extracted_dir
                    else html_file.name
                )
                html_content_map[rel_path] = content
                # Also try without path
                html_content_map[html_file.name] = content
            except Exception:
                continue

        for toc_entry in toc:
            href = toc_entry.href
            title = toc_entry.title

            # Find matching content
            content = None
            for key, cnt in html_content_map.items():
                if href in key or key in href:
                    content = cnt
                    break

            if content is None:
                # Try to find by title matching filename
                title_lower = title.lower().replace(" ", "")
                for key, cnt in html_content_map.items():
                    if title_lower in key.lower().replace(" ", "").replace("_", ""):
                        content = cnt
                        break

            if content:
                paragraphs = self._extract_paragraphs(content)
                full_text = " ".join(paragraphs)

                if len(full_text) >= min_text_len:
                    tags = ChapterClassifier.classify(title)

                    # Clean up if title repeats first paragraph
                    if paragraphs and title.lower().endswith(
                        paragraphs[0].lower()[:50]
                    ):
                        paragraphs = paragraphs[1:]

                    chapters.append(
                        Chapter(
                            index=chapter_idx,
                            title=title,
                            paragraphs=paragraphs,
                            tags=tags,
                        )
                    )
                    chapter_idx += 1
            else:
                # Include entry even without content (e.g., part dividers)
                tags = ChapterClassifier.classify(title)
                chapters.append(
                    Chapter(
                        index=chapter_idx,
                        title=title,
                        paragraphs=[],
                        tags=tags,
                    )
                )
                chapter_idx += 1

        return chapters

    def _extract_chapters_from_files(self, min_text_len: int) -> list[Chapter]:
        """Fallback: extract chapters from HTML files directly."""
        chapters = []
        chapter_idx = 1

        current_vol = ""
        current_book = ""

        for html_file in self._html_files:
            try:
                with open(html_file, "r", encoding="utf-8", errors="ignore") as f:
                    content = f.read()
            except Exception:
                continue

            # Skip non-chapter files
            lower_name = html_file.name.lower()
            if any(
                skip in lower_name
                for skip in [
                    "cover",
                    "copyright",
                    "toc",
                    "nav",
                    "titlepage",
                    "dedication",
                ]
            ):
                continue

            soup = BeautifulSoup(content, "html.parser")
            self._clean_soup(soup)

            elements = soup.find_all(["h1", "h2", "h3", "h4", "p", "div"])

            current_chapter_title = html_file.stem
            current_paragraphs: list[str] = []

            for elem in elements:
                if elem.find_parent(["p", "h1", "h2", "h3", "h4"]):
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
                            chapter_idx - 1,
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
                    chapters,
                    current_chapter_title,
                    current_paragraphs,
                    min_text_len,
                    chapter_idx - 1,
                )

        return chapters

    def _extract_paragraphs(self, html_content: str) -> list[str]:
        """Extract paragraphs from HTML content."""
        soup = BeautifulSoup(html_content, "html.parser")
        self._clean_soup(soup)

        paragraphs = []
        for elem in soup.find_all(["p", "div"]):
            if elem.find_parent(["p", "div"]):
                continue
            text = self._clean_text(elem.get_text(" "))
            if text and len(text) >= 2:
                paragraphs.append(text)

        return paragraphs

    def _add_chapter(
        self,
        chapters: list[Chapter],
        title: str,
        paragraphs: list[str],
        min_len: int,
        toc_index: int,
    ):
        content = " ".join(paragraphs)
        if len(content) < min_len:
            return

        final_title = title if title else f"Chapter {len(chapters) + 1}"

        if paragraphs and final_title.lower().endswith(paragraphs[0].lower()[:50]):
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

    def _clean_soup(self, soup: BeautifulSoup):
        """Clean HTML of unwanted elements."""
        for t in soup.find_all(["script", "style", "nav"]):
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
        """Extract cover image."""
        return self._cover_info

    def cleanup(self):
        """Clean up temporary extraction directory."""
        if self._temp_dir and Path(self._temp_dir).exists():
            try:
                shutil.rmtree(self._temp_dir)
            except Exception:
                pass

    def __del__(self):
        """Cleanup on deletion."""
        self.cleanup()


__all__ = ["MobiReader"]
