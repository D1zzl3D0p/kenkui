"""
FB2 (FictionBook) ebook reader implementation.

Provides EbookReader interface for FB2 files using standard XML parsing.
"""

from __future__ import annotations

import re
import zipfile
from pathlib import Path
from xml.etree import ElementTree as ET

from ..chapter_classifier import ChapterClassifier
from ..helpers import Chapter
from . import EbookMetadata, EbookReader, Registry, TocEntry


FB2_NS = {"fb": "http://www.gribuser.ru/xml/fictionbook/2/0"}


@Registry.register
class Fb2Reader(EbookReader):
    """FB2 (FictionBook) ebook reader."""

    SUPPORTED_EXTENSIONS = {".fb2", ".fb2.zip"}

    def __init__(self, filepath: Path, verbose: bool = False):
        super().__init__(filepath, verbose)
        self._tree: ET.ElementTree | None = None
        self._root: ET.Element | None = None
        self._cover_data: tuple[bytes | None, str | None] = (None, None)

        self._parse()

    def _parse(self):
        """Parse the FB2 file (can be XML or ZIP)."""
        if self.filepath.suffix.lower() == ".zip" or self.filepath.name.endswith(
            ".fb2.zip"
        ):
            self._parse_zip()
        else:
            self._parse_xml(self.filepath)

    def _parse_zip(self):
        """Parse FB2 file inside a ZIP archive."""
        try:
            with zipfile.ZipFile(self.filepath, "r") as zf:
                # Find the fb2 file inside
                fb2_files = [f for f in zf.namelist() if f.lower().endswith(".fb2")]
                if fb2_files:
                    content = zf.read(fb2_files[0])
                    self._parse_xml_content(content)

                    # Try to find cover
                    self._extract_cover_from_zip(zf)
        except Exception as e:
            if self.verbose:
                print(f"Error parsing ZIP: {e}")

    def _parse_xml(self, filepath: Path):
        """Parse FB2 XML file directly."""
        try:
            with open(filepath, "rb") as f:
                content = f.read()
            self._parse_xml_content(content)

            # Try to find cover in same directory
            self._extract_cover_from_path(filepath)
        except Exception as e:
            if self.verbose:
                print(f"Error parsing XML: {e}")

    def _parse_xml_content(self, content: bytes):
        """Parse XML content into element tree."""
        try:
            # Remove XML declaration if present
            if content.startswith(b"<?xml"):
                content = content.split(b"?>", 1)[1]

            self._tree = ET.fromstring(content)
            self._root = self._tree

            # Handle namespace
            if "}" in self._tree.tag:
                ns = self._tree.tag.split("}")[0].strip("{")
                FB2_NS["fb"] = ns

        except ET.ParseError as e:
            if self.verbose:
                print(f"XML parse error: {e}")

    def _extract_cover_from_path(self, filepath: Path):
        """Try to find cover image in same directory."""
        base = filepath.with_suffix("")
        for ext in [".jpg", ".jpeg", ".png"]:
            cover_path = filepath.parent / f"{base.name}-cover{ext}"
            if not cover_path.exists():
                cover_path = filepath.parent / f"cover{ext}"

            if cover_path.exists():
                try:
                    with open(cover_path, "rb") as f:
                        data = f.read()
                    mime = "image/jpeg" if ext in (".jpg", ".jpeg") else "image/png"
                    self._cover_data = (data, mime)
                    return
                except Exception:
                    pass

    def _extract_cover_from_zip(self, zf: zipfile.ZipFile):
        """Extract cover from ZIP archive."""
        for name in zf.namelist():
            lower_name = name.lower()
            if "cover" in lower_name and lower_name.endswith((".jpg", ".jpeg", ".png")):
                try:
                    data = zf.read(name)
                    mime = (
                        "image/jpeg"
                        if lower_name.endswith((".jpg", ".jpeg"))
                        else "image/png"
                    )
                    self._cover_data = (data, mime)
                    return
                except Exception:
                    pass

    def get_metadata(self) -> EbookMetadata:
        """Extract metadata from FB2."""
        title = self.filepath.stem
        author = None
        language = None
        publisher = None
        description = None

        if self._root is None:
            return EbookMetadata(
                title=title,
                author=author,
                language=language,
                publisher=publisher,
                description=description,
                cover_image=self._cover_data[0],
                cover_mime_type=self._cover_data[1],
            )

        # Get title
        title_info = self._root.find(".//fb:title-info", FB2_NS)
        if title_info is None:
            title_info = self._root.find(".//title-info")

        if title_info is not None:
            # Title
            title_elem = title_info.find(".//fb:book-title", FB2_NS)
            if title_elem is None:
                title_elem = title_info.find(".//book-title")
            if title_elem is not None and title_elem.text:
                title = title_elem.text.strip()

            # Author
            authors = []
            for author_elem in title_info.findall(".//fb:author", FB2_NS):
                if author_elem is None:
                    author_elem = title_info.find(".//author")
                if author_elem is not None:
                    first = author_elem.find(".//fb:first-name", FB2_NS)
                    if first is None:
                        first = author_elem.find(".//first-name")
                    last = author_elem.find(".//fb:last-name", FB2_NS)
                    if last is None:
                        last = author_elem.find(".//last-name")

                    name_parts = []
                    if first is not None and first.text:
                        name_parts.append(first.text.strip())
                    if last is not None and last.text:
                        name_parts.append(last.text.strip())

                    if name_parts:
                        authors.append(" ".join(name_parts))

            if authors:
                author = ", ".join(authors)

            # Language
            lang_elem = title_info.find(".//fb:lang", FB2_NS)
            if lang_elem is None:
                lang_elem = title_info.find(".//lang")
            if lang_elem is not None and lang_elem.text:
                language = lang_elem.text.strip()

            # Publisher
            publish_info = title_info.find(".//fb:publish-info", FB2_NS)
            if publish_info is None:
                publish_info = title_info.find(".//publish-info")
            if publish_info is not None:
                publisher_elem = publish_info.find(".//fb:publisher", FB2_NS)
                if publisher_elem is None:
                    publisher_elem = publish_info.find(".//publisher")
                if publisher_elem is not None and publisher_elem.text:
                    publisher = publisher_elem.text.strip()

            # Description/Annotation
            annotation = title_info.find(".//fb:annotation", FB2_NS)
            if annotation is None:
                annotation = title_info.find(".//annotation")
            if annotation is not None:
                desc_parts = []
                for p in annotation.findall(".//fb:p", FB2_NS):
                    if p is None:
                        p = annotation.find(".//p")
                    if p is not None and p.text:
                        desc_parts.append(p.text.strip())
                if desc_parts:
                    description = "\n".join(desc_parts)

        return EbookMetadata(
            title=title,
            author=author,
            language=language,
            publisher=publisher,
            description=description,
            cover_image=self._cover_data[0],
            cover_mime_type=self._cover_data[1],
        )

    def get_toc(self) -> list[TocEntry]:
        """Extract table of contents from FB2."""
        toc = []

        if self._root is None:
            return toc

        # FB2 structure: body > section > section...
        # Title can be in <title><p> or <title><p>subtitle</p></title>

        # First try to get from description TOC
        toc_elem = self._root.find(".//fb:toc", FB2_NS)
        if toc_elem is None:
            toc_elem = self._root.find(".//toc")

        if toc_elem is not None:
            for link in toc_elem.findall(".//fb:link", FB2_NS):
                if link is None:
                    link = toc_elem.find(".//link")
                if link is not None:
                    title = link.find(".//fb:p", FB2_NS)
                    if title is None:
                        title = link.find(".//p")
                    title_text = (
                        title.text if title is not None and title.text else "Untitled"
                    )
                    href = link.get("href", "")

                    toc.append(
                        TocEntry(
                            title=title_text.strip(),
                            href=href,
                        )
                    )

            if toc:
                return toc

        # Fallback: extract from body sections
        bodies = self._root.findall(".//fb:body", FB2_NS)
        if not bodies:
            bodies = self._root.findall(".//body")

        level = 0
        for body in bodies:
            self._extract_sections_from_body(body, toc, level)

        return toc

    def _extract_sections_from_body(
        self, element: ET.Element, toc: list[TocEntry], level: int
    ):
        """Recursively extract section titles from body."""
        for section in element.findall(".//fb:section", FB2_NS):
            if section is None:
                section = element.findall(".//section")

            # Get title
            title = section.find(".//fb:title//fb:p", FB2_NS)
            if title is None:
                title = section.find(".//title//p")
            if title is None:
                title = section.find(".//fb:subtitle", FB2_NS)
            if title is None:
                title = section.find(".//subtitle")

            title_text = ""
            if title is not None and title.text:
                title_text = title.text.strip()
            elif title is not None:
                title_text = " ".join(t.strip() for t in title.itertext() if t.strip())

            if title_text:
                toc.append(
                    TocEntry(
                        title=title_text,
                        href="",
                        level=level,
                    )
                )

            # Recurse into nested sections
            self._extract_sections_from_body(section, toc, level + 1)

    def get_chapters(self, min_text_len: int = 50) -> list[Chapter]:
        """Extract chapters from FB2."""
        chapters = []

        if self._root is None:
            return chapters

        # Get all sections in the main body
        bodies = self._root.findall(".//fb:body", FB2_NS)
        if not bodies:
            bodies = self._root.findall(".//body")

        # Use first body as main content
        main_body = bodies[0] if bodies else None

        if main_body is None:
            return chapters

        chapter_idx = 1
        self._extract_chapters_recursive(main_body, chapters, "", 0, chapter_idx)

        # Renumber chapters
        for i, ch in enumerate(chapters):
            ch.index = i + 1

        return chapters

    def _extract_chapters_recursive(
        self,
        element: ET.Element,
        chapters: list[Chapter],
        parent_title: str,
        level: int,
        chapter_idx: int,
    ) -> int:
        """Recursively extract chapters from sections."""
        sections = element.findall(".//fb:section", FB2_NS)
        if not sections:
            sections = element.findall(".//section")

        if not sections:
            # This section has content, extract it
            content = self._extract_section_content(element)
            full_text = " ".join(content)

            if len(full_text) >= 50:
                # Get title from this element
                title = parent_title or f"Section {len(chapters) + 1}"

                # Skip if just whitespace
                if not title.strip():
                    title = f"Section {len(chapters) + 1}"

                tags = ChapterClassifier.classify(title)

                chapters.append(
                    Chapter(
                        index=chapter_idx,
                        title=title,
                        paragraphs=content,
                        tags=tags,
                        toc_index=len(chapters),
                    )
                )
                chapter_idx += 1
        else:
            for section in sections:
                # Get title
                title_elem = section.find(".//fb:title//fb:p", FB2_NS)
                if title_elem is None:
                    title_elem = section.find(".//title//p")
                if title_elem is None:
                    title_elem = section.find(".//fb:subtitle", FB2_NS)
                if title_elem is None:
                    title_elem = section.find(".//subtitle")

                section_title = ""
                if title_elem is not None and title_elem.text:
                    section_title = title_elem.text.strip()
                elif title_elem is not None:
                    section_title = " ".join(
                        t.strip() for t in title_elem.itertext() if t.strip()
                    )

                # Build full title path
                if parent_title and section_title:
                    full_title = f"{parent_title}: {section_title}"
                elif parent_title:
                    full_title = parent_title
                else:
                    full_title = section_title or f"Section {len(chapters) + 1}"

                chapter_idx = self._extract_chapters_recursive(
                    section, chapters, full_title, level + 1, chapter_idx
                )

        return chapter_idx

    def _extract_section_content(self, section: ET.Element) -> list[str]:
        """Extract text content from a section."""
        paragraphs = []

        # Get all p tags in this section
        for p in section.findall(".//fb:p", FB2_NS):
            if p is None:
                p = section.find(".//p")
            if p is not None:
                text = "".join(p.itertext()).strip()
                if text:
                    paragraphs.append(text)

        return paragraphs

    def get_cover(self) -> tuple[bytes | None, str | None]:
        """Extract cover image."""
        return self._cover_data


__all__ = ["Fb2Reader"]
