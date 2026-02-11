"""Shared utilities for kenkui - common functions used across modules."""

from __future__ import annotations

import os
import re
import zipfile
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Tuple, Optional

# Constants
DEFAULT_VOICES = [
    "alba",
    "marius",
    "javert",
    "jean",
    "fantine",
    "cosette",
    "eponine",
    "azelma",
]

VOICE_DESCRIPTIONS = {
    "alba": "American Male",
    "marius": "American Male",
    "javert": "American Male",
    "jean": "American Male",
    "fantine": "British Female",
    "cosette": "American Female",
    "eponine": "British Female",
    "azelma": "American Female",
}


def batch_text(paragraphs: list[str], max_chars: int = 1000) -> list[str]:
    """Batch paragraphs into ~max_chars character chunks.

    Splits long paragraphs at sentence boundaries for natural breaks.
    """
    batched = []

    for p in paragraphs:
        if len(p) > max_chars:
            sentences = re.split(r"(?<=[.!?])\s+", p)
            current_chunk: list[str] = []
            current_len = 0

            for sentence in sentences:
                sent_len = len(sentence)
                if current_len + sent_len + (1 if current_chunk else 0) > max_chars:
                    if current_chunk:
                        batched.append(" ".join(current_chunk))
                    current_chunk = [sentence]
                    current_len = sent_len
                else:
                    current_chunk.append(sentence)
                    current_len += sent_len + (1 if len(current_chunk) > 1 else 0)

            if current_chunk:
                batched.append(" ".join(current_chunk))
        else:
            batched.append(p)

    return batched


def extract_epub_cover(epub_path: Path) -> Tuple[Optional[bytes], Optional[str]]:
    """Extract cover image from EPUB file.

    Returns:
        Tuple of (image_data, mime_type) or (None, None) if not found.
    """
    try:
        with zipfile.ZipFile(str(epub_path), "r") as epub:
            container = epub.read("META-INF/container.xml")
            tree = ET.fromstring(container)

            ns = {"container": "urn:oasis:names:tc:opendocument:xmlns:container"}
            rootfile = tree.find(".//container:rootfile", ns)
            if rootfile is None:
                return None, None

            opf_path = rootfile.get("full-path")
            if opf_path is None:
                return None, None

            opf_content = epub.read(opf_path)
            opf_tree = ET.fromstring(opf_content)

            namespaces = {
                "opf": "http://www.idpf.org/2007/opf",
                "dc": "http://purl.org/dc/elements/1.1/",
            }

            cover_id = None

            # Method 1: Look for meta tag with name="cover"
            for meta in opf_tree.findall('.//opf:meta[@name="cover"]', namespaces):
                cover_id = meta.get("content")
                break

            # Method 2: Look for item with properties="cover-image"
            if not cover_id:
                for item in opf_tree.findall(
                    './/opf:item[@properties="cover-image"]', namespaces
                ):
                    cover_id = item.get("id")
                    break

            if cover_id:
                for item in opf_tree.findall(".//opf:item", namespaces):
                    if item.get("id") == cover_id:
                        cover_href = item.get("href")
                        if cover_href is None:
                            continue
                        mime_type = item.get("media-type", "")
                        opf_dir = os.path.dirname(opf_path) or ""
                        cover_path = os.path.join(opf_dir, cover_href).replace(
                            "\\", "/"
                        )

                        cover_data = epub.read(cover_path)

                        if not mime_type:
                            ext = os.path.splitext(cover_path)[1].lower()
                            mime_type = {
                                ".jpg": "image/jpeg",
                                ".jpeg": "image/jpeg",
                                ".png": "image/png",
                            }.get(ext, "image/jpeg")

                        return cover_data, mime_type

            # Fallback: Look for common cover image names
            for name in epub.namelist():
                lower_name = name.lower()
                if "cover" in lower_name and any(
                    lower_name.endswith(ext) for ext in [".jpg", ".jpeg", ".png"]
                ):
                    cover_data = epub.read(name)
                    ext = os.path.splitext(name)[1].lower()
                    mime_type = {
                        ".jpg": "image/jpeg",
                        ".jpeg": "image/jpeg",
                        ".png": "image/png",
                    }.get(ext, "image/jpeg")
                    return cover_data, mime_type

    except Exception:
        pass

    return None, None


def sanitize_filename(name: str) -> str:
    """Remove special characters from filename."""
    return re.sub(r'[\\/*?:"<>|]', "", name).strip()


def clean_text(text: str) -> str:
    """Normalize text and handle encoding issues."""
    text = text.encode("utf-8", errors="replace").decode("utf-8")
    return re.sub(r"\s+", " ", text).strip()


__all__ = [
    "DEFAULT_VOICES",
    "VOICE_DESCRIPTIONS",
    "batch_text",
    "extract_epub_cover",
    "sanitize_filename",
    "clean_text",
]
