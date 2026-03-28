"""Shared utilities for kenkui - common functions used across modules."""

from __future__ import annotations

import os
import re
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path

# Re-export from voice_registry — the registry is the single source of truth.
from .voice_registry import BUILTIN_VOICE_NAMES as BUILTIN_VOICE_NAMES  # noqa: F401


def batch_text(
    paragraphs: list[str],
    max_chars: int = 800,
    merge_short: bool = True,
) -> list[str]:
    """Batch paragraphs into ~max_chars character chunks for TTS.

    Algorithm:
    - Paragraphs shorter than ``max_chars`` are *merged* into a running buffer
      (when ``merge_short=True``) so that many short dialogue lines become a
      single, efficient TTS call instead of dozens of tiny ones.
    - Paragraphs longer than ``max_chars`` are split at sentence boundaries
      and each sentence-chunk is appended directly (never merged with a
      subsequent short paragraph — that would create unnatural boundaries).
    - When ``merge_short=False`` every paragraph is emitted individually
      (only splitting if it exceeds ``max_chars``).  Use this mode for
      multi-voice segments where speaker boundaries must not be crossed.

    Args:
        paragraphs:  List of text segments (paragraphs, dialogue lines, etc.)
        max_chars:   Target maximum characters per TTS call.
        merge_short: If True, accumulate short paragraphs into batches up to
                     ``max_chars``.  If False, each paragraph is its own item.

    Returns:
        List of text chunks ready for individual TTS calls.
    """
    if not paragraphs:
        return []

    result: list[str] = []
    buffer: list[str] = []
    buffer_len: int = 0

    def _flush_buffer():
        if buffer:
            result.append(" ".join(buffer))
            buffer.clear()

    def _split_long(text: str) -> list[str]:
        """Split a single long paragraph at sentence boundaries."""
        sentences = re.split(r"(?<=[.!?])\s+", text)
        chunks: list[str] = []
        current: list[str] = []
        current_len = 0
        for sentence in sentences:
            slen = len(sentence)
            sep = 1 if current else 0
            if current_len + sep + slen > max_chars:
                if current:
                    chunks.append(" ".join(current))
                current = [sentence]
                current_len = slen
            else:
                current.append(sentence)
                current_len += sep + slen
        if current:
            chunks.append(" ".join(current))
        return chunks

    for para in paragraphs:
        if not para.strip():
            continue

        if len(para) > max_chars:
            # Long paragraph: flush any pending buffer first, then split
            _flush_buffer()
            buffer_len = 0
            for chunk in _split_long(para):
                result.append(chunk)
        elif merge_short:
            sep = 1 if buffer else 0
            if buffer_len + sep + len(para) > max_chars:
                _flush_buffer()
                buffer_len = 0
            buffer.append(para)
            buffer_len += (1 if len(buffer) > 1 else 0) + len(para)
        else:
            # merge_short=False: each paragraph is its own item
            result.append(para)

    _flush_buffer()
    return result


# ---------------------------------------------------------------------------
# TTS text normalization
# ---------------------------------------------------------------------------

# n't contractions that are commonly mispronounced by the TTS engine.
# Handles both straight apostrophe (') and right single quotation mark (').
# Only n't forms are expanded — other contractions ("I'm", "we're") are left
# alone to preserve natural speech cadence.
_NONT_MAP: dict[str, str] = {
    "won't": "will not",
    "can't": "cannot",
    "don't": "do not",
    "doesn't": "does not",
    "didn't": "did not",
    "isn't": "is not",
    "aren't": "are not",
    "wasn't": "was not",
    "weren't": "were not",
    "haven't": "have not",
    "hasn't": "has not",
    "hadn't": "had not",
    "couldn't": "could not",
    "wouldn't": "would not",
    "shouldn't": "should not",
    "mustn't": "must not",
    "needn't": "need not",
    "shan't": "shall not",
}

# Build a single compiled regex that matches any contraction (case-insensitive).
# The right single quotation mark (U+2019) is treated as an apostrophe.
_NONT_PATTERN = re.compile(
    r"\b(" + "|".join(re.escape(k) for k in _NONT_MAP) + r")\b",
    re.IGNORECASE,
)


def _replace_contraction(m: re.Match) -> str:
    """Return the expansion with the same capitalisation as the matched token."""
    token = m.group(0)
    expansion = _NONT_MAP[token.lower().replace("\u2019", "'")]
    if token.isupper():
        return expansion.upper()
    if token[0].isupper():
        return expansion[0].upper() + expansion[1:]
    return expansion


def normalize_for_tts(text: str) -> str:
    """Expand n't contractions so the TTS engine pronounces them correctly.

    Examples::

        >>> normalize_for_tts("He doesn't know and wasn't sure.")
        'He does not know and was not sure.'
        >>> normalize_for_tts("DON'T")
        'DO NOT'
        >>> normalize_for_tts("Don't")
        'Do not'
        >>> normalize_for_tts("I'm ready")   # non-n't contraction → unchanged
        "I'm ready"
    """
    # Normalise curly apostrophes to straight so the pattern matches both forms.
    text = text.replace("\u2019", "'")
    return _NONT_PATTERN.sub(_replace_contraction, text)


def extract_epub_cover(epub_path: Path) -> tuple[bytes | None, str | None]:
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
                for item in opf_tree.findall('.//opf:item[@properties="cover-image"]', namespaces):
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
                        cover_path = os.path.join(opf_dir, cover_href).replace("\\", "/")

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
