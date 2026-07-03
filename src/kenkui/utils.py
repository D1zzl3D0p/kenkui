"""Shared utilities for kenkui - common functions used across modules."""

from __future__ import annotations

import os
import re
import xml.etree.ElementTree as ET
import zipfile
from enum import Enum
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

        def _append_chunk(chunk: str) -> None:
            if len(chunk) <= max_chars:
                chunks.append(chunk)
                return
            for start in range(0, len(chunk), max_chars):
                chunks.append(chunk[start : start + max_chars].strip())

        for sentence in sentences:
            slen = len(sentence)
            sep = 1 if current else 0
            if current_len + sep + slen > max_chars:
                if current:
                    _append_chunk(" ".join(current))
                current = [sentence]
                current_len = slen
            else:
                current.append(sentence)
                current_len += sep + slen
        if current:
            _append_chunk(" ".join(current))
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


class ApostropheMode(str, Enum):
    """Controls how apostrophes/contractions are processed before TTS.

    keep              – pass text through unchanged
    always_remove     – strip every apostrophe (straight + curly U+2018/U+2019)
    remove_contractions – strip apostrophe only from known contractions
    expand_contractions – expand all known contractions to full form (default)
    """

    KEEP = "keep"
    ALWAYS_REMOVE = "always_remove"
    REMOVE_CONTRACTIONS = "remove_contractions"
    EXPAND_CONTRACTIONS = "expand_contractions"


# ---------------------------------------------------------------------------
# TTS text normalization
# ---------------------------------------------------------------------------

_ALL_CONTRACTIONS_MAP: dict[str, str] = {
    # n't forms (superset of _NONT_MAP)
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
    # Subject + verb: I
    "i'm": "i am",
    "i've": "i have",
    "i'll": "i will",
    "i'd": "i would",
    # Subject + verb: you
    "you're": "you are",
    "you've": "you have",
    "you'll": "you will",
    "you'd": "you would",
    # Subject + verb: he
    "he's": "he is",
    "he'll": "he will",
    "he'd": "he would",
    # Subject + verb: she
    "she's": "she is",
    "she'll": "she will",
    "she'd": "she would",
    # Subject + verb: it
    "it's": "it is",
    "it'll": "it will",
    # Subject + verb: we
    "we're": "we are",
    "we've": "we have",
    "we'll": "we will",
    "we'd": "we would",
    # Subject + verb: they
    "they're": "they are",
    "they've": "they have",
    "they'll": "they will",
    "they'd": "they would",
    # Impersonal
    "that's": "that is",
    "that'll": "that will",
    "that'd": "that would",
    "there's": "there is",
    "there'll": "there will",
    "let's": "let us",
    "who's": "who is",
    "who'd": "who would",
    "who'll": "who will",
    "what's": "what is",
    "what'd": "what did",
    "what'll": "what will",
    "where's": "where is",
    "when's": "when is",
    "why's": "why is",
    "how's": "how is",
}

_ALL_CONTRACTIONS_PATTERN = re.compile(
    r"\b("
    + "|".join(re.escape(k) for k in sorted(_ALL_CONTRACTIONS_MAP, key=len, reverse=True))
    + r")\b",
    re.IGNORECASE,
)


def _expand_all_contraction(m: re.Match) -> str:
    """Expand a contraction from `_ALL_CONTRACTIONS_MAP` preserving case."""
    token = m.group(0)
    expansion = _ALL_CONTRACTIONS_MAP[token.lower()]
    if token.isupper():
        return expansion.upper()
    if token[0].isupper():
        return expansion[0].upper() + expansion[1:]
    return expansion


_TERMINAL_PUNCT = frozenset(".!?…\u2026")
_CLOSING_QUOTES = frozenset("\"'\u201c\u201d\u2018\u2019")
_CLOSING_QUOTES_STR = "\"'\u201c\u201d\u2018\u2019"


def ensure_terminal_punct(text: str) -> str:
    """Append a period if *text* doesn't end with terminal punctuation.

    Handles closing quotes transparently: ``"Hello"`` → unchanged if the char
    before the quote is already punctuation; otherwise a period is inserted
    before the closing quote cluster.
    """
    stripped = text.rstrip()
    if not stripped:
        return text
    last = stripped[-1]
    if last in _TERMINAL_PUNCT:
        return text
    if last in _CLOSING_QUOTES:
        inner = stripped.rstrip(_CLOSING_QUOTES_STR)
        if inner and inner[-1] in _TERMINAL_PUNCT:
            return text
        quotes = stripped[len(inner):]
        return inner + "." + quotes
    return stripped + "."


def normalize_for_tts(text: str, mode: ApostropheMode = ApostropheMode.EXPAND_CONTRACTIONS) -> str:
    """Normalize apostrophes/contractions for TTS according to *mode*.

    keep              → text unchanged
    always_remove     → strip all apostrophes (straight + curly)
    remove_contractions → strip apostrophe only from known contractions
    expand_contractions → expand all known contractions to full form (default)

    Case is preserved: ALL_CAPS → ALL_CAPS, Title → Title, else lower.
    Non-contraction possessives and proper names (e.g. ``O'Brien``) are
    untouched in ``remove_contractions`` and ``expand_contractions`` modes
    because they do not appear in ``_ALL_CONTRACTIONS_MAP``.

    Note: ``remove_contractions`` and ``expand_contractions`` normalize
    U+2019 (right single quote) to a straight apostrophe before matching,
    but leave U+2018 (left single quote) unchanged.
    """
    if mode == ApostropheMode.KEEP:
        return text
    if mode == ApostropheMode.ALWAYS_REMOVE:
        return re.sub(r"['\u2018\u2019]", "", text)
    # Contraction modes: normalise curly right-quote to straight apostrophe first.
    text = text.replace("\u2019", "'")
    if mode == ApostropheMode.REMOVE_CONTRACTIONS:
        return _ALL_CONTRACTIONS_PATTERN.sub(lambda m: m.group(0).replace("'", ""), text)
    # EXPAND_CONTRACTIONS (default)
    return _ALL_CONTRACTIONS_PATTERN.sub(_expand_all_contraction, text)


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
    "ApostropheMode",
    "batch_text",
    "clean_text",
    "ensure_terminal_punct",
    "extract_epub_cover",
    "normalize_for_tts",
    "sanitize_filename",
]
