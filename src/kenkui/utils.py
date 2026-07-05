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


class NumberNormalizationMode(str, Enum):
    """Controls how a number category is rendered before TTS."""

    RAW = "raw"
    WORDS = "words"
    DIGITS = "digits"
    GROUPED_DIGITS = "grouped_digits"


# ---------------------------------------------------------------------------
# TTS text normalization
# ---------------------------------------------------------------------------

_DIGIT_WORDS = {
    "0": "zero",
    "1": "one",
    "2": "two",
    "3": "three",
    "4": "four",
    "5": "five",
    "6": "six",
    "7": "seven",
    "8": "eight",
    "9": "nine",
}

_DEFAULT_NUMBER_NORMALIZATION = {
    "phone_numbers_mode": NumberNormalizationMode.GROUPED_DIGITS.value,
    "identifiers_mode": NumberNormalizationMode.DIGITS.value,
    "cardinals_mode": NumberNormalizationMode.WORDS.value,
    "decimals_mode": NumberNormalizationMode.WORDS.value,
    "ordinals_mode": NumberNormalizationMode.WORDS.value,
    "percentages_mode": NumberNormalizationMode.WORDS.value,
    "identifier_min_digits": 7,
}

_PHONE_RE = re.compile(r"(?<!\d)(\d{3})[-. ](\d{3})[-. ](\d{4})(?!\d)")
_PERCENT_RE = re.compile(r"(?<![\w.])([+-]?\d+(?:,\d{3})*(?:\.\d+)?)\s*%(?!\w)")
_DECIMAL_RE = re.compile(r"(?<![\w.])([+-]?\d+(?:,\d{3})*)\.(\d+)(?![\w.])")
_ORDINAL_RE = re.compile(r"(?<![\w.])([+-]?\d+(?:,\d{3})*)(st|nd|rd|th)(?!\w)", re.I)
_CARDINAL_RE = re.compile(r"(?<![\w.])([+-]?(?:\d{1,3}(?:,\d{3})+|\d+))(?!\w)")

_ONES = (
    "zero",
    "one",
    "two",
    "three",
    "four",
    "five",
    "six",
    "seven",
    "eight",
    "nine",
    "ten",
    "eleven",
    "twelve",
    "thirteen",
    "fourteen",
    "fifteen",
    "sixteen",
    "seventeen",
    "eighteen",
    "nineteen",
)
_TENS = (
    "",
    "",
    "twenty",
    "thirty",
    "forty",
    "fifty",
    "sixty",
    "seventy",
    "eighty",
    "ninety",
)
_SCALES = ("", "thousand", "million", "billion", "trillion")
_IRREGULAR_ORDINALS = {
    "one": "first",
    "two": "second",
    "three": "third",
    "five": "fifth",
    "eight": "eighth",
    "nine": "ninth",
    "twelve": "twelfth",
}

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


def _number_normalization_value(config: object | None, key: str):
    if config is None:
        return _DEFAULT_NUMBER_NORMALIZATION[key]
    if isinstance(config, dict):
        return config.get(key, _DEFAULT_NUMBER_NORMALIZATION[key])
    return getattr(config, key, _DEFAULT_NUMBER_NORMALIZATION[key])


def _number_mode(config: object | None, key: str) -> NumberNormalizationMode:
    value = _number_normalization_value(config, key)
    if isinstance(value, NumberNormalizationMode):
        return value
    try:
        return NumberNormalizationMode(str(value))
    except ValueError:
        return NumberNormalizationMode(_DEFAULT_NUMBER_NORMALIZATION[key])


def _identifier_min_digits(config: object | None) -> int:
    try:
        value = int(_number_normalization_value(config, "identifier_min_digits"))
    except (TypeError, ValueError):
        value = int(_DEFAULT_NUMBER_NORMALIZATION["identifier_min_digits"])
    return max(1, value)


def _clean_number_token(token: str) -> str:
    return token.replace(",", "")


def _digits_to_words(text: str) -> str:
    return " ".join(_DIGIT_WORDS[ch] for ch in text if ch.isdigit())


def _fallback_cardinal(value: int) -> str:
    if value < 0:
        return "minus " + _fallback_cardinal(abs(value))
    if value < 20:
        return _ONES[value]
    if value < 100:
        tens, ones = divmod(value, 10)
        return _TENS[tens] if ones == 0 else f"{_TENS[tens]} {_ONES[ones]}"
    if value < 1000:
        hundreds, rest = divmod(value, 100)
        head = f"{_ONES[hundreds]} hundred"
        return head if rest == 0 else f"{head} {_fallback_cardinal(rest)}"

    parts: list[str] = []
    scale_index = 0
    remaining = value
    while remaining and scale_index < len(_SCALES):
        remaining, chunk = divmod(remaining, 1000)
        if chunk:
            scale = _SCALES[scale_index]
            words = _fallback_cardinal(chunk)
            parts.append(f"{words} {scale}".strip())
        scale_index += 1
    if remaining:
        return str(value)
    return " ".join(reversed(parts))


def _engine_cardinal(token: str) -> str:
    value = int(_clean_number_token(token))
    try:
        import inflect

        words = inflect.engine().number_to_words(value, andword="")
    except Exception:
        words = _fallback_cardinal(value)
    return str(words).replace("-", " ").replace(",", "").strip()


def _fallback_ordinal(token: str) -> str:
    words = _engine_cardinal(token)
    head, sep, tail = words.rpartition(" ")
    target = tail if sep else words
    if target in _IRREGULAR_ORDINALS:
        ordinal = _IRREGULAR_ORDINALS[target]
    elif target.endswith("y"):
        ordinal = target[:-1] + "ieth"
    else:
        ordinal = target + "th"
    return f"{head} {ordinal}".strip() if sep else ordinal


def _engine_ordinal(token: str) -> str:
    value = int(_clean_number_token(token))
    try:
        import inflect

        words = inflect.engine().number_to_words(inflect.engine().ordinal(value), andword="")
    except Exception:
        words = _fallback_ordinal(token)
    return str(words).replace("-", " ").replace(",", "").strip()


def _decimal_to_words(integer: str, fraction: str) -> str:
    return f"{_engine_cardinal(integer)} point {_digits_to_words(fraction)}"


def normalize_numbers_for_tts(text: str, config: object | None = None) -> str:
    """Normalize number-like spans for TTS according to *config*.

    The function protects each matched span with a placeholder as it goes, so a
    category set to ``raw`` is still shielded from later, lower-priority rules.
    """
    if not text:
        return text

    replacements: list[str] = []

    def protect(value: str) -> str:
        marker = f"\x00KENKUI_NUM_{len(replacements)}\x00"
        replacements.append(value)
        return marker

    def replace_phone(match: re.Match) -> str:
        original = match.group(0)
        mode = _number_mode(config, "phone_numbers_mode")
        if mode == NumberNormalizationMode.RAW:
            return protect(original)
        groups = match.groups()
        if mode == NumberNormalizationMode.WORDS:
            return protect(_engine_cardinal("".join(groups)))
        return protect(", ".join(_digits_to_words(group) for group in groups))

    def replace_percent(match: re.Match) -> str:
        original = match.group(0)
        mode = _number_mode(config, "percentages_mode")
        if mode == NumberNormalizationMode.RAW:
            return protect(original)
        number = match.group(1)
        if mode in {NumberNormalizationMode.DIGITS, NumberNormalizationMode.GROUPED_DIGITS}:
            spoken = _digits_to_words(number)
        elif "." in number:
            integer, fraction = _clean_number_token(number).split(".", 1)
            spoken = _decimal_to_words(integer, fraction)
        else:
            spoken = _engine_cardinal(number)
        return protect(f"{spoken} percent")

    def replace_decimal(match: re.Match) -> str:
        original = match.group(0)
        mode = _number_mode(config, "decimals_mode")
        if mode == NumberNormalizationMode.RAW:
            return protect(original)
        integer, fraction = match.groups()
        if mode in {NumberNormalizationMode.DIGITS, NumberNormalizationMode.GROUPED_DIGITS}:
            return protect(_digits_to_words(_clean_number_token(integer) + fraction))
        return protect(_decimal_to_words(integer, fraction))

    def replace_ordinal(match: re.Match) -> str:
        original = match.group(0)
        mode = _number_mode(config, "ordinals_mode")
        if mode == NumberNormalizationMode.RAW:
            return protect(original)
        number = match.group(1)
        if mode in {NumberNormalizationMode.DIGITS, NumberNormalizationMode.GROUPED_DIGITS}:
            return protect(_digits_to_words(number))
        return protect(_engine_ordinal(number))

    def replace_cardinal(match: re.Match) -> str:
        original = match.group(0)
        digits = _clean_number_token(match.group(1)).lstrip("+-")
        if len(digits) >= _identifier_min_digits(config):
            mode = _number_mode(config, "identifiers_mode")
            if mode == NumberNormalizationMode.RAW:
                return protect(original)
            if mode == NumberNormalizationMode.WORDS:
                return protect(_engine_cardinal(match.group(1)))
            return protect(_digits_to_words(digits))

        mode = _number_mode(config, "cardinals_mode")
        if mode == NumberNormalizationMode.RAW:
            return protect(original)
        if mode in {NumberNormalizationMode.DIGITS, NumberNormalizationMode.GROUPED_DIGITS}:
            return protect(_digits_to_words(digits))
        return protect(_engine_cardinal(match.group(1)))

    normalized = _PHONE_RE.sub(replace_phone, text)
    normalized = _PERCENT_RE.sub(replace_percent, normalized)
    normalized = _DECIMAL_RE.sub(replace_decimal, normalized)
    normalized = _ORDINAL_RE.sub(replace_ordinal, normalized)
    normalized = _CARDINAL_RE.sub(replace_cardinal, normalized)
    for index, value in enumerate(replacements):
        normalized = normalized.replace(f"\x00KENKUI_NUM_{index}\x00", value)
    return normalized


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
    "NumberNormalizationMode",
    "batch_text",
    "clean_text",
    "ensure_terminal_punct",
    "extract_epub_cover",
    "normalize_numbers_for_tts",
    "normalize_for_tts",
    "sanitize_filename",
]
