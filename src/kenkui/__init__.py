"""Kenkui - Convert Ebooks to Audiobooks with custom voice samples.

This package provides tools for converting EPUB files to audiobooks using
text-to-speech synthesis with support for custom voice samples.

Example:
    >>> from kenkui import EpubReader, AudioBuilder, Config
    >>> reader = EpubReader(Path("book.epub"))
    >>> chapters = reader.extract_chapters()
    >>> config = Config(voice="alba", epub_path=Path("book.epub"))
    >>> builder = AudioBuilder(config)
    >>> builder.run()
"""

import importlib.metadata

from .helpers import (
    Config,
    Chapter,
    AudioResult,
    parse_range_string,
    interactive_select,
    check_huggingface_access,
    get_bundled_voices,
    print_available_voices,
)
from .parsing import EpubReader, AudioBuilder
from .workers import worker_process_chapter
from .chapter_classifier import ChapterTags, ChapterClassifier
from .chapter_filter import ChapterFilter, FilterPreset, FilterOperation

try:
    __version__ = importlib.metadata.version("kenkui")
except importlib.metadata.PackageNotFoundError:
    __version__ = "0.5.0"

__author__ = "Sumner MacArthur"
__license__ = "GPL-3.0"

__all__ = [
    # Core classes
    "Config",
    "Chapter",
    "AudioResult",
    "EpubReader",
    "AudioBuilder",
    # Chapter classification and filtering
    "ChapterTags",
    "ChapterClassifier",
    "ChapterFilter",
    "FilterPreset",
    "FilterOperation",
    # Functions
    "parse_range_string",
    "interactive_select",
    "check_huggingface_access",
    "get_bundled_voices",
    "print_available_voices",
    "worker_process_chapter",
    # Metadata
    "__version__",
    "__author__",
    "__license__",
]
