"""Kenkui — Convert Ebooks to Audiobooks with custom voice samples.

This package provides tools for converting ebooks to audiobooks using
text-to-speech synthesis with support for custom voice samples.

Example:
    >>> from kenkui import EpubReader, AudioBuilder, ProcessingConfig
    >>> reader = EpubReader(Path("book.epub"))
    >>> config = ProcessingConfig(voice="alba", ebook_path=Path("book.epub"), ...)
    >>> builder = AudioBuilder(config)
    >>> builder.run()
"""

import importlib.metadata

from .chapter_classifier import ChapterClassifier, ChapterTags
from .chapter_filter import ChapterFilter, FilterOperation, FilterPreset
from .helpers import get_bundled_voices
from .huggingface_auth import (
    check_voice_access,
    ensure_huggingface_access,
    is_custom_voice,
)
from .models import (
    AudioResult,
    Chapter,
    CharacterInfo,
    NarrationMode,
    NLPResult,
    ProcessingConfig,
    Segment,
)
from .parsing import AudioBuilder
from .readers.epub import EpubReader
from .voice_loader import load_voice
from .workers import worker_process_chapter

try:
    __version__ = importlib.metadata.version("kenkui")
except importlib.metadata.PackageNotFoundError:
    __version__ = "1.0.0"

__author__ = "Sumner MacArthur"
__license__ = "GPL-3.0"

__all__ = [
    "ProcessingConfig",
    "Chapter",
    "Segment",
    "AudioResult",
    # Multi-voice / NLP
    "NLPResult",
    "CharacterInfo",
    "NarrationMode",
    # Ebook reading
    "EpubReader",
    "AudioBuilder",
    # Chapter handling
    "ChapterTags",
    "ChapterClassifier",
    "ChapterFilter",
    "FilterPreset",
    "FilterOperation",
    # Voices
    "get_bundled_voices",
    "load_voice",
    # Workers
    "worker_process_chapter",
    # HuggingFace auth
    "ensure_huggingface_access",
    "is_custom_voice",
    "check_voice_access",
    # Package metadata
    "__version__",
    "__author__",
    "__license__",
]
