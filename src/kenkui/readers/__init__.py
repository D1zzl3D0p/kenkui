"""
Ebook reader abstraction layer.

This module provides a common interface for reading different ebook formats,
allowing the rest of the application to work with any format uniformly.

Architecture:
- EbookReader: Abstract base class defining the interface
- Registry: Format-to-reader mapping with auto-discovery
- Implementations: EpubReader, MobiReader, etc.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path

from ..helpers import Chapter


@dataclass
class EbookMetadata:
    """Common metadata structure across all ebook formats."""

    title: str
    author: str | None = None
    language: str | None = None
    publisher: str | None = None
    description: str | None = None
    cover_image: bytes | None = None
    cover_mime_type: str | None = None


@dataclass
class TocEntry:
    """Represents a single entry in a table of contents."""

    title: str
    href: str  # Internal reference (filename or anchor)
    level: int = 0  # Nesting level (0 = top-level)


class EbookReader(ABC):
    """Abstract base class for ebook readers.

    All ebook format implementations must inherit from this class
    and implement the required methods. This ensures a consistent
    interface across different formats.
    """

    # Subclasses must define supported extensions
    # e.g., SUPPORTED_EXTENSIONS = {".epub"}
    SUPPORTED_EXTENSIONS: set[str] = set()

    def __init__(self, filepath: Path, verbose: bool = False):
        """Initialize the reader with an ebook file path.

        Args:
            filepath: Path to the ebook file
            verbose: Enable verbose logging for debugging
        """
        self.filepath = filepath
        self.verbose = verbose
        self._metadata: EbookMetadata | None = None

    @abstractmethod
    def get_metadata(self) -> EbookMetadata:
        """Extract metadata from the ebook.

        Returns:
            EbookMetadata object with available metadata fields
        """
        pass

    @abstractmethod
    def get_toc(self) -> list[TocEntry]:
        """Extract the table of contents.

        Returns:
            List of TocEntry objects representing the TOC structure
        """
        pass

    @abstractmethod
    def get_chapters(self, min_text_len: int = 50) -> list[Chapter]:
        """Extract chapters from the ebook.

        This is the main method for content extraction. It should:
        1. Parse the TOC to identify chapter boundaries
        2. Extract text content for each chapter
        3. Return Chapter objects with paragraphs and metadata

        Args:
            min_text_len: Minimum text length for a chapter to be included

        Returns:
            List of Chapter objects in reading order
        """
        pass

    @abstractmethod
    def get_cover(self) -> tuple[bytes | None, str | None]:
        """Extract the cover image.

        Returns:
            Tuple of (image_data, mime_type) or (None, None) if no cover
        """
        pass

    def count_chapters(self) -> int:
        """Quickly count chapters without extracting full content.

        This is a lightweight operation that just counts TOC entries.
        Returns 0 if no TOC is available.

        Returns:
            Number of chapters in the ebook
        """
        toc = self.get_toc()
        return len(toc)

    @property
    def extension(self) -> str:
        """Return the file extension (e.g., '.epub')."""
        return self.filepath.suffix.lower()

    @property
    def format_name(self) -> str:
        """Return a human-readable format name (e.g., 'EPUB')."""
        return self.extension.lstrip(".").upper()


class Registry:
    """Registry for ebook format readers.

    Provides auto-discovery of readers and factory methods for creating
    the appropriate reader for a given file.
    """

    _readers: dict[str, type[EbookReader]] = {}

    @classmethod
    def register(cls, reader_class: type[EbookReader]) -> type[EbookReader]:
        """Register a reader class for its supported extensions.

        Args:
            reader_class: EbookReader subclass to register

        Returns:
            The same class (for use as decorator)
        """
        for ext in reader_class.SUPPORTED_EXTENSIONS:
            cls._readers[ext.lower()] = reader_class
        return reader_class

    @classmethod
    def get_reader_class(cls, filepath: Path) -> type[EbookReader] | None:
        """Get the appropriate reader class for a file.

        Args:
            filepath: Path to the ebook file

        Returns:
            Reader class or None if no reader found for extension
        """
        ext = filepath.suffix.lower()
        return cls._readers.get(ext)

    @classmethod
    def create_reader(cls, filepath: Path, verbose: bool = False) -> EbookReader:
        """Create a reader instance for the given file.

        Args:
            filepath: Path to the ebook file
            verbose: Enable verbose logging

        Returns:
            EbookReader instance for the file

        Raises:
            ValueError: If no reader is available for the file extension
        """
        reader_class = cls.get_reader_class(filepath)
        if reader_class is None:
            supported = ", ".join(sorted(cls._readers.keys()))
            raise ValueError(
                f"No reader available for '{filepath.suffix}' files. "
                f"Supported formats: {supported}"
            )
        return reader_class(filepath, verbose)

    @classmethod
    def supported_extensions(cls) -> set[str]:
        """Return all supported file extensions."""
        return set(cls._readers.keys())

    @classmethod
    def is_supported(cls, filepath: Path) -> bool:
        """Check if a file format is supported."""
        return filepath.suffix.lower() in cls._readers


def get_reader(filepath: Path, verbose: bool = False) -> EbookReader:
    """Convenience function to create a reader.

    This is the main entry point for creating readers in application code.

    Args:
        filepath: Path to the ebook file
        verbose: Enable verbose logging

    Returns:
        EbookReader instance for the file
    """
    return Registry.create_reader(filepath, verbose)


# Import implementations to register them with the Registry
from . import epub
from . import mobi
from . import fb2

__all__ = [
    "EbookReader",
    "EbookMetadata",
    "TocEntry",
    "Registry",
    "get_reader",
    # Re-export implementations
    "epub",
    "mobi",
    "fb2",
]
