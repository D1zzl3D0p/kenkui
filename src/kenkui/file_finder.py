"""Fast file discovery utilities for finding ebook files recursively.

Uses os.scandir() for optimal performance with cross-platform hidden folder detection.
Supports multiple ebook formats through a pluggable architecture.
"""

from __future__ import annotations

import os
import sys
from collections.abc import Iterator
from pathlib import Path

# Import the Registry to get supported extensions dynamically
from .readers import Registry


def _is_hidden_unix(path: str, name: str) -> bool:
    """Check if a file/directory is hidden on Unix-like systems (Linux, macOS).

    Unix convention: Hidden files/directories start with a dot.
    Also checks for macOS metadata files (._*).

    Easy to modify: Add more patterns to the checks below.
    """
    return name.startswith(".") or name.startswith("._")


def _is_hidden_windows(path: str, name: str) -> bool:
    """Check if a file/directory is hidden on Windows.

    Windows convention: Uses FILE_ATTRIBUTE_HIDDEN flag.
    Falls back to name-based check if attribute check fails.

    Easy to modify: Add more patterns to the checks below.
    """
    try:
        import ctypes

        FILE_ATTRIBUTE_HIDDEN = 0x02
        full_path = os.path.join(path, name)

        attrs = ctypes.windll.kernel32.GetFileAttributesW(full_path)
        if attrs == -1:  # Error getting attributes
            return name.startswith(".")  # Fallback

        return bool(attrs & FILE_ATTRIBUTE_HIDDEN)
    except Exception:
        # Fallback to Unix-style detection if Windows API fails
        return name.startswith(".")


# Platform-specific hidden detection function
if sys.platform == "win32":
    _is_hidden = _is_hidden_windows
else:
    _is_hidden = _is_hidden_unix


# Get supported extensions from Registry
SUPPORTED_EXTENSIONS = Registry.supported_extensions()


def find_ebook_files(
    root_path: Path,
    search_hidden: bool = False,
    max_depth: int = 10,
    extensions: set[str] | None = None,
) -> Iterator[Path]:
    """Fast recursive finder for ebook files using os.scandir().

    This is significantly faster than pathlib.rglob() because:
    - Uses C-level directory iteration (os.scandir)
    - Doesn't create Path objects for every entry
    - Prunes directories early (skips hidden folders before entering)
    - Uses generators for memory efficiency

    Args:
        root_path: Starting directory for search
        search_hidden: If True, include hidden directories in search
        max_depth: Maximum recursion depth (default: 10)
        extensions: Specific extensions to search for (default: all supported)

    Yields:
        Path objects to ebook files

    Example:
        >>> for ebook in find_ebook_files(Path("/books")):
        ...     print(ebook)

        >>> # Only EPUB files
        >>> for epub in find_ebook_files(Path("/books"), extensions={".epub"}):
        ...     print(epub)
    """
    if extensions is None:
        extensions = SUPPORTED_EXTENSIONS

    root_str = str(root_path.resolve())
    yield from _scandir_recursive(root_str, 0, search_hidden, max_depth, extensions)


# Backwards compatibility alias
find_epub_files = find_ebook_files


def _scandir_recursive(
    path: str,
    depth: int,
    search_hidden: bool,
    max_depth: int,
    extensions: set[str],
) -> Iterator[Path]:
    """Internal recursive scandir implementation.

    Args:
        path: Current directory path (as string for os.scandir)
        depth: Current recursion depth
        search_hidden: Whether to enter hidden directories
        max_depth: Maximum allowed depth
        extensions: Set of file extensions to look for

    Yields:
        Path objects to ebook files
    """
    if depth > max_depth:
        return

    try:
        with os.scandir(path) as scanner:
            for entry in scanner:
                try:
                    # Check if this entry is hidden (platform-specific)
                    is_hidden = _is_hidden(path, entry.name)

                    if entry.is_dir(follow_symlinks=False):
                        # Skip hidden directories unless requested
                        if is_hidden and not search_hidden:
                            continue

                        # Recurse into subdirectory
                        yield from _scandir_recursive(
                            entry.path,
                            depth + 1,
                            search_hidden,
                            max_depth,
                            extensions,
                        )

                    elif entry.is_file(follow_symlinks=False):
                        # Check if extension matches (case-insensitive)
                        ext = Path(entry.name).suffix.lower()
                        if ext in extensions:
                            yield Path(entry.path)

                except (OSError, PermissionError):
                    # Skip files/directories we can't access
                    continue

    except (OSError, PermissionError):
        # Skip directories we can't read
        return


def count_ebook_files(
    root_path: Path,
    search_hidden: bool = False,
    max_depth: int = 10,
    extensions: set[str] | None = None,
) -> int:
    """Count ebook files without building a full list (memory efficient).

    Args:
        root_path: Starting directory for search
        search_hidden: If True, include hidden directories in search
        max_depth: Maximum recursion depth (default: 10)
        extensions: Specific extensions to count (default: all supported)

    Returns:
        Number of ebook files found
    """
    count = 0
    for _ in find_ebook_files(root_path, search_hidden, max_depth, extensions):
        count += 1
    return count


# Backwards compatibility aliases
count_epub_files = count_ebook_files


__all__ = [
    "find_ebook_files",
    "find_epub_files",  # backwards compatibility
    "count_ebook_files",
    "count_epub_files",  # backwards compatibility
    "SUPPORTED_EXTENSIONS",
]
