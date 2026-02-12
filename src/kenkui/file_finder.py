"""Fast file discovery utilities for finding EPUB files recursively.

Uses os.scandir() for optimal performance with cross-platform hidden folder detection.
"""

import os
import sys
from collections.abc import Iterator
from pathlib import Path


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


def find_epub_files(
    root_path: Path,
    search_hidden: bool = False,
    max_depth: int = 10,
) -> Iterator[Path]:
    """Fast recursive finder for EPUB files using os.scandir().

    This is significantly faster than pathlib.rglob() because:
    - Uses C-level directory iteration (os.scandir)
    - Doesn't create Path objects for every entry
    - Prunes directories early (skips hidden folders before entering)
    - Uses generators for memory efficiency

    Args:
        root_path: Starting directory for search
        search_hidden: If True, include hidden directories in search
        max_depth: Maximum recursion depth (default: 10)

    Yields:
        Path objects to EPUB files

    Example:
        >>> for epub in find_epub_files(Path("/books")):
        ...     print(epub)
    """
    root_str = str(root_path.resolve())
    yield from _scandir_recursive(root_str, 0, search_hidden, max_depth)


def _scandir_recursive(
    path: str,
    depth: int,
    search_hidden: bool,
    max_depth: int,
) -> Iterator[Path]:
    """Internal recursive scandir implementation.

    Args:
        path: Current directory path (as string for os.scandir)
        depth: Current recursion depth
        search_hidden: Whether to enter hidden directories
        max_depth: Maximum allowed depth

    Yields:
        Path objects to EPUB files
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
                        )

                    elif entry.is_file(follow_symlinks=False):
                        # Check for .epub extension (case-insensitive)
                        if entry.name.lower().endswith(".epub"):
                            yield Path(entry.path)

                except (OSError, PermissionError):
                    # Skip files/directories we can't access
                    continue

    except (OSError, PermissionError):
        # Skip directories we can't read
        return


def count_epub_files(
    root_path: Path,
    search_hidden: bool = False,
    max_depth: int = 10,
) -> int:
    """Count EPUB files without building a full list (memory efficient).

    Args:
        root_path: Starting directory for search
        search_hidden: If True, include hidden directories in search
        max_depth: Maximum recursion depth (default: 10)

    Returns:
        Number of EPUB files found
    """
    count = 0
    for _ in find_epub_files(root_path, search_hidden, max_depth):
        count += 1
    return count


__all__ = [
    "find_epub_files",
    "count_epub_files",
]
