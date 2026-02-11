# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- New CLI-based chapter filtering system with regex support
  - `--include-chapter`/`-I` flag for regex-based chapter inclusion (repeatable)
  - `--exclude-chapter`/`-X` flag for regex-based chapter exclusion (repeatable)
  - `--preview` flag to preview which chapters will be included/excluded
  - Operations applied sequentially with later operations taking precedence
  - Smart initialization: `--include-chapter` starts from empty, `--exclude-chapter` starts from all
- `FilterOperation` dataclass for structured filter operations
- `register_preset()` method for extensible preset registration
- Comprehensive logging of chapter filter decisions
- 20 new unit tests for chapter filtering functionality
- New test file: `tests/test_chapter_filter.py`

### Changed

- **BREAKING**: Removed `--select-chapters` flag and Textual-based interactive TUI
- **BREAKING**: Removed `chapter_selector.py` module entirely
- `Config` dataclass now uses `chapter_filters: list[FilterOperation]` instead of `interactive_chapters` and `chapter_filter_preset` fields
- Updated CLI help text to document new filtering options
- Updated README with comprehensive chapter filtering documentation and examples
- Updated `__init__.py` exports to remove `chapter_selector` references

### Removed

- `textual` dependency from `pyproject.toml`
- `chapter_selector.py` (448 lines of Textual-based TUI code)
- Interactive chapter selection functionality
- All references to `interactive_select_chapters` and `ChapterSelectorApp`

### Fixed

- Chapter filtering now properly initializes selection based on first operation type
- Exclude operations without presets now correctly start with all chapters
- Include operations without presets now correctly start with empty selection

## [0.5.0] - 2025-02-08

### Added

- Interactive chapter selection with `--select-chapters` flag
- Textual-based TUI for chapter selection with search and filter options
- Chapter filter presets: `all`, `content-only`, `chapters-only`, `with-parts`
- `--chapter-preset` flag for preset selection
- Chapter classification system with automatic tagging
  - Front matter detection (Acknowledgments, Preface, Introduction, etc.)
  - Back matter detection (References, Index, Bibliography, etc.)
  - Title page detection
  - Part/Book divider detection
- `chapter_classifier.py` module for automatic chapter tagging
- `chapter_filter.py` module for filter preset management
- `chapter_selector.py` module for interactive TUI
- Batch processing for multiple EPUB files
- Automatic output naming based on book title
- `--fix-audiobook` flag for repairing metadata and missing chapters

### Changed

- Refactored CLI argument parsing with better help text
- Improved error handling and user feedback
- Enhanced progress tracking with ETA calculation
- Better voice validation and error messages

### Fixed

- Memory usage optimizations for large EPUBs
- Better handling of malformed EPUB files
- Fixed progress bar display issues
- Resolved threading issues with worker processes

## [0.4.0] - 2025-02-02

### Added

- Custom voice support via Hugging Face URLs (`hf://user/repo/voice.wav`)
- Local voice file support (`.wav` files)
- `--list-voices` flag to display available voices
- Voice descriptions and bundled voice detection
- `--verbose`/`-v` flag for detailed logging
- `--debug`/`-d` flag for full tracebacks
- `--keep` flag to preserve temporary files
- `--version` flag to display version information
- Hugging Face authentication check on startup

### Changed

- Updated default voice to "alba"
- Improved voice selection interface
- Better error messages for missing dependencies

### Fixed

- Fixed audio stitching issues with certain EPUB structures
- Resolved cover extraction failures
- Fixed temporary directory cleanup on interruption

## [0.3.0] - 2025-01-29

### Added

- Multithreaded processing with `--workers`/`-w` flag
- Automatic CPU core detection for worker count
- Progress bars with Rich library
- ETA calculation based on processing speed
- Batch processing for chapter text
- `--output`/`-o` flag for output directory specification

### Changed

- Replaced single-threaded processing with ProcessPoolExecutor
- Improved memory efficiency for large books
- Better progress visualization

### Fixed

- Fixed audio quality issues at chapter boundaries
- Resolved memory leaks in long-running conversions
- Fixed progress bar not updating correctly

## [0.2.0] - 2025-01-25

### Added

- M4B output format with chapter support
- Automatic cover embedding from EPUB
- Metadata preservation (title, author, etc.)
- Chapter marker support in output files
- `fix_audiobook.py` utility module
- `utils.py` with cover extraction and text cleaning

### Changed

- Switched from MP3 to M4B as default output format
- Improved audio encoding quality

### Fixed

- Fixed chapter boundary detection
- Resolved encoding issues with special characters
- Fixed cover image format detection

## [0.1.0] - 2025-01-20

### Added

- Initial release
- Basic EPUB to audiobook conversion
- Support for pocket-tts voices
- Single-file processing
- Basic CLI interface
- Chapter extraction from EPUB
- Audio generation with pause insertion
- `parsing.py` with EpubReader and AudioBuilder
- `workers.py` for TTS processing
- `helpers.py` with Config and utility functions

---

## Migration Guide

### From 0.5.0 to Unreleased

If you were using `--select-chapters` for interactive selection, you'll need to use the new filtering flags:

**Old way:**
```bash
kenkui book.epub --select-chapters
```

**New way:**
```bash
# Preview what would be selected
kenkui book.epub --preview

# Use regex patterns to filter
kenkui book.epub --include-chapter "Chapter [0-9]+"

# Or combine with presets
kenkui book.epub --chapter-preset content-only --exclude-chapter "Appendix"
```

The new system is more flexible and scriptable, allowing precise control over chapter selection without interactive prompts.
