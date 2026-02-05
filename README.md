# Kenkui

Kenkui is basically a fancy wrapper for [Kyutai's pocket-tts](https://github.com/kyutai-labs/pocket-tts), with support for
ebook parsing. It is multithreaded, and runs faster than any other tool I've
used, so I figured I'd start a project to make it easier to use.

## Features

- Freaky fast audiobook generation
- No GPU needed, 100% cpu
- Super High Quality Text-to-Speech
- State of the Art Tools
- Multithreaded
- Custom voices
- Chapter selection
- Batch processing
- **Automatic cover embedding** from EPUB to M4B
- **Fix existing audiobooks** (salvage chapters, add missing ones, fix metadata)

## Requirements

- python 3.7+
- pip, pipx, or uv
- ffmpeg
- mutagen (installed automatically)

## Quick Start

Dependencies:

```bash
uv tool install kenkui
kenkui <your ebook name>.epub
```

You can also install using pip or pipx!

```bash
pip install kenkui

pipx install kenkui
```

## Usage

You can pass a file or directory into kenkui, and it will search directories
recursively for .epub files and convert them to m4b files.

### Basic Examples

```bash
# Convert a single book
kenkui book.epub

# Convert entire library directory
kenkui library/

# Specify output directory
kenkui book.epub -o output/

# Use specific voice
kenkui book.epub --voice alba
```

### Voice Selection

Use `--voice` to specify the voice you want to use.

- accepts:
  - the eight default voices of pocket-tts (alba, marius, javert,
    jean, fantine, cosette, eponine, azelma)
  - .wav files locally on computer (kenkui ships with quite a few extra)
  - voices from Hugging Face (hf://user/repo/voice.wav)

Use `--list-voices` to see all available voices.

### Chapter Selection

Use `--select-chapters` to interactively choose which chapters to convert.

### Fix Audiobooks

Use `--fix-audiobook` to fix existing M4B files:

```bash
# Scan audiobook for missing chapters and fix metadata
kenkui --fix-audiobook book.epub book.m4b

# With verbose output
kenkui --fix-audiobook book.epub book.m4b --verbose


```

What `--fix-audiobook` does:
1. **Scans for missing chapters** - Compares EPUB table of contents with M4B chapters
2. **Salvages existing audio** - Extracts matching chapters from existing M4B (no regeneration needed)
3. **Generates missing chapters** - Asks to generate any chapters not found in M4B
4. **Rebuilds audiobook** - Stitches all audio in proper EPUB chapter order
5. **Embeds cover** - Extracts cover from EPUB and embeds into fixed M4B
6. **Creates new file** - Outputs `book_fixed.m4b` (original preserved as backup)

### Custom voices

In order to use your own custom voice, make sure to record a 5-10 second
clip of the person speaking, with minimal background noise or crosstalk.
We highly recommend using [some sort of tool](https://podcast.adobe.com/en/enhance) to clean the audio.

### Examples

```bash
# Basic conversion
kenkui book.epub

# With voice selection
kenkui library/ --select-books --voice alba
kenkui book.epub --select-chapters --voice ~/Downloads/voice.wav

# With output directory and workers
kenkui book.epub -o output/ -w 4

# Fix an existing audiobook
kenkui --fix-audiobook book.epub book.m4b --verbose
```

## Notes

At this time we do not plan on supporting mp3, not because it's hard, but
because m4b is a wonderful format. There is also currently not support for
ebook formats other than epub. I'm sure it'll get added in the future.

For similar reasons, currently only pocket-tts is supported as the tts
provider. It's the smallest, fastest, and most feature complete at the moment.

## Special Thanks

Thank you to the Guttenberg Project for providing some books included in
kenkui!

## Changelog

### v0.4.0

- **Automatic cover embedding** - Covers from EPUB are now automatically embedded into M4B output
- **Fix audiobooks** - New `--fix-audiobook` command for:
  - Detecting missing chapters via fuzzy title matching
  - Salvaging existing audio from M4B files
  - Generating only missing chapters
  - Fixing metadata and embedding covers
- **Fuzzy chapter matching** - Smart matching between EPUB and M4B chapter titles
- **Chapter salvage** - Reuse existing audio instead of regenerating when possible
