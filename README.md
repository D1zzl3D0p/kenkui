# kenkui

![Python](https://img.shields.io/badge/python-3.7+-blue)
![Platform](https://img.shields.io/badge/platform-macOS%20%7C%20Linux%20%7C%20Windows-lightgrey)
![License](https://img.shields.io/github/license/D1zzl3D0p/kenkui)
![PyPI](https://img.shields.io/pypi/v/kenkui)

> **Freaky fast audiobook generation from EPUBs. No GPU. No nonsense.**

kenkui turns EPUB ebooks into high-quality M4B audiobooks using state-of-the-art text-to-speech — **entirely on CPU**, and faster than anything else I've used.

It's built on top of [Kyutai's pocket-tts](https://github.com/kyutai-labs/pocket-tts), with all the annoying parts handled for you: chapter parsing, batching, metadata, covers, voices, and sane defaults.

If you have ebooks and want audiobooks, kenkui is for you.

---

## ✨ Features

- Freaky fast audiobook generation
- No GPU needed, 100% CPU
- Super high-quality text-to-speech
- Multithreaded
- EPUB-aware chapter parsing
- **Flexible chapter filtering with regex patterns and presets**
- Custom voices
- Batch processing
- Automatic cover embedding (EPUB → M4B)
- Sensible defaults, minimal configuration

---

## 🚀 Quick Start

kenkui is intentionally easy to install and easy to use.

### Requirements

- Python **3.7+**
- `ffmpeg` (https://ffmpeg.org/download.html)
- One Python installer:
  - `uv` (recommended): https://docs.astral.sh/uv/getting-started/installation/
  - or `pip`
  - or `pipx`

### Install

Recommended:

```bash
uv tool install kenkui
```

Alternatives if you prefer:

```bash
pip install kenkui
```

```bash
pipx install kenkui
```

### Run

```bash
kenkui book.epub
```

That's it. You'll get a `book.m4b` alongside your EPUB.

You can also point Kenkui at a directory, and it will recursively convert all EPUBs it finds.

---

## 📚 Usage

You can pass either a single EPUB file or a directory.

```bash
# Convert a single book
kenkui book.epub

# Convert an entire library
kenkui library/

# Specify output directory
kenkui book.epub -o output/
```

### 🎙️ Voice Selection

Use `--voice` to choose a voice.

Accepted inputs:

- One of pocket-tts's default voices:
  ```
  alba, marius, javert, jean, fantine, cosette, eponine, azelma
  ```
- A local `.wav` file
- A Hugging Face-hosted voice:
  ```
  hf://user/repo/voice.wav
  ```

To see everything Kenkui can currently use:

```bash
kenkui --list-voices
```

### 🎭 Custom Voices

To use your own voice, record a **5–10 second** clip of clean speech with minimal background noise or crosstalk.

Cleaning the audio makes a noticeable difference. Tools like Adobe's Enhance Speech work well:
https://podcast.adobe.com/en/enhance

---

## 📖 Chapter Filtering

Kenkui includes a powerful chapter filtering system that lets you control exactly which chapters get converted. This is useful for skipping front matter (copyright pages, tables of contents), back matter (appendices, references), or selecting specific chapters.

### Preview Mode

Before converting, use `--preview` to see which chapters will be included or excluded:

```bash
kenkui book.epub --preview
```

This shows a table with all chapters, their include/exclude status, and their classification tags.

### Filter Presets

Kenkui comes with four built-in presets:

- **`content-only`** (default) — Exclude front matter, back matter, and title pages
- **`all` — Include all chapters
- **`chapters-only` — Only main chapters (exclude part/book dividers)
- **`with-parts` — Main chapters plus Part/Book dividers

Use `--chapter-preset` to select a preset:

```bash
kenkui book.epub --chapter-preset all
```

### Regex Filtering

For more control, use regex patterns to include or exclude chapters by their titles:

```bash
# Include only chapters matching a pattern (starts from empty selection)
kenkui book.epub --include-chapter "Chapter [0-9]+"

# Exclude chapters matching a pattern (starts from all chapters)
kenkui book.epub --exclude-chapter "Appendix"

# Combine multiple patterns
kenkui book.epub \
  --include-chapter "^CHAPTER I[—\s]" \
  --include-chapter "^CHAPTER II[—\s]" \
  --exclude-chapter "Appendix"
```

**Notes on regex filtering:**
- Patterns are matched against chapter titles
- Matching is case-sensitive by default (use `(?i)` for case-insensitive)
- `--include-chapter` without a preset starts with an empty selection and adds matching chapters
- `--exclude-chapter` without a preset starts with all chapters and removes matching ones

### Combining Presets and Regex

You can combine presets with regex patterns. Operations are applied in the order specified:

```bash
# Start with 'all' preset, then exclude chapters
kenkui book.epub --chapter-preset all --exclude-chapter "CHAPTER [0-9]+"

# Complex filter chain
kenkui book.epub \
  --chapter-preset content-only \
  --include-chapter "Introduction" \
  --exclude-chapter "Part [IVX]+"
```

The later operations override earlier ones, giving you precise control.

---

## FAQ

**Do I need a GPU?**  
No. kenkui is 100% CPU-based.

**Is it actually fast?**  
Yes. That's the entire point of the project.

**What output format does it use?**  
M4B, with chapters, metadata, and embedded covers.

**Can it generate MP3s?**  
No. This is intentional — M4B is a significantly better format for audiobooks.

**Does it support formats other than EPUB?**  
Not currently. EPUB only, for now.

**Does it upload my books anywhere?**  
No. Everything runs locally. Internet access is only needed if you pull voices from Hugging Face.

---

## Non-Goals

kenkui is not meant to be:

- A general-purpose text-to-speech framework
- A GUI application
- An MP3 audiobook generator
- A pluggable frontend for every TTS backend available

The focus is narrow by design: fast, high-quality audiobook generation from EPUBs, with minimal friction.

---

## 🙏 Special Thanks

Thanks to **Project Gutenberg** for providing some of the public-domain books included with Kenkui.
