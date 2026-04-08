# kenkui

![Python](https://img.shields.io/badge/python-3.12+-blue)
![Platform](https://img.shields.io/badge/platform-macOS%20%7C%20Linux%20%7C%20Windows-lightgrey)
![License](https://img.shields.io/github/license/D1zzl3D0p/kenkui)
![PyPI](https://img.shields.io/pypi/v/kenkui)

> **Freaky fast audiobook generation from ebooks. No GPU. No nonsense.**

kenkui turns ebooks into high-quality M4B audiobooks using state-of-the-art text-to-speech — **entirely on CPU**, and faster than anything else I've used.

It's built on top of [Kyutai's pocket-tts](https://github.com/kyutai-labs/pocket-tts), with all the annoying parts handled for you: chapter parsing, batching, metadata, covers, voices, and sane defaults.

If you have ebooks and want audiobooks, kenkui is for you.

---

## ✨ Features

- Freaky fast audiobook generation
- No GPU needed, 100% CPU
- Super high-quality text-to-speech
- Multithreaded
- Supports EPUB, MOBI/AZW, and FB2
- Interactive wizard with smart defaults and Escape to go back
- Job queue with live progress dashboard
- **Multi-voice narration** — different voices for different characters, powered by an LLM
- **Chapter-voice mode** — assign a distinct voice to each chapter
- Three tiers of voices: compiled, built-in, and custom
- Flexible chapter selection with presets and manual override
- Broadcast-quality audio post-processing chain
- Automatic cover embedding
- Sensible defaults, minimal configuration

---

## 🚀 Quick Start

kenkui is intentionally easy to install and easy to use.

### One-line installer (macOS / Linux)

```bash
curl -sSL https://raw.githubusercontent.com/D1zzl3D0p/kenkui/main/install.sh | bash
```

### One-line installer (Windows)

```powershell
powershell -Command "irm https://raw.githubusercontent.com/D1zzl3D0p/kenkui/main/install.ps1 | iex"
```

### Requirements

- Python **3.12+**
- One Python installer: `uv` (recommended), `pip`, or `pipx`

### Manual install

```bash
uv tool install kenkui
```

Or with pip/pipx:

```bash
pip install kenkui
# or
pipx install kenkui
```

Compiled voices (~440 MB) are downloaded automatically on first run. To download them ahead of time:

```bash
kenkui voices download
```

### Run

```bash
kenkui book.epub
```

That's it. An interactive wizard walks you through the setup, then kenkui queues the job, starts the worker, and shows a live progress dashboard. You'll get a `book.m4b` alongside your ebook when it's done.

---

## 📚 Usage

### Interactive wizard (default)

```bash
kenkui book.epub
```

Walks you through:

1. Chapter selection
2. Narration mode (single voice, multi-voice, or chapter-voice)
3. Optional per-job quality overrides
4. Output directory
5. Voice assignment (fast character scan for multi-voice, then voice picker)
6. Final confirmation

Press **Escape** at any step to go back to the previous one.

Then auto-starts the queue with a live Rich dashboard.

### Headless mode

Pass a config file with `-c` to skip the wizard entirely:

```bash
kenkui book.epub -c my-config.toml
```

Loads config, queues the job, starts the worker, and streams progress to the terminal. Exits 0 on success, 1 on failure.

### Add to queue without starting

```bash
kenkui add book.epub
```

Queues the job interactively but doesn't auto-start processing. Useful when you want to queue several books first.

```bash
# Headless queue-only (no auto-start)
kenkui add book.epub -c my-config.toml
```

### Queue management

```bash
# Snapshot: show all jobs in a Rich table and exit
kenkui queue

# Live-refreshing dashboard (Ctrl+C to exit)
kenkui queue --live

# Start processing the next pending job
kenkui queue start

# Start processing + enter live dashboard
kenkui queue start --live

# Stop the current job
kenkui queue stop
```

---

## 🎙️ Narration Modes

### Single voice

The default. One voice narrates everything.

### Multi-voice (character narration)

kenkui uses an NLP pipeline to identify characters in the book and assigns each a distinct voice. The narrator gets its own voice too.

**Requirements:**
- [Ollama](https://ollama.com) running locally (`ollama serve`)
- NLP model pulled (default: `llama3.2`) — `ollama pull llama3.2`
- spaCy model — kenkui downloads this automatically if missing

The wizard checks all requirements and shows a status table before proceeding.

**How it works:**

The wizard runs a fast character scan (seconds) before voice assignment so you can assign voices and walk away — the slower LLM attribution phase runs unattended in the background before TTS begins.

**Assignment modes in the wizard:**

- **Simple** — all male characters share one voice, all females share another
- **Advanced** — individual voice per character; kenkui auto-assigns by gender and resolves conflicts for characters that appear in the same chapter, then lets you review and adjust

When reviewing assignments, the voice picker shows which other characters are already using each voice (e.g. `← Rand al'Thor`) so you can avoid conflicts at a glance. Press **Enter** to accept all assignments.

The Ollama model used for attribution is configurable via `nlp_model` in your config.

### Chapter-voice mode

Assign a distinct voice to each chapter. The wizard presents each chapter title and lets you pick a voice for it.

---

## 🎭 Voice System

Voices come in three tiers:

| Tier | Source | Auth required? |
|------|--------|---------------|
| **Compiled** | Downloaded from HuggingFace on first run | No |
| **Built-in** | 8 pocket-tts defaults | No |
| **Custom** | `.wav` files (user-provided or fetched) | Yes (HuggingFace) |

**Built-in voices:**
```
alba, marius, javert, jean, fantine, cosette, eponine, azelma
```

### Voice manager (interactive TUI)

```bash
kenkui voices
```

Launches an interactive voice manager: browse and audition voices, manage the auto-assignment exclusion pool, and look up the character cast for a completed multi-voice book.

### Listing voices

```bash
kenkui voices list

# Filter by metadata
kenkui voices list --gender Female
kenkui voices list --accent Scottish
kenkui voices list --dataset VCTK
kenkui voices list --source compiled
```

### Auditioning voices

```bash
kenkui voices audition <voice>

# Custom preview text
kenkui voices audition <voice> --text "Your preview text here."
```

Synthesizes a short clip and opens it in your system audio player.

### Downloading compiled voices

Compiled voices are downloaded automatically on first run. To download them manually or re-download:

```bash
kenkui voices download

# Force a fresh re-download
kenkui voices download --force
```

Voices are stored at `~/.local/share/kenkui/voices/`.

### Fetching custom voices

```bash
kenkui voices fetch --repo user/repo-name

# Or set an env var
KENKUI_VOICES_REPO=user/repo-name kenkui voices fetch
```

Downloads `.wav` files from a HuggingFace repo to `~/.local/share/kenkui/voices/uncompiled/`. Requires a free HuggingFace account (see FAQ).

### Managing the auto-assignment pool

```bash
# Exclude a voice from multi-voice auto-assignment
kenkui voices exclude <voice>

# Restore an excluded voice
kenkui voices include <voice>
```

Excluded voices are still available for manual assignment in the wizard; they're just skipped during automatic gender-based matching.

### Looking up a book's cast

```bash
kenkui voices cast <title>
```

Displays the character→voice cast for a completed multi-voice book (fuzzy-matched by title).

### Using your own voice

Record a **5–10 second** clip of clean speech with minimal background noise or crosstalk. Cleaning the audio makes a noticeable difference — tools like Adobe's Enhance Speech work well:
<https://podcast.adobe.com/en/enhance>

You can pass a local `.wav` file directly in the wizard, or use a Hugging Face URL:

```
hf://user/repo/voice.wav
```

---

## ⚙️ Configuration

kenkui uses TOML config files stored at `~/.config/kenkui/` (XDG).

```bash
# Create or edit the default config
kenkui config default

# Create a named config
kenkui config fast-mode

# Use a named config
kenkui book.epub -c fast-mode
```

Named configs without a path separator are automatically looked up in `~/.config/kenkui/`.

### Key settings

| Key | Default | Description |
|-----|---------|-------------|
| `workers` | `cpu_count - 2` | Parallel TTS worker processes |
| `m4b_bitrate` | `96k` | Output audio bitrate |
| `temp` | `0.7` | Sampling temperature (lower = stable, higher = expressive) |
| `lsd_decode_steps` | `1` | LSD decode steps (higher = better quality, slower) |
| `noise_clamp` | off | Noise clamp (~3.0 reduces audio glitches) |
| `eos_threshold` | `-4.0` | End-of-speech detection threshold |
| `frames_after_eos` | auto | Frames after EOS cutoff (0 = suppress trailing noise) |
| `default_voice` | `alba` | Fallback voice when no per-job override |
| `default_chapter_preset` | `content-only` | Default chapter filter preset |
| `default_output_dir` | next to source | Where to write output files |
| `pause_line_ms` | `800` | Pause between lines (ms) |
| `pause_chapter_ms` | `2000` | Pause between chapters (ms) |
| `pause_scene_break_ms` | `4000` | Pause at scene breaks (ms) |
| `nlp_model` | `llama3.2` | Ollama model for multi-voice speaker attribution |
| `nlp_confidence_threshold` | `0` | Min attribution confidence score; 0 = second-pass disabled |
| `nlp_review_model` | `""` | Ollama model for second-pass retry; `""` = same as `nlp_model` |
| `excluded_voices` | `[]` | Voices excluded from auto-assignment (still available manually) |

### Per-job quality overrides

The interactive wizard lets you override quality settings for a single job without changing your config. These are the same settings as above, prefixed with `job_`:

`job_temp`, `job_lsd_decode_steps`, `job_noise_clamp`, `job_eos_threshold`, `job_m4b_bitrate`, `job_pause_line_ms`, `job_pause_chapter_ms`, `job_frames_after_eos`

---

## 📖 Chapter Selection

The wizard lets you choose which chapters to include. Available presets:

| Preset | Description |
|--------|-------------|
| `content-only` | Body chapters only, skips front/back matter *(default)* |
| `chapters-only` | Titled chapters only |
| `with-parts` | Chapters and part headings |
| `all` | Every item in the ebook |
| `none` | Skip everything |

After selecting a preset, the wizard shows a checkbox list of all chapters with the preset's defaults pre-selected. You can toggle individual chapters from there.

---

## 🔊 Audio Post-Processing

kenkui applies a broadcast-quality effects chain to every chapter WAV before stitching:

1. Noise reduction
2. High-pass filter (removes low-end rumble)
3. Low shelf EQ (reduces boominess)
4. Presence boost (clarity)
5. De-esser
6. Compressor
7. Limiter
8. Autogain (EBU R128 normalization)

Final loudness normalization is available as an optional step. All parameters are configurable via the `[post_processing]` section of your config.

---

## FAQ

**Do I need a GPU?**
No. kenkui is 100% CPU-based.

**Is it actually fast?**
Yes. That's the entire point of the project.

**What output format does it use?**
M4B, with chapters, metadata, and embedded covers.

**What ebook formats does it support?**
EPUB, MOBI/AZW/AZW3/AZW4, and FB2.

**Can it generate MP3s?**
No. This is intentional — M4B is a significantly better format for audiobooks.

**How does multi-voice narration work?**
kenkui runs a two-stage NLP pipeline: first BookNLP and spaCy identify characters and their dialogue; then an Ollama LLM resolves ambiguous attribution. The result is a speaker map where each character speaks in their assigned voice and the narrator fills everything else.

**Why do I need Ollama for multi-voice?**
The LLM resolves ambiguous dialogue attribution that rule-based systems can't handle reliably. It runs locally via Ollama — nothing leaves your machine.

**Why do I need a HuggingFace account for custom voices?**
The pocket-tts model is gated on HuggingFace, meaning the authors require users to accept their terms before downloading. This only applies to custom uncompiled `.wav` voices — compiled voices and built-ins require no authentication at all.

When you first use a custom voice, kenkui guides you through creating a free HuggingFace account, generating a read-only token, and accepting the model's terms. You only need to do this once.

**Where are voices stored?**
Compiled voices are downloaded to `~/.local/share/kenkui/voices/compiled/` on first run. Custom voices go to `~/.local/share/kenkui/voices/uncompiled/`. Run `kenkui voices download --force` to re-download from scratch.

**Does it upload my books anywhere?**
No. Everything runs locally. Internet access is only needed to pull models from HuggingFace or Ollama the first time.

**Why isn't kenkui finding my ebook in a hidden directory?**
kenkui doesn't search hidden directories by default. Pass the file directly:

```bash
kenkui /path/to/hidden/directory/book.epub
```

---

## Non-Goals

kenkui is not meant to be:

- A general-purpose text-to-speech framework
- A GUI application
- An MP3 audiobook generator
- A pluggable frontend for every TTS backend available

The focus is narrow by design: fast, high-quality audiobook generation from ebooks, with minimal friction.

---

## 🙏 Special Thanks

Thanks to **Project Gutenberg** for providing some of the public-domain books included with kenkui.

---

## Voice Dataset Credits

kenkui's compiled voices are derived from two publicly available speech corpora.

### CSTR VCTK Corpus

> Veaux, Christoph; Yamagishi, Junichi; MacDonald, Kirsten. (2019). *CSTR VCTK Corpus: English Multi-speaker Corpus for CSTR Voice Cloning Toolkit*. University of Edinburgh. The Centre for Speech Technology Research (CSTR).

Licensed under [Creative Commons Attribution 4.0 (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/).
Commercial use is permitted with attribution.

### EARS Dataset

Licensed under [Creative Commons Attribution-NonCommercial 4.0 (CC BY-NC 4.0)](https://creativecommons.org/licenses/by-nc/4.0/).

> **Note:** Compiled voices sourced from EARS (identifiable by `EARS` in the voice name via `kenkui voices list`) **may not be used for commercial purposes**. If you are building a commercial product with kenkui, use only VCTK-sourced or built-in voices.
