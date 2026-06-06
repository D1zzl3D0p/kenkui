# kenkui

![Python](https://img.shields.io/badge/python-3.12+-blue)
![Platform](https://img.shields.io/badge/platform-macOS%20%7C%20Linux%20%7C%20Windows-lightgrey)
![License](https://img.shields.io/github/license/D1zzl3D0p/kenkui)
![PyPI](https://img.shields.io/pypi/v/kenkui)

> Ebook-to-audiobook conversion engine for Python clients.

kenkui is a Python library that converts ebooks into high-quality M4B audiobooks using [Kyutai's pocket-tts](https://github.com/kyutai-labs/pocket-tts), running entirely on CPU.

**Looking for the interactive CLI?** Install [kentui](https://github.com/D1zzl3D0p/kentui) — it's the interactive front-end built on top of this library.

---

## Install

```bash
pip install kenkui
```

Or with uv:

```bash
uv add kenkui
```

---

## Quick Start

```python
from pathlib import Path

import kenkui

# Load config (creates default at ~/.config/kenkui/config.toml on first run)
config = kenkui.load_config()

# Build a ProcessingConfig. Client apps own prompts, queues, and transport.
proc = kenkui.ProcessingConfig(
    voice="alba",
    ebook_path=Path("book.epub"),
    output_path=Path("."),
    pause_line_ms=config.pause_line_ms,
    pause_chapter_ms=config.pause_chapter_ms,
    workers=config.workers,
    m4b_bitrate=config.m4b_bitrate,
    keep_temp=config.keep_temp,
    debug_html=False,
    chapter_filters=[],
)

# Run the conversion
ok = kenkui.run_job(proc)
```

---

## Features

- Freaky fast M4B audiobook generation — 100% CPU, no GPU
- Multithreaded chapter processing
- Supports EPUB, MOBI/AZW/AZW3/AZW4, and FB2
- Multi-voice narration via NLP speaker attribution (Ollama, Anthropic, OpenAI, Google)
- Voice pool template for automatic voice assignment by role + gender + rank
- Chapter-voice mode: distinct voice per chapter
- Broadcast-quality audio post-processing chain
- Credits chapter: synthesized audio appended to every m4b
- Flexible chapter selection (presets + manual override)
- Series support: cross-book character roster with pinned voice assignments

---

## Library Boundary

`kenkui` is the reusable core. It owns parsing, config models, NLP/cache logic,
voice selection, rendering workers, post-processing, and public dataclasses.

External clients own user interaction, HTTP routes, queues, deployment policy,
notifications, and remote execution. `kentui` is the interactive terminal client.
`kenkui-server` can wrap this library as a local or remote service without moving
server policy into this package.

---

## API Reference

### Config

```python
config = kenkui.load_config()                  # default config
config = kenkui.load_config("fast-mode")       # named config
config = kenkui.load_config("/path/to/cfg.toml")

kenkui.save_config(config)
kenkui.save_config(config, "fast-mode")

names: list[str] = kenkui.list_configs()
```

### Book parsing

```python
result = kenkui.parse_book("book.epub")
# result.chapters, result.metadata, result.book_hash

filtered = kenkui.filter_chapters(result.book_hash, selection)
```

### NLP

```python
# Stage 1-2: entity scan + character clustering
scan = kenkui.fast_scan("book.epub", nlp_model="llama3.2")
# scan.characters: list[CharacterInfo]

# Stages 1-4: full pipeline with speaker attribution
result = kenkui.full_analysis(
    "book.epub",
    nlp_model="llama3.2",
    progress_callback=lambda pct, msg: print(f"{pct}% {msg}"),
)
# result.characters, result.chapters (annotated)
```

### Voices

```python
voices = kenkui.list_voices()
voices = kenkui.list_voices(gender="Female", accent="British", source="compiled")

voice = kenkui.get_voice("alba")   # VoiceInfo | None

cast = kenkui.suggest_cast(
    roster=scan.characters,
    excluded_voices=["alba"],
    default_voice="sarah",
)
# cast.speaker_voices: dict[str, str]

narrator = kenkui.recommend_narrator(scan.characters, default_voice="alba")

result = kenkui.exclude_voice("marius")   # ExcludeResult
result = kenkui.include_voice("marius")   # IncludeResult

preview = kenkui.audition_voice("alba", text="Hello world.")
# preview.audio_path, preview.duration_ms

dl = kenkui.download_voice(force=False)   # DownloadResult
dl = kenkui.fetch_voice(repo_id="user/repo")
```

### Series

```python
series_list = kenkui.list_series()
entry = kenkui.get_series("my-series")
entry = kenkui.create_series("My Series")
kenkui.update_series(entry)
```

### HuggingFace auth

```python
result = kenkui.authenticate_huggingface("hf_token_here")
# result.authenticated, result.username
```

### Job runner

```python
from pathlib import Path

from kenkui import ProcessingConfig
from kenkui.chapter_filter import FilterOperation

config = ProcessingConfig(
    voice="alba",
    ebook_path=Path("book.epub"),
    output_path=Path("."),
    pause_line_ms=800,
    pause_chapter_ms=2000,
    workers=4,
    m4b_bitrate="96k",
    keep_temp=False,
    debug_html=False,
    chapter_filters=[FilterOperation("preset", "content-only")],
)

ok: bool = kenkui.run_job(config)
ok: bool = kenkui.run_job(
    config,
    progress_callback=lambda pct, chapter, eta: print(f"{pct:.0f}% {chapter}"),
)
```

---

## Voice System

Voices come in three tiers:

| Tier | Source | Auth required? |
|------|--------|---------------|
| **Compiled** | Downloaded from HuggingFace on first run | No |
| **Built-in** | 8 pocket-tts defaults | No |
| **Custom** | `.wav` files (user-provided or fetched) | Yes (HuggingFace) |

**Built-in voices:** `alba, marius, javert, jean, fantine, cosette, eponine, azelma`

---

## Configuration

kenkui uses TOML config files stored under the XDG config directory, typically
`~/.config/kenkui/`. Cache/state files live under the XDG cache directory,
typically `~/.cache/kenkui/`.

Settings are environment-aware through the `KENKUI_` prefix. Environment values
take precedence when constructing `AppConfig`; saved TOML files remain the local
convenience path for desktop clients.

| Key | Default | Description |
|-----|---------|-------------|
| `workers` | `cpu_count - 2` | Parallel TTS worker processes |
| `m4b_bitrate` | `96k` | Output audio bitrate |
| `temp` | `0.7` | Sampling temperature |
| `lsd_decode_steps` | `1` | LSD decode steps |
| `default_voice` | `alba` | Fallback voice |
| `nlp_provider` | `ollama` | NLP backend |
| `nlp_model` | `llama3.2` | Model for speaker attribution |
| `credits_enabled` | `true` | Append synthesized credits audio |

Logging is 12-factor friendly: library modules log through Python logging and do
not require file logging. Clients may configure stdout/stderr or file handlers.
Credentials should be supplied through the environment or explicit client-owned
auth flows. Local credential files are convenience only, not the preferred
deployment path.

---

## Non-Goals

kenkui is not a general-purpose TTS framework, GUI app, CLI, queue server, cloud
control plane, benchmarking system, or MP3 generator. The focus is narrow:
fast, high-quality audiobook generation from ebooks.

---

## Special Thanks

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

> **Note:** Compiled voices sourced from EARS (identifiable by `EARS` in the voice name via `kenkui.list_voices()`) **may not be used for commercial purposes**. If you are building a commercial product with kenkui, use only VCTK-sourced or built-in voices.
