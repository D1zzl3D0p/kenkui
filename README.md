# Kenkui

[![PyPI](https://img.shields.io/pypi/v/kenkui)](https://pypi.org/project/kenkui/)
[![Python](https://img.shields.io/pypi/pyversions/kenkui)](https://pypi.org/project/kenkui/)
[![CI](https://github.com/D1zzl3D0p/kenkui/actions/workflows/ci.yml/badge.svg)](https://github.com/D1zzl3D0p/kenkui/actions/workflows/ci.yml)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue)](https://github.com/D1zzl3D0p/kenkui/blob/main/LICENSE)

Kenkui turns EPUB books into M4B audiobooks. Speech is synthesized locally on
your CPU with [Pocket-TTS](https://github.com/kyutai-labs/pocket-tts), and a
book can be read by a single narrator or by a full cast, with a different voice
for each character.

```python
import kenkui as kk

if __name__ == "__main__":
    kk.load_voice("eponine")
    kk.magic_run("book.epub", narrator="eponine")  # writes book.m4b
```

- **One narrator or a full cast.** Characters are discovered locally with spaCy
  or by a language model, dialogue is attributed to its speaker, and voices are
  cast deterministically, staying consistent across a whole series.
- **Fix it without re-rendering.** Read the script a book will be spoken from,
  correct a speaker, a pause, or a pronunciation, and preview only that line.
- **Local and predictable.** Speech synthesis never touches the network.
  Voices are downloaded only when you ask for them, and a failed or cancelled
  render never leaves a half-written file behind.
- **A typed, immutable Python API.** Every step returns a new pipeline you can
  branch, inspect, and reuse.

## The Kenkui suite

| Project | What it is | License |
| --- | --- | --- |
| **kenkui** (this repository) | The Python library: the EPUB-to-M4B pipeline. | Apache-2.0 |
| [kenkui-server](https://github.com/D1zzl3D0p/kenkui-server) | A self-hostable API server that runs jobs on this library. | AGPL-3.0-only |
| [kenkui-studio](https://github.com/D1zzl3D0p/kenkui-studio) | A web and desktop app for creating audiobooks through the server. | AGPL-3.0-only |
| [kenkui-voices](https://huggingface.co/datasets/D1zzl3D0p/kenkui-voices) | The 95 precompiled voices in the built-in catalog. | per-voice |

## Install

Kenkui needs Python 3.11–3.13, and FFmpeg 5 or newer on your `PATH`.

```console
pip install kenkui

brew install ffmpeg                  # macOS
sudo apt-get install -y ffmpeg       # Debian / Ubuntu
```

The install is large, because Pocket-TTS depends on PyTorch. Voices are
downloaded separately, on first use: about 225 MB per language, plus about
6.5 MB per voice. To discover characters offline, also install the spaCy
extra:

```console
pip install "kenkui[spacy]"
python -m spacy download en_core_web_lg
```

## Usage

### One narrator

```python
import kenkui as kk

if __name__ == "__main__":
    kk.load_voice("eponine")  # one-time download; later runs are offline

    book = kk.book("book.epub")
    for chapter in book.inspect().chapters:
        print(chapter.id, chapter.title)

    result = (
        book.pronounce(numbers="standard")
        .pauses(chapter_ms=1500, paragraph_ms=300)
        .assign_voice("eponine")
        .tts()
        .metadata(title="My Book", author="An Author")
        .write("book.m4b", on_event=print)
    )
    print(result.output, result.stats.duration_ms)
```

Rendering starts worker processes, and each one re-imports your script, so
keep rendering under `if __name__ == "__main__":`.

### A full cast

Quote attribution uses any [LiteLLM](https://docs.litellm.ai/) model. Set the
provider's API key in your environment, for example `OPENROUTER_API_KEY`.

```python
import kenkui as kk

MODEL = "openrouter/deepseek/deepseek-v4-flash"

if __name__ == "__main__":
    for voice in ("eponine", "paul", "anna", "charles", "jane", "george"):
        kk.load_voice(voice)

    (
        kk.book("book.epub")
        .series("my-series", book=1)  # characters keep their voices in later books
        .infer_characters("spacy")  # or a LiteLLM model
        .attribute_quotes(MODEL)
        .assign_voices(narrator="eponine", unknown="paul", cast={"alice": "anna"})
        .tts()
        .write("book.m4b")
    )
```

### Correct one line and hear the fix

```python
book = kk.book("book.epub").annotations().assign_voice("eponine")
where = {"chapter": "*", "paragraph": 1}

for row in book.script().at(where):  # what will be said, and why
    print(row.path, row.character, row.provenance, row.text[:60])

book = book.attribute("irulan", where=where).silence(900, where=where)
book.select(where).preview("probe.wav")  # renders only those lines
book.write_annotations()  # saved beside the EPUB for the next render
```

## Examples

[`examples/`](https://github.com/D1zzl3D0p/kenkui/tree/main/examples) contains runnable scripts, from inspecting a book to
batch-rendering a library. Most take the path to an EPUB as their only argument:

| Script | Shows |
| --- | --- |
| [`01_inspect.py`](https://github.com/D1zzl3D0p/kenkui/blob/main/examples/01_inspect.py) | Metadata, chapter IDs, validation, and the voice catalog, with no downloads |
| [`02_quickstart.py`](https://github.com/D1zzl3D0p/kenkui/blob/main/examples/02_quickstart.py) | `load_voice()` and `magic_run()` |
| [`03_explicit_pipeline.py`](https://github.com/D1zzl3D0p/kenkui/blob/main/examples/03_explicit_pipeline.py) | Chapter selection, pronunciation, pauses, cover art, progress events, cancellation, error codes |
| [`04_multi_voice.py`](https://github.com/D1zzl3D0p/kenkui/blob/main/examples/04_multi_voice.py) | Character inference, attribution, casting, and inspecting the cast before rendering |
| [`05_review_characters.py`](https://github.com/D1zzl3D0p/kenkui/blob/main/examples/05_review_characters.py) | Correcting the discovered roster before attribution |
| [`06_series_and_style.py`](https://github.com/D1zzl3D0p/kenkui/blob/main/examples/06_series_and_style.py) | Series voice continuity, lexicons, and a reusable `house_style` with `.pipe()` |
| [`07_dial_in.py`](https://github.com/D1zzl3D0p/kenkui/blob/main/examples/07_dial_in.py) | The script → correct → preview → save loop |
| [`08_library_batch.py`](https://github.com/D1zzl3D0p/kenkui/blob/main/examples/08_library_batch.py) | Rendering a whole library from a table of books |
| [`09_voices.py`](https://github.com/D1zzl3D0p/kenkui/blob/main/examples/09_voices.py) | Loading, listing, and registering your own voice |

## Documentation

Full documentation is at **<https://d1zzl3d0p.github.io/kenkui/>**:

- [Getting started](https://d1zzl3d0p.github.io/kenkui/getting-started/)
- [Guide](https://d1zzl3d0p.github.io/kenkui/usage/): every feature, in depth
- [API reference](https://d1zzl3d0p.github.io/kenkui/api/)
- [Models and voices](https://d1zzl3d0p.github.io/kenkui/models-and-voices/): the catalog, and voice rights
- [Troubleshooting](https://d1zzl3d0p.github.io/kenkui/troubleshooting/): stable error codes

## Contributing

See [CONTRIBUTING.md](https://github.com/D1zzl3D0p/kenkui/blob/main/CONTRIBUTING.md). Changes are recorded in
[CHANGELOG.md](https://github.com/D1zzl3D0p/kenkui/blob/main/CHANGELOG.md).

## License

Kenkui is licensed under the [Apache License 2.0](https://github.com/D1zzl3D0p/kenkui/blob/main/LICENSE). Voices and model
weights are separate works with their own terms, and every built-in voice
ships marked as not cleared for commercial use. See
[Models and voices](https://d1zzl3d0p.github.io/kenkui/models-and-voices/).
