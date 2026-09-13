# Getting started

This page takes you from installing Kenkui to rendering a book with a full
cast. Each step matches a runnable script in
[`examples/`](https://github.com/D1zzl3D0p/kenkui/tree/main/examples).

## 1. Install

You need CPython 3.11, 3.12, or 3.13, and FFmpeg 5 or newer.

```console
pip install kenkui

brew install ffmpeg                  # macOS
sudo apt-get install -y ffmpeg       # Debian / Ubuntu
```

Pocket-TTS brings in PyTorch, so expect a download of more than a gigabyte.
See [installation](installation.md) for details.

## 2. Look at your book

Before anything is downloaded or rendered, check that Kenkui reads your book
the way you expect:

```python
import kenkui as kk

book = kk.book("book.epub")
inspection = book.inspect()
print(inspection.metadata.title, inspection.metadata.author)
for chapter in inspection.chapters:
    print(chapter.id, chapter.title, chapter.speech_characters)
```

Chapter IDs such as `ch-v1-2cc5ecca00df5ea4a766ea67` are stable for a given
book. They are how you select chapters and address corrections later on.
Image-only pages, such as covers, are skipped.

[`01_inspect.py`](https://github.com/D1zzl3D0p/kenkui/blob/main/examples/01_inspect.py)
also lists the voices you can choose from.

## 3. Render with one narrator

```python
import kenkui as kk

if __name__ == "__main__":
    kk.load_voice("eponine")
    result = kk.magic_run("book.epub", narrator="eponine")
    print(result.output)
```

`load_voice()` downloads the English engine and the voice the first time it is
called, and does nothing on later calls. Rendering itself never downloads.

!!! warning "Keep rendering under `if __name__ == "__main__":`"
    Rendering runs in worker processes, and each worker imports your script
    again. Without the guard, every worker would start the whole render over.

To control the details, build the pipeline yourself. `magic_run()` is shorthand
for this:

```python
import kenkui as kk

if __name__ == "__main__":
    kk.load_voice("eponine")
    result = (
        kk.book("book.epub")
        .pronounce(numbers="standard")  # read "1984" and "£5" aloud properly
        .pauses(chapter_ms=1500, paragraph_ms=300)
        .assign_voice("eponine")
        .tts()
        .metadata(title="My Book", author="An Author", cover="source")
        .write("book.m4b", workers="auto", overwrite=True)
    )
```

Every method returns a **new** pipeline and leaves the original unchanged. See
[`03_explicit_pipeline.py`](https://github.com/D1zzl3D0p/kenkui/blob/main/examples/03_explicit_pipeline.py)
for chapter selection, progress events, cancellation, and error handling.

## 4. Give each character a voice

A cast takes three steps: find the characters, work out who speaks each line of
dialogue, and assign voices.

```console
pip install "kenkui[spacy]"
python -m spacy download en_core_web_lg
export OPENROUTER_API_KEY=...        # or any provider LiteLLM supports
```

```python
import kenkui as kk

MODEL = "openrouter/deepseek/deepseek-v4-flash"

if __name__ == "__main__":
    for voice in ("eponine", "paul", "anna", "charles", "jane", "george"):
        kk.load_voice(voice)  # casting draws only on loaded voices

    resolved = (
        kk.book("book.epub")
        .infer_characters("spacy")  # local; a LiteLLM model also works
        .attribute_quotes(MODEL)  # the step that calls a model
        .assign_voices(narrator="eponine", unknown="paul")
        .resolve()  # optional: see the cast before rendering
    )
    print(dict(resolved.inspect().casting.assignments))
    resolved.tts().write("book.m4b")
```

Casting is deterministic: the same book with the same loaded voices always gets
the same cast. Attribution is stored, so running this again, or casting the
book differently, makes no new model calls. Use `cast={"alice": "anna"}` to pin
a character to a voice.

To keep characters' voices across volumes, add
`.series("my-series", book=1)`. Before attribution, you can also
[review and correct the character list](usage.md#review-characters-before-attribution).

## 5. Fix what sounds wrong

`script()` shows what each line will be spoken as, and why. Corrections are
rules addressed by position, and a preview renders only the lines they touch:

```python
import kenkui as kk

if __name__ == "__main__":
    book = kk.book("book.epub").annotations().assign_voice("eponine")
    where = {"chapter": "*", "paragraph": 1}

    for row in book.script().at(where):
        print(row.path, row.character, row.provenance, row.text[:60])

    book = (
        book.attribute("irulan", where=where)
        .silence(900, where=where)
        .pronounce({"Atreides": "Ah-tray-deez"})
    )
    book.select(where).preview("probe.wav")
    book.write_annotations()  # saved as book.kenkui.json beside the EPUB
```

The next render that calls `.annotations()` picks these corrections up. Audio
from the preview is cached and reused by the full render.

## Where to next

- [`06_series_and_style.py`](https://github.com/D1zzl3D0p/kenkui/blob/main/examples/06_series_and_style.py):
  keep your taste in one reusable `house_style` function.
- [`08_library_batch.py`](https://github.com/D1zzl3D0p/kenkui/blob/main/examples/08_library_batch.py):
  render a whole library.
- [`09_voices.py`](https://github.com/D1zzl3D0p/kenkui/blob/main/examples/09_voices.py):
  manage voices, and register a voice you recorded.
- The [guide](usage.md) covers every option. The [API reference](api.md) lists
  every signature.
