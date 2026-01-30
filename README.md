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

## Requirements

- python
- pip, pipx, or uv
- ffmpeg

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

Use `--voice` to specify the voice you want to use.

- accepts:
  - the six default voices of pocket-tts (alba, marius, javert,
    jean, fantine, cosette, eponine, azelma)
  - .wav files locally on computer (kenkui ships with quite a few extra)
  - voices and safetensor files from hugging face (starts with hf://)

Use `--list-voices` to see all of the voices kenkui ships with.

Use `--select-chapters` and `--select-books` to specify which chapters or books
you want to use.

> [!NOTE]
> go [to kyutai's official website](https://kyutai.org/blog/2026-01-13-pocket-tts) to try the default voices out!

### Custom voices

In order to use your own custom voice, make sure to record a 5-10 second
clip of the person speaking, with minimal background noise or crosstalk.
We highly recommend using [some sort of tool](https://podcast.adobe.com/en/enhance) to clean the audio.

### Examples

```bash
kenkui book.epub
kenkui library/ --select-books --voice alba
kenkui book.epub -o output/ -j 4
kenkui book.epub --select-chapters --voice ~/Downloads/voice.wav
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

## Roadmap

- Add other models?
- Make the debug menu useful
- make --select chapters and select-books look as nice as list-voices
