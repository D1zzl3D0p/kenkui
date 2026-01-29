# Kenkui

Kenkui is basically a fancy wrapper for Kyutai's pocket-tts, with support for
ebook parsing. It is multithreaded, and runs faster than any other tool I've
used, so I figured I'd start a project to make it easier to use.

## Quick Start

```bash
uv tool install kenkui
kenkui <your ebook name>.epub
```

## Usage

You can pass a file or directory into kenkui, and it will search directories
recursively for .epub files and convert them to m4b files.

Use `--voice` to specify the voice you want to use.

Use `--list-voices` to see all of the voices available.

Use `--select-chapters` and `--select-books` to specify which chapters or books
you want to use.

To test out the default voices, go [to kyutai's official website](https://kyutai.org/blog/2026-01-13-pocket-tts) to try them out.

## Notes

At this time we do not plan on supporting mp3, not because it's hard, but
because m4b is a wonderful format. There is also currently not support for
ebook formats other than epub. I'm sure it'll get added in the future.

For similar reasons, currently only pocket-tts is supported as the tts
provider. It's the smallest, fastest, and most feature complete at the moment.
