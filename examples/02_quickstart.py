"""Render a whole book with one narrator, in the fewest possible lines.

The first run downloads the English engine (about 225 MB) and one voice
embedding (about 6.5 MB). Later runs reuse them and work offline. Requires
``ffmpeg`` and ``ffprobe`` on PATH.

    python examples/02_quickstart.py path/to/book.epub
"""

from __future__ import annotations

import sys
from pathlib import Path

import kenkui as kk


def main(epub: Path) -> None:
    """Provision a narrator voice and write ``book.m4b`` beside ``book.epub``."""
    # Rendering never downloads. load_voice() is the one explicit step that
    # makes a voice renderable; it is idempotent, so leaving it in is free.
    kk.load_voice("eponine")

    # magic_run() is the defaulted path: whole book, one voice, output beside
    # the EPUB. Pass multi=True for a character cast (see 04_multi_voice.py).
    result = kk.magic_run(epub, narrator="eponine")
    minutes = result.stats.duration_ms / 60_000
    print(f"wrote {result.output} ({minutes:.1f} minutes)")


# The guard is required, not style: rendering spawns worker processes, and
# each worker re-imports this file. Without it, every worker would start the
# whole render again.
if __name__ == "__main__":
    match sys.argv[1:]:
        case [path]:
            main(Path(path))
        case _:
            raise SystemExit(__doc__)
