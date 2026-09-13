"""Build a single-voice render step by step, with every control visible.

This is what ``magic_run()`` does, spelled out: choose chapters, shape speech,
set metadata, watch progress, and handle failures by stable error code.

    python examples/03_explicit_pipeline.py path/to/book.epub
"""

from __future__ import annotations

import sys
from pathlib import Path

import kenkui as kk

NARRATOR = "eponine"


def report(event: kk.ExecutionEvent) -> None:
    """Print stage boundaries, measured progress, and warnings."""
    if isinstance(event, kk.StageStarted):
        print(f"== {event.stage}")
    elif isinstance(event, kk.StageProgress):
        chapter = f" ({event.chapter_id})" if event.chapter_id else ""
        print(f"   {event.stage}: {event.completed}/{event.total}{chapter}")
    elif isinstance(event, kk.Warning):
        print(f"!! {event.code}: {event.message}")


def main(epub: Path) -> None:
    """Render the first three chapters of a book to ``<book>.sample.m4b``."""
    kk.load_voice(NARRATOR)

    # Every method returns a new frozen pipeline, so `base` can be branched
    # into as many variants as you like without one affecting another.
    base = kk.book(epub)
    chapters = base.inspect().chapters
    first, last = chapters[0].id, chapters[min(2, len(chapters) - 1)].id

    cover = epub.with_name("cover.jpg")
    pipeline = (
        base.select_chapter_range(first, last)
        # Speech shaping is opt-in. pronounce() enables number reading and a
        # small built-in lexicon; your own entries always win over it.
        .pronounce({"Cthulhu": "kuh-THOO-loo"}, numbers="standard")
        # Pause durations are free to retune later: cached audio is reused.
        .pauses(chapter_ms=1500, heading_after_ms=600, paragraph_ms=300)
        .assign_voice(NARRATOR)
        .tts()
        # Metadata may come after tts(). "source" keeps the EPUB's own cover.
        .metadata(cover=cover if cover.exists() else "source")
    )
    assert base.operations == ()  # branching never mutates

    # validate() is cheap and runs again inside write(); calling it first lets
    # you report every problem at once instead of failing on the first.
    validation = pipeline.validate()
    if not validation.is_valid:
        for issue in validation.errors:
            print(f"{issue.code.value}: {issue.message}")
        raise SystemExit(1)

    token = kk.CancellationToken()  # call token.cancel() from another thread
    try:
        result = pipeline.write(
            epub.with_name(f"{epub.stem}.sample.m4b"),
            on_event=report,
            cancel=token,
            workers="auto",  # or an int; bounded by chapter count and 16
            overwrite=True,  # publication is atomic either way
        )
    except kk.KenkuiError as error:
        # Branch on the machine-readable code, never on message text.
        if error.code is kk.ErrorCode.FFMPEG_NOT_FOUND:
            raise SystemExit("install ffmpeg first: brew install ffmpeg") from error
        raise

    stats = result.stats
    print(f"\nwrote {result.output}")
    print(
        f"  chapters: {stats.rendered_chapters}, segments: "
        f"{stats.synthesized_segments}, audio: {stats.duration_ms / 1000:.0f}s"
    )


if __name__ == "__main__":
    match sys.argv[1:]:
        case [path]:
            main(Path(path))
        case _:
            raise SystemExit(__doc__)
