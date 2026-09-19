"""Public rendering bounds, and the estimates that describe a run before it starts.

A chapter's PCM is streamed to its own part file as each segment arrives, so a
run holds one segment in memory rather than one chapter: a chapter's length costs
disk and time, never memory. Nothing here caps how long a chapter may be. A book
whose endnotes run fourteen hours is a strange book, not a broken one, and only
the person choosing chapters can say which it is -- so length is reported rather
than refused, as a `Warning` event during planning and as an estimate a caller
can show while chapters are still being chosen.

What remains is a bound on untrusted worker output rather than on content: one
segment's PCM, and one run's total. Both are far past any book.
"""

from __future__ import annotations

MAX_SEGMENT_PCM_BYTES = 64 * 1024 * 1024
# About 370 hours of 24 kHz mono audio. This bounds a misbehaving worker, not a
# long book: no selection of real chapters approaches it.
MAX_TOTAL_PCM_BYTES = 64 * 1024 * 1024 * 1024

# Measured across rendered books, including endnote-dense non-fiction, which
# runs at the slow end. Estimates from it are approximate by nature: they exist
# to tell six hours from thirty minutes, not to predict a duration exactly.
TYPICAL_SPEECH_CHARACTERS_PER_SECOND = 13
# Real chapters run to about two hours. Past this, a chapter is usually a whole
# book's endnotes or front matter swept into one file, which is worth saying out
# loud before someone waits for it.
LONG_CHAPTER_HOURS = 6.0


def estimated_audio_hours(speech_characters: int) -> float:
    """Estimate the hours of narration a count of speech characters becomes."""
    return speech_characters / TYPICAL_SPEECH_CHARACTERS_PER_SECOND / 3600


def is_long_chapter(speech_characters: int) -> bool:
    """Report whether one chapter is long enough to be worth remarking on."""
    return estimated_audio_hours(speech_characters) >= LONG_CHAPTER_HOURS
