"""Stage 3: Overlapping paragraph-boundary chunk splitter.

Chapters are split into ~600-800 word chunks that:
- Break ONLY at paragraph boundaries (never mid-sentence).
- Carry ~100 words of leading overlap from the previous chunk so the LLM
  has enough conversational context to continue A-B-A-B speaker momentum.
- Know which ``Quote`` IDs (by ``para_index``) fall inside them.

Public API
----------
chunk_paragraphs(paragraphs, quotes, target_words, overlap_words) → list[Chunk]
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .models import Quote


@dataclass
class Chunk:
    """One overlapping window over a chapter's paragraphs."""

    text: str  # Paragraphs joined by "\n\n"
    para_indices: list[int]  # Paragraph indices in this chunk (may overlap neighbours)
    quote_ids: list[int] = field(default_factory=list)  # Quote IDs whose para is in this chunk


def chunk_paragraphs(
    paragraphs: list[str],
    quotes: list[Quote],
    target_words: int = 700,
    overlap_words: int = 100,
) -> list[Chunk]:
    """Split *paragraphs* into overlapping chunks.

    Args:
        paragraphs:    Chapter paragraphs in order.
        quotes:        Pre-extracted quotes (used to map quote_ids into chunks).
        target_words:  Approximate words per chunk (break at next paragraph boundary).
        overlap_words: Approximate words of overlap carried from the previous chunk.

    Returns:
        List of ``Chunk`` objects in chapter order.  A chapter with fewer
        words than *target_words* produces exactly one chunk.
    """
    if not paragraphs:
        return []

    # Pre-compute word counts per paragraph once.
    para_words = [len(p.split()) for p in paragraphs]
    # Build a fast lookup: para_index → [quote_ids]
    para_to_quotes: dict[int, list[int]] = {}
    for q in quotes:
        para_to_quotes.setdefault(q.para_index, []).append(q.id)

    chunks: list[Chunk] = []
    start = 0

    while start < len(paragraphs):
        # Accumulate paragraphs until we reach the target word count.
        word_count = 0
        end = start
        while end < len(paragraphs):
            word_count += para_words[end]
            end += 1
            if word_count >= target_words:
                break

        indices = list(range(start, end))
        chunk_quote_ids: list[int] = []
        for idx in indices:
            chunk_quote_ids.extend(para_to_quotes.get(idx, []))

        chunks.append(
            Chunk(
                text="\n\n".join(paragraphs[start:end]),
                para_indices=indices,
                quote_ids=chunk_quote_ids,
            )
        )

        if end >= len(paragraphs):
            break

        # Walk back from `end` until we've accumulated `overlap_words` worth
        # of paragraphs; the next chunk starts there.
        overlap_count = 0
        next_start = end
        while next_start > start + 1:
            next_start -= 1  # step back first, then measure
            overlap_count += para_words[next_start]
            if overlap_count >= overlap_words:
                break

        # Always advance by at least one paragraph to prevent infinite loops.
        start = max(start + 1, next_start)

    return chunks
