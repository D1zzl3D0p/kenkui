"""Tests for kenkui.nlp.chunker — overlapping paragraph-boundary chunker."""

from __future__ import annotations

import pytest

from kenkui.nlp.chunker import Chunk, chunk_paragraphs
from kenkui.nlp.models import Quote


def _make_quote(qid: int, para_index: int) -> Quote:
    return Quote(id=qid, text=f'"Quote {qid}."', para_index=para_index, char_offset=0)


class TestChunkParagraphsEmpty:
    def test_empty_input(self):
        assert chunk_paragraphs([], []) == []

    def test_single_short_paragraph(self):
        chunks = chunk_paragraphs(["Hello world."], [])
        assert len(chunks) == 1
        assert chunks[0].text == "Hello world."
        assert chunks[0].para_indices == [0]


class TestChunkParagraphsCoverage:
    def test_all_paragraphs_covered(self):
        """Every paragraph index must appear in at least one chunk."""
        paras = [f"Paragraph {i} " + "word " * 50 for i in range(20)]
        chunks = chunk_paragraphs(paras, [], target_words=100, overlap_words=20)
        covered = set()
        for c in chunks:
            covered.update(c.para_indices)
        assert covered == set(range(len(paras)))

    def test_first_para_in_first_chunk(self):
        paras = ["word " * 400, "word " * 400]
        chunks = chunk_paragraphs(paras, [], target_words=300)
        assert 0 in chunks[0].para_indices

    def test_last_para_in_last_chunk(self):
        paras = ["word " * 400 for _ in range(6)]
        chunks = chunk_paragraphs(paras, [], target_words=300)
        last_idx = len(paras) - 1
        assert last_idx in chunks[-1].para_indices


class TestChunkParagraphsOverlap:
    def test_overlap_creates_shared_paragraphs(self):
        """Consecutive chunks should share at least one paragraph when content is large."""
        paras = ["word " * 200 for _ in range(10)]
        chunks = chunk_paragraphs(paras, [], target_words=300, overlap_words=100)
        if len(chunks) >= 2:
            shared = set(chunks[0].para_indices) & set(chunks[1].para_indices)
            assert len(shared) >= 1

    def test_no_infinite_loop_single_huge_para(self):
        """A single paragraph larger than target_words must not loop forever."""
        huge_para = "word " * 5000
        chunks = chunk_paragraphs([huge_para], [], target_words=300)
        assert len(chunks) == 1


class TestChunkParagraphsQuoteMaping:
    def test_quote_ids_mapped_to_correct_chunk(self):
        paras = ["word " * 400, '"Said Alice."', "word " * 400]
        q = _make_quote(0, para_index=1)  # quote in para 1
        chunks = chunk_paragraphs(paras, [q], target_words=300)
        # Find which chunk contains para 1
        chunk_with_quote = next(c for c in chunks if 1 in c.para_indices)
        assert 0 in chunk_with_quote.quote_ids

    def test_no_quotes_gives_empty_quote_ids(self):
        paras = ["Narrator text. " * 20]
        chunks = chunk_paragraphs(paras, [])
        assert all(c.quote_ids == [] for c in chunks)

    def test_multiple_quotes_in_same_chunk(self):
        paras = ['"A."', '"B."', '"C."']
        quotes = [_make_quote(i, i) for i in range(3)]
        chunks = chunk_paragraphs(paras, quotes, target_words=300)
        all_ids = [qid for c in chunks for qid in c.quote_ids]
        assert sorted(all_ids) == [0, 1, 2]


class TestChunkText:
    def test_chunk_text_joins_with_double_newline(self):
        paras = ["First.", "Second.", "Third."]
        chunks = chunk_paragraphs(paras, [], target_words=300)
        assert chunks[0].text == "First.\n\nSecond.\n\nThird."

    def test_single_para_chunk_text_no_separator(self):
        chunks = chunk_paragraphs(["Only paragraph."], [], target_words=300)
        assert chunks[0].text == "Only paragraph."
