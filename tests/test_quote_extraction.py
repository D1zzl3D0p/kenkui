"""Quote spans partition chapter text, deterministically and without a model.

Only where dialogue is, never who speaks it: attribution is a separate,
model-driven step. Keeping extraction model-free means its correctness never
depends on a provider, and Task 3's segmentation depends on the partition
being exact -- a gap silently drops audio, an overlap silently duplicates it.
"""
# ruff: noqa: RUF001, RUF002 - the typographic quotes are the data under test;
# writing them as escapes would make every case unreadable.

from __future__ import annotations

from itertools import pairwise

import pytest

from kenkui._domain.quotes import TextSpan, extract_spans

CASES = [
    pytest.param('He said, "Go away." She left.', 3, id="straight"),
    pytest.param("He said, “Go away.” She left.", 3, id="curly"),
    pytest.param('"Go away."', 1, id="quote-only"),
    pytest.param("No dialogue at all.", 1, id="narration-only"),
    pytest.param('"A," said B, "and C."', 3, id="interrupted"),
    pytest.param('Before. "One." Between. "Two." After.', 5, id="two-quotes"),
]


@pytest.mark.parametrize(("text", "expected"), CASES)
def test_spans_partition_the_text_exactly(text: str, expected: int) -> None:
    """Every character belongs to exactly one span, in source order."""
    spans = extract_spans("ch1", text)
    assert len(spans) == expected
    assert "".join(text[s.start : s.end] for s in spans) == text


@pytest.mark.parametrize(("text", "expected"), CASES)
def test_spans_are_ordered_and_contiguous(text: str, expected: int) -> None:
    """No gaps and no overlaps, which is what makes it a partition."""
    assert expected  # the count is asserted above; this case list is shared
    spans = extract_spans("ch1", text)
    assert spans[0].start == 0
    assert spans[-1].end == len(text)
    assert all(a.end == b.start for a, b in pairwise(spans))


def test_empty_text_yields_no_spans() -> None:
    """An empty chapter has nothing to partition."""
    assert extract_spans("ch1", "") == ()


def test_dialogue_spans_are_marked_and_narration_is_not() -> None:
    """The flag is what attribution later fills in; narration is never asked."""
    spans = extract_spans("ch1", 'He said, "Go away." She left.')
    assert [s.is_dialogue for s in spans] == [False, True, False]


def test_extraction_cannot_express_a_speaker() -> None:
    """Extraction finds where dialogue is; who said it needs a model.

    TextSpan carries no speaker field at all, so the separation is structural
    rather than a convention someone has to remember.
    """
    assert not hasattr(TextSpan("ch1", 0, 1, is_dialogue=True), "character_id")


def test_apostrophes_do_not_open_a_quote() -> None:
    """A possessive must not open a span, or most prose becomes dialogue.

    Safe because the single quote is not a delimiter here at all, which is
    also why single-quoted British dialogue goes undetected.
    """
    spans = extract_spans("ch1", "Darcy's horse and Bingley's carriage.")
    assert len(spans) == 1
    assert spans[0].is_dialogue is False


def test_single_quoted_dialogue_is_not_detected() -> None:
    """A documented limitation, asserted so a future change is deliberate."""
    spans = extract_spans("ch1", "'Go away,' he said.")
    assert len(spans) == 1
    assert spans[0].is_dialogue is False


def test_a_quote_mark_inside_a_word_does_not_open_a_span() -> None:
    """An inch mark or a typo should not turn the rest of a line into speech."""
    spans = extract_spans("ch1", 'a wo"rd"s end')
    assert len(spans) == 1
    assert spans[0].is_dialogue is False


def test_an_unbalanced_quote_stays_narration() -> None:
    """A stray delimiter must not swallow the rest of the chapter."""
    spans = extract_spans("ch1", 'She began, "and never finished.')
    assert len(spans) == 1
    assert spans[0].is_dialogue is False


def test_nested_quotes_do_not_break_the_partition() -> None:
    """A curly pair containing straight quotes stays one span."""
    text = "He said, “She told me 'go away' twice.” Done."
    spans = extract_spans("ch1", text)
    assert "".join(text[s.start : s.end] for s in spans) == text


def test_extraction_is_deterministic() -> None:
    """The plan fingerprint depends on this."""
    text = '"A," said B, "and C." Then D spoke.'
    assert extract_spans("ch1", text) == extract_spans("ch1", text)


def test_the_chapter_id_is_carried_on_every_span() -> None:
    """Spans from many chapters are pooled, so each must name its own."""
    spans = extract_spans("ch-7", 'He said, "Go away." She left.')
    assert {s.chapter_id for s in spans} == {"ch-7"}


# Scare quotes and acronyms are quoted but nobody says them. Marking them as
# dialogue would render an aside in a character's voice, which is audible.
SCARE_CASES = [
    pytest.param('He was what they called "gifted" then.', id="called"),
    pytest.param("She is a so-called “expert” now.", id="so-called"),
    pytest.param('The device, known as "the loom", hummed.', id="known-as"),
    pytest.param('A ship dubbed "Perseverance" sailed.', id="dubbed"),
    pytest.param('She works for "NATO" now.', id="acronym"),
    pytest.param('He studied "DNA" for years.', id="short-acronym"),
]


@pytest.mark.parametrize("text", SCARE_CASES)
def test_scare_quotes_are_not_dialogue(text: str) -> None:
    """A term being labelled is written but never said aloud."""
    spans = extract_spans("ch1", text)
    assert not any(s.is_dialogue for s in spans), text


@pytest.mark.parametrize("text", SCARE_CASES)
def test_scare_quotes_still_partition_exactly(text: str) -> None:
    """Declining to mark them dialogue must not drop the quote marks."""
    spans = extract_spans("ch1", text)
    assert "".join(text[s.start : s.end] for s in spans) == text


def test_real_dialogue_is_not_mistaken_for_a_scare_quote() -> None:
    """The label heuristic must not swallow ordinary speech."""
    spans = extract_spans("ch1", 'She called out. "Go away," he said.')
    assert sum(1 for s in spans if s.is_dialogue) == 1


def test_curly_single_quotes_are_dialogue() -> None:
    """British typography: ‘...’ is speech, unlike the straight form."""
    spans = extract_spans("ch1", "Before ‘Go away,’ he said.")
    dialogue = [s for s in spans if s.is_dialogue]
    assert len(dialogue) == 1


def test_a_curly_apostrophe_does_not_open_dialogue() -> None:
    """Don’t and Darcy’s are far commoner than single-quoted speech."""
    text = "Don’t touch Darcy’s horse."
    spans = extract_spans("ch1", text)
    assert not any(s.is_dialogue for s in spans)


def test_curly_single_quotes_inside_dialogue_stay_inside() -> None:
    """A nested quotation is part of the utterance, not a second one."""
    text = "“She said ‘go away’ to me.”"
    spans = extract_spans("ch1", text)
    assert len(spans) == 1
    assert spans[0].is_dialogue is True
