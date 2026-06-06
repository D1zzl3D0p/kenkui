"""Unit tests for src/kenkui/nlp/annotator.py."""

from __future__ import annotations

from kenkui.nlp.annotator import annotate_chapter
from kenkui.nlp.models import Quote

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_quote(qid: int, text: str, para_index: int, char_offset: int, kind: str = "dialogue") -> Quote:
    return Quote(id=qid, text=text, para_index=para_index, char_offset=char_offset, kind=kind)


def _para_offsets(paragraphs: list[str]) -> list[int]:
    """Compute per-para start offsets matching quotes.py logic (joined by \\n\\n)."""
    offsets = []
    cursor = 0
    for p in paragraphs:
        offsets.append(cursor)
        cursor += len(p) + 2
    return offsets


# ---------------------------------------------------------------------------
# Test 1 — No quotes in paragraph → [NARRATOR] prefix
# ---------------------------------------------------------------------------


def test_no_quotes_paragraph():
    paragraphs = ["The room was silent."]
    result = annotate_chapter(paragraphs, [], {}, {})
    assert result == "[NARRATOR] The room was silent."


# ---------------------------------------------------------------------------
# Test 2 — Single quote with no hints
# ---------------------------------------------------------------------------


def test_single_quote_no_hints():
    paragraphs = ['"Hello there."']
    offsets = _para_offsets(paragraphs)
    quotes = [_make_quote(0, '"Hello there."', 0, offsets[0])]
    result = annotate_chapter(paragraphs, quotes, {}, {})
    assert result == '[QUOTE:0] "Hello there."'


# ---------------------------------------------------------------------------
# Test 3 — Leading and trailing narrator text around a quote
# ---------------------------------------------------------------------------


def test_leading_trailing_narrator():
    para = 'She smiled. "Come in," she said. The door creaked.'
    paragraphs = [para]
    offsets = _para_offsets(paragraphs)
    quote_text = '"Come in,"'
    q_start = para.index(quote_text)
    quotes = [_make_quote(0, quote_text, 0, offsets[0] + q_start)]
    result = annotate_chapter(paragraphs, quotes, {}, {})
    lines = result.split("\n\n")
    assert any(line.startswith("[NARRATOR]") and "She smiled." in line for line in lines)
    # Tag may include hint/pronoun attributes, so check prefix "[QUOTE:0"
    assert any(line.startswith("[QUOTE:0") for line in lines)
    assert any(line.startswith("[NARRATOR]") and "The door creaked." in line for line in lines)


# ---------------------------------------------------------------------------
# Test 4 — hint= extraction from "said X" pattern
# ---------------------------------------------------------------------------


def test_hint_said_alias():
    para = '"I will go," said Darrow. He turned away.'
    paragraphs = [para]
    offsets = _para_offsets(paragraphs)
    quote_text = '"I will go,"'
    q_start = para.index(quote_text)
    quotes = [_make_quote(0, quote_text, 0, offsets[0] + q_start)]
    alias_to_slug = {"darrow": "darrow_of_lykos"}
    result = annotate_chapter(paragraphs, quotes, alias_to_slug, {})
    assert 'hint="darrow_of_lykos"' in result


# ---------------------------------------------------------------------------
# Test 5 — pronoun= extraction from nearby "he said"
# ---------------------------------------------------------------------------


def test_pronoun_he_said():
    para = '"Let us proceed," he said quietly.'
    paragraphs = [para]
    offsets = _para_offsets(paragraphs)
    quote_text = '"Let us proceed,"'
    q_start = para.index(quote_text)
    quotes = [_make_quote(0, quote_text, 0, offsets[0] + q_start)]
    result = annotate_chapter(paragraphs, quotes, {}, {})
    assert 'pronoun="he/him"' in result


# ---------------------------------------------------------------------------
# Test 6 — Both hint= and pronoun= on the same tag
# ---------------------------------------------------------------------------


def test_hint_and_pronoun_coexist():
    para = '"Enough," said Cassius. He stepped forward.'
    paragraphs = [para]
    offsets = _para_offsets(paragraphs)
    quote_text = '"Enough,"'
    q_start = para.index(quote_text)
    quotes = [_make_quote(0, quote_text, 0, offsets[0] + q_start)]
    alias_to_slug = {"cassius": "cassius_au_bellona"}
    result = annotate_chapter(paragraphs, quotes, alias_to_slug, {})
    assert 'hint="cassius_au_bellona"' in result
    assert 'pronoun="he/him"' in result


# ---------------------------------------------------------------------------
# Test 7 — guess= extraction from alias right after closing quote
# ---------------------------------------------------------------------------


def test_guess_alias_after_closing_quote():
    # Tier-2 guess: alias follows directly after the closing quote character.
    para2 = '"Not now," Lyria turned to leave.'
    paragraphs2 = [para2]
    offsets2 = _para_offsets(paragraphs2)
    quote_text2 = '"Not now,"'
    q_start2 = para2.index(quote_text2)
    quotes2 = [_make_quote(0, quote_text2, 0, offsets2[0] + q_start2)]
    alias_to_slug = {"lyria": "lyria_of_vox"}

    result = annotate_chapter(paragraphs2, quotes2, alias_to_slug, {})
    assert 'guess="lyria_of_vox"' in result


# ---------------------------------------------------------------------------
# Test 8 — Multiple paragraphs: no-quotes, quotes, no-quotes
# ---------------------------------------------------------------------------


def test_multiple_paragraphs():
    paragraphs = [
        "The hall was dark.",
        '"Who goes there?" the guard demanded.',
        "Silence followed.",
    ]
    offsets = _para_offsets(paragraphs)
    quote_text = '"Who goes there?"'
    q_start = paragraphs[1].index(quote_text)
    quotes = [_make_quote(0, quote_text, 1, offsets[1] + q_start)]
    result = annotate_chapter(paragraphs, quotes, {}, {})
    segments = result.split("\n\n")
    assert segments[0] == "[NARRATOR] The hall was dark."
    assert any("[QUOTE:0]" in s for s in segments)
    assert segments[-1] == "[NARRATOR] Silence followed."


# ---------------------------------------------------------------------------
# Test 9 — Empty paragraph list → empty string
# ---------------------------------------------------------------------------


def test_empty_paragraphs():
    result = annotate_chapter([], [], {}, {})
    assert result == ""


# ---------------------------------------------------------------------------
# Test 10 — Pronoun suppressed when no attribution verb nearby
# ---------------------------------------------------------------------------


def test_pronoun_suppressed_without_verb():
    # "he" appears in the narration after the quote, but there is no
    # attribution verb (said/asked/etc.) within ±60 chars — pronoun= must
    # NOT appear in the tag.
    para = '"He went home." The cat sat on the mat.'
    paragraphs = [para]
    offsets = _para_offsets(paragraphs)
    quote_text = '"He went home."'
    q_start = para.index(quote_text)
    quotes = [_make_quote(0, quote_text, 0, offsets[0] + q_start)]
    result = annotate_chapter(paragraphs, quotes, {}, {})
    assert "pronoun=" not in result
