"""Tests for strip_scare_quotes() in kenkui.nlp.quotes."""

import pytest
from kenkui.nlp.quotes import strip_scare_quotes


def _strip(text: str) -> str:
    """Helper: run strip_scare_quotes on a single paragraph."""
    return strip_scare_quotes([text])[0]


def test_called_label_strips_marks():
    para = 'They called "the lazy approach" a success.'
    result = _strip(para)
    assert '"' not in result
    assert '\u201c' not in result
    assert 'the lazy approach' in result


def test_so_called_label_strips_marks():
    para = 'This is so-called "the lazy approach" in action.'
    result = _strip(para)
    assert 'the lazy approach' in result
    assert '"the lazy approach"' not in result


def test_acronym_unesco_strips_marks():
    para = 'The organization "UNESCO" was founded in 1945.'
    result = _strip(para)
    assert 'UNESCO' in result
    assert '"UNESCO"' not in result


def test_acronym_nato_strips_marks():
    para = 'She mentioned that "NATO" had expanded.'
    result = _strip(para)
    assert 'NATO' in result
    assert '"NATO"' not in result


def test_real_dialogue_not_stripped():
    para = '"Hello," he said with a smile.'
    result = _strip(para)
    assert result == para


def test_no_quotes_unchanged():
    para = 'A paragraph with no quotes at all.'
    result = _strip(para)
    assert result == para


def test_scare_quote_nested_inside_dialogue():
    para = '"We gave it what you\'d call a \u201csoft landing,\u201d" she said.'
    result = _strip(para)
    # Inner scare quote should be stripped (preceded by "call a")
    assert 'soft landing,' in result
    assert '\u201csoft landing,\u201d' not in result
    # Outer dialogue quote marks should remain
    assert result.startswith('\u201c') or result.startswith('"')


def test_known_as_label_strips_marks():
    para = 'He was known as "the Reaper" in the underworld.'
    result = _strip(para)
    assert 'the Reaper' in result
    assert '"the Reaper"' not in result


def test_titled_label_strips_marks():
    para = 'The book is titled "War and Peace" and spans many volumes.'
    result = _strip(para)
    assert 'War and Peace' in result
    assert '"War and Peace"' not in result


def test_return_list_same_length():
    paragraphs = [
        'They called "the lazy approach" best.',
        '"Hello," he said.',
        'No quotes here.',
    ]
    result = strip_scare_quotes(paragraphs)
    assert len(result) == len(paragraphs)


def test_curly_quote_acronym_stripped():
    para = 'The agency known as \u201cCIA\u201d denied involvement.'
    result = _strip(para)
    assert 'CIA' in result
    assert '\u201cCIA\u201d' not in result


def test_two_acronyms_same_paragraph_both_stripped():
    """Two qualifying acronym spans in one paragraph must both be stripped
    without corrupting the text (guards against C1: shared close-pos bug)."""
    para = 'Both "UNESCO" and "NATO" were mentioned.'
    result = _strip(para)
    assert 'UNESCO' in result
    assert 'NATO' in result
    assert '"UNESCO"' not in result
    assert '"NATO"' not in result
    # The non-quote content must survive intact.
    assert 'Both ' in result
    assert ' and ' in result
    assert ' were mentioned.' in result
