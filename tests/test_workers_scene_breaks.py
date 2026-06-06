"""Tests for scene-break detection in workers and EPUB reader."""
from __future__ import annotations

import pytest

# ---------------------------------------------------------------------------
# TestIsSceneBreak
# ---------------------------------------------------------------------------

class TestIsSceneBreak:
    """Unit tests for workers._is_scene_break()."""

    @pytest.mark.parametrize("text", [
        "",           # empty string
        "   ",        # whitespace only
        "\t\n",       # tabs/newlines
        "* * *",      # three asterisks
        "* *",        # two asterisks
        "***",        # no spaces
        "**",         # two together
        "---",        # three dashes
        "--",         # two dashes
        "\u2014\u2014",  # two em-dashes
        "\u2014\u2014\u2014",  # three em-dashes
        "#",          # single hash
        "  #  ",      # hash with surrounding spaces
        "  * * *  ",  # asterisks with surrounding spaces
    ])
    def test_is_scene_break_true(self, text):
        from kenkui.workers import _is_scene_break
        assert _is_scene_break(text) is True

    @pytest.mark.parametrize("text", [
        "This is a paragraph.",
        "Chapter 1",
        "She said goodbye.",
        "* This is NOT a break",   # asterisk at start of sentence
        "#Title",                   # hash immediately followed by text
        "Some --- text",            # dashes mid-sentence
        "Hello World",
        "a",
    ])
    def test_is_scene_break_false(self, text):
        from kenkui.workers import _is_scene_break
        assert _is_scene_break(text) is False


class TestSharedSceneBreakRules:
    def test_shared_helper_matches_worker_helper(self):
        from kenkui.text_rules import is_scene_break
        from kenkui.workers import _is_scene_break

        samples = ["", "* * *", "---", "#", "Chapter 1", "Some --- text"]
        assert [is_scene_break(s) for s in samples] == [_is_scene_break(s) for s in samples]


# ---------------------------------------------------------------------------
# TestSplitAtSceneBreaks
# ---------------------------------------------------------------------------

class TestSplitAtSceneBreaks:
    """Unit tests for workers._split_at_scene_breaks()."""

    def test_empty_string_sentinels_split_into_groups(self):
        from kenkui.workers import _split_at_scene_breaks
        paras = ["Para 1.", "", "Para 2.", "", "Para 3."]
        groups = _split_at_scene_breaks(paras)
        assert len(groups) == 3
        assert groups[0] == ["Para 1."]
        assert groups[1] == ["Para 2."]
        assert groups[2] == ["Para 3."]

    def test_asterisk_break_splits_two_groups(self):
        from kenkui.workers import _split_at_scene_breaks
        paras = ["A.", "* * *", "B."]
        groups = _split_at_scene_breaks(paras)
        assert len(groups) == 2
        assert groups[0] == ["A."]
        assert groups[1] == ["B."]

    def test_no_breaks_returns_single_group(self):
        from kenkui.workers import _split_at_scene_breaks
        paras = ["A.", "B.", "C."]
        groups = _split_at_scene_breaks(paras)
        assert len(groups) == 1
        assert groups[0] == ["A.", "B.", "C."]

    def test_leading_break_produces_no_empty_first_group(self):
        from kenkui.workers import _split_at_scene_breaks
        paras = ["", "A.", "B."]
        groups = _split_at_scene_breaks(paras)
        assert len(groups) == 1
        assert groups[0] == ["A.", "B."]

    def test_trailing_break_produces_no_empty_last_group(self):
        from kenkui.workers import _split_at_scene_breaks
        paras = ["A.", "B.", ""]
        groups = _split_at_scene_breaks(paras)
        assert len(groups) == 1
        assert groups[0] == ["A.", "B."]

    def test_consecutive_breaks_collapse_to_single_split(self):
        from kenkui.workers import _split_at_scene_breaks
        paras = ["A.", "", "", "B."]
        groups = _split_at_scene_breaks(paras)
        assert len(groups) == 2

    def test_empty_input_returns_single_empty_group(self):
        from kenkui.workers import _split_at_scene_breaks
        groups = _split_at_scene_breaks([])
        assert groups == [[]]

    def test_all_breaks_returns_single_empty_group(self):
        from kenkui.workers import _split_at_scene_breaks
        groups = _split_at_scene_breaks(["", "* * *", ""])
        assert groups == [[]]

    def test_break_markers_not_included_in_groups(self):
        from kenkui.workers import _split_at_scene_breaks
        paras = ["A.", "---", "B."]
        groups = _split_at_scene_breaks(paras)
        for group in groups:
            assert "---" not in group


# ---------------------------------------------------------------------------
# TestEpubLinespacePreservation
# ---------------------------------------------------------------------------

# Long enough paragraph text so that Strategy 3 (< 100 chars) doesn't trigger
# and overwrite the linespace sentinels Strategy 2 correctly placed.
_LONG_FIRST = "This is the first paragraph of the section, containing enough text to exceed the threshold."
_LONG_SECOND = "This is the second paragraph after the scene break, also long enough to exceed the threshold."
_LONG_THIRD = "This is the third paragraph in yet another scene, still long enough to stay above the limit."


class TestEpubLinespacePreservation:
    """Verify that epub.py preserves linespace paragraphs as scene-break sentinels."""

    def _make_soup(self, html: str):
        from bs4 import BeautifulSoup
        return BeautifulSoup(html, "html.parser")

    def _get_reader(self):
        from kenkui.readers.epub import EpubReader
        reader = EpubReader.__new__(EpubReader)
        return reader

    def test_linespace_lowercase_produces_empty_sentinel(self):
        html = f"""
        <html><body>
          <p>{_LONG_FIRST}</p>
          <p class="linespace">&#160;</p>
          <p>{_LONG_SECOND}</p>
        </body></html>
        """
        soup = self._make_soup(html)
        reader = self._get_reader()
        paras = reader._extract_chapter_paragraphs(soup)
        assert "" in paras, f"Expected empty sentinel in: {paras}"

    def test_linespace_uppercase_produces_empty_sentinel(self):
        html = f"""
        <html><body>
          <p>{_LONG_FIRST}</p>
          <p class="LINESPACE">&#160;</p>
          <p>{_LONG_SECOND}</p>
        </body></html>
        """
        soup = self._make_soup(html)
        reader = self._get_reader()
        paras = reader._extract_chapter_paragraphs(soup)
        assert "" in paras, f"Expected empty sentinel in: {paras}"

    def test_linespace_splits_into_two_scene_groups(self):
        from kenkui.workers import _split_at_scene_breaks
        html = f"""
        <html><body>
          <p>{_LONG_FIRST}</p>
          <p class="linespace">&#160;</p>
          <p>{_LONG_SECOND}</p>
        </body></html>
        """
        soup = self._make_soup(html)
        reader = self._get_reader()
        paras = reader._extract_chapter_paragraphs(soup)
        groups = _split_at_scene_breaks(paras)
        assert len(groups) == 2, f"Expected 2 groups, got {len(groups)}: {groups}"
        # First group contains first paragraph
        assert any("first paragraph" in p for p in groups[0])
        # Second group contains second paragraph
        assert any("second paragraph" in p for p in groups[1])

    def test_multiple_linespaces_produce_multiple_splits(self):
        from kenkui.workers import _split_at_scene_breaks
        html = f"""
        <html><body>
          <p>{_LONG_FIRST}</p>
          <p class="linespace">&#160;</p>
          <p>{_LONG_SECOND}</p>
          <p class="linespace">&#160;</p>
          <p>{_LONG_THIRD}</p>
        </body></html>
        """
        soup = self._make_soup(html)
        reader = self._get_reader()
        paras = reader._extract_chapter_paragraphs(soup)
        groups = _split_at_scene_breaks(paras)
        assert len(groups) == 3, f"Expected 3 groups, got {len(groups)}"

    def test_normal_paragraphs_not_treated_as_breaks(self):
        html = f"""
        <html><body>
          <p>{_LONG_FIRST}</p>
          <p>{_LONG_SECOND}</p>
          <p>{_LONG_THIRD}</p>
        </body></html>
        """
        soup = self._make_soup(html)
        reader = self._get_reader()
        paras = reader._extract_chapter_paragraphs(soup)
        assert "" not in paras
