"""Tests for kenkui.booknlp_processor.

Covers:
- BOOKNLP_AVAILABLE flag
- Cache key generation (_book_hash)
- Cache read/write round-trip
- Narrator paragraph merging (_assemble_segments)
- Token→paragraph mapping (_find_para_by_byte)
- Segment ordering correctness
- _parse_quotes_file and _parse_tokens_file
- _load_annotated_chapters helper in parsing.py
- AnnotatedChaptersCacheMissError when cache file is missing
"""

from __future__ import annotations

import json
import textwrap
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from kenkui.booknlp_processor import (
    BOOKNLP_AVAILABLE,
    BookNLPResult,
    _assemble_segments,
    _book_hash,
    _build_para_to_quote,
    _build_token_to_para_map,
    _extract_characters,
    _find_para_by_byte,
    _parse_quotes_file,
    _parse_tokens_file,
    _ParaLocation,
    _write_input_file,
    cache_result,
    ensure_spacy_model,
    get_cached_result,
)
from kenkui.models import Chapter, CharacterInfo, Segment


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_chapter(index: int, paragraphs: list[str]) -> Chapter:
    return Chapter(index=index, title=f"Chapter {index}", paragraphs=paragraphs)


# ---------------------------------------------------------------------------
# BOOKNLP_AVAILABLE flag
# ---------------------------------------------------------------------------


class TestBookNLPAvailableFlag:
    """Tests for BOOKNLP_AVAILABLE() and reset_booknlp_check().

    The function has two gates:
      1. ``booknlp`` package must be importable.
      2. ``en_core_web_sm`` spaCy model must be present on disk.

    Both must be true for the function to return True.  The result is cached;
    reset_booknlp_check() clears the cache for a fresh re-evaluation.
    """

    def setup_method(self):
        """Each test gets a clean cache."""
        from kenkui.booknlp_processor import reset_booknlp_check

        reset_booknlp_check()

    def teardown_method(self):
        from kenkui.booknlp_processor import reset_booknlp_check

        reset_booknlp_check()

    def test_flag_returns_bool(self):
        assert isinstance(BOOKNLP_AVAILABLE(), bool)

    def test_returns_false_when_booknlp_package_missing(self):
        """If the booknlp package cannot be imported, return False."""
        import kenkui.booknlp_processor as bmod

        original = bmod._booknlp_available
        bmod._booknlp_available = None
        try:
            with patch.dict("sys.modules", {"booknlp": None, "booknlp.booknlp": None}):
                result = BOOKNLP_AVAILABLE()
            assert result is False
        finally:
            bmod._booknlp_available = original

    def _fake_modules(self, model_present: bool) -> tuple:
        """Return (fake_booknlp_module, fake_spacy_module) for sys.modules injection.

        Both spacy and booknlp may or may not be installed in the test
        runner's environment.  We stub them out entirely in sys.modules so
        the tests are hermetic and never depend on the real packages.
        """
        import types

        fake_booknlp = types.ModuleType("booknlp.booknlp")
        fake_booknlp.BookNLP = MagicMock()  # type: ignore[attr-defined]

        fake_util = MagicMock()
        fake_util.is_package.return_value = model_present
        fake_spacy = types.ModuleType("spacy")
        fake_spacy.util = fake_util  # type: ignore[attr-defined]

        return fake_booknlp, fake_spacy

    def test_returns_false_when_spacy_model_missing(self):
        """Package importable but en_core_web_sm absent → return False.

        This is the bug case: booknlp imports fine but BookNLP.__init__ will
        crash with OSError when spacy.load('en_core_web_sm') is called.
        BOOKNLP_AVAILABLE() must catch this before routing the user to
        BookNLPAnalysisScreen.
        """
        import kenkui.booknlp_processor as bmod

        bmod._booknlp_available = None

        fake_booknlp, fake_spacy = self._fake_modules(model_present=False)
        with patch.dict("sys.modules", {"booknlp.booknlp": fake_booknlp, "spacy": fake_spacy}):
            result = BOOKNLP_AVAILABLE()

        assert result is False

    def test_returns_true_when_package_and_model_present(self):
        """Both package importable and model present → return True."""
        import kenkui.booknlp_processor as bmod

        bmod._booknlp_available = None

        fake_booknlp, fake_spacy = self._fake_modules(model_present=True)
        with patch.dict("sys.modules", {"booknlp.booknlp": fake_booknlp, "spacy": fake_spacy}):
            result = BOOKNLP_AVAILABLE()

        assert result is True

    def test_result_is_cached(self):
        """A second call should not re-run the check."""
        import kenkui.booknlp_processor as bmod

        bmod._booknlp_available = None

        fake_booknlp, fake_spacy = self._fake_modules(model_present=True)
        with patch.dict("sys.modules", {"booknlp.booknlp": fake_booknlp, "spacy": fake_spacy}):
            BOOKNLP_AVAILABLE()
            BOOKNLP_AVAILABLE()
            BOOKNLP_AVAILABLE()

        # is_package is on fake_spacy.util — check it was only called once
        assert fake_spacy.util.is_package.call_count == 1, (
            "is_package should only be called once (result is cached)"
        )

    def test_reset_clears_cache_and_allows_recheck(self):
        """reset_booknlp_check() causes the next call to re-evaluate."""
        from kenkui.booknlp_processor import reset_booknlp_check

        import kenkui.booknlp_processor as bmod

        bmod._booknlp_available = None

        fake_booknlp, fake_spacy_false = self._fake_modules(model_present=False)
        _, fake_spacy_true = self._fake_modules(model_present=True)

        # First call: model absent → False
        with patch.dict(
            "sys.modules", {"booknlp.booknlp": fake_booknlp, "spacy": fake_spacy_false}
        ):
            assert BOOKNLP_AVAILABLE() is False

        # Reset + re-check with model now present → True
        reset_booknlp_check()
        with patch.dict("sys.modules", {"booknlp.booknlp": fake_booknlp, "spacy": fake_spacy_true}):
            assert BOOKNLP_AVAILABLE() is True

    def test_run_analysis_raises_if_unavailable(self):
        """If booknlp is not installed, run_analysis must raise RuntimeError."""
        with patch("kenkui.booknlp_processor.BOOKNLP_AVAILABLE", return_value=False):
            from kenkui.booknlp_processor import run_analysis

            with pytest.raises(RuntimeError, match="not available"):
                run_analysis([], Path("/fake/book.epub"))


# ---------------------------------------------------------------------------
# ensure_spacy_model
# ---------------------------------------------------------------------------


class TestEnsureSpacyModel:
    """Tests for ensure_spacy_model().

    ensure_spacy_model() no longer uses spacy.cli.download (which calls
    sys.exit() on failure and silently kills background threads).  Instead it
    invokes the installer directly via subprocess.run, trying uv then pip.
    These tests mock subprocess.run and spacy.util.is_package.
    """

    def _make_spacy_mock(self, model_present: bool):
        """Return a fake spacy module whose util.is_package matches model_present."""
        import types

        call_count = {"n": 0}

        def is_package(name):
            call_count["n"] += 1
            if model_present:
                return True
            # First call is the pre-install check; subsequent calls simulate
            # the state after a successful installation.
            return call_count["n"] > 1

        spacy_mock = MagicMock()
        spacy_mock.util = types.SimpleNamespace(is_package=is_package)
        return spacy_mock

    def _make_subprocess_success(self):
        """Return a subprocess.run mock that always reports success."""
        proc = MagicMock()
        proc.returncode = 0
        proc.stdout = ""
        proc.stderr = ""
        return MagicMock(return_value=proc)

    def _make_subprocess_fail(self, error_msg: str = "install failed"):
        """Return a subprocess.run mock that always reports failure."""
        proc = MagicMock()
        proc.returncode = 1
        proc.stdout = error_msg
        proc.stderr = error_msg
        return MagicMock(return_value=proc)

    def test_no_op_when_model_already_present(self):
        """If spaCy reports the model is installed, no installer is invoked."""
        spacy_mock = self._make_spacy_mock(model_present=True)
        subprocess_mock = self._make_subprocess_success()
        with (
            patch.dict("sys.modules", {"spacy": spacy_mock}),
            patch("kenkui.booknlp_processor.subprocess") as sp,
        ):
            ensure_spacy_model()
            sp.run.assert_not_called()

    def test_installer_invoked_when_model_missing(self):
        """If the model is absent, subprocess.run is called to install it."""
        spacy_mock = self._make_spacy_mock(model_present=False)
        subprocess_mock = self._make_subprocess_success()
        with (
            patch.dict("sys.modules", {"spacy": spacy_mock}),
            patch("kenkui.booknlp_processor.subprocess.run", subprocess_mock),
        ):
            ensure_spacy_model()
        subprocess_mock.assert_called()
        # At least one call should contain the model wheel URL
        all_args = [str(call) for call in subprocess_mock.call_args_list]
        assert any("en_core_web_sm" in a for a in all_args)

    def test_progress_callback_called_during_download(self):
        """Status messages containing the model name are forwarded to the callback."""
        spacy_mock = self._make_spacy_mock(model_present=False)
        subprocess_mock = self._make_subprocess_success()
        messages: list[str] = []
        with (
            patch.dict("sys.modules", {"spacy": spacy_mock}),
            patch("kenkui.booknlp_processor.subprocess.run", subprocess_mock),
        ):
            ensure_spacy_model(progress_callback=messages.append)
        assert any("en_core_web_sm" in m for m in messages)

    def test_raises_runtime_error_if_spacy_missing(self):
        """If spaCy itself is not installed, a clear RuntimeError is raised."""
        with patch.dict("sys.modules", {"spacy": None}):
            with pytest.raises((RuntimeError, ImportError)):
                ensure_spacy_model()

    def test_uv_tried_first_then_pip_fallback(self):
        """uv is tried first; if uv is not found, pip is tried next."""
        spacy_mock = self._make_spacy_mock(model_present=False)

        # uv raises FileNotFoundError (not installed), pip succeeds
        def fake_run(cmd, **kwargs):
            if cmd[0] == "uv":
                raise FileNotFoundError("uv not found")
            proc = MagicMock()
            proc.returncode = 0
            proc.stdout = ""
            proc.stderr = ""
            return proc

        with (
            patch.dict("sys.modules", {"spacy": spacy_mock}),
            patch("kenkui.booknlp_processor.subprocess.run", side_effect=fake_run),
        ):
            ensure_spacy_model()  # should succeed via pip fallback

    def test_raises_runtime_error_when_all_installers_fail(self):
        """RuntimeError is raised (not SystemExit) when every installer fails.

        This is the core contract: ensure_spacy_model() must NEVER call
        sys.exit() — it must always raise RuntimeError so the calling thread
        can catch it and surface it as a UI error.
        """
        spacy_mock = self._make_spacy_mock(model_present=False)
        subprocess_mock = self._make_subprocess_fail("network error")
        with (
            patch.dict("sys.modules", {"spacy": spacy_mock}),
            patch("kenkui.booknlp_processor.subprocess.run", subprocess_mock),
        ):
            with pytest.raises(RuntimeError):
                ensure_spacy_model()

    def test_no_system_exit_on_pip_failure(self):
        """SystemExit must never propagate out of ensure_spacy_model().

        In a uv-managed environment spacy.cli.run_command calls sys.exit()
        when pip is absent.  Our implementation bypasses spacy.cli entirely,
        so SystemExit should never escape — but we test the boundary explicitly.
        """
        spacy_mock = self._make_spacy_mock(model_present=False)

        def fake_run(cmd, **kwargs):
            # Simulate the old spacy.cli behaviour
            raise SystemExit(1)

        with (
            patch.dict("sys.modules", {"spacy": spacy_mock}),
            patch("kenkui.booknlp_processor.subprocess.run", side_effect=fake_run),
        ):
            # SystemExit must NOT propagate — should be caught and re-raised as RuntimeError
            # (The for/else in ensure_spacy_model treats FileNotFoundError as "try next";
            #  SystemExit from subprocess.run itself would propagate because it is not
            #  FileNotFoundError — but subprocess.run itself will NOT raise SystemExit;
            #  that was the old spacy.cli path which we no longer use.
            #  This test documents the expected boundary.)
            try:
                ensure_spacy_model()
            except SystemExit:
                pytest.fail(
                    "ensure_spacy_model() let SystemExit propagate — "
                    "this would silently kill the background download thread"
                )
            except RuntimeError:
                pass  # correct: failure is surfaced as RuntimeError


# ---------------------------------------------------------------------------
# Cache key
# ---------------------------------------------------------------------------


class TestBookHash:
    def test_hash_is_hex_string(self):
        h = _book_hash(Path(__file__))
        assert isinstance(h, str)
        assert len(h) == 16
        assert all(c in "0123456789abcdef" for c in h)

    def test_different_paths_give_different_hashes(self):
        h1 = _book_hash(Path("/a/book.epub"))
        h2 = _book_hash(Path("/b/book.epub"))
        assert h1 != h2

    def test_nonexistent_path_does_not_crash(self):
        h = _book_hash(Path("/nonexistent/path/book.epub"))
        assert isinstance(h, str)


# ---------------------------------------------------------------------------
# Cache read / write round-trip
# ---------------------------------------------------------------------------


class TestCacheRoundTrip:
    def test_cache_write_then_read(self, tmp_path):
        """cache_result followed by get_cached_result returns an equivalent result."""
        with patch("kenkui.booknlp_processor.CACHE_DIR", tmp_path):
            book_path = tmp_path / "book.epub"
            book_path.write_text("fake epub")

            chars = [
                CharacterInfo("ALICE-0", "Alice", quote_count=10, gender_pronoun="she"),
            ]
            ch = _make_chapter(0, ["Para 1.", "Para 2."])
            ch.segments = [
                Segment("Para 1.", "NARRATOR", 0),
                Segment("Para 2.", "ALICE-0", 1),
            ]
            result = BookNLPResult(
                characters=chars,
                chapters=[ch],
                book_hash=_book_hash(book_path),
            )

            cache_path = cache_result(result, book_path)
            assert cache_path.exists()

            loaded = get_cached_result(book_path)
            assert loaded is not None
            assert len(loaded.characters) == 1
            assert loaded.characters[0].display_name == "Alice"
            assert len(loaded.chapters) == 1
            assert loaded.chapters[0].segments is not None
            assert len(loaded.chapters[0].segments) == 2

    def test_get_cached_result_returns_none_if_missing(self, tmp_path):
        with patch("kenkui.booknlp_processor.CACHE_DIR", tmp_path):
            book_path = tmp_path / "missing.epub"
            assert get_cached_result(book_path) is None

    def test_cache_overwrite(self, tmp_path):
        """Writing a second result for the same book overwrites the first."""
        with patch("kenkui.booknlp_processor.CACHE_DIR", tmp_path):
            book_path = tmp_path / "book.epub"
            book_path.write_text("fake epub")

            h = _book_hash(book_path)
            r1 = BookNLPResult(
                characters=[CharacterInfo("A-0", "Alice", 5)],
                chapters=[],
                book_hash=h,
            )
            r2 = BookNLPResult(
                characters=[
                    CharacterInfo("A-0", "Alice", 5),
                    CharacterInfo("B-0", "Bob", 3),
                ],
                chapters=[],
                book_hash=h,
            )
            cache_result(r1, book_path)
            cache_result(r2, book_path)

            loaded = get_cached_result(book_path)
            assert loaded is not None
            assert len(loaded.characters) == 2


# ---------------------------------------------------------------------------
# _find_para_by_byte
# ---------------------------------------------------------------------------


class TestFindParaByByte:
    def _make_locations(self):
        return [
            _ParaLocation(chapter_index=0, para_index=0, byte_start=0, byte_end=10),
            _ParaLocation(chapter_index=0, para_index=1, byte_start=12, byte_end=25),
            _ParaLocation(chapter_index=1, para_index=0, byte_start=27, byte_end=40),
        ]

    def test_finds_first_paragraph(self):
        locs = self._make_locations()
        result = _find_para_by_byte(locs, 5)
        assert result is not None
        assert result.para_index == 0

    def test_finds_second_paragraph(self):
        locs = self._make_locations()
        result = _find_para_by_byte(locs, 15)
        assert result is not None
        assert result.para_index == 1
        assert result.chapter_index == 0

    def test_finds_chapter_boundary(self):
        locs = self._make_locations()
        result = _find_para_by_byte(locs, 30)
        assert result is not None
        assert result.chapter_index == 1

    def test_gap_between_paragraphs_returns_none(self):
        locs = self._make_locations()
        result = _find_para_by_byte(locs, 11)  # gap between 10 and 12
        assert result is None

    def test_empty_locations(self):
        assert _find_para_by_byte([], 0) is None


# ---------------------------------------------------------------------------
# _parse_quotes_file
# ---------------------------------------------------------------------------


class TestParseQuotesFile:
    def test_parses_valid_tsv(self, tmp_path):
        content = textwrap.dedent("""\
            quote_start\tquote_end\tmention_start\tmention_end\tmention_text\tchar_id\tquote_text
            0\t5\t6\t7\tshe\tELIZABETH-0\tIt is a truth universally acknowledged.
            10\t15\t16\t17\the\tDARCY-1\tYou must allow me to tell you...
        """)
        f = tmp_path / "book.quotes"
        f.write_text(content)
        quotes = _parse_quotes_file(f)
        assert len(quotes) == 2
        assert quotes[0]["char_id"] == "ELIZABETH-0"
        assert quotes[1]["quote_start"] == 10

    def test_missing_file_returns_empty(self, tmp_path):
        result = _parse_quotes_file(tmp_path / "missing.quotes")
        assert result == []

    def test_skips_malformed_lines(self, tmp_path):
        content = "header\nonly_one_column\n"
        f = tmp_path / "book.quotes"
        f.write_text(content)
        result = _parse_quotes_file(f)
        assert result == []


# ---------------------------------------------------------------------------
# _parse_tokens_file
# ---------------------------------------------------------------------------


class TestParseTokensFile:
    def test_parses_valid_tsv(self, tmp_path):
        content = textwrap.dedent("""\
            para_ID\tsentence_ID\ttoken_ID_sentence\ttoken_ID_doc\tword\tlemma\tbyte_onset\tbyte_offset\tPOS\tdep\tdep_head\tevent
            0\t0\t0\t0\tIt\tit\t0\t2\tPRP\tnsubj\t1\tO
            0\t0\t1\t1\tis\tbe\t3\t5\tVBZ\tROOT\t1\tO
        """)
        f = tmp_path / "book.tokens"
        f.write_text(content)
        tokens = _parse_tokens_file(f)
        assert len(tokens) == 2
        assert tokens[0]["token_id"] == 0
        assert tokens[0]["byte_onset"] == 0
        assert tokens[1]["byte_onset"] == 3

    def test_missing_file_returns_empty(self, tmp_path):
        result = _parse_tokens_file(tmp_path / "missing.tokens")
        assert result == []


# ---------------------------------------------------------------------------
# _write_input_file
# ---------------------------------------------------------------------------


class TestWriteInputFile:
    def test_writes_paragraphs(self, tmp_path):
        chapters = [
            _make_chapter(0, ["Hello world.", "Second paragraph."]),
            _make_chapter(1, ["Chapter two first para."]),
        ]
        out = tmp_path / "book.txt"
        locations = _write_input_file(chapters, out)
        text = out.read_text(encoding="utf-8")
        assert "Hello world." in text
        assert "Chapter two first para." in text
        assert len(locations) == 3  # 2 paras in ch0 + 1 para in ch1

    def test_location_chapter_indices(self, tmp_path):
        chapters = [
            _make_chapter(0, ["Para A."]),
            _make_chapter(5, ["Para B."]),
        ]
        out = tmp_path / "book.txt"
        locations = _write_input_file(chapters, out)
        assert locations[0].chapter_index == 0
        assert locations[1].chapter_index == 5


# ---------------------------------------------------------------------------
# _assemble_segments — narrator merging
# ---------------------------------------------------------------------------


class TestAssembleSegments:
    def test_all_narrator_when_no_quotes(self):
        ch = _make_chapter(0, ["Para 1.", "Para 2.", "Para 3."])
        segs = _assemble_segments(ch, [], {}, {})
        assert len(segs) == 1
        assert segs[0].speaker == "NARRATOR"
        assert "Para 1." in segs[0].text
        assert "Para 3." in segs[0].text

    def test_dialogue_para_becomes_character_segment(self):
        ch = _make_chapter(0, ["Narrator text.", "She said hello.", "More narration."])
        # Para 1 is attributed to ALICE-0
        para_to_quote = {1: "ALICE-0"}
        segs = _assemble_segments(ch, [], {}, para_to_quote)
        speakers = [s.speaker for s in segs]
        assert "ALICE-0" in speakers
        assert "NARRATOR" in speakers

    def test_consecutive_narrator_paras_merged(self):
        ch = _make_chapter(0, ["N1.", "N2.", "N3.", "Dialogue.", "N4.", "N5."])
        para_to_quote = {3: "BOB-0"}
        segs = _assemble_segments(ch, [], {}, para_to_quote)
        narrator_segs = [s for s in segs if s.speaker == "NARRATOR"]
        # N1-N3 merged into one, N4-N5 merged into another
        assert len(narrator_segs) == 2
        assert "N1." in narrator_segs[0].text
        assert "N4." in narrator_segs[1].text

    def test_segment_indices_are_sequential(self):
        ch = _make_chapter(0, ["N.", "D.", "N2."])
        para_to_quote = {1: "ALICE-0"}
        segs = _assemble_segments(ch, [], {}, para_to_quote)
        indices = [s.index for s in segs]
        assert indices == sorted(indices)

    def test_empty_chapter_returns_empty(self):
        ch = _make_chapter(0, [])
        segs = _assemble_segments(ch, [], {}, {})
        assert segs == []


# ---------------------------------------------------------------------------
# _load_annotated_chapters and AnnotatedChaptersCacheMissError (parsing.py)
# ---------------------------------------------------------------------------


class TestLoadAnnotatedChapters:
    def test_loads_chapters_from_json(self, tmp_path):
        from kenkui.parsing import _load_annotated_chapters

        cache_data = {
            "book_hash": "abc123",
            "chapters": [
                {
                    "index": 0,
                    "title": "Ch 0",
                    "paragraphs": ["p1", "p2"],
                    "segments": [
                        {"text": "p1", "speaker": "NARRATOR", "index": 0},
                    ],
                },
                {
                    "index": 1,
                    "title": "Ch 1",
                    "paragraphs": ["p3"],
                },
            ],
        }
        cache_file = tmp_path / "cache.json"
        cache_file.write_text(json.dumps(cache_data), encoding="utf-8")

        chapters = _load_annotated_chapters(cache_file, included_indices=[])
        assert len(chapters) == 2
        assert chapters[0].segments is not None
        assert chapters[1].segments is None

    def test_filters_by_included_indices(self, tmp_path):
        from kenkui.parsing import _load_annotated_chapters

        cache_data = {
            "book_hash": "xyz",
            "chapters": [
                {"index": 0, "title": "Ch 0", "paragraphs": ["p0"]},
                {"index": 1, "title": "Ch 1", "paragraphs": ["p1"]},
                {"index": 2, "title": "Ch 2", "paragraphs": ["p2"]},
            ],
        }
        cache_file = tmp_path / "cache.json"
        cache_file.write_text(json.dumps(cache_data), encoding="utf-8")

        chapters = _load_annotated_chapters(cache_file, included_indices=[0, 2])
        assert len(chapters) == 2
        assert chapters[0].index == 0
        assert chapters[1].index == 2

    def test_raises_cache_miss_if_file_missing(self, tmp_path):
        from kenkui.parsing import (
            AnnotatedChaptersCacheMissError,
            _load_annotated_chapters,
        )

        missing = tmp_path / "nonexistent.json"
        with pytest.raises(AnnotatedChaptersCacheMissError, match="CACHE_MISS"):
            _load_annotated_chapters(missing, included_indices=[])

    def test_sorted_by_chapter_index(self, tmp_path):
        from kenkui.parsing import _load_annotated_chapters

        cache_data = {
            "book_hash": "order",
            "chapters": [
                {"index": 3, "title": "Ch 3", "paragraphs": []},
                {"index": 1, "title": "Ch 1", "paragraphs": []},
                {"index": 2, "title": "Ch 2", "paragraphs": []},
            ],
        }
        cache_file = tmp_path / "cache.json"
        cache_file.write_text(json.dumps(cache_data), encoding="utf-8")

        chapters = _load_annotated_chapters(cache_file, included_indices=[])
        assert [ch.index for ch in chapters] == [1, 2, 3]


# ---------------------------------------------------------------------------
# _extract_characters
# ---------------------------------------------------------------------------


class TestExtractCharacters:
    """Tests for _extract_characters() parsing the BookNLP .book JSON.

    Fixtures use the **real BookNLP 1.0.8 schema** (``mentions`` / ``count``)
    unless the test is specifically exercising backward-compatibility with the
    older hypothetical schema (``names`` / ``quoteCount``).

    Real 1.0.8 character schema::

        {
          "id": 13,
          "mentions": {
            "proper":  [{"n": "Jane", "c": 9}],
            "common":  [],
            "pronoun": [{"n": "her", "c": 8}, ...]
          },
          "count": 25,
          "g": {"argmax": "she/her", "inference": {...}}
        }
    """

    def _write_book_json(self, tmp_path, characters: list[dict]) -> Path:
        """Write a minimal .book JSON file."""
        path = tmp_path / "book.book"
        path.write_text(json.dumps({"characters": characters}), encoding="utf-8")
        return path

    # -- Real schema (BookNLP 1.0.8) ----------------------------------------

    def test_basic_character_extraction(self, tmp_path):
        """Standard case: real BookNLP 1.0.8 schema with mentions and count."""
        data = [
            {
                "id": 42,
                "mentions": {
                    "proper": [{"n": "Elizabeth", "c": 10}, {"n": "Eliza", "c": 3}],
                    "common": [],
                    "pronoun": [{"n": "she", "c": 8}],
                },
                "count": 50,
                "g": {"argmax": "she/her", "inference": {}},
            },
            {
                "id": 7,
                "mentions": {
                    "proper": [{"n": "Darcy", "c": 8}],
                    "common": [],
                    "pronoun": [{"n": "he", "c": 6}],
                },
                "count": 30,
                "g": {"argmax": "he/him/his", "inference": {}},
            },
        ]
        path = self._write_book_json(tmp_path, data)
        chars = _extract_characters(path)
        assert len(chars) == 2
        # Sorted by count descending
        assert chars[0].display_name == "Elizabeth"
        assert chars[0].quote_count == 50
        assert chars[0].gender_pronoun == "she/her"
        assert chars[0].character_id == "ELIZABETH-42"
        assert chars[1].display_name == "Darcy"
        assert chars[1].character_id == "DARCY-7"

    def test_picks_most_common_proper_name(self, tmp_path):
        """When multiple proper name forms exist, the one with highest 'c' wins."""
        data = [
            {
                "id": 1,
                "mentions": {
                    "proper": [{"n": "Liz", "c": 2}, {"n": "Elizabeth Bennet", "c": 20}],
                    "common": [],
                    "pronoun": [],
                },
                "count": 22,
            }
        ]
        path = self._write_book_json(tmp_path, data)
        chars = _extract_characters(path)
        assert chars[0].display_name == "Elizabeth Bennet"

    def test_falls_back_to_pronoun_when_no_proper_name(self, tmp_path):
        """No proper names → use the most common pronoun as display_name."""
        data = [
            {
                "id": 99,
                "mentions": {
                    "proper": [],
                    "common": [],
                    "pronoun": [{"n": "he", "c": 5}, {"n": "him", "c": 2}],
                },
                "count": 7,
            }
        ]
        path = self._write_book_json(tmp_path, data)
        chars = _extract_characters(path)
        assert chars[0].display_name == "he"

    def test_falls_back_to_char_id_when_no_names_or_pronouns(self, tmp_path):
        """No names and no pronouns → display_name falls back to char_id string."""
        data = [
            {
                "id": 99,
                "mentions": {"proper": [], "common": [], "pronoun": []},
                "count": 1,
            }
        ]
        path = self._write_book_json(tmp_path, data)
        chars = _extract_characters(path)
        assert chars[0].display_name == "99"
        assert chars[0].character_id == "99-99"

    def test_uses_count_as_prominence(self, tmp_path):
        """Real schema: 'count' field drives the sort order."""
        data = [
            {
                "id": 1,
                "mentions": {"proper": [{"n": "Minor", "c": 1}], "common": [], "pronoun": []},
                "count": 5,
            },
            {
                "id": 2,
                "mentions": {"proper": [{"n": "Major", "c": 1}], "common": [], "pronoun": []},
                "count": 100,
            },
            {
                "id": 3,
                "mentions": {"proper": [{"n": "Mid", "c": 1}], "common": [], "pronoun": []},
                "count": 20,
            },
        ]
        path = self._write_book_json(tmp_path, data)
        chars = _extract_characters(path)
        counts = [c.quote_count for c in chars]
        assert counts == sorted(counts, reverse=True)
        assert chars[0].display_name == "Major"

    def test_missing_count_defaults_to_zero(self, tmp_path):
        """Neither 'count' nor 'quoteCount' present → quote_count = 0."""
        data = [
            {
                "id": 1,
                "mentions": {"proper": [{"n": "Alice", "c": 1}], "common": [], "pronoun": []},
            }
        ]
        path = self._write_book_json(tmp_path, data)
        chars = _extract_characters(path)
        assert chars[0].quote_count == 0

    def test_missing_gender_defaults_to_empty(self, tmp_path):
        """Missing 'g' field → gender_pronoun = ''."""
        data = [
            {
                "id": 1,
                "mentions": {"proper": [{"n": "Pat", "c": 1}], "common": [], "pronoun": []},
                "count": 2,
            }
        ]
        path = self._write_book_json(tmp_path, data)
        chars = _extract_characters(path)
        assert chars[0].gender_pronoun == ""

    def test_null_gender_defaults_to_empty(self, tmp_path):
        """Explicit null 'g' → gender_pronoun = ''."""
        data = [
            {
                "id": 1,
                "mentions": {"proper": [{"n": "Alex", "c": 1}], "common": [], "pronoun": []},
                "count": 2,
                "g": None,
            }
        ]
        path = self._write_book_json(tmp_path, data)
        chars = _extract_characters(path)
        assert chars[0].gender_pronoun == ""

    def test_empty_characters_array(self, tmp_path):
        """An empty characters array returns an empty list."""
        path = self._write_book_json(tmp_path, [])
        assert _extract_characters(path) == []

    def test_missing_file_returns_empty(self, tmp_path):
        """A missing .book file returns an empty list without raising."""
        assert _extract_characters(tmp_path / "nonexistent.book") == []

    def test_malformed_json_returns_empty(self, tmp_path):
        """Corrupt JSON returns an empty list without raising."""
        path = tmp_path / "bad.book"
        path.write_text("{ not valid json", encoding="utf-8")
        assert _extract_characters(path) == []

    def test_character_id_format(self, tmp_path):
        """character_id should be DISPLAY_NAME_UPPERCASED-raw_id."""
        data = [
            {
                "id": 5,
                "mentions": {"proper": [{"n": "Mr Darcy", "c": 1}], "common": [], "pronoun": []},
                "count": 1,
            }
        ]
        path = self._write_book_json(tmp_path, data)
        chars = _extract_characters(path)
        # spaces replaced with underscores, uppercased
        assert chars[0].character_id == "MR_DARCY-5"

    # -- Backward-compat: legacy schema (names / quoteCount) ----------------

    def test_legacy_names_field_still_works(self, tmp_path):
        """Old 'names' field (flat list) is accepted as a fallback.

        This ensures data cached or generated by older code remains usable
        without requiring a re-analysis.
        """
        data = [
            {
                "id": 42,
                "names": [{"n": "Elizabeth", "c": 10}, {"n": "Eliza", "c": 3}],
                "quoteCount": 50,
                "g": {"argmax": "she"},
            }
        ]
        path = self._write_book_json(tmp_path, data)
        chars = _extract_characters(path)
        assert chars[0].display_name == "Elizabeth"
        assert chars[0].quote_count == 50
        assert chars[0].character_id == "ELIZABETH-42"

    def test_legacy_quote_count_field_preferred_over_count(self, tmp_path):
        """When both 'quoteCount' (old) and 'count' (new) are present,
        'quoteCount' wins because it is more precise (actual speech count)."""
        data = [
            {
                "id": 1,
                "mentions": {"proper": [{"n": "Jane", "c": 5}], "common": [], "pronoun": []},
                "count": 100,  # total mentions
                "quoteCount": 42,  # actual speech acts (more precise)
            }
        ]
        path = self._write_book_json(tmp_path, data)
        chars = _extract_characters(path)
        assert chars[0].quote_count == 42


# ---------------------------------------------------------------------------
# _build_token_to_para_map and _build_para_to_quote
# ---------------------------------------------------------------------------


class TestBuildTokenToParaMap:
    """Tests for _build_token_to_para_map."""

    def _make_locations(self) -> list[_ParaLocation]:
        return [
            _ParaLocation(chapter_index=0, para_index=0, byte_start=0, byte_end=20),
            _ParaLocation(chapter_index=0, para_index=1, byte_start=22, byte_end=45),
            _ParaLocation(chapter_index=1, para_index=0, byte_start=47, byte_end=70),
        ]

    def test_maps_token_to_correct_paragraph(self):
        locations = self._make_locations()
        tokens = [
            {"token_id": 0, "byte_onset": 5, "byte_offset": 8},  # ch0 para0
            {"token_id": 1, "byte_onset": 30, "byte_offset": 35},  # ch0 para1
            {"token_id": 2, "byte_onset": 55, "byte_offset": 60},  # ch1 para0
        ]
        mapping = _build_token_to_para_map(tokens, locations)
        assert mapping[0].chapter_index == 0
        assert mapping[0].para_index == 0
        assert mapping[1].chapter_index == 0
        assert mapping[1].para_index == 1
        assert mapping[2].chapter_index == 1
        assert mapping[2].para_index == 0

    def test_token_in_gap_not_mapped(self):
        """A token whose byte_onset falls between paragraphs is not mapped."""
        locations = self._make_locations()
        tokens = [{"token_id": 99, "byte_onset": 21, "byte_offset": 22}]  # gap
        mapping = _build_token_to_para_map(tokens, locations)
        assert 99 not in mapping

    def test_empty_tokens_gives_empty_map(self):
        locations = self._make_locations()
        assert _build_token_to_para_map([], locations) == {}

    def test_empty_locations_gives_empty_map(self):
        tokens = [{"token_id": 0, "byte_onset": 5, "byte_offset": 8}]
        assert _build_token_to_para_map(tokens, []) == {}


class TestBuildParaToQuote:
    """Tests for _build_para_to_quote, including char_id_map resolution."""

    def _loc(self, chapter_index, para_index, byte_start=0, byte_end=10):
        return _ParaLocation(chapter_index, para_index, byte_start, byte_end)

    def test_basic_attribution(self):
        token_to_para = {0: self._loc(0, 0), 5: self._loc(0, 1)}
        quotes = [
            {"quote_start": 0, "char_id": "42", "quote_text": "..."},
            {"quote_start": 5, "char_id": "7", "quote_text": "..."},
        ]
        result = _build_para_to_quote(0, quotes, token_to_para)
        assert result == {0: "42", 1: "7"}

    def test_char_id_map_resolves_to_coref_id(self):
        """When char_id_map is provided, raw IDs are resolved to coref IDs."""
        token_to_para = {0: self._loc(0, 2)}
        quotes = [{"quote_start": 0, "char_id": "42", "quote_text": "hello"}]
        char_id_map = {"42": "ELIZABETH_BENNETT-42"}
        result = _build_para_to_quote(0, quotes, token_to_para, char_id_map)
        assert result == {2: "ELIZABETH_BENNETT-42"}

    def test_unmapped_char_id_uses_raw(self):
        """If char_id is not in char_id_map, the raw id is used as fallback."""
        token_to_para = {0: self._loc(0, 0)}
        quotes = [{"quote_start": 0, "char_id": "99", "quote_text": "x"}]
        char_id_map = {"42": "SOMEONE-42"}  # 99 not present
        result = _build_para_to_quote(0, quotes, token_to_para, char_id_map)
        assert result == {0: "99"}

    def test_skips_unknown_char_id(self):
        """char_id of '-1' is ignored (BookNLP sentinel for unknown speaker)."""
        token_to_para = {0: self._loc(0, 0)}
        quotes = [{"quote_start": 0, "char_id": "-1", "quote_text": "?"}]
        result = _build_para_to_quote(0, quotes, token_to_para)
        assert result == {}

    def test_skips_quote_in_wrong_chapter(self):
        """Quotes whose token maps to a different chapter are ignored."""
        # token 0 maps to chapter 1, but we are building for chapter 0
        token_to_para = {0: self._loc(1, 0)}
        quotes = [{"quote_start": 0, "char_id": "42", "quote_text": "x"}]
        result = _build_para_to_quote(0, quotes, token_to_para)
        assert result == {}

    def test_first_attribution_wins(self):
        """When two quotes map to the same paragraph, the first one wins."""
        token_to_para = {0: self._loc(0, 0), 1: self._loc(0, 0)}
        quotes = [
            {"quote_start": 0, "char_id": "1", "quote_text": "first"},
            {"quote_start": 1, "char_id": "2", "quote_text": "second"},
        ]
        result = _build_para_to_quote(0, quotes, token_to_para)
        assert result == {0: "1"}

    def test_token_not_in_mapping_skipped(self):
        """If quote_start token is not in token_to_para, the quote is skipped."""
        token_to_para: dict = {}
        quotes = [{"quote_start": 999, "char_id": "42", "quote_text": "x"}]
        result = _build_para_to_quote(0, quotes, token_to_para)
        assert result == {}


# ---------------------------------------------------------------------------
# run_analysis with mocked BookNLP
# ---------------------------------------------------------------------------


class TestRunAnalysis:
    """Tests for run_analysis() using a mocked BookNLP instance.

    We patch:
    - BOOKNLP_AVAILABLE → True (or False for the negative case)
    - booknlp.booknlp.BookNLP → a MagicMock whose .process() writes synthetic
      output files into the temp directory
    """

    def _write_synthetic_outputs(self, output_dir: Path, book_id: str):
        """Write minimal .tokens, .quotes, and .book files for testing."""
        # .tokens  — two tokens in chapter 0 paragraph 0
        tokens_content = (
            "paragraph_ID\tsentence_ID\ttoken_ID_sentence\ttoken_ID_doc"
            "\tword\tlemma\tbyte_onset\tbyte_offset\tPOS\tdep\tdep_head\tevent\n"
            "0\t0\t0\t0\tShe\tshe\t0\t3\tPRP\tnsubj\t1\tO\n"
            "0\t0\t1\t1\tsaid\tsay\t4\t8\tVBD\tROOT\t1\tO\n"
        )
        (output_dir / f"{book_id}.tokens").write_text(tokens_content, encoding="utf-8")

        # .quotes — one quote attributed to char_id "42"
        quotes_content = (
            "quote_start\tquote_end\tmention_start\tmention_end"
            "\tmention_text\tchar_id\tquote_text\n"
            "0\t1\t2\t3\tShe\t42\tHello world.\n"
        )
        (output_dir / f"{book_id}.quotes").write_text(quotes_content, encoding="utf-8")

        # .book JSON — real BookNLP 1.0.8 schema
        book_json = json.dumps(
            {
                "characters": [
                    {
                        "id": 42,
                        "mentions": {
                            "proper": [{"n": "Alice", "c": 5}],
                            "common": [],
                            "pronoun": [{"n": "she", "c": 3}],
                        },
                        "count": 1,
                        "g": {"argmax": "she/her", "inference": {}},
                    }
                ]
            }
        )
        (output_dir / f"{book_id}.book").write_text(book_json, encoding="utf-8")

    def _make_booknlp_mock(self, output_dir: Path, book_id: str):
        """Return a mock BookNLP class whose process() writes synthetic outputs."""
        mock_instance = MagicMock()
        mock_instance.process.side_effect = lambda input_path, out_dir, bid: (
            self._write_synthetic_outputs(Path(out_dir), bid)
        )
        mock_class = MagicMock(return_value=mock_instance)
        return mock_class

    def test_returns_booknlp_result(self, tmp_path):
        """run_analysis returns a BookNLPResult with characters and chapters."""
        from kenkui.booknlp_processor import run_analysis

        book_path = tmp_path / "book.epub"
        book_path.write_text("fake")

        chapter = Chapter(index=0, title="Ch 0", paragraphs=["She said hello."])

        mock_class = MagicMock()
        mock_instance = mock_class.return_value
        mock_instance.process.side_effect = lambda inp, out, bid: self._write_synthetic_outputs(
            Path(out), bid
        )

        import types

        fake_booknlp_module = types.ModuleType("booknlp.booknlp")
        fake_booknlp_module.BookNLP = mock_class  # type: ignore[attr-defined]

        import kenkui.booknlp_processor as bmod

        original_available = bmod.BOOKNLP_AVAILABLE
        bmod.BOOKNLP_AVAILABLE = lambda: True  # type: ignore[assignment]
        try:
            with patch.dict("sys.modules", {"booknlp.booknlp": fake_booknlp_module}):
                result = run_analysis([chapter], book_path, model_size="small")
        finally:
            bmod.BOOKNLP_AVAILABLE = original_available  # type: ignore[assignment]

        assert isinstance(result, BookNLPResult)
        assert len(result.characters) == 1
        assert result.characters[0].display_name == "Alice"
        assert len(result.chapters) == 1

    def test_segments_speaker_matches_character_id(self, tmp_path):
        """Segment.speaker values must appear in the characters list as character_id.

        This is the regression test for the speaker ID mismatch bug:
        previously Segment.speaker stored raw char_id ('42') but speaker_voices
        was keyed by composite coref_id ('ALICE-42'), causing all characters
        to fall back to the narrator voice.
        """
        from kenkui.booknlp_processor import run_analysis

        book_path = tmp_path / "book.epub"
        book_path.write_text("fake")

        # Single paragraph that will be attributed to char 42
        chapter = Chapter(index=0, title="Ch 0", paragraphs=["She said hello."])

        mock_class = MagicMock()
        mock_instance = mock_class.return_value
        mock_instance.process.side_effect = lambda inp, out, bid: self._write_synthetic_outputs(
            Path(out), bid
        )

        import types

        fake_booknlp_module = types.ModuleType("booknlp.booknlp")
        fake_booknlp_module.BookNLP = mock_class  # type: ignore[attr-defined]

        import kenkui.booknlp_processor as bmod

        original_available = bmod.BOOKNLP_AVAILABLE
        bmod.BOOKNLP_AVAILABLE = lambda: True  # type: ignore[assignment]
        try:
            with patch.dict("sys.modules", {"booknlp.booknlp": fake_booknlp_module}):
                result = run_analysis([chapter], book_path)
        finally:
            bmod.BOOKNLP_AVAILABLE = original_available  # type: ignore[assignment]

        character_ids = {c.character_id for c in result.characters}
        for ch in result.chapters:
            if ch.segments:
                for seg in ch.segments:
                    if seg.speaker != "NARRATOR":
                        assert seg.speaker in character_ids, (
                            f"Segment speaker '{seg.speaker}' not in character_ids "
                            f"{character_ids} — speaker/character_id mismatch"
                        )

    def test_raises_when_booknlp_unavailable(self, tmp_path):
        """run_analysis raises RuntimeError when BookNLP is not available."""
        from kenkui.booknlp_processor import run_analysis

        import kenkui.booknlp_processor as bmod

        original = bmod.BOOKNLP_AVAILABLE
        bmod.BOOKNLP_AVAILABLE = lambda: False  # type: ignore[assignment]
        try:
            with pytest.raises(RuntimeError, match="not available"):
                run_analysis([], tmp_path / "book.epub")
        finally:
            bmod.BOOKNLP_AVAILABLE = original  # type: ignore[assignment]

    def test_progress_callback_called(self, tmp_path):
        """progress_callback receives status strings during analysis."""
        from kenkui.booknlp_processor import run_analysis

        book_path = tmp_path / "book.epub"
        book_path.write_text("fake")
        chapter = Chapter(index=0, title="Ch 0", paragraphs=["Text."])

        messages: list[str] = []

        mock_class = MagicMock()
        mock_instance = mock_class.return_value
        mock_instance.process.side_effect = lambda inp, out, bid: self._write_synthetic_outputs(
            Path(out), bid
        )

        import types

        fake_mod = types.ModuleType("booknlp.booknlp")
        fake_mod.BookNLP = mock_class  # type: ignore[attr-defined]

        import kenkui.booknlp_processor as bmod

        original = bmod.BOOKNLP_AVAILABLE
        bmod.BOOKNLP_AVAILABLE = lambda: True  # type: ignore[assignment]
        try:
            with patch.dict("sys.modules", {"booknlp.booknlp": fake_mod}):
                run_analysis([chapter], book_path, progress_callback=messages.append)
        finally:
            bmod.BOOKNLP_AVAILABLE = original  # type: ignore[assignment]

        assert any(
            "analysis" in m.lower() or "booknlp" in m.lower() or "segment" in m.lower()
            for m in messages
        )

    def test_book_hash_in_result(self, tmp_path):
        """Result carries a non-empty book_hash for cache keying."""
        from kenkui.booknlp_processor import _book_hash, run_analysis

        book_path = tmp_path / "book.epub"
        book_path.write_text("fake")
        chapter = Chapter(index=0, title="Ch 0", paragraphs=["Text."])

        mock_class = MagicMock()
        mock_instance = mock_class.return_value
        mock_instance.process.side_effect = lambda inp, out, bid: self._write_synthetic_outputs(
            Path(out), bid
        )

        import types

        fake_mod = types.ModuleType("booknlp.booknlp")
        fake_mod.BookNLP = mock_class  # type: ignore[attr-defined]

        import kenkui.booknlp_processor as bmod

        original = bmod.BOOKNLP_AVAILABLE
        bmod.BOOKNLP_AVAILABLE = lambda: True  # type: ignore[assignment]
        try:
            with patch.dict("sys.modules", {"booknlp.booknlp": fake_mod}):
                result = run_analysis([chapter], book_path)
        finally:
            bmod.BOOKNLP_AVAILABLE = original  # type: ignore[assignment]

        assert result.book_hash == _book_hash(book_path)


# ---------------------------------------------------------------------------
# _write_input_file — byte offset accuracy (binary mode regression)
# ---------------------------------------------------------------------------


class TestWriteInputFileBinaryOffsets:
    """Verify that _write_input_file records byte offsets that align with
    the actual encoded file content, so token byte_onset values from BookNLP
    (which reads the file in binary mode) match our _ParaLocation ranges."""

    def test_byte_offsets_bracket_paragraph_text(self, tmp_path):
        """byte_start and byte_end in each _ParaLocation must bracket the
        UTF-8 encoded paragraph bytes in the written file."""
        chapters = [
            Chapter(index=0, title="Ch 0", paragraphs=["Hello world.", "Second para."]),
            Chapter(index=1, title="Ch 1", paragraphs=["Third para."]),
        ]
        out = tmp_path / "book.txt"
        locations = _write_input_file(chapters, out)

        raw_bytes = out.read_bytes()
        for loc in locations:
            # The slice of the file at [byte_start:byte_end] should decode to
            # the paragraph text we wrote.
            chapter = chapters[loc.chapter_index]
            para = chapter.paragraphs[loc.para_index]
            extracted = raw_bytes[loc.byte_start : loc.byte_end].decode("utf-8")
            assert extracted == para, (
                f"Byte range [{loc.byte_start}:{loc.byte_end}] decoded to "
                f"{extracted!r}, expected {para!r}"
            )

    def test_byte_offsets_with_unicode_text(self, tmp_path):
        """Multi-byte Unicode characters must be handled correctly."""
        # "café" is 5 bytes in UTF-8 (c, a, f, é=2 bytes)
        chapters = [Chapter(index=0, title="Ch 0", paragraphs=["Café au lait.", "Naïve résumé."])]
        out = tmp_path / "unicode.txt"
        locations = _write_input_file(chapters, out)

        raw_bytes = out.read_bytes()
        for loc in locations:
            para = chapters[0].paragraphs[loc.para_index]
            extracted = raw_bytes[loc.byte_start : loc.byte_end].decode("utf-8")
            assert extracted == para


# ---------------------------------------------------------------------------
# BookNLP availability smoke-test
# ---------------------------------------------------------------------------


class TestBookNLPSmoke:
    """Smoke tests that exercise the *real* BookNLP stack.

    These tests are skipped automatically when the spaCy model is absent
    (``BOOKNLP_AVAILABLE() is False``), so CI without the model stays green.
    When the model IS installed they verify:

    * ``BookNLP("en", {...})`` constructs without raising.
    * ``run_analysis()`` processes a minimal text end-to-end and returns a
      ``BookNLPResult`` with the expected shape.

    The point of these tests is to catch the class of bug where
    ``BOOKNLP_AVAILABLE()`` returns ``True`` but ``BookNLP.__init__`` crashes
    (e.g. missing ``en_core_web_sm``), so we exercise the constructor path
    directly rather than only checking import-time availability.
    """

    def setup_method(self):
        from kenkui.booknlp_processor import reset_booknlp_check

        reset_booknlp_check()

    def teardown_method(self):
        from kenkui.booknlp_processor import reset_booknlp_check

        reset_booknlp_check()

    @pytest.fixture(autouse=True)
    def skip_if_unavailable(self):
        """Skip the whole class when BookNLP is not fully operational."""
        from kenkui.booknlp_processor import BOOKNLP_AVAILABLE

        if not BOOKNLP_AVAILABLE():
            pytest.skip("BookNLP / en_core_web_sm not available in this environment")

    def test_booknlp_constructor_does_not_raise(self):
        """BookNLP('en', model_params) must not raise OSError or RuntimeError.

        Catches two classes of failure:
        - OSError [E050]: en_core_web_sm missing (spaCy model absent)
        - RuntimeError (unexpected key 'bert.embeddings.position_ids'):
          booknlp 1.0.8 model checkpoints were saved with transformers ~4.x
          where position_ids was a persistent buffer.  transformers 5.x made
          it non-persistent, so loading with strict=True fails.  The shim in
          run_analysis() patches load_state_dict to strict=False; this test
          applies the same shim to verify the constructor path is covered.
        """
        import torch.nn as nn
        from booknlp.booknlp import BookNLP  # type: ignore[import]

        model_params = {"pipeline": "entity,quote,coref", "model": "small"}

        orig = nn.Module.load_state_dict

        def _permissive(self, state_dict, strict=True, **kwargs):
            return orig(self, state_dict, strict=False, **kwargs)

        nn.Module.load_state_dict = _permissive  # type: ignore[method-assign]
        try:
            instance = BookNLP("en", model_params)
        finally:
            nn.Module.load_state_dict = orig  # type: ignore[method-assign]

        assert instance is not None

    def test_run_analysis_returns_result(self, tmp_path):
        """run_analysis() completes on a tiny synthetic book without error."""
        from kenkui.booknlp_processor import run_analysis

        book_path = tmp_path / "tiny.epub"
        book_path.write_text("placeholder")

        chapters = [
            Chapter(
                index=0,
                title="Chapter 1",
                paragraphs=[
                    'Alice said "Hello."',
                    "She walked into the room.",
                    'Bob replied "Hi there."',
                ],
            )
        ]

        result = run_analysis(chapters, book_path, model_size="small")

        assert isinstance(result, BookNLPResult)
        assert isinstance(result.characters, list)
        assert isinstance(result.chapters, list)
        assert len(result.chapters) == 1
        assert result.chapters[0].segments is not None
        assert isinstance(result.book_hash, str) and len(result.book_hash) > 0

    def test_run_analysis_segment_speakers_are_valid(self, tmp_path):
        """When BookNLP resolves a quote to a named character, Segment.speaker
        must use the composite coref_id (e.g. 'ALICE-42'), not the raw numeric
        char_id ('42').

        Regression test for the speaker-ID mismatch bug.  On very short text
        BookNLP may attribute some quotes to entity id 0 or other implicit
        referents that don't appear as named characters in the .book JSON —
        those fall back to the raw id (acceptable).  We only assert that
        *when* a speaker id IS resolvable (present in character_ids), the
        composite form is used rather than the raw numeric form.
        """
        from kenkui.booknlp_processor import run_analysis

        book_path = tmp_path / "book.epub"
        book_path.write_text("placeholder")

        chapters = [
            Chapter(
                index=0,
                title="Chapter 1",
                paragraphs=[
                    'Mary said "Good morning."',
                    "The sun rose over the hills.",
                    'John replied "Good morning to you too."',
                    "They sat down to breakfast.",
                ],
            )
        ]

        result = run_analysis(chapters, book_path, model_size="small")
        character_ids = {c.character_id for c in result.characters}
        # Build a set of raw numeric ids that ARE in the character list,
        # so we know which speakers *should* have been resolved.
        raw_ids_in_chars = {c.character_id.rsplit("-", 1)[-1] for c in result.characters}

        for chapter in result.chapters:
            for seg in chapter.segments or []:
                if seg.speaker == "NARRATOR":
                    continue
                # Only assert composite form when the raw id maps to a known character.
                # Quotes attributed to implicit referents (e.g. id '0') that have no
                # named character entry in .book JSON legitimately use the raw id.
                if seg.speaker in raw_ids_in_chars:
                    assert seg.speaker in character_ids, (
                        f"Segment.speaker={seg.speaker!r} is a raw char_id but "
                        f"should have been resolved to a composite coref_id. "
                        f"character_ids={character_ids!r}"
                    )
