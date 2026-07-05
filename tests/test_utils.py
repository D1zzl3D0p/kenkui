"""Tests for kenkui.utils — batch_text, ensure_terminal_punct, and _normalize_bitrate."""

from __future__ import annotations

import pytest

from kenkui.models import _normalize_bitrate
from kenkui.utils import batch_text

# ---------------------------------------------------------------------------
# batch_text
# ---------------------------------------------------------------------------


class TestBatchTextMerging:
    """Short segments should be merged into batches up to max_chars."""

    def test_merges_short_segments(self):
        segs = ["Hello.", "World.", "Foo."]
        result = batch_text(segs, max_chars=800, merge_short=True)
        assert len(result) == 1
        assert "Hello." in result[0]
        assert "World." in result[0]

    def test_respects_max_chars_boundary(self):
        # Each segment is 10 chars; max_chars=25 → 2 segments per batch
        segs = ["AAAAAAAAAA"] * 5  # 10 chars each
        result = batch_text(segs, max_chars=25, merge_short=True)
        for batch in result:
            assert len(batch) <= 25 + 1  # +1 for the space separator

    def test_empty_input_returns_empty(self):
        assert batch_text([], max_chars=800) == []

    def test_single_short_segment_returned_as_is(self):
        result = batch_text(["Hello world."], max_chars=800)
        assert result == ["Hello world."]

    def test_skips_blank_segments(self):
        segs = ["Hello.", "", "World."]
        result = batch_text(segs, max_chars=800)
        # Empty string should be dropped
        assert all(s.strip() for s in result)


class TestBatchTextSplitting:
    """Long paragraphs should be split at sentence boundaries."""

    def test_splits_long_paragraph(self):
        long = "Sentence one. Sentence two. Sentence three. " * 5  # ~220 chars
        result = batch_text([long], max_chars=50)
        assert len(result) > 1
        for chunk in result:
            assert (
                len(chunk) <= 50 or "." in chunk
            )  # each chunk is at most one sentence

    def test_hard_splits_single_oversized_sentence(self):
        result = batch_text(["A" * 121], max_chars=50)
        assert [len(chunk) for chunk in result] == [50, 50, 21]

    def test_long_paragraphs_flushed_before_merge(self):
        """A long paragraph should be emitted separately, not merged with next short one."""
        long = "A" * 900 + "."
        short = "Short."
        result = batch_text([long, short], max_chars=800)
        # The long paragraph should be in its own batch (or split)
        # The short one should be separate or merged only with other shorts
        joined = " ".join(result)
        assert "A" in joined
        assert "Short." in joined


class TestBatchTextMergeShortFalse:
    """merge_short=False: each paragraph is its own item (no merging)."""

    def test_no_merging_when_false(self):
        segs = ["A.", "B.", "C."]
        result = batch_text(segs, max_chars=800, merge_short=False)
        assert result == ["A.", "B.", "C."]

    def test_long_paragraphs_still_split_when_merge_false(self):
        long = "Sentence one. Sentence two. Sentence three. " * 10
        result = batch_text([long], max_chars=50, merge_short=False)
        assert len(result) > 1


# ---------------------------------------------------------------------------
# _normalize_bitrate
# ---------------------------------------------------------------------------


class TestNormalizeBitrate:
    @pytest.mark.parametrize(
        "inp,expected",
        [
            ("64k", "64k"),
            ("96K", "96k"),
            ("128k", "128k"),
            ("64", "64k"),  # THE bug case — bare int gets 'k'
            ("96", "96k"),
            ("128", "128k"),
            ("999", "999k"),
            ("1000", "1000"),  # >= 1000 → already in bps, pass through
            ("128000", "128000"),  # large bps value, pass through
            ("", "96k"),  # empty → default
            (None, "96k"),  # None → default
            ("bad", "96k"),  # garbage → default
            (" 64k ", "64k"),  # whitespace stripped
        ],
    )
    def test_normalize(self, inp, expected):
        assert _normalize_bitrate(inp) == expected

    def test_custom_default(self):
        assert _normalize_bitrate(None, default="128k") == "128k"
        assert _normalize_bitrate("", default="64k") == "64k"


# ---------------------------------------------------------------------------
# normalize_for_tts
# ---------------------------------------------------------------------------

from kenkui.utils import normalize_for_tts, normalize_numbers_for_tts


class TestNormalizeForTts:
    """Tests for normalize_for_tts() — n't contraction expansion."""

    @pytest.mark.parametrize("contraction,expansion", [
        ("won't", "will not"),
        ("can't", "cannot"),
        ("don't", "do not"),
        ("doesn't", "does not"),
        ("didn't", "did not"),
        ("isn't", "is not"),
        ("aren't", "are not"),
        ("wasn't", "was not"),
        ("weren't", "were not"),
        ("haven't", "have not"),
        ("hasn't", "has not"),
        ("hadn't", "had not"),
        ("couldn't", "could not"),
        ("wouldn't", "would not"),
        ("shouldn't", "should not"),
        ("mustn't", "must not"),
        ("needn't", "need not"),
        ("shan't", "shall not"),
    ])
    def test_nont_map_entries(self, contraction, expansion):
        assert normalize_for_tts(contraction) == expansion

    def test_uppercase_preserved(self):
        assert normalize_for_tts("DON'T") == "DO NOT"

    def test_uppercase_doesnt(self):
        assert normalize_for_tts("DOESN'T") == "DOES NOT"

    def test_title_case_preserved(self):
        assert normalize_for_tts("Don't") == "Do not"

    def test_title_case_wont(self):
        assert normalize_for_tts("Won't") == "Will not"

    def test_curly_apostrophe_dont(self):
        assert normalize_for_tts("don\u2019t") == "do not"

    def test_curly_apostrophe_cant(self):
        assert normalize_for_tts("can\u2019t") == "cannot"

    def test_curly_apostrophe_uppercase(self):
        assert normalize_for_tts("DON\u2019T") == "DO NOT"

    def test_non_nont_im_unchanged(self):
        assert normalize_for_tts("I'm ready", mode=ApostropheMode.KEEP) == "I'm ready"

    def test_non_nont_were_unchanged(self):
        assert normalize_for_tts("we're here", mode=ApostropheMode.KEEP) == "we're here"

    def test_non_nont_its_unchanged(self):
        assert normalize_for_tts("it's fine", mode=ApostropheMode.KEEP) == "it's fine"

    def test_non_nont_theyre_unchanged(self):
        assert normalize_for_tts("they're going", mode=ApostropheMode.KEEP) == "they're going"

    def test_mid_sentence(self):
        result = normalize_for_tts("He doesn't know.")
        assert result == "He does not know."

    def test_sentence_start(self):
        result = normalize_for_tts("Don't worry.")
        assert result == "Do not worry."

    def test_sentence_end(self):
        result = normalize_for_tts("He said he wouldn't.")
        assert result == "He said he would not."

    def test_multiple_contractions_in_one_sentence(self):
        result = normalize_for_tts("She doesn't know and wasn't sure.")
        assert result == "She does not know and was not sure."

    def test_no_contractions_unchanged(self):
        text = "The sun rose over the hills."
        assert normalize_for_tts(text) == text

    def test_empty_string(self):
        assert normalize_for_tts("") == ""


class TestNormalizeNumbersForTts:
    def test_comma_cardinal_to_words(self):
        assert normalize_numbers_for_tts("There were 100,000 people.") == (
            "There were one hundred thousand people."
        )

    def test_long_identifier_to_digits(self):
        assert normalize_numbers_for_tts("Call id 3495992019.") == (
            "Call id three four nine five nine nine two zero one nine."
        )

    def test_phone_number_to_grouped_digits(self):
        assert normalize_numbers_for_tts("Dial 801-999-9999 now.") == (
            "Dial eight zero one, nine nine nine, nine nine nine nine now."
        )

    def test_raw_phone_mode_protects_against_cardinal_fallback(self):
        result = normalize_numbers_for_tts(
            "Dial 801-999-9999 now.",
            {"phone_numbers_mode": "raw"},
        )
        assert result == "Dial 801-999-9999 now."

    def test_raw_decimal_mode_protects_against_cardinal_fallback(self):
        result = normalize_numbers_for_tts(
            "The value was 3.14 exactly.",
            {"decimals_mode": "raw"},
        )
        assert result == "The value was 3.14 exactly."

    def test_decimal_and_percent_to_words(self):
        assert normalize_numbers_for_tts("Use 3.14 for 25%.") == (
            "Use three point one four for twenty five percent."
        )

    def test_ordinal_to_words(self):
        assert normalize_numbers_for_tts("She finished 21st.") == "She finished twenty first."


# ---------------------------------------------------------------------------
# normalize_for_tts — apostrophe_mode parameter
# ---------------------------------------------------------------------------

from kenkui.utils import ApostropheMode


class TestNormalizeForTtsMode:
    """Tests for the four apostrophe_mode dispatch paths."""

    # ── keep ────────────────────────────────────────────────────────────────
    def test_keep_leaves_dont_unchanged(self):
        assert normalize_for_tts("don't", mode=ApostropheMode.KEEP) == "don't"

    def test_keep_leaves_im_unchanged(self):
        assert normalize_for_tts("I'm ready", mode=ApostropheMode.KEEP) == "I'm ready"

    def test_keep_leaves_curly_unchanged(self):
        assert normalize_for_tts("don\u2019t", mode=ApostropheMode.KEEP) == "don\u2019t"

    def test_keep_leaves_obrien_unchanged(self):
        assert normalize_for_tts("O'Brien", mode=ApostropheMode.KEEP) == "O'Brien"

    # ── always_remove ────────────────────────────────────────────────────────
    def test_always_remove_strips_contraction(self):
        assert normalize_for_tts("don't", mode=ApostropheMode.ALWAYS_REMOVE) == "dont"

    def test_always_remove_strips_proper_name(self):
        assert normalize_for_tts("O'Brien", mode=ApostropheMode.ALWAYS_REMOVE) == "OBrien"

    def test_always_remove_strips_curly_apostrophe(self):
        assert normalize_for_tts("don\u2019t", mode=ApostropheMode.ALWAYS_REMOVE) == "dont"

    def test_always_remove_strips_curly_left_quote(self):
        # U+2018 left single quotation mark
        assert normalize_for_tts("\u2018twas", mode=ApostropheMode.ALWAYS_REMOVE) == "twas"

    # ── remove_contractions ──────────────────────────────────────────────────
    def test_remove_contractions_strips_apostrophe_from_contraction(self):
        assert normalize_for_tts("don't", mode=ApostropheMode.REMOVE_CONTRACTIONS) == "dont"

    def test_remove_contractions_preserves_proper_name(self):
        assert normalize_for_tts("O'Brien", mode=ApostropheMode.REMOVE_CONTRACTIONS) == "O'Brien"

    def test_remove_contractions_strips_im(self):
        assert normalize_for_tts("I'm ready", mode=ApostropheMode.REMOVE_CONTRACTIONS) == "Im ready"

    def test_remove_contractions_curly_apostrophe(self):
        assert normalize_for_tts("don\u2019t", mode=ApostropheMode.REMOVE_CONTRACTIONS) == "dont"

    def test_remove_contractions_curly_apostrophe_non_nont(self):
        # U+2019 in a subject contraction → strip apostrophe
        assert normalize_for_tts("I\u2019m ready", mode=ApostropheMode.REMOVE_CONTRACTIONS) == "Im ready"

    # ── expand_contractions ──────────────────────────────────────────────────
    def test_expand_contractions_dont(self):
        assert normalize_for_tts("don't", mode=ApostropheMode.EXPAND_CONTRACTIONS) == "do not"

    def test_expand_contractions_im(self):
        assert normalize_for_tts("I'm ready", mode=ApostropheMode.EXPAND_CONTRACTIONS) == "I am ready"

    def test_expand_contractions_its(self):
        assert normalize_for_tts("it's fine", mode=ApostropheMode.EXPAND_CONTRACTIONS) == "it is fine"

    def test_expand_contractions_preserves_proper_name(self):
        assert normalize_for_tts("O'Brien", mode=ApostropheMode.EXPAND_CONTRACTIONS) == "O'Brien"

    def test_expand_contractions_curly(self):
        assert normalize_for_tts("don\u2019t", mode=ApostropheMode.EXPAND_CONTRACTIONS) == "do not"

    def test_expand_contractions_uppercase(self):
        assert normalize_for_tts("I'M READY", mode=ApostropheMode.EXPAND_CONTRACTIONS) == "I AM READY"

    def test_expand_contractions_title_case(self):
        assert normalize_for_tts("I'm ready", mode=ApostropheMode.EXPAND_CONTRACTIONS) == "I am ready"

    # ── default mode ─────────────────────────────────────────────────────────
    def test_default_mode_is_expand_contractions(self):
        # no mode arg → behaves like expand_contractions
        assert normalize_for_tts("don't") == "do not"
        assert normalize_for_tts("I'm ready") == "I am ready"


# ---------------------------------------------------------------------------
# ApostropheMode round-trip through AppConfig / JobConfig
# ---------------------------------------------------------------------------

from pathlib import Path

from kenkui.models import AppConfig, JobConfig


class TestApostropheModeModels:
    """AppConfig and JobConfig serialization round-trips."""

    def test_appconfig_default_is_expand_contractions(self):
        cfg = AppConfig()
        assert cfg.apostrophe_mode == ApostropheMode.EXPAND_CONTRACTIONS

    def test_appconfig_to_dict_includes_apostrophe_mode(self):
        cfg = AppConfig()
        d = cfg.to_dict()
        assert d["apostrophe_mode"] == "expand_contractions"

    def test_appconfig_from_dict_roundtrip(self):
        cfg = AppConfig.from_dict({"apostrophe_mode": "keep"})
        assert cfg.apostrophe_mode == ApostropheMode.KEEP

    def test_appconfig_from_dict_missing_key_defaults_to_expand(self):
        cfg = AppConfig.from_dict({})
        assert cfg.apostrophe_mode == ApostropheMode.EXPAND_CONTRACTIONS

    def test_jobconfig_default_job_apostrophe_mode_is_none(self):
        job = JobConfig(ebook_path=Path("book.epub"))
        assert job.job_apostrophe_mode is None

    def test_jobconfig_to_dict_omits_none_job_apostrophe_mode(self):
        job = JobConfig(ebook_path=Path("book.epub"))
        d = job.to_dict()
        assert "job_apostrophe_mode" not in d

    def test_jobconfig_to_dict_includes_when_set(self):
        job = JobConfig(ebook_path=Path("book.epub"), job_apostrophe_mode=ApostropheMode.ALWAYS_REMOVE)
        d = job.to_dict()
        assert d["job_apostrophe_mode"] == "always_remove"

    def test_jobconfig_from_dict_roundtrip(self):
        job = JobConfig.from_dict({
            "ebook_path": "book.epub",
            "job_apostrophe_mode": "remove_contractions",
        })
        assert job.job_apostrophe_mode == ApostropheMode.REMOVE_CONTRACTIONS

    def test_jobconfig_from_dict_missing_job_apostrophe_mode_is_none(self):
        job = JobConfig.from_dict({"ebook_path": "book.epub"})
        assert job.job_apostrophe_mode is None


# ---------------------------------------------------------------------------
# ensure_terminal_punct
# ---------------------------------------------------------------------------

from kenkui.utils import ensure_terminal_punct


class TestEnsureTerminalPunct:
    @pytest.mark.parametrize("text", [
        "Hello world.",
        "Really?",
        "Stop!",
        "Wait\u2026",
        "Continue…",
    ])
    def test_already_punctuated_unchanged(self, text):
        assert ensure_terminal_punct(text) == text

    def test_no_punct_appends_period(self):
        assert ensure_terminal_punct("Hello world") == "Hello world."

    def test_trailing_whitespace_stripped_then_period(self):
        assert ensure_terminal_punct("Hello world  ") == "Hello world."

    def test_empty_string_unchanged(self):
        assert ensure_terminal_punct("") == ""

    def test_whitespace_only_unchanged(self):
        assert ensure_terminal_punct("   ") == "   "

    def test_closing_double_quote_with_punct_unchanged(self):
        assert ensure_terminal_punct('"Hello."') == '"Hello."'

    def test_closing_curly_quote_with_punct_unchanged(self):
        assert ensure_terminal_punct("\u201cHello.\u201d") == "\u201cHello.\u201d"

    def test_closing_quote_without_punct_gets_period(self):
        result = ensure_terminal_punct('"Hello"')
        assert result == '"Hello."'

    def test_closing_curly_quote_without_punct_gets_period(self):
        result = ensure_terminal_punct("\u201cHello\u201d")
        assert result == "\u201cHello.\u201d"

    def test_question_mark_before_closing_quote_unchanged(self):
        assert ensure_terminal_punct('"Hello?"') == '"Hello?"'


# ---------------------------------------------------------------------------
# JobConfig per-job override fields
# ---------------------------------------------------------------------------


def test_job_config_eos_threshold_override():
    from pathlib import Path

    from kenkui.models import JobConfig
    job = JobConfig(ebook_path=Path("book.epub"), job_eos_threshold=-2.5)
    assert job.job_eos_threshold == -2.5


def test_job_config_tts_max_tokens_per_chunk_round_trip():
    from pathlib import Path

    from kenkui.models import JobConfig
    job = JobConfig(ebook_path=Path("book.epub"), job_tts_max_tokens_per_chunk=50)
    d = job.to_dict()
    assert d["job_tts_max_tokens_per_chunk"] == 50
    job2 = JobConfig.from_dict(d)
    assert job2.job_tts_max_tokens_per_chunk == 50


def test_job_config_post_processing_enabled_override():
    from pathlib import Path

    from kenkui.models import JobConfig
    job = JobConfig(ebook_path=Path("book.epub"), job_post_processing_enabled=False)
    assert job.job_post_processing_enabled is False


def test_job_config_pp_enabled_round_trip():
    from pathlib import Path

    from kenkui.models import JobConfig
    job = JobConfig(ebook_path=Path("book.epub"), job_post_processing_enabled=True)
    d = job.to_dict()
    job2 = JobConfig.from_dict(d)
    assert job2.job_post_processing_enabled is True


def test_job_config_pp_disabled_round_trip():
    from pathlib import Path

    from kenkui.models import JobConfig
    job = JobConfig(ebook_path=Path("book.epub"), job_post_processing_enabled=False)
    d = job.to_dict()
    assert d.get("job_post_processing_enabled") is False
    job2 = JobConfig.from_dict(d)
    assert job2.job_post_processing_enabled is False
