"""Tests for multi-voice data models added in the BookNLP integration.

Covers:
- NarrationMode enum
- CharacterInfo dataclass
- Segment to_dict / from_dict
- Chapter to_dict / from_dict (with and without segments)
- JobConfig round-trip with multi-voice fields
- AppConfig round-trip with nlp_model
- ProcessingConfig multi-voice fields
- Backward compatibility (old dicts without new fields)
"""

from __future__ import annotations

from pathlib import Path

import pytest

from kenkui.models import (
    AppConfig,
    Chapter,
    CharacterInfo,
    ChapterSelection,
    JobConfig,
    NarrationMode,
    ProcessingConfig,
    Segment,
)


# ---------------------------------------------------------------------------
# NarrationMode
# ---------------------------------------------------------------------------


class TestNarrationMode:
    def test_single_value(self):
        assert NarrationMode.SINGLE.value == "single"

    def test_multi_value(self):
        assert NarrationMode.MULTI.value == "multi"

    def test_from_string_single(self):
        assert NarrationMode("single") == NarrationMode.SINGLE

    def test_from_string_multi(self):
        assert NarrationMode("multi") == NarrationMode.MULTI

    def test_invalid_raises(self):
        with pytest.raises(ValueError):
            NarrationMode("invalid")


# ---------------------------------------------------------------------------
# CharacterInfo
# ---------------------------------------------------------------------------


class TestCharacterInfo:
    def test_defaults(self):
        c = CharacterInfo(character_id="FOO-0", display_name="Foo")
        assert c.quote_count == 0
        assert c.gender_pronoun == ""

    def test_all_fields(self):
        c = CharacterInfo(
            character_id="ELIZABETH_BENNETT-0",
            display_name="Elizabeth Bennet",
            quote_count=142,
            gender_pronoun="she",
        )
        assert c.character_id == "ELIZABETH_BENNETT-0"
        assert c.display_name == "Elizabeth Bennet"
        assert c.quote_count == 142
        assert c.gender_pronoun == "she"


# ---------------------------------------------------------------------------
# Segment serialization
# ---------------------------------------------------------------------------


class TestSegmentSerialization:
    def test_to_dict_contains_all_fields(self):
        s = Segment(text="Hello world.", speaker="ELIZABETH_BENNETT-0", index=3)
        d = s.to_dict()
        assert d["text"] == "Hello world."
        assert d["speaker"] == "ELIZABETH_BENNETT-0"
        assert d["index"] == 3

    def test_from_dict_round_trip(self):
        original = Segment(text="Narration.", speaker="NARRATOR", index=0)
        restored = Segment.from_dict(original.to_dict())
        assert restored.text == original.text
        assert restored.speaker == original.speaker
        assert restored.index == original.index

    def test_from_dict_defaults(self):
        s = Segment.from_dict({"text": "Hello."})
        assert s.speaker == "NARRATOR"
        assert s.index == 0

    def test_speaker_preserved(self):
        s = Segment.from_dict({"text": "Said she.", "speaker": "JANE-1", "index": 5})
        assert s.speaker == "JANE-1"
        assert s.index == 5


# ---------------------------------------------------------------------------
# Chapter serialization (with and without segments)
# ---------------------------------------------------------------------------


class TestChapterSerialization:
    def test_to_dict_no_segments(self):
        ch = Chapter(index=1, title="Chapter One", paragraphs=["Para 1.", "Para 2."])
        d = ch.to_dict()
        assert d["index"] == 1
        assert d["title"] == "Chapter One"
        assert d["paragraphs"] == ["Para 1.", "Para 2."]
        assert "segments" not in d

    def test_to_dict_with_segments(self):
        segs = [
            Segment("Narration text.", "NARRATOR", 0),
            Segment("Dialogue.", "DARCY-2", 1),
        ]
        ch = Chapter(index=0, title="Ch 0", paragraphs=["p"], segments=segs)
        d = ch.to_dict()
        assert "segments" in d
        assert len(d["segments"]) == 2
        assert d["segments"][1]["speaker"] == "DARCY-2"

    def test_from_dict_no_segments(self):
        d = {"index": 2, "title": "Chapter Two", "paragraphs": ["A.", "B."]}
        ch = Chapter.from_dict(d)
        assert ch.index == 2
        assert ch.title == "Chapter Two"
        assert ch.paragraphs == ["A.", "B."]
        assert ch.segments is None

    def test_from_dict_with_segments(self):
        d = {
            "index": 0,
            "title": "Prologue",
            "paragraphs": ["Para."],
            "segments": [
                {"text": "Narration.", "speaker": "NARRATOR", "index": 0},
                {"text": "Speech.", "speaker": "ALICE-0", "index": 1},
            ],
        }
        ch = Chapter.from_dict(d)
        assert ch.segments is not None
        assert len(ch.segments) == 2
        assert ch.segments[0].speaker == "NARRATOR"
        assert ch.segments[1].speaker == "ALICE-0"

    def test_round_trip_with_segments(self):
        segs = [Segment("Text.", "BOB-1", 0), Segment("More.", "NARRATOR", 1)]
        original = Chapter(index=5, title="Five", paragraphs=["x"], segments=segs)
        restored = Chapter.from_dict(original.to_dict())
        assert restored.index == 5
        assert restored.segments is not None
        assert len(restored.segments) == 2
        assert restored.segments[0].speaker == "BOB-1"

    def test_round_trip_no_segments(self):
        original = Chapter(index=3, title="Three", paragraphs=["a", "b"])
        restored = Chapter.from_dict(original.to_dict())
        assert restored.segments is None
        assert restored.paragraphs == ["a", "b"]


# ---------------------------------------------------------------------------
# JobConfig multi-voice fields
# ---------------------------------------------------------------------------


class TestJobConfigMultiVoice:
    def _base_job(self, **kwargs) -> JobConfig:
        return JobConfig(ebook_path=Path("/tmp/book.epub"), **kwargs)

    def test_default_narration_mode(self):
        job = self._base_job()
        assert job.narration_mode == NarrationMode.SINGLE

    def test_default_speaker_voices_empty(self):
        job = self._base_job()
        assert job.speaker_voices == {}

    def test_default_annotated_chapters_path_none(self):
        job = self._base_job()
        assert job.annotated_chapters_path is None

    def test_to_dict_includes_multi_voice(self):
        job = self._base_job(
            narration_mode=NarrationMode.MULTI,
            speaker_voices={"DARCY-0": "jean", "ELIZABETH-1": "cosette"},
            annotated_chapters_path=Path("/tmp/cache.json"),
        )
        d = job.to_dict()
        assert d["narration_mode"] == "multi"
        assert d["speaker_voices"]["DARCY-0"] == "jean"
        assert d["annotated_chapters_path"] == "/tmp/cache.json"

    def test_from_dict_round_trip(self):
        original = self._base_job(
            narration_mode=NarrationMode.MULTI,
            speaker_voices={"ALICE-0": "cosette"},
            annotated_chapters_path=Path("/tmp/my_cache.json"),
        )
        restored = JobConfig.from_dict(original.to_dict())
        assert restored.narration_mode == NarrationMode.MULTI
        assert restored.speaker_voices == {"ALICE-0": "cosette"}
        assert restored.annotated_chapters_path == Path("/tmp/my_cache.json")

    def test_backward_compat_missing_multi_voice_fields(self):
        """Old JobConfig dicts without narration_mode/speaker_voices should load."""
        old_data = {
            "ebook_path": "/tmp/old.epub",
            "voice": "alba",
            "chapter_selection": {},
            "name": "Old Book",
        }
        job = JobConfig.from_dict(old_data)
        assert job.narration_mode == NarrationMode.SINGLE
        assert job.speaker_voices == {}
        assert job.annotated_chapters_path is None


# ---------------------------------------------------------------------------
# AppConfig nlp_model field
# ---------------------------------------------------------------------------


class TestAppConfigNLPModel:
    def test_default_nlp_model(self):
        assert AppConfig().nlp_model == "llama3.2"

    def test_to_dict_includes_nlp_model(self):
        cfg = AppConfig(nlp_model="phi3:mini")
        d = cfg.to_dict()
        assert d["nlp_model"] == "phi3:mini"

    def test_from_dict_round_trip(self):
        cfg = AppConfig.from_dict({"nlp_model": "phi3:mini"})
        assert cfg.nlp_model == "phi3:mini"

    def test_missing_nlp_model_defaults(self):
        cfg = AppConfig.from_dict({})
        assert cfg.nlp_model == "llama3.2"


# ---------------------------------------------------------------------------
# ProcessingConfig multi-voice fields
# ---------------------------------------------------------------------------


class TestProcessingConfigMultiVoice:
    def _make_cfg(self, **kwargs):
        from kenkui.chapter_filter import FilterOperation

        return ProcessingConfig(
            voice="alba",
            ebook_path=Path("/tmp/book.epub"),
            output_path=Path("/tmp"),
            pause_line_ms=400,
            pause_chapter_ms=2000,
            workers=2,
            m4b_bitrate="96k",
            keep_temp=False,
            debug_html=False,
            chapter_filters=[FilterOperation("preset", "content-only")],
            **kwargs,
        )

    def test_default_speaker_voices_empty(self):
        cfg = self._make_cfg()
        assert cfg.speaker_voices == {}

    def test_default_annotated_chapters_path_none(self):
        cfg = self._make_cfg()
        assert cfg.annotated_chapters_path is None

    def test_speaker_voices_stored(self):
        voices = {"NARRATOR": "alba", "HOLMES-0": "jean"}
        cfg = self._make_cfg(speaker_voices=voices)
        assert cfg.speaker_voices["HOLMES-0"] == "jean"

    def test_annotated_chapters_path_stored(self):
        path = Path("/tmp/cache.json")
        cfg = self._make_cfg(annotated_chapters_path=path)
        assert cfg.annotated_chapters_path == path
