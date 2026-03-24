"""Tests for kenkui.models — AppConfig, Segment, ChapterSelection round-trips."""

from __future__ import annotations

from pathlib import Path

from kenkui.models import (
    AppConfig,
    Chapter,
    CharacterInfo,
    ChapterPreset,
    ChapterSelection,
    NLPResult,
    Segment,
)


class TestAppConfigDefaults:
    def test_default_voice(self):
        assert AppConfig().default_voice == "alba"

    def test_default_chapter_preset(self):
        assert AppConfig().default_chapter_preset == "content-only"

    def test_default_output_dir_none(self):
        assert AppConfig().default_output_dir is None

    def test_default_bitrate(self):
        assert AppConfig().m4b_bitrate == "96k"

    def test_default_temp(self):
        assert AppConfig().temp == 0.7

    def test_default_lsd_decode_steps(self):
        assert AppConfig().lsd_decode_steps == 1

    def test_default_nlp_model(self):
        assert AppConfig().nlp_model == "llama3.2"


class TestAppConfigRoundTrip:
    def test_to_dict_from_dict_identity(self):
        original = AppConfig(
            name="test",
            workers=4,
            verbose=True,
            m4b_bitrate="128k",
            pause_line_ms=500,
            pause_chapter_ms=3000,
            temp=0.8,
            lsd_decode_steps=2,
            noise_clamp=3.0,
            default_voice="cosette",
            default_chapter_preset="chapters-only",
            default_output_dir=Path("/tmp/audio"),
        )
        restored = AppConfig.from_dict(original.to_dict())
        assert restored.name == "test"
        assert restored.workers == 4
        assert restored.verbose is True
        assert restored.m4b_bitrate == "128k"
        assert restored.temp == 0.8
        assert restored.lsd_decode_steps == 2
        assert restored.noise_clamp == 3.0
        assert restored.default_voice == "cosette"
        assert restored.default_chapter_preset == "chapters-only"
        assert restored.default_output_dir == Path("/tmp/audio")

    def test_bitrate_normalized_on_load(self):
        cfg = AppConfig.from_dict({"m4b_bitrate": "64"})
        assert cfg.m4b_bitrate == "64k"

    def test_missing_keys_use_defaults(self):
        cfg = AppConfig.from_dict({})
        assert cfg.name == "default"
        assert cfg.m4b_bitrate == "96k"
        assert cfg.default_voice == "alba"
        assert cfg.default_chapter_preset == "content-only"

    def test_nlp_model_round_trip(self):
        cfg = AppConfig(nlp_model="phi3:mini")
        restored = AppConfig.from_dict(cfg.to_dict())
        assert restored.nlp_model == "phi3:mini"

    def test_new_fields_backward_compatible(self):
        """Old YAML files without the new fields should load cleanly."""
        old_data = {
            "name": "legacy",
            "workers": 4,
            "m4b_bitrate": "64k",
            # No default_voice, default_chapter_preset, default_output_dir, nlp_model
        }
        cfg = AppConfig.from_dict(old_data)
        assert cfg.default_voice == "alba"
        assert cfg.default_chapter_preset == "content-only"
        assert cfg.default_output_dir is None
        assert cfg.nlp_model == "llama3.2"  # default when field absent


class TestChapterSelectionRoundTrip:
    def test_preset_round_trip(self):
        sel = ChapterSelection(preset=ChapterPreset.CONTENT_ONLY, included=[1, 2], excluded=[3])
        restored = ChapterSelection.from_dict(sel.to_dict())
        assert restored.preset == ChapterPreset.CONTENT_ONLY
        assert restored.included == [1, 2]
        assert restored.excluded == [3]

    def test_custom_preset_round_trip(self):
        sel = ChapterSelection(preset=ChapterPreset.CUSTOM, included=[0, 2, 5])
        restored = ChapterSelection.from_dict(sel.to_dict())
        assert restored.preset == ChapterPreset.CUSTOM
        assert restored.included == [0, 2, 5]

    def test_empty_selection(self):
        sel = ChapterSelection()
        d = sel.to_dict()
        assert d["included"] == []
        assert d["excluded"] == []


class TestSegment:
    def test_segment_defaults(self):
        s = Segment(text="Hello world.")
        assert s.speaker == "NARRATOR"
        assert s.index == 0

    def test_segment_custom_speaker(self):
        s = Segment(text="I disagree!", speaker="alice", index=3)
        assert s.speaker == "alice"
        assert s.index == 3

    def test_segment_text_preserved(self):
        text = "The quick brown fox."
        s = Segment(text=text)
        assert s.text == text


class TestChapterWithSegments:
    def _make_tags(self):
        from kenkui.chapter_classifier import ChapterTags

        return ChapterTags(is_chapter=True)

    def test_chapter_segments_none_by_default(self):

        ch = Chapter(index=0, title="Ch 1", paragraphs=["Para."])
        assert ch.segments is None

    def test_chapter_with_segments(self):
        segs = [Segment("Narration.", "NARRATOR", 0), Segment("Dialogue.", "alice", 1)]
        ch = Chapter(index=0, title="Ch 1", paragraphs=["Para."], segments=segs)
        assert ch.segments is not None
        assert len(ch.segments) == 2
        assert ch.segments[1].speaker == "alice"


class TestCharacterInfo:
    def test_round_trip(self):
        c = CharacterInfo(
            character_id="Harry Potter",
            display_name="Harry Potter",
            quote_count=42,
            gender_pronoun="he",
        )
        restored = CharacterInfo.from_dict(c.to_dict())
        assert restored.character_id == "Harry Potter"
        assert restored.quote_count == 42
        assert restored.gender_pronoun == "he"

    def test_defaults_on_from_dict(self):
        c = CharacterInfo.from_dict({"character_id": "Alice"})
        assert c.display_name == "Alice"
        assert c.quote_count == 0
        assert c.gender_pronoun == ""


class TestNLPResult:
    def _make_result(self):
        ch = Chapter(
            index=0,
            title="Chapter One",
            paragraphs=["Para."],
            segments=[Segment("Para.", "NARRATOR", 0)],
        )
        ci = CharacterInfo(
            character_id="Alice Wonderland",
            display_name="Alice Wonderland",
            quote_count=5,
        )
        return NLPResult(characters=[ci], chapters=[ch], book_hash="abc123")

    def test_round_trip(self):
        result = self._make_result()
        restored = NLPResult.from_dict(result.to_dict())
        assert restored.book_hash == "abc123"
        assert len(restored.characters) == 1
        assert restored.characters[0].character_id == "Alice Wonderland"
        assert len(restored.chapters) == 1
        assert restored.chapters[0].title == "Chapter One"

    def test_segments_preserved_through_round_trip(self):
        result = self._make_result()
        restored = NLPResult.from_dict(result.to_dict())
        assert restored.chapters[0].segments is not None
        assert restored.chapters[0].segments[0].speaker == "NARRATOR"

    def test_empty_result(self):
        empty = NLPResult(characters=[], chapters=[], book_hash="")
        restored = NLPResult.from_dict(empty.to_dict())
        assert restored.characters == []
        assert restored.chapters == []
