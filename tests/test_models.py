"""Tests for kenkui.models — AppConfig, Segment, ChapterSelection round-trips."""

from __future__ import annotations

from pathlib import Path

from kenkui.models import (
    AppConfig,
    Chapter,
    ChapterPreset,
    ChapterSelection,
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

    def test_new_fields_backward_compatible(self):
        """Old YAML files without the new fields should load cleanly."""
        old_data = {
            "name": "legacy",
            "workers": 4,
            "m4b_bitrate": "64k",
            # No default_voice, default_chapter_preset, default_output_dir
        }
        cfg = AppConfig.from_dict(old_data)
        assert cfg.default_voice == "alba"
        assert cfg.default_chapter_preset == "content-only"
        assert cfg.default_output_dir is None


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
