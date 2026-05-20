"""Tests for kenkui parsing functionality."""

import json
from pathlib import Path

import pytest

from kenkui.models import Chapter
from kenkui.readers.epub import EpubReader

TEST_EPUB = Path("src/kenkui/samples/Les Miserables - Victor Hugo.epub")


class TestEpubReader:
    """Tests for the EpubReader class."""

    @pytest.fixture
    def reader(self):
        """Create an EpubReader instance for testing."""
        return EpubReader(TEST_EPUB)

    def test_reader_initialization(self, reader):
        """Test that EpubReader can be initialized."""
        assert reader is not None
        assert reader.filepath == TEST_EPUB

    def test_get_book_title(self, reader):
        """Test extracting book title from EPUB."""
        title = reader.get_metadata().title
        assert isinstance(title, str)
        assert len(title) > 0
        # Title should be sanitized (no special chars)
        assert "<" not in title
        assert ">" not in title

    def test_extract_chapters_returns_list(self, reader):
        """Test that get_chapters returns a list."""
        chapters = reader.get_chapters()
        assert isinstance(chapters, list)

    def test_extract_chapters_returns_chapter_objects(self, reader):
        """Test that get_chapters returns Chapter objects."""
        chapters = reader.get_chapters()
        if chapters:  # Only test if chapters were found
            for chapter in chapters:
                assert isinstance(chapter, Chapter)
                assert hasattr(chapter, "index")
                assert hasattr(chapter, "title")
                assert hasattr(chapter, "paragraphs")
                assert isinstance(chapter.index, int)
                assert isinstance(chapter.title, str)
                assert isinstance(chapter.paragraphs, list)


class TestChapterDataclass:
    """Tests for the Chapter dataclass."""

    def test_chapter_creation(self):
        """Test creating a Chapter instance."""
        chapter = Chapter(
            index=1, title="Test Chapter", paragraphs=["Paragraph 1", "Paragraph 2"]
        )
        assert chapter.index == 1
        assert chapter.title == "Test Chapter"
        assert len(chapter.paragraphs) == 2

    def test_chapter_with_empty_paragraphs(self):
        """Test creating a Chapter with empty paragraphs."""
        chapter = Chapter(index=2, title="Empty Chapter", paragraphs=[])
        assert chapter.index == 2
        assert chapter.paragraphs == []


class TestLoadAnnotatedChaptersSpeakerSlugNormalization:
    """Test that _load_annotated_chapters normalizes speaker strings to slugs at read time."""

    def _make_cache(self, tmp_path: Path, extra_segments: list | None = None) -> Path:
        base_segments = [
            {"text": "He said something.", "speaker": "Darrow", "index": 0, "is_scene_break": False},
            {"text": "She replied.", "speaker": "Rhonna", "index": 1, "is_scene_break": False},
            {"text": "Narration.", "speaker": "NARRATOR", "index": 2, "is_scene_break": False},
            {"text": "Unknown speaker.", "speaker": "Unknown", "index": 3, "is_scene_break": False},
            {"text": "Mixed Case Name.", "speaker": "Sevro Au Barca", "index": 4, "is_scene_break": False},
        ]
        cache_data = {
            "chapters": [
                {
                    "index": 0,
                    "title": "Chapter 1",
                    "paragraphs": [],
                    "toc_index": 0,
                    "segments": base_segments + (extra_segments or []),
                }
            ]
        }
        cache_file = tmp_path / "nlp_cache.json"
        cache_file.write_text(json.dumps(cache_data), encoding="utf-8")
        return cache_file

    def test_slug_normalization_speakers(self, tmp_path):
        """Non-sentinel speakers are slugified; NARRATOR and Unknown are preserved."""
        from kenkui.parsing import _load_annotated_chapters

        cache_path = self._make_cache(tmp_path)
        chapters = _load_annotated_chapters(cache_path, [])

        assert len(chapters) == 1
        segs = {s.index: s for s in chapters[0].segments}

        # Non-sentinels must be slugified
        assert segs[0].speaker == "darrow"
        assert segs[1].speaker == "rhonna"
        assert segs[4].speaker == "sevro_au_barca"

        # Sentinel values must be left unchanged
        assert segs[2].speaker == "NARRATOR"
        assert segs[3].speaker == "Unknown"

    def test_slug_normalization_null_speaker_does_not_crash(self, tmp_path):
        """A segment with speaker=null in the cache must not raise AttributeError."""
        from kenkui.parsing import _load_annotated_chapters

        extra = [{"text": "* * *", "speaker": None, "index": 5, "is_scene_break": False}]
        cache_path = self._make_cache(tmp_path, extra_segments=extra)
        chapters = _load_annotated_chapters(cache_path, [])

        segs = {s.index: s for s in chapters[0].segments}
        # Segment.from_dict yields None for an explicit null; the guard must leave it as-is.
        assert segs[5].speaker is None

    def test_slug_normalization_skips_scene_break_segments(self, tmp_path):
        """Scene-break segments are never slug-normalised, regardless of their speaker field."""
        from kenkui.parsing import _load_annotated_chapters

        extra = [{"text": "* * *", "speaker": "Some Name", "index": 6, "is_scene_break": True}]
        cache_path = self._make_cache(tmp_path, extra_segments=extra)
        chapters = _load_annotated_chapters(cache_path, [])

        segs = {s.index: s for s in chapters[0].segments}
        assert segs[6].speaker == "Some Name"


class TestWarnUnresolvableSpeakers:
    """Tests for _warn_unresolvable_speakers pre-flight helper."""

    def _make_chapter(self, segments):
        """Build a minimal Chapter with the given segments list."""
        from kenkui.models import Chapter, Segment

        segs = [
            Segment(text=s["text"], speaker=s["speaker"], index=i,
                    is_scene_break=s.get("is_scene_break", False))
            for i, s in enumerate(segments)
        ]
        ch = Chapter(index=0, title="Ch 1", paragraphs=[])
        ch.segments = segs
        return ch

    def test_no_mapping_emits_warning(self):
        """Speaker with no entry in speaker_voices → warning logged."""
        from kenkui.parsing import _warn_unresolvable_speakers

        warnings = []
        ch = self._make_chapter([{"text": "Hello.", "speaker": "darrow"}])
        _warn_unresolvable_speakers([ch], {}, warnings.append)

        assert len(warnings) == 1
        assert "darrow" in warnings[0]
        assert "no voice mapping" in warnings[0]

    def test_missing_safetensors_emits_warning(self):
        """Speaker mapped to a nonexistent .safetensors path → warning logged."""
        from unittest.mock import patch
        from kenkui.parsing import _warn_unresolvable_speakers

        warnings = []
        ch = self._make_chapter([{"text": "Hello.", "speaker": "darrow"}])
        fake_path = "/nonexistent/darrow.safetensors"
        speaker_voices = {"darrow": fake_path}

        with patch("kenkui.parsing.load_voice", return_value=fake_path):
            _warn_unresolvable_speakers([ch], speaker_voices, warnings.append)

        assert len(warnings) == 1
        assert "darrow" in warnings[0]
        assert "not found on disk" in warnings[0]

    def test_valid_non_safetensors_voice_no_warning(self):
        """Speaker mapped to a non-safetensors voice (built-in) → no warning."""
        from unittest.mock import patch
        from kenkui.parsing import _warn_unresolvable_speakers

        warnings = []
        ch = self._make_chapter([{"text": "Hello.", "speaker": "darrow"}])
        speaker_voices = {"darrow": "alba"}

        with patch("kenkui.parsing.load_voice", return_value="alba.pt"):
            _warn_unresolvable_speakers([ch], speaker_voices, warnings.append)

        assert warnings == []

    def test_sentinel_speakers_skipped(self):
        """NARRATOR and Unknown sentinels do not trigger warnings."""
        from kenkui.parsing import _warn_unresolvable_speakers

        warnings = []
        ch = self._make_chapter([
            {"text": "Narration.", "speaker": "NARRATOR"},
            {"text": "?", "speaker": "Unknown"},
        ])
        _warn_unresolvable_speakers([ch], {}, warnings.append)

        assert warnings == []

    def test_scene_break_segments_skipped(self):
        """Segments with is_scene_break=True are not checked."""
        from kenkui.parsing import _warn_unresolvable_speakers

        warnings = []
        ch = self._make_chapter([
            {"text": "* * *", "speaker": "darrow", "is_scene_break": True},
        ])
        _warn_unresolvable_speakers([ch], {}, warnings.append)

        assert warnings == []
