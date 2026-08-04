"""Tests for kenkui parsing functionality."""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from kenkui.chapter_filter import FilterOperation
from kenkui.models import Chapter, ProcessingConfig
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


class TestPdfTranscriptOutput:
    def _make_builder(self, tmp_path: Path):
        from kenkui.parsing import AudioBuilder

        cfg = ProcessingConfig(
            voice="alba",
            ebook_path=tmp_path / "book.pdf",
            output_path=tmp_path / "out",
            pause_line_ms=400,
            pause_chapter_ms=2000,
            workers=2,
            m4b_bitrate="96k",
            keep_temp=False,
            debug_html=False,
            chapter_filters=[FilterOperation("preset", "content-only")],
        )
        return AudioBuilder(cfg)

    def test_pdf_transcripts_are_written_to_output_dir(self, tmp_path):
        builder = self._make_builder(tmp_path)
        emitted: list[str] = []
        builder.console = SimpleNamespace(emit=lambda msg, style="": emitted.append(msg))
        builder._reader = SimpleNamespace(
            get_transcript_sections=lambda: [
                SimpleNamespace(
                    title="Chapter 1",
                    start_page=0,
                    end_page=0,
                    raw_paragraphs=["Raw paragraph."],
                    filtered_paragraphs=["Filtered paragraph."],
                )
            ]
        )
        chapters = [Chapter(index=0, title="Chapter 1", paragraphs=["Filtered paragraph."])]

        builder._write_pdf_transcripts(tmp_path / "out" / "book.m4b", chapters)

        raw_path = tmp_path / "out" / "book.transcript.raw.txt"
        filtered_path = tmp_path / "out" / "book.transcript.filtered.txt"
        assert raw_path.exists()
        assert filtered_path.exists()
        assert "Raw paragraph." in raw_path.read_text(encoding="utf-8")
        assert "Filtered paragraph." in filtered_path.read_text(encoding="utf-8")
        assert emitted
        assert str(tmp_path / "out") in emitted[-1]


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

    def test_non_roster_speaker_added_to_roster_when_roster_available(self, tmp_path):
        """Annotated cache speakers missing from the roster remain assignable."""
        from kenkui.models import CharacterInfo, FastScanResult
        from kenkui.nlp.models import CharacterRecord, CharacterRoster
        from kenkui.parsing import _load_annotated_chapters

        cache_path = self._make_cache(
            tmp_path,
            [{"text": "Ahoy.", "speaker": "fisherman", "index": 7, "is_scene_break": False}],
        )
        roster = CharacterRoster(characters=[
            CharacterRecord(slug="darrow", canonical_name="Darrow", aliases=["Darrow"])
        ])
        roster_cache = tmp_path / "roster.json"
        roster_cache.write_text(
            json.dumps({
                "roster_data": FastScanResult(
                    roster=roster,
                    characters=[CharacterInfo(character_id="darrow", display_name="Darrow")],
                    book_hash="hash",
                ).to_dict()
            }),
            encoding="utf-8",
        )

        chapters = _load_annotated_chapters(cache_path, [], roster_cache)
        segs = {s.index: s for s in chapters[0].segments}

        assert segs[0].speaker == "darrow"
        assert segs[7].speaker == "fisherman"

        updated = json.loads(roster_cache.read_text(encoding="utf-8"))
        result = FastScanResult.from_dict(updated["roster_data"])
        absorbed = result.roster.by_slug("fisherman")
        assert absorbed is not None
        assert absorbed.canonical_name == "Fisherman"
        assert any(c.character_id == "fisherman" for c in result.characters)


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


class TestAutoAssignUnmappedSpeakers:
    def _make_chapter(self, segments):
        from kenkui.models import Chapter, Segment

        segs = [
            Segment(text=s["text"], speaker=s["speaker"], index=i,
                    is_scene_break=s.get("is_scene_break", False))
            for i, s in enumerate(segments)
        ]
        ch = Chapter(index=0, title="Ch 1", paragraphs=[])
        ch.segments = segs
        return ch

    def test_auto_assigns_only_valid_roster_speakers(self):
        from unittest.mock import patch

        from kenkui.models import Chapter, Segment
        from kenkui.parsing import _auto_assign_unmapped_speakers

        chapter = Chapter(
            index=0,
            title="Ch 1",
            paragraphs=[],
            segments=[
                Segment(text="Named.", speaker="darrow", index=0),
                Segment(text="Role.", speaker="fisherman", index=1),
            ],
        )

        with patch(
            "kenkui.services.voice_service.list_voices",
            return_value=[
                SimpleNamespace(voice_id="alba", pool_enabled=True, status="available", gender="female"),
                SimpleNamespace(voice_id="cedar", pool_enabled=True, status="available", gender="male"),
            ],
        ):
            assigned = _auto_assign_unmapped_speakers(
                [chapter],
                {},
                "alba",
                lambda _msg: None,
                roster_slugs={"darrow"},
            )

        assert "darrow" in assigned
        assert "fisherman" not in assigned

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


class TestRosterCacheLoaders:
    """Tests for _load_character_genders and _load_roster_slugs error paths."""

    def test_load_character_genders_bad_json_logs_warning(self, tmp_path, caplog):
        """Corrupt JSON in roster cache logs a warning and returns empty dict."""
        import logging

        from kenkui.parsing import _load_character_genders

        bad_file = tmp_path / "roster.json"
        bad_file.write_text("not-valid-json", encoding="utf-8")

        with caplog.at_level(logging.WARNING, logger="kenkui.parsing"):
            result = _load_character_genders(bad_file)

        assert result == {}
        assert any("character genders" in r.message for r in caplog.records)

    def test_load_roster_slugs_missing_file_logs_warning(self, tmp_path, caplog):
        """Missing roster cache file logs a warning and returns empty set."""
        import logging

        from kenkui.parsing import _load_roster_slugs

        missing = tmp_path / "nonexistent.json"

        with caplog.at_level(logging.WARNING, logger="kenkui.parsing"):
            result = _load_roster_slugs(missing)

        assert result == set()
        assert any("roster slugs" in r.message for r in caplog.records)

    def test_load_character_genders_none_path_returns_empty(self):
        """None path returns empty dict without logging."""
        from kenkui.parsing import _load_character_genders

        assert _load_character_genders(None) == {}

    def test_load_roster_slugs_none_path_returns_empty(self):
        """None path returns empty set without logging."""
        from kenkui.parsing import _load_roster_slugs

        assert _load_roster_slugs(None) == set()
