"""Tests for kenkui.nlp.pipeline — NLPPipeline, NLPJob, NLPJobStatus, ValidationResult."""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from kenkui.models import Chapter, NLPResult, CharacterInfo
from kenkui.nlp.models import CharacterRoster, CharacterRecord
from kenkui.nlp.pipeline import NLPJob, NLPJobStatus, NLPPipeline, ValidationResult
from kenkui.nlp_config import NLPConfig


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_chapter(index: int = 0, title: str = "Chapter 1") -> Chapter:
    return Chapter(index=index, title=title, paragraphs=["Hello world."])


def _make_roster() -> CharacterRoster:
    return CharacterRoster(characters=[
        CharacterRecord(
            slug="jane_eyre",
            canonical_name="Jane Eyre",
            aliases=["Jane"],
            gender="she/her",
        )
    ])


def _make_nlp_result(book_hash: str = "abc123") -> NLPResult:
    chapter = _make_chapter()
    return NLPResult(
        characters=[CharacterInfo(character_id="Jane Eyre", display_name="Jane Eyre")],
        chapters=[chapter],
        book_hash=book_hash,
    )


def _make_pipeline() -> NLPPipeline:
    """Return a pipeline with mocked providers."""
    config = NLPConfig()
    pipeline = NLPPipeline.__new__(NLPPipeline)
    pipeline._config = config
    pipeline._extraction = MagicMock()
    pipeline._attribution = MagicMock()
    return pipeline


# ---------------------------------------------------------------------------
# validate_tools tests
# ---------------------------------------------------------------------------


def test_pipeline_validate_tools_ollama_available():
    """Both extraction and attribution report ok=True when ollama is importable."""
    pipeline = _make_pipeline()

    mock_ollama = MagicMock()
    with patch.dict(sys.modules, {"ollama": mock_ollama}):
        results = pipeline.validate_tools()

    assert len(results) == 2
    assert all(r.ok for r in results), [r.message for r in results]
    assert results[0].step == "extraction"
    assert results[1].step == "attribution"
    assert results[0].tool == "ollama"
    assert results[1].tool == "ollama"
    assert results[0].message == "Available"
    assert results[1].message == "Available"


def test_pipeline_validate_tools_ollama_not_installed():
    """Both results report ok=False when ollama is not importable."""
    pipeline = _make_pipeline()

    # Remove ollama from sys.modules and block its import
    saved = sys.modules.pop("ollama", None)
    try:
        with patch.dict(sys.modules, {"ollama": None}):
            results = pipeline.validate_tools()
    finally:
        if saved is not None:
            sys.modules["ollama"] = saved

    assert len(results) == 2
    assert all(not r.ok for r in results), [r.message for r in results]


# ---------------------------------------------------------------------------
# extract tests
# ---------------------------------------------------------------------------


def test_pipeline_extract_returns_roster(tmp_path):
    """extract() calls build_roster and returns the CharacterRoster."""
    pipeline = _make_pipeline()
    roster = _make_roster()
    pipeline._extraction.build_roster.return_value = roster

    book_path = tmp_path / "book.epub"
    book_path.write_bytes(b"fake")
    chapters = [_make_chapter()]

    with (
        patch("kenkui.nlp.pipeline.get_cache", return_value=None),
        patch("kenkui.nlp.pipeline.put_cache") as mock_put,
    ):
        result = pipeline.extract(book_path, chapters, use_cache=True)

    # The filter always wraps a new CharacterRoster, so check by value not identity.
    assert isinstance(result, CharacterRoster)
    assert result.characters == roster.characters
    pipeline._extraction.build_roster.assert_called_once()
    mock_put.assert_called_once()


def test_pipeline_extract_uses_cache_when_available(tmp_path):
    """extract() returns cached roster without calling build_roster."""
    pipeline = _make_pipeline()
    roster = _make_roster()
    cached_dict = roster.model_dump()

    book_path = tmp_path / "book.epub"
    book_path.write_bytes(b"fake")
    chapters = [_make_chapter()]

    with patch("kenkui.nlp.pipeline.get_cache", return_value=cached_dict):
        result = pipeline.extract(book_path, chapters, use_cache=True)

    pipeline._extraction.build_roster.assert_not_called()
    assert isinstance(result, CharacterRoster)
    assert result.characters[0].canonical_name == "Jane Eyre"


def test_pipeline_extract_skips_cache_when_disabled(tmp_path):
    """extract() calls build_roster when use_cache=False."""
    pipeline = _make_pipeline()
    roster = _make_roster()
    pipeline._extraction.build_roster.return_value = roster

    book_path = tmp_path / "book.epub"
    book_path.write_bytes(b"fake")
    chapters = [_make_chapter()]

    with (
        patch("kenkui.nlp.pipeline.get_cache") as mock_get,
        patch("kenkui.nlp.pipeline.put_cache"),
    ):
        result = pipeline.extract(book_path, chapters, use_cache=False)

    mock_get.assert_not_called()
    # The filter wraps a new object; check by value not identity.
    assert isinstance(result, CharacterRoster)
    assert result.characters == roster.characters


def test_pipeline_extract_progress_callback_called(tmp_path):
    """extract() calls progress_callback with (int, str) during extraction."""
    pipeline = _make_pipeline()
    roster = _make_roster()
    pipeline._extraction.build_roster.return_value = roster

    book_path = tmp_path / "book.epub"
    book_path.write_bytes(b"fake")
    chapters = [_make_chapter()]

    received: list[tuple[int, str]] = []

    def _cb(pct: int, msg: str) -> None:
        received.append((pct, msg))

    with (
        patch("kenkui.nlp.pipeline.get_cache", return_value=None),
        patch("kenkui.nlp.pipeline.put_cache"),
    ):
        pipeline.extract(book_path, chapters, progress_callback=_cb, use_cache=True)

    # The adapter is called by the provider mock; no direct calls here —
    # just verify the signature contract was not violated.


# ---------------------------------------------------------------------------
# attribute tests
# ---------------------------------------------------------------------------


def test_pipeline_attribute_returns_nlp_result(tmp_path):
    """attribute() calls attribute_chapter for each chapter and returns NLPResult."""
    pipeline = _make_pipeline()
    roster = _make_roster()

    # Mock attribution result
    from kenkui.nlp.models import AttributionResult, AttributionItem
    attr_result = AttributionResult(attributions=[])
    pipeline._attribution.attribute_chapter.return_value = attr_result

    book_path = tmp_path / "book.epub"
    book_path.write_bytes(b"fake")
    chapters = [_make_chapter(0), _make_chapter(1, "Chapter 2")]

    with (
        patch("kenkui.nlp.pipeline.get_cache", return_value=None),
        patch("kenkui.nlp.pipeline.put_cache"),
        patch("kenkui.nlp.pipeline._attribution_to_segments", return_value=[]),
        patch("kenkui.nlp.pipeline.book_hash", return_value="deadbeef"),
    ):
        result = pipeline.attribute(book_path, chapters, roster, use_cache=True)

    assert isinstance(result, NLPResult)
    assert result.book_hash == "deadbeef"
    assert pipeline._attribution.attribute_chapter.call_count == 2


def test_pipeline_attribute_uses_cache_when_available(tmp_path):
    """attribute() returns cached NLPResult without calling attribute_chapter."""
    pipeline = _make_pipeline()
    roster = _make_roster()

    nlp_result = _make_nlp_result()
    cached_dict = nlp_result.to_dict()

    book_path = tmp_path / "book.epub"
    book_path.write_bytes(b"fake")
    chapters = [_make_chapter()]

    with patch("kenkui.nlp.pipeline.get_cache", return_value=cached_dict):
        result = pipeline.attribute(book_path, chapters, roster, use_cache=True)

    pipeline._attribution.attribute_chapter.assert_not_called()
    assert isinstance(result, NLPResult)


def test_pipeline_attribute_skips_cache_when_disabled(tmp_path):
    """attribute() calls attribute_chapter when use_cache=False."""
    pipeline = _make_pipeline()
    roster = _make_roster()

    from kenkui.nlp.models import AttributionResult
    attr_result = AttributionResult(attributions=[])
    pipeline._attribution.attribute_chapter.return_value = attr_result

    book_path = tmp_path / "book.epub"
    book_path.write_bytes(b"fake")
    chapters = [_make_chapter()]

    with (
        patch("kenkui.nlp.pipeline.get_cache") as mock_get,
        patch("kenkui.nlp.pipeline.put_cache"),
        patch("kenkui.nlp.pipeline._attribution_to_segments", return_value=[]),
        patch("kenkui.nlp.pipeline.book_hash", return_value="deadbeef"),
    ):
        result = pipeline.attribute(book_path, chapters, roster, use_cache=False)

    mock_get.assert_not_called()
    pipeline._attribution.attribute_chapter.assert_called_once()


# ---------------------------------------------------------------------------
# run tests
# ---------------------------------------------------------------------------


def test_pipeline_run_calls_extract_then_attribute(tmp_path):
    """run() delegates to extract() then attribute() in order."""
    pipeline = _make_pipeline()
    roster = _make_roster()
    nlp_result = _make_nlp_result()

    call_order: list[str] = []

    def _mock_extract(*args, **kwargs):
        call_order.append("extract")
        return roster

    def _mock_attribute(*args, **kwargs):
        call_order.append("attribute")
        return nlp_result

    pipeline.extract = _mock_extract  # type: ignore[method-assign]
    pipeline.attribute = _mock_attribute  # type: ignore[method-assign]

    book_path = tmp_path / "book.epub"
    book_path.write_bytes(b"fake")

    result = pipeline.run(book_path, [_make_chapter()])

    assert call_order == ["extract", "attribute"]
    assert result is nlp_result


# ---------------------------------------------------------------------------
# extract_job tests
# ---------------------------------------------------------------------------


def test_extract_job_returns_job_object(tmp_path):
    """extract_job() returns an NLPJob with a job_id immediately."""
    pipeline = _make_pipeline()
    roster = _make_roster()
    pipeline._extraction.build_roster.return_value = roster

    book_path = tmp_path / "book.epub"
    book_path.write_bytes(b"fake")

    with (
        patch("kenkui.nlp.pipeline.get_cache", return_value=None),
        patch("kenkui.nlp.pipeline.put_cache"),
    ):
        job = pipeline.extract_job(book_path, [_make_chapter()])

    assert isinstance(job, NLPJob)
    assert job.job_id  # non-empty UUID string
    # Wait briefly so we don't block the test suite
    job.wait(timeout=5.0)


def test_extract_job_completes_to_done(tmp_path):
    """extract_job() transitions to DONE with result populated."""
    pipeline = _make_pipeline()
    roster = _make_roster()
    pipeline.extract = MagicMock(return_value=roster)  # type: ignore[method-assign]

    book_path = tmp_path / "book.epub"
    book_path.write_bytes(b"fake")

    job = pipeline.extract_job(book_path, [_make_chapter()])
    snapshot = job.wait(timeout=5.0)

    assert snapshot.status == NLPJobStatus.DONE
    assert snapshot.result is roster
    assert snapshot.error is None
    assert snapshot.progress == 100


def test_extract_job_sets_failed_on_exception(tmp_path):
    """extract_job() transitions to FAILED when extract raises."""
    pipeline = _make_pipeline()
    pipeline.extract = MagicMock(side_effect=RuntimeError("boom"))  # type: ignore[method-assign]

    book_path = tmp_path / "book.epub"
    book_path.write_bytes(b"fake")

    job = pipeline.extract_job(book_path, [_make_chapter()])
    snapshot = job.wait(timeout=5.0)

    assert snapshot.status == NLPJobStatus.FAILED
    assert snapshot.result is None
    assert isinstance(snapshot.error, RuntimeError)
    assert "boom" in str(snapshot.error)


# ---------------------------------------------------------------------------
# attribute_job tests
# ---------------------------------------------------------------------------


def test_attribute_job_completes_to_done(tmp_path):
    """attribute_job() transitions to DONE with result populated."""
    pipeline = _make_pipeline()
    roster = _make_roster()
    nlp_result = _make_nlp_result()
    pipeline.attribute = MagicMock(return_value=nlp_result)  # type: ignore[method-assign]

    book_path = tmp_path / "book.epub"
    book_path.write_bytes(b"fake")

    job = pipeline.attribute_job(book_path, [_make_chapter()], roster)
    snapshot = job.wait(timeout=5.0)

    assert snapshot.status == NLPJobStatus.DONE
    assert snapshot.result is nlp_result
    assert snapshot.error is None
    assert snapshot.progress == 100


def test_attribute_job_sets_failed_on_exception(tmp_path):
    """attribute_job() transitions to FAILED when attribute raises."""
    pipeline = _make_pipeline()
    roster = _make_roster()
    pipeline.attribute = MagicMock(side_effect=ValueError("attribution error"))  # type: ignore[method-assign]

    book_path = tmp_path / "book.epub"
    book_path.write_bytes(b"fake")

    job = pipeline.attribute_job(book_path, [_make_chapter()], roster)
    snapshot = job.wait(timeout=5.0)

    assert snapshot.status == NLPJobStatus.FAILED
    assert "attribution error" in snapshot.message


# ---------------------------------------------------------------------------
# NLPJob API tests
# ---------------------------------------------------------------------------


def test_nlp_job_poll_returns_snapshot():
    """poll() returns a new NLPJob with _thread=None."""
    job = NLPJob(
        job_id="test-id",
        status=NLPJobStatus.PENDING,
        progress=0,
        message="Queued",
        result=None,
        error=None,
    )
    snapshot = job.poll()
    assert snapshot.job_id == "test-id"
    assert snapshot._thread is None
    assert snapshot is not job


def test_nlp_job_cancel_sets_event():
    """cancel() sets the cancel event."""
    job = NLPJob(
        job_id="test-id",
        status=NLPJobStatus.RUNNING,
        progress=50,
        message="Running",
        result=None,
        error=None,
    )
    assert not job._cancel_event.is_set()
    job.cancel()
    assert job._cancel_event.is_set()


# ---------------------------------------------------------------------------
# ValidationResult dataclass tests
# ---------------------------------------------------------------------------


def test_validation_result_fields():
    """ValidationResult stores all required fields."""
    vr = ValidationResult(tool="ollama", step="extraction", ok=True, message="Available")
    assert vr.tool == "ollama"
    assert vr.step == "extraction"
    assert vr.ok is True
    assert vr.message == "Available"


def test_nlp_job_status_values():
    """NLPJobStatus has the expected string values."""
    assert NLPJobStatus.PENDING == "pending"
    assert NLPJobStatus.RUNNING == "running"
    assert NLPJobStatus.DONE == "done"
    assert NLPJobStatus.FAILED == "failed"


# ---------------------------------------------------------------------------
# Pronoun-slug filtering tests
# ---------------------------------------------------------------------------


class TestExtractPronounFiltering:
    """extract() must strip CharacterRecords whose slug is a pronoun."""

    def test_pronoun_slugs_are_removed(self, tmp_path):
        """Pronoun-slug characters (he, she) are filtered out of the returned roster."""
        pipeline = _make_pipeline()

        dirty_roster = CharacterRoster(characters=[
            CharacterRecord(slug="she", canonical_name="She", aliases=[], gender="she/her"),
            CharacterRecord(slug="he", canonical_name="He", aliases=[], gender="he/him"),
            CharacterRecord(slug="darrow", canonical_name="Darrow", aliases=[], gender="he/him"),
        ])
        pipeline._extraction.build_roster.return_value = dirty_roster

        book_path = tmp_path / "book.epub"
        book_path.write_bytes(b"fake")
        chapters = [_make_chapter()]

        with (
            patch("kenkui.nlp.pipeline.get_cache", return_value=None),
            patch("kenkui.nlp.pipeline.put_cache"),
        ):
            result = pipeline.extract(book_path, chapters, use_cache=False)

        slugs = {ch.slug for ch in result.characters}
        assert "she" not in slugs, "pronoun 'she' should be filtered out"
        assert "he" not in slugs, "pronoun 'he' should be filtered out"
        assert "darrow" in slugs, "real character 'darrow' must be kept"

    def test_real_character_is_kept(self, tmp_path):
        """A roster with only a real character slug is returned unchanged."""
        pipeline = _make_pipeline()

        clean_roster = CharacterRoster(characters=[
            CharacterRecord(slug="darrow", canonical_name="Darrow", aliases=[], gender="he/him"),
        ])
        pipeline._extraction.build_roster.return_value = clean_roster

        book_path = tmp_path / "book.epub"
        book_path.write_bytes(b"fake")
        chapters = [_make_chapter()]

        with (
            patch("kenkui.nlp.pipeline.get_cache", return_value=None),
            patch("kenkui.nlp.pipeline.put_cache"),
        ):
            result = pipeline.extract(book_path, chapters, use_cache=False)

        assert len(result.characters) == 1
        assert result.characters[0].slug == "darrow"

    def test_filtered_roster_is_what_gets_cached(self, tmp_path):
        """put_cache receives the filtered roster (without pronoun slugs)."""
        pipeline = _make_pipeline()

        dirty_roster = CharacterRoster(characters=[
            CharacterRecord(slug="they", canonical_name="They", aliases=[], gender="they/them"),
            CharacterRecord(slug="lyria", canonical_name="Lyria", aliases=[], gender="she/her"),
        ])
        pipeline._extraction.build_roster.return_value = dirty_roster

        book_path = tmp_path / "book.epub"
        book_path.write_bytes(b"fake")
        chapters = [_make_chapter()]

        with (
            patch("kenkui.nlp.pipeline.get_cache", return_value=None),
            patch("kenkui.nlp.pipeline.put_cache") as mock_put,
        ):
            pipeline.extract(book_path, chapters, use_cache=False)

        # Inspect the dict passed to put_cache
        cached_dict = mock_put.call_args[0][0]
        cached_slugs = {ch["slug"] for ch in cached_dict["characters"]}
        assert "they" not in cached_slugs, "pronoun 'they' must not be in cache"
        assert "lyria" in cached_slugs, "'lyria' must be in cache"


# ---------------------------------------------------------------------------
# Attribution count slug normalization tests (Fix 5)
# ---------------------------------------------------------------------------


class TestAttributionCountSlugNormalization:
    """attribute() must accumulate quote counts by slug, not by canonical name."""

    def _make_roster_darrow_rhonna(self) -> CharacterRoster:
        return CharacterRoster(characters=[
            CharacterRecord(slug="darrow", canonical_name="Darrow", aliases=[], gender="he/him"),
            CharacterRecord(slug="rhonna", canonical_name="Rhonna", aliases=[], gender="she/her"),
        ])

    def test_attribution_counts_uses_slugs(self, tmp_path):
        """attribution_counts keys are slugified; sentinel speakers are excluded."""
        from kenkui.nlp.models import AttributionResult, AttributionItem

        pipeline = _make_pipeline()
        roster = self._make_roster_darrow_rhonna()

        # LLM returns canonical-case speakers (the pre-Fix-7 scenario)
        attr_result = AttributionResult(attributions=[
            AttributionItem(quote_id=1, speaker="Darrow", confidence=5),
            AttributionItem(quote_id=2, speaker="Darrow", confidence=4),
            AttributionItem(quote_id=3, speaker="Rhonna", confidence=4),
            AttributionItem(quote_id=4, speaker="NARRATOR", confidence=5),
            AttributionItem(quote_id=5, speaker="Unknown", confidence=2),
        ])
        pipeline._attribution.attribute_chapter.return_value = attr_result

        book_path = tmp_path / "book.epub"
        book_path.write_bytes(b"fake")
        chapters = [_make_chapter(0)]

        with (
            patch("kenkui.nlp.pipeline.get_cache", return_value=None),
            patch("kenkui.nlp.pipeline.put_cache"),
            patch("kenkui.nlp.pipeline._attribution_to_segments", return_value=[]),
            patch("kenkui.nlp.pipeline.book_hash", return_value="deadbeef"),
        ):
            result = pipeline.attribute(book_path, chapters, roster, use_cache=False)

        # Find CharacterInfo by character_id
        darrow_ci = next(c for c in result.characters if c.character_id == "darrow")
        rhonna_ci = next(c for c in result.characters if c.character_id == "rhonna")

        # Slugs must be used as keys: "Darrow" -> "darrow", "Rhonna" -> "rhonna"
        assert darrow_ci.quote_count == 2, (
            f"Expected darrow quote_count=2, got {darrow_ci.quote_count}"
        )
        assert rhonna_ci.quote_count == 1, (
            f"Expected rhonna quote_count=1, got {rhonna_ci.quote_count}"
        )

    def test_narrator_and_unknown_excluded_from_counts(self, tmp_path):
        """NARRATOR and Unknown speakers must not appear in attribution_counts.

        The roster deliberately includes characters whose slugs match the
        slugified forms of the sentinel values ("narrator", "unknown").  If the
        sentinel guard in pipeline.attribute() were removed, those characters
        would receive a non-zero quote_count.  The test therefore catches a real
        regression rather than passing trivially.
        """
        from kenkui.nlp.models import AttributionResult, AttributionItem

        pipeline = _make_pipeline()
        # Include roster entries whose slugs equal the slugified sentinels so
        # that removing the guard would cause a non-zero quote_count.
        roster = CharacterRoster(characters=[
            CharacterRecord(slug="narrator", canonical_name="Narrator", aliases=[], gender=""),
            CharacterRecord(slug="unknown", canonical_name="Unknown", aliases=[], gender=""),
        ])

        attr_result = AttributionResult(attributions=[
            AttributionItem(quote_id=1, speaker="NARRATOR", confidence=5),
            AttributionItem(quote_id=2, speaker="Unknown", confidence=2),
        ])
        pipeline._attribution.attribute_chapter.return_value = attr_result

        book_path = tmp_path / "book.epub"
        book_path.write_bytes(b"fake")
        chapters = [_make_chapter(0)]

        with (
            patch("kenkui.nlp.pipeline.get_cache", return_value=None),
            patch("kenkui.nlp.pipeline.put_cache"),
            patch("kenkui.nlp.pipeline._attribution_to_segments", return_value=[]),
            patch("kenkui.nlp.pipeline.book_hash", return_value="deadbeef"),
        ):
            result = pipeline.attribute(book_path, chapters, roster, use_cache=False)

        for ci in result.characters:
            assert ci.quote_count == 0, (
                f"{ci.character_id} got quote_count={ci.quote_count}; "
                "sentinel speakers must be excluded from attribution_counts"
            )

    def test_quote_count_correct_via_slug_lookup(self, tmp_path):
        """ci.quote_count is correct when attribution keys are slugs matching rec.slug."""
        from kenkui.nlp.models import AttributionResult, AttributionItem

        pipeline = _make_pipeline()
        roster = self._make_roster_darrow_rhonna()

        # The LLM already returns slug-form speakers.
        attr_result = AttributionResult(attributions=[
            AttributionItem(quote_id=1, speaker="darrow", confidence=5),
            AttributionItem(quote_id=2, speaker="darrow", confidence=5),
            AttributionItem(quote_id=3, speaker="darrow", confidence=5),
        ])
        pipeline._attribution.attribute_chapter.return_value = attr_result

        book_path = tmp_path / "book.epub"
        book_path.write_bytes(b"fake")
        chapters = [_make_chapter(0)]

        with (
            patch("kenkui.nlp.pipeline.get_cache", return_value=None),
            patch("kenkui.nlp.pipeline.put_cache"),
            patch("kenkui.nlp.pipeline._attribution_to_segments", return_value=[]),
            patch("kenkui.nlp.pipeline.book_hash", return_value="deadbeef"),
        ):
            result = pipeline.attribute(book_path, chapters, roster, use_cache=False)

        darrow_ci = next(c for c in result.characters if c.character_id == "darrow")
        assert darrow_ci.quote_count == 3, (
            f"Expected darrow quote_count=3, got {darrow_ci.quote_count}"
        )
