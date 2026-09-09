"""Contract tests for the WP2 public pipeline and value surface."""

from __future__ import annotations

import logging
from contextlib import contextmanager
from dataclasses import FrozenInstanceError
from pathlib import Path
from typing import TYPE_CHECKING, assert_type, cast, get_args

import pytest

import kenkui as kk
from kenkui._domain.casting import CharacterProfile
from kenkui._domain.operations import (
    AssignVoices,
    AttributeQuotes,
    InferCharacters,
    Pauses,
    Pronunciations,
    Series,
    SpokenForm,
    SynthesizeSpeech,
)
from kenkui._resolution import _log_ungendered_cast
from kenkui.voices.types import PerceivedGender, Voice

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator
    from typing import Any

EXPECTED_OPERATION_COUNT = 4
IO_ERROR_MESSAGE = "pipeline construction performed I/O"
PAUSE_CHAPTER_MS = 1500
PAUSE_HEADING_MS = 600
PAUSE_PARAGRAPH_MS = 250


def test_ordinary_functions_compose_with_fluent_methods() -> None:
    """Function arguments and return types survive the pipeline boundary."""

    def configure(
        pipeline: kk.Pipeline, narrator: str, *, paragraph_ms: int
    ) -> kk.Pipeline:
        return pipeline.assign_voice(narrator).pauses(paragraph_ms=paragraph_ms)

    def describe(pipeline: kk.Pipeline, *, prefix: str) -> str:
        return prefix + pipeline.source.path.name

    original = kk.epub("book.epub")
    configured = original.pipe(configure, "narrator", paragraph_ms=250).tts()
    assert_type(configured, kk.Pipeline)
    assert configured.operations == (
        original.assign_voice("narrator").pauses(paragraph_ms=250).tts().operations
    )
    assert original.operations == ()
    report = configured.pipe(describe, prefix="Selected: ")
    assert_type(report, str)
    assert report == "Selected: book.epub"


def test_construction_is_lazy_and_book_dispatches_without_reading(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Constructors and fluent methods record intent without filesystem effects."""
    original_exists = Path.exists
    original_read_bytes = Path.read_bytes

    def guarded_exists(path: Path) -> bool:
        if path == Path("unread.epub"):
            raise AssertionError(IO_ERROR_MESSAGE)
        return original_exists(path)

    def guarded_read_bytes(path: Path) -> bytes:
        if path == Path("unread.epub"):
            raise AssertionError(IO_ERROR_MESSAGE)
        return original_read_bytes(path)

    monkeypatch.setattr(Path, "exists", guarded_exists)
    monkeypatch.setattr(Path, "read_bytes", guarded_read_bytes)

    source = kk.book("unread.epub")
    pipeline = (
        source.select_chapters("chapter-1", "chapter-2")
        .assign_voice("narrator")
        .tts()
        .metadata(title=None, author="Writer", cover=None)
    )

    assert source.source.path == Path("unread.epub")
    assert source.source.format == "epub"
    assert source.operations == ()
    assert len(pipeline.operations) == EXPECTED_OPERATION_COUNT


def test_magic_run_writes_a_single_voice_book_beside_the_source(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Missing single-voice intent or a wrong default destination loses the render."""
    captured: list[tuple[Path, tuple[object, ...]]] = []
    expected = object()

    def capture_write(self: kk.Pipeline, output: str | Path) -> object:
        captured.append((Path(output), self.operations))
        return expected

    monkeypatch.setattr(kk.Pipeline, "write", capture_write)

    magic_run = getattr(kk, "magic_run", None)
    assert magic_run is not None
    assert magic_run("novel.epub", narrator="eponine") is expected

    assert captured == [
        (
            Path("novel.m4b"),
            (
                AssignVoices(
                    narrator_voice_id="eponine",
                    unknown_voice_id="eponine",
                    cast=(),
                    method="gendered",
                ),
                SynthesizeSpeech(),
            ),
        )
    ]


def test_magic_run_uses_the_default_openrouter_model_for_multi_voice(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Omitting multi-voice analysis would silently collapse character dialogue."""
    captured: list[tuple[Path, tuple[object, ...]]] = []

    def capture_write(self: kk.Pipeline, output: str | Path) -> object:
        captured.append((Path(output), self.operations))
        return object()

    monkeypatch.setattr(kk.Pipeline, "write", capture_write)

    magic_run = getattr(kk, "magic_run", None)
    assert magic_run is not None
    magic_run("novel.epub", narrator="eponine", multi=True)

    assert captured == [
        (
            Path("novel.m4b"),
            (
                InferCharacters("openrouter/deepseek/deepseek-v4-flash"),
                AttributeQuotes("openrouter/deepseek/deepseek-v4-flash"),
                AssignVoices(
                    narrator_voice_id="eponine",
                    unknown_voice_id="eponine",
                    cast=(),
                    method="gendered",
                ),
                SynthesizeSpeech(),
            ),
        )
    ]


def test_pipeline_is_frozen_branchable_and_operations_are_values() -> None:
    """Fluent calls return independent immutable operation chains."""
    root = kk.epub("book.epub")
    first = root.assign_voice("first")
    second = root.assign_voice("second")

    assert root.operations == ()
    assert first != second
    assert first.operations[:-1] == second.operations[:-1]
    assert first.operations is not second.operations
    with pytest.raises(FrozenInstanceError):
        first.source = root.source  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        first.operations[-1].narrator_voice_id = "x"  # type: ignore[misc,union-attr]


def test_duplicate_and_contradictory_operations_have_stable_codes() -> None:
    """Invalid semantic chains fail deterministically at their cheapest boundary."""
    with pytest.raises(kk.ValidationError) as duplicate:
        kk.epub("book.epub").select_chapters("a").select_chapters("b")
    assert duplicate.value.code == kk.ErrorCode.DUPLICATE_OPERATION

    with pytest.raises(kk.ValidationError) as missing_voice:
        kk.epub("book.epub").tts()
    assert missing_voice.value.code == kk.ErrorCode.VOICE_REQUIRED

    rendered = kk.epub("book.epub").assign_voice("voice").tts()
    with pytest.raises(kk.ValidationError) as ordering:
        rendered.select_chapters("chapter")
    assert ordering.value.code == kk.ErrorCode.INVALID_OPERATION_ORDER


def test_operation_arguments_are_validated_stably() -> None:
    """Invalid public arguments do not leak incidental Python errors."""
    cases: list[tuple[Callable[[], object], kk.ErrorCode]] = [
        (lambda: kk.epub("book.epub").select_chapters(), kk.ErrorCode.EMPTY_SELECTION),
        (
            lambda: kk.epub("book.epub").select_chapters("same", "same"),
            kk.ErrorCode.DUPLICATE_CHAPTER_ID,
        ),
        (lambda: kk.epub("book.epub").assign_voice("  "), kk.ErrorCode.INVALID_VOICE),
        (
            lambda: kk.epub("book.epub").metadata(cover=cast("Any", "remote")),
            kk.ErrorCode.INVALID_METADATA,
        ),
    ]
    for operation, code in cases:
        with pytest.raises(kk.ValidationError) as caught:
            operation()
        assert caught.value.code == code


def test_metadata_choices_are_semantic_and_immutable() -> None:
    """Inheritance, overrides, source-cover, and omitted-cover remain explicit."""
    root = kk.epub("book.epub")
    inherited = root.metadata()
    omitted = root.metadata(title="Title", author="Author", cover=None)

    assert inherited.metadata_intent == kk.MetadataIntent()
    assert omitted.metadata_intent == kk.MetadataIntent(
        title="Title", author="Author", cover=None
    )
    assert root.metadata_intent is None


def test_book_rejects_unsupported_formats_with_public_error() -> None:
    """Generic format dispatch rejects unsupported suffixes without touching source."""
    with pytest.raises(kk.SourceError) as caught:
        kk.book("book.pdf")
    assert caught.value.code == kk.ErrorCode.UNSUPPORTED_FORMAT
    assert "book.pdf" not in str(caught.value)


def test_validate_is_inexpensive_and_reports_stable_issues(tmp_path: Path) -> None:
    """Validation checks source and semantic requirements without parsing EPUB data."""
    missing = kk.epub(tmp_path / "missing.epub")
    invalid = missing.validate()

    assert not invalid.is_valid
    assert {issue.code for issue in invalid.issues} == {
        kk.ErrorCode.SOURCE_NOT_FOUND,
        kk.ErrorCode.VOICE_REQUIRED,
        kk.ErrorCode.TTS_REQUIRED,
    }
    existing = tmp_path / "book.epub"
    existing.write_bytes(b"not parsed in WP2")
    valid = kk.epub(existing).assign_voice("voice").tts().validate()
    assert valid == kk.ValidationResult()
    assert valid.is_valid


def test_inspect_rejects_malformed_source_and_write_is_stable_placeholder(
    tmp_path: Path,
) -> None:
    """Inspection is authoritative while later rendering remains unavailable."""
    source = tmp_path / "book.epub"
    source.write_bytes(b"placeholder")
    pipeline = kk.epub(source).assign_voice("voice").tts()

    with pytest.raises(kk.SourceError) as inspection:
        pipeline.inspect()
    assert inspection.value.code == kk.ErrorCode.MALFORMED_EPUB

    with pytest.raises(kk.RenderError) as rendering:
        pipeline.write(tmp_path / "book.m4b")
    assert rendering.value.code == kk.ErrorCode.RENDERER_UNAVAILABLE
    assert not (tmp_path / "book.m4b").exists()


def test_write_checks_semantics_controls_and_destination_before_placeholder(
    tmp_path: Path,
) -> None:
    """Write performs authoritative cheap checks and never alters an existing file."""
    source = tmp_path / "book.epub"
    source.write_bytes(b"placeholder")
    output = tmp_path / "book.m4b"
    output.write_bytes(b"keep")

    with pytest.raises(kk.ValidationError) as no_tts:
        kk.epub(source).assign_voice("voice").write(tmp_path / "new.m4b")
    assert no_tts.value.code == kk.ErrorCode.TTS_REQUIRED

    pipeline = kk.epub(source).assign_voice("voice").tts()
    with pytest.raises(kk.ValidationError) as bad_workers:
        pipeline.write(tmp_path / "new.m4b", workers=0)
    assert bad_workers.value.code == kk.ErrorCode.INVALID_WORKERS

    with pytest.raises(kk.EncodingError) as exists:
        pipeline.write(output)
    assert exists.value.code == kk.ErrorCode.OUTPUT_EXISTS
    assert output.read_bytes() == b"keep"

    with pytest.raises(kk.RenderError):
        pipeline.write(output, workers="auto", overwrite=True)
    assert output.read_bytes() == b"keep"


def test_public_values_events_errors_and_results_are_frozen() -> None:
    """Public DTOs expose typed immutable contracts before execution exists."""
    metadata = kk.BookMetadata(title="Title", author="Author", cover_available=True)
    inspection = kk.BookInspection(
        metadata=metadata,
        chapters=(kk.ChapterInspection("id", 0, "One", 12),),
    )
    voice = kk.Voice(
        id="voice",
        name="Voice",
        enabled=True,
        provenance=None,
        license_id="CC0-1.0",
        commercial_use_allowed=True,
    )
    stats = kk.ExecutionStats(12, 12, 1, 1, 100)
    result = kk.Result(Path("out.m4b"), stats)
    event = kk.StageProgress(2, "render", 1, 2, "id")
    error = kk.SourceError(kk.ErrorCode.SOURCE_NOT_FOUND)

    for value, attribute in [
        (metadata, "title"),
        (inspection, "chapters"),
        (voice, "name"),
        (stats, "duration_ms"),
        (result, "output"),
        (event, "completed"),
        (error, "code"),
    ]:
        with pytest.raises(FrozenInstanceError):
            setattr(value, attribute, None)

    assert set(get_args(kk.ExecutionEvent)) == {
        kk.CastResolved,
        kk.Started,
        kk.StageStarted,
        kk.StageProgress,
        kk.StageCompleted,
        kk.Warning,
        kk.Completed,
    }


def test_cancellation_token_is_cooperative_and_error_is_stable() -> None:
    """The token is explicit, thread-safe state with one public cancellation error."""
    token = kk.CancellationToken()
    assert not token.cancelled
    token.cancel()
    assert token.cancelled
    with pytest.raises(kk.CancelledError) as caught:
        token.raise_if_cancelled()
    assert caught.value.code == kk.ErrorCode.CANCELLED


def test_public_errors_preserve_normal_exception_propagation() -> None:
    """Interpreter-managed traceback updates must not mask public errors."""

    @contextmanager
    def boundary() -> Iterator[None]:
        yield

    with pytest.raises(kk.SourceError) as caught, boundary():
        raise kk.SourceError(kk.ErrorCode.SOURCE_NOT_FOUND)
    assert caught.value.code == kk.ErrorCode.SOURCE_NOT_FOUND


def test_public_exports_are_intentional() -> None:
    """The facade publishes the documented intent, inspection, and result values."""
    expected = {
        "__version__",
        "Engine",
        "add_voice",
        "list_voices",
        "load_voice",
        "remove_voice",
        "unload_voice",
        "list_castings",
        "list_series",
        "magic_run",
        "read_lexicon",
        "remove_casting",
        "remove_attribution",
        "remove_series",
        "SeriesCharacter",
        "SeriesRecord",
        "CastResolved",
        "CastingInspection",
        "CharacterProfile",
        "CharacterRoster",
        "Collision",
        "SpeakerSpan",
        "book",
        "builtin_lexicon",
        "epub",
        "Pipeline",
        "Source",
        "MetadataIntent",
        "BookMetadata",
        "ChapterInspection",
        "BookInspection",
        "Voice",
        "ValidationIssue",
        "ValidationResult",
        "ExecutionStats",
        "Result",
        "CancellationToken",
        "ExecutionEvent",
        "Started",
        "StageStarted",
        "StageProgress",
        "StageCompleted",
        "Warning",
        "Completed",
        "ErrorCode",
        "KenkuiError",
        "ValidationError",
        "SourceError",
        "VoiceError",
        "ModelError",
        "RenderError",
        "EncodingError",
        "CancelledError",
    }
    assert set(kk.__all__) == expected
    assert not hasattr(kk, "CacheStore")


def test_pronounce_records_intent_without_effects() -> None:
    """The operation captures configuration as an immutable value."""
    pipeline = kk.epub("book.epub").pronounce({"Cthulhu": "kuh-THOO-loo"})
    recorded = pipeline.operations[0]
    assert isinstance(recorded, SpokenForm)
    assert recorded.numbers == "conservative"
    assert recorded.builtin_lexicon is True
    assert recorded.lexicon == ()
    tuning = pipeline.operations[1]
    assert isinstance(tuning, Pronunciations)
    assert tuning.rules[0].value == (("Cthulhu", "kuh-THOO-loo"),)
    assert tuning.rules[0].where.is_whole_book()


def test_pronounce_is_branchable_and_absent_by_default() -> None:
    """A pipeline that never calls pronounce records no spoken-form intent."""
    root = kk.epub("book.epub")
    branch = root.pronounce()
    assert root.operations == ()
    assert any(isinstance(item, SpokenForm) for item in branch.operations)


def test_pronounce_rejects_an_unknown_tier() -> None:
    """Only the four defined tiers are accepted."""
    with pytest.raises(kk.ValidationError) as error:
        kk.epub("book.epub").pronounce(numbers="wild")
    assert error.value.code is kk.ErrorCode.INVALID_PRONUNCIATION


def test_pronounce_rejects_a_malformed_entry() -> None:
    """Caller entries are validated at the Pipeline boundary, not at render."""
    with pytest.raises(kk.ValidationError) as error:
        kk.epub("book.epub").pronounce({"": "x"})
    assert error.value.code is kk.ErrorCode.INVALID_PRONUNCIATION


def test_pronounce_style_can_be_replaced() -> None:
    """Repeated pronunciation calls replace the global spoken-form settings."""
    pipeline = kk.epub("book.epub").pronounce().pronounce(numbers="off")
    assert pipeline.operations == (SpokenForm(numbers="off"),)


def test_pauses_records_five_independent_durations() -> None:
    """Each boundary kind is separately variable."""
    pipeline = kk.epub("book.epub").pauses(
        chapter_ms=1500, heading_after_ms=600, paragraph_ms=250
    )
    recorded = pipeline.operations[0]
    assert isinstance(recorded, Pauses)
    assert recorded.chapter_ms == PAUSE_CHAPTER_MS
    assert recorded.heading_after_ms == PAUSE_HEADING_MS
    assert recorded.paragraph_ms == PAUSE_PARAGRAPH_MS
    assert recorded.heading_before_ms == 0
    assert recorded.line_ms == 0


@pytest.mark.parametrize(
    "kwargs",
    [
        {"chapter_ms": -1},
        {"line_ms": -100},
        {"paragraph_ms": 60_001},
    ],
)
def test_pauses_rejects_out_of_range_durations(kwargs: dict[str, int]) -> None:
    """Negative and absurd durations are refused at the Pipeline boundary."""
    with pytest.raises(kk.ValidationError) as error:
        kk.epub("book.epub").pauses(**kwargs)
    assert error.value.code is kk.ErrorCode.INVALID_PAUSE


def test_pauses_defaults_to_silence_free() -> None:
    """Calling pauses with no argument enables nothing."""
    recorded = kk.epub("book.epub").pauses().operations[0]
    assert isinstance(recorded, Pauses)
    assert recorded == Pauses()


def _traited_voice(voice_id: str, gender: PerceivedGender) -> Voice:
    return Voice(
        id=voice_id,
        name=voice_id.title(),
        enabled=True,
        provenance="test",
        license_id="CC-BY-4.0",
        commercial_use_allowed=False,
        language="english",
        state="loaded",
        perceived_gender=gender,
    )


def _speaker(character_id: str, gender: str | None) -> CharacterProfile:
    return CharacterProfile(character_id, character_id.title(), gender, 100, ("ch1",))


def test_an_ungendered_cast_is_logged_for_the_operator(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """The silence that let a pool carrying no gender traits go unnoticed.

    `candidates` falls back to the whole pool, so a gendered cast that
    cannot be honoured renders happily in arbitrary voices. Nothing else in
    the suite can observe that, because the fallback is the designed
    behaviour; only the log distinguishes it from a cast that worked.
    """
    with caplog.at_level(logging.WARNING):
        _log_ungendered_cast(
            "gendered",
            (_speaker("her", "feminine"),),
            (_traited_voice("c", "masculine"),),
        )
    assert "ungendered_cast" in caplog.text


def test_a_served_cast_logs_nothing(caplog: pytest.LogCaptureFixture) -> None:
    """No noise when the pool can honour the method."""
    pool = (_traited_voice("c", "masculine"), _traited_voice("a", "feminine"))
    with caplog.at_level(logging.WARNING):
        _log_ungendered_cast("gendered", (_speaker("her", "feminine"),), pool)
    assert "ungendered_cast" not in caplog.text


def test_the_random_method_is_never_reported(caplog: pytest.LogCaptureFixture) -> None:
    """The random method never promised a gendered pool."""
    with caplog.at_level(logging.WARNING):
        _log_ungendered_cast(
            "random",
            (_speaker("her", "feminine"),),
            (_traited_voice("c", "masculine"),),
        )
    assert "ungendered_cast" not in caplog.text


def test_series_records_intent_without_reading_anything() -> None:
    """Membership is declared, never derived: no EPUB carries it."""
    pipeline = kk.epub("book.epub").series("stormlight", book=3)
    recorded = pipeline.operations[-1]
    assert isinstance(recorded, Series)
    assert recorded.series_id == "stormlight"
    assert recorded.book == 3  # noqa: PLR2004 - the book number passed in above
    assert recorded.allow_recast is False
    assert recorded.allow_narrator_change is False


def test_an_empty_series_id_is_refused() -> None:
    """A series with no name cannot be looked up again."""
    with pytest.raises(kk.ValidationError) as error:
        kk.epub("book.epub").series("   ")
    assert error.value.code == kk.ErrorCode.INVALID_SERIES


def test_a_negative_book_number_is_refused() -> None:
    """A non-positive book number cannot order a series."""
    with pytest.raises(kk.ValidationError) as error:
        kk.epub("book.epub").series("stormlight", book=0)
    assert error.value.code == kk.ErrorCode.INVALID_SERIES


def test_two_series_declarations_replace_membership() -> None:
    """A book belongs to the most recently declared series."""
    pipeline = kk.epub("book.epub").series("a").series("b")
    assert pipeline.operations == (Series("b"),)
