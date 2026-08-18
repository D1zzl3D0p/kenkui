"""Contract tests for the WP2 public pipeline and value surface."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import FrozenInstanceError
from pathlib import Path
from typing import TYPE_CHECKING, cast, get_args

import pytest

import kenkui as kk

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator
    from typing import Any

EXPECTED_OPERATION_COUNT = 5
IO_ERROR_MESSAGE = "pipeline construction performed I/O"


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
        .normalize_text()
        .assign_voice("narrator")
        .tts()
        .metadata(title=None, author="Writer", cover=None)
    )

    assert source.source.path == Path("unread.epub")
    assert source.source.format == "epub"
    assert source.operations == ()
    assert len(pipeline.operations) == EXPECTED_OPERATION_COUNT


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
        first.operations[-1].voice_id = "changed"  # type: ignore[misc,union-attr]


def test_duplicate_and_contradictory_operations_have_stable_codes() -> None:
    """Invalid semantic chains fail deterministically at their cheapest boundary."""
    with pytest.raises(kk.ValidationError) as duplicate:
        kk.epub("book.epub").normalize_text().normalize_text()
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
    """The facade publishes only the approved WP2 concepts."""
    expected = {
        "__version__",
        "book",
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
