"""Exact selection, bounded edge changes, and reusable WAV probes."""

from __future__ import annotations

import pickle
import wave
from dataclasses import replace
from typing import TYPE_CHECKING

import pytest

import kenkui as kk
from conftest import CH08_ID, CH09_ID
from helpers import make_epub, xhtml
from kenkui._domain import planning, selection
from kenkui._domain.grid import build_grid
from kenkui._domain.operations import SpokenForm
from kenkui._domain.paths import parse_pattern
from kenkui._domain.planning import compile_execution_plan
from kenkui._domain.spoken import to_spoken
from kenkui._execution.cache import CacheStore
from kenkui.errors import EncodingError, ValidationError

_MAX_EDGES = 2
_PCM_SAMPLE_BYTES = 2
_SELECTED_PARAGRAPH = 2
_FIRST_SENTENCE = 2
_LAST_SENTENCE = 160

if TYPE_CHECKING:
    from pathlib import Path

    from kenkui._domain.grid import Unit
    from kenkui._domain.planning import ExecutionPlan


def _plan(book: kk.Pipeline) -> ExecutionPlan:
    checkpoint = book._resolved  # noqa: SLF001
    assert checkpoint is not None
    return compile_execution_plan(
        book.tts(),
        book.inspect(),
        source_bytes_hash=checkpoint.source_hash,
        resolved_voice=checkpoint.bindings.voice,
        model_revision=checkpoint.bindings.model_revision,
        spans=checkpoint.spans,
        assignments=checkpoint.cast_assignments,
        cast_voices=checkpoint.bindings.cast_voices,
    )


@pytest.fixture
def renderable_book(resolved_book: kk.Pipeline) -> kk.Pipeline:
    """Complete the shared review fixture's missing character-discovery intent."""
    return resolved_book.infer_characters("fake/model").resolve()


def test_select_is_lazy_immutable_and_preserves_checkpoint(
    resolved_book: kk.Pipeline, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Selection records a frozen union and keeps source resolution intact."""
    pattern: dict[str, object] = {"chapter": CH08_ID, "paragraph": [2]}

    def forbidden(*_args: object, **_kwargs: object) -> None:
        pytest.fail("selection declaration performed I/O or resolution")

    monkeypatch.setattr("kenkui.pipeline.inspect_epub", forbidden)
    monkeypatch.setattr("kenkui.pipeline.resolve_inputs", forbidden)
    selected = resolved_book.select(pattern)
    pattern["paragraph"] = 1
    assert selected._resolved is resolved_book._resolved  # noqa: SLF001
    assert len(selected.operations) == len(resolved_book.operations) + 1
    assert {row.path.paragraph for row in selected.script()} == {2}
    assert len(list(resolved_book.script())) > len(list(selected.script()))


def test_select_union_is_exact_ordered_and_deduplicated(
    resolved_book: kk.Pipeline,
) -> None:
    """Overlapping selectors do not duplicate rows or fill omitted subtrees."""
    book = resolved_book.select(
        {"chapter": CH09_ID},
        {"chapter": CH08_ID, "paragraph": 2, "sentence": -1},
        parse_pattern({"chapter": CH09_ID}),
    )
    rows = list(book.script())
    expected = [
        *resolved_book.script().at(
            {"chapter": CH08_ID, "paragraph": 2, "sentence": -1}
        ),
        *resolved_book.script().at({"chapter": CH09_ID}),
    ]
    assert [row.path for row in rows] == [row.path for row in expected]
    assert "".join(segment.text for segment in _plan(book).segments) == "".join(
        row.text for row in rows
    )
    with pytest.raises(KeyError):
        book.script()[next(iter(resolved_book.script())).path]


@pytest.mark.parametrize("first", ["select", "chapters", "range"])
@pytest.mark.parametrize("second", ["select", "chapters", "range"])
def test_selection_modes_are_mutually_exclusive(
    resolved_book: kk.Pipeline, first: str, second: str
) -> None:
    """All directions, including duplicate modes, reject another selection."""
    methods = {
        "select": lambda book: book.select({"chapter": CH08_ID}),
        "chapters": lambda book: book.select_chapters(CH08_ID),
        "range": lambda book: book.select_chapter_range(CH08_ID, CH09_ID),
    }
    with pytest.raises(ValidationError) as caught:
        methods[second](methods[first](resolved_book))
    assert caught.value.code == kk.ErrorCode.DUPLICATE_OPERATION


def test_select_rejects_empty_patterns_and_reports_no_matches(
    resolved_book: kk.Pipeline,
) -> None:
    """No arguments fail eagerly; valid selectors with no leaves fail on inspection."""
    with pytest.raises(ValidationError) as caught:
        resolved_book.select()
    assert caught.value.code == kk.ErrorCode.EMPTY_SELECTION
    with pytest.raises(ValidationError) as caught:
        resolved_book.select({"paragraph": 999}).inspect()
    assert caught.value.code == kk.ErrorCode.EMPTY_SELECTION


def test_empty_selection_fails_before_provider_work(
    renderable_book: kk.Pipeline,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An unresolved empty probe validates its source ranges before attribution."""

    def forbidden(*_args: object, **_kwargs: object) -> None:
        pytest.fail("empty selection reached model work")

    monkeypatch.setattr("kenkui._characters.resolve_attribution", forbidden)
    unresolved = replace(renderable_book, _resolved=None)
    with pytest.raises(ValidationError) as caught:
        unresolved.select({"paragraph": 999}).resolve()
    assert caught.value.code == kk.ErrorCode.EMPTY_SELECTION


def test_select_applies_before_resolution_and_preserves_source_snapshot(
    renderable_book: kk.Pipeline,
) -> None:
    """Selection before resolution shares the same full source checkpoint basis."""
    source = renderable_book.source.path
    base = kk.book(source).assign_voice("ivy")
    selected = base.select({"chapter": CH09_ID}).resolve()
    assert [chapter.id for chapter in selected.inspect().chapters] == [CH09_ID]
    checkpoint = selected._resolved  # noqa: SLF001
    assert checkpoint is not None
    assert [chapter.id for chapter in checkpoint.inspection.chapters] == [
        CH08_ID,
        CH09_ID,
    ]
    assert _plan(selected).segments == _plan(base.resolve()).segments[-1:]
    source.write_bytes(b"source changed")
    assert [row.path.chapter for row in selected.script()] == [CH09_ID]
    with pytest.raises(kk.SourceError) as caught:
        selected.tts().preview(source.with_suffix(".wav"), workers=1)
    assert caught.value.code == kk.ErrorCode.SOURCE_CHANGED


def test_select_keeps_roster_checkpoint(renderable_book: kk.Pipeline) -> None:
    """Selecting after discovery retains the exact reviewed roster checkpoint."""
    roster_book = renderable_book.resolve(until="characters")
    selected = roster_book.select({"chapter": CH09_ID})
    assert selected._roster is roster_book._roster  # noqa: SLF001
    assert selected._roster is not None  # noqa: SLF001
    assert selected.inspect().roster is roster_book.inspect().roster
    assert [chapter.id for chapter in selected.inspect().chapters] == [CH09_ID]


def test_selected_inspection_hides_planning_basis_from_repr_and_equality(
    renderable_book: kk.Pipeline,
) -> None:
    """The private source basis does not change the public view or persisted plan."""
    selected = renderable_book.select({"chapter": CH09_ID})
    inspection = selected.inspect()
    assert "_planning_chapters" not in repr(inspection)
    assert inspection == replace(inspection, _planning_chapters=())
    restored = pickle.loads(pickle.dumps(inspection))  # noqa: S301
    assert restored._planning_chapters == inspection._planning_chapters  # noqa: SLF001
    assert "_planning_chapters" not in repr(_plan(selected))


def test_select_respects_tts_order(renderable_book: kk.Pipeline) -> None:
    """Selection follows the established pre-TTS order of the chapter modes."""
    with pytest.raises(ValidationError) as caught:
        renderable_book.tts().select({"paragraph": 1})
    assert caught.value.code == kk.ErrorCode.INVALID_OPERATION_ORDER


@pytest.fixture
def long_book(resolved_book: kk.Pipeline, tmp_path: Path) -> kk.Pipeline:
    """Build a second chapter with selections crossing ordinary render chunks."""
    paragraphs = [
        " ".join(
            f"Paragraph {paragraph} sentence {sentence}." for sentence in range(180)
        )
        for paragraph in range(1, 5)
    ]
    source = make_epub(
        tmp_path / "long.epub",
        chapters={
            "ch08": xhtml("<p>Earlier chapter.</p>"),
            "ch09": xhtml("".join(f"<p>{text}</p>" for text in paragraphs)),
        },
        spine=["ch08", "ch09"],
    )
    return kk.book(source).assign_voice(resolved_book._assigned_voice_id()).resolve()  # noqa: SLF001


@pytest.mark.pocket_real
def test_selection_preserves_interior_segments_and_only_clips_edges(
    long_book: kk.Pipeline,
) -> None:
    """Use the real planner; one contiguous selection changes at most two chunks."""
    whole = _plan(long_book)
    selected = long_book.select(
        {"chapter": CH09_ID, "paragraph": 2, "sentence": "2..160"}
    )
    part = _plan(selected)
    chapter = long_book.inspect().chapters[1]
    units = [
        unit
        for unit in build_grid(chapter)
        if unit.paragraph == _SELECTED_PARAGRAPH
        and _FIRST_SENTENCE <= unit.sentence <= _LAST_SENTENCE
    ]
    lower, upper = units[0].start, units[-1].end
    offset = 0
    interior = []
    edge = []
    for segment in whole.segments:
        if segment.chapter_id != chapter.id:
            continue
        end = offset + len(segment.text)
        if lower <= offset and end <= upper:
            interior.append(segment)
        elif offset < upper and lower < end:
            edge.append(segment)
        offset = end
    assert len(interior) >= _MAX_EDGES
    assert interior == [segment for segment in part.segments if segment in interior]
    changed = [segment for segment in part.segments if segment not in interior]
    assert len(changed) == len(edge) <= _MAX_EDGES
    assert [segment.ordinal for segment in changed] == [
        segment.ordinal for segment in edge
    ]
    assert part.segments[0].ordinal > 0
    assert (
        "".join(segment.text for segment in part.segments) == chapter.text[lower:upper]
    )
    assert part.total_speech_characters == upper - lower
    assert part.output.chapters[0].speech_characters == upper - lower
    assert part.trailing_silence_ms[-1] == 0


@pytest.mark.parametrize("numbers", ["off", "standard"])
def test_spoken_selection_uses_exact_offsets_and_preserves_full_identities(
    long_book: kk.Pipeline, numbers: str
) -> None:
    """Repeated prose and number expansion do not move source selection edges."""
    book = long_book.pronounce(numbers=numbers, builtin=False)
    whole = _plan(book)
    assert _plan(book.select({})).segments == whole.segments
    selected = book.select({"chapter": CH09_ID, "paragraph": 2, "sentence": "2..160"})
    text = "".join(row.text for row in selected.script())
    expected = to_spoken(text, numbers=numbers, lexicon=(), builtin=False)  # type: ignore[arg-type]
    part = _plan(selected)
    assert "".join(segment.text for segment in part.segments) == expected
    assert part.total_speech_characters == len(text)
    whole_by_id = {segment.id: segment for segment in whole.segments}
    assert (
        len([segment for segment in part.segments if segment.id not in whole_by_id])
        <= _MAX_EDGES
    )


def test_selected_planning_reuses_each_chapter_grid(
    long_book: kk.Pipeline, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Selection clipping and scoped transforms do not rescan grid boundaries."""
    selected = long_book.select(
        {"chapter": CH09_ID, "paragraph": 2, "sentence": "2..160"}
    )
    inspection = selected.inspect()
    chapters = inspection._planning_chapters or inspection.chapters  # noqa: SLF001
    real_build = build_grid
    calls: list[str] = []

    def counted(chapter: kk.ChapterInspection) -> tuple[Unit, ...]:
        calls.append(chapter.id)
        return real_build(chapter)

    monkeypatch.setattr(planning, "build_grid", counted)
    monkeypatch.setattr(selection, "build_grid", counted)

    checkpoint = selected._resolved  # noqa: SLF001
    assert checkpoint is not None
    compile_execution_plan(
        selected.tts(),
        inspection,
        source_bytes_hash=checkpoint.source_hash,
        resolved_voice=checkpoint.bindings.voice,
        model_revision=checkpoint.bindings.model_revision,
        spans=checkpoint.spans,
        assignments=checkpoint.cast_assignments,
        cast_voices=checkpoint.bindings.cast_voices,
    )

    assert calls == [chapter.id for chapter in chapters]


def test_selection_does_not_guess_inside_cross_unit_pronunciation(
    renderable_book: kk.Pipeline,
) -> None:
    """A rule spanning sentence edges cannot leak excluded words into a probe."""
    # A directly-constructed SpokenForm is the narrowest way to place a
    # cross-unit entry: it is exactly one whole-book lexicon and no grid.
    # Scoped Pronunciations rules have their own coverage in
    # test_tuning_merge.py, which pins that they never reach outside a scope.
    spoken = SpokenForm(
        numbers="off",
        builtin_lexicon=False,
        lexicon=(("Alpha one. Alpha two", "Replacement"),),
    )
    book = replace(renderable_book, operations=(*renderable_book.operations, spoken))
    assert _plan(book.select({})).segments == _plan(book).segments
    selected = book.select({"chapter": CH08_ID, "paragraph": 1, "sentence": 2})
    assert (
        "".join(segment.text for segment in _plan(selected).segments).strip()
        == "Alpha two."
    )


@pytest.mark.parametrize("whitespace_ms", [300, 0, None])
@pytest.mark.parametrize("include_whitespace", [True, False])
def test_selected_whitespace_gaps_match_script_and_planning(
    renderable_book: kk.Pipeline,
    tmp_path: Path,
    whitespace_ms: int | None,
    *,
    include_whitespace: bool,
) -> None:
    """Selected whitespace carries zero rows and settles replacements consistently."""
    source = make_epub(
        tmp_path / "quotes.epub",
        chapters={"one": xhtml('<p>"Alpha." "Beta."</p>')},
        spine=["one"],
    )
    book = (
        kk.book(source)
        .assign_voice(renderable_book._assigned_voice_id())  # noqa: SLF001
        .silence(900, where={"sentence": 1, "phrase": 1})
    )
    if whitespace_ms is not None:
        book = book.silence(whitespace_ms, where={"sentence": 1, "phrase": 2})
    book = book.resolve().select({} if include_whitespace else {"phrase": 1})
    rows = [row for row in book.script() if row.text.strip()]
    plan = _plan(book)
    assert [row.text.strip() for row in rows] == [
        segment.text.strip() for segment in plan.segments
    ]
    expected = (
        whitespace_ms if include_whitespace and whitespace_ms is not None else 900
    )
    assert (
        tuple(row.silence_after_ms for row in rows)
        == plan.trailing_silence_ms
        == (expected, 0)
    )


def test_disjoint_equal_text_selections_have_distinct_edge_ids(
    renderable_book: kk.Pipeline, tmp_path: Path
) -> None:
    """Two identical phrases in one original segment remain separate cache entries."""
    source = make_epub(
        tmp_path / "repeated.epub",
        chapters={"one": xhtml("<p>Same. Gap. Same. Tail.</p>")},
        spine=["one"],
    )
    book = (
        kk.book(source)
        .assign_voice(renderable_book._assigned_voice_id())  # noqa: SLF001
        .resolve()
        .select({"sentence": [1, 3]})
    )
    segments = _plan(book).segments
    assert [segment.text for segment in segments] == ["Same. ", "Same. "]
    assert len({segment.id for segment in segments}) == len(segments)
    assert len({segment.ordinal for segment in segments}) == 1


def test_preview_is_real_wav_and_retains_cache_for_full_render(
    renderable_book: kk.Pipeline, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Sparse source ordinals pass publication and the full render reuses their PCM."""
    checkpoint = renderable_book._resolved  # noqa: SLF001
    assert checkpoint is not None
    cache = CacheStore(tmp_path / "probe-cache")
    book = replace(
        renderable_book,
        _resolved=replace(
            checkpoint, bindings=replace(checkpoint.bindings, cache_store=cache)
        ),
    )
    selected = book.select({"chapter": CH09_ID})
    events: list[kk.ExecutionEvent] = []
    target = tmp_path / "preview.wav"
    operations = selected.operations
    result = selected.preview(target, workers=1, on_event=events.append)
    assert selected.operations == operations
    assert result.output == target
    assert events
    with wave.open(str(target), "rb") as stream:
        assert stream.getsampwidth() == _PCM_SAMPLE_BYTES
        assert stream.getnframes() > 0
        assert stream.getcomptype() == "NONE"
    recorded: list[str] = []
    lookup = cache.lookup

    def tracked(*args: object, **kwargs: object) -> object:
        audio = lookup(*args, **kwargs)  # type: ignore[arg-type]
        if audio is not None:
            recorded.append(audio.segment_id)
        return audio

    monkeypatch.setattr(
        CacheStore, "lookup", lambda _self, *args, **kwargs: tracked(*args, **kwargs)
    )
    book.tts().write_m4b(tmp_path / "whole.m4b", workers=1)
    assert recorded == [segment.id for segment in _plan(selected).segments]


@pytest.mark.parametrize("suffix", [".m4b", ".M4B", ".mp3"])
def test_preview_rejects_non_wav_outputs(
    renderable_book: kk.Pipeline, tmp_path: Path, suffix: str
) -> None:
    """A probe cannot be mistaken for an M4B or silently mislabeled."""
    with pytest.raises(ValidationError) as caught:
        renderable_book.tts().preview(tmp_path / f"probe{suffix}")
    assert caught.value.code == kk.ErrorCode.INVALID_OUTPUT


def test_preview_reuses_validation_and_overwrite_controls(
    renderable_book: kk.Pipeline, tmp_path: Path
) -> None:
    """Validation is shared with publication before any rendering starts."""
    with pytest.raises(ValidationError) as caught:
        kk.book(renderable_book.source.path).preview(tmp_path / "probe.wav")
    assert caught.value.code == kk.ErrorCode.VOICE_REQUIRED
    with pytest.raises(ValidationError) as caught:
        renderable_book.tts().preview(tmp_path / "probe.wav", workers=0)
    assert caught.value.code == kk.ErrorCode.INVALID_WORKERS
    target = tmp_path / "probe.wav"
    target.write_bytes(b"original")
    with pytest.raises(EncodingError) as encoding:
        renderable_book.tts().preview(target)
    assert encoding.value.code == kk.ErrorCode.OUTPUT_EXISTS
    assert target.read_bytes() == b"original"
    renderable_book.tts().preview(target, workers=1, overwrite=True)
    assert target.read_bytes().startswith(b"RIFF")


def test_preview_skips_metadata_cover_reads(
    renderable_book: kk.Pipeline, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Invalid publication cover paths do not affect a metadata-free WAV probe."""

    def forbidden(*_args: object, **_kwargs: object) -> None:
        pytest.fail("preview attempted publication cover I/O")

    monkeypatch.setattr("kenkui._execution.coordinator.read_cover", forbidden)
    target = tmp_path / "probe.WAV"
    renderable_book.metadata(
        title="Private title", author="Private author", cover=tmp_path / "missing.jpg"
    ).tts().preview(target, workers=1)
    content = target.read_bytes()
    assert content.startswith(b"RIFF")
    assert b"Private title" not in content
    assert b"Private author" not in content
    assert b"LIST" not in content
