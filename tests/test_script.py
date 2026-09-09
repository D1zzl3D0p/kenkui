"""Lazy review rows preserve canonical text, provenance, and authoring anchors."""

from __future__ import annotations

from dataclasses import FrozenInstanceError, replace
from typing import TYPE_CHECKING

import pytest

import kenkui as kk
from conftest import CH08_ID, CH09_ID
from helpers import make_epub, xhtml
from kenkui._domain.grid import build_grid
from kenkui._domain.operations import Attributions, Pronunciations, Silences
from kenkui._domain.paths import Path, parse_pattern, path_of
from kenkui._domain.planning import (
    SpeakerSpan,
    compile_execution_plan,
    effective_spans,
    manual_gaps,
)
from kenkui._domain.sidecar import authoring_snapshot
from kenkui._domain.tuning import Rule

if TYPE_CHECKING:
    from pathlib import Path as FilePath

    from kenkui._domain.grid import Unit


def test_rows_cover_the_chapter_in_order(resolved_book: kk.Pipeline) -> None:
    """Rows are a lossless partition in canonical order, including whitespace."""
    rows = list(resolved_book.script().at({"chapter": CH08_ID}))
    assert rows == sorted(
        rows,
        key=lambda row: (
            row.path.paragraph or 0,
            row.path.line or 0,
            row.path.sentence or 0,
            row.path.phrase or 0,
        ),
    )
    assert "".join(row.text for row in rows) == resolved_book.inspect().chapters[0].text


def test_provenance_reports_the_winning_rule(resolved_book: kk.Pipeline) -> None:
    """Specificity wins first; equal patterns use their declaration order."""
    book = (
        resolved_book.attribute("irulan", where={"chapter": "*", "paragraph": 1})
        .attribute("jessica", where={"chapter": CH08_ID, "paragraph": 1})
        .attribute("paul", where={"chapter": CH08_ID, "paragraph": 1})
        .attribute("narrator", where={"chapter": CH08_ID})
    )
    row = next(iter(book.script().at({"chapter": CH08_ID, "paragraph": 1})))
    assert (row.character, row.provenance, row.rule_index) == ("paul", "rule", 2)


def test_machine_provenance_when_no_rule_matches(resolved_book: kk.Pipeline) -> None:
    """An unattributed resolved chapter uses the default narration layer."""
    rows = list(resolved_book.script().at({"chapter": CH08_ID}))
    assert rows
    assert {row.provenance for row in rows} <= {"machine", "default"}
    assert all(row.rule_index is None for row in rows)


def test_script_works_before_resolution(
    unresolved_book: kk.Pipeline, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Inspection, even with model intent, never invokes resolution or models."""

    def forbidden(*_args: object, **_kwargs: object) -> None:
        pytest.fail("script inspection reached a model or resolution boundary")

    monkeypatch.setattr("kenkui.pipeline.resolve_inputs", forbidden)
    monkeypatch.setattr("kenkui._resolution._attribution_client", forbidden)
    book = unresolved_book.attribute_quotes("fake/model").attribute(
        "irulan", where={"paragraph": 1}
    )
    rows = list(book.script().at({"chapter": CH08_ID}))
    assert {row.provenance for row in rows} == {"unresolved", "rule"}
    assert all(row.character is None for row in rows if row.provenance == "unresolved")


def test_chapters_materialize_lazily(
    resolved_book: kk.Pipeline, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Each chapter builds once, and reading cache metadata does no work."""
    calls: list[str] = []

    def tracked(chapter: kk.ChapterInspection) -> tuple[Unit, ...]:
        calls.append(chapter.id)
        return build_grid(chapter)

    monkeypatch.setattr("kenkui.script.build_grid", tracked)
    script = resolved_book.script()
    assert tuple(script.materialized) == ()
    assert script.warnings == ()
    assert calls == []
    first = list(script.at(parse_pattern({"chapter": CH08_ID})))
    assert script.materialized == (CH08_ID,)
    assert list(script.at({"chapter": CH08_ID})) == first
    assert script[first[0].path] is first[0]
    assert calls == [CH08_ID]
    assert [row.path.chapter for row in script][-1] == CH09_ID
    assert calls == [CH08_ID, CH09_ID]


def test_path_lookup_and_rows_are_immutable(unresolved_book: kk.Pipeline) -> None:
    """An exact path yields its frozen row; missing/ambiguous paths raise."""
    script = unresolved_book.script()
    row = next(iter(script.at({"chapter": CH08_ID})))
    assert script[row.path] is row
    with pytest.raises(FrozenInstanceError):
        row.text = "changed"  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        row.path.paragraph = 9  # type: ignore[misc]
    with pytest.raises(KeyError):
        script[Path(chapter="missing", paragraph=1)]
    with pytest.raises(KeyError):
        script[Path(chapter=CH08_ID)]
    with pytest.raises(KeyError):
        script[Path(chapter=CH08_ID, paragraph=999)]


def test_filters_use_last_and_set_semantics(unresolved_book: kk.Pipeline) -> None:
    """Filters use the complete chapter's sibling counts, never filtered counts."""
    script = unresolved_book.script()
    rows = list(script.at({"chapter": CH08_ID, "paragraph": [1, 2], "sentence": -1}))
    units = build_grid(unresolved_book.inspect().chapters[0])
    assert rows
    for row in rows:
        assert row.path.sentence == max(
            unit.sentence
            for unit in units
            if (unit.paragraph, unit.line) == (row.path.paragraph, row.path.line)
        )
    assert list(script.at({"chapter": "missing"})) == []


def test_machine_and_rule_rows_agree_with_planning(resolved_book: kk.Pipeline) -> None:
    """The script retains provenance while agreeing with the effective spans."""
    checkpoint = resolved_book._resolved  # noqa: SLF001
    assert checkpoint is not None
    chapter = checkpoint.inspection.chapters[0]
    machine = (SpeakerSpan(chapter.id, 0, len(chapter.text), "jessica"),)
    casting = checkpoint.inspection.casting
    assert casting is not None
    inspection = replace(checkpoint.inspection, casting=replace(casting, spans=machine))
    book = replace(
        resolved_book,
        _resolved=replace(checkpoint, spans=machine, inspection=inspection),
    )
    book = book.attribute("irulan", where={"chapter": chapter.id, "paragraph": 1})
    effective = effective_spans(chapter, machine, book.operations)
    rows = list(book.script().at({"chapter": chapter.id}))
    for unit, row in zip(build_grid(chapter), rows, strict=True):
        assert row.character == next(
            span.character_id
            for span in effective
            if span.start <= unit.start < span.end
        )
    assert {row.provenance for row in rows} == {"rule", "machine"}


def test_flags_and_silences_match_the_grid(unresolved_book: kk.Pipeline) -> None:
    """A subtree silence appears only on its final leaf, including explicit zero."""
    book = (
        unresolved_book.pauses(paragraph_ms=500, chapter_ms=1200)
        .silence(0, where={"chapter": CH08_ID, "paragraph": 1})
        .silence(900, where={"chapter": CH08_ID, "paragraph": 2, "sentence": 1})
    )
    chapter = book.inspect().chapters[0]
    rows = list(book.script().at({"chapter": chapter.id}))
    units = build_grid(chapter)
    assert [(row.is_dialogue, row.is_emphasised) for row in rows] == [
        (unit.is_dialogue, unit.is_emphasised) for unit in units
    ]
    for index, gap in manual_gaps(chapter, book.operations).items():
        assert rows[index].silence_after_ms == gap
    chapter_gap = 1200
    assert rows[-1].silence_after_ms == chapter_gap
    assert list(book.script())[-1].silence_after_ms == 0


@pytest.mark.parametrize("kind", [Attributions, Silences, Pronunciations])
def test_anchor_digest_mismatch_is_reported(
    resolved_book: kk.Pipeline, kind: type[Attributions | Silences | Pronunciations]
) -> None:
    """Stale anchors remain at the authored path and warn on materialization."""
    value = (
        "jessica"
        if kind is Attributions
        else 900
        if kind is Silences
        else (("A", "B"),)
    )
    rule = Rule(
        parse_pattern({"chapter": CH08_ID, "paragraph": 1}),
        value,
        0,
        digest="sha256:deadbeefdeadbeef",
    )
    book = replace(resolved_book, operations=(*resolved_book.operations, kind((rule,))))
    script = book.script()
    before = script.warnings
    assert before == ()
    list(script.at({"chapter": CH08_ID}))
    (warning,) = script.warnings
    assert warning.code == kk.ErrorCode.ANCHOR_DIGEST_MISMATCH
    assert warning.severity == "warning"
    assert "digest" in warning.message
    assert CH08_ID in warning.message
    assert "¶1" in warning.message
    assert before == ()
    assert list(script.at({"chapter": CH08_ID}))
    assert script.warnings == (warning,)


def test_unchanged_sidecar_subtrees_do_not_warn(unresolved_book: kk.Pipeline) -> None:
    """Drift checks hash the exact same whole subtree as sidecar authoring."""
    book = unresolved_book.attribute(
        "irulan", where={"chapter": CH08_ID, "paragraph": 1}
    )
    snapshot = authoring_snapshot(book.operations, book.inspect().chapters)
    script = replace(book, operations=snapshot).script()
    list(script)
    assert script.warnings == ()


def test_match_count_waits_for_the_whole_rule_scope(
    unresolved_book: kk.Pipeline,
) -> None:
    """A wildcard count cannot be compared to an incomplete chapter subtotal."""
    rule = Rule(
        parse_pattern({"chapter": "*", "paragraph": 1}), "irulan", 0, matched=999
    )
    script = replace(unresolved_book, operations=(Attributions((rule,)),)).script()
    list(script.at({"chapter": CH09_ID}))
    assert tuple(script.warnings) == ()
    list(script.at({"chapter": CH08_ID}))
    (warning,) = script.warnings
    assert warning.code == kk.ErrorCode.PATTERN_MATCH_COUNT_DRIFT
    assert "999" in warning.message
    assert "3" in warning.message


def test_exact_chapter_count_and_missing_digest(unresolved_book: kk.Pipeline) -> None:
    """Missing anchors and zero-match drift surface without materializing others."""
    rules = (
        Rule(
            parse_pattern({"chapter": CH08_ID, "paragraph": 999}),
            "paul",
            0,
            digest="sha256:deadbeefdeadbeef",
        ),
        Rule(
            parse_pattern({"chapter": CH08_ID, "paragraph": "2..3"}),
            "jessica",
            1,
            matched=0,
        ),
    )
    script = replace(unresolved_book, operations=(Attributions(rules),)).script()
    list(script.at({"chapter": CH08_ID}))
    assert [warning.code for warning in script.warnings] == [
        kk.ErrorCode.ANCHOR_DIGEST_MISMATCH,
        kk.ErrorCode.PATTERN_MATCH_COUNT_DRIFT,
    ]
    assert script.materialized == (CH08_ID,)


def test_sidecar_roundtrip_retains_count_and_digest(
    unresolved_book: kk.Pipeline, tmp_path: FilePath
) -> None:
    """Actual authored and reloaded anchors validate across every rule family."""
    tuned = (
        unresolved_book.attribute("irulan", where={"paragraph": 1})
        .silence(500, where={"chapter": CH08_ID, "paragraph": 1})
        .pronounce({"Alpha": "Alfa"}, where={"chapter": CH08_ID})
    )
    path = tuned.write_annotations(tmp_path / "book.kenkui.json")
    script = unresolved_book.annotations(path).script()
    assert len(list(script)) == sum(
        len(build_grid(chapter)) for chapter in tuned.inspect().chapters
    )
    assert script.warnings == ()


def test_selection_keeps_source_paths(unresolved_book: kk.Pipeline) -> None:
    """Selecting a chapter preserves its source IDs and canonical coordinates."""
    script = unresolved_book.select_chapters(CH09_ID).script()
    rows = list(script)
    expected = build_grid(unresolved_book.inspect().chapters[1])
    assert [row.path for row in rows] == [path_of(unit) for unit in expected]
    assert script.materialized == (CH09_ID,)


@pytest.mark.parametrize("use_range", [False, True])
def test_selected_scripts_defer_broad_counts(
    unresolved_book: kk.Pipeline, monkeypatch: pytest.MonkeyPatch, *, use_range: bool
) -> None:
    """A source-wide saved count is never compared with a selected subtotal."""
    rules = (
        Rule(parse_pattern({"paragraph": 1}), "irulan", 0, matched=999),
        Rule(parse_pattern({"chapter": CH08_ID}), "jessica", 1, matched=999),
        Rule(parse_pattern({"chapter": CH09_ID}), "paul", 2, matched=999),
    )
    book = replace(unresolved_book, operations=(Attributions(rules),))
    book = (
        book.select_chapter_range(CH09_ID, CH09_ID)
        if use_range
        else book.select_chapters(CH09_ID)
    )
    calls: list[str] = []

    def tracked(chapter: kk.ChapterInspection) -> tuple[Unit, ...]:
        calls.append(chapter.id)
        return build_grid(chapter)

    monkeypatch.setattr("kenkui.script.build_grid", tracked)
    script = book.script()
    list(script)
    (warning,) = script.warnings
    assert "rule[2]" in warning.message
    assert calls == [CH09_ID]


def test_whole_book_iteration_stops_at_the_first_requested_row(
    unresolved_book: kk.Pipeline,
) -> None:
    """Creating an iterator does no grid work, and next builds one chapter."""
    script = unresolved_book.script()
    rows = iter(script)
    assert tuple(script.materialized) == ()
    next(rows)
    assert script.materialized == (CH08_ID,)
    assert script[{"chapter": CH09_ID, "paragraph": 1}].path.chapter == CH09_ID


def test_emphasis_and_structural_pauses_are_visible(tmp_path: FilePath) -> None:
    """Parsed emphasis and derived paragraph/line/heading gaps reach review rows."""
    source = make_epub(
        tmp_path / "emphasis.epub",
        chapters={
            "one": xhtml(
                "<h1>Title</h1><p><em>First line.</em><br/>Second line.</p><p>Last.</p>"
            )
        },
        spine=["one"],
    )
    book = kk.book(source).pauses(heading_after_ms=700, line_ms=200, paragraph_ms=400)
    rows = list(book.script())
    assert any(row.is_emphasised for row in rows)
    assert any(not row.is_emphasised for row in rows)
    assert [row.silence_after_ms for row in rows] == [700, 200, 400, 0]


def test_drift_findings_follow_declaration_order_not_access_order(
    unresolved_book: kk.Pipeline,
) -> None:
    """Incremental diagnostics retain each operation family's original rule index."""
    rules = (
        Rule(parse_pattern({"chapter": CH08_ID}), "irulan", 40, matched=999),
        Rule(parse_pattern({"chapter": CH09_ID}), "jessica", 3, matched=999),
    )
    script = replace(unresolved_book, operations=(Attributions(rules),)).script()
    list(script.at({"chapter": CH09_ID}))
    snapshot = script.warnings
    list(script.at({"chapter": CH08_ID}))
    assert "rule[3]" in snapshot[0].message
    assert [warning.message.split(" at ")[0] for warning in script.warnings] == [
        "Attributions rule[40]",
        "Attributions rule[3]",
    ]
    with pytest.raises(FrozenInstanceError):
        snapshot[0].message = "changed"  # type: ignore[misc]


def test_missing_chapter_anchor_warns_without_a_grid(
    unresolved_book: kk.Pipeline,
) -> None:
    """An absent source chapter is detectable when its empty query is consumed."""
    rule = Rule(
        parse_pattern({"chapter": "missing"}),
        "irulan",
        0,
        digest="sha256:deadbeefdeadbeef",
    )
    script = replace(unresolved_book, operations=(Attributions((rule,)),)).script()
    assert tuple(script.warnings) == ()
    assert list(script.at({"chapter": "missing"})) == []
    (warning,) = script.warnings
    assert warning.code == kk.ErrorCode.ANCHOR_DIGEST_MISMATCH
    assert script.materialized == ()


def test_stale_digest_does_not_relocate_the_declared_rule(
    unresolved_book: kk.Pipeline,
) -> None:
    """A stale paragraph anchor still targets that paragraph and visibly warns."""
    rule = Rule(
        parse_pattern({"chapter": CH08_ID, "paragraph": 1}),
        "irulan",
        0,
        digest="sha256:deadbeefdeadbeef",
    )
    script = replace(unresolved_book, operations=(Attributions((rule,)),)).script()
    rows = list(script)
    assert all(
        row.path.paragraph == 1 and row.path.chapter == CH08_ID
        for row in rows
        if row.character == "irulan"
    )
    assert len(script.warnings) == 1


@pytest.mark.parametrize("whitespace_ms", [300, 0, None])
def test_whitespace_gaps_match_compiled_spoken_gaps(
    tmp_path: FilePath, whitespace_ms: int | None
) -> None:
    """Whitespace keeps its row but settles its gap onto the previous spoken row."""
    source = make_epub(
        tmp_path / "quotes.epub",
        chapters={"one": xhtml('<p>"Alpha." "Beta."</p>')},
        spine=["one"],
    )
    book = (
        kk.book(source)
        .assign_voice("ivy")
        .silence(900, where={"sentence": 1, "phrase": 1})
    )
    if whitespace_ms is not None:
        book = book.silence(whitespace_ms, where={"sentence": 1, "phrase": 2})
    inspection = book.inspect()
    script = book.script()
    rows = list(script)
    voice = kk.Voice(
        id="ivy",
        name="Ivy",
        enabled=True,
        provenance="fixture",
        license_id="CC0-1.0",
        commercial_use_allowed=True,
        language="en",
        state="loaded",
        content_fingerprint="a" * 64,
        compatible_model_revisions=("fake-v1",),
    )
    plan = compile_execution_plan(
        book.tts(),
        inspection,
        source_bytes_hash="b" * 64,
        resolved_voice=voice,
        model_revision="fake-v1",
    )
    assert [row.text for row in rows] == ['"Alpha."', " ", '"Beta."']
    assert rows[1].silence_after_ms == 0
    assert [row.text.strip() for row in rows if row.text.strip()] == [
        segment.text.strip() for segment in plan.segments
    ]
    assert tuple(row.silence_after_ms for row in rows if row.text.strip()) == (
        plan.trailing_silence_ms
    )
    assert plan.trailing_silence_ms == (
        900 if whitespace_ms is None else whitespace_ms,
        0,
    )
