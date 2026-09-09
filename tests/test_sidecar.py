"""Versioned sidecar persistence and immutable loading contracts."""

from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

import kenkui as kk
from conftest import CH08_ID, CH09_ID
from helpers import make_epub, xhtml
from kenkui._domain import sidecar
from kenkui._domain.grid import build_grid, unit_digest
from kenkui._domain.operations import (
    Annotations,
    Attributions,
    Pronunciations,
    Silences,
)
from kenkui._domain.paths import parse_pattern
from kenkui._domain.sidecar import SIDECAR_VERSION, deserialize, serialize, sidecar_path
from kenkui._domain.tuning import Rule, resolve_rules
from kenkui.errors import ErrorCode, ValidationError

if TYPE_CHECKING:
    from kenkui._domain.grid import Unit
    from kenkui.inspection import ChapterInspection


def test_round_trip_preserves_tuning(epub_path: Path, tmp_path: Path) -> None:
    """Saving preserves patterns, values and declaration order on reload."""
    book = (
        kk.book(epub_path)
        .attribute("irulan", where={"chapter": "*", "paragraph": 1})
        .attribute("paul", where={"chapter": CH08_ID, "paragraph": 2})
        .silence(900, where={"chapter": CH08_ID, "paragraph": 1})
        .pronounce({"Atreides": "Ah-tray-deez"})
    )
    written = book.write_annotations(tmp_path / "b.kenkui.json")
    reloaded = kk.book(epub_path).annotations(written)
    original = tuple(
        op
        for op in book.operations
        if isinstance(op, (Attributions, Silences, Pronunciations))
    )
    loaded = tuple(
        replace(
            op, rules=tuple(replace(r, digest=None, matched=None) for r in op.rules)
        )
        for op in reloaded.operations
        if isinstance(op, (Attributions, Silences, Pronunciations))
    )
    assert loaded == original
    assert all(rule.digest is None for op in original for rule in op.rules)


def test_sidecar_defaults_beside_the_epub(epub_path: Path) -> None:
    """The default path uses the source stem and supports default loading."""
    book = kk.book(epub_path).attribute("irulan")
    written = book.write_annotations()
    assert written == epub_path.with_suffix(".kenkui.json")
    loaded = kk.book(epub_path).annotations()
    assert any(isinstance(op, Attributions) for op in loaded.operations)


def test_digest_tracks_exact_loaded_bytes(epub_path: Path, tmp_path: Path) -> None:
    """Even formatting-only edits change the loaded content identity."""
    written = kk.book(epub_path).attribute("a").write_annotations(tmp_path / "s.json")
    one = kk.book(epub_path).annotations(written)
    raw = written.read_bytes()
    written.write_bytes(raw + b"\n")
    two = kk.book(epub_path).annotations(written)
    first = next(op for op in one.operations if isinstance(op, Annotations))
    second = next(op for op in two.operations if isinstance(op, Annotations))
    assert first.digest == hashlib.sha256(raw).hexdigest()
    assert first.digest != second.digest
    assert one.operations != two.operations


def test_loading_twice_is_refused(epub_path: Path, tmp_path: Path) -> None:
    """The unique loader is rejected before a second file read."""
    written = kk.book(epub_path).attribute("a").write_annotations(tmp_path / "s.json")
    loaded = kk.book(epub_path).annotations(written)
    written.unlink()
    with pytest.raises(ValidationError) as error:
        loaded.annotations(written)
    assert error.value.code == ErrorCode.DUPLICATE_OPERATION


@pytest.mark.parametrize("raw", ["[]", '{"kenkui_sidecar": 9}', "{", "\ufffd"])
def test_malformed_sidecar_is_refused(
    epub_path: Path, tmp_path: Path, raw: str
) -> None:
    """Malformed files and unknown formats use the public sidecar error."""
    bad = tmp_path / "bad.json"
    bad.write_text(raw, encoding="utf-8")
    with pytest.raises(ValidationError) as error:
        kk.book(epub_path).annotations(bad)
    assert error.value.code == ErrorCode.INVALID_SIDECAR


def test_only_tuning_serializes(epub_path: Path, tmp_path: Path) -> None:
    """Voice, pause style, bibliography and prior loader stay out of JSON."""
    book = (
        kk.book(epub_path)
        .assign_voice("ivy")
        .pauses(paragraph_ms=400)
        .metadata(title="Book")
        .attribute("a")
    )
    written = book.write_annotations(tmp_path / "s.json")
    loaded = book.annotations(written)
    payload = json.loads(loaded.write_annotations(written).read_text())
    assert set(payload) == {"kenkui_sidecar", "attributions"}


def test_saving_records_leaf_and_subtree_digests(epub_path: Path) -> None:
    """An exact paragraph hash covers its entire canonical text span."""
    chapter = kk.book(epub_path).inspect().chapters[0]
    units = build_grid(chapter)
    book = (
        kk.book(epub_path)
        .attribute("a", where={"chapter": CH08_ID, "paragraph": 1})
        .attribute("b", where={"chapter": CH08_ID, "paragraph": 1, "sentence": 2})
    )
    payload = json.loads(book.write_annotations().read_text())
    paragraph = [unit for unit in units if unit.paragraph == 1]
    subtree = replace(paragraph[0], end=paragraph[-1].end)
    assert payload["attributions"][0]["digest"] == unit_digest(subtree, chapter.text)
    assert payload["attributions"][1]["digest"] == unit_digest(
        paragraph[1], chapter.text
    )
    assert "matched" not in payload["attributions"][0]


def test_patterns_record_counts_and_last_resolves(epub_path: Path) -> None:
    """Whole-book, wildcards, sets, spans and last retain matched leaf counts."""
    chapters = kk.book(epub_path).inspect().chapters
    units = tuple(unit for chapter in chapters for unit in build_grid(chapter))
    book = kk.book(epub_path).attribute(
        "a",
        where=(
            {},
            {"chapter": "*", "paragraph": 1},
            {"chapter": CH08_ID, "paragraph": [1]},
            {"chapter": CH08_ID, "paragraph": "1..2"},
            {"paragraph": -1},
            {"chapter": "missing"},
        ),
    )
    payload = json.loads(book.write_annotations().read_text())
    expected = [
        len(units),
        sum(unit.paragraph == 1 for unit in units),
        sum(unit.chapter_id == CH08_ID and unit.paragraph == 1 for unit in units),
        sum(unit.chapter_id == CH08_ID for unit in units),
        sum(
            unit.paragraph == (2 if unit.chapter_id == CH08_ID else 1) for unit in units
        ),
        0,
    ]
    assert [rule["matched"] for rule in payload["attributions"]] == expected
    loaded = kk.book(epub_path).annotations()
    operation = next(op for op in loaded.operations if isinstance(op, Attributions))
    assert [rule.matched for rule in operation.rules] == expected


def test_loading_extends_rules_and_keeps_checkpoints(
    resolved_book: kk.Pipeline, tmp_path: Path
) -> None:
    """Loaded rules form a baseline; inline rules and checkpoints survive."""
    source = (
        kk.book(resolved_book.source.path)
        .attribute("loaded", where=({}, {"paragraph": 1}))
        .silence(0)
        .pronounce({"Paul": "Pawl"})
    )
    written = source.write_annotations(tmp_path / "s.json")
    base = resolved_book.attribute("inline")
    loaded = base.annotations(str(written)).attribute("later")
    operation = next(op for op in loaded.operations if isinstance(op, Attributions))
    assert [rule.value for rule in operation.rules] == [
        "loaded",
        "loaded",
        "inline",
        "later",
    ]
    assert [rule.index for rule in operation.rules] == list(range(4))
    annotation = next(op for op in loaded.operations if isinstance(op, Annotations))
    assert annotation.loaded == {"attributions": 2, "silences": 1, "pronunciations": 1}
    assert next(op for op in base.operations if isinstance(op, Attributions)).rules == (
        Rule(parse_pattern({}), "inline", 0),
    )
    assert loaded._resolved is base._resolved  # noqa: SLF001
    assert loaded._roster is base._roster  # noqa: SLF001
    with pytest.raises(TypeError):
        annotation.loaded["attributions"] = 9  # type: ignore[index]


def test_inline_rules_win_equal_specificity_after_loading(epub_path: Path) -> None:
    """Inline overrides win equal patterns even when declared before loading."""
    written = kk.book(epub_path).attribute("from-file").write_annotations()
    book = kk.book(epub_path).attribute("inline-first").attribute("inline-last")
    loaded = book.annotations(written)
    rules = next(op.rules for op in loaded.operations if isinstance(op, Attributions))
    unit = build_grid(book.inspect().chapters[0])[0]
    decision = resolve_rules(unit, None, rules, {})
    assert decision.value == "inline-last"
    assert decision.rule_index == rules[-1].index
    assert [rule.value for rule in rules] == [
        "from-file",
        "inline-first",
        "inline-last",
    ]
    assert [rule.index for rule in rules] == list(range(len(rules)))


def _shared_tuning(book: kk.Pipeline) -> kk.Pipeline:
    """Apply shared code declarations that may already be saved in a sidecar."""
    where = {"chapter": CH08_ID, "paragraph": 1}
    return (
        book.attribute("shared", where=where)
        .silence(250, where=where)
        .pronounce({"Paul": "Pawl"}, where=where)
    )


def test_repeated_save_load_deduplicates_shared_tuning(epub_path: Path) -> None:
    """All tuning kinds remain stable despite inline indices/metadata differing."""
    saved = _shared_tuning(
        kk.book(epub_path)
        .attribute("book", where={"paragraph": 2})
        .silence(900, where={"paragraph": 2})
        .pronounce({"Chani": "Chah-nee"}, where={"paragraph": 2})
    )
    written = saved.write_annotations()
    original = written.read_bytes()
    for _cycle in range(3):
        inline = _shared_tuning(kk.book(epub_path))
        loaded = inline.annotations(written)
        counts = next(
            op.loaded for op in loaded.operations if isinstance(op, Annotations)
        )
        for kind in (Attributions, Silences, Pronunciations):
            rules = next(op.rules for op in loaded.operations if isinstance(op, kind))
            code_rules = next(
                op.rules for op in inline.operations if isinstance(op, kind)
            )
            assert len(rules) == counts[kind.__name__.lower()]
            assert [rule.index for rule in rules] == [0, 1]
            assert rules[-1].digest is not None
            assert code_rules[0].index == 0
            assert code_rules[0].digest is None
        loaded.write_annotations(written)
        assert written.read_bytes() == original


def test_deduplication_retains_distinct_rules_and_source_repetitions(
    epub_path: Path,
) -> None:
    """Only exact cross-source duplicates disappear; meaningful order is retained."""
    written = (
        kk.book(epub_path)
        .attribute("shared")
        .attribute("file-middle")
        .attribute("shared")
        .write_annotations()
    )
    inline = (
        kk.book(epub_path)
        .attribute("shared")
        .attribute("shared", where={"paragraph": 1})
        .attribute("inline")
        .attribute("inline-middle")
        .attribute("inline")
    )
    loaded = inline.annotations(written)
    rules = next(op.rules for op in loaded.operations if isinstance(op, Attributions))
    assert [rule.value for rule in rules] == [
        "shared",
        "file-middle",
        "shared",
        "shared",
        "inline",
        "inline-middle",
        "inline",
    ]
    assert [rule.index for rule in rules] == list(range(len(rules)))
    counts = next(op.loaded for op in loaded.operations if isinstance(op, Annotations))
    assert counts["attributions"] == len(
        json.loads(written.read_text())["attributions"]
    )
    assert all(rule.matched is not None for rule in rules[: counts["attributions"]])
    assert all(rule.matched is None for rule in rules[counts["attributions"] :])
    assert rules[counts["attributions"]].where == parse_pattern({"paragraph": 1})


def test_codec_preserves_array_order_and_detaches_payload() -> None:
    """Indices describe declaration order, and mutable JSON cannot change rules."""
    rules = (
        Rule(parse_pattern({"paragraph": [3, 1]}), "first", 10, matched=4),
        Rule(parse_pattern({"sentence": "2..4"}), "last", 0),
    )
    payload = serialize((Silences(()), Attributions(rules)))
    loaded = deserialize(payload)
    assert isinstance(loaded[0], Silences)
    assert isinstance(loaded[1], Attributions)
    assert [rule.value for rule in loaded[1].rules] == ["first", "last"]
    assert [rule.index for rule in loaded[1].rules] == [0, 1]
    assert serialize(loaded) == payload
    words = {"Paul": "Pawl"}
    pronunciation = deserialize(
        {"kenkui_sidecar": 1, "pronunciations": [{"where": {}, "words": words}]}
    )
    words["Paul"] = "changed"
    assert isinstance(pronunciation[0], Pronunciations)
    assert pronunciation[0].rules[0].value == (("Paul", "Pawl"),)


@pytest.mark.parametrize(
    "payload",
    [
        None,
        [],
        {},
        {"kenkui_sidecar": True},
        {"kenkui_sidecar": 1.0},
        {"kenkui_sidecar": "1"},
        {"kenkui_sidecar": SIDECAR_VERSION, "unknown": []},
        {"kenkui_sidecar": SIDECAR_VERSION, "attributions": {}},
        {"kenkui_sidecar": SIDECAR_VERSION, "attributions": [None]},
    ],
)
def test_invalid_payload_schema_is_refused(payload: object) -> None:
    """Only supported versioned objects with known array sections load."""
    with pytest.raises(ValidationError) as error:
        deserialize(payload)
    assert error.value.code == ErrorCode.INVALID_SIDECAR


@pytest.mark.parametrize(
    "entry",
    [
        {"character": "a"},
        {"where": {}},
        {"where": [], "character": "a"},
        {"where": {"word": 1}, "character": "a"},
        {"where": {"paragraph": True}, "character": "a"},
        {"where": {"paragraph": "3..1"}, "character": "a"},
        {"where": {}, "character": ""},
        {"where": {}, "character": 3},
        {"where": {}, "character": "a", "unknown": 1},
        {"where": {}, "character": "a", "digest": "wrong"},
        {"where": {}, "character": "a", "digest": None},
        {"where": {}, "character": "a", "matched": -1},
        {"where": {}, "character": "a", "matched": True},
        {"where": {}, "character": "a", "matched": 1.0},
        {"where": {}, "character": "a", "matched": None},
        {
            "where": {},
            "character": "a",
            "digest": "sha256:0123456789abcdef",
            "matched": 1,
        },
    ],
)
def test_invalid_rule_schema_is_refused(entry: object) -> None:
    """Invalid paths, values and drift metadata all use INVALID_SIDECAR."""
    with pytest.raises(ValidationError) as error:
        deserialize({"kenkui_sidecar": 1, "attributions": [entry]})
    assert error.value.code == ErrorCode.INVALID_SIDECAR


@pytest.mark.parametrize(
    ("section", "entry"),
    [
        ("silences", {"where": {}, "ms": True}),
        ("silences", {"where": {}, "ms": -1}),
        ("silences", {"where": {}, "ms": 60_001}),
        ("silences", {"where": {}, "ms": 1.5}),
        ("pronunciations", {"where": {}, "words": []}),
        ("pronunciations", {"where": {}, "words": {"Paul": 1}}),
        ("pronunciations", {"where": {}, "words": {"Paul": ""}}),
        ("pronunciations", {"where": {}, "words": {"Paul": "Pawl", "paul": "Paul"}}),
    ],
)
def test_silence_and_lexicon_validation(section: str, entry: object) -> None:
    """Loading applies the same numeric and lexicon bounds as inline tuning."""
    with pytest.raises(ValidationError) as error:
        deserialize({"kenkui_sidecar": 1, section: [entry]})
    assert error.value.code == ErrorCode.INVALID_SIDECAR


def test_load_does_not_inspect_source(epub_path: Path, tmp_path: Path) -> None:
    """Loading is independent of EPUB availability and model resolution."""
    written = kk.book(epub_path).attribute("a").write_annotations(tmp_path / "s.json")
    missing = kk.book(tmp_path / "not-present.epub").annotations(written)
    assert any(isinstance(op, Attributions) for op in missing.operations)


def test_saving_builds_only_touched_grids_and_ignores_render_selection(
    epub_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Identity selection cannot silently remove tuning anchored elsewhere."""
    inspected: list[str] = []

    def record(chapter: ChapterInspection) -> tuple[Unit, ...]:
        inspected.append(chapter.id)
        return build_grid(chapter)

    monkeypatch.setattr(sidecar, "build_grid", record)
    book = (
        kk.book(epub_path)
        .select_chapters(CH09_ID)
        .attribute("a", where={"chapter": CH08_ID})
    )
    payload = json.loads(book.write_annotations().read_text())
    assert inspected == [CH08_ID]
    assert "digest" in payload["attributions"][0]


@pytest.mark.parametrize(
    "body", ["<p>First. Next.</p><p>First. Next.</p>", "<p>First.</p><p>Next.</p>"]
)
def test_sparse_selection_of_distinct_subtrees_uses_count(
    tmp_path: Path, body: str
) -> None:
    """Distinct addressed subtrees require counts even when they are adjacent."""
    epub = make_epub(
        tmp_path / "sparse.epub",
        chapters={"ch": xhtml(body)},
        spine=["ch"],
    )
    chapter = kk.book(epub).inspect().chapters[0]
    book = kk.book(epub).attribute("a", where={"chapter": chapter.id, "sentence": 1})
    payload = json.loads(book.write_annotations().read_text())
    assert payload["attributions"][0]["matched"] == body.count("<p>")
    assert "digest" not in payload["attributions"][0]


def test_rewriting_refreshes_anchors_without_mutating_loaded_snapshot(
    epub_path: Path,
) -> None:
    """Saving refreshes the file while an existing loaded digest stays frozen."""
    book = kk.book(epub_path).attribute("a", where={"chapter": CH08_ID})
    written = book.write_annotations()
    loaded = kk.book(epub_path).annotations(written)
    original = loaded.operations
    make_epub(
        epub_path, chapters={"ch08": xhtml("<p>Changed text.</p>")}, spine=["ch08"]
    )
    loaded.write_annotations()
    refreshed = kk.book(epub_path).annotations(written)
    assert loaded.operations == original
    assert loaded.operations != refreshed.operations
    assert (
        next(op for op in loaded.operations if isinstance(op, Attributions))
        .rules[0]
        .digest
        != next(op for op in refreshed.operations if isinstance(op, Attributions))
        .rules[0]
        .digest
    )


def test_failed_publication_preserves_previous_file(
    epub_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A failed atomic replace neither truncates saved work nor leaves a temp."""
    written = kk.book(epub_path).attribute("original").write_annotations()
    original = written.read_bytes()

    def fail_replace(_self: Path, _target: Path) -> None:
        raise OSError

    monkeypatch.setattr(Path, "replace", fail_replace)
    with pytest.raises(ValidationError) as error:
        kk.book(epub_path).attribute("changed").write_annotations()
    assert error.value.code == ErrorCode.INVALID_SIDECAR
    assert written.read_bytes() == original
    assert list(written.parent.glob(f".{written.name}.*.tmp")) == []


def test_sidecar_cannot_overwrite_source(epub_path: Path) -> None:
    """An explicit annotation destination cannot replace the source EPUB."""
    original = epub_path.read_bytes()
    with pytest.raises(ValidationError) as error:
        kk.book(epub_path).attribute("a").write_annotations(epub_path)
    assert error.value.code == ErrorCode.INVALID_SIDECAR
    assert epub_path.read_bytes() == original


def test_file_errors_are_sanitized(epub_path: Path, tmp_path: Path) -> None:
    """Missing paths, directories and invalid UTF-8 raise a public error."""
    invalid = tmp_path / "invalid.json"
    invalid.write_bytes(b"\xff")
    for path in (tmp_path / "missing.json", tmp_path, invalid):
        with pytest.raises(ValidationError) as error:
            kk.book(epub_path).annotations(path)
        assert error.value.code == ErrorCode.INVALID_SIDECAR
    with pytest.raises(ValidationError) as error:
        kk.book(epub_path).attribute("a").write_annotations(
            tmp_path / "missing" / "s.json"
        )
    assert error.value.code == ErrorCode.INVALID_SIDECAR


def test_empty_sidecar_is_supported_and_stable(tmp_path: Path) -> None:
    """A version-only document loads once and saves without inspecting an EPUB."""
    book = kk.book(tmp_path / "Title. Author.epub")
    written = book.write_annotations()
    assert sidecar_path(book.source.path) == tmp_path / "Title. Author.kenkui.json"
    assert json.loads(written.read_text()) == {"kenkui_sidecar": SIDECAR_VERSION}
    loaded = book.annotations()
    original = written.read_bytes()
    loaded.write_annotations()
    assert written.read_bytes() == original
