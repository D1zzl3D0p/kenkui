"""Accumulating tuning and replaceable pipeline intent contracts."""

from __future__ import annotations

from dataclasses import FrozenInstanceError
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

import kenkui as kk
from kenkui._domain.operations import (
    Annotations,
    Attributions,
    Pauses,
    Pronunciations,
    Silences,
    SpokenForm,
    SynthesizeSpeech,
    replace_or_append,
)
from kenkui._domain.paths import parse_pattern

if TYPE_CHECKING:
    from collections.abc import Callable

    from kenkui.pipeline import WhereArg


def test_attribute_accumulates_in_declaration_order(epub_path: Path) -> None:
    """One operation holds ordered rules while branches remain independent."""
    base = kk.book(epub_path)
    first = base.attribute("irulan")
    branch = first.attribute("paul", where={"chapter": "ch08"})
    operations = [op for op in branch.operations if isinstance(op, Attributions)]
    assert len(operations) == 1
    assert [rule.value for rule in operations[0].rules] == ["irulan", "paul"]
    assert [rule.index for rule in operations[0].rules] == [0, 1]
    assert operations[0].rules[0].where.is_whole_book()
    assert operations[0].rules[1].where == parse_pattern({"chapter": "ch08"})
    assert base.operations == ()
    assert len(first.operations[0].rules) == 1  # type: ignore[union-attr]


def test_tuning_accepts_patterns_and_freezes_where(epub_path: Path) -> None:
    """Patterns and sparse raw mappings select the same immutable coordinates."""
    where: dict[str, object] = {"sentence": [1, 3]}
    expected = parse_pattern(where)
    branch = (
        kk.book(epub_path).attribute("a", where=where).attribute("b", where=expected)
    )
    where["sentence"] = 5
    operation = branch.operations[0]
    assert isinstance(operation, Attributions)
    assert all(rule.where == expected for rule in operation.rules)
    with pytest.raises(FrozenInstanceError):
        operation.rules = ()  # type: ignore[misc]


@pytest.mark.parametrize(
    ("tune", "kind", "value"),
    [
        (lambda b, w: b.attribute("paul", where=w), Attributions, "paul"),
        (lambda b, w: b.silence(900, where=w), Silences, 900),
        (
            lambda b, w: b.pronounce({"Paul": "Pawl"}, where=w),
            Pronunciations,
            (("Paul", "Pawl"),),
        ),
    ],
)
def test_tuple_selectors_accumulate_in_order_and_freeze_inputs(
    resolved_book: kk.Pipeline,
    tune: Callable[[kk.Pipeline, WhereArg], kk.Pipeline],
    kind: type[Attributions | Silences | Pronunciations],
    value: object,
) -> None:
    """Mixed tuples and mapping tuples add consecutive immutable declarations."""
    sentences = [1, 3]
    first: dict[str, object] = {"sentence": sentences}
    second = parse_pattern({"chapter": "ch08"})
    third: dict[str, object] = {"paragraph": 2}
    fourth: dict[str, object] = {"phrase": 1}
    expected = [parse_pattern({}), parse_pattern(first), second]
    expected.extend([parse_pattern(third), parse_pattern(fourth)])

    base = tune(resolved_book, None)
    branch = tune(tune(base, (first, second)), (third, fourth))
    sentences.append(5)
    first["chapter"] = "changed"
    third["paragraph"] = 7
    fourth.clear()

    operations = [op for op in branch.operations if isinstance(op, kind)]
    assert len(operations) == 1
    assert [rule.where for rule in operations[0].rules] == expected
    assert [rule.index for rule in operations[0].rules] == list(range(len(expected)))
    assert [rule.value for rule in operations[0].rules] == [value] * len(expected)
    assert len(next(op.rules for op in base.operations if isinstance(op, kind))) == 1
    assert branch._resolved is resolved_book._resolved  # noqa: SLF001
    assert branch._roster is resolved_book._roster  # noqa: SLF001
    assert branch.resolve() is branch


@pytest.mark.parametrize(
    "tune",
    [
        lambda b, w: b.attribute("paul", where=w),
        lambda b, w: b.silence(900, where=w),
        lambda b, w: b.pronounce({"Paul": "Pawl"}, where=w),
    ],
)
def test_invalid_tuple_member_is_rejected_eagerly(
    epub_path: Path, tune: Callable[[kk.Pipeline, WhereArg], kk.Pipeline]
) -> None:
    """Every tuple entry is validated before a public call returns a branch."""
    base = kk.book(epub_path)
    with pytest.raises(kk.ValidationError) as error:
        tune(base, ({"chapter": "ch08"}, {"sentence": 0}))
    assert error.value.code == kk.ErrorCode.INVALID_PATTERN
    assert base.operations == ()


def test_silence_accumulates_and_zero_is_allowed(epub_path: Path) -> None:
    """A zero-valued silence rule explicitly removes a derived pause."""
    branch = (
        kk.book(epub_path)
        .silence(900, where={"chapter": "ch08"})
        .silence(0, where={"chapter": "ch09"})
        .silence(60_000)
    )
    operation = branch.operations[0]
    assert isinstance(operation, Silences)
    assert [rule.value for rule in operation.rules] == [900, 0, 60_000]
    assert [rule.index for rule in operation.rules] == [0, 1, 2]


@pytest.mark.parametrize("duration", [-1, 60_001, True, 1.5, "10", None])
def test_invalid_silence_is_refused(epub_path: Path, duration: object) -> None:
    """Only integer millisecond values within the pause budget are accepted."""
    with pytest.raises(kk.ValidationError) as error:
        kk.book(epub_path).silence(duration)  # type: ignore[arg-type]
    assert error.value.code == kk.ErrorCode.INVALID_PAUSE


def test_pronunciation_rules_accumulate_and_style_replaces(epub_path: Path) -> None:
    """Scoped lexicons compose; the latest call chooses global spoken features."""
    lexicon = {"Paul": "Pawl"}
    branch = (
        kk.book(epub_path)
        .pronounce(lexicon, numbers="off")
        .pronounce({"Paul": "Pol"}, where={"chapter": "ch08"}, numbers="standard")
        .pronounce(numbers="aggressive", builtin=False, roman=False)
    )
    lexicon["Paul"] = "mutated"
    rules = next(op.rules for op in branch.operations if isinstance(op, Pronunciations))
    assert [rule.value for rule in rules] == [(("Paul", "Pawl"),), (("Paul", "Pol"),)]
    assert [rule.index for rule in rules] == [0, 1]
    assert rules[0].where.is_whole_book()
    assert rules[1].where == parse_pattern({"chapter": "ch08"})
    spoken = next(op for op in branch.operations if isinstance(op, SpokenForm))
    assert spoken.numbers == "aggressive"
    assert spoken.builtin_lexicon is False
    assert spoken.features == (("roman", False),)
    assert spoken.lexicon == ()


@pytest.mark.parametrize(
    "tune",
    [
        lambda book: book.attribute("narrator"),
        lambda book: book.silence(0),
        lambda book: book.pronounce({"Paul": "Pawl"}),
        lambda book: book.pauses(paragraph_ms=500),
        lambda book: book.metadata(title="Revised"),
    ],
)
def test_tuning_preserves_resolution(
    resolved_book: kk.Pipeline, tune: Callable[[kk.Pipeline], kk.Pipeline]
) -> None:
    """Human tuning consumes the existing machine snapshot without resolving again."""
    branch = tune(resolved_book)
    assert branch._resolved is resolved_book._resolved  # noqa: SLF001
    assert branch._roster is resolved_book._roster  # noqa: SLF001
    assert branch.resolve() is branch


@pytest.mark.parametrize(
    ("first", "second"),
    [
        (lambda b: b.pauses(paragraph_ms=400), lambda b: b.pauses(paragraph_ms=500)),
        (lambda b: b.assign_voice("a"), lambda b: b.assign_voice("b")),
        (lambda b: b.metadata(title="a"), lambda b: b.metadata(title="b")),
        (lambda b: b.series("a"), lambda b: b.series("b")),
        (lambda b: b.infer_characters("a"), lambda b: b.infer_characters("b")),
        (lambda b: b.attribute_quotes("a"), lambda b: b.attribute_quotes("b")),
    ],
)
def test_style_and_identity_re_calls_replace(
    epub_path: Path,
    first: Callable[[kk.Pipeline], kk.Pipeline],
    second: Callable[[kk.Pipeline], kk.Pipeline],
) -> None:
    """Replace settings at their original position with last-call-wins semantics."""
    base = kk.book(epub_path)
    original = first(base)
    assert second(original).operations == second(base).operations
    assert original.operations == first(base).operations


@pytest.mark.parametrize(
    "retune",
    [
        lambda book: book.assign_voice("different"),
        lambda book: book.attribute_quotes("different/model"),
        lambda book: book.infer_characters("different/model"),
        lambda book: book.series("different"),
    ],
)
def test_model_and_cast_changes_invalidate_resolution(
    resolved_book: kk.Pipeline, retune: Callable[[kk.Pipeline], kk.Pipeline]
) -> None:
    """Replacement must not reuse attribution or casting derived from old intent."""
    assert retune(resolved_book)._resolved is None  # noqa: SLF001
    assert resolved_book._resolved is not None  # noqa: SLF001


def test_replacement_preserves_position_and_tts_order(epub_path: Path) -> None:
    """Existing settings remain editable after TTS; new pre-TTS stages are refused."""
    base = kk.book(epub_path).pauses(paragraph_ms=400).assign_voice("ivy").tts()
    branch = base.pauses(paragraph_ms=500).tts().metadata(title="After synthesis")
    assert branch.operations[0] == Pauses(paragraph_ms=500)
    assert sum(isinstance(op, SynthesizeSpeech) for op in branch.operations) == 1
    with pytest.raises(kk.ValidationError) as error:
        base.silence(500)
    assert error.value.code == kk.ErrorCode.INVALID_OPERATION_ORDER
    assert replace_or_append((Pauses(), SynthesizeSpeech()), Pauses(line_ms=1)) == (
        Pauses(line_ms=1),
        SynthesizeSpeech(),
    )


@pytest.mark.parametrize(
    ("first", "second"),
    [
        (lambda b: b.select_chapters("a"), lambda b: b.select_chapters("b")),
        (lambda b: b.select_chapters("a"), lambda b: b.select_chapter_range("a", "b")),
        (lambda b: b.select_chapter_range("a", "b"), lambda b: b.select_chapters("a")),
        (
            lambda b: b.select_chapter_range("a", "b"),
            lambda b: b.select_chapter_range("b", "c"),
        ),
    ],
)
def test_selection_operations_still_refuse_duplicates(
    epub_path: Path,
    first: Callable[[kk.Pipeline], kk.Pipeline],
    second: Callable[[kk.Pipeline], kk.Pipeline],
) -> None:
    """Selecting twice remains incompatible in both directions."""
    with pytest.raises(kk.ValidationError) as error:
        second(first(kk.book(epub_path)))
    assert error.value.code == kk.ErrorCode.DUPLICATE_OPERATION


def test_annotations_defensively_freeze_loaded_data() -> None:
    """A frozen annotation operation owns a read-only snapshot of its input."""
    loaded = {"chapter": 10}
    operation = Annotations(Path("timing.json"), "digest", loaded)
    loaded["chapter"] = 20
    assert operation.loaded == {"chapter": 10}
    with pytest.raises(TypeError):
        operation.loaded["chapter"] = 30  # type: ignore[index]
