"""A series is contradicted before a model call, not after paying for one."""

from __future__ import annotations

from kenkui._characters import store
from kenkui._domain.operations import AssignVoices, Series
from kenkui.errors import ErrorCode
from kenkui.validation import series_intent_errors


def _record() -> store.SeriesRecord:
    return store.SeriesRecord(
        series_id="s",
        narrator_voice_id="eponine",
        characters=(
            store.SeriesCharacter(
                "kaladin", "Kaladin", "masculine", "alf", 100, ("Kaladin",)
            ),
        ),
    )


def _operations(**overrides: object) -> tuple[object, ...]:
    # Popped before the dict feeds Series(**series): update() would otherwise
    # copy "narrator" along with it, and Series has no such field.
    narrator = overrides.pop("narrator", "eponine")
    series: dict[str, object] = {"series_id": "s"}
    series.update(overrides)
    return (
        AssignVoices(
            narrator_voice_id=narrator,  # type: ignore[arg-type]
            unknown_voice_id="eponine",
            cast=(),
            method="gendered",
        ),
        Series(**series),  # type: ignore[arg-type]
    )


def test_a_pinned_voice_missing_from_the_pool_fails() -> None:
    """The operator unloaded a voice between volumes."""
    errors = series_intent_errors(_operations(), _record(), frozenset({"aoife"}))
    assert ErrorCode.SERIES_VOICE_MISSING in errors


def test_allow_recast_accepts_the_loss() -> None:
    """Explicit permission lets the render proceed without the lost voice."""
    errors = series_intent_errors(
        _operations(allow_recast=True), _record(), frozenset({"aoife"})
    )
    assert ErrorCode.SERIES_VOICE_MISSING not in errors


def test_a_changed_narrator_fails() -> None:
    """A series narrator changing is almost always a mistake."""
    errors = series_intent_errors(
        _operations(narrator="marius"), _record(), frozenset({"alf", "marius"})
    )
    assert ErrorCode.SERIES_NARRATOR_CHANGED in errors


def test_allow_narrator_change_accepts_it() -> None:
    """Explicit permission lets a new narrator voice proceed."""
    errors = series_intent_errors(
        _operations(narrator="marius", allow_narrator_change=True),
        _record(),
        frozenset({"alf", "marius"}),
    )
    assert ErrorCode.SERIES_NARRATOR_CHANGED not in errors


def test_a_first_volume_has_nothing_to_contradict() -> None:
    """No stored series means no constraint."""
    assert series_intent_errors(_operations(), None, frozenset({"alf"})) == ()


def test_a_pipeline_without_a_series_is_unaffected() -> None:
    """Every existing pipeline keeps validating exactly as it did."""
    only_voices = (
        AssignVoices(
            narrator_voice_id="eponine", unknown_voice_id="eponine",
            cast=(), method="gendered",
        ),
    )
    assert series_intent_errors(only_voices, _record(), frozenset()) == ()
