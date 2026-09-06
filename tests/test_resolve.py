"""resolve() steps a pipeline forward without changing what it means."""

from __future__ import annotations

import pytest

import kenkui as kk
from kenkui._domain.operations import AssignVoices, AttributeQuotes, InferCharacters

_THREE_OPERATIONS = 3


def _pipeline(model: str = "fake/model") -> kk.Pipeline:
    return (
        kk.epub("book.epub")
        .infer_characters(model=model)
        .attribute_quotes(model=model)
        .assign_voices(narrator="eponine", method="gendered")
    )


def test_building_a_pipeline_performs_no_work() -> None:
    """Construction records intent; nothing is parsed, fetched, or inferred."""
    pipeline = _pipeline()
    assert len(pipeline.operations) == _THREE_OPERATIONS


def test_a_binding_bug_is_not_retried_with_different_casting(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A resolver TypeError must not silently fall back to another signature."""
    calls: list[str] = []

    def broken_binding(voice_id: str) -> None:
        calls.append(voice_id)
        message = "binding implementation bug"
        raise TypeError(message)

    monkeypatch.setattr("kenkui.pipeline._execution_bindings", broken_binding)
    with pytest.raises(TypeError, match="binding implementation bug"):
        kk.epub("book.epub").assign_voice("eponine").resolve()
    assert calls == ["eponine"]


def test_assign_voice_is_the_degenerate_cast() -> None:
    """One VoicePlan and one renderer serve single and multi voice alike."""
    single = kk.epub("book.epub").assign_voice("eponine")
    plural = kk.epub("book.epub").assign_voices(narrator="eponine")
    assert single.operations == plural.operations


def test_cancelled_resolution_never_binds_resources(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An already cancelled request must perform no resolution work."""

    def unexpected_binding(_voice_id: str) -> None:
        pytest.fail("cancelled resolution reached voice binding")

    monkeypatch.setattr("kenkui.pipeline._execution_bindings", unexpected_binding)
    token = kk.CancellationToken()
    token.cancel()
    with pytest.raises(kk.CancelledError):
        kk.epub("book.epub").assign_voice("eponine").resolve(cancel=token)


def test_unknown_defaults_to_the_narrator_voice() -> None:
    """A line nobody could place sounds like narration rather than vanishing."""
    casting = kk.epub("book.epub").assign_voices(narrator="eponine").operations[-1]
    assert isinstance(casting, AssignVoices)
    assert casting.unknown_voice_id == "eponine"


def test_unknown_can_be_set_independently() -> None:
    """Unattributed speech can be made audibly distinct from narration."""
    casting = (
        kk.epub("book.epub")
        .assign_voices(narrator="eponine", unknown="paul")
        .operations[-1]
    )
    assert isinstance(casting, AssignVoices)
    assert casting.unknown_voice_id == "paul"


def test_the_cast_is_order_independent() -> None:
    """Two callers writing the same cast differently must record it the same."""
    forward = kk.epub("book.epub").assign_voices(
        narrator="eponine", cast={"a": "anna", "b": "vera"}
    )
    reverse = kk.epub("book.epub").assign_voices(
        narrator="eponine", cast={"b": "vera", "a": "anna"}
    )
    assert forward.operations == reverse.operations


def test_operation_order_does_not_matter() -> None:
    """Only the before_tts rule constrains ordering; the planner reads by type."""
    forward = _pipeline()
    reverse = (
        kk.epub("book.epub")
        .assign_voices(narrator="eponine", method="gendered")
        .attribute_quotes(model="fake/model")
        .infer_characters(model="fake/model")
    )
    assert set(map(type, forward.operations)) == set(map(type, reverse.operations))


def test_resolve_leaves_the_receiver_alone() -> None:
    """Immutable in shape even though it is an effect."""
    original = _pipeline()
    assert original._resolved is None  # noqa: SLF001


def test_appending_drops_resolved_values() -> None:
    """Changing intent invalidates resolution; re-resolving is a store lookup."""
    resolved = kk.Pipeline(
        kk.epub("book.epub").source,
        _pipeline().operations,
        object(),  # type: ignore[arg-type]
    )
    assert resolved.tts()._resolved is None  # noqa: SLF001


def test_resolved_values_are_not_intent() -> None:
    """They must never reach the plan or its fingerprint."""
    base = _pipeline()
    carrying = kk.Pipeline(base.source, base.operations, object())  # type: ignore[arg-type]
    assert carrying.operations == base.operations


def test_attributing_without_a_roster_is_rejected_by_presence_not_order() -> None:
    """A presence rule, so chaining order stays free."""
    issues = (
        kk.epub("book.epub")
        .attribute_quotes(model="fake/model")
        .assign_voices(narrator="eponine")
        .validate()
        .issues
    )
    assert kk.ErrorCode.ATTRIBUTION_UNAVAILABLE in {issue.code for issue in issues}


def test_inferring_alone_is_fine() -> None:
    """A roster with no attribution is harmless: nothing consumes it."""
    issues = (
        kk.epub("book.epub")
        .infer_characters(model="fake/model")
        .assign_voices(narrator="eponine")
        .validate()
        .issues
    )
    assert kk.ErrorCode.ATTRIBUTION_UNAVAILABLE not in {issue.code for issue in issues}


def test_the_model_id_must_be_real() -> None:
    """A blank model id would reach the provider as a broken request."""
    for blank in ("", "   "):
        try:
            kk.epub("book.epub").infer_characters(model=blank)
        except kk.ValidationError:
            continue
        msg = f"blank model {blank!r} was accepted"
        raise AssertionError(msg)


def test_operations_are_named_for_what_they_do() -> None:
    """The three character operations, and nothing else."""
    kinds = {type(op) for op in _pipeline().operations}
    assert kinds == {InferCharacters, AttributeQuotes, AssignVoices}


def test_an_unknown_casting_method_is_rejected_at_intent_time() -> None:
    """A typo must fail where it is written, not silently render one voice.

    The solver checks the method too, but it only runs when attribution is
    configured and some character is left for it to place. A single-voice
    pipeline reaches neither, so without this the run just narrates.
    """
    with pytest.raises(kk.ValidationError) as caught:
        kk.epub("book.epub").assign_voices(narrator="eponine", method="astrology")
    assert caught.value.code is kk.ErrorCode.CASTING_METHOD_UNKNOWN


def test_a_non_path_cover_is_invalid_intent_rather_than_a_crash() -> None:
    """metadata() validates its own argument instead of raising TypeError."""
    with pytest.raises(kk.ValidationError) as caught:
        kk.epub("book.epub").metadata(cover=17)  # type: ignore[arg-type]
    assert caught.value.code is kk.ErrorCode.INVALID_METADATA


def test_a_cast_without_attribution_is_rejected() -> None:
    """Named characters are unreachable when nothing attributes quotes.

    Resolution returns early with no assignments when attribution is absent,
    so the cast and a distinct unknown voice are discarded in silence and the
    book renders entirely in the narrator's voice.
    """
    issues = (
        kk.epub("book.epub")
        .assign_voices(narrator="eponine", cast={"javert": "charles"})
        .tts()
        .validate()
        .issues
    )
    assert kk.ErrorCode.CAST_UNATTRIBUTED in {issue.code for issue in issues}


def test_a_distinct_unknown_voice_without_attribution_is_rejected() -> None:
    """The unknown voice only ever speaks lines attribution failed to place."""
    issues = (
        kk.epub("book.epub")
        .assign_voices(narrator="eponine", unknown="charles")
        .tts()
        .validate()
        .issues
    )
    assert kk.ErrorCode.CAST_UNATTRIBUTED in {issue.code for issue in issues}


def test_a_narrator_alone_is_not_a_cast() -> None:
    """Single-voice rendering must stay the frictionless default."""
    issues = (
        kk.epub("book.epub").assign_voices(narrator="eponine").tts().validate().issues
    )
    assert kk.ErrorCode.CAST_UNATTRIBUTED not in {issue.code for issue in issues}
