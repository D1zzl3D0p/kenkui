"""Exhaustive classification of immutable operation families."""

from kenkui._domain import operations as ops
from kenkui._domain.tiers import tier_of


def test_every_operation_is_classified() -> None:
    """A new operation must not disappear from all three tier views."""
    unclassified = [
        cls.__name__
        for cls in vars(ops).values()
        if isinstance(cls, type)
        and issubclass(cls, ops.Operation)
        and tier_of(cls) is None
    ]
    assert unclassified == []


def test_tuning_operations_are_the_ones_that_serialize() -> None:
    """Human corrections are the portable tuning tier."""
    assert tier_of(ops.Attributions) == "tuning"
    assert tier_of(ops.Silences) == "tuning"
    assert tier_of(ops.Pronunciations) == "tuning"
    assert tier_of(ops.Annotations) == "tuning"


def test_style_and_identity_are_separated() -> None:
    """Production choices differ from identity and chapter selection."""
    assert tier_of(ops.Pauses) == "style"
    assert tier_of(ops.AssignVoices) == "style"
    assert tier_of(ops.SynthesizeSpeech) == "style"
    assert tier_of(ops.SpokenForm) == "style"
    assert tier_of(ops.InferCharacters) == "style"
    assert tier_of(ops.AttributeQuotes) == "style"
    assert tier_of(ops.MetadataIntent) == "identity"
    assert tier_of(ops.Series) == "identity"
    assert tier_of(ops.SelectChapters) == "identity"
    assert tier_of(ops.Select) == "identity"
    assert tier_of(ops.SelectChapterRange) == "identity"


def test_unknown_operation_is_unclassified() -> None:
    """Unknown classes do not silently receive a default tier."""
    assert tier_of(object) is None  # type: ignore[arg-type]
