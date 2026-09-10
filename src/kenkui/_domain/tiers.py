"""Explicit tier classification for every semantic operation family."""

from typing import Literal, TypeAlias

from kenkui._domain import operations as ops

Tier: TypeAlias = Literal["identity", "tuning", "style"]

_TIERS: dict[type[ops.Operation], Tier] = {
    ops.SelectChapters: "identity",
    ops.Select: "identity",
    ops.SelectChapterRange: "identity",
    ops.MetadataIntent: "identity",
    ops.Series: "identity",
    ops.Attributions: "tuning",
    ops.Silences: "tuning",
    ops.Pronunciations: "tuning",
    ops.Annotations: "tuning",
    ops.SpokenForm: "style",
    ops.Pauses: "style",
    ops.InferCharacters: "style",
    ops.AttributeQuotes: "style",
    ops.AssignVoices: "style",
    ops.SynthesizeSpeech: "style",
}


def tier_of(operation: type[ops.Operation]) -> Tier | None:
    """Return the declared tier, leaving unknown families unclassified."""
    return _TIERS.get(operation)
