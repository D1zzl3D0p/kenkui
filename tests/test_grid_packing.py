"""Hierarchical packing over transformed addressable-grid text."""

from __future__ import annotations

import random
from pathlib import Path as FilePath

import pytest

from kenkui._domain.grid import build_grid, build_structure_index
from kenkui._domain.grid_packing import (
    FallbackCut,
    PackedRange,
    PackingInput,
    SpokenMapping,
    SpokenRegion,
    pack_grid,
)
from kenkui.inspection import ChapterInspection


def chapter(text: str) -> ChapterInspection:
    """Build one minimal canonical chapter."""
    return ChapterInspection(
        id="ch01",
        index=0,
        title="One",
        speech_characters=len(text),
        text=text,
        emphasis=(),
    )


def request(
    canonical: str,
    *,
    budget: int,
    spoken: str | None = None,
    mappings: tuple[SpokenMapping, ...] = (),
    mandatory: frozenset[int] = frozenset(),
) -> PackingInput:
    """Build a full-chapter packer input."""
    leaves = build_grid(chapter(canonical))
    return PackingInput(
        leaves=leaves,
        ranges=build_structure_index(leaves),
        spoken_regions=(
            SpokenRegion(0, len(canonical), spoken or canonical, mappings),
        ),
        mandatory_cuts=mandatory,
        character_budget=budget,
    )


def texts(packed: tuple[PackedRange, ...], spoken: str) -> tuple[str, ...]:
    """Return spoken slices without making the packer own duplicate text."""
    return tuple(spoken[item.spoken_start : item.spoken_end] for item in packed)


def test_descends_to_sentences_then_greedily_combines_adjacent_pieces() -> None:
    """An over-budget line descends; later fitting sentences still combine."""
    source = "A sentence. Another one. Third."

    packed = pack_grid(request(source, budget=20))

    assert texts(packed, source) == ("A sentence. ", "Another one. Third.")
    assert [(item.canonical_start, item.canonical_end) for item in packed] == [
        (0, 12),
        (12, len(source)),
    ]


def test_descends_to_phrases_before_using_an_emergency_cut() -> None:
    """Clause-level leaves are ordinary candidates, not fallback cuts."""
    source = "One, two, three."

    packed = pack_grid(request(source, budget=10))

    assert texts(packed, source) == ("One, two, ", "three.")
    assert all(item.fallback_cut_after is None for item in packed)


def test_mandatory_cut_prevents_an_otherwise_fitting_join() -> None:
    """Semantic boundaries divide packing intervals before greedy joining."""
    source = "One. Two. Three."
    cut = len("One. Two. ")

    packed = pack_grid(request(source, budget=100, mandatory=frozenset({cut})))

    assert texts(packed, source) == (source[:cut], source[cut:])
    assert packed[0].canonical_end == cut
    assert packed[1].canonical_start == cut


@pytest.mark.parametrize(
    ("source", "budget", "expected", "reason"),
    [
        (
            "abcdefghij-klmnopqrst",
            12,
            ("abcdefghij-", "klmnopqrst"),
            FallbackCut.PUNCTUATION_OR_HYPHEN,
        ),
        (
            "alpha beta gamma",
            10,
            ("alpha ", "beta gamma"),
            FallbackCut.WHITESPACE,
        ),
        (
            "x" * 25,
            10,
            ("x" * 10, "x" * 10, "x" * 5),
            FallbackCut.HARD_TOKEN,
        ),
    ],
)
def test_isolated_leaf_fallback_is_ranked_and_characterized(
    source: str,
    budget: int,
    expected: tuple[str, ...],
    reason: FallbackCut,
) -> None:
    """Fallback prefers punctuation/hyphen, then whitespace, then hard cuts."""
    packed = pack_grid(request(source, budget=budget))

    assert texts(packed, source) == expected
    assert all(item.fallback_cut_after is reason for item in packed[:-1])
    assert packed[-1].fallback_cut_after is None


def test_spoken_expansion_controls_fit_while_ranges_remain_canonical() -> None:
    """Packing budgets transformed text and projects its edges back to source."""
    canonical = "IV then."
    spoken = "Four then."
    mapping = SpokenMapping(0, 2, 0, 4)

    packed = pack_grid(request(canonical, spoken=spoken, mappings=(mapping,), budget=6))

    assert texts(packed, spoken) == ("Four ", "then.")
    assert [(item.canonical_start, item.canonical_end) for item in packed] == [
        (0, 3),
        (3, len(canonical)),
    ]


def test_regions_keep_independent_transformations_and_mandatory_edge() -> None:
    """Region-local spoken forms concatenate without crossing their shared cut."""
    canonical = "IV lead"
    spoken = "Four leed"
    leaves = build_grid(chapter(canonical))
    packing = PackingInput(
        leaves=leaves,
        ranges=build_structure_index(leaves),
        spoken_regions=(
            SpokenRegion(0, 2, "Four", (SpokenMapping(0, 2, 0, 4),)),
            SpokenRegion(2, 7, " leed", (SpokenMapping(3, 7, 1, 5),)),
        ),
        mandatory_cuts=frozenset({2}),
        character_budget=100,
    )

    packed = pack_grid(packing)

    assert texts(packed, spoken) == ("Four", " leed")
    assert [(item.canonical_start, item.canonical_end) for item in packed] == [
        (0, 2),
        (2, 7),
    ]


def test_fallback_inside_one_expanded_token_uses_stable_canonical_envelopes() -> None:
    """Spoken subranges disambiguate emergency pieces inside one replacement."""
    canonical = "X"
    spoken = "abcdefghij"
    mapping = SpokenMapping(0, 1, 0, len(spoken))

    packed = pack_grid(request(canonical, spoken=spoken, mappings=(mapping,), budget=4))

    assert texts(packed, spoken) == ("abcd", "efgh", "ij")
    assert all((item.canonical_start, item.canonical_end) == (0, 1) for item in packed)
    assert [item.fallback_cut_after for item in packed] == [
        FallbackCut.HARD_TOKEN,
        FallbackCut.HARD_TOKEN,
        None,
    ]


def test_ordinary_boundaries_are_grid_or_mandatory_edges() -> None:
    """Only explicitly characterized fallback cuts may land within one leaf."""
    source = "First, clause. Second sentence.\n\nFinal paragraph."
    cut = source.index("Second")
    packed = pack_grid(request(source, budget=17, mandatory=frozenset({cut})))
    leaves = build_grid(chapter(source))
    grid_edges = {leaf.start for leaf in leaves} | {leaf.end for leaf in leaves}

    for item in packed[:-1]:
        if item.fallback_cut_after is None:
            assert item.canonical_end in grid_edges | {cut}


def test_randomized_packing_is_exact_bounded_deterministic_and_mandatory() -> None:
    """Generated prose preserves every spoken character under varied budgets."""
    rng = random.Random(81357)  # noqa: S311 - deterministic property fixture
    vocabulary = ("alpha", "beta", "gamma", "delta", "epsilon", "zeta")
    separators = (" ", ", ", "; ", ". ", "\n", "\n\n", "-")
    for _case in range(100):
        source = (
            "".join(
                word + rng.choice(separators)
                for word in rng.choices(vocabulary, k=rng.randint(2, 25))
            )
            + "omega."
        )
        leaves = build_grid(chapter(source))
        edges = sorted({leaf.start for leaf in leaves} | {leaf.end for leaf in leaves})
        mandatory = frozenset(rng.sample(edges[1:-1], k=min(2, len(edges) - 2)))
        packing = request(
            source,
            budget=rng.randint(4, 40),
            mandatory=mandatory,
        )

        first = pack_grid(packing)
        second = pack_grid(packing)

        assert first == second
        assert "".join(texts(first, source)) == source
        assert all(
            0 < item.spoken_end - item.spoken_start <= packing.character_budget
            for item in first
        )
        assert mandatory <= {item.canonical_end for item in first[:-1]}


@pytest.mark.parametrize("budget", [0, -1, True])
def test_budget_must_be_a_positive_integer(budget: int) -> None:
    """Invalid hard ceilings fail before producing a partial result."""
    with pytest.raises(ValueError, match="positive budget"):
        pack_grid(request("Text.", budget=budget))


def test_mandatory_cut_inside_a_spoken_replacement_is_rejected() -> None:
    """Callers must transform semantic regions independently at their edges."""
    mapping = SpokenMapping(0, 2, 0, 4)
    with pytest.raises(ValueError, match="inside a spoken replacement"):
        pack_grid(
            request(
                "IV then.",
                spoken="Four then.",
                mappings=(mapping,),
                budget=100,
                mandatory=frozenset({1}),
            )
        )


def test_packer_has_no_pipeline_attribution_synthesis_or_pause_imports() -> None:
    """The packing module remains a pure domain leaf/index consumer."""
    source = (
        FilePath(__file__).parents[1] / "src" / "kenkui" / "_domain" / "grid_packing.py"
    ).read_text()
    forbidden = ("pipeline", "_characters", "_tts", "planning", "Pauses")
    assert all(token not in source for token in forbidden)
