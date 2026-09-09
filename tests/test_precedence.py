"""Tests for rule precedence and overlap diagnostics."""

# The test names state the behavior without duplicative function docstrings.
# ruff: noqa: D103

from kenkui._domain.grid import Unit
from kenkui._domain.paths import SiblingCounts, parse_pattern
from kenkui._domain.tuning import Rule, overlap_warnings, resolve_rules


def rule(index: int, where: dict[str, object], value: str) -> Rule:
    return Rule(where=parse_pattern(where), value=value, index=index)


def test_rule_has_an_optional_anchor_digest() -> None:
    pattern = parse_pattern({"chapter": "ch08", "paragraph": 3})
    assert Rule(pattern, "paul", 0).digest is None
    assert Rule(pattern, "paul", 0, digest="sha256:abc").digest == "sha256:abc"


def test_no_rules_falls_through_to_machine(
    unit_ch08_p3_s2: Unit, siblings: SiblingCounts
) -> None:
    decision = resolve_rules(unit_ch08_p3_s2, "narrator", (), siblings)
    assert decision.value == "narrator"
    assert decision.provenance == "machine"


def test_no_rules_and_no_machine_is_the_default(
    unit_ch08_p3_s2: Unit, siblings: SiblingCounts
) -> None:
    decision = resolve_rules(unit_ch08_p3_s2, None, (), siblings)
    assert decision.provenance == "default"


def test_any_rule_beats_machine(unit_ch08_p3_s2: Unit, siblings: SiblingCounts) -> None:
    rules = (rule(0, {"chapter": "*"}, "irulan"),)
    decision = resolve_rules(unit_ch08_p3_s2, "paul", rules, siblings)
    assert decision.value == "irulan"
    assert decision.provenance == "rule"


def test_subset_wins_regardless_of_order(
    unit_ch08_p3_s2: Unit, siblings: SiblingCounts
) -> None:
    wide = rule(0, {"chapter": "ch08"}, "paul")
    narrow = rule(1, {"chapter": "ch08", "paragraph": 3, "sentence": 2}, "jessica")
    assert (
        resolve_rules(unit_ch08_p3_s2, None, (wide, narrow), siblings).value
        == "jessica"
    )
    assert (
        resolve_rules(unit_ch08_p3_s2, None, (narrow, wide), siblings).value
        == "jessica"
    )


def test_incomparable_overlap_falls_back_to_declaration_order(
    unit_ch08_p1_s1: Unit, siblings: SiblingCounts
) -> None:
    a = rule(0, {"chapter": "ch08", "paragraph": "*"}, "paul")
    b = rule(1, {"chapter": "*", "paragraph": 1}, "irulan")
    assert resolve_rules(unit_ch08_p1_s1, None, (a, b), siblings).value == "irulan"
    assert resolve_rules(unit_ch08_p1_s1, None, (b, a), siblings).value == "paul"


def test_incomparable_overlap_is_reported() -> None:
    a = rule(0, {"chapter": "ch08", "paragraph": "*"}, "paul")
    b = rule(1, {"chapter": "*", "paragraph": 1}, "irulan")
    assert overlap_warnings((a, b)) == ((0, 1),)


def test_nested_rules_do_not_warn() -> None:
    wide = rule(0, {"chapter": "ch08"}, "paul")
    narrow = rule(1, {"chapter": "ch08", "paragraph": 3}, "jessica")
    assert overlap_warnings((wide, narrow)) == ()


def test_disjoint_rules_do_not_warn() -> None:
    ch08 = rule(3, {"chapter": "ch08"}, "paul")
    ch09 = rule(7, {"chapter": "ch09"}, "jessica")
    assert overlap_warnings((ch08, ch09)) == ()


def test_partially_overlapping_sets_warn_with_rule_indices() -> None:
    a = rule(3, {"paragraph": [1, 2]}, "paul")
    b = rule(7, {"paragraph": "2..3"}, "jessica")
    assert overlap_warnings((a, b)) == ((3, 7),)


def test_rule_index_is_reported_for_provenance(
    unit_ch08_p3_s2: Unit, siblings: SiblingCounts
) -> None:
    rules = (rule(0, {"chapter": "*"}, "a"), rule(1, {"chapter": "ch08"}, "b"))
    assert resolve_rules(unit_ch08_p3_s2, None, rules, siblings).rule_index == 1


def test_later_rule_position_not_rule_index_breaks_incomparable_tie(
    unit_ch08_p1_s1: Unit, siblings: SiblingCounts
) -> None:
    earlier = rule(99, {"chapter": "ch08"}, "earlier")
    later = rule(2, {"paragraph": 1}, "later")
    decision = resolve_rules(unit_ch08_p1_s1, None, (earlier, later), siblings)
    assert decision.value == "later"
    assert decision.rule_index == later.index


def test_dominated_late_rule_cannot_displace_an_incomparable_maximum(
    unit_ch08_p1_s1: Unit, siblings: SiblingCounts
) -> None:
    narrow = rule(0, {"paragraph": 1, "sentence": [1, 2]}, "narrow")
    incomparable = rule(1, {"paragraph": 1, "sentence": [1, 3]}, "later maximum")
    dominated = rule(2, {"paragraph": [1, 2], "sentence": [1, 2]}, "dominated")
    decision = resolve_rules(
        unit_ch08_p1_s1, None, (narrow, incomparable, dominated), siblings
    )
    assert decision.value == "later maximum"
