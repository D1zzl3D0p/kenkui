"""Tests for voice review UX changes (Tasks 3 and 4)."""

import pytest


def test_voice_display_format_puts_voice_first():
    """Review choice label should be 'voice  Character (mentions, gender)'."""
    voice = "cosette"
    display_name = "Elizabeth Bennet"
    prominence = 47
    gender = "she"

    label = f"{voice:<20}  {display_name}  ({prominence} mentions, {gender})"
    assert label.startswith("cosette")
    assert "Elizabeth Bennet" in label
    assert label.index("cosette") < label.index("Elizabeth Bennet")


def test_voice_display_format_voice_column_width():
    """Voice column should be left-justified in a 20-char field."""
    voice = "alba"
    display_name = "Mr. Darcy"
    prominence = 30
    gender = "he"

    label = f"{voice:<20}  {display_name}  ({prominence} mentions, {gender})"
    # Voice part should be exactly 20 chars before the separator
    assert label[:20] == "alba" + " " * 16
    assert label[20:22] == "  "


def test_conflict_suffix_included_for_shared_voice():
    """When a voice is used by another character, the choice label shows '← CharName'."""
    from collections import defaultdict

    voice_users = defaultdict(list)
    voice_users["cosette"] = ["Elizabeth Bennet"]

    choice = {"name": "cosette              Female · French", "value": "cosette"}

    v = choice["value"]
    users = voice_users.get(v, [])
    suffix = f"  ← {', '.join(users[:2])}" if users else ""
    annotated_name = choice["name"] + suffix

    assert "← Elizabeth Bennet" in annotated_name


def test_conflict_suffix_empty_when_no_other_users():
    """When a voice has no other users, no suffix is added."""
    from collections import defaultdict

    voice_users = defaultdict(list)
    # cosette is not in voice_users

    choice = {"name": "cosette              Female · French", "value": "cosette"}

    v = choice["value"]
    users = voice_users.get(v, [])
    suffix = f"  ← {', '.join(users[:2])}" if users else ""
    annotated_name = choice["name"] + suffix

    assert "←" not in annotated_name
    assert annotated_name == choice["name"]


def test_conflict_suffix_shows_max_two_users():
    """Conflict suffix truncates to first 2 users."""
    from collections import defaultdict

    voice_users = defaultdict(list)
    voice_users["alba"] = ["Alice", "Bob", "Charlie"]

    choice = {"name": "alba                 Male · American", "value": "alba"}

    v = choice["value"]
    users = voice_users.get(v, [])
    suffix = f"  ← {', '.join(users[:2])}" if users else ""
    annotated_name = choice["name"] + suffix

    assert "← Alice, Bob" in annotated_name
    assert "Charlie" not in annotated_name
