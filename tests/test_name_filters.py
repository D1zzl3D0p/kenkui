"""Tests for kenkui.nlp._filters._is_proper_name."""
from __future__ import annotations

import pytest

from kenkui.nlp._filters import _is_proper_name


# ---------------------------------------------------------------------------
# Cases that should be rejected (return False)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("phrase", [
    # Long descriptive phrases with article starters
    "a pleasant - faced Andoran whose eyes almost glowed when she spoke of making certain that he lived to face the Dark One",
    "a short , hard - bitten fellow named Gerard Arganda , who was shaking his head so hard",
    "A tall woman with red hair and blue eyes and a very long description",
    # Quantifier/article starters
    "Any Aes Sedai who had not sworn to him",
    "the Dragon Reborn",
    # Relative clause indicators
    "Sisters who knew Nynaeve had learned to take care with that word around her",
    "woman standing near the door",
    "man named Gerard",
    # Possessive starters
    "His guard",
    "Their leader",
    # Demonstrative starters
    "This woman",
    # Single common nouns
    "Sisters",
    "Man",
    "Lord",
])
def test_rejected_phrases(phrase):
    assert _is_proper_name(phrase) is False


# ---------------------------------------------------------------------------
# Cases that should be accepted (return True)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("name", [
    "Rand al'Thor",
    "Nynaeve al'Meara",
    "Mat Cauthon",
    "Perrin Aybara",
    "Egwene al'Vere",
    "Min Farshaw",
    "alba",
    "Elizabeth Bennet",
    "Mr Darcy",
    "Jean Valjean",
    "Faile ni Bashere t'Aybara",  # exactly 4 words — at limit
    "Aes Sedai",
])
def test_accepted_names(name):
    assert _is_proper_name(name) is True
