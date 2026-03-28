"""Tests for wizard back-navigation (Task 2)."""

import pytest


def test_goback_moves_to_previous_step():
    """GoBack from step 1 re-runs step 0."""
    call_log = []

    def step0(state):
        call_log.append("step0")
        return {**state, "a": 1}

    def step1(state):
        call_log.append("step1")
        if len([x for x in call_log if x == "step1"]) == 1:
            raise GoBack  # First call raises GoBack
        return {**state, "b": 2}

    from kenkui.cli.add import GoBack
    steps = [step0, step1]
    state = {}
    i = 0
    while i < len(steps):
        try:
            state = steps[i](state)
            i += 1
        except GoBack:
            i = max(0, i - 1)

    assert call_log == ["step0", "step1", "step0", "step1"]
    assert state == {"a": 1, "b": 2}


def test_escape_on_first_step_stays_at_step_0():
    """GoBack at step 0 stays at step 0, not negative index."""
    from kenkui.cli.add import GoBack
    call_count = [0]

    def step0(state):
        call_count[0] += 1
        if call_count[0] == 1:
            raise GoBack
        return state

    steps = [step0]
    state = {}
    i = 0
    while i < len(steps):
        try:
            state = steps[i](state)
            i += 1
        except GoBack:
            i = max(0, i - 1)

    assert call_count[0] == 2  # ran twice: once raised GoBack, once succeeded


def test_wizard_execute_raises_goback_on_keyboard_interrupt():
    """_wizard_execute converts KeyboardInterrupt to GoBack."""
    from unittest.mock import MagicMock
    from kenkui.cli.add import GoBack, _wizard_execute

    mock_prompt = MagicMock()
    mock_prompt.execute.side_effect = KeyboardInterrupt

    with pytest.raises(GoBack):
        _wizard_execute(mock_prompt)


def test_wizard_execute_raises_goback_on_eoferror():
    """_wizard_execute converts EOFError to GoBack."""
    from unittest.mock import MagicMock
    from kenkui.cli.add import GoBack, _wizard_execute

    mock_prompt = MagicMock()
    mock_prompt.execute.side_effect = EOFError

    with pytest.raises(GoBack):
        _wizard_execute(mock_prompt)


def test_wizard_execute_returns_value_on_success():
    """_wizard_execute returns the result of prompt.execute() on success."""
    from unittest.mock import MagicMock
    from kenkui.cli.add import _wizard_execute

    mock_prompt = MagicMock()
    mock_prompt.execute.return_value = "cosette"

    result = _wizard_execute(mock_prompt)
    assert result == "cosette"
