"""Provisioning failures use stable public codes with sanitized messages."""
# ruff: noqa: D103

from __future__ import annotations

import pytest

from kenkui import ErrorCode, VoiceError

_EXPECTED = {
    ErrorCode.VOICE_NOT_PROVISIONED: "voice_not_provisioned",
    ErrorCode.VOICE_UNKNOWN: "voice_unknown",
    ErrorCode.ENGINE_NOT_CLONING_CAPABLE: "engine_not_cloning_capable",
    ErrorCode.VOICE_VARIETY_INVALID: "voice_variety_invalid",
}


@pytest.mark.parametrize(("code", "value"), list(_EXPECTED.items()))
def test_code_value_is_stable(code: ErrorCode, value: str) -> None:
    assert code.value == value


@pytest.mark.parametrize("code", list(_EXPECTED))
def test_default_message_is_present_and_sanitized(code: ErrorCode) -> None:
    message = str(VoiceError(code))
    assert message
    assert "/" not in message


def test_not_provisioned_message_names_the_fix() -> None:
    assert "load_voice" in str(VoiceError(ErrorCode.VOICE_NOT_PROVISIONED))
