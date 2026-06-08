from __future__ import annotations

from unittest.mock import MagicMock, patch

from kenkui.voice_pool import auto_populate_from_voices


def _info(voice_id: str, gender: str):
    v = MagicMock()
    v.voice_id = voice_id
    v.gender = gender
    return v


def test_auto_populate_uses_pool_enabled_catalog_voices():
    voices = [
        _info("m1", "Male"),
        _info("m2", "Male"),
        _info("m3", "Male"),
        _info("f1", "Female"),
    ]

    with patch("kenkui.services.voice_service.list_voices", return_value=voices) as mocked:
        template = auto_populate_from_voices()

    mocked.assert_called_once_with(pool_enabled=True, status="available")
    assert template.protagonist["male"].named[1] == "m1"
    assert template.protagonist["male"].pool == ["m3"]
    assert template.protagonist["female"].named[1] == "f1"
