from __future__ import annotations

from types import SimpleNamespace

import kenkui
from kenkui import voice_metadata as vm


def test_public_api_reexports():
    assert kenkui.voice_label is vm.voice_label
    for name in (
        "voice_id",
        "voice_source",
        "voice_source_group",
        "voice_excluded",
        "voice_label",
    ):
        assert name in kenkui.__all__


def test_voice_id_prefers_voice_id_then_name():
    assert vm.voice_id(SimpleNamespace(voice_id="alba")) == "alba"
    assert vm.voice_id(SimpleNamespace(name="legacy")) == "legacy"
    assert vm.voice_id(SimpleNamespace()) == ""


def test_voice_source_prefers_origin_then_source():
    assert vm.voice_source(SimpleNamespace(origin="compiled")) == "compiled"
    assert vm.voice_source(SimpleNamespace(source="builtin")) == "builtin"


def test_voice_source_group_mapping():
    assert vm.voice_source_group(SimpleNamespace(origin="kenkui_compiled")) == "compiled"
    assert vm.voice_source_group(SimpleNamespace(origin="pocket_tts_builtin")) == "builtin"
    assert vm.voice_source_group(SimpleNamespace(origin="uncompiled")) == "custom"
    assert vm.voice_source_group(SimpleNamespace(origin="other")) == "other"


def test_voice_excluded_prefers_pool_enabled():
    assert vm.voice_excluded(SimpleNamespace(pool_enabled=False)) is True
    assert vm.voice_excluded(SimpleNamespace(pool_enabled=True)) is False
    assert vm.voice_excluded(SimpleNamespace(excluded=True)) is True
    assert vm.voice_excluded(SimpleNamespace()) is False


def test_voice_label_precedence():
    assert vm.voice_label(SimpleNamespace(display_label="A", voice_id="x")) == "A"
    assert vm.voice_label(SimpleNamespace(description="D", voice_id="x")) == "D"
    assert vm.voice_label(SimpleNamespace(display_name="N", voice_id="x")) == "N"
    assert vm.voice_label(SimpleNamespace(voice_id="x")) == "x"


def test_nlp_cache_dir_ends_with_nlp_cache():
    assert kenkui.nlp_cache_dir().name == "nlp_cache"
