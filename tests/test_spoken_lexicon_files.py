"""A caller's pronunciation table can live in a file, not only in a literal."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import pytest

import kenkui as kk

if TYPE_CHECKING:
    from pathlib import Path


def test_a_plain_object_is_read_as_entries(tmp_path: Path) -> None:
    """The obvious file shape works without ceremony."""
    path = tmp_path / "mine.json"
    path.write_text(json.dumps({"Cthulhu": "kuh-THOO-loo", "Nyarlathotep": "nyar"}))
    assert kk.read_lexicon(path) == {
        "Cthulhu": "kuh-THOO-loo",
        "Nyarlathotep": "nyar",
    }


def test_the_shipped_file_shape_is_also_accepted(tmp_path: Path) -> None:
    """Copying the built-in table and editing it must just work."""
    path = tmp_path / "mine.json"
    path.write_text(json.dumps({"version": "mine-v1", "entries": {"quay": "key"}}))
    assert kk.read_lexicon(path) == {"quay": "key"}


def test_the_built_in_table_is_readable() -> None:
    """A caller extending the shipped entries has to be able to see them."""
    shipped = kk.builtin_lexicon()
    assert shipped["colonel"] == "kernel"
    # A copy, so editing it cannot corrupt the process-wide cached table.
    shipped["colonel"] = "tampered"
    assert kk.builtin_lexicon()["colonel"] == "kernel"


def test_a_missing_file_fails_with_a_source_code(tmp_path: Path) -> None:
    """A path that is not there is a source problem, not a parse one."""
    with pytest.raises(kk.SourceError) as error:
        kk.read_lexicon(tmp_path / "absent.json")
    assert error.value.code == kk.ErrorCode.SOURCE_NOT_FOUND


def test_malformed_json_is_a_pronunciation_error(tmp_path: Path) -> None:
    """A file that is not JSON is bad pronunciation input."""
    path = tmp_path / "broken.json"
    path.write_text("{not json")
    with pytest.raises(kk.ValidationError) as error:
        kk.read_lexicon(path)
    assert error.value.code == kk.ErrorCode.INVALID_PRONUNCIATION


def test_entries_are_validated_on_read(tmp_path: Path) -> None:
    """The same bounds a literal gets, so a file cannot smuggle bad input in."""
    path = tmp_path / "clashing.json"
    path.write_text(json.dumps({"Quay": "key", "quay": "kway"}))
    with pytest.raises(kk.ValidationError) as error:
        kk.read_lexicon(path)
    assert error.value.code == kk.ErrorCode.INVALID_PRONUNCIATION


def test_a_read_lexicon_feeds_pronounce(tmp_path: Path) -> None:
    """The whole point: what comes out of the file goes into the stage."""
    path = tmp_path / "mine.json"
    path.write_text(json.dumps({"Cthulhu": "kuh-THOO-loo"}))
    pipeline = kk.epub("book.epub").pronounce(kk.read_lexicon(path))
    assert pipeline.operations[-1].lexicon == (("Cthulhu", "kuh-THOO-loo"),)
