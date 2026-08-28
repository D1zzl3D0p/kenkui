"""Volume two sounds like volume one."""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING

import kenkui as kk
from conftest import log_field
from kenkui._audio.m4b import FakeArtifactAssembler
from kenkui._characters import store
from kenkui._domain.operations import Series
from kenkui._execution.coordinator import ExecutionBindings
from kenkui._execution.process_pool import EngineSpecification
from kenkui.pipeline import _log_series_overrides
from test_epub import make_epub, xhtml

if TYPE_CHECKING:
    from pathlib import Path

    import pytest

# Two loaded voices, neither the narrator, so the solver has somewhere to
# spread the cast. Language matches the fixture narrator below.
_ALF = kk.Voice(
    id="alf",
    name="Alf",
    enabled=True,
    provenance="fixture",
    license_id="CC0-1.0",
    commercial_use_allowed=True,
    language="en",
    state="loaded",
)
_AOIFE = kk.Voice(
    id="aoife",
    name="Aoife",
    enabled=True,
    provenance="fixture",
    license_id="CC0-1.0",
    commercial_use_allowed=True,
    language="en",
    state="loaded",
)
_NARRATOR = kk.Voice(
    id="eponine",
    name="Eponine",
    enabled=True,
    provenance="fixture",
    license_id="CC0-1.0",
    commercial_use_allowed=True,
    language="en",
    state="loaded",
)


class _Roster:
    """Names one character, and attributes every quote to them."""

    def __init__(self, character_id: str) -> None:
        self.character_id = character_id

    def complete(self, model: str, prompt: str) -> str:
        assert model
        if "List the speaking characters" in prompt:
            return json.dumps(
                {
                    "characters": [
                        {
                            "id": self.character_id,
                            "name": self.character_id.title(),
                            "gender": "feminine",
                        }
                    ]
                }
            )
        return json.dumps(
            {"attributions": [{"quote_id": 0, "speaker": self.character_id}]}
        )


def _stub_resolution(monkeypatch: pytest.MonkeyPatch, character_id: str) -> None:
    """Replace the model boundary and the voice pool with deterministic doubles.

    `resolve_attribution` takes a `client` argument directly (see
    `tests/test_attribution.py`), but `Pipeline` has no public parameter that
    reaches it -- inventing one just for this test would put a model concern
    in front of every caller of the public API. Instead this monkeypatches
    `kenkui.pipeline._attribution_client`, the private seam added alongside
    it, the same way existing tests already monkeypatch
    `kenkui.pipeline._execution_bindings` to skip real engine resolution.
    """
    monkeypatch.setattr(
        "kenkui.pipeline._attribution_client", lambda: _Roster(character_id)
    )
    monkeypatch.setattr(
        "kenkui.pipeline._execution_bindings",
        lambda: ExecutionBindings(
            EngineSpecification.fake(), FakeArtifactAssembler(), _NARRATOR, "fake-v1"
        ),
    )
    monkeypatch.setattr("kenkui.voices.provision.list_voices", lambda: (_ALF, _AOIFE))


def _render_volume(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, character_id: str, book: int
) -> dict[str, str]:
    """Resolve one volume of a series and return its cast assignments."""
    _stub_resolution(monkeypatch, character_id)
    path = make_epub(
        tmp_path / f"volume-{book}.epub",
        chapters={"one": xhtml(f'<h1>One</h1><p>"Hello," said {character_id}.</p>')},
        spine=("one",),
    )
    resolved = (
        kk.epub(path)
        .series("s", book=book)
        .infer_characters("fake/model")
        .attribute_quotes("fake/model")
        .assign_voices(narrator="eponine")
        .resolve()
    )
    return dict(resolved._resolved.cast_assignments)  # noqa: SLF001


def test_a_returning_character_keeps_their_voice(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The whole point: one person, one voice, across volumes."""
    first = _render_volume(tmp_path, monkeypatch, "javert", book=1)
    second = _render_volume(tmp_path, monkeypatch, "javert", book=2)
    assert first["javert"] == second["javert"]


def test_a_newcomer_does_not_take_a_spent_voice(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Volume two keeps spreading rather than restarting the count."""
    first = _render_volume(tmp_path, monkeypatch, "javert", book=1)
    second = _render_volume(tmp_path, monkeypatch, "cosette", book=2)
    assert second["cosette"] != first["javert"]


def test_a_book_outside_a_series_touches_no_series_state(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Every existing pipeline keeps behaving exactly as it did.

    Single-voice resolution never reaches attribution, so only the engine
    binding -- needed for *any* call to ``resolve()``, series or not -- has
    to be stubbed; the model and voice-pool doubles above are irrelevant
    here.
    """
    monkeypatch.setattr(
        "kenkui.pipeline._execution_bindings",
        lambda: ExecutionBindings(
            EngineSpecification.fake(), FakeArtifactAssembler(), _NARRATOR, "fake-v1"
        ),
    )
    path = make_epub(
        tmp_path / "solo.epub",
        chapters={"one": xhtml('<h1>One</h1><p>"Hello," said javert.</p>')},
        spine=("one",),
    )
    kk.epub(path).assign_voice("eponine").resolve()
    assert store.list_series() == ()


def test_a_re_rendered_volume_does_not_double_its_own_contribution(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Re-rendering the same volume is ordinary, not a second volume's worth.

    ``resolve()``/``write()`` run this wiring on every call, so a caller who
    re-renders volume one twice must not see its speech counted twice in the
    series -- it would silently skew how later volumes spread their cast.
    """
    _render_volume(tmp_path, monkeypatch, "javert", book=1)
    once = store.read_series("s")
    assert once is not None
    first_total = next(
        c for c in once.characters if c.canonical_id == "javert"
    ).spoken_characters

    _render_volume(tmp_path, monkeypatch, "javert", book=1)
    twice = store.read_series("s")
    assert twice is not None
    second_total = next(
        c for c in twice.characters if c.canonical_id == "javert"
    ).spoken_characters

    assert second_total == first_total


def _known(voice_id: str = "alf") -> store.SeriesCharacter:
    return store.SeriesCharacter(
        canonical_id="javert",
        display_name="Javert",
        gender="masculine",
        voice_id=voice_id,
        spoken_characters=500,
        aliases=("Javert",),
    )


def test_a_dropped_pin_is_logged_for_the_operator(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """allow_recast lets a render proceed; the log says it had to."""
    series = Series(series_id="s", allow_recast=True)
    stored = store.SeriesRecord("s", "eponine", (_known(),))
    with caplog.at_level(logging.WARNING):
        _log_series_overrides(series, stored, ["javert"], "eponine")
    assert "series_override" in caplog.text
    record = caplog.records[0]
    assert log_field(record, "boundary") == "series"
    assert log_field(record, "dropped_pins") == "javert"


def test_an_adopted_narrator_is_logged_for_the_operator(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """allow_narrator_change lets a render proceed; the log says it had to."""
    series = Series(series_id="s", allow_narrator_change=True)
    stored = store.SeriesRecord("s", "eponine", (_known(),))
    with caplog.at_level(logging.WARNING):
        _log_series_overrides(series, stored, [], "cosette")
    assert "series_override" in caplog.text


def test_an_honoured_series_logs_nothing(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """No noise when nothing had to be forced."""
    series = Series(series_id="s")
    stored = store.SeriesRecord("s", "eponine", (_known(),))
    with caplog.at_level(logging.WARNING):
        _log_series_overrides(series, stored, [], "eponine")
    assert "series_override" not in caplog.text


