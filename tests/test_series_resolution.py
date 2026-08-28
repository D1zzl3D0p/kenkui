"""Volume two sounds like volume one."""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING

import pytest

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
# Sorts before "alf" -- used to prove a re-solve (rather than a persisted
# pin) actually ran, since the greedy solver's tie-break is alphabetical.
_AARON = kk.Voice(
    id="aaron",
    name="Aaron",
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
        _log_series_overrides(series, stored, ["javert"], [], "eponine")
    assert "series_override" in caplog.text
    record = caplog.records[0]
    assert log_field(record, "boundary") == "series"
    assert log_field(record, "dropped_pins") == "javert"


def test_an_overridden_pin_is_logged_for_the_operator(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A caller's explicit cast= beating a series pin is still worth saying."""
    series = Series(series_id="s")
    stored = store.SeriesRecord("s", "eponine", (_known(),))
    with caplog.at_level(logging.WARNING):
        _log_series_overrides(series, stored, [], ["javert"], "eponine")
    assert "series_override" in caplog.text
    record = caplog.records[0]
    assert log_field(record, "overridden_pins") == "javert"


def test_an_adopted_narrator_is_logged_for_the_operator(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """allow_narrator_change lets a render proceed; the log says it had to."""
    series = Series(series_id="s", allow_narrator_change=True)
    stored = store.SeriesRecord("s", "eponine", (_known(),))
    with caplog.at_level(logging.WARNING):
        _log_series_overrides(series, stored, [], [], "cosette")
    assert "series_override" in caplog.text


def test_an_honoured_series_logs_nothing(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """No noise when nothing had to be forced."""
    series = Series(series_id="s")
    stored = store.SeriesRecord("s", "eponine", (_known(),))
    with caplog.at_level(logging.WARNING):
        _log_series_overrides(series, stored, [], [], "eponine")
    assert "series_override" not in caplog.text




def test_a_pin_cannot_land_a_character_on_the_narrators_voice(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """The narrator's voice is reserved.

    ``resolve_cast`` never reports a collision for an explicit pin -- that
    check only runs for characters the solver placed itself -- so a pin
    equal to the narrator's voice would otherwise land a character on it
    with nothing to show for it.
    """
    store.write_series(
        store.SeriesRecord(
            "s",
            "eponine",
            (
                store.SeriesCharacter(
                    canonical_id="javert",
                    display_name="Javert",
                    gender="feminine",
                    voice_id="eponine",
                    spoken_characters=100,
                    aliases=("Javert",),
                ),
            ),
        )
    )
    _stub_resolution(monkeypatch, "javert")
    # The narrator's own voice is also enumerated by list_voices(), which is
    # exactly what makes it reachable as a pin: nothing here treats it as
    # special until the reserved-id exclusion does.
    monkeypatch.setattr(
        "kenkui.voices.provision.list_voices", lambda: (_ALF, _AOIFE, _NARRATOR)
    )
    path = make_epub(
        tmp_path / "volume-2.epub",
        chapters={"one": xhtml('<h1>One</h1><p>"Hello," said javert.</p>')},
        spine=("one",),
    )
    with caplog.at_level(logging.WARNING):
        resolved = (
            kk.epub(path)
            .series("s", book=2, allow_recast=True)
            .infer_characters("fake/model")
            .attribute_quotes("fake/model")
            .assign_voices(narrator="eponine")
            .resolve()
        )
    assignments = dict(resolved._resolved.cast_assignments)  # noqa: SLF001
    assert assignments["javert"] != "eponine"
    assert "series_override" in caplog.text


def test_a_pin_for_a_different_language_voice_is_dropped_and_logged(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """validate()'s pool and _resolve_all's pool differ by language filtering.

    A pin naming a voice that is loaded but of another language passes
    validate(), which never filters by language, and reaches here with no
    allow_recast involved at all. It must still be dropped, and reported.
    """
    store.write_series(
        store.SeriesRecord(
            "s",
            "eponine",
            (
                store.SeriesCharacter(
                    canonical_id="javert",
                    display_name="Javert",
                    gender="feminine",
                    voice_id="foreign",
                    spoken_characters=100,
                    aliases=("Javert",),
                ),
            ),
        )
    )
    _stub_resolution(monkeypatch, "javert")
    foreign = kk.Voice(
        id="foreign",
        name="Foreign",
        enabled=True,
        provenance="fixture",
        license_id="CC0-1.0",
        commercial_use_allowed=True,
        language="fr",
        state="loaded",
    )
    monkeypatch.setattr(
        "kenkui.voices.provision.list_voices", lambda: (_ALF, _AOIFE, foreign)
    )
    path = make_epub(
        tmp_path / "volume-2.epub",
        chapters={"one": xhtml('<h1>One</h1><p>"Hello," said javert.</p>')},
        spine=("one",),
    )
    with caplog.at_level(logging.WARNING):
        resolved = (
            kk.epub(path)
            .series("s", book=2)
            .infer_characters("fake/model")
            .attribute_quotes("fake/model")
            .assign_voices(narrator="eponine")
            .resolve()
        )
    assignments = dict(resolved._resolved.cast_assignments)  # noqa: SLF001
    assert assignments["javert"] != "foreign"
    assert "series_override" in caplog.text


def test_allow_recast_converges_once_the_dropped_pin_is_replaced(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A dropped pin must persist what the solver actually chose.

    Otherwise every render re-pins the dead voice, drops it again, and
    re-solves from scratch -- which a *widening* pool can answer
    differently each time, since the greedy solver's tie-break is
    alphabetical over whatever pool it is handed. Persisting the
    replacement is what lets allow_recast settle on one voice, which is the
    property that matters -- not any single render's answer.
    """
    store.write_series(
        store.SeriesRecord(
            "s",
            "eponine",
            (
                store.SeriesCharacter(
                    canonical_id="javert",
                    display_name="Javert",
                    gender="feminine",
                    voice_id="ghost",
                    spoken_characters=100,
                    aliases=("Javert",),
                ),
            ),
        )
    )
    path = make_epub(
        tmp_path / "volume-2.epub",
        chapters={"one": xhtml('<h1>One</h1><p>"Hello," said javert.</p>')},
        spine=("one",),
    )

    # The client and engine bindings are stubbed once, directly -- not via
    # `_stub_resolution`, which would also reset the voice pool back to its
    # default on every call and mask the pool actually being widened below.
    monkeypatch.setattr(
        "kenkui.pipeline._attribution_client", lambda: _Roster("javert")
    )
    monkeypatch.setattr(
        "kenkui.pipeline._execution_bindings",
        lambda: ExecutionBindings(
            EngineSpecification.fake(), FakeArtifactAssembler(), _NARRATOR, "fake-v1"
        ),
    )

    def _render() -> str:
        resolved = (
            kk.epub(path)
            .series("s", book=2, allow_recast=True)
            .infer_characters("fake/model")
            .attribute_quotes("fake/model")
            .assign_voices(narrator="eponine")
            .resolve()
        )
        return dict(resolved._resolved.cast_assignments)["javert"]  # noqa: SLF001

    monkeypatch.setattr("kenkui.voices.provision.list_voices", lambda: (_ALF, _AOIFE))
    first = _render()
    assert first == "alf"

    # The pool widens with a voice that sorts before "alf". If this render
    # still had to re-solve from scratch -- because the first render's
    # replacement was never persisted -- it would pick "aaron" instead.
    monkeypatch.setattr(
        "kenkui.voices.provision.list_voices", lambda: (_AARON, _ALF, _AOIFE)
    )
    second = _render()
    assert second == first


def test_an_explicit_cast_overrides_a_series_pin(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """The caller's render-time cast is more specific than a stored series default.

    It is also the only way to correct a series pin without discarding the
    whole series, so the series then adopts what the caller chose.
    """
    first = _render_volume(tmp_path, monkeypatch, "javert", book=1)
    assert first["javert"] == "alf"  # deterministic: alf sorts before aoife

    _stub_resolution(monkeypatch, "javert")
    path = make_epub(
        tmp_path / "volume-2.epub",
        chapters={"one": xhtml('<h1>One</h1><p>"Hello," said javert.</p>')},
        spine=("one",),
    )
    with caplog.at_level(logging.WARNING):
        resolved = (
            kk.epub(path)
            .series("s", book=2)
            .infer_characters("fake/model")
            .attribute_quotes("fake/model")
            .assign_voices(narrator="eponine", cast={"javert": "aoife"})
            .resolve()
        )
    assignments = dict(resolved._resolved.cast_assignments)  # noqa: SLF001
    assert assignments["javert"] == "aoife"
    assert "series_override" in caplog.text

    record = store.read_series("s")
    assert record is not None
    javert = next(c for c in record.characters if c.canonical_id == "javert")
    assert javert.voice_id == "aoife"


def test_allow_narrator_change_adopts_the_new_narrator(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """The spec: allow_narrator_change records the new narrator from here on."""
    _render_volume(tmp_path, monkeypatch, "javert", book=1)
    record = store.read_series("s")
    assert record is not None
    assert record.narrator_voice_id == "eponine"

    _stub_resolution(monkeypatch, "javert")
    path = make_epub(
        tmp_path / "volume-2.epub",
        chapters={"one": xhtml('<h1>One</h1><p>"Hello," said javert.</p>')},
        spine=("one",),
    )
    with caplog.at_level(logging.WARNING):
        (
            kk.epub(path)
            .series("s", book=2, allow_narrator_change=True)
            .infer_characters("fake/model")
            .attribute_quotes("fake/model")
            .assign_voices(narrator="cosette")
            .resolve()
        )
    assert "series_override" in caplog.text
    adopted = store.read_series("s")
    assert adopted is not None
    assert adopted.narrator_voice_id == "cosette"

    # Having settled, the same narrator on the next render is no longer a
    # change worth reporting -- the state converged instead of nagging
    # forever about a switch that already happened.
    caplog.clear()
    _stub_resolution(monkeypatch, "javert")
    path3 = make_epub(
        tmp_path / "volume-3.epub",
        chapters={"one": xhtml('<h1>One</h1><p>"Hello," said javert.</p>')},
        spine=("one",),
    )
    with caplog.at_level(logging.WARNING):
        (
            kk.epub(path3)
            .series("s", book=3, allow_narrator_change=True)
            .infer_characters("fake/model")
            .attribute_quotes("fake/model")
            .assign_voices(narrator="cosette")
            .resolve()
        )
    assert "series_override" not in caplog.text


def test_an_unauthorised_narrator_change_does_not_corrupt_the_series(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """resolve() never validates, so an unauthorised change must not stick.

    Otherwise ``write()`` refuses this pipeline (SERIES_NARRATOR_CHANGED)
    while ``resolve().write()`` would already have repointed the series to
    whatever ``narrator=`` said -- a typo permanently redirecting a series
    with no permission ever granted.
    """
    _render_volume(tmp_path, monkeypatch, "javert", book=1)
    before = store.read_series("s")
    assert before is not None
    assert before.narrator_voice_id == "eponine"

    _stub_resolution(monkeypatch, "javert")
    path = make_epub(
        tmp_path / "volume-2.epub",
        chapters={"one": xhtml('<h1>One</h1><p>"Hello," said javert.</p>')},
        spine=("one",),
    )
    pipeline = (
        kk.epub(path)
        .series("s", book=2)  # no allow_narrator_change
        .infer_characters("fake/model")
        .attribute_quotes("fake/model")
        .assign_voices(narrator="cosette")
    )
    pipeline.resolve()

    unchanged = store.read_series("s")
    assert unchanged is not None
    assert unchanged.narrator_voice_id == "eponine"

    # The discrepancy this render just introduced must still be visible to
    # the next validate(), the same as if resolve() had never run.
    issues = {issue.code for issue in pipeline.validate().issues}
    assert kk.ErrorCode.SERIES_NARRATOR_CHANGED in issues


def test_a_render_that_fails_at_binding_leaves_the_series_unchanged(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The series must not record a cast the render never actually produced.

    The narrator alone resolves fine -- that binding runs early, purely to
    read the narrator's language for the pool filter -- but resolving the
    *whole* cast's bindings is what can fail on an unprovisioned voice, and
    it must be checked before, not after, the series is written.
    """
    seed = store.SeriesRecord(
        "s",
        "eponine",
        (
            store.SeriesCharacter(
                canonical_id="javert",
                display_name="Javert",
                gender="feminine",
                voice_id="alf",
                spoken_characters=100,
                aliases=("Javert",),
            ),
        ),
    )
    store.write_series(seed)
    monkeypatch.setattr(
        "kenkui.pipeline._attribution_client", lambda: _Roster("javert")
    )
    monkeypatch.setattr("kenkui.voices.provision.list_voices", lambda: (_ALF, _AOIFE))

    def _bindings(_voice_id: str, *, also: tuple[str, ...] = ()) -> ExecutionBindings:
        if also:
            raise kk.VoiceError(kk.ErrorCode.VOICE_DISABLED)
        return ExecutionBindings(
            EngineSpecification.fake(), FakeArtifactAssembler(), _NARRATOR, "fake-v1"
        )

    monkeypatch.setattr("kenkui.pipeline._execution_bindings", _bindings)

    path = make_epub(
        tmp_path / "volume-2.epub",
        chapters={"one": xhtml('<h1>One</h1><p>"Hello," said javert.</p>')},
        spine=("one",),
    )
    with pytest.raises(kk.VoiceError):
        (
            kk.epub(path)
            .series("s", book=2)
            .infer_characters("fake/model")
            .attribute_quotes("fake/model")
            .assign_voices(narrator="eponine")
            .resolve()
        )

    assert store.read_series("s") == seed


def test_an_unauthorised_drop_renders_but_leaves_the_series_unchanged(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Without allow_recast, a dropped pin must not launder itself away.

    The solver still has to choose something for the render -- the speech
    cannot vanish -- but persisting that choice would erase the operator's
    only signal that the pinned voice went missing, and the next
    validate() would fall silent about a problem that still exists.
    """
    seed = store.SeriesRecord(
        "s",
        "eponine",
        (
            store.SeriesCharacter(
                canonical_id="javert",
                display_name="Javert",
                gender="feminine",
                voice_id="ghost",
                spoken_characters=100,
                aliases=("Javert",),
            ),
        ),
    )
    store.write_series(seed)
    _stub_resolution(monkeypatch, "javert")
    path = make_epub(
        tmp_path / "volume-2.epub",
        chapters={"one": xhtml('<h1>One</h1><p>"Hello," said javert.</p>')},
        spine=("one",),
    )
    pipeline = (
        kk.epub(path)
        .series("s", book=2)  # no allow_recast
        .infer_characters("fake/model")
        .attribute_quotes("fake/model")
        .assign_voices(narrator="eponine")
    )
    resolved = pipeline.resolve()
    assignments = dict(resolved._resolved.cast_assignments)  # noqa: SLF001
    assert assignments["javert"] != "ghost"  # the render still had to choose

    # Speech and aliases still accumulate normally -- only the voice itself
    # must not launder away the missing pin. Contrast with
    # `test_allow_recast_converges_once_the_dropped_pin_is_replaced`, where
    # the same drop, authorised, *does* replace the voice.
    persisted = store.read_series("s")
    assert persisted is not None
    javert = next(c for c in persisted.characters if c.canonical_id == "javert")
    assert javert.voice_id == "ghost"

    issues = {issue.code for issue in pipeline.validate().issues}
    assert kk.ErrorCode.SERIES_VOICE_MISSING in issues


class _OffRoster:
    """Names one character, then answers every quote with someone else.

    Exactly what `attribution.py` mints a ``role:<slug>@<chapter_id>`` for:
    a speaker the roster never listed, kept distinct per chapter because
    chapter 40's officer is not chapter 12's.
    """

    def complete(self, model: str, prompt: str) -> str:
        assert model
        if "List the speaking characters" in prompt:
            return json.dumps(
                {
                    "characters": [
                        {"id": "javert", "name": "Javert", "gender": "feminine"}
                    ]
                }
            )
        return json.dumps(
            {"attributions": [{"quote_id": 0, "speaker": "officer"}]}
        )


def test_a_minted_role_never_becomes_a_series_character(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A one-scene part is not a person the series should remember.

    Minted roles are deliberately distinct per chapter, so persisting them
    inverts that decision across volumes -- volume one's officer would be
    merged with volume four's -- and, because every volume mints the same
    slugs, it was also what made an ordinary first volume grow a new
    canonical on every re-render.
    """
    monkeypatch.setattr("kenkui.pipeline._attribution_client", _OffRoster)
    monkeypatch.setattr(
        "kenkui.pipeline._execution_bindings",
        lambda: ExecutionBindings(
            EngineSpecification.fake(), FakeArtifactAssembler(), _NARRATOR, "fake-v1"
        ),
    )
    monkeypatch.setattr("kenkui.voices.provision.list_voices", lambda: (_ALF, _AOIFE))
    path = make_epub(
        tmp_path / "volume-1.epub",
        chapters={
            "one": xhtml('<h1>One</h1><p>"Hello," said the officer.</p>'),
            "two": xhtml('<h1>Two</h1><p>"Again," said the officer.</p>'),
        },
        spine=("one", "two"),
    )

    def _render() -> dict[str, str]:
        resolved = (
            kk.epub(path)
            .series("s", book=1)
            .infer_characters("fake/model")
            .attribute_quotes("fake/model")
            .assign_voices(narrator="eponine")
            .resolve()
        )
        return dict(resolved._resolved.cast_assignments)  # noqa: SLF001

    first = _render()
    # Still cast normally inside their own book: excluding them from the
    # series must not drop their speech.
    minted = sorted(key for key in first if key.startswith("role:officer@"))
    assert len(minted) == 2  # noqa: PLR2004 - one officer per chapter, by design
    for _ in range(2):
        assert _render() == first
        record = store.read_series("s")
        assert record is not None
        assert record.characters == ()


class _TwoSpeakers:
    """Two women whose surface forms both reach one person the series knows."""

    def complete(self, model: str, prompt: str) -> str:
        assert model
        if "List the speaking characters" in prompt:
            return json.dumps(
                {
                    "characters": [
                        {"id": "elizabeth", "name": "Elizabeth", "gender": "feminine"},
                        {"id": "lizzy", "name": "Lizzy", "gender": "feminine"},
                    ]
                }
            )
        return json.dumps(
            {
                "attributions": [
                    {"quote_id": 0, "speaker": "elizabeth"},
                    {"quote_id": 1, "speaker": "lizzy"},
                ]
            }
        )


def test_re_rendering_a_volume_neither_mints_nor_recasts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Rendering one volume twice is one volume, not two.

    Both of this book's characters reach the series' one Elizabeth Bennet,
    so `match_roster` withholds her from both -- correctly. Each then
    joined the series under a minted canonical, and the next render, finding
    its own slug taken, minted another; the series grew by two people per
    render. Their own accumulated speech also fed back into `prior_load`,
    so the solver re-cast them against an inflated count and the voices
    swapped between renders -- which changes the segment identity and
    re-synthesizes the whole book.
    """
    store.write_series(
        store.SeriesRecord(
            "s",
            "eponine",
            (
                store.SeriesCharacter(
                    canonical_id="elizabeth-bennet",
                    display_name="Elizabeth Bennet",
                    gender="feminine",
                    voice_id="alf",
                    spoken_characters=100,
                    aliases=("Elizabeth", "Lizzy"),
                ),
            ),
        )
    )
    monkeypatch.setattr("kenkui.pipeline._attribution_client", _TwoSpeakers)
    monkeypatch.setattr(
        "kenkui.pipeline._execution_bindings",
        lambda: ExecutionBindings(
            EngineSpecification.fake(), FakeArtifactAssembler(), _NARRATOR, "fake-v1"
        ),
    )
    monkeypatch.setattr("kenkui.voices.provision.list_voices", lambda: (_ALF, _AOIFE))
    path = make_epub(
        tmp_path / "volume-1.epub",
        chapters={
            # Deliberately lopsided: Elizabeth's own accumulated speech is
            # what tips `prior_load` far enough to move her off the voice
            # she was given last render.
            "one": xhtml(
                "<h1>One</h1><p>"
                '"Hello, and a great deal more besides, for I have been '
                "talking at some length about the weather and the roads and "
                'everything else that comes to mind," said Elizabeth.</p>'
                '<p>"Indeed," said Lizzy.</p>'
            )
        },
        spine=("one",),
    )

    def _render() -> dict[str, str]:
        resolved = (
            kk.epub(path)
            .series("s", book=1)
            .infer_characters("fake/model")
            .attribute_quotes("fake/model")
            .assign_voices(narrator="eponine")
            .resolve()
        )
        return dict(resolved._resolved.cast_assignments)  # noqa: SLF001

    first = _render()
    for _ in range(3):
        assert _render() == first
        record = store.read_series("s")
        assert record is not None
        assert {c.canonical_id for c in record.characters} == {
            "elizabeth-bennet",
            "elizabeth",
            "lizzy",
        }


_FOREIGN = kk.Voice(
    id="foreign",
    name="Foreign",
    enabled=True,
    provenance="fixture",
    license_id="CC0-1.0",
    commercial_use_allowed=True,
    language="fr",
    state="loaded",
)


def _series_pipeline(tmp_path: Path, name: str = "volume-2.epub") -> kk.Pipeline:
    """Build a second volume of series "s", ready to validate."""
    path = make_epub(
        tmp_path / name,
        chapters={"one": xhtml('<h1>One</h1><p>"Hello," said javert.</p>')},
        spine=("one",),
    )
    return (
        kk.epub(path)
        .series("s", book=2)
        .infer_characters("fake/model")
        .attribute_quotes("fake/model")
        .assign_voices(narrator="eponine")
    )


def _seed(voice_id: str) -> None:
    store.write_series(
        store.SeriesRecord(
            "s",
            "eponine",
            (
                store.SeriesCharacter(
                    canonical_id="javert",
                    display_name="Javert",
                    gender="feminine",
                    voice_id=voice_id,
                    spoken_characters=100,
                    aliases=("Javert",),
                ),
            ),
        )
    )


def test_validate_refuses_a_pin_of_the_wrong_language(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """validate() has to judge the pool the render will actually cast from.

    `_resolve_all` keeps only voices matching the narrator's language, so a
    pin naming a loaded French voice passes a language-blind validate(), is
    dropped at render time, and -- correctly, without allow_recast -- is not
    persisted. The character then sounds different in this volume and every
    later one, with only a log line to say so.
    """
    _seed("foreign")
    monkeypatch.setattr(
        "kenkui.voices.provision.list_voices",
        lambda: (_ALF, _AOIFE, _NARRATOR, _FOREIGN),
    )
    issues = {issue.code for issue in _series_pipeline(tmp_path).validate().issues}
    assert kk.ErrorCode.SERIES_VOICE_MISSING in issues


def test_validate_refuses_a_pin_equal_to_this_renders_narrator(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The narrator's voice is reserved, so a pin on it is no pin at all."""
    _seed("eponine")
    monkeypatch.setattr(
        "kenkui.voices.provision.list_voices", lambda: (_ALF, _AOIFE, _NARRATOR)
    )
    issues = {issue.code for issue in _series_pipeline(tmp_path).validate().issues}
    assert kk.ErrorCode.SERIES_VOICE_MISSING in issues


def test_validate_still_accepts_a_pin_the_render_can_honour(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Tightening the pool must not start refusing ordinary series renders."""
    _seed("alf")
    monkeypatch.setattr(
        "kenkui.voices.provision.list_voices",
        lambda: (_ALF, _AOIFE, _NARRATOR, _FOREIGN),
    )
    issues = {issue.code for issue in _series_pipeline(tmp_path).validate().issues}
    assert kk.ErrorCode.SERIES_VOICE_MISSING not in issues


def test_a_missing_series_voice_names_its_characters_for_the_operator(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """The refusal stands; the sanitized issue cannot say who it is about.

    `ValidationIssue` carries a stable code and public text and nothing
    else, so an operator told only "series_voice_missing" has fifty
    characters to search by hand. Spec D asks for the characters and the
    lost voice by name, which is an operator's log line, the same way a
    collision is.
    """
    _seed("ghost")
    monkeypatch.setattr(
        "kenkui.voices.provision.list_voices", lambda: (_ALF, _AOIFE, _NARRATOR)
    )
    with caplog.at_level(logging.WARNING):
        _series_pipeline(tmp_path).validate()
    named = [r for r in caplog.records if r.getMessage() == "series_voice_missing"]
    assert len(named) == 1
    assert log_field(named[0], "boundary") == "series"
    assert log_field(named[0], "series_id") == "s"
    assert log_field(named[0], "characters") == "Javert"
    assert log_field(named[0], "voice_ids") == "ghost"
