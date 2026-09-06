"""Regression coverage for the barebones multi-book example."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import TYPE_CHECKING, Self

if TYPE_CHECKING:
    import pytest


EXAMPLE = Path(__file__).parents[1] / "spikes/examples/example.py"


def load_example() -> ModuleType:
    """Load the script without executing its main block."""
    spec = importlib.util.spec_from_file_location("example", EXAMPLE)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class Pipeline:
    """Record calls made by the explicit rendering pipeline."""

    def __init__(self) -> None:
        """Initialize an empty call log."""
        self.calls: list[tuple[str, tuple[object, ...], dict[str, object]]] = []

    def _record(
        self, name: str, *args: object, **kwargs: object
    ) -> Self:
        self.calls.append((name, args, kwargs))
        return self

    def pronounce(self, lexicon: dict[str, str]) -> Self:
        """Record pronunciation configuration."""
        return self._record("pronounce", lexicon)

    def metadata(self, **metadata: object) -> Self:
        """Record metadata configuration."""
        return self._record("metadata", **metadata)

    def infer_characters(self, model: str) -> Self:
        """Record character inference configuration."""
        return self._record("infer_characters", model)

    def attribute_quotes(self, model: str) -> Self:
        """Record quote attribution configuration."""
        return self._record("attribute_quotes", model)

    def assign_voices(self, **voices: str) -> Self:
        """Record voice assignment configuration."""
        return self._record("assign_voices", **voices)

    def series(self, series: str, *, book: int) -> Self:
        """Record series configuration."""
        return self._record("series", series, book=book)

    def tts(self) -> Self:
        """Record TTS configuration."""
        return self._record("tts")

    def write(self, output: Path, **options: object) -> Self:
        """Record the output location and execution options."""
        return self._record("write", output, **options)


def test_explicit_run_keeps_cover_and_uses_spacy(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Verify the explicit pipeline keeps cover metadata and uses spaCy."""
    example = load_example()
    pipeline = Pipeline()
    epub = tmp_path / "Book.epub"
    epub.touch()
    cover = tmp_path / "cover.jpg"
    cover.touch()

    def book(path: Path) -> Pipeline:
        assert path == epub
        return pipeline

    monkeypatch.setattr(
        example,
        "kk",
        SimpleNamespace(book=book, builtin_lexicon=dict),
    )

    example.explicit_run("Book", "Author", epub, "series", 1)

    assert ("infer_characters", ("spacy",), {}) in pipeline.calls
    assert (
        "attribute_quotes",
        ("openrouter/deepseek/deepseek-v4-flash",),
        {},
    ) in pipeline.calls
    assert (
        "metadata",
        (),
        {"title": "Book", "author": "Author", "cover": cover},
    ) in pipeline.calls
    assert ("series", ("series",), {"book": 1}) in pipeline.calls
    assert (
        "write",
        (tmp_path / "Book.m4b",),
        {
            "on_event": example.report_progress,
            # Pinned to the performance-core count rather than left on "auto";
            # overwrite because every book in the list already has an M4B.
            "workers": example.WORKERS,
            "overwrite": True,
        },
    ) in pipeline.calls


def test_magic_run_uses_the_attribution_model(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Verify the compact helper receives the attribution model."""
    example = load_example()
    calls: list[tuple[object, dict[str, object]]] = []
    epub = tmp_path / "Book.epub"

    def fake_magic_run(path: Path, **kwargs: object) -> str:
        calls.append((path, kwargs))
        return "result"

    monkeypatch.setattr(
        example, "kk", SimpleNamespace(magic_run=fake_magic_run)
    )

    assert example.magic_run(epub) == "result"
    assert calls == [
        (
            epub,
            {
                "narrator": "ivy",
                "multi": True,
                "model": "openrouter/deepseek/deepseek-v4-flash",
            },
        )
    ]


def test_main_renders_every_listed_book(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify the simple loop renders every configured book."""
    example = load_example()
    rendered: list[tuple[object, ...]] = []

    def explicit_run(*book: object) -> None:
        rendered.append(book)

    monkeypatch.setattr(example, "explicit_run", explicit_run)
    def configure_logging(**_kwargs: object) -> None:
        return None

    monkeypatch.setattr(example.logging, "basicConfig", configure_logging)

    example.main()

    assert rendered == [
        (
            title,
            author,
            example.LIBRARY / author / folder / f"{title} - {author}.epub",
            series,
            volume,
        )
        for title, author, folder, series, volume in example.BOOKS
    ]


def test_main_continues_after_a_kenkui_write_error(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """A failed book must not prevent later books from rendering."""
    example = load_example()
    monkeypatch.setattr(example, "BOOKS", example.BOOKS[:2])
    rendered: list[str] = []

    def explicit_run(title: str, *_args: object) -> None:
        rendered.append(title)
        if title == example.BOOKS[0][0]:
            raise example.kk.EncodingError(example.kk.ErrorCode.OUTPUT_EXISTS)

    monkeypatch.setattr(example, "explicit_run", explicit_run)
    example.main()

    assert rendered == [example.BOOKS[0][0], example.BOOKS[1][0]]
    assert "failed" in caplog.text.lower()
