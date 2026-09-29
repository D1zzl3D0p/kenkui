"""Chapter labels come from navigation, not incidental epigraph headings."""

from __future__ import annotations

from typing import TYPE_CHECKING
from zipfile import ZipFile

import pytest

from helpers import make_epub, xhtml
from kenkui._epub.parser import inspect_epub
from kenkui.errors import ErrorCode, SourceError

if TYPE_CHECKING:
    from pathlib import Path


def _navigation(path: Path, *, nav: str = "", ncx: str = "") -> None:
    with ZipFile(path) as archive:
        contents = {name: archive.read(name) for name in archive.namelist()}
    package = contents["OPS/package.opf"].decode()
    if nav:
        package = package.replace(
            "</manifest>",
            '<item id="nav" href="nav.xhtml" properties="nav" '
            'media-type="application/xhtml+xml"/></manifest>',
        )
        contents["OPS/nav.xhtml"] = xhtml(nav).encode()
    if ncx:
        package = package.replace(
            "</manifest>",
            '<item id="ncx" href="toc.ncx" '
            'media-type="application/x-dtbncx+xml"/></manifest>',
        ).replace("<spine>", '<spine toc="ncx">')
        contents["OPS/toc.ncx"] = f"<ncx><navMap>{ncx}</navMap></ncx>".encode()
    contents["OPS/package.opf"] = package.encode()
    with ZipFile(path, "w") as archive:
        for name, data in contents.items():
            archive.writestr(name, data)


def _point(href: str, title: str) -> str:
    return (
        f"<navPoint><navLabel><text>{title}</text></navLabel>"
        f'<content src="{href}"/></navPoint>'
    )


def test_ncx_titles_and_calibre_continuations(tmp_path: Path) -> None:
    """Reproduce opaque titles and epigraph headings in split Dune chapters."""
    source = make_epub(
        tmp_path / "book.epub",
        chapters={
            "c8P_split_000": xhtml("<p>Opening.</p>", title="c8P"),
            "c8P_split_001": xhtml("<p>Ending.</p>", title="c8P"),
            "c9G": xhtml("<p>Epigraph.</p><h2>BY THE HISTORIAN</h2><p>Story.</p>"),
            "extra": xhtml("<h1>Afterword</h1><p>Notes.</p>"),
        },
        spine=["c8P_split_000", "c8P_split_001", "c9G", "extra"],
    )
    before = inspect_epub(source)
    _navigation(
        source,
        ncx=_point("text/c8P_split_000.xhtml", "Chapter 1")
        + _point("text/c9G.xhtml", "Chapter 2"),
    )
    after = inspect_epub(source)
    assert [c.title for c in after.chapters] == ["Chapter 1", "Chapter 2"]
    assert after.chapters[0].text == "Opening.\n\nEnding."
    assert after.chapters[1].text.endswith("Afterword\n\nNotes.")
    assert "\n\n".join(c.text for c in after.chapters) == "\n\n".join(
        c.text for c in before.chapters
    )
    assert after.chapters[1].id != before.chapters[1].id


def test_epub3_toc_precedes_ncx_and_ignores_other_navigation(tmp_path: Path) -> None:
    """Only the TOC supplies chapter names, with EPUB 3 preferred over NCX."""
    source = make_epub(
        tmp_path / "book.epub",
        chapters={"one": xhtml('<h1 id="start">Epigraph credit</h1><p>Story.</p>')},
        spine=["one"],
    )
    _navigation(
        source,
        nav='<nav xmlns:epub="http://www.idpf.org/2007/ops" epub:type="landmarks">'
        '<a href="text/one.xhtml">Wrong</a></nav>'
        '<nav xmlns:epub="http://www.idpf.org/2007/ops" epub:type="toc">'
        '<ol><li><a href="text/one.xhtml#start">Chapter <span>One</span></a>'
        '<ol><li><a href="text/one.xhtml#later">Section</a></li></ol>'
        "</li></ol></nav>",
        ncx=_point("text/one.xhtml", "Old title"),
    )
    assert inspect_epub(source).chapters[0].title == "Chapter One"


@pytest.mark.parametrize("href", ["https://example.com/", "../../outside", "missing"])
def test_unusable_navigation_links_do_not_replace_fallback(
    tmp_path: Path, href: str
) -> None:
    """External, unsafe, and missing targets cannot supply a chapter label."""
    source = make_epub(
        tmp_path / "book.epub", chapters={"one": xhtml("<h1>One</h1>")}, spine=["one"]
    )
    _navigation(source, ncx=_point(href, "Wrong"))
    assert inspect_epub(source).chapters[0].title == "One"


def test_fragment_spine_uses_exact_navigation_target(tmp_path: Path) -> None:
    """A document-level label must not mask labels for individual fragments."""
    body = xhtml(
        '<section id="a"><p>A.</p></section><section id="b"><p>B.</p></section>'
    )
    source = make_epub(
        tmp_path / "book.epub",
        chapters={"a": body, "b": body},
        spine=["a", "b"],
        hrefs={"a": "text/shared.xhtml#a", "b": "text/shared.xhtml#b"},
    )
    _navigation(
        source,
        ncx=_point("text/shared.xhtml#a", "First")
        + _point("text/shared.xhtml#b", "Second"),
    )
    assert [c.title for c in inspect_epub(source).chapters] == ["First", "Second"]


@pytest.mark.parametrize("name", ["c3CN", "c3CN_split_001"])
def test_filename_document_titles_use_numbered_fallback(
    tmp_path: Path, name: str
) -> None:
    """Opaque source filenames are not useful audiobook chapter names."""
    source = make_epub(
        tmp_path / "book.epub",
        chapters={name: xhtml("<p>Notes.</p>", title="c3CN")},
        spine=[name],
    )
    assert inspect_epub(source).chapters[0].title == "Untitled section 1"


def test_navigation_skips_empty_entries_and_incomplete_points(tmp_path: Path) -> None:
    """Incomplete optional labels leave the meaningful document title available."""
    source = make_epub(
        tmp_path / "book.epub",
        chapters={"one": xhtml("<p>Story.</p>", title="A meaningful title")},
        spine=["one"],
    )
    _navigation(
        source,
        ncx=_point("text/one.xhtml", " ")
        + _point("", "Empty target")
        + '<navPoint><content src="text/one.xhtml"/></navPoint>',
    )
    assert inspect_epub(source).chapters[0].title == "A meaningful title"


def test_toc_intervals_preserve_unlisted_content_and_annotations(
    tmp_path: Path,
) -> None:
    """Physical files and epigraph credits never create extra TOC chapters."""
    source = make_epub(
        tmp_path / "book.epub",
        chapters={
            "front": xhtml("<p>Preface.</p>", title="front"),
            "one": xhtml("<h1>ONE</h1><p><em>Opening.</em></p>"),
            "extra": xhtml("<p>Fragment.</p><h2>CREDIT</h2><hr/><p>Scene.</p>"),
            "two": xhtml("<h1>TWO</h1><p>Ending.</p>"),
            "report": xhtml("<p><em>Report.</em></p>", title="report"),
        },
        spine=["front", "one", "extra", "two", "report"],
    )
    before = inspect_epub(source)
    _navigation(
        source,
        ncx=_point("text/one.xhtml", "Chapter 1")
        + _point("text/two.xhtml", "Chapter 2"),
    )
    after = inspect_epub(source)
    assert [c.title for c in after.chapters] == [
        "Untitled section 1",
        "Chapter 1",
        "Chapter 2",
    ]
    assert "\n\n".join(c.text for c in before.chapters) == "\n\n".join(
        c.text for c in after.chapters
    )
    for attribute in ("emphasis", "heading_ranges", "scene_ranges"):
        original = [
            c.text[a:b] for c in before.chapters for a, b in getattr(c, attribute)
        ]
        grouped = [
            c.text[a:b] for c in after.chapters for a, b in getattr(c, attribute)
        ]
        assert grouped == original
    assert after.chapters[1].id != before.chapters[1].id
    assert inspect_epub(source) == after


def test_fragment_toc_splits_one_file_in_reading_order(tmp_path: Path) -> None:
    """TOC anchors partition siblings, not just the target element's subtree."""
    source = make_epub(
        tmp_path / "book.epub",
        chapters={
            "one": xhtml(
                '<p>Before.</p><h1 id="a">Repeated</h1><p><em>First.</em></p>'
                '<h1 id="b">Repeated</h1><p>Second.</p><hr/><p>Last scene.</p>'
            )
        },
        spine=["one"],
    )
    _navigation(
        source,
        ncx=_point("text/one.xhtml#b", "Chapter 2")
        + _point("text/one.xhtml#a", "Chapter 1")
        + _point("text/one.xhtml#a", "Duplicate link"),
    )
    chapters = inspect_epub(source).chapters
    assert [c.text for c in chapters] == [
        "Before.",
        "Repeated\n\nFirst.",
        "Repeated\n\nSecond.\n\nLast scene.",
    ]
    assert [c.title for c in chapters[1:]] == ["Chapter 1", "Chapter 2"]
    assert len({c.id for c in chapters}) == len(chapters)
    assert chapters[1].text[slice(*chapters[1].emphasis[0])] == "First."
    assert chapters[2].text[slice(*chapters[2].scene_ranges[0])] == "Last scene."


@pytest.mark.parametrize("target", ["missing", "hidden", "duplicate"])
def test_invalid_toc_anchors_do_not_name_or_drop_text(
    tmp_path: Path, target: str
) -> None:
    """Invalid fragment links cannot silently become document-level titles."""
    source = make_epub(
        tmp_path / "book.epub",
        chapters={
            "one": xhtml(
                '<p>Body.</p><p id="hidden" hidden="">Hidden.</p>'
                '<span id="duplicate"/><span id="duplicate"/><h2>Internal credit</h2>',
                title="one",
            )
        },
        spine=["one"],
    )
    _navigation(source, ncx=_point(f"text/one.xhtml#{target}", "Wrong"))
    chapter = inspect_epub(source).chapters[0]
    assert chapter.title == "Untitled section 1"
    assert chapter.text == "Body.\n\nInternal credit"


def test_missing_toc_joins_splits_and_uses_sequential_neutral_labels(
    tmp_path: Path,
) -> None:
    """Fallback labels use logical section positions and reject internal credits."""
    source = make_epub(
        tmp_path / "book.epub",
        chapters={
            "x_split_000": xhtml("<p>A.</p>", title="x"),
            "x_split_001": xhtml("<p>B.</p>", title="x"),
            "y": xhtml("<p>C.</p><h2>CREDIT</h2>", title="y"),
            "z": xhtml("<h1>Afterword</h1><p>D.</p>", title="z"),
        },
        spine=["x_split_000", "x_split_001", "y", "z"],
    )
    chapters = inspect_epub(source).chapters
    assert [c.title for c in chapters] == [
        "Untitled section 1",
        "Untitled section 2",
        "Afterword",
    ]
    assert chapters[0].text == "A.\n\nB."


def test_malformed_epub3_navigation_falls_back_to_ncx(tmp_path: Path) -> None:
    """Optional malformed navigation must not make readable content unusable."""
    source = make_epub(
        tmp_path / "book.epub",
        chapters={"one": xhtml("<p>Body.</p>")},
        spine=["one"],
    )
    _navigation(source, nav="<nav>", ncx=_point("text/one.xhtml", "Chapter 1"))
    assert inspect_epub(source).chapters[0].title == "Chapter 1"


def test_authored_duplicate_titles_and_number_restarts_are_preserved(
    tmp_path: Path,
) -> None:
    """TOC labels are authoritative even when numbers repeat or skip."""
    source = make_epub(
        tmp_path / "book.epub",
        chapters={name: xhtml(f"<p>{name}.</p>") for name in ("a", "b", "c")},
        spine=["a", "b", "c"],
    )
    _navigation(
        source,
        ncx=_point("text/a.xhtml", "Chapter 1")
        + _point("text/b.xhtml", "Chapter 1")
        + _point("text/c.xhtml", "Chapter 50"),
    )
    chapters = inspect_epub(source).chapters
    assert [c.title for c in chapters] == ["Chapter 1", "Chapter 1", "Chapter 50"]
    assert len({c.id for c in chapters}) == len(chapters)


def test_empty_anchor_before_heading_is_a_valid_toc_boundary(tmp_path: Path) -> None:
    """Empty publisher anchors still point to the following readable content."""
    source = make_epub(
        tmp_path / "book.epub",
        chapters={
            "one": xhtml(
                '<p>Prefix.</p><a id="start"/>'
                '<h1 id="heading">Opening</h1><p>Story.</p>'
            )
        },
        spine=["one"],
    )
    _navigation(
        source,
        ncx=_point("text/one.xhtml#start", "Chapter 1")
        + _point("text/one.xhtml#heading", "Alternate title"),
    )
    chapters = inspect_epub(source).chapters
    assert [c.text for c in chapters] == ["Prefix.", "Opening\n\nStory."]
    assert chapters[1].title == "Chapter 1"


def test_fragment_chapter_ids_survive_prose_and_title_edits(tmp_path: Path) -> None:
    """Semantic anchors, not current character offsets, identify chapters."""
    ids = []
    for text in ("Short", "Much longer revised prose"):
        source = make_epub(
            tmp_path / "book.epub",
            chapters={
                "one": xhtml(
                    f'<h1 id="a">{text}</h1><p>First.</p>'
                    '<h1 id="b">Second</h1><p>Last.</p>'
                )
            },
            spine=["one"],
        )
        _navigation(
            source,
            ncx=_point("text/one.xhtml#a", text) + _point("text/one.xhtml#b", "Second"),
        )
        ids.append([c.id for c in inspect_epub(source).chapters])
    assert ids[0] == ids[1]


def test_toc_expansion_respects_chapter_limit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Repeated documents with fragment chapters cannot bypass the spine budget."""
    source = make_epub(
        tmp_path / "book.epub",
        chapters={"one": xhtml('<p id="a">First.</p><p id="b">Second.</p>')},
        spine=["one", "one"],
    )
    _navigation(
        source,
        ncx=_point("text/one.xhtml#a", "First") + _point("text/one.xhtml#b", "Second"),
    )
    monkeypatch.setattr("kenkui._epub.parser.MAX_SPINE_CHAPTERS", 2)
    with pytest.raises(SourceError) as error:
        inspect_epub(source)
    assert error.value.code == ErrorCode.ARCHIVE_LIMIT
