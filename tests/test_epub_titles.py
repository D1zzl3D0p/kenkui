"""Chapter labels come from navigation, not incidental epigraph headings."""

from __future__ import annotations

from typing import TYPE_CHECKING
from zipfile import ZipFile

import pytest

from helpers import make_epub, xhtml
from kenkui._epub.parser import inspect_epub

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
    assert [c.title for c in after.chapters] == [
        "Chapter 1",
        "Chapter 1 (part 2)",
        "Chapter 2",
        "Afterword",
    ]
    assert [(c.id, c.text, c.headings, c.heading_ranges) for c in after.chapters] == [
        (c.id, c.text, c.headings, c.heading_ranges) for c in before.chapters
    ]


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
    assert inspect_epub(source).chapters[0].title == "Chapter 1"


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
