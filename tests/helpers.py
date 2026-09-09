"""Shared EPUB-building helpers for tests.

Moved out of ``test_epub.py`` so any test module can build a tiny,
project-authored EPUB fixture without importing another test module.
"""

from __future__ import annotations

from typing import TYPE_CHECKING
from zipfile import ZIP_STORED, ZipFile

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence
    from pathlib import Path

CONTAINER = """<?xml version="1.0"?>
<container xmlns="urn:oasis:names:tc:opendocument:xmlns:container"
 version="1.0">
 <rootfiles><rootfile full-path="OPS/package.opf"
  media-type="application/oebps-package+xml"/></rootfiles>
</container>"""


def make_epub(  # noqa: PLR0913 - options intentionally map EPUB concepts.
    path: Path,
    *,
    chapters: Mapping[str, str | bytes],
    spine: Sequence[str],
    hrefs: Mapping[str, str] | None = None,
    title: str = "Tiny Book",
    author: str = "Ada Author",
    cover: bool = False,
) -> Path:
    """Generate a project-authored tiny EPUB fixture."""
    hrefs = hrefs or {name: f"text/{name}.xhtml" for name in chapters}
    cover_item = (
        '<item id="cover" href="images/cover.png" media-type="image/png" '
        'properties="cover-image"/>'
        if cover
        else ""
    )
    manifest = "".join(
        f'<item id="{name}" href="{hrefs[name]}" media-type="application/xhtml+xml"/>'
        for name in chapters
    )
    spine_xml = "".join(f'<itemref idref="{name}"/>' for name in spine)
    opf = f"""<?xml version="1.0"?>
<package xmlns="http://www.idpf.org/2007/opf" version="3.0">
 <metadata xmlns:dc="http://purl.org/dc/elements/1.1/">
  <dc:title>{title}</dc:title><dc:creator>{author}</dc:creator>
  {('<meta name="cover" content="cover"/>' if cover else "")}
 </metadata>
 <manifest>{manifest}{cover_item}</manifest><spine>{spine_xml}</spine>
</package>"""
    with ZipFile(path, "w") as archive:
        archive.writestr("mimetype", "application/epub+zip", compress_type=ZIP_STORED)
        archive.writestr("META-INF/container.xml", CONTAINER)
        archive.writestr("OPS/package.opf", opf)
        written: set[str] = set()
        for name, body in chapters.items():
            member = hrefs[name].split("#", maxsplit=1)[0]
            if member not in written:
                archive.writestr(f"OPS/{member}", body)
                written.add(member)
        if cover:
            archive.writestr(
                "OPS/images/cover.png",
                bytes.fromhex(
                    "89504e470d0a1a0a0000000d4948445200000001000000010804000000"
                    "b51c0c020000000b4944415478da6364f80f00010501012718e3660000"
                    "000049454e44ae426082"
                ),
            )
    return path


def xhtml(body: str, *, title: str = "Document title") -> str:
    """Wrap project-authored XHTML body content."""
    return (
        '<html xmlns="http://www.w3.org/1999/xhtml"><head>'
        f"<title>{title}</title></head><body>{body}</body></html>"
    )
