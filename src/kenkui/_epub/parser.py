"""Secure deterministic EPUB package and visible-text parser."""

from __future__ import annotations

import re
from collections import Counter
from io import BytesIO
from typing import TYPE_CHECKING, cast
from xml.etree import ElementTree as ET
from zipfile import BadZipFile, LargeZipFile, ZipFile

from defusedxml import DefusedXmlException
from defusedxml.ElementTree import iterparse as safe_iterparse

from kenkui._domain.text import normalize_text
from kenkui.errors import ErrorCode, SourceError
from kenkui.inspection import BookInspection, BookMetadata, ChapterInspection

from .archive import validate_archive
from .identifiers import chapter_id
from .paths import canonical_member, resolve_member

if TYPE_CHECKING:
    from pathlib import Path
    from xml.etree.ElementTree import Element
    from zipfile import ZipFile as TypedZipFile

_CONTAINER = "META-INF/container.xml"
_NAMED_SPACES = re.compile(
    rb"&(?:nbsp|ensp|emsp|thinsp|hairsp|numsp|puncsp|nnbsp);", re.IGNORECASE
)
_BLOCK_ELEMENTS = frozenset(
    {
        "address",
        "article",
        "aside",
        "blockquote",
        "dd",
        "div",
        "dl",
        "dt",
        "figcaption",
        "figure",
        "footer",
        "form",
        "h1",
        "h2",
        "h3",
        "h4",
        "h5",
        "h6",
        "header",
        "hr",
        "li",
        "main",
        "nav",
        "ol",
        "p",
        "pre",
        "section",
        "table",
        "tbody",
        "td",
        "tfoot",
        "th",
        "thead",
        "tr",
        "ul",
    }
)
_IGNORED_ELEMENTS = frozenset({"script", "style", "noscript", "template"})

# Parsing and speech-materialization work limits supplement the ZIP byte limits.
MAX_SPINE_CHAPTERS = 10_000
MAX_SPEECH_CHARACTERS = 50_000_000
MAX_ELEMENT_DEPTH = 256
MAX_XML_ELEMENTS = 100_000
MAX_XML_ATTRIBUTES = 100_000
MAX_ATTRIBUTES_PER_ELEMENT = 1_000


class _TextEmitter:
    """Emit semantic boundaries and prevent accidental inline word joining."""

    def __init__(self) -> None:
        self.parts: list[str] = []

    def text(self, value: str | None) -> None:
        if not value:
            return
        if (
            self.parts
            and self.parts[-1]
            and self.parts[-1][-1].isalnum()
            and value[0].isalnum()
        ):
            self.parts.append(" ")
        self.parts.append(value)

    def boundary(self, count: int) -> None:
        self.parts.append("\n" * count)

    def value(self) -> str:
        return "".join(self.parts)


def _local(tag: str) -> str:
    return tag.rsplit("}", maxsplit=1)[-1].lower()


def _enforce_xml_limits(
    depth: int, elements: int, attributes: int, element_attributes: int
) -> None:
    if (
        depth > MAX_ELEMENT_DEPTH
        or elements > MAX_XML_ELEMENTS
        or attributes > MAX_XML_ATTRIBUTES
        or element_attributes > MAX_ATTRIBUTES_PER_ELEMENT
    ):
        raise SourceError(ErrorCode.ARCHIVE_LIMIT)


def _xml(data: bytes) -> Element:
    try:
        events = safe_iterparse(BytesIO(data), events=("start", "end"), forbid_dtd=True)
        depth = 0
        elements = 0
        attributes = 0
        root: Element | None = None
        for event, element in events:
            if event == "start":
                depth += 1
                if depth == 1:
                    root = element
                elements += 1
                element_attributes = len(element.attrib)
                attributes += element_attributes
                _enforce_xml_limits(depth, elements, attributes, element_attributes)
            else:
                depth -= 1
    except SourceError:
        raise
    except (
        DefusedXmlException,
        ET.ParseError,
        LookupError,
        UnicodeError,
        ValueError,
    ) as error:
        raise SourceError(ErrorCode.MALFORMED_EPUB) from error
    else:
        return cast("Element", root)


def _xhtml(data: bytes) -> Element:
    data = _NAMED_SPACES.sub(b"&#160;", data)
    names = ("nbsp", "ensp", "emsp", "thinsp", "hairsp", "numsp", "puncsp", "nnbsp")
    for encoding in ("utf-16-le", "utf-16-be", "utf-32-le", "utf-32-be"):
        pattern = re.compile(
            b"|".join(re.escape(f"&{name};".encode(encoding)) for name in names),
            re.IGNORECASE,
        )
        data = pattern.sub("&#160;".encode(encoding), data)
    return _xml(data)


def _one(elements: list[Element]) -> Element:
    if len(elements) != 1:
        raise SourceError(ErrorCode.MALFORMED_EPUB)
    return elements[0]


def _hidden(element: Element) -> bool:
    attributes = {_local(key): value for key, value in element.attrib.items()}
    if "hidden" in attributes or attributes.get("aria-hidden", "").lower() == "true":
        return True
    style = re.sub(r"\s+", "", attributes.get("style", "").lower())
    return "display:none" in style or "visibility:hidden" in style


def _emit_element(element: Element, emitter: _TextEmitter) -> None:
    tag = _local(element.tag)
    if tag in _IGNORED_ELEMENTS or _hidden(element):
        return
    if tag == "br":
        emitter.boundary(1)
        return
    block = tag in _BLOCK_ELEMENTS
    if block:
        emitter.boundary(2)
    emitter.text(element.text)
    for child in element:
        _emit_element(child, emitter)
        emitter.text(child.tail)
    if block:
        emitter.boundary(2)


def _body_fragment_index(
    root: Element,
) -> tuple[Element, dict[str, Element | None]]:
    bodies = [element for element in root.iter() if _local(element.tag) == "body"]
    body = _one(bodies)
    fragments: dict[str, Element | None] = {}
    for element in body.iter():
        fragment = element.attrib.get("id")
        if fragment:
            fragments[fragment] = element if fragment not in fragments else None
    return body, fragments


def _body_scope(
    body: Element, fragments: dict[str, Element | None], fragment: str
) -> Element:
    if not fragment:
        return body
    match = fragments.get(fragment)
    if match is None:
        raise SourceError(ErrorCode.MALFORMED_EPUB)
    return match


def _visible_headings(element: Element) -> list[str]:
    """Collect heading text without descending through hidden ancestors."""
    if _local(element.tag) in _IGNORED_ELEMENTS or _hidden(element):
        return []
    if _local(element.tag) in {"h1", "h2", "h3", "h4", "h5", "h6"}:
        emitter = _TextEmitter()
        _emit_element(element, emitter)
        title = normalize_text(emitter.value())
        return [title] if title else []
    headings: list[str] = []
    for child in element:
        headings.extend(_visible_headings(child))
    return headings


def _chapter_text(
    root: Element,
    body: Element,
    fragments: dict[str, Element | None],
    fragment: str,
) -> tuple[str, str]:
    scope = _body_scope(body, fragments, fragment)
    emitter = _TextEmitter()
    _emit_element(scope, emitter)
    text = normalize_text(emitter.value())
    if not text:
        return "", ""
    headings = _visible_headings(scope)
    title = headings[0] if headings else ""
    if not title:
        document_titles = [
            element for element in root.iter() if _local(element.tag) == "title"
        ]
        title = (
            normalize_text(" ".join(document_titles[0].itertext()))
            if document_titles
            else ""
        )
    return title, text


def _member_map(archive: TypedZipFile) -> dict[str, str]:
    return {
        canonical_member(info.filename.rstrip("/")): info.filename
        for info in archive.infolist()
        if not info.is_dir()
    }


def _read(archive: TypedZipFile, members: dict[str, str], name: str) -> bytes:
    try:
        return archive.read(members[name])
    except (
        KeyError,
        BadZipFile,
        RuntimeError,
        NotImplementedError,
        OSError,
    ) as error:
        raise SourceError(ErrorCode.MALFORMED_EPUB) from error


def _package_path(container: Element) -> str:
    rootfiles = [
        element for element in container.iter() if _local(element.tag) == "rootfile"
    ]
    rootfile = _one(rootfiles)
    full_path = rootfile.attrib.get("full-path", "")
    return canonical_member(full_path)


def _metadata(
    package: Element, manifest: dict[str, tuple[str, str, str]], members: set[str]
) -> BookMetadata:
    titles = [element for element in package.iter() if _local(element.tag) == "title"]
    creators = [
        element for element in package.iter() if _local(element.tag) == "creator"
    ]
    title = normalize_text("".join(titles[0].itertext())) if titles else None
    author = normalize_text("".join(creators[0].itertext())) if creators else None
    cover_ids = {
        element.attrib.get("content", "")
        for element in package.iter()
        if _local(element.tag) == "meta"
        and element.attrib.get("name", "").lower() == "cover"
    }
    cover_available = any(
        path in members
        and ("cover-image" in properties.split() or item_id in cover_ids)
        for item_id, (path, _fragment, properties) in manifest.items()
    )
    return BookMetadata(title or None, author or None, cover_available)


def _document(
    archive: TypedZipFile,
    members: dict[str, str],
    member: str,
    cache: dict[str, tuple[Element, Element, dict[str, Element | None]]],
) -> tuple[Element, Element, dict[str, Element | None]]:
    document = cache.get(member)
    if document is None:
        root = _xhtml(_read(archive, members, member))
        body, fragments = _body_fragment_index(root)
        document = (root, body, fragments)
        cache[member] = document
    return document


def _spine_chapters(
    archive: TypedZipFile,
    members: dict[str, str],
    manifest: dict[str, tuple[str, str, str]],
    spine_element: Element,
) -> tuple[ChapterInspection, ...]:
    spine_items = [item for item in spine_element if _local(item.tag) == "itemref"]
    if not spine_items:
        raise SourceError(ErrorCode.EMPTY_SELECTION)
    if len(spine_items) > MAX_SPINE_CHAPTERS:
        raise SourceError(ErrorCode.ARCHIVE_LIMIT)
    occurrences: Counter[tuple[str, str]] = Counter()
    material_cache: dict[tuple[str, str], tuple[str, str]] = {}
    document_cache: dict[str, tuple[Element, Element, dict[str, Element | None]]] = {}
    speech_characters = 0
    chapters: list[ChapterInspection] = []
    for itemref in spine_items:
        idref = itemref.attrib.get("idref", "")
        try:
            member, fragment, _properties = manifest[idref]
        except KeyError as error:
            raise SourceError(ErrorCode.MALFORMED_EPUB) from error
        if member not in members:
            raise SourceError(ErrorCode.MALFORMED_EPUB)
        identity = (member, fragment)
        occurrence = occurrences[identity]
        occurrences[identity] += 1
        material = material_cache.get(identity)
        if material is None:
            document = _document(archive, members, member, document_cache)
            material = _chapter_text(*document, fragment)
            material_cache[identity] = material
        title, text = material
        if not text:
            # Image-only pages (covers, title pages, plates) carry no speech.
            continue
        speech_characters += len(text)
        if speech_characters > MAX_SPEECH_CHARACTERS:
            raise SourceError(ErrorCode.ARCHIVE_LIMIT)
        index = len(chapters)
        chapters.append(
            ChapterInspection(
                chapter_id(member, occurrence, fragment),
                index,
                title or f"Chapter {index + 1}",
                len(text),
                text,
            )
        )
    if not chapters:
        raise SourceError(ErrorCode.EMPTY_CHAPTER)
    return tuple(chapters)


def _parse(archive: TypedZipFile) -> BookInspection:
    validate_archive(archive)
    members = _member_map(archive)
    if _read(archive, members, "mimetype") != b"application/epub+zip":
        raise SourceError(ErrorCode.MALFORMED_EPUB)
    container = _xml(_read(archive, members, _CONTAINER))
    package_path = _package_path(container)
    package = _xml(_read(archive, members, package_path))

    manifest_elements = [
        element for element in package.iter() if _local(element.tag) == "manifest"
    ]
    spine_elements = [
        element for element in package.iter() if _local(element.tag) == "spine"
    ]
    manifest_element = _one(manifest_elements)
    spine_element = _one(spine_elements)
    manifest: dict[str, tuple[str, str, str]] = {}
    for item in manifest_element:
        if _local(item.tag) != "item":
            continue
        item_id = item.attrib.get("id", "")
        href = item.attrib.get("href", "")
        if not item_id or not href or item_id in manifest:
            raise SourceError(ErrorCode.MALFORMED_EPUB)
        path, fragment = resolve_member(package_path, href)
        manifest[item_id] = (path, fragment, item.attrib.get("properties", ""))

    chapters = _spine_chapters(archive, members, manifest, spine_element)
    return BookInspection(_metadata(package, manifest, set(members)), chapters)


def inspect_epub(path: Path) -> BookInspection:
    """Inspect an EPUB without extracting it or loading synthesis dependencies."""
    try:
        with ZipFile(path, "r") as archive:
            return _parse(archive)
    except SourceError:
        raise
    except (BadZipFile, LargeZipFile, OSError, ValueError) as error:
        raise SourceError(ErrorCode.MALFORMED_EPUB) from error
