"""Focused WP3 EPUB parsing, normalization, and selection tests."""

from __future__ import annotations

import shutil
import unicodedata
from dataclasses import FrozenInstanceError
from typing import TYPE_CHECKING
from zipfile import ZIP_DEFLATED, ZIP_STORED, ZipFile, ZipInfo

import pytest

import kenkui as kk
from kenkui._domain.selection import select_chapters
from kenkui._domain.text import SPACE_CODEPOINTS, normalize_text
from kenkui._epub import parser as epub_parser
from kenkui._epub.archive import ArchiveLimits, validate_archive
from kenkui._epub.identifiers import chapter_id
from kenkui._epub.paths import resolve_member

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence
    from pathlib import Path

EXPECTED_DUPLICATE_SPINE_CHAPTERS = 3

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


def test_inspection_metadata_visible_text_boundaries_cover_and_immutability(
    tmp_path: Path,
) -> None:
    """Inspection parses metadata and exact visible normalized chapter speech."""
    source = make_epub(
        tmp_path / "book.epub",
        chapters={
            "one": xhtml(
                "<h1>Same title</h1><p>Hello <em>brave</em>world<br/>again.</p>"
                "<script>bad</script><style>bad</style><noscript>bad</noscript>"
                '<p hidden="hidden">bad</p><div aria-hidden="true">bad</div>'
                "<p>Last&nbsp;line.</p>"
            )
        },
        spine=["one"],
        cover=True,
    )
    inspection = kk.epub(source).inspect()

    assert inspection.metadata == kk.BookMetadata(
        title="Tiny Book", author="Ada Author", cover_available=True
    )
    assert len(inspection.chapters) == 1
    chapter = inspection.chapters[0]
    assert chapter.title == "Same title"
    assert chapter.text == "Same title\n\nHello brave world\nagain.\n\nLast line."
    assert chapter.speech_characters == len(chapter.text)
    with pytest.raises(FrozenInstanceError):
        chapter.text = "changed"  # type: ignore[misc]


def test_ids_are_relocatable_title_independent_and_occurrence_fragment_sensitive(
    tmp_path: Path,
) -> None:
    """Every spine occurrence has a deterministic semantic, location-free ID."""
    hrefs = {"one": "text/ch.xhtml#part-a", "two": "text/ch.xhtml#part-b"}
    document = xhtml(
        '<section id="part-a"><h1>Same</h1>A</section>'
        '<section id="part-b"><h1>Same</h1>B</section>'
    )
    chapters = {"one": document, "two": document}
    source = make_epub(
        tmp_path / "first.epub",
        chapters=chapters,
        spine=["one", "one", "two"],
        hrefs=hrefs,
    )
    relocated = tmp_path / "nested" / "relocated.epub"
    relocated.parent.mkdir()
    shutil.copyfile(source, relocated)

    first = kk.epub(source).inspect()
    second = kk.epub(relocated).inspect()
    assert first == second
    assert (
        len({chapter.id for chapter in first.chapters})
        == EXPECTED_DUPLICATE_SPINE_CHAPTERS
    )
    assert [chapter.title for chapter in first.chapters] == ["Same", "Same", "Same"]
    assert first.chapters[0].id != first.chapters[1].id
    assert first.chapters[0].id != first.chapters[2].id

    retitled = make_epub(
        tmp_path / "retitled.epub",
        chapters={"one": document.replace("Same", "Changed"), "two": document},
        spine=["one", "one", "two"],
        hrefs=hrefs,
        title="A different title",
    )
    assert [c.id for c in kk.epub(retitled).inspect().chapters] == [
        c.id for c in first.chapters
    ]


def test_normalization_is_versioned_nfc_idempotent_and_preserves_content() -> None:
    """Normalization has documented deterministic whitespace and Unicode semantics."""
    raw = "  Cafe\u0301\u00a0\tWORD\r\n \r\n\r\n punctuation!?  "
    expected = "Café WORD\n\npunctuation!?"
    assert normalize_text(raw) == expected
    assert normalize_text(expected) == expected
    assert unicodedata.is_normalized("NFC", normalize_text(raw))


def test_normalization_idempotence_over_generated_spacing_samples() -> None:
    """Exercise idempotence over every explicitly mapped spacing character."""
    samples = [
        f"left{space}{space}right\r\n\rthird\n\n\nfourth"
        for space in sorted(SPACE_CODEPOINTS)
    ]
    samples.extend(["", "plain", "e\u0301", "\t surrounded \t"])
    for sample in samples:
        once = normalize_text(sample)
        assert normalize_text(once) == once


@pytest.mark.parametrize(
    ("base", "href", "expected"),
    [
        ("OPS/package.opf", "text/../ch.xhtml#x", ("OPS/ch.xhtml", "x")),
        ("package.opf", "a%20b.xhtml", ("a b.xhtml", "")),
    ],
)
def test_canonical_posix_resolution(
    base: str, href: str, expected: tuple[str, str]
) -> None:
    """Safe hrefs resolve canonically and preserve fragment identity."""
    assert resolve_member(base, href) == expected


@pytest.mark.parametrize(
    "href", ["../../../evil", "/absolute", "https://evil/x", "a\\b", "%2e%2e/%2e%2e/x"]
)
def test_canonical_posix_resolution_rejects_unsafe_members(href: str) -> None:
    """Traversal, external, absolute, and non-POSIX member paths are rejected."""
    with pytest.raises(kk.SourceError) as caught:
        resolve_member("OPS/package.opf", href)
    assert caught.value.code == kk.ErrorCode.UNSAFE_ARCHIVE_PATH


def test_selection_all_explicit_and_inclusive_range(tmp_path: Path) -> None:
    """Inspection applies all, explicit caller-order, and inclusive range semantics."""
    source = make_epub(
        tmp_path / "book.epub",
        chapters={name: xhtml(f"<h1>{name}</h1><p>{name}</p>") for name in "abcd"},
        spine=list("abcd"),
    )
    all_chapters = kk.epub(source).inspect().chapters
    ids = [chapter.id for chapter in all_chapters]
    assert kk.epub(source).select_chapters(ids[2], ids[0]).inspect().chapters == (
        all_chapters[2],
        all_chapters[0],
    )
    assert kk.epub(source).select_chapter_range(ids[1], ids[3]).inspect().chapters == (
        all_chapters[1],
        all_chapters[2],
        all_chapters[3],
    )

    for pipeline, code in [
        (kk.epub(source).select_chapters("missing"), kk.ErrorCode.CHAPTER_NOT_FOUND),
        (
            kk.epub(source).select_chapter_range(ids[3], ids[1]),
            kk.ErrorCode.REVERSED_CHAPTER_RANGE,
        ),
    ]:
        with pytest.raises(kk.ValidationError) as caught:
            pipeline.inspect()
        assert caught.value.code == code


def test_selection_helper_rejects_ambiguous_duplicate_available_ids() -> None:
    """Materialized selection never silently chooses duplicate IDs."""
    chapters = (
        kk.ChapterInspection("same", 0, "A", 1, "a"),
        kk.ChapterInspection("same", 1, "B", 1, "b"),
    )
    with pytest.raises(kk.ValidationError) as caught:
        select_chapters(chapters, ("same",))
    assert caught.value.code == kk.ErrorCode.DUPLICATE_CHAPTER_ID


def test_empty_spine_and_empty_visible_chapter_have_stable_errors(
    tmp_path: Path,
) -> None:
    """Invalid empty selections and chapters fail at authoritative inspection."""
    empty_spine = make_epub(
        tmp_path / "empty-spine.epub", chapters={"one": xhtml("text")}, spine=[]
    )
    empty_chapter = make_epub(
        tmp_path / "empty-chapter.epub",
        chapters={"one": xhtml("<script>only hidden</script>")},
        spine=["one"],
    )
    for source, code in [
        (empty_spine, kk.ErrorCode.EMPTY_SELECTION),
        (empty_chapter, kk.ErrorCode.EMPTY_CHAPTER),
    ]:
        with pytest.raises(kk.SourceError) as caught:
            kk.epub(source).inspect()
        assert caught.value.code == code


def test_malformed_missing_and_traversal_epubs_have_stable_errors(
    tmp_path: Path,
) -> None:
    """Malformed ZIP/XML/container/manifest paths do not leak parser exceptions."""
    not_zip = tmp_path / "not-zip.epub"
    not_zip.write_bytes(b"not a zip")
    malformed = tmp_path / "malformed.epub"
    with ZipFile(malformed, "w") as archive:
        archive.writestr("mimetype", "application/epub+zip")
        archive.writestr("META-INF/container.xml", "<broken")
    traversal = tmp_path / "traversal.epub"
    bad_container = CONTAINER.replace("OPS/package.opf", "../../evil.opf")
    with ZipFile(traversal, "w") as archive:
        archive.writestr("mimetype", "application/epub+zip")
        archive.writestr("META-INF/container.xml", bad_container)
    for source, code in [
        (not_zip, kk.ErrorCode.MALFORMED_EPUB),
        (malformed, kk.ErrorCode.MALFORMED_EPUB),
        (traversal, kk.ErrorCode.UNSAFE_ARCHIVE_PATH),
    ]:
        with pytest.raises(kk.SourceError) as caught:
            kk.epub(source).inspect()
        assert caught.value.code == code


def test_missing_opf_malformed_xhtml_and_invalid_spine_are_rejected(
    tmp_path: Path,
) -> None:
    """Required package layers and spine references have one sanitized failure."""
    missing_container = tmp_path / "missing-container.epub"
    with ZipFile(missing_container, "w") as archive:
        archive.writestr("mimetype", "application/epub+zip")

    missing_opf = tmp_path / "missing-opf.epub"
    with ZipFile(missing_opf, "w") as archive:
        archive.writestr("mimetype", "application/epub+zip")
        archive.writestr("META-INF/container.xml", CONTAINER)

    invalid_spine = make_epub(
        tmp_path / "invalid-spine.epub",
        chapters={"one": xhtml("text")},
        spine=["missing"],
    )
    malformed_xhtml = make_epub(
        tmp_path / "malformed-xhtml.epub",
        chapters={"one": "<html><body><broken></body></html>"},
        spine=["one"],
    )
    entity_xhtml = make_epub(
        tmp_path / "entity-xhtml.epub",
        chapters={
            "one": (
                '<!DOCTYPE html [<!ENTITY x "unsafe">]><html><body>&x;</body></html>'
            )
        },
        spine=["one"],
    )
    for source in (
        missing_container,
        missing_opf,
        invalid_spine,
        malformed_xhtml,
        entity_xhtml,
    ):
        with pytest.raises(kk.SourceError) as caught:
            kk.epub(source).inspect()
        assert caught.value.code == kk.ErrorCode.MALFORMED_EPUB


def test_archive_limits_members_sizes_total_ratio_and_duplicate_names(
    tmp_path: Path,
) -> None:
    """All ZIP metadata is validated before any member content is consumed."""
    cases: list[tuple[str, ArchiveLimits, list[tuple[str, bytes]], kk.ErrorCode]] = [
        (
            "members",
            ArchiveLimits(max_members=1),
            [("a", b"1"), ("b", b"2")],
            kk.ErrorCode.ARCHIVE_LIMIT,
        ),
        (
            "member",
            ArchiveLimits(max_member_size=1),
            [("a", b"12")],
            kk.ErrorCode.ARCHIVE_LIMIT,
        ),
        (
            "total",
            ArchiveLimits(max_total_size=2),
            [("a", b"12"), ("b", b"3")],
            kk.ErrorCode.ARCHIVE_LIMIT,
        ),
        (
            "ratio",
            ArchiveLimits(max_ratio=1.0),
            [("a", b"x" * 1000)],
            kk.ErrorCode.ARCHIVE_LIMIT,
        ),
        (
            "duplicate",
            ArchiveLimits(),
            [("a", b"1"), ("a", b"2")],
            kk.ErrorCode.MALFORMED_EPUB,
        ),
        (
            "traversal",
            ArchiveLimits(),
            [("../a", b"1")],
            kk.ErrorCode.UNSAFE_ARCHIVE_PATH,
        ),
    ]
    for name, limits, members, code in cases:
        source = tmp_path / f"{name}.epub"
        with ZipFile(source, "w", compression=ZIP_DEFLATED) as archive:
            for member, data in members:
                archive.writestr(member, data)
        with ZipFile(source) as archive, pytest.raises(kk.SourceError) as caught:
            validate_archive(archive, limits)
        assert caught.value.code == code


def test_archive_rejects_encrypted_metadata_flag(tmp_path: Path) -> None:
    """Encrypted members are rejected deterministically before reading."""
    source = tmp_path / "encrypted.epub"
    with ZipFile(source, "w") as archive:
        info = ZipInfo("a")
        archive.writestr(info, b"data")
    with ZipFile(source) as archive:
        archive.filelist[0].flag_bits |= 0x1
        with pytest.raises(kk.SourceError) as caught:
            validate_archive(archive)
    assert caught.value.code == kk.ErrorCode.MALFORMED_EPUB


def test_identifier_algorithm_has_explicit_version_and_semantic_inputs() -> None:
    """Stable IDs vary only with canonical path, occurrence, and fragment."""
    baseline = chapter_id("OPS/text/a.xhtml", 0, "fragment")
    assert baseline.startswith("ch-v1-")
    assert baseline == chapter_id("OPS/text/a.xhtml", 0, "fragment")
    assert baseline != chapter_id("OPS/text/b.xhtml", 0, "fragment")
    assert baseline != chapter_id("OPS/text/a.xhtml", 1, "fragment")
    assert baseline != chapter_id("OPS/text/a.xhtml", 0, "other")


@pytest.mark.parametrize("encoding", ["utf-16", "utf-32"])
def test_encoding_aware_xml_parser_rejects_non_ascii_entity_declarations(
    tmp_path: Path, encoding: str
) -> None:
    """UTF-16/32 DTDs cannot bypass entity hardening via embedded NUL bytes."""
    document = (
        (
            '<?xml version="1.0" encoding="UTF-8"?>'
            '<!DOCTYPE html [<!ENTITY x "unsafe">]>'
            "<html><body>&x;</body></html>"
        )
        .replace("UTF-8", encoding.upper())
        .encode(encoding)
    )
    source = make_epub(
        tmp_path / f"entity-{encoding}.epub",
        chapters={"one": document},
        spine=["one"],
    )

    with pytest.raises(kk.SourceError) as caught:
        kk.epub(source).inspect()
    assert caught.value.code == kk.ErrorCode.MALFORMED_EPUB


def test_spine_count_and_cumulative_speech_work_are_bounded(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Repeated spine references cannot amplify bounded archive input without limit."""
    source = make_epub(
        tmp_path / "amplified.epub",
        chapters={"one": xhtml("<p>12345</p>")},
        spine=["one", "one", "one"],
    )
    monkeypatch.setattr(epub_parser, "MAX_SPINE_CHAPTERS", 2)
    with pytest.raises(kk.SourceError) as caught:
        kk.epub(source).inspect()
    assert caught.value.code == kk.ErrorCode.ARCHIVE_LIMIT

    monkeypatch.setattr(epub_parser, "MAX_SPINE_CHAPTERS", 3)
    monkeypatch.setattr(epub_parser, "MAX_SPEECH_CHARACTERS", 12)
    with pytest.raises(kk.SourceError) as caught:
        kk.epub(source).inspect()
    assert caught.value.code == kk.ErrorCode.ARCHIVE_LIMIT


def test_duplicate_spine_material_is_parsed_once_but_keeps_occurrences(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Duplicate canonical member/fragment occurrences share materialization work."""
    source = make_epub(
        tmp_path / "duplicates.epub",
        chapters={"one": xhtml("<h1>One</h1><p>Text</p>")},
        spine=["one", "one"],
    )
    original = epub_parser._xhtml  # noqa: SLF001 - verifies the cache boundary.
    calls = 0

    def counting_xhtml(data: bytes) -> object:
        nonlocal calls
        calls += 1
        return original(data)

    monkeypatch.setattr(epub_parser, "_xhtml", counting_xhtml)
    chapters = kk.epub(source).inspect().chapters

    assert calls == 1
    assert [chapter.text for chapter in chapters] == ["One\n\nText"] * 2
    assert chapters[0].id != chapters[1].id


def test_distinct_fragments_parse_and_index_canonical_member_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Distinct fragment material shares one document parse and body-ID index."""
    document = xhtml(
        '<section id="first"><h1>First</h1><p>Alpha</p></section>'
        '<section id="second"><h1>Second</h1><p>Beta</p></section>'
    )
    source = make_epub(
        tmp_path / "fragments.epub",
        chapters={"one": document, "two": document},
        spine=["one", "two"],
        hrefs={"one": "text/shared.xhtml#first", "two": "text/shared.xhtml#second"},
    )
    original_xhtml = epub_parser._xhtml  # noqa: SLF001 - verifies cache boundary.
    original_index = epub_parser._body_fragment_index  # noqa: SLF001
    parse_calls = 0
    index_calls = 0

    def counting_xhtml(data: bytes) -> object:
        nonlocal parse_calls
        parse_calls += 1
        return original_xhtml(data)

    def counting_index(root: object) -> object:
        nonlocal index_calls
        index_calls += 1
        return original_index(root)  # type: ignore[arg-type]

    monkeypatch.setattr(epub_parser, "_xhtml", counting_xhtml)
    monkeypatch.setattr(epub_parser, "_body_fragment_index", counting_index)

    chapters = kk.epub(source).inspect().chapters

    assert parse_calls == 1
    assert index_calls == 1
    assert [(chapter.title, chapter.text) for chapter in chapters] == [
        ("First", "First\n\nAlpha"),
        ("Second", "Second\n\nBeta"),
    ]


def test_unknown_xml_encoding_is_sanitized_by_public_pipeline(tmp_path: Path) -> None:
    """Codec lookup failures never escape the public inspection boundary."""
    document = (
        b'<?xml version="1.0" encoding="x-kenkui-unknown"?>'
        b"<html><body><p>text</p></body></html>"
    )
    source = make_epub(
        tmp_path / "unknown-encoding.epub",
        chapters={"one": document},
        spine=["one"],
    )

    with pytest.raises(kk.SourceError) as caught:
        kk.epub(source).inspect()
    assert caught.value.code == kk.ErrorCode.MALFORMED_EPUB


def test_event_parser_limits_shallow_wide_nodes_and_excessive_attributes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Start-event work limits reject wide and attribute-heavy XML trees."""
    wide = make_epub(
        tmp_path / "wide.epub",
        chapters={"one": xhtml("<p>x</p>" * 12)},
        spine=["one"],
    )
    monkeypatch.setattr(epub_parser, "MAX_XML_ELEMENTS", 10)
    with pytest.raises(kk.SourceError) as caught:
        kk.epub(wide).inspect()
    assert caught.value.code == kk.ErrorCode.ARCHIVE_LIMIT

    monkeypatch.setattr(epub_parser, "MAX_XML_ELEMENTS", 100)
    monkeypatch.setattr(epub_parser, "MAX_ATTRIBUTES_PER_ELEMENT", 3)
    attributed = make_epub(
        tmp_path / "attributes.epub",
        chapters={"one": xhtml('<p a="1" b="2" c="3" d="4">x</p>')},
        spine=["one"],
    )
    with pytest.raises(kk.SourceError) as caught:
        kk.epub(attributed).inspect()
    assert caught.value.code == kk.ErrorCode.ARCHIVE_LIMIT


def test_deep_xhtml_has_stable_limit_error_and_template_is_inert(
    tmp_path: Path,
) -> None:
    """Deep trees never leak RecursionError, and template content is not speech."""
    depth = epub_parser.MAX_ELEMENT_DEPTH + 1
    document = xhtml(
        f"{'<div>' * depth}visible{'</div>' * depth}<template>inert</template>"
    )
    source = make_epub(
        tmp_path / "deep.epub", chapters={"one": document}, spine=["one"]
    )
    with pytest.raises(kk.SourceError) as caught:
        kk.epub(source).inspect()
    assert caught.value.code == kk.ErrorCode.ARCHIVE_LIMIT

    shallow = make_epub(
        tmp_path / "template.epub",
        chapters={"one": xhtml("<p>spoken</p><template>inert</template>")},
        spine=["one"],
    )
    assert kk.epub(shallow).inspect().chapters[0].text == "spoken"


def test_source_only_pipeline_can_inspect_fixture_epub(tmp_path: Path) -> None:
    """Inspection reads a valid EPUB without voice or synthesis intent."""
    fixture_epub = make_epub(
        tmp_path / "fixture.epub",
        chapters={"chapter": xhtml("<p>Inspectable source.</p>")},
        spine=("chapter",),
    )

    inspection = kk.epub(fixture_epub).inspect()

    assert inspection.chapters


def test_inspection_logs_safe_parse_context(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """Inspection logs only structured parse metadata, never source contents."""
    source = make_epub(
        tmp_path / "secret-source.epub",
        chapters={"chapter": xhtml("<p>Secret source text.</p>")},
        spine=("chapter",),
    )
    caplog.set_level("INFO", logger="kenkui.pipeline")

    kk.epub(source).inspect()

    record = next(
        entry
        for entry in caplog.records
        if getattr(entry, "event", None) == "inspection_completed"
    )
    assert record.boundary == "parse"
    assert record.chapter_count == 1
    assert str(source) not in caplog.text
    assert "Secret source text." not in caplog.text
