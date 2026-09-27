"""Prose recovery and protected negatives for the native cleanup steps."""
# ruff: noqa: PLR2004

from dataclasses import replace

from kenkui.pdf_processing import (
    PdfBlock,
    PdfDocument,
    PdfLine,
    PdfPage,
    reconstruct_paragraphs,
    remove_furniture,
)


def _page(
    number: int, rows: tuple[tuple[str, float, float, float, float], ...]
) -> PdfPage:
    lines = tuple(
        PdfLine(f"{number}:{i}", number, text, (x, y, end, y + size), size, "Body")
        for i, (text, x, y, end, size) in enumerate(rows)
    )
    blocks = tuple(
        PdfBlock(line.id, number, line.text, sources=(line.id,), box=line.box)
        for line in lines
    )
    return PdfPage(number, 600, 800, blocks, lines)


def _document(*pages: PdfPage) -> PdfDocument:
    return PdfDocument("a" * 64, pages, tuple(b for p in pages for b in p.blocks))


def test_repeated_headers_removed_but_large_title_preserved() -> None:
    """Matching text and location alone must not delete the book title."""
    pages = tuple(
        _page(
            i,
            (
                ("The Quiet Book", 60, 30, 350, 24 if i == 1 else 10),
                ("Ordinary prose remains on this page.", 60, 200, 500, 12),
            ),
        )
        for i in range(1, 5)
    )
    document = _document(*pages)
    result = remove_furniture(document)
    assert [b.text for b in result.narration].count("The Quiet Book") == 1
    assert len(result.edits) == 3
    assert result.pages == document.pages
    assert remove_furniture(result) == result


def test_prose_unwrapped_but_indentation_and_dialogue_preserved() -> None:
    """A new paragraph and a new speaker survive compatible body fonts."""
    page = _page(
        1,
        (
            ("The first prose line continues into the", 60, 150, 500, 12),
            ("next line which finishes the thought.", 60, 166, 500, 12),
            ("An indented paragraph should remain separate.", 84, 182, 500, 12),
            ('"Another speaker starts here," she said.', 60, 198, 500, 12),
        ),
    )
    document = _document(page)
    result = reconstruct_paragraphs(document)
    assert len(result.narration) == 3
    assert result.narration[0].text == (
        "The first prose line continues into the next line which finishes the thought."
    )
    assert result.pages == document.pages
    assert reconstruct_paragraphs(result) == result


def test_monospaced_lines_and_sparse_pages_are_not_unwrapped() -> None:
    """Without supported prose geometry, retain the extracted boundaries."""
    page = _page(
        1,
        (
            ("A short page should be preserved.", 60, 150, 500, 12),
            ("Even if these two lines align.", 60, 166, 500, 12),
        ),
    )
    assert reconstruct_paragraphs(_document(page)) == _document(page)
    mono = replace(
        page, lines=tuple(replace(line, font="Courier") for line in page.lines)
    )
    assert reconstruct_paragraphs(_document(mono)) == _document(mono)


def test_header_folios_with_repeated_titles_are_removed() -> None:
    """Page numbers may be at the top, alone or attached to a running title."""
    pages = tuple(
        _page(
            i,
            (
                (str(i) if i % 2 else f"A Running Title {i}", 60, 30, 500, 10),
                ("The prose on this page is preserved.", 60, 150, 500, 12),
                ("There is more ordinary body text here.", 60, 166, 500, 12),
                ("The final line is also ordinary prose.", 60, 182, 500, 12),
            ),
        )
        for i in range(1, 11)
    )
    document = _document(*pages)
    result = remove_furniture(document)
    assert len(result.edits) == 10
    assert all("prose" in b.text or "body text" in b.text for b in result.narration)
    assert remove_furniture(result) == result


def test_repeated_years_and_interior_numbers_are_retained() -> None:
    """Number-looking text must follow the page sequence at the page edge."""
    pages = tuple(
        _page(
            i,
            (
                ("2020", 60, 30, 100, 12),
                ("Body prose should remain unchanged.", 60, 150, 500, 12),
                (str(i), 60, 200, 100, 12),
            ),
        )
        for i in range(1, 7)
    )
    document = _document(*pages)
    assert remove_furniture(document) == document


def test_hyphenated_continuation_across_facing_pages() -> None:
    """A page break can split a word even when left and right margins differ."""
    from kenkui.pdf_processing import repair_line_break_words  # noqa: PLC0415

    left = _page(
        1,
        (
            ("The investment begins with sound research.", 60, 150, 500, 12),
            ("A second body line supplies margin evidence.", 60, 166, 500, 12),
            ("We examine the value of this invest-", 60, 680, 500, 12),
        ),
    )
    right = _page(
        2,
        (
            ("ment before proceeding further.", 80, 150, 520, 12),
            ("The next paragraph has a clear indentation.", 104, 190, 520, 12),
            ("This line continues the next paragraph.", 80, 206, 520, 12),
        ),
    )
    document = _document(left, right)
    result = repair_line_break_words(reconstruct_paragraphs(document))
    assert any(
        block.text
        == "We examine the value of this investment before proceeding further."
        for block in result.narration
    )
    assert result.pages == document.pages
