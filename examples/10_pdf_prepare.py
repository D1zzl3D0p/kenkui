"""Prepare embedded PDF text and inspect the cleanup audit without running TTS.

Install kenkui[pdf] first. The native recipe removes repeated furniture and
corroborated footnotes, reconstructs paragraphs and repairs source-supported word
breaks. For optional layout/OCR, install kenkui[pdf-layout], provision assets,
and pass their directory as the second argument.

    python examples/10_pdf_prepare.py path/to/book.pdf [path/to/models]
"""

from __future__ import annotations

import sys
from pathlib import Path

import kenkui as kk
from kenkui.pdf_processing import PdfLayoutOptions


def main(pdf: Path, models: str | None = None) -> None:
    """Print prepared prose and a concise summary of source evidence and edits."""
    prepared = (
        kk.book(pdf)
        .pdf_processing(
            mode="auto" if models else "native",
            layout=PdfLayoutOptions(artifacts_path=models),
        )
        .prepare()
    )
    report = prepared.pdf_report()
    print(f"Pages: {len(report.pages)}; cleanup edits: {len(report.edits)}")
    for issue in report.issues:
        print(f"{issue.severity}: {issue.code}: {issue.message}")
    for chapter in prepared.inspect().chapters:
        print(f"\n{chapter.id}: {chapter.speech_characters} characters\n")
        print(chapter.text)
    # Continue with a provisioned narrator when satisfied:
    # prepared.assign_voice("eponine").tts().write("book.m4b")


if __name__ == "__main__":
    match sys.argv[1:]:
        case [path]:
            main(Path(path))
        case [path, models]:
            main(Path(path), models)
        case _:
            raise SystemExit(__doc__)
