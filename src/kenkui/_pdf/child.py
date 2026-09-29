"""Private command-line entrypoint for a disposable extraction process."""

from __future__ import annotations

import json
import sys
from pathlib import Path

from kenkui.errors import ErrorCode, KenkuiError

from .docling import extract_layout
from .options import PdfLayoutOptions, PdfOptions
from .protocol import SCHEMA, write_result


def main() -> None:
    """Consume a caller-owned request and return only sanitized primitive records."""
    request_path, result_path = map(Path, sys.argv[1:])
    request = json.loads(request_path.read_text())
    layout = PdfLayoutOptions(**request["layout"])
    options = PdfOptions(
        mode="auto",
        language=request["language"],
        layout=layout,
        max_pages=request["max_pages"],
        max_characters=request["max_characters"],
    )

    def progress(stage: str) -> None:
        target = Path(request["progress"])
        temporary = target.with_suffix(".tmp")
        temporary.write_text(json.dumps(stage))
        temporary.replace(target)

    try:
        document = extract_layout(
            Path(request["path"]), request["source_hash"], options, progress
        )
        write_result(result_path, document, layout.max_output_bytes)
    except KenkuiError as error:
        result_path.write_text(
            json.dumps({"schema": SCHEMA, "error": error.code.value})
        )
    except FileNotFoundError:
        result_path.write_text(
            json.dumps({"schema": SCHEMA, "error": ErrorCode.PDF_ASSETS_MISSING.value})
        )
    except Exception:  # noqa: BLE001 - no paths, source text or backend traces cross IPC
        result_path.write_text(
            json.dumps({"schema": SCHEMA, "error": ErrorCode.PDF_WORKER_FAILED.value})
        )


if __name__ == "__main__":
    main()
