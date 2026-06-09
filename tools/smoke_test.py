#!/usr/bin/env python3
"""Runtime smoke test: verify voice catalog discovery and preview pipeline.

Usage:
    python tools/smoke_test.py

Requires pocket-tts to be installed. Tests:
  1. List all voices from the catalog (builtin + compiled if downloaded)
  2. Generate a preview WAV for the builtin voice "alba"
  3. If compiled voices are available (downloaded), generate a preview for the
     first available compiled voice.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from kenkui.services.voice_service import list_voices, prepare_voice_preview
from kenkui.voice_registry import get_catalog


def main() -> None:
    voices = list_voices()
    print(f"Catalog: {len(voices)} voice(s)")
    for v in voices:
        print(f"  {v.voice_id:<42} {v.origin:<20} {v.status}")

    print("\nTesting builtin voice 'alba'...")
    result = prepare_voice_preview("alba", force=True)
    out = Path(result.audio_path)
    assert out.exists() and out.stat().st_size > 0, f"Preview missing or empty: {out}"
    print(f"  OK: {out} ({out.stat().st_size:,} bytes)")

    compiled = get_catalog().filter(origin="kenkui_compiled", status="available")
    if not compiled:
        print("\nNo compiled voices downloaded; skipping compiled voice test.")
        print('Run: python -c "from kenkui.voice_download import download_voices; download_voices()"')
    else:
        v = compiled[0]
        print(f"\nTesting compiled voice {v.voice_id!r}...")
        result = prepare_voice_preview(v.voice_id, force=True)
        out = Path(result.audio_path)
        assert out.exists() and out.stat().st_size > 0, f"Preview missing or empty: {out}"
        print(f"  OK: {out} ({out.stat().st_size:,} bytes)")

    print("\nSmoke test passed.")


if __name__ == "__main__":
    main()
