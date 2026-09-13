"""Manage voices, including one you recorded yourself.

A voice is ``registered`` (known, not on disk), ``loaded`` (downloaded and
hash-pinned), or ``missing`` (the manifest says loaded, the file is gone).
Engines -- one per language, about 225 MB -- are downloaded when a voice
needs one and pruned when no loaded voice does.

    python examples/09_voices.py                   # list and load
    python examples/09_voices.py narrator.wav      # also register your own
"""

from __future__ import annotations

import sys
from pathlib import Path

import kenkui as kk


def show() -> None:
    """Print loaded voices and the disk their engines use."""
    loaded = [v for v in kk.list_voices() if v.state == "loaded"]
    for voice in loaded:
        print(
            f"  {voice.id:<16} {voice.language:<10} {voice.variety:<13} "
            f"commercial={voice.commercial_use_allowed}"
        )
    engines = {v.engine for v in loaded if v.engine is not None}
    megabytes = sum(e.size_bytes for e in engines) / 1_000_000
    print(f"  {len(engines)} engine(s), {megabytes:.0f} MB")


def main(recording: Path | None) -> None:
    """Load three feminine and three masculine English voices, then show them."""
    # There are no bulk verbs: compose over list_voices(). Each voice is about
    # 6.5 MB, so load a pool sized to your casts rather than the whole catalog.
    english = [v for v in kk.list_voices() if v.language == "english"]
    for gender in ("feminine", "masculine"):
        for voice in [v for v in english if v.perceived_gender == gender][:3]:
            kk.load_voice(voice.id)
    show()

    if recording is not None:
        # Kenkui infers no rights metadata: every field is required. Compiling
        # a .wav needs the gated kyutai/pocket-tts weights (accept the terms on
        # Hugging Face and log in). A .safetensors embedding needs no model.
        kk.add_voice(
            recording,
            voice_id="house-narrator",
            name="House Narrator",
            language="english",
            provenance="recorded 2026-09-01 with documented consent",
            license_id="proprietary",
            commercial_use_allowed=True,
            voice_rights="owned outright",
        )
        kk.load_voice("house-narrator")
        show()

    # unload_voice() frees disk but keeps the entry and its rights metadata;
    # remove_voice() forgets a voice you added entirely.
    #   kk.unload_voice("house-narrator")
    #   kk.remove_voice("house-narrator")

    # Attribution and casts are stored so re-rendering costs no model calls.
    # remove_casting() is free to rebuild; remove_attribution() is not.
    print(f"{len(kk.list_castings())} stored cast(s), {len(kk.list_series())} series")


if __name__ == "__main__":
    match sys.argv[1:]:
        case []:
            main(None)
        case [path]:
            main(Path(path))
        case _:
            raise SystemExit(__doc__)
