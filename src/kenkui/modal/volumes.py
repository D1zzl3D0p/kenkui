from __future__ import annotations

import modal  # type: ignore[import]

llm_weights_volume = modal.Volume.from_name("kenkui-llm-weights", create_if_missing=True)
pocket_tts_volume = modal.Volume.from_name("kenkui-pocket-tts", create_if_missing=True)
voice_registry_volume = modal.Volume.from_name("kenkui-voice-registry", create_if_missing=True)

VOLUME_MOUNTS = {
    "/weights/llm": llm_weights_volume,
    "/weights/pocket-tts": pocket_tts_volume,
    "/weights/voices": voice_registry_volume,
}
