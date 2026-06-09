"""Helpers for compiling Pocket TTS voice assets into loadable state files."""

from __future__ import annotations

import shutil
from functools import lru_cache
from pathlib import Path

import safetensors

DEFAULT_VOICE_PACK_LANGUAGE = "english_2026-04"


@lru_cache(maxsize=4)
def _load_model(language: str = DEFAULT_VOICE_PACK_LANGUAGE):
    from pocket_tts import TTSModel

    return TTSModel.load_model(language=language)


def is_legacy_audio_prompt_asset(path: Path) -> bool:
    """Return True when *path* stores the legacy flat ``audio_prompt`` tensor."""
    try:
        with safetensors.safe_open(path, framework="pt") as f:
            keys = list(f.keys())
    except Exception:
        return False
    return keys == ["audio_prompt"] or ("audio_prompt" in keys and all("/" not in k for k in keys))


def compile_audio_prompt_source(
    source: str | Path,
    output_path: Path,
    *,
    language: str = DEFAULT_VOICE_PACK_LANGUAGE,
    truncate: bool = True,
) -> Path:
    """Compile an audio source to a modern Pocket TTS voice-state safetensors file."""
    from pocket_tts.models.tts_model import export_model_state

    output_path.parent.mkdir(parents=True, exist_ok=True)
    model = _load_model(language)
    model_state = model.get_state_for_audio_prompt(source, truncate=truncate)
    export_model_state(model_state, output_path)
    return output_path


def migrate_voice_asset(
    source_path: Path,
    output_path: Path,
    *,
    language: str = DEFAULT_VOICE_PACK_LANGUAGE,
) -> Path:
    """Copy a current voice-state asset or convert a legacy prompt tensor."""
    if source_path.resolve() == output_path.resolve():
        return output_path

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if is_legacy_audio_prompt_asset(source_path):
        from pocket_tts.models.tts_model import export_model_state, init_states

        with safetensors.safe_open(source_path, framework="pt") as f:
            prompt = f.get_tensor("audio_prompt")

        model = _load_model(language)
        state = init_states(model.flow_lm, batch_size=1, sequence_length=prompt.shape[1])
        model._run_flow_lm_and_increment_step(model_state=state, audio_conditioning=prompt)
        export_model_state(state, output_path)
        return output_path

    shutil.copy2(source_path, output_path)
    return output_path
