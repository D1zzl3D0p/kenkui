from __future__ import annotations

import os

GPU_TIERS: list[tuple[str, str, int]] = [
    ("70b", "A100", 1),
    ("13b", "A10G", 1),
    ("8b",  "A10G", 1),
    ("7b",  "T4",   1),
    ("3b",  "T4",   1),
]

DEFAULT_GPU = "T4"


def resolve_gpu_tier(model_name: str) -> str:
    override = os.environ.get("KENKUI_MODAL_NLP_GPU_OVERRIDE", "").strip()
    if override:
        return override

    lower = model_name.lower()
    for pattern, tier, _count in GPU_TIERS:
        if pattern in lower:
            return tier

    return DEFAULT_GPU
