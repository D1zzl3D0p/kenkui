from __future__ import annotations

import modal  # type: ignore[import]

nlp_image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "spacy>=3.0.0",
        "booknlp>=1.0.8",
        "litellm>=1.0.0",
        "instructor>=1.0.0",
        "pydantic>=2.0.0",
        "boto3>=1.35.0",
    )
    .run_commands("python -m spacy download en_core_web_sm")
)

tts_image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("ffmpeg")
    .pip_install(
        "pocket-tts>=2.0.0",
        "pydub>=0.25.0",
        "mutagen>=1.45.0",
        "imageio-ffmpeg>=0.5.0",
        "pedalboard>=0.9",
        "noisereduce>=3.0",
        "ffmpeg-normalize>=1.26",
        "boto3>=1.35.0",
    )
)
