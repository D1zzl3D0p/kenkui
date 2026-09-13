"""The terms a catalog voice carries, derived from the dataset it came from.

A built-in voice's license and rights statement are properties of its source
recording's dataset, not of the individual voice: every VCTK voice shares
VCTK's terms. Recording them once per dataset keeps ninety-odd copies of the
same sentence from drifting apart, and makes an unrecognized source fall back
to conservative, unreviewed terms rather than to whatever was last pasted.

These are engineering records of what each source publishes, not legal
conclusions. Every entry is marked not cleared for commercial use.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final, Literal

Dataset = Literal[
    "vctk", "ears", "expresso", "common-voice", "kyutai-donation", "unreviewed"
]


@dataclass(frozen=True, slots=True)
class DatasetRights:
    """License, rights statement, and commercial default shared by a dataset."""

    license_id: str
    voice_rights: str
    commercial_use_allowed: bool = False


_NONCOMMERCIAL: Final = (
    "Treat as research-only or noncommercial unless your own review of the "
    "source terms concludes otherwise."
)

_RIGHTS: Final[dict[Dataset, DatasetRights]] = {
    "vctk": DatasetRights(
        "CC-BY-4.0",
        "Derived from the VCTK corpus via kyutai/tts-voices. Review the VCTK "
        "terms and speaker consent for your intended use before commercial "
        "deployment.",
    ),
    "ears": DatasetRights(
        "CC-BY-NC-4.0", f"Derived from the EARS corpus. {_NONCOMMERCIAL}"
    ),
    "expresso": DatasetRights(
        "CC-BY-NC-4.0", f"Derived from the Expresso dataset. {_NONCOMMERCIAL}"
    ),
    "common-voice": DatasetRights(
        "CC0-1.0",
        "Derived from Common Voice via kyutai/pocket-tts. Review the Common Voice "
        "terms for your intended use before commercial deployment.",
    ),
    "kyutai-donation": DatasetRights(
        "unreviewed",
        "Voice donation distributed by kyutai. Confirm the donor's permission "
        "scope for your intended use.",
    ),
    "unreviewed": DatasetRights(
        "unreviewed",
        "Source terms have not been reviewed. Confirm the recording's license "
        "and speaker consent for your intended use.",
    ),
}

# Path prefixes inside kyutai's repositories that identify a corpus. Anything
# else kyutai distributes is a voice donation.
_KYUTAI_CORPORA: Final[tuple[tuple[str, Dataset], ...]] = (
    ("hf://kyutai/tts-voices/vctk/", "vctk"),
    ("hf://kyutai/tts-voices/ears/", "ears"),
    ("hf://kyutai/tts-voices/expresso/", "expresso"),
    ("hf://kyutai/pocket-tts/common_voice_", "common-voice"),
)
_KYUTAI_REPOSITORIES: Final = ("hf://kyutai/tts-voices/", "hf://kyutai/pocket-tts/")

# The voice pack names its source corpus directly.
_PACK_DATASETS: Final[dict[str, Dataset]] = {"VCTK": "vctk", "EARS": "ears"}


def dataset_for_origin(origin_url: str) -> Dataset:
    """Return the dataset a built-in voice's source recording belongs to."""
    for prefix, dataset in _KYUTAI_CORPORA:
        if origin_url.startswith(prefix):
            return dataset
    if origin_url.startswith(_KYUTAI_REPOSITORIES):
        return "kyutai-donation"
    return "unreviewed"


def dataset_for_pack(name: str) -> Dataset:
    """Return the dataset a voice-pack record names, or ``"unreviewed"``."""
    return _PACK_DATASETS.get(name, "unreviewed")


def rights_for(dataset: Dataset) -> DatasetRights:
    """Return the terms every voice from ``dataset`` carries."""
    return _RIGHTS[dataset]
