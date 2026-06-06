"""Public dataclasses and enums for kenkui library state.

The package is split by domain, while this module preserves the historical
``kenkui.models`` import path for applications and serialized cache loaders.
"""

from .audio import AudioResult, ProcessingConfig
from .book import BookInfo, ChapterSelection, CharacterInfo, CharacterRecord
from .common import (
    AttributionExecutionMode,
    AttributionTool,
    ChapterPreset,
    CostStatus,
    ExtractionTool,
    JobStatus,
    NarrationMode,
    NlpExecutionMode,
    TTSExecutionMode,
    _migrate_speaker_voices_keys,
    _normalize_bitrate,
)
from .config import AppConfig, PostProcessingConfig
from .job import JobConfig
from .nlp import Chapter, FastScanResult, NLPResult, Segment
from .queue import QueueItem

__all__ = [
    "AppConfig",
    "AttributionExecutionMode",
    "AttributionTool",
    "AudioResult",
    "BookInfo",
    "Chapter",
    "ChapterPreset",
    "ChapterSelection",
    "CharacterInfo",
    "CharacterRecord",
    "CostStatus",
    "ExtractionTool",
    "FastScanResult",
    "JobConfig",
    "JobStatus",
    "NarrationMode",
    "NLPResult",
    "NlpExecutionMode",
    "PostProcessingConfig",
    "ProcessingConfig",
    "QueueItem",
    "Segment",
    "TTSExecutionMode",
    "_migrate_speaker_voices_keys",
    "_normalize_bitrate",
]
