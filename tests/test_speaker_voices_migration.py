"""Tests for speaker_voices legacy-key migration in JobConfig.from_dict()."""

from __future__ import annotations

from kenkui.models import JobConfig, _migrate_speaker_voices_keys

# ---------------------------------------------------------------------------
# Unit: _migrate_speaker_voices_keys
# ---------------------------------------------------------------------------


def test_slug_keys_unchanged():
    d = {"elizabeth_bennet": "voice_a", "mr_darcy": "voice_b"}
    assert _migrate_speaker_voices_keys(d) == d


def test_preserved_special_keys_unchanged():
    d = {"NARRATOR": "alba", "Unknown": "alba", "SCENE_BREAK": ""}
    assert _migrate_speaker_voices_keys(d) == d


def test_canonical_name_converted_to_slug():
    result = _migrate_speaker_voices_keys({"Elizabeth Bennet": "voice_a"})
    assert result == {"elizabeth_bennet": "voice_a"}


def test_canonical_name_with_honorific_converted():
    result = _migrate_speaker_voices_keys({"Mr. Darcy": "voice_b"})
    assert result == {"mr_darcy": "voice_b"}


def test_name_with_apostrophe_converted():
    result = _migrate_speaker_voices_keys({"Rand al'Thor": "voice_c"})
    assert result == {"rand_althor": "voice_c"}


def test_mixed_dict_preserves_and_migrates():
    d = {
        "NARRATOR": "alba",
        "Elizabeth Bennet": "voice_a",
        "mr_darcy": "voice_b",
        "Mr. Wickham": "voice_c",
    }
    result = _migrate_speaker_voices_keys(d)
    assert result["NARRATOR"] == "alba"
    assert result["elizabeth_bennet"] == "voice_a"
    assert result["mr_darcy"] == "voice_b"
    assert result["mr_wickham"] == "voice_c"
    assert "Elizabeth Bennet" not in result
    assert "Mr. Wickham" not in result


# ---------------------------------------------------------------------------
# Integration: JobConfig.from_dict() migration
# ---------------------------------------------------------------------------


def _minimal_job_dict(speaker_voices: dict) -> dict:
    return {
        "ebook_path": "/tmp/book.epub",
        "speaker_voices": speaker_voices,
    }


def test_job_config_from_dict_migrates_legacy_keys():
    data = _minimal_job_dict({"Jane Eyre": "voice_a", "NARRATOR": "alba"})
    cfg = JobConfig.from_dict(data)
    assert "jane_eyre" in cfg.speaker_voices
    assert cfg.speaker_voices["jane_eyre"] == "voice_a"
    assert cfg.speaker_voices["NARRATOR"] == "alba"
    assert "Jane Eyre" not in cfg.speaker_voices


def test_job_config_from_dict_leaves_slug_keys_intact():
    data = _minimal_job_dict({"jane_eyre": "voice_a", "NARRATOR": "alba"})
    cfg = JobConfig.from_dict(data)
    assert cfg.speaker_voices == {"jane_eyre": "voice_a", "NARRATOR": "alba"}


def test_job_config_from_dict_empty_speaker_voices():
    data = _minimal_job_dict({})
    cfg = JobConfig.from_dict(data)
    assert cfg.speaker_voices == {}
