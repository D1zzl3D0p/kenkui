from __future__ import annotations

from pathlib import Path

from kenkui.models import (
    AppConfig,
    AttributionExecutionMode,
    JobConfig,
    NlpExecutionMode,
)


def test_nlp_execution_mode_defaults_local():
    cfg = AppConfig.from_dict({})
    assert cfg.nlp_execution_mode == NlpExecutionMode.LOCAL


def test_attribution_execution_mode_defaults_local():
    cfg = AppConfig.from_dict({})
    assert cfg.attribution_execution_mode == AttributionExecutionMode.LOCAL


def test_nlp_execution_mode_modal_from_dict():
    cfg = AppConfig.from_dict({"nlp_execution_mode": "modal"})
    assert cfg.nlp_execution_mode == NlpExecutionMode.MODAL


def test_attribution_execution_mode_litellm_from_dict():
    cfg = AppConfig.from_dict({"attribution_execution_mode": "litellm"})
    assert cfg.attribution_execution_mode == AttributionExecutionMode.LITELLM


def test_app_config_to_dict_includes_execution_modes():
    cfg = AppConfig.from_dict({
        "nlp_execution_mode": "modal",
        "attribution_execution_mode": "litellm",
    })
    d = cfg.to_dict()
    assert d["nlp_execution_mode"] == "modal"
    assert d["attribution_execution_mode"] == "litellm"


def test_job_config_per_job_execution_mode_overrides(tmp_path):
    job = JobConfig(
        ebook_path=tmp_path / "book.epub",
        job_nlp_execution_mode=NlpExecutionMode.MODAL,
        job_attribution_execution_mode=AttributionExecutionMode.LOCAL,
    )
    d = job.to_dict()
    assert d["job_nlp_execution_mode"] == "modal"
    assert d["job_attribution_execution_mode"] == "local"


def test_job_config_round_trips_execution_modes(tmp_path):
    job = JobConfig(
        ebook_path=tmp_path / "book.epub",
        job_nlp_execution_mode=NlpExecutionMode.MODAL,
    )
    restored = JobConfig.from_dict(job.to_dict())
    assert restored.job_nlp_execution_mode == NlpExecutionMode.MODAL
    assert restored.job_attribution_execution_mode is None


def test_job_config_execution_mode_defaults_none(tmp_path):
    job = JobConfig(ebook_path=tmp_path / "book.epub")
    assert job.job_nlp_execution_mode is None
    assert job.job_attribution_execution_mode is None
