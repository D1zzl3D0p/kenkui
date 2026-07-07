from __future__ import annotations

from pathlib import Path

from kenkui.models import AppConfig, CostStatus, JobConfig, JobStatus, QueueItem
from kenkui.services.queue_manager import QueueManager


def _item(job_id: str, status: JobStatus = JobStatus.PENDING) -> QueueItem:
    return QueueItem(
        id=job_id,
        job=JobConfig(ebook_path=Path("book.epub")),
        status=status,
        execution_provider="local",
    )


def _manager(tmp_path) -> QueueManager:
    return QueueManager(queue_file=tmp_path / "queue.toml", app_config=AppConfig())


def test_add_persists_and_reloads(tmp_path):
    qm = _manager(tmp_path)
    qm.add(_item("aaa"))
    assert qm.queue_file.exists()

    reloaded = _manager(tmp_path)
    assert [i.id for i in reloaded.all_items] == ["aaa"]


def test_remove_only_allows_terminal_states(tmp_path):
    qm = _manager(tmp_path)
    qm.add(_item("aaa", JobStatus.PROCESSING))
    assert qm.remove("aaa") is False
    qm.complete("aaa", "out.m4b")
    assert qm.remove("aaa") is True
    assert qm.all_items == []


def test_status_views_partition_items(tmp_path):
    qm = _manager(tmp_path)
    qm.add(_item("p", JobStatus.PENDING))
    qm.add(_item("c", JobStatus.COMPLETED))
    qm.add(_item("f", JobStatus.FAILED))
    assert [i.id for i in qm.pending_items] == ["p"]
    assert [i.id for i in qm.completed_items] == ["c"]
    assert [i.id for i in qm.failed_items] == ["f"]
    assert qm.next_pending().id == "p"


def test_current_item_prefers_processing_then_paused(tmp_path):
    qm = _manager(tmp_path)
    qm.add(_item("paused", JobStatus.PAUSED))
    assert qm.current_item.id == "paused"
    qm.add(_item("proc", JobStatus.PROCESSING))
    assert qm.current_item.id == "proc"
    assert qm.processing_item().id == "proc"


def test_reset_stale_processing_moves_to_pending(tmp_path):
    qm = _manager(tmp_path)
    qm.add(_item("stuck", JobStatus.PROCESSING))
    # Simulate a fresh boot loading a queue that has a PROCESSING item.
    reloaded = _manager(tmp_path)
    item = reloaded.get("stuck")
    assert item.status == JobStatus.PENDING
    assert item.progress == 0.0


def test_update_job_metadata_coerces_cost_status(tmp_path):
    qm = _manager(tmp_path)
    qm.add(_item("aaa"))
    qm.update_job_metadata("aaa", cost_status="final", remote_job_id="r1")
    item = qm.get("aaa")
    assert item.cost_status == CostStatus.FINAL
    assert item.remote_job_id == "r1"


def test_update_job_metadata_skips_none_values(tmp_path):
    qm = _manager(tmp_path)
    qm.add(_item("aaa"))
    qm.update_job_metadata("aaa", remote_job_id="keep")
    qm.update_job_metadata("aaa", remote_job_id=None)
    assert qm.get("aaa").remote_job_id == "keep"


def test_complete_and_fail_transitions(tmp_path):
    qm = _manager(tmp_path)
    qm.add(_item("done"))
    qm.add(_item("bad"))
    qm.complete("done", "final.m4b")
    qm.fail("bad", "boom")
    assert qm.get("done").status == JobStatus.COMPLETED
    assert qm.get("done").progress == 100.0
    assert qm.get("bad").status == JobStatus.FAILED
    assert qm.get("bad").error_message == "boom"


def test_clear_all_returns_removed_count(tmp_path):
    qm = _manager(tmp_path)
    qm.add(_item("a"))
    qm.add(_item("b"))
    assert qm.clear_all() == 2
    assert qm.all_items == []


def test_app_config_setter_persists(tmp_path):
    qm = _manager(tmp_path)
    cfg = AppConfig.from_dict({"nlp_provider": "ollama"})
    qm.app_config = cfg
    reloaded = _manager(tmp_path)
    assert reloaded.app_config.nlp_provider == "ollama"
