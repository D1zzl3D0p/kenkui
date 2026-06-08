from __future__ import annotations

import ast
from pathlib import Path

import kenkui


def test_root_public_api_exports_core_entrypoints():
    expected = {
        "AppConfig",
        "ApostropheMode",
        "ProcessingConfig",
        "PostProcessingConfig",
        "load_config",
        "parse_book",
        "fast_scan",
        "full_analysis",
        "list_voices",
        "suggest_cast",
        "set_voice_pool_enabled",
        "prepare_voice_preview",
        "import_custom_voice",
        "compiled_voices_available",
        "get_huggingface_status",
        "ProviderCredentials",
        "book_hash",
        "list_cached_rosters",
        "get_cached_nlp_result",
        "cache_nlp_result",
        "cache_roster",
        "nlp_attribution_cache_path",
        "SeriesManifest",
        "slugify",
        "list_local_series",
        "load_local_series",
        "save_local_series",
        "list_roster_candidates",
        "match_characters",
        "authenticate_huggingface",
        "run_job",
        "ProgressEvent",
        "StageRecord",
        "load_stage_records",
    }

    assert expected <= set(kenkui.__all__)
    for name in expected:
        assert hasattr(kenkui, name)


def test_models_package_preserves_public_import_path():
    from kenkui.models import AppConfig, Chapter, JobConfig, Segment

    assert AppConfig().m4b_bitrate == "96k"
    assert Chapter(index=0, title="One", paragraphs=[]).title == "One"
    assert Segment("hello").speaker == "NARRATOR"
    assert JobConfig(ebook_path=Path("book.epub")).name == "book"


def test_core_source_has_no_builtin_print_or_input_calls():
    src_root = Path(__file__).resolve().parents[1] / "src" / "kenkui"
    offenders: list[str] = []

    for path in src_root.rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id in {"print", "input"}
            ):
                offenders.append(f"{path.relative_to(src_root)}:{node.lineno}:{node.func.id}")

    assert offenders == []
