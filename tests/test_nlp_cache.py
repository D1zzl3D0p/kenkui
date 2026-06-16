"""Tests for kenkui.nlp._cache."""
from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import patch

import kenkui.nlp as legacy_nlp
from kenkui.models import Chapter, CharacterInfo, FastScanResult, NLPResult
from kenkui.nlp._cache import (
    CacheMeta,
    _cache_filename,
    _model_slug,
    clear_checkpoints,
    delete_cache,
    get_cache,
    get_chapter_checkpoint,
    list_caches,
    put_cache,
    put_chapter_checkpoint,
)
from kenkui.nlp.models import CharacterRecord, CharacterRoster

FAKE_BOOK = Path("/fake/book.epub")
FAKE_HASH = "abc123def456"


def _fake_hash(path):
    return FAKE_HASH


class TestModelSlug:
    def test_lowercase_passthrough(self):
        assert _model_slug("llama3") == "llama3"

    def test_dots_and_dashes_replaced(self):
        assert _model_slug("claude-3.5-sonnet") == "claude_3_5_sonnet"

    def test_empty_returns_default(self):
        assert _model_slug("") == "default"

    def test_uppercase_lowercased(self):
        assert _model_slug("GPT-4o") == "gpt_4o"


class TestCacheFilename:
    def test_filename_format(self):
        name = _cache_filename("abc123", "extraction", "ollama", "llama3.2")
        assert name == "abc123-extraction-ollama-llama3_2.json"

    def test_attribution_step(self):
        name = _cache_filename("abc123", "attribution", "litellm", "claude-3-5-sonnet")
        assert name == "abc123-attribution-litellm-claude_3_5_sonnet.json"


class TestPutAndGetCache:
    def test_roundtrip(self, tmp_path):
        with patch("kenkui.nlp._cache._get_cache_dir", return_value=tmp_path), \
             patch("kenkui.nlp._cache._book_hash_fn", _fake_hash, create=True):
            with patch("kenkui.nlp.book_hash", _fake_hash):
                payload = {"roster_data": {"characters": []}}
                path = put_cache(payload, FAKE_BOOK, "extraction", "ollama", "llama3.2", cache_dir=tmp_path)
                result = get_cache(FAKE_BOOK, "extraction", "ollama", "llama3.2", cache_dir=tmp_path)

        assert result is not None
        assert result["roster_data"] == {"characters": []}
        assert result["step"] == "extraction"
        assert result["tool"] == "ollama"
        assert result["model"] == "llama3.2"
        assert "created_at" in result

    def test_put_is_atomic(self, tmp_path):
        """Verify .tmp file is cleaned up (replaced, not left behind)."""
        with patch("kenkui.nlp.book_hash", _fake_hash):
            put_cache({}, FAKE_BOOK, "extraction", "ollama", "test", cache_dir=tmp_path)
        tmp_files = list(tmp_path.glob("*.tmp"))
        assert tmp_files == []

    def test_get_returns_none_when_missing(self, tmp_path):
        with patch("kenkui.nlp.book_hash", _fake_hash):
            result = get_cache(FAKE_BOOK, "extraction", "ollama", "llama3.2", cache_dir=tmp_path)
        assert result is None


class TestChapterCheckpoints:
    def test_roundtrip_verifies_chapter_fingerprint(self, tmp_path):
        chapter = Chapter(index=1, title="One", paragraphs=["Alice spoke."])
        changed = Chapter(index=1, title="One", paragraphs=["Alice shouted."])

        with patch("kenkui.nlp.book_hash", _fake_hash):
            put_chapter_checkpoint(
                {"value": 1},
                FAKE_BOOK,
                chapter,
                step="attribution",
                tool="openrouter",
                model="model/a",
                cache_dir=tmp_path,
            )
            cached = get_chapter_checkpoint(
                FAKE_BOOK,
                chapter,
                step="attribution",
                tool="openrouter",
                model="model/a",
                cache_dir=tmp_path,
            )
            stale = get_chapter_checkpoint(
                FAKE_BOOK,
                changed,
                step="attribution",
                tool="openrouter",
                model="model/a",
                cache_dir=tmp_path,
            )

        assert cached == {"value": 1}
        assert stale is None

    def test_clear_checkpoints_removes_matching_tree(self, tmp_path):
        chapter = Chapter(index=1, title="One", paragraphs=["Alice spoke."])

        with patch("kenkui.nlp.book_hash", _fake_hash):
            path = put_chapter_checkpoint(
                {"value": 1},
                FAKE_BOOK,
                chapter,
                step="attribution",
                tool="openrouter",
                model="model/a",
                cache_dir=tmp_path,
            )
            clear_checkpoints(
                FAKE_BOOK,
                step="attribution",
                tool="openrouter",
                model="model/a",
                cache_dir=tmp_path,
            )

        assert not path.exists()

    def test_get_returns_none_on_corrupt_json(self, tmp_path):
        with patch("kenkui.nlp.book_hash", _fake_hash):
            path = tmp_path / f"{FAKE_HASH}-extraction-ollama-llama3_2.json"
            path.write_text("not json", encoding="utf-8")
            result = get_cache(FAKE_BOOK, "extraction", "ollama", "llama3.2", cache_dir=tmp_path)
        assert result is None


class TestListCaches:
    def _write_entry(self, cache_dir, step, tool, model, created_at_str):
        name = _cache_filename(FAKE_HASH, step, tool, model)
        data = {
            "created_at": created_at_str,
            "step": step,
            "tool": tool,
            "model": model,
            "book_hash": FAKE_HASH,
        }
        (cache_dir / name).write_text(json.dumps(data), encoding="utf-8")

    def test_empty_dir_returns_empty(self, tmp_path):
        with patch("kenkui.nlp.book_hash", _fake_hash):
            assert list_caches(FAKE_BOOK, cache_dir=tmp_path) == []

    def test_returns_correct_meta(self, tmp_path):
        self._write_entry(tmp_path, "extraction", "ollama", "llama3.2", "2026-01-01T00:00:00+00:00")
        with patch("kenkui.nlp.book_hash", _fake_hash):
            metas = list_caches(FAKE_BOOK, cache_dir=tmp_path)
        assert len(metas) == 1
        m = metas[0]
        assert m.step == "extraction"
        assert m.tool == "ollama"
        assert m.book_hash == FAKE_HASH

    def test_sorted_newest_first(self, tmp_path):
        self._write_entry(tmp_path, "extraction", "ollama", "llama3.2", "2026-01-01T00:00:00+00:00")
        self._write_entry(tmp_path, "attribution", "ollama", "llama3.2", "2026-06-01T00:00:00+00:00")
        with patch("kenkui.nlp.book_hash", _fake_hash):
            metas = list_caches(FAKE_BOOK, cache_dir=tmp_path)
        assert metas[0].step == "attribution"  # newer
        assert metas[1].step == "extraction"   # older

    def test_step_filter(self, tmp_path):
        self._write_entry(tmp_path, "extraction", "ollama", "llama3.2", "2026-01-01T00:00:00+00:00")
        self._write_entry(tmp_path, "attribution", "ollama", "llama3.2", "2026-01-02T00:00:00+00:00")
        with patch("kenkui.nlp.book_hash", _fake_hash):
            metas = list_caches(FAKE_BOOK, step="extraction", cache_dir=tmp_path)
        assert len(metas) == 1
        assert metas[0].step == "extraction"


class TestDeleteCache:
    def test_deletes_file(self, tmp_path):
        f = tmp_path / "test.json"
        f.write_text("{}", encoding="utf-8")
        meta = CacheMeta(
            path=f, step="extraction", tool="ollama", model="llama3.2",
            created_at=datetime.now(tz=UTC), description="", book_hash=FAKE_HASH,
        )
        delete_cache(meta)
        assert not f.exists()

    def test_delete_missing_file_is_safe(self, tmp_path):
        meta = CacheMeta(
            path=tmp_path / "nonexistent.json", step="extraction", tool="ollama",
            model="llama3.2", created_at=datetime.now(tz=UTC),
            description="", book_hash=FAKE_HASH,
        )
        delete_cache(meta)  # must not raise


class TestLegacyModelAwareCaches:
    def _book(self, tmp_path: Path) -> Path:
        book = tmp_path / "book.epub"
        book.write_bytes(b"fake")
        return book

    def _scan_result(self, book: Path) -> FastScanResult:
        roster = CharacterRoster(characters=[
            CharacterRecord(slug="alice", canonical_name="Alice", aliases=["Alice"])
        ])
        return FastScanResult(
            roster=roster,
            characters=[CharacterInfo(character_id="alice", display_name="Alice")],
            book_hash=legacy_nlp.book_hash(book),
        )

    def _nlp_result(self, book: Path) -> NLPResult:
        return NLPResult(
            characters=[CharacterInfo(character_id="alice", display_name="Alice")],
            chapters=[Chapter(index=0, title="One", paragraphs=[])],
            book_hash=legacy_nlp.book_hash(book),
        )

    def test_roster_caches_for_two_models_write_different_files(self, tmp_path):
        book = self._book(tmp_path)
        result = self._scan_result(book)
        with patch("kenkui.nlp.CONFIG_DIR", tmp_path):
            path_a = legacy_nlp.cache_roster(
                result, book, method="llm", provider="openrouter", model="anthropic/claude-3.5"
            )
            path_b = legacy_nlp.cache_roster(
                result, book, method="llm", provider="openrouter", model="openai/gpt-4o"
            )

        assert path_a != path_b
        assert "openrouter-anthropic_claude_3_5" in path_a.name
        assert "openrouter-openai_gpt_4o" in path_b.name

    def test_attribution_caches_for_two_models_write_different_files(self, tmp_path):
        book = self._book(tmp_path)
        result = self._nlp_result(book)
        with patch("kenkui.nlp.CONFIG_DIR", tmp_path):
            path_a = legacy_nlp.cache_result(
                result, book, provider="openrouter", model="anthropic/claude-3.5"
            )
            path_b = legacy_nlp.cache_result(
                result, book, provider="openrouter", model="openai/gpt-4o"
            )

        assert path_a != path_b
        assert "openrouter-anthropic_claude_3_5" in path_a.name
        assert "openrouter-openai_gpt_4o" in path_b.name

    def test_legacy_provider_only_roster_read_as_fallback(self, tmp_path):
        book = self._book(tmp_path)
        result = self._scan_result(book)
        with patch("kenkui.nlp.CONFIG_DIR", tmp_path):
            legacy_path = tmp_path / "nlp_cache" / legacy_nlp._roster_cache_name(
                book, "llm", "openrouter"
            )
            legacy_path.parent.mkdir(parents=True)
            legacy_path.write_text(
                json.dumps({
                    "created_at": "2026-01-01T00:00:00",
                    "method": "llm",
                    "provider": "openrouter",
                    "model": "",
                    "roster_data": result.to_dict(),
                }),
                encoding="utf-8",
            )
            cached = legacy_nlp.get_cached_roster(
                book, method="llm", provider="openrouter", model="new/model"
            )

        assert cached is not None
        assert cached.roster.characters[0].slug == "alice"

    def test_exact_model_roster_cache_wins_over_legacy_fallback(self, tmp_path):
        book = self._book(tmp_path)
        old = self._scan_result(book)
        new_roster = CharacterRoster(characters=[
            CharacterRecord(slug="bob", canonical_name="Bob", aliases=["Bob"])
        ])
        new = FastScanResult(
            roster=new_roster,
            characters=[CharacterInfo(character_id="bob", display_name="Bob")],
            book_hash=legacy_nlp.book_hash(book),
        )
        with patch("kenkui.nlp.CONFIG_DIR", tmp_path):
            legacy_nlp.cache_roster(old, book, method="llm", provider="openrouter")
            legacy_nlp.cache_roster(new, book, method="llm", provider="openrouter", model="new/model")
            cached = legacy_nlp.get_cached_roster(
                book, method="llm", provider="openrouter", model="new/model"
            )

        assert cached is not None
        assert cached.roster.characters[0].slug == "bob"

    def test_legacy_provider_only_attribution_read_as_fallback(self, tmp_path):
        book = self._book(tmp_path)
        result = self._nlp_result(book)
        with patch("kenkui.nlp.CONFIG_DIR", tmp_path):
            legacy_path = legacy_nlp.cache_result(result, book, provider="openrouter")
            cached = legacy_nlp.get_cached_result(
                book, provider="openrouter", model="new/model"
            )

        assert cached is not None
        assert legacy_path.name.endswith("-openrouter.json")
        assert cached.characters[0].character_id == "alice"
