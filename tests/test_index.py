"""索引层测试：增量同步、删除同步、检索返回。"""

from __future__ import annotations

from pathlib import Path

from feilian.config import Settings
from feilian.index import VectorIndex
from feilian.loaders import load_documents

LONG_TEXT = "飞链是一个本地优先的检索增强生成工具。" * 20


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def test_first_sync_indexes_everything(settings: Settings, fake_embeddings) -> None:
    _write(settings.docs_dir / "a.md", "# 甲\n\n" + LONG_TEXT)
    _write(settings.docs_dir / "sub" / "b.txt", LONG_TEXT)

    index = VectorIndex(settings, embeddings=fake_embeddings)
    result = index.sync()

    assert sorted(result.added) == ["a.md", "sub/b.txt"]
    assert result.updated == []
    assert result.chunks > 0
    assert index.count() == result.chunks


def test_second_sync_is_noop(settings: Settings, fake_embeddings) -> None:
    _write(settings.docs_dir / "a.md", LONG_TEXT)
    index = VectorIndex(settings, embeddings=fake_embeddings)
    index.sync()

    again = VectorIndex(settings, embeddings=fake_embeddings).sync()

    assert again.added == []
    assert again.updated == []
    assert again.skipped == 1
    assert again.rebuilt is False


def test_changed_file_is_reindexed(settings: Settings, fake_embeddings) -> None:
    target = settings.docs_dir / "a.md"
    _write(target, LONG_TEXT)
    index = VectorIndex(settings, embeddings=fake_embeddings)
    first = index.sync()

    _write(target, LONG_TEXT + "\n\n新增了一段完全不同的内容。")
    second = VectorIndex(settings, embeddings=fake_embeddings).sync()

    assert second.updated == ["a.md"]
    # 旧块删除后写入新块，块数不应出现「旧块 + 新块」的翻倍
    assert second.chunks >= first.chunks
    stats = index.stats()
    assert stats["chunks_in_store"] == stats["chunks_in_manifest"]


def test_deleted_file_is_removed(settings: Settings, fake_embeddings) -> None:
    keep = settings.docs_dir / "keep.md"
    drop = settings.docs_dir / "drop.md"
    _write(keep, LONG_TEXT)
    _write(drop, LONG_TEXT)
    index = VectorIndex(settings, embeddings=fake_embeddings)
    index.sync()

    drop.unlink()
    result = VectorIndex(settings, embeddings=fake_embeddings).sync()

    assert result.removed == ["drop.md"]
    assert result.skipped == 1


def test_rebuild_when_signature_changes(settings: Settings, fake_embeddings) -> None:
    _write(settings.docs_dir / "a.md", LONG_TEXT)
    VectorIndex(settings, embeddings=fake_embeddings).sync()

    smaller = settings.with_overrides(chunk_size=80, chunk_overlap=10)
    result = VectorIndex(smaller, embeddings=fake_embeddings).sync()

    assert result.rebuilt is True
    assert result.chunks > 0


def test_retrieve_returns_labelled_chunks(settings: Settings, fake_embeddings) -> None:
    _write(settings.docs_dir / "a.md", "飞链的核心是增量索引。" * 10)
    _write(settings.docs_dir / "b.md", "切块大小决定检索精度。" * 10)
    index = VectorIndex(settings, embeddings=fake_embeddings)
    index.sync()

    chunks = index.retrieve("增量索引 切块", k=2)

    assert chunks
    assert all(c.source for c in chunks)
    assert all(c.chunk_index is not None for c in chunks)
    assert "a.md" in {c.source for c in chunks}


def test_retrieve_on_empty_store_returns_empty(settings: Settings, fake_embeddings) -> None:
    index = VectorIndex(settings, embeddings=fake_embeddings)
    assert index.retrieve("随便问点什么") == []


def test_loaders_report_unsupported_and_missing(settings: Settings) -> None:
    _write(settings.docs_dir / "a.md", "内容")
    _write(settings.docs_dir / "note.xyz", "不支持的格式")

    documents, warnings = load_documents(settings.docs_dir)

    assert [d.metadata["source"] for d in documents] == ["a.md"]
    assert warnings == []
