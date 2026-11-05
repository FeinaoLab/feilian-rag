"""向量索引：持久化 + 增量同步。

原始 demo 每次启动都 Chroma.from_documents 重建全量索引，文档一多就慢。
这里改成：
1. 向量库落盘到 .feilian/chroma，进程退出不丢；
2. manifest.json 记录每个文件的指纹（sha1）与它产生的 chunk id；
3. 同步时只处理「新增 / 内容变化 / 已删除」的文件，其余直接跳过；
4. 切块参数或嵌入模型变了，manifest 签名不匹配，自动整库重建，避免维度对不上。
"""

from __future__ import annotations

import hashlib
import json
import time
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from .config import Settings
from .errors import DocumentError, StoreError
from .loaders import iter_files, load_file

MANIFEST_VERSION = 3


@dataclass
class RetrievedChunk:
    """一条检索结果。"""

    document: Document
    score: float | None = None

    @property
    def source(self) -> str:
        return str(self.document.metadata.get("source", "unknown"))

    @property
    def page(self) -> int | None:
        page = self.document.metadata.get("page")
        return int(page) if isinstance(page, int) else None

    @property
    def chunk_index(self) -> int | None:
        idx = self.document.metadata.get("chunk_index")
        return int(idx) if isinstance(idx, int) else None

    def label(self) -> str:
        bits = [self.source]
        if self.page is not None:
            bits.append(f"第{self.page}页")
        if self.chunk_index is not None:
            bits.append(f"块{self.chunk_index}")
        return " · ".join(bits)


@dataclass
class SyncResult:
    added: list[str] = field(default_factory=list)
    updated: list[str] = field(default_factory=list)
    removed: list[str] = field(default_factory=list)
    skipped: int = 0
    chunks: int = 0
    rebuilt: bool = False
    warnings: list[str] = field(default_factory=list)

    @property
    def changed(self) -> bool:
        return bool(self.added or self.updated or self.removed or self.rebuilt)

    def summary(self) -> str:
        parts = [
            f"新增 {len(self.added)}",
            f"更新 {len(self.updated)}",
            f"删除 {len(self.removed)}",
            f"跳过 {self.skipped}",
            f"当前块数 {self.chunks}",
        ]
        if self.rebuilt:
            parts.insert(0, "已整库重建")
        return "｜".join(parts)


def _fingerprint(path: Path) -> str:
    digest = hashlib.sha1()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def _chunk_id(relative: str, index: int, fingerprint: str) -> str:
    return f"{relative}::{index}::{fingerprint[:8]}"


class VectorIndex:
    """包一层 Chroma，负责同步、统计与检索。"""

    def __init__(self, settings: Settings, embeddings: Embeddings | None = None) -> None:
        self.settings = settings
        self._embeddings = embeddings
        self._store = None

    # ---------- 基础组件 ----------

    @property
    def embeddings(self) -> Embeddings:
        if self._embeddings is None:
            from .providers import build_embeddings

            self._embeddings = build_embeddings(self.settings)
        return self._embeddings

    @property
    def store(self):
        if self._store is None:
            try:
                from langchain_chroma import Chroma
            except ImportError as exc:  # pragma: no cover
                raise StoreError("缺少 langchain-chroma，请执行 pip install -r requirements.txt") from exc

            self.settings.chroma_dir.mkdir(parents=True, exist_ok=True)
            self._store = Chroma(
                collection_name=self.settings.collection,
                embedding_function=self.embeddings,
                persist_directory=str(self.settings.chroma_dir),
            )
        return self._store

    @property
    def splitter(self) -> RecursiveCharacterTextSplitter:
        return RecursiveCharacterTextSplitter(
            chunk_size=self.settings.chunk_size,
            chunk_overlap=self.settings.chunk_overlap,
            separators=["\n\n", "\n", "。", "！", "？", "；", " ", ""],
        )

    # ---------- manifest ----------

    def _signature(self) -> dict[str, Any]:
        """决定索引是否可复用的参数签名。"""
        return {
            "provider": self.settings.provider,
            "embedding_model": self.settings.embedding_model,
            "chunk_size": self.settings.chunk_size,
            "chunk_overlap": self.settings.chunk_overlap,
            "collection": self.settings.collection,
        }

    def _read_manifest(self) -> dict[str, Any]:
        path = self.settings.manifest_path
        if not path.exists():
            return {"version": MANIFEST_VERSION, "signature": self._signature(), "files": {}}
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return {"version": MANIFEST_VERSION, "signature": self._signature(), "files": {}}
        if data.get("version") != MANIFEST_VERSION:
            return {"version": MANIFEST_VERSION, "signature": self._signature(), "files": {}}
        data.setdefault("files", {})
        return data

    def _write_manifest(self, manifest: dict[str, Any]) -> None:
        self.settings.store_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            "version": MANIFEST_VERSION,
            "signature": self._signature(),
            "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "files": manifest.get("files", {}),
        }
        self.settings.manifest_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True),
            encoding="utf-8",
        )

    # ---------- 同步 ----------

    def sync(self, rebuild: bool = False) -> SyncResult:
        """把 docs 目录同步进向量库。"""
        result = SyncResult()
        manifest = self._read_manifest()

        if manifest.get("signature") != self._signature():
            rebuild = True
        if rebuild:
            manifest = {"version": MANIFEST_VERSION, "signature": self._signature(), "files": {}}
            self.reset()
            result.rebuilt = True

        files = iter_files(self.settings.docs_dir, exclude=[self.settings.store_dir])
        known: dict[str, dict] = dict(manifest.get("files", {}))
        seen: set[str] = set()

        for path in files:
            relative = path.relative_to(self.settings.docs_dir).as_posix()
            seen.add(relative)
            try:
                fingerprint = _fingerprint(path)
            except OSError as exc:
                result.warnings.append(f"读取失败，已跳过：{relative}（{exc}）")
                continue

            previous = known.get(relative)
            if previous and previous.get("fingerprint") == fingerprint and not rebuild:
                result.skipped += 1
                continue

            if previous and previous.get("ids"):
                try:
                    self.store.delete(ids=list(previous["ids"]))
                except Exception as exc:  # pragma: no cover - 库损坏时兜底
                    result.warnings.append(f"旧块清理失败：{relative}（{exc}）")

            try:
                single = load_file(path, self.settings.docs_dir)
            except DocumentError as exc:
                result.warnings.append(str(exc))
                continue
            except Exception as exc:  # pragma: no cover
                result.warnings.append(f"解析失败：{relative}（{exc}）")
                continue
            if not single:
                result.warnings.append(f"没有可用的文本内容，已跳过：{relative}")
                known.pop(relative, None)
                continue

            chunks = self.splitter.split_documents(single)
            ids = []
            for index, chunk in enumerate(chunks):
                chunk.metadata["source"] = relative
                chunk.metadata["chunk_index"] = index
                chunk.metadata["fingerprint"] = fingerprint[:8]
                ids.append(_chunk_id(relative, index, fingerprint))

            try:
                self.store.add_documents(documents=chunks, ids=ids)
            except Exception as exc:
                raise StoreError(f"写入向量库失败（{relative}）：{exc}") from exc

            known[relative] = {"fingerprint": fingerprint, "chunks": len(chunks), "ids": ids}
            if previous:
                result.updated.append(relative)
            else:
                result.added.append(relative)

        for stale in sorted(set(known) - seen):
            ids = known[stale].get("ids", [])
            if ids:
                try:
                    self.store.delete(ids=list(ids))
                except Exception as exc:  # pragma: no cover
                    result.warnings.append(f"删除旧文件块失败：{stale}（{exc}）")
            known.pop(stale, None)
            result.removed.append(stale)

        self._write_manifest({"files": known})
        result.chunks = self.count()
        return result

    def reset(self) -> None:
        """清空向量库（不动物理文件，只删集合内容）。"""
        try:
            self.store.reset_collection()
        except Exception:
            try:
                ids = self.store.get(include=[])["ids"]  # type: ignore[index]
                if ids:
                    self.store.delete(ids=ids)
            except Exception as exc:  # pragma: no cover
                raise StoreError(f"清空向量库失败：{exc}") from exc

    # ---------- 查询 ----------

    def count(self) -> int:
        try:
            return int(self.store._collection.count())  # noqa: SLF001 - 官方未暴露 count
        except Exception:
            try:
                return len(self.store.get(include=[])["ids"])  # type: ignore[index]
            except Exception:  # pragma: no cover
                return 0

    def stats(self) -> dict[str, Any]:
        manifest = self._read_manifest()
        files = manifest.get("files", {})
        known_ids = {i for meta in files.values() for i in meta.get("ids", [])}
        return {
            "files": len(files),
            "chunks_in_manifest": sum(int(meta.get("chunks", 0)) for meta in files.values()),
            "chunks_in_store": self.count(),
            "orphan_chunks": max(0, self.count() - len(known_ids)),
            "store_dir": str(self.settings.store_dir),
            "signature": manifest.get("signature", {}),
        }

    def retrieve(self, question: str, k: int | None = None) -> list[RetrievedChunk]:
        """检索相关块，按相关度从高到低；带阈值过滤。"""
        k = k or self.settings.top_k
        if self.count() == 0:
            return []

        if self.settings.search_type == "mmr":
            documents = self.store.max_marginal_relevance_search(
                question, k=k, fetch_k=max(self.settings.fetch_k, k * 4)
            )
            return [RetrievedChunk(document=d, score=None) for d in documents]

        threshold = self.settings.score_threshold
        try:
            with warnings.catch_warnings():
                # 某些嵌入（尤其是测试用的假向量）会算出 [0,1] 外的相关性，
                # 底层会刷 UserWarning，这里静音后统一裁剪，不让日志淹掉真正的告警。
                warnings.simplefilter("ignore", UserWarning)
                pairs = self.store.similarity_search_with_relevance_scores(question, k=k)
        except Exception:  # pragma: no cover - 不同后端评分实现差异
            documents = self.store.similarity_search(question, k=k)
            return [RetrievedChunk(document=d, score=None) for d in documents]

        results = [
            RetrievedChunk(document=doc, score=min(1.0, max(0.0, float(score))))
            for doc, score in pairs
        ]
        if threshold is not None:
            filtered = [c for c in results if c.score is None or c.score >= threshold]
            if filtered:
                results = filtered
        return results
