"""测试公共件：只提供离线的假模型和临时配置。"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")

from feilian.config import Settings  # noqa: E402
from feilian.testing import FakeEmbeddings  # noqa: E402


@pytest.fixture()
def settings(tmp_path: Path) -> Settings:
    docs = tmp_path / "docs"
    docs.mkdir()
    return Settings(
        provider="ollama",
        chat_model="fake",
        embedding_model="fake-embed",
        docs_dir=docs,
        store_dir=tmp_path / ".feilian",
        collection="test_collection",
        chunk_size=200,
        chunk_overlap=30,
        top_k=3,
        fetch_k=6,
        score_threshold=None,
        search_type="similarity",
        history_turns=2,
    )


@pytest.fixture()
def fake_embeddings() -> FakeEmbeddings:
    return FakeEmbeddings()
