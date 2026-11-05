"""离线可用的假模型，供测试和「没装模型也想跑一遍」时使用。

    from feilian.testing import FakeEmbeddings, FakeChatModel

它们不联网、不发请求，行为完全可复现，适合在 CI 里验证索引与问答链路。
"""

from __future__ import annotations

import hashlib
from typing import Any

from langchain_core.embeddings import Embeddings


class FakeEmbeddings(Embeddings):
    """把文本映射成固定维度的确定性向量，语义无关但足以跑通全链路。"""

    def __init__(self, dim: int = 16) -> None:
        self.dim = dim

    def _vector(self, text: str) -> list[float]:
        digest = hashlib.sha256(text.encode("utf-8")).digest()
        vector = [b / 255.0 for b in digest[: self.dim]]
        vector[0] += 0.001  # 避免零向量导致余弦相似度不可用
        return vector

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._vector(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._vector(text)


class _Message:
    """最小可用的消息对象，只带 content 字段。"""

    def __init__(self, content: str) -> None:
        self.content = content


class FakeChatModel:
    """记录收到的消息，返回固定答案。"""

    def __init__(self, reply: str = "这是根据资料 [1] 得到的答案。") -> None:
        self.reply = reply
        self.calls: list[list[Any]] = []

    def invoke(self, messages: list[Any]) -> _Message:
        self.calls.append(messages)
        return _Message(self.reply)
