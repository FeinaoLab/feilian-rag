"""飞链 RAG：把本地文档链成一条可追溯的答案链。

    from feilian import RAGChat, Settings

    chat = RAGChat(Settings.load())
    print(chat.ask("这份文档讲了什么？").render())
"""

from __future__ import annotations

from .chain import Answer, RAGChat, Source
from .config import Settings
from .errors import ConfigError, FeilianError, ProviderError, StoreError
from .index import SyncResult, VectorIndex

__version__ = "0.1.0"

__all__ = [
    "Answer",
    "ConfigError",
    "FeilianError",
    "ProviderError",
    "RAGChat",
    "Settings",
    "Source",
    "StoreError",
    "SyncResult",
    "VectorIndex",
    "__version__",
]
