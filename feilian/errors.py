"""飞链 RAG 的异常定义。

所有对外抛出的异常都从这里派生，方便 CLI 统一捕获并给出人话提示，
而不是把 requests / urllib 的原始堆栈甩给用户。
"""

from __future__ import annotations


class FeilianError(Exception):
    """飞链 RAG 的基类异常。"""


class ConfigError(FeilianError):
    """配置缺失或非法。"""


class ProviderError(FeilianError):
    """模型侧不可用：服务没启动、密钥无效、模型未拉取等。"""


class StoreError(FeilianError):
    """向量库读写异常。"""


class DocumentError(FeilianError):
    """文档读取异常。"""
