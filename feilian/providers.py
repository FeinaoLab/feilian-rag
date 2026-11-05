"""模型工厂：按配置造 Chat 模型和 Embedding 模型。

要点：
1. 所有第三方 SDK 的 import 都放在函数内部，避免「只用 Ollama 却因为没装 openai 而启动失败」；
2. 造模型前先做一次轻量健康检查，报错信息说人话（例如「Ollama 没启动」而不是 ConnectionRefused）；
3. 测试可以直接传 fake 模型进来，不必联网。
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from typing import Any

from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseChatModel

from .config import Settings
from .errors import ProviderError


def _ollama_models(base_url: str, timeout: float = 3.0) -> list[str]:
    """返回 Ollama 已安装的模型名列表，连不上则抛 ProviderError。"""
    url = base_url.rstrip("/") + "/api/tags"
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:  # noqa: S310 - 本机地址
            payload = json.loads(resp.read().decode("utf-8"))
    except (urllib.error.URLError, OSError, json.JSONDecodeError) as exc:
        raise ProviderError(
            f"连不上 Ollama（{base_url}）。请先运行 `ollama serve`，或把 OLLAMA_BASE_URL 改成本机实际地址。"
        ) from exc
    return [m.get("name", "") for m in payload.get("models", [])]


def _match_model(wanted: str, available: list[str]) -> str | None:
    """Ollama 允许省略 :latest，这里做一次宽松匹配。"""
    if wanted in available:
        return wanted
    for name in available:
        if name.split(":")[0] == wanted.split(":")[0]:
            return name
    return None


def ensure_ollama_ready(settings: Settings) -> dict[str, Any]:
    """确认 Ollama 在跑，且对话模型和嵌入模型都已拉取。"""
    available = _ollama_models(settings.ollama_base_url)
    chat = _match_model(settings.chat_model, available)
    emb_model = settings.embedding_model
    if emb_model.startswith("text-embedding"):
        emb_model = "nomic-embed-text"
    emb = _match_model(emb_model, available)

    missing = []
    if chat is None:
        missing.append(f"对话模型 {settings.chat_model}（ollama pull {settings.chat_model}）")
    if emb is None:
        missing.append(f"嵌入模型 {emb_model}（ollama pull {emb_model}）")
    if missing:
        raise ProviderError("Ollama 里缺少模型：" + "；".join(missing))

    return {"provider": "ollama", "base_url": settings.ollama_base_url, "models": available}


def build_embeddings(settings: Settings) -> Embeddings:
    if settings.provider == "openai":
        try:
            from langchain_openai import OpenAIEmbeddings
        except ImportError as exc:  # pragma: no cover
            raise ProviderError("缺少 langchain-openai，请先 pip install -r requirements.txt") from exc
        kwargs: dict[str, Any] = {
            "model": settings.embedding_model,
            "api_key": settings.openai_api_key,
        }
        if settings.openai_base_url:
            kwargs["base_url"] = settings.openai_base_url
        return OpenAIEmbeddings(**kwargs)

    try:
        from langchain_ollama import OllamaEmbeddings
    except ImportError as exc:  # pragma: no cover
        raise ProviderError("缺少 langchain-ollama，请先 pip install -r requirements.txt") from exc

    model = settings.embedding_model
    if model.startswith("text-embedding"):
        model = "nomic-embed-text"
    return OllamaEmbeddings(model=model, base_url=settings.ollama_base_url)


def build_chat_model(settings: Settings) -> BaseChatModel:
    if settings.provider == "openai":
        try:
            from langchain_openai import ChatOpenAI
        except ImportError as exc:  # pragma: no cover
            raise ProviderError("缺少 langchain-openai，请先 pip install -r requirements.txt") from exc
        kwargs: dict[str, Any] = {
            "model": settings.chat_model,
            "api_key": settings.openai_api_key,
            "temperature": settings.temperature,
        }
        if settings.openai_base_url:
            kwargs["base_url"] = settings.openai_base_url
        return ChatOpenAI(**kwargs)

    try:
        from langchain_ollama import ChatOllama
    except ImportError as exc:  # pragma: no cover
        raise ProviderError("缺少 langchain-ollama，请先 pip install -r requirements.txt") from exc
    return ChatOllama(
        model=settings.chat_model,
        base_url=settings.ollama_base_url,
        temperature=settings.temperature,
    )


def health_check(settings: Settings) -> dict[str, Any]:
    """给 `feilian doctor` 用的体检：不真的发一次推理请求，省时省钱。"""
    report: dict[str, Any] = {"provider": settings.provider, "ok": False, "detail": ""}
    if settings.provider == "ollama":
        info = ensure_ollama_ready(settings)
        report.update(ok=True, detail=f"Ollama 可用，已安装 {len(info['models'])} 个模型", models=info["models"])
        return report

    if not settings.openai_api_key:
        report["detail"] = "OPENAI_API_KEY 为空"
        return report
    report.update(ok=True, detail="OpenAI 配置就绪（未发起网络请求）")
    return report
