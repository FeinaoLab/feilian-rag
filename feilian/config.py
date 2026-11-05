"""飞链 RAG · 配置层。

设计原则：
1. 所有可调参数集中在 Settings 一个不可变对象里，方便测试时用 replace() 覆盖；
2. 只从环境变量 / .env 读一次，读不到就用安全的默认值；
3. 校验只做「必须现在就报错」的检查，其余留到真正调用时再说。
"""

from __future__ import annotations

import os
from dataclasses import dataclass, replace
from pathlib import Path

from dotenv import load_dotenv

from .errors import ConfigError

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ENV_FILE = PROJECT_ROOT / ".env"

load_dotenv(ENV_FILE)


def _env(key: str, default: str = "") -> str:
    value = os.getenv(key)
    if value is None:
        return default
    return value.strip()


def _env_int(key: str, default: int) -> int:
    raw = _env(key)
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError as exc:  # pragma: no cover - 只在手改 .env 出错时触发
        raise ConfigError(f"环境变量 {key} 需要是整数，当前是 {raw!r}") from exc


def _env_float(key: str, default: float) -> float:
    raw = _env(key)
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError as exc:  # pragma: no cover
        raise ConfigError(f"环境变量 {key} 需要是数字，当前是 {raw!r}") from exc


def _env_optional_float(key: str) -> float | None:
    raw = _env(key)
    if not raw or raw.lower() in {"none", "null", "off"}:
        return None
    try:
        return float(raw)
    except ValueError as exc:  # pragma: no cover
        raise ConfigError(f"环境变量 {key} 需要是数字或留空，当前是 {raw!r}") from exc


@dataclass(frozen=True)
class Settings:
    """一次运行所需的全部配置。"""

    provider: str = "ollama"
    chat_model: str = "llama3.2"
    embedding_model: str = "nomic-embed-text"
    temperature: float = 0.0

    openai_api_key: str = ""
    openai_base_url: str = ""
    ollama_base_url: str = "http://localhost:11434"

    docs_dir: Path = PROJECT_ROOT / "docs"
    store_dir: Path = PROJECT_ROOT / ".feilian"
    collection: str = "feilian"

    chunk_size: int = 700
    chunk_overlap: int = 120

    top_k: int = 4
    fetch_k: int = 20
    score_threshold: float | None = 0.2
    search_type: str = "similarity"

    history_turns: int = 6

    @classmethod
    def load(cls) -> "Settings":
        """从 .env / 环境变量装配配置。"""
        docs_dir = Path(_env("DOCS_DIR") or "docs")
        if not docs_dir.is_absolute():
            docs_dir = PROJECT_ROOT / docs_dir

        store_dir = Path(_env("STORE_DIR") or ".feilian")
        if not store_dir.is_absolute():
            store_dir = PROJECT_ROOT / store_dir

        settings = cls(
            provider=_env("LLM_PROVIDER", "ollama").lower(),
            chat_model=_env("OPENAI_MODEL" if _env("LLM_PROVIDER", "ollama").lower() == "openai" else "OLLAMA_MODEL")
            or ("gpt-4o-mini" if _env("LLM_PROVIDER", "ollama").lower() == "openai" else "llama3.2"),
            embedding_model=_env("EMBEDDING_MODEL", "nomic-embed-text"),
            temperature=_env_float("TEMPERATURE", 0.0),
            openai_api_key=_env("OPENAI_API_KEY"),
            openai_base_url=_env("OPENAI_BASE_URL"),
            ollama_base_url=_env("OLLAMA_BASE_URL", "http://localhost:11434"),
            docs_dir=docs_dir,
            store_dir=store_dir,
            collection=_env("COLLECTION", "feilian"),
            chunk_size=_env_int("CHUNK_SIZE", 700),
            chunk_overlap=_env_int("CHUNK_OVERLAP", 120),
            top_k=_env_int("TOP_K", 4),
            fetch_k=_env_int("FETCH_K", 20),
            score_threshold=_env_optional_float("SCORE_THRESHOLD"),
            search_type=_env("SEARCH_TYPE", "similarity").lower(),
            history_turns=_env_int("HISTORY_TURNS", 6),
        )
        settings.validate()
        return settings

    def validate(self) -> None:
        if self.provider not in {"openai", "ollama"}:
            raise ConfigError(f"LLM_PROVIDER 只能是 openai 或 ollama，当前是 {self.provider!r}")
        if self.provider == "openai" and not self.openai_api_key:
            raise ConfigError("使用 openai 时必须提供 OPENAI_API_KEY，请检查 .env")
        if self.search_type not in {"similarity", "mmr"}:
            raise ConfigError(f"SEARCH_TYPE 只能是 similarity 或 mmr，当前是 {self.search_type!r}")
        if self.chunk_overlap >= self.chunk_size:
            raise ConfigError("CHUNK_OVERLAP 必须小于 CHUNK_SIZE")
        if self.top_k <= 0:
            raise ConfigError("TOP_K 必须大于 0")

    def with_overrides(self, **kwargs) -> "Settings":
        """返回一份覆盖了部分字段的新配置（原地不可变）。"""
        clean = {k: v for k, v in kwargs.items() if v is not None}
        new = replace(self, **clean)
        new.validate()
        return new

    @property
    def manifest_path(self) -> Path:
        return self.store_dir / "manifest.json"

    @property
    def chroma_dir(self) -> Path:
        return self.store_dir / "chroma"

    def describe(self) -> str:
        emb_hint = self.embedding_model
        if self.provider == "ollama" and emb_hint.startswith("text-embedding"):
            emb_hint = "nomic-embed-text"
        return (
            f"provider={self.provider} chat={self.chat_model} embedding={emb_hint} "
            f"chunk={self.chunk_size}/{self.chunk_overlap} top_k={self.top_k} "
            f"search={self.search_type} threshold={self.score_threshold}"
        )
