"""问答链：检索 + 带引用的生成 + 多轮记忆。

和原始 demo 的差别：
1. 答案必须带编号引用，末尾给出「资料来源」清单，模型说什么都能回查；
2. 资料里没有时明确回「查不到」，而不是顺着问题编；
3. 保留最近 N 轮对话，追问「那第二步呢」这种指代能接上。
"""

from __future__ import annotations

from dataclasses import dataclass, field

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage

from .config import Settings
from .index import RetrievedChunk, VectorIndex

SYSTEM_PROMPT = """你是「飞链」知识库助手。你会拿到若干段从用户本地文档里检索出来的资料，然后回答问题。

必须遵守的规则：
1. 只依据资料回答，不要使用资料之外的知识，也不要凭印象补全细节。
2. 每个结论后面用方括号标出它来自哪一段资料，例如 [1]、[2]。
3. 如果资料里没有答案，直接说「资料里没有找到相关内容」，并说明你查到了哪些相近的内容；严禁编造。
4. 如果资料之间互相矛盾，把冲突点指出来，不要自行裁决。
5. 用简体中文回答，先给结论，再给必要的细节，不要复述整个资料。
"""


@dataclass
class Source:
    """一条可回查的引用。"""

    index: int
    label: str
    snippet: str
    score: float | None = None

    def as_dict(self) -> dict:
        return {"index": self.index, "label": self.label, "score": self.score, "snippet": self.snippet}

    def render(self) -> str:
        score = f"（相关度 {self.score:.3f}）" if self.score is not None else ""
        return f"[{self.index}] {self.label}{score}\n    {self.snippet}"


@dataclass
class Answer:
    text: str
    sources: list[Source] = field(default_factory=list)
    found: bool = True

    def as_dict(self) -> dict:
        return {"answer": self.text, "found": self.found, "sources": [s.as_dict() for s in self.sources]}

    def render(self) -> str:
        if not self.found:
            return self.text
        lines = [self.text.strip(), "", "资料来源："]
        lines.extend(s.render() for s in self.sources)
        return "\n".join(lines)


def _snippet(text: str, limit: int = 120) -> str:
    flat = " ".join(text.split())
    return flat if len(flat) <= limit else flat[:limit] + "…"


class RAGChat:
    """把索引、模型和会话状态串起来的门面。"""

    def __init__(
        self,
        settings: Settings,
        index: VectorIndex | None = None,
        chat_model: BaseChatModel | None = None,
    ) -> None:
        self.settings = settings
        self.index = index or VectorIndex(settings)
        self._chat_model = chat_model
        self._history: list[BaseMessage] = []

    @property
    def chat_model(self) -> BaseChatModel:
        if self._chat_model is None:
            from .providers import build_chat_model

            self._chat_model = build_chat_model(self.settings)
        return self._chat_model

    # ---------- 会话状态 ----------

    def reset(self) -> None:
        self._history.clear()

    @property
    def history_turns(self) -> int:
        return len(self._history) // 2

    def _trim_history(self) -> None:
        limit = self.settings.history_turns * 2
        if limit <= 0:
            self._history.clear()
        elif len(self._history) > limit:
            self._history = self._history[-limit:]

    # ---------- 问答 ----------

    def _build_messages(self, question: str, chunks: list[RetrievedChunk]) -> list[BaseMessage]:
        context_blocks = []
        for i, chunk in enumerate(chunks, start=1):
            context_blocks.append(f"[{i}] 来源：{chunk.label()}\n{chunk.document.page_content.strip()}")
        context = "\n\n".join(context_blocks)

        messages: list[BaseMessage] = [SystemMessage(content=SYSTEM_PROMPT)]
        messages.extend(self._history)
        messages.append(
            HumanMessage(
                content=f"资料：\n{context}\n\n问题：{question}\n\n请依据上面的资料回答，并标注引用编号。"
            )
        )
        return messages

    def ask(self, question: str, k: int | None = None) -> Answer:
        question = question.strip()
        if not question:
            return Answer(text="请输入一个问题。", found=False)

        chunks = self.index.retrieve(question, k=k)
        if not chunks:
            return Answer(
                text="资料里没有找到相关内容。可以先把相关文档放进 docs/ 目录，然后执行 `feilian index`。",
                found=False,
            )

        response = self.chat_model.invoke(self._build_messages(question, chunks))
        text = response.content if isinstance(response.content, str) else str(response.content)

        sources = [
            Source(
                index=i,
                label=chunk.label(),
                snippet=_snippet(chunk.document.page_content),
                score=chunk.score,
            )
            for i, chunk in enumerate(chunks, start=1)
        ]

        self._history.extend([HumanMessage(content=question), AIMessage(content=text)])
        self._trim_history()

        return Answer(text=text.strip(), sources=sources, found=True)
