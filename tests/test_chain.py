"""问答层测试：引用、拒答、多轮记忆。"""

from __future__ import annotations

from pathlib import Path

from feilian.chain import RAGChat
from feilian.testing import FakeChatModel
from feilian.config import Settings
from feilian.index import VectorIndex

TEXT = "飞链的索引是增量的，第二次同步只会跳过未变化的文件。切块大小建议五百到八百字符。"


def _build(settings: Settings, fake_embeddings, reply: str | None = None) -> RAGChat:
    Path(settings.docs_dir / "kb.md").write_text(TEXT * 3, encoding="utf-8")
    index = VectorIndex(settings, embeddings=fake_embeddings)
    index.sync()
    return RAGChat(settings, index=index, chat_model=FakeChatModel() if reply is None else FakeChatModel(reply))


def test_answer_carries_sources(settings: Settings, fake_embeddings) -> None:
    chat = _build(settings, fake_embeddings)

    answer = chat.ask("索引是怎么更新的？")

    assert answer.found is True
    assert answer.sources
    assert answer.sources[0].label.startswith("kb.md")
    assert "资料来源" in answer.render()


def test_empty_index_answers_without_calling_model(settings: Settings, fake_embeddings) -> None:
    model = FakeChatModel()
    chat = RAGChat(settings, index=VectorIndex(settings, embeddings=fake_embeddings), chat_model=model)

    answer = chat.ask("没有资料时应该怎么办？")

    assert answer.found is False
    assert model.calls == []
    assert "没有找到" in answer.text


def test_history_is_trimmed(settings: Settings, fake_embeddings) -> None:
    chat = _build(settings, fake_embeddings)

    for _ in range(5):
        chat.ask("切块大小怎么定？")

    # history_turns=2，所以最多保留 4 条消息（2 轮）
    assert len(chat._history) <= settings.history_turns * 2  # noqa: SLF001
    assert chat.history_turns <= settings.history_turns


def test_reset_clears_history(settings: Settings, fake_embeddings) -> None:
    chat = _build(settings, fake_embeddings)
    chat.ask("增量索引是什么？")
    assert chat.history_turns == 1

    chat.reset()

    assert chat.history_turns == 0


def test_prompt_contains_context_and_question(settings: Settings, fake_embeddings) -> None:
    chat = _build(settings, fake_embeddings)
    chat.ask("切块建议多大？")

    messages = chat.chat_model.calls[0]
    human = messages[-1].content
    assert "资料" in human
    assert "切块建议多大？" in human
    assert messages[0].content.startswith("你是「飞链」知识库助手")
