"""飞链 RAG 的命令行入口。

    feilian index     同步文档到向量库（增量）
    feilian ask "问题" 单次问答
    feilian chat      交互式多轮问答
    feilian stats     看索引状态
    feilian doctor    环境体检
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .chain import RAGChat
from .config import Settings
from .errors import FeilianError
from .index import VectorIndex
from .providers import health_check


def _print_banner(settings: Settings, chunks: int) -> None:
    print("=" * 64)
    print("  飞链 RAG · 知识检索问答")
    print("=" * 64)
    print(f"  配置   {settings.describe()}")
    print(f"  文档   {settings.docs_dir}")
    print(f"  索引   {settings.store_dir}（{chunks} 块）")
    print("=" * 64)
    print("  直接输入问题提问；/index 增量同步；/stats 索引状态；/reset 清空对话；/quit 退出")
    print("=" * 64)


def _cmd_index(args: argparse.Namespace, settings: Settings) -> int:
    index = VectorIndex(settings)
    result = index.sync(rebuild=args.rebuild)
    print(f"[索引] {result.summary()}")
    for name in result.added:
        print(f"  + {name}")
    for name in result.updated:
        print(f"  ~ {name}")
    for name in result.removed:
        print(f"  - {name}")
    for warning in result.warnings:
        print(f"  ! {warning}")
    if not result.changed:
        print("  文档没有变化，索引直接复用。")
    return 0


def _cmd_ask(args: argparse.Namespace, settings: Settings) -> int:
    index = VectorIndex(settings)
    if index.count() == 0:
        print("[提示] 向量库还是空的，先执行 `feilian index` 建索引。")
        return 1
    chat = RAGChat(settings, index=index)
    answer = chat.ask(args.question, k=args.k)
    if args.json:
        print(json.dumps(answer.as_dict(), ensure_ascii=False, indent=2))
    else:
        print(answer.render())
    return 0 if answer.found else 2


def _cmd_chat(args: argparse.Namespace, settings: Settings) -> int:
    index = VectorIndex(settings)
    chat = RAGChat(settings, index=index)
    _print_banner(settings, index.count())

    while True:
        try:
            line = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n再见。")
            return 0

        if not line:
            continue
        lowered = line.lower()
        if lowered in {"/quit", "/exit", "quit", "exit", "q"}:
            print("再见。")
            return 0
        if lowered in {"/index", "index"}:
            result = index.sync()
            print(f"[索引] {result.summary()}")
            for warning in result.warnings:
                print(f"  ! {warning}")
            continue
        if lowered in {"/stats", "stats"}:
            print(json.dumps(index.stats(), ensure_ascii=False, indent=2))
            continue
        if lowered in {"/reset", "reset"}:
            chat.reset()
            print("[会话] 已清空上下文。")
            continue
        if lowered in {"/help", "help"}:
            print("  /index 增量同步文档  /stats 索引状态  /reset 清空对话  /quit 退出")
            continue

        answer = chat.ask(line)
        print()
        print(answer.render())


def _cmd_stats(args: argparse.Namespace, settings: Settings) -> int:
    print(json.dumps(VectorIndex(settings).stats(), ensure_ascii=False, indent=2))
    return 0


def _cmd_doctor(args: argparse.Namespace, settings: Settings) -> int:
    print(f"[配置] {settings.describe()}")
    print(f"[文档] {settings.docs_dir}（存在：{settings.docs_dir.exists()}）")
    report = health_check(settings)
    print(f"[模型] {'OK' if report['ok'] else 'FAIL'} · {report['detail']}")
    if report["ok"]:
        for name in report.get("models", []):
            print(f"    - {name}")
    try:
        stats = VectorIndex(settings).stats()
        print(f"[索引] 文件 {stats['files']} · 块 {stats['chunks_in_store']} · 目录 {stats['store_dir']}")
    except FeilianError as exc:
        print(f"[索引] 读取失败：{exc}")
    return 0 if report["ok"] else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="feilian",
        description="飞链 RAG：把本地文档链成可追溯的答案。",
    )
    parser.add_argument("--provider", choices=["openai", "ollama"], help="临时覆盖 LLM_PROVIDER")
    parser.add_argument("--docs", help="临时覆盖文档目录")
    parser.add_argument("--k", type=int, help="临时覆盖检索条数")

    sub = parser.add_subparsers(dest="command", required=True)

    p_index = sub.add_parser("index", help="同步文档到向量库（增量）")
    p_index.add_argument("--rebuild", action="store_true", help="忽略 manifest，整库重建")
    p_index.set_defaults(func=_cmd_index)

    p_ask = sub.add_parser("ask", help="单次问答")
    p_ask.add_argument("question", help="要问的问题")
    p_ask.add_argument("--json", action="store_true", help="输出 JSON（含引用明细）")
    p_ask.set_defaults(func=_cmd_ask)

    p_chat = sub.add_parser("chat", help="交互式多轮问答")
    p_chat.set_defaults(func=_cmd_chat)

    p_stats = sub.add_parser("stats", help="查看索引状态")
    p_stats.set_defaults(func=_cmd_stats)

    p_doctor = sub.add_parser("doctor", help="环境体检")
    p_doctor.set_defaults(func=_cmd_doctor)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        settings = Settings.load().with_overrides(
            provider=args.provider,
            docs_dir=Path(args.docs) if args.docs else None,
            top_k=args.k,
        )
        return int(args.func(args, settings))
    except FeilianError as exc:
        print(f"[错误] {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("\n再见。")
        return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
