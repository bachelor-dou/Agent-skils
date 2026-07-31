#!/usr/bin/env python
"""命令行对话入口:`python -m hot_project.agent_cli`,自然语言提问,quit 退出。

日志只写文件、不打控制台 —— 对话本身就在控制台上,再混进 INFO 日志就没法看了。
"""

from __future__ import annotations

import logging
import sys

from . import config
from .agent import build
from .common import logs
from .infra import llm

try:
    import readline        # 让上下方向键能翻历史。Windows 上没有,不影响使用
except ImportError:        # pragma: no cover
    readline = None

logger = logging.getLogger("hot_project")

HISTORY_LINES = 200
QUIT_WORDS = frozenset({"quit", "exit", "q"})


def preflight() -> list[str]:
    """开跑前检查配置,返回要提醒用户的话。缺什么都不阻止启动 —— 有些问题不需要联网。"""
    notes = []
    if not config.github_tokens():
        notes.append("没配 GitHub token(GITHUB_TOKENS),查仓库、出榜都会失败。\n"
                     "  export GITHUB_TOKENS='ghp_xxx,ghp_yyy'")
    if not llm.get().configured():
        notes.append("没配任何 LLM(config.LLM_MODELS 里的平台都没有 key),对话没法进行。\n"
                     "  export LLM_A_KEY='...'")
    return notes


def main() -> int:
    log_path = logs.setup(config.LOG_DIR, "cli", console=False)
    if readline is not None:
        readline.set_history_length(HISTORY_LINES)

    print("=" * 60)
    print("  GitHub 热门项目发现 Agent")
    print("  输入自然语言指令;quit / exit / q 退出")
    print(f"  日志:{log_path}")
    print("=" * 60)
    for note in preflight():
        print(f"\n提示:{note}")
    print()

    agent = build()
    while True:
        try:
            line = input("你> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n再见!")
            return 0
        if not line:
            continue
        if line.lower() in QUIT_WORDS:
            print("再见!")
            return 0
        print("思考中...\n")
        try:
            print(f"Agent> {agent.chat(line)}\n")
        except Exception as e:      # noqa: BLE001 —— 一次提问失败不该让整个会话退出
            logger.exception("对话异常")
            print(f"Agent> 这一轮出错了:{e}(详情见日志)\n")


if __name__ == "__main__":
    sys.exit(main())
