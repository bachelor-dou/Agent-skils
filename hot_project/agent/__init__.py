"""agent —— ReAct 循环(loop)、会话历史与压缩(history)、系统提示词(prompts)。

这一层只认 `tools.Registry`,不认识任何具体工具,加一个工具不需要碰这里。
"""

from __future__ import annotations

from .history import Session
from .loop import Agent

__all__ = ["Agent", "Session", "build"]


def build() -> Agent:
    """生产用的 Agent:共享 LLM 客户端 + 全部工具 + 共享的 GitHub 门面。"""
    return Agent()
