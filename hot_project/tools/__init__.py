"""tools —— Agent 能调用的能力,一个工具一条 `Tool` 声明。

    spec.py          Param / Tool / Registry / Ctx —— 契约本身
    repo_tools.py    单仓库:增长、介绍、画像、搜索、收藏
    report_tools.py  读历史报告:整体分析、star 轨迹
    rank_tools.py    三张榜(昂贵,执行前要用户确认)
    local_tools.py   零成本:查库、关键词表、取回暂存、Trending

    ranking.py       榜单流水线(rank_tools 和周报 cron 共用)
    report.py        Markdown 报告生成
    describe.py      项目介绍:抓素材 + 提示词 + 调 LLM

给模型看的 JSON schema 和运行时校验规则都从同一个 `Param` 长出来,不可能漂移。
提示词也在这一层:「用中文、四段式、不许编造」是产品决定,「某家挂了换哪家」才是基础设施。
"""

from __future__ import annotations

from . import local_tools, rank_tools, repo_tools, report_tools
from .spec import Ctx, Param, Registry, Tool

__all__ = ["Ctx", "Param", "Registry", "Tool", "registry"]

_shared: Registry | None = None


def registry() -> Registry:
    """全部工具。进程内共享 —— 它是只读的。"""
    global _shared
    if _shared is None:
        _shared = Registry([*rank_tools.TOOLS, *repo_tools.TOOLS,
                            *report_tools.TOOLS, *local_tools.TOOLS])
    return _shared
