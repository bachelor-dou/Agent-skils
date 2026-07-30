"""零成本工具:查本地库、看关键词表、取回暂存结果、抓 Trending。

前三个纯本地读取。Trending 抓的是 HTML 页面,不吃 token 也不受 Search 限流,
所以也算「便宜」那一档。
"""

from __future__ import annotations

import json
import logging

from .. import config
from ..common.timeutil import age_days
from ..infra.store import snapshots, universe
from ..provider.github import trending as trending_api
from . import describe
from .spec import Param, Tool

logger = logging.getLogger("hot_project")

TRENDING_CONDENSE_MAX = 70


def get_db_info(ctx, args: dict) -> dict:
    """不给 repo → 库概览;给 repo → 那条记录。"""
    projects = universe.load()
    if repo := (args.get("repo") or "").strip():
        info = projects.get(repo)
        if info is None:
            return {"repo": repo, "in_db": False,
                    "message": f"{repo} 不在本地库里(可能 star 没到 {config.MIN_STAR},"
                               f"或者还没被每日发现扫到)。"}
        return {"repo": repo, "in_db": True, **info}

    days = snapshots.available_dates()
    return {
        "project_count": len(projects),
        "snapshot_days": len(days),
        "latest_snapshot": str(days[-1]) if days else None,
        "oldest_snapshot": str(days[0]) if days else None,
        "min_star": config.MIN_STAR,
        "with_description": sum(1 for i in projects.values() if i.get("desc")),
    }


def get_keyword_catalog(ctx, args: dict) -> dict:
    """返回预设关键词分组表。

    这张表约 4k 字符。常驻 system 提示词的话,对不做关键词榜的对话是纯浪费,
    所以改成模型判断需要挑词时按需来取。
    """
    return {
        "categories": config.SEARCH_KEYWORDS,
        "usage": "从相关分组挑关键词,再补充分组没覆盖到的英文搜索词,"
                 "一起传给 keyword_ranking 的 keywords 参数。",
    }


def recall_tool_result(ctx, args: dict) -> dict:
    """取回之前因体积过大被暂存的工具结果。"""
    ref = str(args.get("ref", "")).strip()
    store = (ctx.state.tool_state.get("offloaded", {}) if ctx.state else {})
    raw = store.get(ref)
    if raw is None:
        return {"error": f"没找到暂存结果 {ref}。可用的 ref:{sorted(store)}"}
    try:
        return {"ref": ref, "result": json.loads(raw)}
    except (json.JSONDecodeError, TypeError):
        return {"ref": ref, "result": raw}


def fetch_trending(ctx, args: dict) -> dict:
    """抓 GitHub Trending。`all` = 日/周/月三榜合并去重。"""
    period = args.get("trending_range", trending_api.DEFAULT_PERIOD)
    periods = trending_api.PERIODS if period == "all" else (period,)

    merged: dict[str, dict] = {}
    for one in periods:
        for repo in ctx.gh.trending(one):
            merged.setdefault(repo["full_name"], repo)

    items = list(merged.values())
    if items:
        for repo, short in zip(items, describe.condense(items, TRENDING_CONDENSE_MAX)):
            repo["short_desc"] = short
    return {"trending_range": period, "count": len(items), "repos": items}


TOOLS = (
    Tool("get_db_info",
         "【数据库查询】查本地库(不联网)。不传 repo → 概览:收录了多少仓库、"
         "快照覆盖到哪天、入库门槛是多少;传 repo → 那个仓库的缓存记录"
         "(star、创建时间、语言、已生成的介绍)。想知道「榜单数据新不新」时先看概览。",
         get_db_info,
         (Param("repo", "str", "可选,查特定仓库;不传返回概览。", default=None),)),
    Tool("get_keyword_catalog",
         "【关键词分组表】返回预设的搜索关键词分组全表(本地读取,零成本)。"
         "做关键词榜(keyword_ranking)需要挑词时先调用本工具查看各组关键词,"
         "再挑选+补充后传入 keywords。",
         get_keyword_catalog),
    Tool("recall_tool_result",
         "【取回暂存结果】重新读取之前因体积过大被暂存的工具结果(本地读取,零成本)。"
         "当你在历史里看到形如 {offloaded:true, ref:'tr3', digest:...} 的存根、"
         "且需要其完整内容时调用。",
         recall_tool_result,
         (Param("ref", "str", "存根里的 ref,如 tr3。"),)),
    Tool("fetch_trending",
         "【Trending】获取 GitHub Trending 列表。all = 日/周/月三榜合并去重。",
         fetch_trending,
         (Param("trending_range", "enum", "daily / weekly(默认) / monthly / all",
                default=trending_api.DEFAULT_PERIOD,
                choices=(*trending_api.PERIODS, "all")),)),
)
