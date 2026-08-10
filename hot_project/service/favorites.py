"""收藏的应用逻辑 —— 「收藏时补概要」这类决策在这里,api_server 只翻译 HTTP。

POST 与 GET 返回同一形状:带出场统计的**权威清单**。前端乐观更新后直接拿响应对账,
分类规则(如「子分类随父清空」)以数据层 `set_favorite` 为准,前端本地那份只是预演。
"""

from __future__ import annotations

from ..infra.data_access import favorites, reports, universe
from . import describe

FAVORITE_DESC_MAX = 60
README_FOR_SHORT_DESC = 1200     # 兜底出概要时喂给模型的 README 截断长度


def short_desc(name: str, saved: dict, gh=None) -> str:
    """收藏卡片上那句中文概要。网页 ☆ 和 agent 的 add_favorite 共用这一份。

    素材三档退让:GitHub 原文简介 → 库里的四段介绍 → README(多一次请求,`gh=None` 则没有)。
    """
    source = (saved.get("gh_desc") or "").strip() or (saved.get("desc") or "").strip()
    if not source and gh is not None and gh.usable:
        readme = (gh.profile(name, want=("readme",)).get("readme") or {}).get("text", "")
        source = readme[:README_FOR_SHORT_DESC].strip()
    if not source:
        return ""
    return describe.condense([{"full_name": name, "description": source}],
                             max_chars=FAVORITE_DESC_MAX)[0]


def listing(user_id: str) -> tuple[list[dict], int]:
    """带「共 N 期上榜 M 期」出场统计的收藏清单。非法 user_id 抛 `ValueError`。"""
    if not favorites.valid_user_id(user_id):
        raise ValueError("无效的 user_id")
    counts, total = reports.appearance_counts()
    return [dict(item, report_count=counts.get(item.get("repo", ""), 0),
                 report_total=total)
            for item in favorites.get(user_id)], total


def _auto_short(repo: str) -> str | None:
    from ..provider.github import client as github
    return short_desc(repo, universe.load().get(repo, {}), github.shared()) or None


def update(user_id: str, repo: str, action: str, *, source_report: str = "",
           category: str | None = None, subcategory: str | None = None,
           short_desc: str | None = None) -> tuple[list[dict], int]:
    """add / remove 一条收藏,返回和 `listing` 同款的权威清单。非法输入抛 `ValueError`。

    概要只在「新收藏且调用方没给」时现生成 —— 不在出报告时给几百个项目预生成,
    也不覆盖用户手写的那句。
    """
    short = None
    if action == "add":
        already = next((x for x in favorites.get(user_id)
                        if x.get("repo") == repo), None)
        if short_desc is not None:
            short = short_desc.strip()[:FAVORITE_DESC_MAX]
        elif already is None:
            short = _auto_short(repo)
    favorites.set_favorite(user_id, repo, action, source_report=source_report,
                           short_desc=short, category=category,
                           subcategory=subcategory)
    return listing(user_id)
