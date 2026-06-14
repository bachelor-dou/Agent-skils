"""原子工具：单仓库增长/介绍/DB 查询/Trending。

单仓库容错解析（repo_growth / describe_project）：用户可给
  - 完整 `owner/repo`：直接精确查；
  - 仅项目名（如 `vllm`）或一句描述：用 provider.search_similar 检索候选。
解析策略——能唯一定位就直接用，真有歧义（多个相近）才返回 disambiguation
候选，交给 LLM 找用户确认。
"""

from ..config import GROWTH_CALC_DAYS
from ..capabilities.describe import get_db_info as _get_db_info, describe_project as _describe_project


def _resolve_repo(ctx, raw: str):
    """把用户输入解析为确定的 owner/repo。

    返回 (full_name, None) 表示已唯一定位；(None, payload) 表示需消歧或未找到。
    """
    raw = (raw or "").strip()
    if not raw:
        return None, {"error": "缺少必需参数 repo"}

    # 形如 owner/repo：先精确查；命中即用，未命中则按 repo 名去搜相似项目
    if "/" in raw:
        if ctx.provider.repo_info(raw) is not None:
            return raw, None
        query = raw.split("/", 1)[1].strip() or raw
    else:
        query = raw

    cands = ctx.provider.search_similar(query, limit=5)
    if not cands:
        return None, {"error": f"未找到与 `{raw}` 匹配的仓库，也没有相似项目。请确认 owner/repo。"}

    # 唯一定位：只有一个候选，或恰有一个「项目名完全相同」的强匹配
    if len(cands) == 1:
        return cands[0].full_name, None
    exact = [c for c in cands if c.full_name.rsplit("/", 1)[-1].lower() == query.lower()]
    if len(exact) == 1:
        return exact[0].full_name, None

    # 有歧义：交给 LLM 找用户确认
    return None, {
        "disambiguation": True,
        "message": f"`{raw}` 有多个相近项目，请向用户确认是哪一个（回复完整 owner/repo）。",
        "candidates": [
            {"full_name": c.full_name, "star": c.star, "desc": c.description} for c in cands
        ],
    }


def repo_growth_handler(ctx, args: dict) -> dict:
    repo, payload = _resolve_repo(ctx, args.get("repo"))
    if payload is not None:
        return payload
    ctx.state.active_repo = repo
    return ctx.provider.repo_growth(repo, growth_calc_days=args.get("growth_calc_days", GROWTH_CALC_DAYS))


def describe_project_handler(ctx, args: dict) -> dict:
    repo, payload = _resolve_repo(ctx, args.get("repo"))
    if payload is not None:
        return payload
    ctx.state.active_repo = repo
    return _describe_project(repo=repo, db=ctx.db, token_mgr=ctx.provider.token_mgr)


def get_db_info_handler(ctx, args: dict) -> dict:
    return _get_db_info(db=ctx.db, repo=args.get("repo"))


def fetch_trending_handler(ctx, args: dict) -> dict:
    return ctx.provider.fetch_trending(trending_range=args.get("trending_range", "weekly"))
