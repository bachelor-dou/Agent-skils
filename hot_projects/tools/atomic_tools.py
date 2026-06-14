"""原子工具：单仓库增长/介绍/DB 查询/Trending。

单仓库模糊消歧：repo_growth / describe_project 先精确查；查不到则用 provider.search_similar
返回相似候选给 LLM，由 LLM 询问用户选择正确的 owner/repo。
"""

from ..config import GROWTH_CALC_DAYS
from ..capabilities.describe import get_db_info as _get_db_info, describe_project as _describe_project


def _disambig(ctx, raw_name: str) -> dict:
    cands = ctx.provider.search_similar(raw_name, limit=5)
    if not cands:
        return {"error": f"未找到仓库: {raw_name}，也没有相似项目。请确认 owner/repo。"}
    return {
        "disambiguation": True,
        "message": f"没找到 `{raw_name}`，你是不是指以下之一？请回复完整 owner/repo。",
        "candidates": [
            {"full_name": c.full_name, "star": c.star, "desc": c.description} for c in cands
        ],
    }


def repo_growth_handler(ctx, args: dict) -> dict:
    repo = (args.get("repo") or "").strip()
    if not repo:
        return {"error": "缺少必需参数 repo"}
    if ctx.provider.repo_info(repo) is None:
        return _disambig(ctx, repo)
    ctx.state.active_repo = repo
    return ctx.provider.repo_growth(repo, growth_calc_days=args.get("growth_calc_days", GROWTH_CALC_DAYS))


def describe_project_handler(ctx, args: dict) -> dict:
    repo = (args.get("repo") or "").strip()
    if not repo:
        return {"error": "缺少必需参数 repo"}
    if ctx.provider.repo_info(repo) is None:
        return _disambig(ctx, repo)
    ctx.state.active_repo = repo
    return _describe_project(repo=repo, db=ctx.db, token_mgr=ctx.provider.token_mgr)


def get_db_info_handler(ctx, args: dict) -> dict:
    return _get_db_info(db=ctx.db, repo=args.get("repo"))


def fetch_trending_handler(ctx, args: dict) -> dict:
    return ctx.provider.fetch_trending(trending_range=args.get("trending_range", "weekly"))
