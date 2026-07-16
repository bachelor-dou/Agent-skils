"""repo_growth 工具：查单个仓库近期 star 增长。"""

from ...config import GROWTH_CALC_DAYS
from ..basic.resolve import resolve_repo


def repo_growth_handler(ctx, args: dict) -> dict:
    repo, payload = resolve_repo(ctx, args.get("repo"))
    if payload is not None:
        return payload
    ctx.state.active_repo = repo
    return ctx.provider.repo_growth(repo, growth_calc_days=args.get("growth_calc_days", GROWTH_CALC_DAYS))
