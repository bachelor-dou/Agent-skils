"""复合榜单工具：综合/新项目/关键词，内部按序跑 ranking_pipeline。

幂等确认守卫：首次调用（或参数变化）返回"请确认参数"并记录参数签名；
用户确认后 LLM 以相同签名再次调用 → 真正执行。
"""

import json

from ..pipeline.ranking_pipeline import run_ranking
from ..pipeline.cache import RankingCache
from ..infra.db import save_db_desc_only

_MODE_LABEL = {"comprehensive": "综合热榜", "hot_new": "新项目热榜", "keyword": "关键词热榜"}


def _sig(mode: str, params: dict) -> str:
    return json.dumps({"mode": mode, **params}, ensure_ascii=False, sort_keys=True, default=str)


def make_ranking_handler(mode: str):
    label = _MODE_LABEL.get(mode, mode)

    def handler(ctx, args: dict) -> dict:
        params = dict(args)
        sig = _sig(mode, params)

        # 幂等确认守卫
        if ctx.state.pending_confirmation_signature != sig:
            ctx.state.pending_confirmation_signature = sig
            return {
                "needs_confirmation": True,
                "mode": mode,
                "params": params,
                "message": f"将执行【{label}】，参数={params}。确认请回复『开始』。",
            }
        ctx.state.pending_confirmation_signature = None

        if ctx.state.ranking_cache is None:
            ctx.state.ranking_cache = RankingCache()

        result = run_ranking(
            ctx.provider, mode=mode, params=params, db=ctx.db,
            cache=ctx.state.ranking_cache, do_report=True, force_refresh=False,
            progress_cb=getattr(ctx, "progress_cb", None),
        )
        # Agent 路径：仅持久化 desc 字段
        save_db_desc_only(ctx.db)

        ranked = result.get("ranked", [])
        return {
            "mode": result.get("mode", mode),
            "ranked_count": len(ranked),
            "candidates_count": result.get("candidates_count", 0),
            "report_path": result.get("report_path", ""),
            "growth_calc_days": result.get("growth_calc_days"),
            "ranked": [
                {"rank": i + 1, "repo": n, "growth": v["growth"], "star": v["star"]}
                for i, (n, v) in enumerate(ranked)
            ],
        }

    return handler
