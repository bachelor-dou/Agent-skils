"""唯一榜单流水线：collect→growth_calc→threshold→rank→report，分阶段缓存。

Agent 复合工具与 scheduled_update 共用本流水线（单一实现）。
持久化（desc_only / 全量 save_db）由调用方负责，本模块保持纯。
"""

import logging

from ..config import GROWTH_CALC_DAYS, STAR_GROWTH_THRESHOLD
from ..ranking import step2_rank_and_select
from ..report import step3_generate_report
from ..infra.db import get_db_age_days
from .cache import RankingCache

logger = logging.getLogger("hot_projects")


def _collect(provider, mode, params) -> list[dict]:
    """收集候选：关键词搜索 +（非 keyword 模式）星段扫描 + Trending 补源。"""
    repos: list[dict] = []
    seen: set[str] = set()

    sr = provider.search_by_keywords(
        categories=params.get("categories"),
        min_star=params["min_star"],
        days_since_created=params.get("days_since_created"),
    )
    raw = sr.get("_raw_repos", [])
    repos.extend(raw)
    seen.update(r["full_name"] for r in raw)

    if mode != "keyword":
        scan = provider.scan_star_range(
            min_star=params["min_star"],
            max_star=params.get("max_star"),
            seen_repos=seen,
            days_since_created=params.get("days_since_created"),
        )
        repos.extend(scan.get("_raw_repos", []))

        tr = provider.fetch_trending(trending_range="all")
        from ..capabilities import trending_repo_to_search_repo
        for r in tr.get("_raw_repos", []):
            fn = r["full_name"]
            if fn not in seen:
                seen.add(fn)
                repos.append(trending_repo_to_search_repo(r))

    return repos


def run_ranking(provider, mode, params, db, cache: RankingCache | None = None,
                do_report: bool = True, force_refresh: bool = False) -> dict:
    """执行榜单流水线，分阶段复用 cache。

    mode: "comprehensive" | "hot_new" | "keyword"
    params: min_star/max_star/categories/days_since_created/growth_calc_days/
            growth_threshold/top_n
    """
    cache = cache or RankingCache()

    # ── 1) collect ──
    collect_sig = {
        "mode": mode, "min_star": params["min_star"], "max_star": params.get("max_star"),
        "categories": params.get("categories"), "days_since_created": params.get("days_since_created"),
    }
    repos = cache.get("collect", collect_sig)
    if repos is None:
        repos = _collect(provider, mode, params)
        cache.set("collect", collect_sig, repos)

    # 综合/关键词榜未指定窗口 → 用 DB 年龄窗口（隐藏行为#1）
    growth_calc_days = params.get("growth_calc_days")
    window_specified = growth_calc_days is not None
    if not window_specified and mode in ("comprehensive", "keyword"):
        age = get_db_age_days(db)
        if db.get("valid") and age and age > 0:
            growth_calc_days = age
    effective_window = growth_calc_days or GROWTH_CALC_DAYS
    days_since = params.get("days_since_created")

    # ── 2) growth_calc（昂贵）──
    growth_sig = {**collect_sig, "growth_calc_days": growth_calc_days, "days_since_created": days_since}
    growth = cache.get("growth_calc", growth_sig)
    if growth is None:
        growth = provider.batch_growth(
            repos, db, growth_threshold=0, days_since_created=days_since,
            growth_calc_days=effective_window, force_refresh=force_refresh,
            window_specified=window_specified,
        )
        cache.set("growth_calc", growth_sig, growth)
    effective_window = growth.get("growth_calc_days", effective_window)

    # ── 3) threshold（廉价过滤）──
    threshold = params.get("growth_threshold", STAR_GROWTH_THRESHOLD)
    thr_sig = {**growth_sig, "growth_threshold": threshold}
    candidates = cache.get("threshold", thr_sig)
    if candidates is None:
        candidates = {k: v for k, v in growth.get("candidates", {}).items() if v["growth"] >= threshold}
        cache.set("threshold", thr_sig, candidates)

    # ── 4) rank（廉价）──
    rank_mode = "hot_new" if mode == "hot_new" else "comprehensive"
    rank_sig = {**thr_sig, "rank_mode": rank_mode, "top_n": params.get("top_n")}
    ranked = cache.get("rank", rank_sig)
    if ranked is None:
        ordered = step2_rank_and_select(
            candidates, mode=rank_mode, db=db,
            days_since_created=days_since,
            prefiltered_days_since_created=days_since,  # 隐藏行为#3：预筛窗口透传
        )
        top_n = params.get("top_n")
        ranked = ordered[:top_n] if top_n else ordered
        cache.set("rank", rank_sig, ranked)

    result = {
        "ranked": ranked,
        "candidates_count": len(candidates),
        "mode": rank_mode,
        "growth_calc_days": effective_window,
    }

    # ── 5) report ──
    if do_report:
        report_path = step3_generate_report(
            ranked, db, mode=rank_mode,
            days_since_created=days_since if rank_mode == "hot_new" else None,
            growth_calc_days=effective_window, growth_threshold=threshold,
            min_star=params["min_star"],
            token_mgr=getattr(provider, "token_mgr", None),
        )
        result["report_path"] = report_path
    return result
