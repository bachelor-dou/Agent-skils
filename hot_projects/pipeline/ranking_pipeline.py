"""唯一榜单流水线：collect→growth_calc→threshold→rank→report，分阶段缓存。

Agent 复合工具与 scheduled_update 共用本流水线（单一实现）。
持久化（desc_only / 全量 save_db）由调用方负责，本模块保持纯。
"""

import logging

from ..config import GROWTH_CALC_DAYS, STAR_GROWTH_THRESHOLD
from ..capabilities.scoring import step2_rank_and_select
from ..capabilities.report import step3_generate_report
from ..infra.db import get_db_age_days
from .cache import RankingCache

logger = logging.getLogger("hot_projects")


def _emit(progress_cb, percent, label: str) -> None:
    """安全回传进度（百分比 0-100 + 文案）；回调异常绝不影响流水线。"""
    if progress_cb is None:
        return
    try:
        progress_cb(int(percent), label)
    except Exception:  # noqa: BLE001
        logger.debug("progress_cb 异常已忽略", exc_info=True)


def _collect(provider, mode, params, progress_cb=None) -> list[dict]:
    """收集候选：关键词搜索 +（非 keyword 模式）星段扫描 + Trending 补源。"""
    repos: list[dict] = []
    seen: set[str] = set()

    _emit(progress_cb, 8, "搜索关键词…")
    sr = provider.search_by_keywords(
        categories=params.get("categories"),
        min_star=params["min_star"],
        days_since_created=params.get("days_since_created"),
        keywords=params.get("keywords"),
    )
    raw = sr.get("_raw_repos", [])
    repos.extend(raw)
    seen.update(r["full_name"] for r in raw)

    if mode != "keyword":
        _emit(progress_cb, 16, "扫描 star 区间…")
        scan = provider.scan_star_range(
            min_star=params["min_star"],
            max_star=params.get("max_star"),
            seen_repos=seen,
            days_since_created=params.get("days_since_created"),
        )
        repos.extend(scan.get("_raw_repos", []))

        _emit(progress_cb, 24, "合并 Trending…")
        tr = provider.fetch_trending(trending_range="all")
        from ..capabilities import trending_repo_to_search_repo
        for r in tr.get("_raw_repos", []):
            fn = r["full_name"]
            if fn not in seen:
                seen.add(fn)
                repos.append(trending_repo_to_search_repo(r))

    return repos


def run_ranking(provider, mode, params, db, cache: RankingCache | None = None,
                do_report: bool = True, force_refresh: bool = False,
                progress_cb=None) -> dict:
    """执行榜单流水线，分阶段复用 cache。

    mode: "comprehensive" | "hot_new" | "keyword"
    params: min_star/max_star/categories/days_since_created/growth_calc_days/
            growth_threshold/top_n
    """
    cache = cache or RankingCache()
    _emit(progress_cb, 3, "开始执行…")

    # ── 1) collect ──
    collect_sig = {
        "mode": mode, "min_star": params["min_star"], "max_star": params.get("max_star"),
        "categories": params.get("categories"), "days_since_created": params.get("days_since_created"),
        "keywords": params.get("keywords"),
    }
    repos = cache.get("collect", collect_sig)
    if repos is None:
        repos = _collect(provider, mode, params, progress_cb=progress_cb)
        cache.set("collect", collect_sig, repos)
    _emit(progress_cb, 30, f"候选收集完成（{len(repos)} 个）")

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
        _emit(progress_cb, 32, "计算项目增长…（较慢）")
        growth = provider.batch_growth(
            repos, db, growth_threshold=0, days_since_created=days_since,
            growth_calc_days=effective_window, force_refresh=force_refresh,
            window_specified=window_specified,
        )
        cache.set("growth_calc", growth_sig, growth)
    effective_window = growth.get("growth_calc_days", effective_window)
    _emit(progress_cb, 65, "增长计算完成")

    # ── 3) threshold（廉价过滤）──
    threshold = params.get("growth_threshold", STAR_GROWTH_THRESHOLD)
    thr_sig = {**growth_sig, "growth_threshold": threshold}
    candidates = cache.get("threshold", thr_sig)
    if candidates is None:
        candidates = {k: v for k, v in growth.get("candidates", {}).items() if v["growth"] >= threshold}
        cache.set("threshold", thr_sig, candidates)
    _emit(progress_cb, 68, f"筛选候选（{len(candidates)} 个达标）")

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
    _emit(progress_cb, 72, "排序完成")

    result = {
        "ranked": ranked,
        "candidates_count": len(candidates),
        "mode": rank_mode,
        "growth_calc_days": effective_window,
    }

    # ── 5) report ──
    if do_report:
        _emit(progress_cb, 74, "生成报告…")

        def _report_cb(frac: float, label: str) -> None:
            _emit(progress_cb, 74 + 25 * max(0.0, min(1.0, frac)), label)

        report_path = step3_generate_report(
            ranked, db, mode=rank_mode,
            days_since_created=days_since if rank_mode == "hot_new" else None,
            growth_calc_days=effective_window, growth_threshold=threshold,
            min_star=params["min_star"],
            token_mgr=getattr(provider, "token_mgr", None),
            progress_cb=_report_cb,
            topic=params.get("topic") if mode == "keyword" else None,
        )
        result["report_path"] = report_path
    _emit(progress_cb, 100, "完成")
    return result
