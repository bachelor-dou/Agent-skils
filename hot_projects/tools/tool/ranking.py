"""榜单复合工具：综合/新项目/关键词三榜 —— 唯一入口，Agent 与定时任务共用。

内部编排:collect→growth_calc→threshold→rank→report,分阶段参数签名缓存;
昂贵执行前用「幂等确认守卫」要用户确认。
持久化(desc_only / 全量 save_db)由调用方负责,本模块保持纯。
"""

import json
import logging

from ...config import (
    GROWTH_CALC_DAYS,
    STAR_GROWTH_THRESHOLD,
    RECENT_GROWTH_DAYS,
    BURST_PROBE_ENABLED,
)
from ..basic.scoring import step2_rank_and_select
from ..basic.report import step3_generate_report
from ...infra.db import get_db_age_days, save_db_desc_only

logger = logging.getLogger("hot_projects")


# ══════════════════════════════════════════════════════════════
# 分阶段签名缓存(会话级):上游阶段变化使下游失效
# ══════════════════════════════════════════════════════════════

STAGE_ORDER = ["collect", "growth_calc", "threshold", "rank", "report"]


def _stage_sig(params: dict) -> str:
    return json.dumps(params, ensure_ascii=False, sort_keys=True, default=str)


class RankingCache:
    """会话级榜单缓存。get 命中需阶段名 + 参数签名一致;set 会失效所有下游阶段。"""

    def __init__(self) -> None:
        self._store: dict[str, tuple[str, object]] = {}
        # 会话级旁路缓存:不随阶段失效。用于缓存"按项目"的稳定事实(如各候选最近 K 天增长),
        # 使阈值/top_n 等下游参数变化时无需对已算过的项目重复发 API。
        self.aux: dict[str, dict] = {}

    def get(self, stage: str, params: dict):
        entry = self._store.get(stage)
        if entry is None or entry[0] != _stage_sig(params):
            return None
        return entry[1]

    def set(self, stage: str, params: dict, payload) -> None:
        self._store[stage] = (_stage_sig(params), payload)
        idx = STAGE_ORDER.index(stage)
        for downstream in STAGE_ORDER[idx + 1:]:
            self._store.pop(downstream, None)


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
        from ..basic import trending_repo_to_search_repo
        for r in tr.get("_raw_repos", []):
            fn = r["full_name"]
            if fn not in seen:
                seen.add(fn)
                repos.append(trending_repo_to_search_repo(r))

    return repos


def _calc_recent_growth(provider, candidates: dict, db: dict, recent_days: int) -> dict:
    """对候选池计算"最近 recent_days 天"增长，供打分做爆发加成。

    复用 batch_growth（实时 API，force_refresh=False 不写 DB）：候选 refreshed_at 与
    recent_days 窗口不匹配 → 逐项回退实时二分，得到的就是最近 recent_days 天增长。

    Returns: {full_name: recent_growth}，仅含成功解析（>=0）的项目。
    """
    if not candidates:
        return {}
    cand_repos = [
        {
            "full_name": fn,
            "star": info.get("star", 0),
            "_raw": {"created_at": info.get("created_at", "")},
        }
        for fn, info in candidates.items()
    ]
    # 这趟是"最近窗口"副计算：把 hot_projects 的 INFO 噪声（批量增长/[GROWTH] 逐条）压到
    # WARNING，避免和主 7 天增长的日志混淆；调用方会打印清晰的探针起止行。
    growth_logger = logging.getLogger("hot_projects")
    prev_level = growth_logger.level
    growth_logger.setLevel(logging.WARNING)
    try:
        res = provider.batch_growth(
            cand_repos, db,
            growth_threshold=0,
            growth_calc_days=recent_days,
            window_specified=True,
            force_refresh=False,
            candidate_log_threshold=10 ** 9,  # 抑制 [OK] 候选 日志
        )
    finally:
        growth_logger.setLevel(prev_level)
    out: dict[str, int] = {}
    for fn, info in res.get("candidates", {}).items():
        g = info.get("growth")
        if isinstance(g, int) and g >= 0:
            out[fn] = g
    return out


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
        if age and age > 0:
            # 未指定窗口：用 DB 年龄当窗口以复用 DB 差值，封顶在默认窗口 GROWTH_CALC_DAYS。
            # cron 每 7 天跑，DB 年龄基本 ≤7；偶尔 >7 则退回默认 7——此时各项目 refreshed_at
            # 比窗口旧出超过 5h 容差，逐项匹配会自动失败、回退实时（DB 实质已过期）。
            # 不再依赖静态 DATA_EXPIRE_DAYS 驱动的 db["valid"]。
            growth_calc_days = min(age, GROWTH_CALC_DAYS)
    effective_window = growth_calc_days or GROWTH_CALC_DAYS
    days_since = params.get("days_since_created")

    # 真实增长阈值：用于 threshold 阶段过滤 + growth_calc 阶段的候选日志展示。
    threshold = params.get("growth_threshold", STAR_GROWTH_THRESHOLD)

    # ── 2) growth_calc（昂贵）──
    # growth_threshold=0：候选池全量收录，使 threshold 阶段可用不同阈值反复重筛而不重算增长
    #   （growth_sig 不含阈值，故缓存阈值无关）。
    # candidate_log_threshold=threshold：[OK] 候选 日志仅展示达标候选，保持"候选=达标"语义。
    growth_sig = {**collect_sig, "growth_calc_days": growth_calc_days, "days_since_created": days_since}
    growth = cache.get("growth_calc", growth_sig)
    if growth is None:
        _emit(progress_cb, 32, "计算项目增长…（较慢）")
        growth = provider.batch_growth(
            repos, db, growth_threshold=0, days_since_created=days_since,
            growth_calc_days=effective_window, force_refresh=force_refresh,
            window_specified=window_specified,
            candidate_log_threshold=threshold,
        )
        cache.set("growth_calc", growth_sig, growth)
    effective_window = growth.get("growth_calc_days", effective_window)
    growth_candidates_count = len(growth.get("candidates", {}))
    collected_count = growth.get("total_checked", len(repos))
    excluded_count = max(0, collected_count - growth_candidates_count)
    logger.info(
        "增长候选池(增长≥0): %s 个（本轮计算 %s，掉星/未确定剔除 %s）。",
        growth_candidates_count, collected_count, excluded_count,
    )
    _emit(progress_cb, 65, "增长计算完成")

    # ── 3) threshold（廉价过滤）──
    thr_sig = {**growth_sig, "growth_threshold": threshold}
    candidates = cache.get("threshold", thr_sig)
    if candidates is None:
        candidates = {k: v for k, v in growth.get("candidates", {}).items() if v["growth"] >= threshold}
        cache.set("threshold", thr_sig, candidates)
    logger.info("达标候选池(growth >= %s): %s 个。", threshold, len(candidates))
    _emit(progress_cb, 68, f"筛选达标候选（{len(candidates)} 个）")

    rank_mode = "hot_new" if mode == "hot_new" else "comprehensive"

    # ── 3.5) recent_growth：候选池最近 K 天增长，给打分做"最近爆发"加成（仅综合/关键词榜）──
    # 按项目缓存在 cache.aux（不随阶段失效）：阈值/top_n 变化时只对新出现的候选发 API。
    recent_probe_count = 0
    boost_applied_count = 0
    if BURST_PROBE_ENABLED and rank_mode == "comprehensive" and candidates:
        recent_by_repo = cache.aux.setdefault("recent_growth", {})
        missing = {fn: info for fn, info in candidates.items() if fn not in recent_by_repo}
        if missing:
            _emit(progress_cb, 70, f"最近 {RECENT_GROWTH_DAYS} 天爆发探针（{len(missing)} 个）")
            logger.info("最近爆发探针: 对 %s 个达标候选实时计算近 %s 天增长…", len(missing), RECENT_GROWTH_DAYS)
            recent_by_repo.update(_calc_recent_growth(provider, missing, db, RECENT_GROWTH_DAYS))
        for fn, info in candidates.items():
            if fn in recent_by_repo:
                info["recent_growth"] = recent_by_repo[fn]
        # 爆发加成生效数：近 K 天速率 > 整窗平均速率（acceleration > 1）
        recent_probe_count = sum(1 for info in candidates.values() if "recent_growth" in info)
        if effective_window > 0:
            for info in candidates.values():
                rg = info.get("recent_growth")
                g = info.get("growth", 0)
                if rg is not None and g > 0 and (rg / RECENT_GROWTH_DAYS) > (g / effective_window):
                    boost_applied_count += 1
        logger.info(
            "最近爆发探针完成: %s 个候选，爆发加成生效 %s 个。",
            recent_probe_count, boost_applied_count,
        )

    # ── 4) rank（廉价）──
    rank_sig = {**thr_sig, "rank_mode": rank_mode, "top_n": params.get("top_n")}
    ranked = cache.get("rank", rank_sig)
    if ranked is None:
        ordered = step2_rank_and_select(
            candidates, mode=rank_mode, db=db,
            days_since_created=days_since,
            prefiltered_days_since_created=days_since,  # 隐藏行为#3：预筛窗口透传
            growth_calc_days=effective_window,
        )
        top_n = params.get("top_n")
        ranked = ordered[:top_n] if top_n else ordered
        cache.set("rank", rank_sig, ranked)
    requested_top_n = params.get("top_n")
    if requested_top_n and len(ranked) < requested_top_n:
        logger.warning(
            "达标候选不足 requested_top_n=%s returned=%s candidates=%s growth_pool=%s",
            requested_top_n, len(ranked), len(candidates), growth_candidates_count,
        )
    _emit(progress_cb, 72, "排序完成")

    result = {
        "ranked": ranked,
        "growth_candidates_count": growth_candidates_count,
        "candidates_count": len(candidates),
        "returned_count": len(ranked),
        "mode": rank_mode,
        "growth_calc_days": effective_window,
        "funnel": {
            "collected": collected_count,
            "db_diff": growth.get("db_diff_count", 0),
            "realtime": growth.get("realtime_count", 0),
            "growth_pool": growth_candidates_count,
            "qualified": len(candidates),
            "recent_probe": recent_probe_count,
            "boost_applied": boost_applied_count,
            "ranked": len(ranked),
        },
    }

    # ── 5) report ──
    if do_report:
        _emit(progress_cb, 74, "生成报告…")

        def _report_cb(frac: float, label: str) -> None:
            _emit(progress_cb, 74 + 25 * max(0.0, min(1.0, frac)), label)

        report_path = step3_generate_report(
            ranked, db, mode=mode,  # 传真实模式：keyword 报告落 _KEY 专用文件名
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


# ══════════════════════════════════════════════════════════════
# Agent 复合工具 handler(三榜共用):幂等确认守卫 + 执行 run_ranking
# ══════════════════════════════════════════════════════════════

_MODE_LABEL = {"comprehensive": "综合热榜", "hot_new": "新项目热榜", "keyword": "关键词热榜"}


def _confirm_sig(mode: str, params: dict) -> str:
    return json.dumps({"mode": mode, **params}, ensure_ascii=False, sort_keys=True, default=str)


def _format_confirm(mode: str, params: dict) -> str:
    """把实际生效的参数原样拼成一句确认文案（展示=执行，不经模型转述、不漏参数）。"""
    label = _MODE_LABEL.get(mode, mode)
    bits: list[str] = []
    kws = params.get("keywords")
    if kws:
        shown = "、".join(kws[:6]) + ("…" if len(kws) > 6 else "")
        bits.append(f"关键词 {len(kws)} 个（{shown}）")
    if params.get("categories"):
        bits.append("类别：" + "、".join(params["categories"]))
    if params.get("topic"):
        bits.append(f"方向：{params['topic']}")
    if params.get("top_n") is not None:
        bits.append(f"Top {params['top_n']}")
    if params.get("min_star") is not None:
        bits.append(f"最低 star={params['min_star']}")
    gt = params.get("growth_threshold")
    if gt is not None:
        bits.append(f"增长阈值={gt}" + ("（不过滤增长）" if gt == 0 else "（近窗口需涨够该 star 才入选）"))
    gcd = params.get("growth_calc_days")
    bits.append(f"增长窗口={gcd}天" if gcd else "增长窗口=默认(按数据库年龄，通常近几天)")
    if params.get("days_since_created") is not None:
        bits.append(f"新项目创建窗口={params['days_since_created']}天")
    if mode != "keyword" and params.get("max_star") is not None:
        bits.append(f"星段上限={params['max_star']}")
    bits.append("生成报告文件（较慢，逐个项目写介绍）" if params.get("generate_report")
                else "不生成报告文件（榜单直接在对话里给）")
    return (f"将执行【{label}】，参数：" + "；".join(bits)
            + "。确认无误请回复『开始』；要改参数（如降低阈值、改关键词）直接说。")


def make_ranking_handler(mode: str):
    """构造某个榜单模式的 Agent 工具 handler。

    确定性确认守卫：首次调用（或参数变化）记录待确认参数并返回"请确认"，由 agent 层直接把
    _format_confirm 的完整参数回显给用户（不经模型转述）。用户确认后模型带 confirm=true 再次
    调用 → 按「首次存下的参数」执行（展示=执行，杜绝二次调用时的参数漂移与重复确认）。
    兼容旧路径：不带 confirm 但以相同签名复调，同样视为确认执行。榜单缓存挂在会话 tool_state。
    """
    label = _MODE_LABEL.get(mode, mode)

    def handler(ctx, args: dict) -> dict:
        params = dict(args)
        confirm = bool(params.pop("confirm", False))
        sig = _confirm_sig(mode, params)

        pending_sig = ctx.state.pending_confirmation_signature
        stored = ctx.state.tool_state.get("pending_ranking") or {}
        # 是否为「确认执行」：有待确认项，且（用户明确 confirm / 或以相同签名复调）
        is_confirm = bool(pending_sig) and (confirm or pending_sig == sig)

        if not is_confirm:
            ctx.state.pending_confirmation_signature = sig
            ctx.state.tool_state["pending_ranking"] = {"mode": mode, "sig": sig, "params": params}
            return {
                "needs_confirmation": True,
                "mode": mode,
                "params": params,
                "message": _format_confirm(mode, params),
            }

        # 执行：优先用「首次确认时存下的参数」，确保与回显完全一致（confirm 复调时模型可能漂移）
        exec_params = stored["params"] if stored.get("mode") == mode and "params" in stored else params
        ctx.state.pending_confirmation_signature = None
        ctx.state.tool_state.pop("pending_ranking", None)
        params = exec_params

        cache = ctx.state.tool_state.get("ranking_cache")
        if cache is None:
            cache = RankingCache()
            ctx.state.tool_state["ranking_cache"] = cache

        # 报告是按需产物：默认只回榜单，避免每次找项目都为 Top N 逐个跑 LLM 写介绍
        do_report = bool(params.pop("generate_report", False))
        result = run_ranking(
            ctx.provider, mode=mode, params=params, db=ctx.db,
            cache=cache, do_report=do_report, force_refresh=False,
            progress_cb=getattr(ctx, "progress_cb", None),
        )
        save_db_desc_only(ctx.db)  # Agent 路径:仅持久化 desc 字段

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
