"""
评分排序（tools/basic · 评分组件）
================================
对候选仓库进行评分排序，支持两种模式。

架构定位：
  tools/basic 底层组件，step2_rank_and_select() 由 ranking 编排直接调用。

评分模式：
  1. comprehensive（综合排名）
     base  = (log(1+growth) * 1000 + log2(1+rate) * 3000) * discount
     score = base * burst_boost
     • growth = 绝对增长量（窗口内），反映项目热度
     • rate   = growth / star，反映爆发力
     • discount：rate > 0.5 时线性折扣（最多 -15%），抑制低基数项目虚高
     • burst_boost：最近 RECENT_GROWTH_DAYS 天速率显著高于整窗平均时给乘法加成，
       让最近几天爆火的项目排名更高（候选未带 recent_growth 时 boost=1）
  2. hot_new（新项目专榜）
     • 仅保留创建时间 <= days_since_created 的仓库
     • 如果候选池已在 batch_check_growth 阶段按相同窗口预筛，直接按增长量降序
     • 否则从 DB 补充 created_at 后再筛选
"""

import logging
import math
from datetime import datetime, timezone

from ...config import (
    DEFAULT_SCORE_MODE,
    DAYS_SINCE_CREATED,
    GROWTH_CALC_DAYS,
    RECENT_GROWTH_DAYS,
    BURST_ALPHA,
    BURST_CAP,
)

logger = logging.getLogger("hot_projects")


def _hydrate_candidate_created_at(
    candidate_map: dict[str, dict],
    db: dict | None,
) -> None:
    """为缺失 created_at 的候选从 DB 补充创建时间。

    created_at 的 API 补全已在 tool_batch_check_growth 初筛阶段完成（写入内存候选；
    仅 force_refresh 路径才落 DB 快照）；此处查 DB 作为二次兜底
    （如 comprehensive 搜索后切 hot_new 排名）。
    """
    if not candidate_map:
        return

    db_projects = db.get("projects", {}) if db else {}

    for full_name, info in candidate_map.items():
        if info.get("created_at"):
            continue

        db_created_at = db_projects.get(full_name, {}).get("created_at", "")
        if db_created_at:
            info["created_at"] = db_created_at


def step2_rank_and_select(
    candidate_map: dict[str, dict],
    mode: str = DEFAULT_SCORE_MODE,
    db: dict | None = None,
    days_since_created: int | None = None,
    prefiltered_days_since_created: int | None = None,
    growth_calc_days: int = GROWTH_CALC_DAYS,
    recent_growth_days: int = RECENT_GROWTH_DAYS,
    burst_alpha: float = BURST_ALPHA,
    burst_cap: float = BURST_CAP,
) -> list[tuple[str, dict]]:
    """
    评分排序 + 截取 Top N。

    评分模式：
      comprehensive — 综合排名：基础分 = (log(1+增长量)·1000 + log2(1+增长率)·3000) × 折扣；
                      再叠加"最近爆发加成"——若候选带 recent_growth（最近 recent_growth_days
                      天增长），最近速率显著高于整窗平均时给乘法加成 boost，
                      使最近几天爆火的项目排名更高（详见 _burst_boost）。
      hot_new       — 新项目专榜：候选池已预筛时直接按增长量排序，否则兜底按创建时间过滤。

    Returns:
        [(full_name, {"growth": int, "star": int, ...}), ...] 按 score 降序，返回全部排序结果。
    """
    _days_created = days_since_created if days_since_created is not None else DAYS_SINCE_CREATED

    def _burst_boost(item: dict) -> float:
        """最近爆发加成（乘法、封顶、不反向惩罚）。

        recent_growth 缺失/无效时返回 1.0，退化为纯基础分排序——保证未做 recent 探针的
        链路（如 Agent 直接排序）完全兼容。
        """
        recent = item.get("recent_growth")
        g = item.get("growth", 0)
        if (
            recent is None or recent < 0 or g <= 0
            or growth_calc_days <= 0 or recent_growth_days <= 0
        ):
            return 1.0
        avg_rate = g / growth_calc_days
        if avg_rate <= 0:
            return 1.0
        recent_rate = recent / recent_growth_days
        acceleration = recent_rate / avg_rate
        burst = min(max(acceleration - 1.0, 0.0), burst_cap)
        return 1.0 + burst_alpha * burst

    def _calc_score(item: dict) -> float:
        g = item["growth"]
        s = item["star"]

        if s <= 0:
            return float(g)

        growth_score = math.log(1 + g) * 1000
        rate = g / s
        rate_score = math.log(1 + rate) / math.log(2) * 3000

        if mode == "comprehensive":
            if rate > 0.5:
                discount = 1.0 - 0.15 * min((rate - 0.5) / 0.5, 1.0)
            else:
                discount = 1.0
            return (growth_score + rate_score) * discount * _burst_boost(item)
        else:
            return float(g)

    def _is_new_project(info: dict) -> bool:
        created_at = info.get("created_at", "")
        if not created_at:
            return False
        try:
            created_date = datetime.strptime(
                created_at[:10], "%Y-%m-%d"
            ).replace(tzinfo=timezone.utc)
            days_since = (datetime.now(timezone.utc) - created_date).days
            return days_since <= _days_created
        except (ValueError, TypeError):
            return False

    if mode == "hot_new":
        if prefiltered_days_since_created == _days_created:
            sorted_candidates = sorted(
                candidate_map.items(),
                key=lambda x: x[1]["growth"],
                reverse=True,
            )
            logger.info(
                f"Step 2 (hot_new): 候选池已前置筛选(<={_days_created}天)，"
                f"直接按增长量排序 {len(candidate_map)} 个。"
            )
        else:
            _hydrate_candidate_created_at(candidate_map, db)
            new_projects = {
                name: info for name, info in candidate_map.items()
                if _is_new_project(info)
            }
            sorted_candidates = sorted(
                new_projects.items(),
                key=lambda x: x[1]["growth"],
                reverse=True,
            )
            logger.info(
                f"Step 2 (hot_new): 兜底按新项目窗口(<={_days_created}天)过滤后，"
                f"保留 {len(new_projects)} 个。"
            )
    else:
        sorted_candidates = sorted(
            candidate_map.items(), key=lambda x: _calc_score(x[1]), reverse=True
        )
        logger.info(
            f"Step 2 (comprehensive): 候选 {len(candidate_map)} 个。"
        )

    logger.info("  Top 10 预览:")
    for i, (name, info) in enumerate(sorted_candidates[:10], 1):
        score = _calc_score(info)
        info["_score"] = score
        recent = info.get("recent_growth")
        recent_txt = f", 近{recent_growth_days}天+{recent}" if recent is not None else ""
        boost_txt = f", boost={_burst_boost(info):.2f}" if recent is not None else ""
        logger.info(
            f"    {i}. {name} (+{info['growth']}{recent_txt}, star={info['star']}"
            f"{boost_txt}, score={score:.0f})"
        )

    # 补充其余候选的 score
    for name, info in sorted_candidates[10:]:
        info["_score"] = _calc_score(info)

    return sorted_candidates
