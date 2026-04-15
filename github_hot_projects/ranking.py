"""
评分排序
========
对候选仓库进行评分排序，支持 comprehensive（综合）和 hot_new（新项目专榜）两种模式。

原 scorer.py，重命名为 ranking.py 以更准确反映职责。
"""

import logging
import math
from datetime import datetime, timezone

from .common.config import (
    DEFAULT_SCORE_MODE,
    NEW_PROJECT_DAYS,
)

logger = logging.getLogger("discover_hot")


def _hydrate_candidate_created_at(
    candidate_map: dict[str, dict],
    db: dict | None,
) -> None:
    """为缺失 created_at 的候选从 DB 补充创建时间。

    API 补全已在 tool_batch_check_growth 初筛阶段完成并存入 DB，
    此处仅查 DB 作为二次兜底（如 comprehensive 搜索后切 hot_new 排名）。
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
    new_project_days: int | None = None,
    prefiltered_new_project_days: int | None = None,
) -> list[tuple[str, dict]]:
    """
    评分排序 + 截取 Top N。

    评分模式：
      comprehensive — 综合排名：log(增长量) + log(增长率)，新项目平滑折扣
    hot_new       — 新项目专榜：候选池已预筛时直接按增长量排序，否则兜底按创建时间过滤

    Returns:
        [(full_name, {"growth": int, "star": int, ...}), ...] 按 score 降序，返回全部排序结果。
    """
    _new_days = new_project_days if new_project_days is not None else NEW_PROJECT_DAYS

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
            return (growth_score + rate_score) * discount
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
            return days_since <= _new_days
        except (ValueError, TypeError):
            return False

    if mode == "hot_new":
        if prefiltered_new_project_days == _new_days:
            sorted_candidates = sorted(
                candidate_map.items(),
                key=lambda x: x[1]["growth"],
                reverse=True,
            )
            logger.info(
                f"Step 2 (hot_new): 候选池已前置筛选(<={_new_days}天)，"
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
                f"Step 2 (hot_new): 兜底按新项目窗口(<={_new_days}天)过滤后，"
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
        logger.info(
            f"    {i}. {name} (+{info['growth']}, star={info['star']}, score={score:.0f})"
        )

    # 补充其余候选的 score
    for name, info in sorted_candidates[10:]:
        info["_score"] = _calc_score(info)

    return sorted_candidates
