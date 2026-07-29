"""
Task 辅助函数
===============
承载 Task 相关的流程辅助逻辑：候选管理、checkpoint、批量增长任务提交。

说明：
  - 任务子类定义保留在 tasks.py。
  - 本模块不在顶层导入 CalcGrowthTask，避免与 tasks.py 形成循环导入。
"""

import json
import logging
import os
from typing import Any

from datetime import timedelta

from ...config import (
    CHECKPOINT_FILE_PATH,
    STAR_GROWTH_THRESHOLD,
    GROWTH_CALC_DAYS,
    DB_DIFF_TOLERANCE_HOURS,
    DB_DIFF_SCALE_MIN_RATIO,
    DB_DIFF_SCALE_MAX_RATIO,
    SNAPSHOT_ANCHOR_TOLERANCE_DAYS,
)
from ..db import (
    get_db_age_days,
    is_project_window_match,
    timestamp_age_days,
)
from ..snapshots import find_anchor, utc_today
from ...datasource.github.token_pool import GitHubTokenPool

logger = logging.getLogger("hot_projects")


def _upsert_candidate(
    candidate_map: dict[str, dict],
    full_name: str,
    growth: int,
    current_star: int,
    created_at: str = "",
    source: str = "",
    log_threshold: int | None = None,
) -> None:
    """更新或插入候选（取更大的 growth 值），保留 created_at。

    log_threshold：仅当 growth >= log_threshold 时打印 [OK] 候选 日志。
    None 表示不限制（始终打印）。候选池本身始终全量收录（供分阶段缓存复用），
    日志只展示达标候选，与旧项目"候选=达标"的打印语义保持一致。
    """
    existing = candidate_map.get(full_name)
    if existing:
        if growth > existing["growth"]:
            existing["growth"] = growth
            existing["star"] = current_star
        if created_at and not existing.get("created_at"):
            existing["created_at"] = created_at
    else:
        candidate_map[full_name] = {
            "growth": growth,
            "star": current_star,
            "created_at": created_at,
        }
        if log_threshold is None or growth >= log_threshold:
            tag = f"({source})" if source else ""
            logger.info(f"  [OK] 候选{tag}: {full_name} | growth={growth} | star={current_star}")


def _load_checkpoint() -> dict:
    """加载断点续传文件。返回 {full_name: {"growth": int | "unresolved", "star": int}} 或空字典。"""
    if not os.path.exists(CHECKPOINT_FILE_PATH):
        return {}
    try:
        with open(CHECKPOINT_FILE_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        logger.info(f"检测到断点续传文件: {len(data)} 个已计算项目。")
        return data
    except (json.JSONDecodeError, IOError):
        return {}


def _save_checkpoint(completed: dict) -> None:
    """增量保存已完成的增长计算结果到断点文件。"""
    try:
        temp_path = CHECKPOINT_FILE_PATH + ".tmp"
        with open(temp_path, "w", encoding="utf-8") as f:
            json.dump(completed, f, ensure_ascii=False)
        os.replace(temp_path, CHECKPOINT_FILE_PATH)
    except IOError as e:
        logger.warning(f"断点文件保存失败: {e}")


def _remove_checkpoint() -> None:
    """流程完整成功后删除断点文件。"""
    try:
        if os.path.exists(CHECKPOINT_FILE_PATH):
            os.remove(CHECKPOINT_FILE_PATH)
    except IOError:
        pass


def _load_growth_anchor(time_window: int) -> tuple[dict[str, int], int] | None:
    """取 T−time_window 那天的每日快照作为本轮锚点，全部仓库共用。

    一轮只挑一次：所有仓库因此落在同一个窗口上，相对排名不受个体刷新时间影响。
    没有可用快照（刚接入、还没攒够历史）返回 None，逐项回退到 DB 差值那几条路。

    Returns:
        (锚点 star 表, 锚点的实际天龄)。漏采几天时实际天龄会大于请求窗口，
        调用方必须据此改写 effective_growth_calc_days——否则 9 天的增长会被当成
        7 天来算速率，打分虚高 29%。
    """
    target = utc_today() - timedelta(days=time_window)
    found = find_anchor(target, SNAPSHOT_ANCHOR_TOLERANCE_DAYS)
    if found is None:
        logger.info(
            "每日快照：找不到 %s ±%d 天的锚点，本轮回退 DB 差值路径。",
            target, SNAPSHOT_ANCHOR_TOLERANCE_DAYS,
        )
        return None

    anchor_day, stars = found
    actual_window = (utc_today() - anchor_day).days
    logger.info(
        "每日快照锚点: %s（含 %d 个仓库，实际窗口 %d 天，请求 %d 天）。",
        anchor_day, len(stars), actual_window, time_window,
    )
    if actual_window != time_window:
        logger.warning(
            "锚点日期偏离请求窗口 %+d 天（漏采导致），本轮按实际 %d 天口径统计："
            "全部仓库共用同一锚点，窗口一致、相对排名不受影响，但报告里的增长是 %d 天的。",
            actual_window - time_window, actual_window, actual_window,
        )
    return stars, actual_window


def _resolve_growth_without_timestamps(
    full_name: str,
    current_star: int,
    created_at: str,
    prev: dict | None,
    time_window: int,
    anchor_stars: dict[str, int] | None = None,
) -> tuple[int, str] | None:
    """不发任何请求就能定下来的窗口增长，按精确度递降四条：

      1. 快照锚点   — T−N 那天的每日快照里有这个仓库 → current − 锚点 star，精确。
                     全部仓库共用同一份锚点，窗口长度一致，是唯一不受个体刷新时间影响的路径。
      2. DB 差值   — 快照年龄 ≈ 窗口（±5h）→ current − 旧 star，精确。
      3. 窗口内新建 — 仓库创建于窗口内 → 全部 star 都是窗口内涨的，精确。
      4. DB 折算   — 快照年龄在窗口的 [0.4, 3.0] 倍内 → 按年龄线性折算到窗口，近似。

    第 2~4 条是 GitHub 2026-06-30 把 stargazers 列表限权给 admin/collaborator 后的过渡方案：
    原先窗口不匹配的项目会落到二分法/采样外推，而那两条路对他人仓库已不可用。
    每日快照攒够一个窗口后，第 1 条会接管绝大多数仓库，第 4 条的近似可以退役。
    返回 None 表示无法在本地定下来，交由调用方走实时估算。
    """
    if anchor_stars:
        anchor_star = anchor_stars.get(full_name)
        if anchor_star is not None:
            return current_star - anchor_star, "快照"

    saved_star = prev.get("star") if prev else None
    refreshed_at = prev.get("refreshed_at", "") if prev else ""

    if saved_star is not None and is_project_window_match(
        refreshed_at, time_window, DB_DIFF_TOLERANCE_HOURS
    ):
        return current_star - saved_star, "DB"

    repo_age = timestamp_age_days(created_at)
    if repo_age is not None and repo_age <= time_window:
        return current_star, "窗口内新建"

    if saved_star is None:
        return None
    snapshot_age = timestamp_age_days(refreshed_at)
    if snapshot_age is None or snapshot_age <= 0:
        return None
    if not (
        time_window * DB_DIFF_SCALE_MIN_RATIO
        <= snapshot_age
        <= time_window * DB_DIFF_SCALE_MAX_RATIO
    ):
        return None

    delta = current_star - saved_star
    if delta <= 0:
        # 掉星/持平无需折算：放大负增长只会制造假象，按原值报即可。
        return delta, "DB折算"
    return round(delta * time_window / snapshot_age), "DB折算"


def _submit_growth_tasks(
    pool: Any,
    token_mgr: GitHubTokenPool,
    raw_repos: dict[str, dict],
    db: dict,
    candidate_map: dict[str, dict],
    growth_ctx: dict,
) -> dict:
    """
    批量增长计算入队：默认先走 checkpoint/DB 差值，再将剩余提交为 CalcGrowthTask。

    growth_ctx 由调用方创建并传入（包含 checkpoint, pending_created_at, db_projects 等共享状态）。
    返回 checkpoint dict。
    """
    from .tasks import CalcGrowthTask

    db_projects = db.get("projects", {})
    growth_threshold = growth_ctx.get("growth_threshold", STAR_GROWTH_THRESHOLD)
    log_threshold = growth_ctx.get("candidate_log_threshold", growth_threshold)
    use_realtime_growth = bool(growth_ctx.get("use_realtime_growth", False))
    can_write_db = bool(growth_ctx.get("can_write_db", False))
    use_checkpoint = bool(growth_ctx.get("use_checkpoint", not use_realtime_growth))

    checkpoint = {} if not use_checkpoint else _load_checkpoint()
    growth_ctx["checkpoint"] = checkpoint

    pending = {
        fn: info for fn, info in raw_repos.items()
        if fn not in candidate_map
    }

    resumed_count = 0
    if use_checkpoint:
        # 从 checkpoint 恢复
        for fn in list(pending.keys()):
            if fn in checkpoint:
                cp = checkpoint[fn]
                growth = cp["growth"]
                # 跳过上轮标记为 unresolved 的仓库（不重复估算，直接从 pending 移除）
                if growth == "unresolved":
                    del pending[fn]
                    resumed_count += 1
                    continue
                current_star = cp["star"]
                created_at = pending[fn].get("created_at", "")
                if growth >= growth_threshold:
                    _upsert_candidate(candidate_map, fn, growth, current_star, created_at, "checkpoint",
                                      log_threshold=log_threshold)
                del pending[fn]
                resumed_count += 1

        if resumed_count:
            logger.info(f"断点续传: 恢复 {resumed_count} 个已计算项目。")

    # DB 差值法：主线程直接处理（模式感知）
    checkpoint_dirty = False
    db_count = 0

    time_window = growth_ctx.get("growth_calc_days", GROWTH_CALC_DAYS)
    window_specified = bool(growth_ctx.get("window_specified", True))
    is_hot_new = bool(growth_ctx.get("is_hot_new", False))
    db_age = get_db_age_days(db)

    # 综合榜未指定窗口时，自动采用 DB 年龄窗口
    if not is_hot_new and not window_specified:
        if db_age is not None and db_age > 0:
            time_window = db_age
            growth_ctx["growth_calc_days"] = time_window
            logger.info(f"综合榜未指定窗口：本轮自动采用 DB 年龄窗口 {time_window} 天。")

    growth_ctx["effective_growth_calc_days"] = time_window

    # ── DB 差值（两层判定）──
    # 第一层（计算窗口 time_window）已在上方确定（Agent 未指定=DB年龄/指定=用户值；
    # 定时=调用方传入并已取 max(指定,默认)）。
    # 第二层（项目级）：项目在 DB 里 + |项目年龄 − time_window| ≤ 容差(5h) → 走差值，
    # 用 seeding 覆盖前捕获的旧快照(prev_snapshot) 计算 current_star − 旧star。
    # 仅新项目榜(is_hot_new) 整体不走差值、全部实时；综合榜按项目级窗口匹配逐项决定。
    prev_snapshot = growth_ctx.get("prev_snapshot", {}) or {}
    # 差值有效性改为「逐项判定」：仅当项目快照 refreshed_at 与本次窗口相差 ≤ 5h 才走差值
    # （见下方 is_project_window_match）。不再依赖由静态 DATA_EXPIRE_DAYS 驱动的 db["valid"]
    # 粗闸——否则 growth_calc_days >= 默认窗口+1 时整库会被误判过期、全量实时（D1）。
    # is_project_window_match 对过旧/过新的快照会自动回退实时，故无需粗闸兜底。
    # 新项目榜(is_hot_new) 仍全部实时。
    allow_diff = not is_hot_new

    scaled_count = 0
    fresh_repo_count = 0
    anchor_count = 0
    anchor_stars = None
    if allow_diff:
        anchor = _load_growth_anchor(time_window)
        if anchor is not None:
            anchor_stars, anchor_window = anchor
            # 锚点顺延（漏采）时窗口会变长，统计口径必须跟着改，否则速率按 7 天算而增长是 9 天的。
            growth_ctx["effective_growth_calc_days"] = anchor_window

        for full_name in list(pending.keys()):
            info = pending[full_name]
            current_star = info["star"]
            created_at = info.get("created_at", "")
            resolved = _resolve_growth_without_timestamps(
                full_name, current_star, created_at,
                prev_snapshot.get(full_name), time_window, anchor_stars,
            )
            if resolved is None:
                continue

            growth, source = resolved
            if source == "快照":
                anchor_count += 1
            elif source == "DB折算":
                scaled_count += 1
            elif source == "窗口内新建":
                fresh_repo_count += 1
            if use_checkpoint:
                checkpoint[full_name] = {"growth": growth, "star": current_star}
                checkpoint_dirty = True
            db_count += 1
            if growth >= growth_threshold:
                _upsert_candidate(candidate_map, full_name, growth, current_star, created_at, source,
                                  log_threshold=log_threshold)
            del pending[full_name]

    if use_realtime_growth:
        logger.info("新项目榜：跳过 DB 差值，全部走实时增长估算。")

    if use_checkpoint and checkpoint_dirty:
        _save_checkpoint(checkpoint)

    # 非 DB 差值：提交 CalcGrowthTask 到池子
    pending_created_at = growth_ctx["pending_created_at"]
    for full_name, info in pending.items():
        pending_created_at[full_name] = info.get("created_at", "")
        pool.submit(CalcGrowthTask(
            _token_mgr=token_mgr,
            full_name=full_name,
            current_star=info["star"],
            repo_item=info["repo_item"],
            _ctx=growth_ctx,
        ))

    growth_ctx["db_diff_count"] = db_count
    growth_ctx["realtime_count"] = len(pending)
    growth_ctx["resumed_count"] = resumed_count

    db_age_info = f"(距上次更新≈{db_age}天)" if db_age is not None else ""
    logger.info(
        f"批量增长计算: {len(pending)} 个任务入队 "
        f"(本地定案{db_age_info} {db_count}，其中每日快照锚点 {anchor_count}、"
        f"窗口内新建 {fresh_repo_count}、按快照年龄折算 {scaled_count}, 续传 {resumed_count}, "
        f"跳过已入选 {len(raw_repos) - len(pending) - db_count - resumed_count})"
    )

    return checkpoint
