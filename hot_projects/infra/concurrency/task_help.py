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

from ...config import (
    CHECKPOINT_FILE_PATH,
    STAR_GROWTH_THRESHOLD,
    GROWTH_CALC_DAYS,
    DB_DIFF_TOLERANCE_HOURS,
)
from ..db import (
    get_db_age_days,
    is_project_window_match,
)
from ...providers.github.token_pool import GitHubTokenPool

logger = logging.getLogger("discover_hot")


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

    if allow_diff:
        for full_name in list(pending.keys()):
            prev = prev_snapshot.get(full_name)
            if not prev:
                continue
            saved_star = prev.get("star")
            if saved_star is None:
                continue
            if not is_project_window_match(
                prev.get("refreshed_at", ""), time_window, DB_DIFF_TOLERANCE_HOURS
            ):
                continue

            info = pending[full_name]
            current_star = info["star"]
            created_at = info.get("created_at", "")
            growth = current_star - saved_star
            if use_checkpoint:
                checkpoint[full_name] = {"growth": growth, "star": current_star}
                checkpoint_dirty = True
            db_count += 1
            if growth >= growth_threshold:
                _upsert_candidate(candidate_map, full_name, growth, current_star, created_at, "DB",
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

    db_age_info = f"(距上次更新≈{db_age}天)" if db_age is not None else ""
    logger.info(
        f"批量增长计算: {len(pending)} 个任务入队 "
        f"(DB差值{db_age_info} {db_count}, 续传 {resumed_count}, "
        f"跳过已入选 {len(raw_repos) - len(pending) - db_count - resumed_count})"
    )

    return checkpoint
