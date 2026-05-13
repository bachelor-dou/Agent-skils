"""
Task 辅助函数
===============
承载 Task 相关的流程辅助逻辑：候选管理、checkpoint、批量增长任务提交。

说明：
  - 任务子类定义保留在 task.py。
  - 本模块不在顶层导入 CalcGrowthTask，避免与 task.py 形成循环导入。
"""

import json
import logging
import os
from datetime import datetime, timezone
from typing import Any

from ..common.config import (
    CHECKPOINT_FILE_PATH,
    STAR_GROWTH_THRESHOLD,
    GROWTH_CALC_DAYS,
)
from ..common.db import (
    is_project_same_batch,
    get_db_age_days,
)
from ..common.async_token_pool import GitHubTokenPool

logger = logging.getLogger("discover_hot")


def _upsert_candidate(
    candidate_map: dict[str, dict],
    full_name: str,
    growth: int,
    current_star: int,
    created_at: str = "",
    source: str = "",
) -> None:
    """更新或插入候选（取更大的 growth 值），保留 created_at。"""
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
        tag = f"({source})" if source else ""
        logger.info(f"  [OK] 候选{tag}: {full_name} | growth={growth} | star={current_star}")


def _load_checkpoint() -> dict:
    """加载断点续传文件。返回 {full_name: {"growth": int, "star": int}} 或空字典。"""
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


def _project_refresh_age_days(project: dict) -> int | None:
    """返回仓库 refreshed_at 距今的天数（按 UTC 日期差），无有效值返回 None。"""
    refreshed_at = project.get("refreshed_at", "")
    if not refreshed_at:
        return None
    try:
        refresh_dt = datetime.strptime(refreshed_at, "%Y-%m-%dT%H:%M:%SZ").replace(
            tzinfo=timezone.utc
        )
        return (datetime.now(timezone.utc).date() - refresh_dt.date()).days
    except ValueError:
        return None


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
    from .task import CalcGrowthTask

    db_projects = db.get("projects", {})
    growth_threshold = growth_ctx.get("growth_threshold", STAR_GROWTH_THRESHOLD)
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
                    _upsert_candidate(candidate_map, fn, growth, current_star, created_at, "checkpoint")
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

    # 判断能否走 DB 差值
    if is_hot_new:
        # 新项目榜：始终实时计算，不走 DB 差值
        can_use_db_diff = False
    else:
        # 综合榜：根据窗口匹配判断
        if window_specified:
            can_use_db_diff = bool(
                db.get("valid", False)
                and db_age is not None
                and db_age == time_window
            )
        else:
            can_use_db_diff = bool(db.get("valid", False))

    # 只在允许 DB 差值且非实时模式时，才尝试走 DB 差值
    if can_use_db_diff and not use_realtime_growth:
        for full_name in list(pending.keys()):
            info = pending[full_name]
            current_star = info["star"]
            created_at = info.get("created_at", "")

            if full_name in db_projects:
                project_age = _project_refresh_age_days(db_projects[full_name])
                project_ok = (
                    project_age == time_window
                    and is_project_same_batch(db_projects[full_name], db)
                )

                if project_ok:
                    saved_star = db_projects[full_name].get("star", 0)
                    growth = current_star - saved_star
                    checkpoint[full_name] = {"growth": growth, "star": current_star}
                    checkpoint_dirty = True
                    db_count += 1
                    if growth >= growth_threshold:
                        _upsert_candidate(candidate_map, full_name, growth, current_star, created_at, "DB")
                    del pending[full_name]

    if use_realtime_growth:
        logger.info("实时计算模式：跳过 checkpoint 和 DB 差值，全部走实时增长估算。")

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
