"""
Task 子类与数据收集辅助
========================
定义搜索 / 扫描 / 增长计算的 Task 子类，以及批量提交、断点续传等辅助函数。

Task 子类（继承 task_base.Task）由 agent_tools 中的 Tool 函数创建并提交到 Pool。

包含：
  - KeywordSearchTask   — 关键词搜索任务
  - ScanSegmentTask     — Star 区间扫描任务
  - CalcGrowthTask      — 增长计算任务
  - _submit_growth_tasks — 批量增长计算入队
  - _upsert_candidate    — 候选更新/插入
  - checkpoint 函数      — 断点续传
"""

import json
import logging
import os
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone

from ..common.config import (
    CHECKPOINT_FILE_PATH,
    MIN_STAR_FILTER,
    SEARCH_REQUEST_INTERVAL,
    STAR_GROWTH_THRESHOLD,
    TIME_WINDOW_DAYS,
)
from ..common.db import (
    update_db_project,
    is_project_same_batch,
    get_db_age_days,
)
from ..common.github_api import search_github_repos
from ..growth_estimator import (
    GROWTH_ESTIMATION_UNRESOLVED,
    estimate_star_growth_binary,
)
from ..common.token_manager import TokenManager
from .task_base import Task
from .worker_pool import TokenWorkerPool

logger = logging.getLogger("discover_hot")

CHECKPOINT_BATCH_SIZE = 10  # checkpoint 批量落盘阈值


# ══════════════════════════════════════════════════════════════
# 候选管理
# ══════════════════════════════════════════════════════════════


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


# ══════════════════════════════════════════════════════════════
# 断点续传
# ══════════════════════════════════════════════════════════════


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
    """返回仓库 refreshed_at 距今的天数（四舍五入），无有效值返回 None。"""
    refreshed_at = project.get("refreshed_at", "")
    if not refreshed_at:
        return None
    try:
        refresh_dt = datetime.strptime(refreshed_at, "%Y-%m-%dT%H:%M:%SZ").replace(
            tzinfo=timezone.utc
        )
        return round((datetime.now(timezone.utc) - refresh_dt).total_seconds() / 86400)
    except ValueError:
        return None


# ══════════════════════════════════════════════════════════════
# Task 子类定义（搜索 / 扫描 / 增长计算）
# ══════════════════════════════════════════════════════════════


@dataclass
class KeywordSearchTask(Task):
    """关键词搜索任务：搜索单个关键词的多页结果。"""

    needs_token: bool = True
    keyword: str = ""
    category: str = ""
    keyword_idx: int = 0
    total_keywords: int = 0
    max_pages: int = 3
    created_after: str = ""
    project_min_star_override: int = 0
    _raw_repos: dict = field(default=None, repr=False)

    def execute(self, token_idx: int | None) -> list[dict]:
        worker_suffix = f", worker={token_idx}" if token_idx is not None else ""
        logger.info(
            f"[{self.keyword_idx}/{self.total_keywords}] 搜索: "
            f"'{self.keyword}' (类别: {self.category}{worker_suffix})"
        )
        collected: list[dict] = []
        project_min_star = self.project_min_star_override if self.project_min_star_override else MIN_STAR_FILTER
        query = self.keyword
        if self.created_after:
            query = f"{query} created:>={self.created_after}"

        for page in range(1, self.max_pages + 1):
            items = search_github_repos(
                self._token_mgr, query, token_idx, page=page, worker_idx=token_idx
            )
            if items is None:
                continue
            if not items:
                break
            for repo_item in items:
                full_name = repo_item.get("full_name", "")
                if not full_name:
                    continue
                current_star = repo_item.get("stargazers_count", 0)
                if current_star < project_min_star:
                    continue
                collected.append({
                    "full_name": full_name,
                    "star": current_star,
                    "repo_item": repo_item,
                    "created_at": repo_item.get("created_at", ""),
                })
            time.sleep(SEARCH_REQUEST_INTERVAL)

        return collected

    def on_result(self, result: list[dict]) -> None:
        if not result or self._raw_repos is None:
            return
        for repo in result:
            fn = repo["full_name"]
            if fn not in self._raw_repos:
                self._raw_repos[fn] = {
                    "star": repo["star"],
                    "repo_item": repo["repo_item"],
                    "created_at": repo["created_at"],
                }

    def __str__(self) -> str:
        return f"KeywordSearch({self.keyword})"


@dataclass
class ScanSegmentTask(Task):
    """Star 区间扫描任务：扫描单个子区间的多页结果。"""

    needs_token: bool = True
    seg_idx: int = 0
    low: int = 0
    high: int = 0
    total_segments: int = 0
    created_after: str = ""
    min_star_filter_override: int = 0
    page_numbers: list[int] | None = None
    retry_round: int = 0
    _raw_repos: dict = field(default=None, repr=False)
    failed_pages: list[int] = field(default_factory=list, init=False, repr=False)

    def execute(self, token_idx: int | None) -> list[dict]:
        self.failed_pages = []
        query = f"stars:{self.low}..{self.high}"
        if self.created_after:
            query = f"{query} created:>={self.created_after}"
        worker_suffix = f" (worker={token_idx})" if token_idx is not None else ""
        retry_suffix = f", retry={self.retry_round}" if self.retry_round else ""
        page_suffix = f", pages={self.page_numbers}" if self.page_numbers else ""
        logger.info(
            f"  子区间 {self.seg_idx}/{self.total_segments}: "
            f"{query}{worker_suffix}{retry_suffix}{page_suffix}"
        )
        collected: list[dict] = []
        min_star_filter = self.min_star_filter_override if self.min_star_filter_override else MIN_STAR_FILTER
        pages = self.page_numbers if self.page_numbers is not None else list(range(1, 11))
        stop_on_empty = self.page_numbers is None

        for page in pages:
            items = search_github_repos(
                self._token_mgr, query, token_idx,
                page=page, sort="updated", auto_star_filter=False, worker_idx=token_idx,
            )
            if items is None:
                self.failed_pages.append(page)
                worker_suffix = f", worker={token_idx}" if token_idx is not None else ""
                failure_action = "加入补偿队列" if self.retry_round == 0 else "补偿后仍失败"
                logger.warning(
                    f"  子区间 {self.seg_idx}/{self.total_segments}: {query}, "
                    f"page={page}{worker_suffix} 连续失败，{failure_action}。"
                )
                continue
            if not items:
                if stop_on_empty:
                    break
                continue
            for repo_item in items:
                full_name = repo_item.get("full_name", "")
                if not full_name:
                    continue
                current_star = repo_item.get("stargazers_count", 0)
                if current_star < min_star_filter:
                    continue
                collected.append({
                    "full_name": full_name,
                    "star": current_star,
                    "repo_item": repo_item,
                    "created_at": repo_item.get("created_at", ""),
                })
            time.sleep(SEARCH_REQUEST_INTERVAL)

        return collected

    def on_result(self, result: list[dict]) -> None:
        if not result or self._raw_repos is None:
            return
        for repo in result:
            fn = repo["full_name"]
            if fn not in self._raw_repos:
                self._raw_repos[fn] = {
                    "star": repo["star"],
                    "repo_item": repo["repo_item"],
                    "created_at": repo["created_at"],
                }

    def __str__(self) -> str:
        pages = f", pages={self.page_numbers}" if self.page_numbers is not None else ""
        retry = f", retry={self.retry_round}" if self.retry_round else ""
        return f"ScanSegment({self.low}..{self.high}{pages}{retry})"


@dataclass
class CalcGrowthTask(Task):
    """
    增长计算任务：计算单个仓库的窗口期 star 增长。

    _ctx 字典由调用方提供，包含：
      checkpoint, pending_created_at, db_projects, candidate_map,
      checkpoint_dirty (list[bool]), completed_since_save (list[int])
    """

    needs_token: bool = True
    full_name: str = ""
    current_star: int = 0
    repo_item: dict = field(default_factory=dict)
    _ctx: dict = field(default=None, repr=False)

    def execute(self, token_idx: int | None) -> tuple[str, int, int]:
        parts = self.full_name.split("/", 1)
        if len(parts) != 2:
            return self.full_name, -1, self.current_star
        owner, repo_name = parts
        logger.info(
            f"  [SEARCH] stargazers 查询: {self.full_name} (star={self.current_star})"
        )
        time_window_days = TIME_WINDOW_DAYS
        if self._ctx is not None:
            time_window_days = self._ctx.get("time_window_days", TIME_WINDOW_DAYS)
        growth = estimate_star_growth_binary(
            self._token_mgr, owner, repo_name, self.current_star,
            token_idx=token_idx,
            time_window_days=time_window_days,
        )
        if growth >= 0 and growth > self.current_star:
            growth = self.current_star
        return self.full_name, growth, self.current_star

    def on_result(self, result: tuple[str, int, int]) -> None:
        if self._ctx is None:
            return
        checkpoint = self._ctx["checkpoint"]
        db_projects = self._ctx["db_projects"]
        candidate_map = self._ctx["candidate_map"]
        pending_created_at = self._ctx["pending_created_at"]
        growth_threshold = self._ctx.get("growth_threshold", STAR_GROWTH_THRESHOLD)
        use_checkpoint = self._ctx.get("use_checkpoint", True)
        update_db = self._ctx.get("update_db", False)

        _, growth, current_star = result
        created_at = pending_created_at.get(self.full_name, "")

        if growth == GROWTH_ESTIMATION_UNRESOLVED:
            logger.warning(
                f"  增长估算未决: {self.full_name}，"
                "采样数据不足，标记为 unresolved 写入 checkpoint。"
            )
            unresolved_count = self._ctx.get("unresolved_count")
            if unresolved_count is not None:
                unresolved_count[0] += 1
            # 写入 checkpoint 标记 unresolved 状态，下次运行跳过而非重复估算
            if use_checkpoint:
                checkpoint[self.full_name] = {"growth": "unresolved", "star": current_star}
                self._ctx["checkpoint_dirty"][0] = True
            return

        if use_checkpoint:
            checkpoint[self.full_name] = {"growth": growth, "star": current_star}
            self._ctx["checkpoint_dirty"][0] = True
            self._ctx["completed_since_save"][0] += 1

        if growth >= 0:
            if update_db:
                update_db_project(db_projects, self.full_name, current_star, self.repo_item)
            if growth >= growth_threshold:
                _upsert_candidate(candidate_map, self.full_name, growth, current_star, created_at)

        if use_checkpoint and self._ctx["completed_since_save"][0] >= CHECKPOINT_BATCH_SIZE:
            _save_checkpoint(checkpoint)
            self._ctx["checkpoint_dirty"][0] = False
            self._ctx["completed_since_save"][0] = 0

    def on_error(self, error: Exception) -> None:
        if self._ctx is None:
            return
        logger.error(f"  增长计算异常: {self.full_name}, {error}")
        fallback_star = self.repo_item.get("stargazers_count", 0)
        use_checkpoint = self._ctx.get("use_checkpoint", True)
        if fallback_star:
            checkpoint = self._ctx["checkpoint"]
            if use_checkpoint:
                checkpoint[self.full_name] = {"growth": -1, "star": fallback_star}
                self._ctx["checkpoint_dirty"][0] = True
                self._ctx["completed_since_save"][0] += 1

    def __str__(self) -> str:
        return f"CalcGrowth({self.full_name})"


# ══════════════════════════════════════════════════════════════
# 批量增长计算入队
# ══════════════════════════════════════════════════════════════


def _submit_growth_tasks(
    pool: TokenWorkerPool,
    token_mgr: TokenManager,
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
    db_projects = db.get("projects", {})
    growth_threshold = growth_ctx.get("growth_threshold", STAR_GROWTH_THRESHOLD)
    force_refresh = bool(growth_ctx.get("force_refresh", False))
    use_checkpoint = bool(growth_ctx.get("use_checkpoint", not force_refresh))

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

    time_window = growth_ctx.get("time_window_days", TIME_WINDOW_DAYS)
    window_specified = bool(growth_ctx.get("window_specified", True))
    is_comprehensive = growth_ctx.get("new_project_days") is None
    db_age = get_db_age_days(db)

    if is_comprehensive and not window_specified:
        if db_age is not None and db_age > 0:
            time_window = db_age
            growth_ctx["time_window_days"] = time_window
            logger.info(f"综合榜未指定窗口：本轮自动采用 DB 年龄窗口 {time_window} 天。")

    growth_ctx["effective_time_window_days"] = time_window

    if is_comprehensive:
        if window_specified:
            can_use_db_diff = bool(
                db.get("valid", False)
                and db_age is not None
                and db_age == time_window
            )
        else:
            can_use_db_diff = bool(db.get("valid", False))
    else:
        # 新项目榜仅依赖仓库级 refreshed_at 与窗口精确匹配（不依赖全局 date）。
        can_use_db_diff = True

    if not force_refresh:
        for full_name in list(pending.keys()):
            info = pending[full_name]
            current_star = info["star"]
            created_at = info.get("created_at", "")

            if can_use_db_diff and full_name in db_projects:
                project_age = _project_refresh_age_days(db_projects[full_name])
                if is_comprehensive:
                    project_ok = (
                        project_age == time_window
                        and is_project_same_batch(db_projects[full_name], db)
                    )
                else:
                    project_ok = project_age == time_window

                if project_ok:
                    saved_star = db_projects[full_name].get("star", 0)
                    growth = current_star - saved_star
                    checkpoint[full_name] = {"growth": growth, "star": current_star}
                    checkpoint_dirty = True
                    db_count += 1
                    if growth >= growth_threshold:
                        _upsert_candidate(candidate_map, full_name, growth, current_star, created_at, "DB")
                    del pending[full_name]
    else:
        logger.info("强制刷新模式：跳过 checkpoint 和 DB 差值，全部走实时增长估算。")

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
