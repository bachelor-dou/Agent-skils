"""
核心流程编排模块
================
将搜索、增长计算、排序、报告生成串联为完整 pipeline。

流程架构：
  Phase 1 - 统一收集：关键词搜索 + Star 范围扫描 + GitHub Trending → 汇入同一 raw_repos map（去重 + star >= 1000）
  Phase 2 - 批量增长计算：对 raw_repos 中全部仓库并行计算窗口期增长 → 筛出候选
  Phase 3 - 评分排序 + 截取 Top N
  Phase 4 - LLM 描述 + 报告生成

并行模型：
  TokenWorkerPool（3 Worker 绑定 3 Token），Phase 1 和 Phase 2 复用同一个池。
  Worker 从共享任务队列取任务执行，限流自行 sleep，Token 失效退出。
  结果通过 result_queue 由主线程 wait_all_done 后单线程合并。

主要函数：
  - main()                         — 入口编排
  - collect_from_trending()        — GitHub Trending 收集（无 API，主线程直接执行）
  - step2_rank_and_select()        — 评分排序 + 截取 Top N
  - step3_generate_report()        — LLM 描述 + Markdown 报告
"""

import json
import logging
import os
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from .config import (
    CHECKPOINT_FILE_PATH,
    GITHUB_TOKENS,
    HOT_PROJECT_COUNT,
    LLM_API_URL,
    LLM_MODEL,
    MIN_STAR_FILTER,
    REPORT_DIR,
    DEFAULT_SCORE_MODE,
    SEARCH_REQUEST_INTERVAL,
    STAR_GROWTH_THRESHOLD,
    STAR_RANGE_MAX,
    STAR_RANGE_MIN,
    TIME_WINDOW_DAYS,
    DATA_EXPIRE_DAYS,
    RATE_LIMIT_WAIT_INTERVAL,
    SEARCH_KEYWORDS,
)
from .db import load_db, save_db, update_db_project
from .github_api import auto_split_star_range, search_github_repos
from .github_trending import fetch_trending
from .growth_estimator import estimate_star_growth_binary
from .llm import call_llm_describe
from .token_manager import TokenManager
from .worker_pool import Task, TokenWorkerPool

logger = logging.getLogger("discover_hot")

CHECKPOINT_BATCH_SIZE = 10  # checkpoint 批量落盘阈值


# ══════════════════════════════════════════════════════════════
# Pipeline Task 定义
# ══════════════════════════════════════════════════════════════


@dataclass
class KeywordSearchTask(Task):
    """关键词搜索任务：搜索单个关键词的多页结果。"""

    needs_token: bool = True
    keyword: str = ""
    category: str = ""
    keyword_idx: int = 0
    total_keywords: int = 0
    _raw_repos: dict = field(default=None, repr=False)

    def execute(self, token_idx: int | None) -> list[dict]:
        logger.info(
            f"[{self.keyword_idx}/{self.total_keywords}] 搜索: "
            f"'{self.keyword}' (类别: {self.category})"
        )
        collected: list[dict] = []

        for page in range(1, 4):
            items = search_github_repos(
                self._token_mgr, self.keyword, token_idx, page=page
            )
            if items is None:
                continue  # 网络失败，跳过该页
            if not items:
                break
            for repo_item in items:
                full_name = repo_item.get("full_name", "")
                if not full_name:
                    continue
                current_star = repo_item.get("stargazers_count", 0)
                if current_star < MIN_STAR_FILTER:
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
    _raw_repos: dict = field(default=None, repr=False)

    def execute(self, token_idx: int | None) -> list[dict]:
        query = f"stars:{self.low}..{self.high}"
        logger.info(f"  子区间 {self.seg_idx}/{self.total_segments}: {query}")
        collected: list[dict] = []

        for page in range(1, 11):
            items = search_github_repos(
                self._token_mgr, query, token_idx,
                page=page, sort="updated", auto_star_filter=False,
            )
            if items is None:
                continue  # 网络失败，跳过该页
            if not items:
                break
            for repo_item in items:
                full_name = repo_item.get("full_name", "")
                if not full_name:
                    continue
                current_star = repo_item.get("stargazers_count", 0)
                if current_star < MIN_STAR_FILTER:
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
        return f"ScanSegment({self.low}..{self.high})"


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
        growth = estimate_star_growth_binary(
            self._token_mgr, owner, repo_name, self.current_star,
            token_idx=token_idx,
        )
        if growth > self.current_star:
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

        _, growth, current_star = result
        created_at = pending_created_at.get(self.full_name, "")

        checkpoint[self.full_name] = {"growth": growth, "star": current_star}
        self._ctx["checkpoint_dirty"][0] = True
        self._ctx["completed_since_save"][0] += 1

        if growth >= 0:
            update_db_project(db_projects, self.full_name, current_star, self.repo_item)
            if growth >= growth_threshold:
                _upsert_candidate(candidate_map, self.full_name, growth, current_star, created_at)

        if self._ctx["completed_since_save"][0] >= CHECKPOINT_BATCH_SIZE:
            _save_checkpoint(checkpoint)
            self._ctx["checkpoint_dirty"][0] = False
            self._ctx["completed_since_save"][0] = 0

    def on_error(self, error: Exception) -> None:
        if self._ctx is None:
            return
        logger.error(f"  增长计算异常: {self.full_name}, {error}")
        fallback_star = self.repo_item.get("stargazers_count", 0)
        if fallback_star:
            db_projects = self._ctx["db_projects"]
            checkpoint = self._ctx["checkpoint"]
            update_db_project(db_projects, self.full_name, fallback_star, self.repo_item)
            checkpoint[self.full_name] = {"growth": -1, "star": fallback_star}
            self._ctx["checkpoint_dirty"][0] = True
            self._ctx["completed_since_save"][0] += 1

    def __str__(self) -> str:
        return f"CalcGrowth({self.full_name})"


# ══════════════════════════════════════════════════════════════
# Phase 1: Trending 收集（无需 Token，主线程直接执行）
# ══════════════════════════════════════════════════════════════


def collect_from_trending(
    raw_repos: dict[str, dict],
    candidate_map: dict[str, dict],
) -> None:
    """
    GitHub Trending 收集。

    策略：
      - daily/weekly 范围：爬取 Trending 页面获取 stars_today（实际为本日/本周增长星数）。
        对于 weekly 范围，stars_today 可直接作为增长指标 ——
        若 stars_today >= STAR_GROWTH_THRESHOLD，直接加入候选（无需查窗口期增长）。
      - monthly 范围：stars_today 为本月增长，粒度太粗，仍需走正常增长计算。

    Trending 仓库如果不满足直接入选条件，则加入 raw_repos 走后续批量增长计算。
    """
    logger.info("GitHub Trending 收集开始")

    trending_repos: list[dict] = []
    for since in ("daily", "weekly", "monthly"):
        repos = fetch_trending(since=since)
        trending_repos.extend(repos)

    # 去重（保留 stars_today 更大的）
    unique: dict[str, dict] = {}
    for r in trending_repos:
        fn = r["full_name"]
        if fn not in unique:
            unique[fn] = r
        else:
            if r.get("stars_today", 0) > unique[fn].get("stars_today", 0):
                unique[fn] = r

    direct_count = 0
    raw_count = 0

    for fn, r in unique.items():
        star = r.get("star", 0)
        if star < MIN_STAR_FILTER:
            continue

        stars_today = r.get("stars_today", 0)
        since = r.get("since", "")

        # weekly 范围：stars_today 是一周增长星数，与 10 天窗口期接近，可直接判断
        # daily 范围的 stars_today 只代表当天增长，粒度太细，不直接入选
        if since == "weekly" and stars_today >= STAR_GROWTH_THRESHOLD:
            if fn not in candidate_map:
                candidate_map[fn] = {
                    "growth": stars_today,
                    "star": star,
                    "created_at": "",
                    "source": f"trending-{since}",
                }
                direct_count += 1
                logger.info(
                    f"  [OK] Trending 直接入选({since}): {fn} | "
                    f"stars_today={stars_today} | star={star}"
                )
            continue

        # 其他（monthly 或 stars_today 不够）→ 加入 raw_repos 走增长计算
        if fn not in raw_repos and fn not in candidate_map:
            repo_item = {
                "full_name": fn,
                "stargazers_count": star,
                "forks_count": r.get("forks", 0),
                "description": r.get("description", ""),
                "language": r.get("language", ""),
                "topics": [],
            }
            raw_repos[fn] = {
                "star": star,
                "repo_item": repo_item,
                "created_at": "",
            }
            raw_count += 1

    logger.info(
        f"Trending 收集完成: {len(unique)} 个仓库, "
        f"直接入选 {direct_count} 个, 加入待计算 {raw_count} 个, "
        f"候选 {len(candidate_map)} 个, raw_repos 累计 {len(raw_repos)} 个。"
    )


# ══════════════════════════════════════════════════════════════
# 断点续传：Checkpoint 辅助
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


# ══════════════════════════════════════════════════════════════
# Phase 2: 统一批量增长计算（使用 TokenWorkerPool）
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
    Phase 2 入队：DB 差值法主线程处理，其余提交为 CalcGrowthTask。

    growth_ctx 由调用方创建并传入（包含 checkpoint, pending_created_at, db_projects 等共享状态）。
    返回 checkpoint dict。
    """
    db_valid = db.get("valid", False)
    db_projects = db.get("projects", {})
    growth_threshold = growth_ctx.get("growth_threshold", STAR_GROWTH_THRESHOLD)

    checkpoint = _load_checkpoint()
    growth_ctx["checkpoint"] = checkpoint

    pending = {
        fn: info for fn, info in raw_repos.items()
        if fn not in candidate_map
    }

    # 从 checkpoint 恢复
    resumed_count = 0
    for fn in list(pending.keys()):
        if fn in checkpoint:
            cp = checkpoint[fn]
            growth = cp["growth"]
            current_star = cp["star"]
            created_at = pending[fn].get("created_at", "")
            repo_item = pending[fn]["repo_item"]
            update_db_project(db_projects, fn, current_star, repo_item)
            if growth >= growth_threshold:
                _upsert_candidate(candidate_map, fn, growth, current_star, created_at, "checkpoint")
            del pending[fn]
            resumed_count += 1

    if resumed_count:
        logger.info(f"断点续传: 恢复 {resumed_count} 个已计算项目。")
        save_db(db)

    # DB 差值法：主线程直接处理
    checkpoint_dirty = False
    db_count = 0

    for full_name in list(pending.keys()):
        info = pending[full_name]
        current_star = info["star"]
        repo_item = info["repo_item"]
        created_at = info.get("created_at", "")

        if full_name in db_projects and db_valid:
            saved_star = db_projects[full_name].get("star", 0)
            growth = current_star - saved_star
            update_db_project(db_projects, full_name, current_star, repo_item)
            checkpoint[full_name] = {"growth": growth, "star": current_star}
            checkpoint_dirty = True
            db_count += 1
            if growth >= growth_threshold:
                _upsert_candidate(candidate_map, full_name, growth, current_star, created_at, "DB")
            del pending[full_name]

    if checkpoint_dirty:
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

    logger.info(
        f"批量增长计算: {len(pending)} 个任务入队 "
        f"(DB差值 {db_count}, 续传 {resumed_count}, "
        f"跳过已入选 {len(raw_repos) - len(pending) - db_count - resumed_count})"
    )

    return checkpoint


# ══════════════════════════════════════════════════════════════
# Step 2: 评分排序
# ══════════════════════════════════════════════════════════════


def _hydrate_candidate_created_at(
    candidate_map: dict[str, dict],
    db: dict | None,
    token_mgr: TokenManager | None,
) -> None:
    """为缺失 created_at 的候选补充创建时间，优先使用 DB，其次请求仓库详情。"""
    if not candidate_map:
        return

    db_projects = db.get("projects", {}) if db else {}

    for full_name, info in candidate_map.items():
        if info.get("created_at"):
            continue

        db_created_at = db_projects.get(full_name, {}).get("created_at", "")
        if db_created_at:
            info["created_at"] = db_created_at
            continue

        if token_mgr is None:
            continue

        try:
            items = search_github_repos(
                token_mgr,
                f"repo:{full_name}",
                token_idx=0,
                page=1,
                per_page=1,
                auto_star_filter=False,
            )
        except Exception as e:
            logger.warning(f"补全创建时间失败: {full_name}, {e}")
            continue
        time.sleep(SEARCH_REQUEST_INTERVAL)
        if not items:
            continue

        repo_item = next(
            (item for item in items if item.get("full_name") == full_name),
            items[0],
        )
        created_at = repo_item.get("created_at", "")
        if not created_at:
            continue

        info["created_at"] = created_at
        update_db_project(
            db_projects,
            full_name,
            info.get("star", repo_item.get("stargazers_count", 0)),
            repo_item,
        )


def step2_rank_and_select(
    candidate_map: dict[str, dict],
    mode: str = DEFAULT_SCORE_MODE,
    db: dict | None = None,
    token_mgr: TokenManager | None = None,
) -> list[tuple[str, dict]]:
    """
    Step 2: 评分排序 + 截取 Top N。

    评分模式：
      comprehensive — 综合排名：log(增长量) + log(增长率)，新项目平滑折扣
    hot_new       — 新项目专榜：优先用 DB 补全创建时间，必要时按仓库名补查后再按增长量排序

    Returns:
        [(full_name, {"growth": int, "star": int, ...}), ...] 按 score 降序，最多 HOT_PROJECT_COUNT 个。
    """
    import math
    from .config import NEW_PROJECT_DAYS

    def _calc_score(item: dict) -> float:
        g = item["growth"]
        s = item["star"]

        if s <= 0:
            return float(g)

        # 增长量（对数压缩极端值）
        growth_score = math.log(1 + g) * 1000
        # 增长率（对数衰减，避免新项目固定加分）
        rate = g / s
        rate_score = math.log(1 + rate) / math.log(2) * 3000

        if mode == "comprehensive":
            # 新项目平滑折扣：rate > 0.5 开始线性衰减，rate=1.0 时 0.85 折
            if rate > 0.5:
                discount = 1.0 - 0.15 * min((rate - 0.5) / 0.5, 1.0)
            else:
                discount = 1.0
            return (growth_score + rate_score) * discount
        else:
            # 兜底：纯增长量
            return float(g)

    def _is_new_project(info: dict) -> bool:
        """判定是否为新项目：创建时间距今 <= NEW_PROJECT_DAYS 天。"""
        created_at = info.get("created_at", "")
        if not created_at:
            return False
        try:
            created_date = datetime.strptime(
                created_at[:10], "%Y-%m-%d"
            ).replace(tzinfo=timezone.utc)
            days_since = (datetime.now(timezone.utc) - created_date).days
            return days_since <= NEW_PROJECT_DAYS
        except (ValueError, TypeError):
            return False

    if mode == "hot_new":
        _hydrate_candidate_created_at(candidate_map, db, token_mgr)
        # 新项目专榜：仅筛选创建时间 <= 45 天的新项目
        new_projects = {
            name: info for name, info in candidate_map.items()
            if _is_new_project(info)
        }
        # 新项目按增长量排序
        sorted_candidates = sorted(
            new_projects.items(),
            key=lambda x: x[1]["growth"],
            reverse=True,
        )
        top_n = sorted_candidates[:HOT_PROJECT_COUNT]
        logger.info(
            f"Step 2 (hot_new): 新项目(<={NEW_PROJECT_DAYS}天) {len(new_projects)} 个 → 取前 {len(top_n)} 个。"
        )
    else:
        sorted_candidates = sorted(
            candidate_map.items(), key=lambda x: _calc_score(x[1]), reverse=True
        )
        top_n = sorted_candidates[:HOT_PROJECT_COUNT]
        logger.info(
            f"Step 2 (comprehensive): 候选 {len(candidate_map)} → 取前 {len(top_n)} 个。"
        )

    logger.info("  Top 10 预览:")
    for i, (name, info) in enumerate(top_n[:10], 1):
        score = _calc_score(info)
        logger.info(
            f"    {i}. {name} (+{info['growth']}, star={info['star']}, score={score:.0f})"
        )

    return top_n


# ══════════════════════════════════════════════════════════════
# Step 3: LLM 描述 + 报告生成
# ══════════════════════════════════════════════════════════════


def step3_generate_report(
    top_projects: list[tuple[str, dict]], db: dict
) -> str:
    """
    Step 3: 为 Top N 项目生成 LLM 描述 + 输出 Markdown 报告。

        流程：
            1. 筛选 desc 为空的项目
            2. 串行调用 LLM 生成缺失描述
            3. LLM 描述写回 DB projects[repo].desc
            4. 写入 report/YYYY-MM-DD.md

    Returns:
        报告文件路径。
    """
    os.makedirs(REPORT_DIR, exist_ok=True)
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    report_path = os.path.join(REPORT_DIR, f"{today}.md")
    db_projects = db.get("projects", {})

    # ── 串行 LLM 调用 ──
    need_llm: list[tuple[int, str, str, dict]] = []
    desc_results: dict[str, str] = {}

    for idx, (full_name, info) in enumerate(top_projects):
        saved = db_projects.get(full_name, {})
        existing_desc = saved.get("desc", "")
        html_url = f"https://github.com/{full_name}"
        if existing_desc:
            desc_results[full_name] = existing_desc
        else:
            need_llm.append((idx + 1, full_name, html_url, saved))

    if need_llm:
        logger.info(f"Step 3: 需要生成描述 {len(need_llm)} 个项目，按顺序调用 LLM...")
        for idx, full_name, html_url, saved in need_llm:
            logger.info(f"[{idx}/{len(top_projects)}] LLM 生成描述: {full_name}")
            desc = call_llm_describe(full_name, saved, html_url)
            if desc:
                desc_results[full_name] = desc
                if full_name in db_projects:
                    db_projects[full_name]["desc"] = desc
            else:
                desc_results.setdefault(full_name, "")

    # ── 生成报告 ──
    lines: list[str] = [f"# GitHub 热门项目 — {today}\n"]
    lines.append(
        f"> 共 {len(top_projects)} 个项目 | "
        f"窗口期: {TIME_WINDOW_DAYS} 天 | "
        f"增长阈值: >={STAR_GROWTH_THRESHOLD} stars | "
        f"最低 star: >={MIN_STAR_FILTER}\n"
    )

    for idx, (full_name, info) in enumerate(top_projects, 1):
        growth = info["growth"]
        star = info["star"]
        html_url = f"https://github.com/{full_name}"
        detailed_desc = desc_results.get(full_name, "")

        lines.append(f"## {idx}. {full_name}（+{growth}，total {star}）\n")
        lines.append(f"链接: {html_url}\n")
        lines.append(f"{detailed_desc}\n")
        lines.append("---\n")

    report_content = "\n".join(lines)
    try:
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(report_content)
    except IOError as e:
        logger.error(f"报告写入失败: {report_path}, {e}")
        return ""

    logger.info(f"Step 3 完成: 报告已写入 {report_path}")
    return report_path


# ══════════════════════════════════════════════════════════════
# 内部辅助
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
# 主入口
# ══════════════════════════════════════════════════════════════


def main() -> None:
    """完整执行流程的入口函数。"""
    token_mgr = TokenManager()
    start_time = time.time()

    logger.info("=" * 60)
    logger.info("GitHub 热门项目发现 — 开始执行")
    logger.info(f"  Token 数量: {len(token_mgr.tokens)}")
    logger.info(f"  LLM: {LLM_API_URL} / {LLM_MODEL}")
    logger.info(f"  时间窗口: {TIME_WINDOW_DAYS} 天")
    logger.info(f"  增长阈值: >={STAR_GROWTH_THRESHOLD}")
    logger.info(f"  最低 star: >={MIN_STAR_FILTER}")
    logger.info(f"  热门数量: {HOT_PROJECT_COUNT}")
    logger.info(f"  DB 过期天数: {DATA_EXPIRE_DAYS}")
    logger.info(f"  搜索间隔: {SEARCH_REQUEST_INTERVAL}s")
    logger.info(f"  限流等待间隔: {RATE_LIMIT_WAIT_INTERVAL}s")
    logger.info(f"  Star 范围扫描: {STAR_RANGE_MIN}..{STAR_RANGE_MAX}")
    logger.info(f"  评分模式: {DEFAULT_SCORE_MODE}")
    logger.info(f"  并行 workers: {len(GITHUB_TOKENS)}")
    logger.info("=" * 60)

    # ── 1. 加载 DB ──
    db = load_db()
    logger.info(
        f"DB 状态: valid={db['valid']}, "
        f"已有 {len(db.get('projects', {}))} 个项目, "
        f"上次更新: {db.get('date', '从未')}"
    )

    # ── Phase 0: Star 范围自动分段（串行，池子未启动，Token#0 独占） ──
    logger.info(f"Star 范围分段 ({STAR_RANGE_MIN}..{STAR_RANGE_MAX}) 开始（串行）")
    segments = auto_split_star_range(token_mgr, STAR_RANGE_MIN, STAR_RANGE_MAX, token_idx=0)
    logger.info(
        f"自动分段完成，共 {len(segments)} 个子区间: "
        + ", ".join(f"[{lo}..{hi}]" for lo, hi in segments)
    )

    # ── 启动 Worker Pool ──
    pool = TokenWorkerPool(token_mgr.tokens)
    pool.start()

    raw_repos: dict[str, dict] = {}
    candidate_map: dict[str, dict] = {}

    try:
        # ── Phase 1: 统一收集（关键词搜索 + Star扫描 并行入队） ──
        total_keywords = sum(len(kws) for kws in SEARCH_KEYWORDS.values())
        keyword_idx = 0
        for category, keywords in SEARCH_KEYWORDS.items():
            for kw in keywords:
                keyword_idx += 1
                pool.submit(KeywordSearchTask(
                    _token_mgr=token_mgr,
                    keyword=kw,
                    category=category,
                    keyword_idx=keyword_idx,
                    total_keywords=total_keywords,
                    _raw_repos=raw_repos,
                ))

        total_segments = len(segments)
        for seg_idx, (low, high) in enumerate(segments, 1):
            pool.submit(ScanSegmentTask(
                _token_mgr=token_mgr,
                seg_idx=seg_idx,
                low=low,
                high=high,
                total_segments=total_segments,
                _raw_repos=raw_repos,
            ))

        # 主线程同时做 Trending（爬 HTML，无需 Token / Pool）
        collect_from_trending(raw_repos, candidate_map)

        # 等待 Phase 1 所有搜索任务完成
        logger.info(
            f"Phase 1: {total_keywords} 关键词 + {total_segments} 段 Star扫描 已入队，"
            f"等待 {pool.active_workers} 个 Worker 完成..."
        )
        pool.wait_all_done()
        phase1_tasks = pool.drain_results()

        logger.info(
            f"Phase 1 收集完成: raw_repos {len(raw_repos)} 个, "
            f"Trending 直接入选 {len(candidate_map)} 个, "
            f"完成任务 {phase1_tasks} 个。"
        )

        # ── Phase 2: 统一批量增长计算 ──
        growth_ctx = {
            "checkpoint": None,  # 由 _submit_growth_tasks 设置
            "pending_created_at": {},
            "db_projects": db.get("projects", {}),
            "candidate_map": candidate_map,
            "growth_threshold": STAR_GROWTH_THRESHOLD,
            "checkpoint_dirty": [False],  # 用 list 包装使子任务可修改
            "completed_since_save": [0],
        }
        checkpoint = _submit_growth_tasks(
            pool, token_mgr, raw_repos, db, candidate_map, growth_ctx
        )

        pool.wait_all_done()
        pool.drain_results()

        # drain_results 后检查 checkpoint 是否有未落盘的
        if growth_ctx["checkpoint_dirty"][0]:
            _save_checkpoint(checkpoint)

        logger.info(f"批量增长计算完成: 候选总数 {len(candidate_map)} 个。")

    finally:
        pool.shutdown()

    if not candidate_map:
        logger.warning("未找到任何满足增长阈值的候选项目。")
        db["valid"] = True
        save_db(db)
        _remove_checkpoint()
        elapsed = time.time() - start_time
        logger.info(f"无候选项目，DB 基线已更新。耗时: {elapsed:.1f}s")
        return

    # ── Phase 3: 排序取 Top N ──
    top_projects = step2_rank_and_select(candidate_map, db=db, token_mgr=token_mgr)

    # ── Phase 4: 生成报告 ──
    report_path = step3_generate_report(top_projects, db)

    # ── 5. 最终落盘 ──
    db["valid"] = True
    save_db(db)
    _remove_checkpoint()

    # ── 6. 统计摘要 ──
    elapsed = time.time() - start_time
    logger.info("=" * 60)
    logger.info("执行完成！")
    logger.info(f"  候选仓库数: {len(candidate_map)}")
    logger.info(f"  热门项目数: {len(top_projects)}")
    logger.info(f"  报告路径: {report_path}")
    logger.info(f"  DB 项目总数: {len(db.get('projects', {}))}")
    logger.info(f"  总耗时: {elapsed:.1f}s ({elapsed / 60:.1f}min)")
    logger.info("=" * 60)
