"""
Agent Tool 定义（执行层 · 工具中枢）
====================================
9 个 Tool 函数 + TOOL_SCHEMAS（供 Agent ReAct 循环调用）。

架构定位：
  执行层核心，连接 Agent 层与各独立执行组件（ranking/report/growth/trending/tasks）。
  agent.py → 【agent_tools.py】→ ranking.py / report.py / growth_estimator.py / ...
  execution/pipeline.py 也通过本模块编排完整流程。

Tool 列表：
  1. search_hot_projects    — 按关键词类别搜索热门仓库
  2. scan_star_range        — 按 star 范围扫描仓库
  3. check_repo_growth      — 查询单个仓库实时详情及近期增长
  4. batch_check_growth     — 批量计算仓库增长并筛选候选
  5. rank_candidates        — 对候选列表混合评分排序
  6. describe_project       — 调用 LLM 生成单个项目描述
  7. generate_report        — 生成完整 Markdown 报告
  8. get_db_info            — 查询 DB 状态和仓库信息
  9. fetch_trending         — 获取 GitHub Trending 热门仓库

内部实现拆分到独立模块：
  - tasks/     — Task 子类、批量提交、断点续传、候选管理
  - ranking.py — 评分排序算法
  - report.py  — 报告生成
"""

import logging
import time
from datetime import datetime, timezone

from .common.config import (
    DEFAULT_SCORE_MODE,
    GITHUB_TOKENS,
    HOT_PROJECT_COUNT,
    HOT_NEW_PROJECT_COUNT,
    MIN_STAR_FILTER,
    NEW_PROJECT_DAYS,
    SEARCH_KEYWORDS,
    SEARCH_REQUEST_INTERVAL,
    STAR_GROWTH_THRESHOLD,
    STAR_RANGE_MAX,
    STAR_RANGE_MIN,
    TIME_WINDOW_DAYS,
)
from .common.db import save_db, update_db_project
from .common.github_api import auto_split_star_range, fetch_repo_info, search_github_repos
from .growth_estimator import (
    GROWTH_ESTIMATION_UNRESOLVED,
    estimate_star_growth_binary,
)
from .common.llm import batch_condense_descriptions, call_llm_describe
from .common.token_manager import TokenManager
from .report import step3_generate_report
from .ranking import step2_rank_and_select
from .tasks import (
    KeywordSearchTask,
    ScanSegmentTask,
    _remove_checkpoint,
    _save_checkpoint,
    _submit_growth_tasks,
    TokenWorkerPool,
)

logger = logging.getLogger("discover_hot")


def coerce_positive_int(value: object, default: int, minimum: int = 1) -> int:
    """将关键数值参数规范到合法正数范围。"""
    if isinstance(value, int) and not isinstance(value, bool) and value >= minimum:
        return value
    return default


def coerce_optional_positive_int(value: object) -> int | None:
    """将可选正整数参数规范化；非法值回退为未启用。"""
    if isinstance(value, int) and not isinstance(value, bool) and value > 0:
        return value
    return None


def coerce_non_negative_int(value: object, default: int) -> int:
    """将阈值类参数规范到非负整数。"""
    if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
        return value
    return default


def coerce_ranking_mode(mode: object) -> str:
    """仅允许已知排名模式，非法值回退为默认模式。"""
    return mode if mode in {"comprehensive", "hot_new"} else DEFAULT_SCORE_MODE


def coerce_star_range(min_star: object, max_star: object) -> tuple[int, int]:
    """规范化 star 扫描区间，并自动修正反向区间。"""
    low = coerce_positive_int(min_star, STAR_RANGE_MIN)
    high = coerce_positive_int(max_star, STAR_RANGE_MAX)
    if high < low:
        low, high = high, low
    return low, high


def _ensure_project_record(
    db_projects: dict[str, dict],
    full_name: str,
    current_star: int,
    repo_item: dict,
    *,
    allow_snapshot_refresh: bool,
) -> dict:
    """确保 DB 中存在仓库记录；仅在允许时刷新快照字段。"""
    if allow_snapshot_refresh:
        update_db_project(db_projects, full_name, current_star, repo_item)
        project = db_projects.setdefault(full_name, {})
        project.setdefault("desc_level", "")
        return project

    readme_url = f"https://github.com/{full_name}/blob/HEAD/README.md"
    description = repo_item.get("description") or ""
    language = repo_item.get("language") or ""
    topics = repo_item.get("topics") or []
    forks = repo_item.get("forks_count", 0)
    created_at = repo_item.get("created_at") or ""

    if full_name not in db_projects:
        db_projects[full_name] = {
            "star": current_star,
            "forks": forks,
            "created_at": created_at,
            "desc": "",
            "desc_level": "",
            "short_desc": description[:500],
            "language": language,
            "topics": topics,
            "readme_url": readme_url,
        }
        return db_projects[full_name]

    project = db_projects[full_name]
    if "readme_url" not in project:
        project["readme_url"] = readme_url
    if created_at and not project.get("created_at"):
        project["created_at"] = created_at
    if description and not project.get("short_desc"):
        project["short_desc"] = description[:500]
    if language and not project.get("language"):
        project["language"] = language
    if topics and not project.get("topics"):
        project["topics"] = topics
    if "desc_level" not in project:
        project["desc_level"] = ""
    return project


def _write_candidate_brief_desc(
    candidate_map: dict[str, dict],
    raw_repos: dict[str, dict],
    db_projects: dict[str, dict],
    *,
    allow_snapshot_refresh: bool,
) -> int:
    """为候选仓库补充 LLM 简述（brief）并写入 DB 的 desc 字段。"""
    if not candidate_map:
        return 0

    pending_payload: list[dict] = []
    pending_names: list[str] = []

    for full_name, candidate in candidate_map.items():
        info = raw_repos.get(full_name, {})
        repo_item = info.get("repo_item", {})
        current_star = candidate.get("star", info.get("star", 0))
        project = _ensure_project_record(
            db_projects,
            full_name,
            current_star,
            repo_item,
            allow_snapshot_refresh=allow_snapshot_refresh,
        )

        existing_desc = (project.get("desc") or "").strip()
        if existing_desc:
            continue

        short_desc = (project.get("short_desc") or repo_item.get("description") or "").strip()
        language = project.get("language") or repo_item.get("language") or ""
        topics = project.get("topics") or repo_item.get("topics") or []

        summary_parts: list[str] = []
        if short_desc:
            summary_parts.append(short_desc)
        if language:
            summary_parts.append(f"语言: {language}")
        if topics:
            summary_parts.append(f"标签: {', '.join(topics[:4])}")
        if not summary_parts:
            summary_parts.append("暂无公开描述信息。")

        pending_names.append(full_name)
        pending_payload.append(
            {
                "full_name": full_name,
                "description": "；".join(summary_parts),
            }
        )

    if not pending_payload:
        return 0

    written = 0
    chunk_size = 40
    for start in range(0, len(pending_payload), chunk_size):
        chunk_payload = pending_payload[start:start + chunk_size]
        chunk_names = pending_names[start:start + chunk_size]
        condensed = batch_condense_descriptions(chunk_payload, max_chars=120)
        for idx, full_name in enumerate(chunk_names):
            desc = (condensed[idx] if idx < len(condensed) else "").strip()
            if not desc:
                continue
            project = db_projects.get(full_name)
            if not project:
                continue
            project["desc"] = desc
            project["desc_level"] = "brief"
            written += 1

    if written:
        logger.info(f"候选简述已写入 DB: {written} 个项目。")

    return written


# ══════════════════════════════════════════════════════════════
# Tool 实现
# ══════════════════════════════════════════════════════════════


def tool_search_hot_projects(
    token_mgr: TokenManager,
    categories: list[str] | None = None,
    project_min_star: int = MIN_STAR_FILTER,
    max_pages: int = 3,
    new_project_days: int | None = None,
    *,
    min_stars: int | None = None,
) -> dict:
    """
    Tool 1: 按关键词类别搜索 GitHub 热门仓库（并行）。

    使用 TokenWorkerPool + KeywordSearchTask 并行搜索。

    Args:
        token_mgr:        TokenManager 实例
        categories:       搜索类别列表（如 ["AI-Agent", "AI-RAG"]），None 则搜索全部
        project_min_star: 关键词搜索项目最低 star 过滤线
        max_pages:        每个关键词搜索的最大页数
        new_project_days: 新项目判定窗口（天），指定后在搜索查询中加入 created:>=date 过滤

    Returns:
        {"repos": [{"full_name": ..., "star": ..., "description": ...}, ...],
         "total": int, "categories_searched": list}
    """
    from datetime import timedelta

    if min_stars is not None and project_min_star == MIN_STAR_FILTER:
        project_min_star = min_stars

    project_min_star = coerce_positive_int(project_min_star, MIN_STAR_FILTER)
    max_pages = coerce_positive_int(max_pages, 3)
    new_project_days = coerce_optional_positive_int(new_project_days)

    if categories:
        keywords_dict = {k: v for k, v in SEARCH_KEYWORDS.items() if k in categories}
        if not keywords_dict:
            return {"repos": [], "total": 0, "categories_searched": [],
                    "error": f"未找到匹配类别，可用类别: {list(SEARCH_KEYWORDS.keys())}"}
    else:
        keywords_dict = SEARCH_KEYWORDS

    # 新项目模式：计算创建时间截止日期
    created_after = ""
    if new_project_days is not None:
        cutoff = datetime.now(timezone.utc) - timedelta(days=new_project_days)
        created_after = cutoff.strftime("%Y-%m-%d")

    raw_repos: dict[str, dict] = {}
    total_keywords = sum(len(kws) for kws in keywords_dict.values())

    # ── 并行搜索：提交 KeywordSearchTask 到 Pool ──
    pool = TokenWorkerPool(token_mgr.tokens)
    pool.start()
    try:
        keyword_idx = 0
        for category, keywords in keywords_dict.items():
            for keyword in keywords:
                keyword_idx += 1
                pool.submit(KeywordSearchTask(
                    _token_mgr=token_mgr,
                    keyword=keyword,
                    category=category,
                    keyword_idx=keyword_idx,
                    total_keywords=total_keywords,
                    max_pages=max_pages,
                    created_after=created_after,
                    project_min_star_override=project_min_star,
                    _raw_repos=raw_repos,
                ))
        pool.wait_all_done()
        pool.drain_results()
    finally:
        pool.shutdown()

    # ── 转换为返回格式 ──
    repos: list[dict] = []
    for fn, info in raw_repos.items():
        repo_item = info["repo_item"]
        star = info["star"]
        if star < project_min_star:
            continue
        repos.append({
            "full_name": fn,
            "star": star,
            "description": (repo_item.get("description") or "")[:200],
            "language": repo_item.get("language") or "",
            "topics": repo_item.get("topics") or [],
            "_raw": repo_item,
        })

    display_repos = [{k: v for k, v in r.items() if k != "_raw"} for r in repos]
    return {
        "repos": display_repos,
        "total": len(repos),
        "categories_searched": list(keywords_dict.keys()),
        "_raw_repos": repos,
    }


def tool_scan_star_range(
    token_mgr: TokenManager,
    min_star: int = STAR_RANGE_MIN,
    max_star: int = STAR_RANGE_MAX,
    seen_repos: set[str] | None = None,
    new_project_days: int | None = None,
) -> dict:
    """
    Tool 2: 按 star 范围扫描仓库（并行）。

    使用 TokenWorkerPool + ScanSegmentTask 并行扫描各子区间。

    阶段隔离：
    Phase 0 — 串行：auto_split_star_range 递归分段（主线程优先 token_idx=0，限流时自动切换其他 token）
      Phase 1 — 并行：ScanSegmentTask 提交到 Pool，N Worker 并行扫描

    Args:
        min_star:         最低星数
        max_star:         最高星数
        seen_repos:       已扫描过的仓库集合（用于去重）
        new_project_days: 新项目判定窗口（天），指定后在查询中加入 created:>=date 过滤
    """
    from datetime import timedelta

    min_star, max_star = coerce_star_range(min_star, max_star)
    new_project_days = coerce_optional_positive_int(new_project_days)

    if seen_repos is None:
        seen_repos = set()

    # 新项目模式：计算创建时间截止日期
    created_after = ""
    extra_query = ""
    min_star_filter = min_star
    if new_project_days is not None:
        cutoff = datetime.now(timezone.utc) - timedelta(days=new_project_days)
        created_after = cutoff.strftime("%Y-%m-%d")
        extra_query = f"created:>={created_after}"

    # ── Phase 0: 串行分段（主线程优先 token_idx=0，必要时自动切换其他 token） ──
    segments = auto_split_star_range(
        token_mgr, min_star, max_star, token_idx=0, extra_query=extra_query
    )
    raw_repos: dict[str, dict] = {}

    # ── Phase 1: 并行扫描各子区间 ──
    pool = TokenWorkerPool(token_mgr.tokens)
    pool.start()
    try:
        segment_tasks: list[ScanSegmentTask] = []
        for seg_idx, (low, high) in enumerate(segments, 1):
            task = ScanSegmentTask(
                _token_mgr=token_mgr,
                seg_idx=seg_idx,
                low=low,
                high=high,
                total_segments=len(segments),
                created_after=created_after,
                min_star_filter_override=min_star_filter,
                _raw_repos=raw_repos,
            )
            segment_tasks.append(task)
            pool.submit(task)
        pool.wait_all_done()
        pool.drain_results()

        retry_tasks: list[ScanSegmentTask] = []
        retried_pages = 0
        for task in segment_tasks:
            if not task.failed_pages:
                continue
            retried_pages += len(task.failed_pages)
            retry_task = ScanSegmentTask(
                _token_mgr=token_mgr,
                seg_idx=task.seg_idx,
                low=task.low,
                high=task.high,
                total_segments=task.total_segments,
                created_after=task.created_after,
                min_star_filter_override=task.min_star_filter_override,
                page_numbers=list(task.failed_pages),
                retry_round=1,
                _raw_repos=raw_repos,
            )
            retry_tasks.append(retry_task)

        if retry_tasks:
            logger.warning(
                f"区间扫描发现 {retried_pages} 个失败页，提交 {len(retry_tasks)} 个页级补偿任务。"
            )
            for task in retry_tasks:
                pool.submit(task)
            pool.wait_all_done()
            pool.drain_results()

            final_failed = [
                (task.low, task.high, page)
                for task in retry_tasks
                for page in task.failed_pages
            ]
            if final_failed:
                failed_preview = ", ".join(
                    f"stars:{low}..{high}/page={page}"
                    for low, high, page in final_failed[:10]
                )
                if len(final_failed) > 10:
                    failed_preview += ", ..."
                logger.error(
                    f"页级补偿后仍有 {len(final_failed)} 个失败页，结果可能不完整: {failed_preview}"
                )
    finally:
        pool.shutdown()

    # ── 去重 + 转换返回格式 ──
    repos: list[dict] = []
    for fn, info in raw_repos.items():
        if fn in seen_repos:
            continue
        seen_repos.add(fn)
        repo_item = info["repo_item"]
        repos.append({
            "full_name": fn,
            "star": info["star"],
            "description": (repo_item.get("description") or "")[:200],
            "language": repo_item.get("language") or "",
            "_raw": repo_item,
        })

    display_repos = [{k: v for k, v in r.items() if k != "_raw"} for r in repos]
    return {
        "repos": display_repos,
        "total": len(repos),
        "star_range": f"{min_star}..{max_star}",
        "segments": len(segments),
        "_raw_repos": repos,
    }


def tool_check_repo_growth(
    token_mgr: TokenManager,
    repo: str,
    db: dict | None = None,
    time_window_days: int = TIME_WINDOW_DAYS,
) -> dict:
    """
    Tool 3: 查询单个仓库近期 star 增长，实时获取项目详情并生成 LLM 描述。

    增长计算始终走实时二分法/采样外推，不走 DB 差值法。
    DB 仅用于读取已有描述缓存和补充静态元数据。

    Args:
        repo: "owner/repo" 格式
        db:   DB 字典（可选，仅用于读取描述缓存）
        time_window_days: 增长统计窗口（天）
    """
    time_window_days = coerce_positive_int(time_window_days, TIME_WINDOW_DAYS)

    parts = repo.split("/", 1)
    if len(parts) != 2:
        return {"error": f"仓库格式错误，应为 owner/repo: {repo}"}

    owner, repo_name = parts

    # 实时获取仓库信息（直接调用 /repos API，避免 Search API 的 422 问题）
    repo_item = fetch_repo_info(token_mgr, owner, repo_name, token_idx=0)
    if not repo_item:
        return {
            "error": f"未找到仓库: {repo}（可能不存在或为私有仓库）",
            "hint": "建议改用 describe_project 获取该项目的描述信息，或用 get_db_info 查询本地数据库。",
        }

    current_star = repo_item.get("stargazers_count", 0)

    growth = estimate_star_growth_binary(
        token_mgr,
        owner,
        repo_name,
        current_star,
        token_idx=0,
        time_window_days=time_window_days,
    )
    if time_window_days != TIME_WINDOW_DAYS:
        method = f"自定义{time_window_days}天窗口，二分法/采样外推"
    else:
        method = "二分法/采样外推"

    if growth == GROWTH_ESTIMATION_UNRESOLVED:
        growth_value = None
        growth_status = "sampling_unresolved"
        meets_threshold = False
        method = f"{method}(未决)"
        growth_warning = "采样数据不足，当前未返回可靠增长估值；本轮结果未写入批处理 checkpoint/DB。"
    else:
        growth_value = growth
        growth_status = "ok"
        meets_threshold = growth >= STAR_GROWTH_THRESHOLD
        growth_warning = ""

    # LLM 生成项目描述（README 浓缩摘要），优先复用 DB 中已有描述
    html_url = repo_item.get("html_url", f"https://github.com/{repo}")
    cached_desc = ""
    if db:
        cached_desc = db.get("projects", {}).get(repo, {}).get("desc", "")
    if cached_desc:
        description = cached_desc
    else:
        repo_info = {
            "short_desc": repo_item.get("description", ""),
            "language": repo_item.get("language", ""),
            "topics": repo_item.get("topics", []),
            "readme_url": f"{html_url}#readme",
        }
        description = call_llm_describe(repo, repo_info, html_url, detail_level="detailed")

    return {
        "repo": repo,
        "current_star": current_star,
        "growth": growth_value,
        "growth_status": growth_status,
        "time_window_days": time_window_days,
        "method": method,
        "meets_threshold": meets_threshold,
        "warning": growth_warning,
        "language": repo_item.get("language", ""),
        "short_desc": (repo_item.get("description") or "")[:200],
        "created_at": repo_item.get("created_at", ""),
        "topics": repo_item.get("topics", []),
        "description": description or "描述生成失败",
    }


def tool_batch_check_growth(
    token_mgr: TokenManager,
    repos: list[dict],
    db: dict,
    growth_threshold: int = STAR_GROWTH_THRESHOLD,
    new_project_days: int | None = None,
    time_window_days: int = TIME_WINDOW_DAYS,
    force_refresh: bool = False,
    refresh_db: bool = False,
    window_specified: bool = True,
) -> dict:
    """
    Tool 4: 批量计算仓库增长并筛选候选。

    使用 TokenWorkerPool + CalcGrowthTask 并行计算。
    当 new_project_days 指定时，先按创建时间筛选新项目，只对新项目计算增长。
    当 force_refresh=True 或 refresh_db=True 时，跳过 checkpoint/DB 差值并走实时估算；
    force_refresh 与 refresh_db 均允许刷新 DB 快照。

    Args:
        repos:            仓库列表（含 full_name, star, _raw）
        db:               DB 字典
        growth_threshold: 增长阈值
        new_project_days: 新项目判定窗口（天），None 则不做创建时间筛选（全量计算）
        time_window_days: 增长统计窗口（天）
        force_refresh:    是否强制实时刷新（不复用 DB 与缓存）
        refresh_db:       是否允许在本轮刷新 DB 快照（供定期任务调用）
        window_specified: 调用方是否显式指定了 time_window_days
    """
    from datetime import timedelta

    growth_threshold = coerce_non_negative_int(growth_threshold, STAR_GROWTH_THRESHOLD)
    new_project_days = coerce_optional_positive_int(new_project_days)
    time_window_days = coerce_positive_int(time_window_days, TIME_WINDOW_DAYS)
    refresh_db = bool(refresh_db)
    window_specified = bool(window_specified)
    effective_force_refresh = force_refresh or refresh_db or time_window_days != TIME_WINDOW_DAYS
    allow_snapshot_refresh = force_refresh or refresh_db

    # 构建 raw_repos 格式
    raw_repos: dict[str, dict] = {}
    for r in repos:
        fn = r["full_name"]
        if fn in raw_repos:
            continue
        raw_item = r.get("_raw", r)
        raw_repos[fn] = {
            "star": r["star"],
            "repo_item": raw_item,
            "created_at": raw_item.get("created_at", ""),
        }

    # ── 补全缺失的 created_at（DB → API），所有模式通用 ──
    db_projects = db.get("projects", {})
    api_fetched_count = 0
    for fn, info in raw_repos.items():
        if info.get("created_at"):
            continue
        db_ca = db_projects.get(fn, {}).get("created_at", "")
        if db_ca:
            info["created_at"] = db_ca
            info["repo_item"]["created_at"] = db_ca
            continue
        try:
            items = search_github_repos(
                token_mgr,
                f"repo:{fn}",
                token_idx=0,
                page=1,
                per_page=1,
                auto_star_filter=False,
            )
            if items:
                repo_item = next(
                    (item for item in items if item.get("full_name") == fn),
                    items[0],
                )
                created_at = repo_item.get("created_at", "")
                if created_at:
                    info["created_at"] = created_at
                    info["repo_item"]["created_at"] = created_at
                    api_fetched_count += 1
            time.sleep(SEARCH_REQUEST_INTERVAL)
        except Exception as e:
            logger.warning(f"API 补全 created_at 失败: {fn}, {e}")
    if api_fetched_count:
        logger.info(f"created_at 补全: API 获取 {api_fetched_count} 个")

    seeded_count = 0
    if allow_snapshot_refresh:
        for fn, info in raw_repos.items():
            repo_item = info.get("repo_item", {})
            if info.get("created_at") and not repo_item.get("created_at"):
                repo_item["created_at"] = info["created_at"]
            update_db_project(db_projects, fn, info.get("star", 0), repo_item)
            seeded_count += 1
        if seeded_count:
            logger.info(f"刷新模式: 初筛阶段已同步 DB 快照 {seeded_count} 个项目。")

    # ── 新项目前置筛选：仅保留创建时间在窗口内的仓库 ──
    skipped_count = 0
    if new_project_days is not None:
        cutoff = datetime.now(timezone.utc) - timedelta(days=new_project_days)
        filtered: dict[str, dict] = {}
        for fn, info in raw_repos.items():
            created_at = info.get("created_at", "")
            if not created_at:
                skipped_count += 1
                continue
            try:
                created_date = datetime.strptime(
                    created_at[:10], "%Y-%m-%d"
                ).replace(tzinfo=timezone.utc)
                if created_date >= cutoff:
                    filtered[fn] = info
                else:
                    skipped_count += 1
            except (ValueError, TypeError):
                skipped_count += 1
        logger.info(
            f"新项目前置筛选(<={new_project_days}天): "
            f"原 {len(raw_repos)} 个 → 保留 {len(filtered)} 个, "
            f"跳过 {skipped_count} 个"
        )
        raw_repos = filtered

    candidate_map: dict[str, dict] = {}

    pool = TokenWorkerPool(token_mgr.tokens)
    pool.start()
    try:
        growth_ctx = {
            "checkpoint": None,
            "pending_created_at": {},
            "db_projects": db.get("projects", {}),
            "candidate_map": candidate_map,
            "growth_threshold": growth_threshold,
            "force_refresh": effective_force_refresh,
            "update_db": allow_snapshot_refresh,
            "window_specified": window_specified,
            "time_window_days": time_window_days,
            "use_checkpoint": (not effective_force_refresh) and window_specified,
            "cache_growth": time_window_days == TIME_WINDOW_DAYS,
            "unresolved_count": [0],
            "checkpoint_dirty": [False],
            "completed_since_save": [0],
        }
        checkpoint = _submit_growth_tasks(
            pool, token_mgr, raw_repos, db, candidate_map, growth_ctx
        )
        pool.wait_all_done()
        pool.drain_results()

        if growth_ctx["checkpoint_dirty"][0]:
            _save_checkpoint(checkpoint)

        _remove_checkpoint()
    finally:
        pool.shutdown()

    brief_desc_written = _write_candidate_brief_desc(
        candidate_map,
        raw_repos,
        db_projects,
        allow_snapshot_refresh=allow_snapshot_refresh,
    )

    db_updated = bool(allow_snapshot_refresh or brief_desc_written > 0)
    effective_time_window = growth_ctx.get("effective_time_window_days", time_window_days)

    return {
        "candidates": candidate_map,
        "total_checked": len(raw_repos),
        "total_input": len(repos),
        "candidates_count": len(candidate_map),
        "unresolved_sampling_count": growth_ctx["unresolved_count"][0],
        "skipped_by_creation_time": skipped_count,
        "threshold": growth_threshold,
        "force_refresh": effective_force_refresh,
        "db_updated": db_updated,
        "seeded_snapshot_count": seeded_count,
        "brief_desc_written": brief_desc_written,
        "time_window_days": effective_time_window,
        "requested_time_window_days": time_window_days,
    }


def tool_rank_candidates(
    candidates: dict[str, dict],
    top_n: int | None = None,
    mode: str = DEFAULT_SCORE_MODE,
    db: dict | None = None,
    new_project_days: int | None = None,
    prefiltered_new_project_days: int | None = None,
    time_window_days: int = TIME_WINDOW_DAYS,
) -> dict:
    """
    Tool 5: 对候选列表评分排序。

    Args:
        candidates:       {full_name: {"growth": int, "star": int, "stars_today": int(可选)}}
        top_n:            取前 N 个
        mode:             评分模式 ("comprehensive" | "hot_new")
        new_project_days: hot_new 模式下新项目判定窗口（天），None 则使用默认值
        prefiltered_new_project_days:
                          候选池在 batch_check_growth 阶段已按该窗口预筛；
                          与 new_project_days 一致时，排名阶段可直接按增长排序
    """
    mode = coerce_ranking_mode(mode)
    time_window_days = coerce_positive_int(time_window_days, TIME_WINDOW_DAYS)
    new_project_days = coerce_optional_positive_int(new_project_days)
    prefiltered_new_project_days = coerce_optional_positive_int(prefiltered_new_project_days)
    if top_n is None:
        top_n = HOT_NEW_PROJECT_COUNT if mode == "hot_new" else HOT_PROJECT_COUNT
    else:
        top_n = coerce_positive_int(
            top_n,
            HOT_NEW_PROJECT_COUNT if mode == "hot_new" else HOT_PROJECT_COUNT,
        )

    top = step2_rank_and_select(
        candidates, mode=mode, db=db,
        new_project_days=new_project_days,
        prefiltered_new_project_days=prefiltered_new_project_days,
    )[:top_n]

    if mode == "hot_new" and db is not None:
        save_db(db)

    ranked = []
    for i, (name, info) in enumerate(top, 1):
        ranked.append({
            "rank": i,
            "repo": name,
            "growth": info["growth"],
            "star": info["star"],
        })

    return {
        "ranked_projects": ranked,
        "total_candidates": len(candidates),
        "returned": len(ranked),
        "mode": mode,
        "_ordered_tuples": top,  # 内部使用
    }


def tool_describe_project(repo: str, db: dict) -> dict:
    """
    Tool 6: 调用 LLM 为单个项目生成描述。

    Args:
        repo: "owner/repo"
        db:   DB 字典
    """
    db_projects = db.get("projects", {})
    saved = db_projects.get(repo, {})
    existing = saved.get("desc", "")
    if existing:
        return {"repo": repo, "description": existing, "source": "DB缓存"}

    html_url = f"https://github.com/{repo}"
    desc = call_llm_describe(repo, saved, html_url, detail_level="detailed")
    if desc and repo in db_projects:
        db_projects[repo]["desc"] = desc
        db_projects[repo]["desc_level"] = "detailed"

    return {
        "repo": repo,
        "description": desc or "描述生成失败",
        "source": "LLM生成",
    }


def tool_generate_report(
    top_projects: list[tuple[str, dict]],
    db: dict,
    mode: str = "comprehensive",
    new_project_days: int | None = None,
    time_window_days: int = TIME_WINDOW_DAYS,
) -> dict:
    """
    Tool 7: 生成完整 Markdown 报告。

    调用 report.step3_generate_report 生成报告。
    """
    report_path = step3_generate_report(
        top_projects,
        db,
        mode=mode,
        new_project_days=new_project_days,
        time_window_days=time_window_days,
    )
    return {"report_path": report_path, "project_count": len(top_projects)}


def tool_get_db_info(db: dict, repo: str | None = None) -> dict:
    """
    Tool 8: 查询 DB 状态或特定仓库信息。

    Args:
        db:   DB 字典
        repo: 可选，查询特定仓库；None 则返回概览
    """
    if repo:
        info = db.get("projects", {}).get(repo)
        if info:
            return {"repo": repo, "info": info, "found": True}
        return {"repo": repo, "found": False}

    return {
        "valid": db.get("valid", False),
        "date": db.get("date", ""),
        "total_projects": len(db.get("projects", {})),
    }


def tool_fetch_trending(
    since: str = "weekly",
    include_all_periods: bool = False,
) -> dict:
    """
    Tool 10: 获取 GitHub Trending 仓库列表。

    两种使用路径：
      路径 1 — 直接展示：用户问 Trending 上有什么，默认返回 weekly
      路径 2 — 候选补充：可抓取 daily / weekly / monthly 三档并去重后加入候选池

    Args:
        since:           时间范围 ("daily" | "weekly" | "monthly")，默认 weekly
        include_all_periods:
                         为 True 时抓取 daily / weekly / monthly 三档并去重汇总
    """
    from .github_trending import fetch_trending, fetch_trending_all

    normalized_since = since if since in {"daily", "weekly", "monthly"} else "weekly"
    if include_all_periods:
        repos = fetch_trending_all()
    else:
        repos = fetch_trending(since=normalized_since)

    period_label = {"daily": "今日增长", "weekly": "本周增长", "monthly": "本月增长"}

    # 用 LLM 批量浓缩描述（最多70字），失败时回退截断
    from .common.llm import batch_condense_descriptions
    condensed = batch_condense_descriptions(repos, max_chars=70)

    if include_all_periods:
        display_repos = [
            {
                "full_name": r["full_name"],
                "star": r["star"],
                "forks": r["forks"],
                "periods": r.get("periods", []),
                "stars_by_period": r.get("stars_by_period", {}),
                "description": condensed[i],
                "language": r["language"],
            }
            for i, r in enumerate(repos)
        ]
    else:
        growth_field = period_label.get(normalized_since, "增长")
        display_repos = [
            {
                "full_name": r["full_name"],
                "star": r["star"],
                "forks": r["forks"],
                growth_field: r["stars_today"],
                "description": condensed[i],
                "language": r["language"],
            }
            for i, r in enumerate(repos)
        ]

    result = {
        "repos": display_repos,
        "count": len(display_repos),
        "include_all_periods": include_all_periods,
        "_raw_repos": repos,  # 内部使用
    }
    if include_all_periods:
        result["periods"] = ["daily", "weekly", "monthly"]
    else:
        result["since"] = normalized_since

    return result


# ══════════════════════════════════════════════════════════════
# Tool Schema 定义（OpenAI Function Calling 格式）
# ══════════════════════════════════════════════════════════════

TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "search_hot_projects",
            "description": (
                "【批量搜索】按关键词类别从 GitHub 批量搜索仓库，用于构建热榜候选池。"
                "可指定搜索类别（如AI-Agent、AI-RAG等），返回满足 star 过滤条件的仓库列表。"
                "仅搜索收集，不计算增长。不适合查询单个特定项目。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "categories": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": (
                            f"搜索类别列表，可选值: {list(SEARCH_KEYWORDS.keys())}。"
                            "不传则搜索全部类别。"
                        ),
                    },
                    "project_min_star": {
                        "type": "integer",
                        "description": f"关键词搜索项目最低 star 过滤线，默认{MIN_STAR_FILTER}",
                        "default": MIN_STAR_FILTER,
                    },
                    "max_pages": {
                        "type": "integer",
                        "description": "每个关键词搜索最大页数，默认3",
                        "default": 3,
                    },
                    "new_project_days": {
                        "type": "integer",
                        "description": (
                            "新项目创建时间窗口（天）。指定后只搜索创建时间在该天数以内的仓库（GitHub API created:>=date）。"
                            "例如用户说'近20天内新创建的项目'则传 20。"
                            "与 time_window_days（增长统计窗口）是完全独立的参数。"
                            "如果用户意图不涉及新项目过滤，不要传此参数。"
                        ),
                    },
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "scan_star_range",
            "description": (
                "【批量扫描】按 star 数量范围扫描 GitHub 仓库，补充关键词搜索未覆盖的热门仓库。"
                "与 search_hot_projects 配合使用。不适合查询单个特定项目。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "min_star": {
                        "type": "integer",
                        "description": f"扫描区间最低星数，默认{STAR_RANGE_MIN}",
                        "default": STAR_RANGE_MIN,
                    },
                    "max_star": {
                        "type": "integer",
                        "description": f"最高星数，默认{STAR_RANGE_MAX}",
                        "default": STAR_RANGE_MAX,
                    },
                    "new_project_days": {
                        "type": "integer",
                        "description": (
                            "新项目创建时间窗口（天）。指定后只扫描创建时间在该天数以内的仓库。"
                            "例如用户说'近20天内新创建的项目'则传 20。"
                            "与 time_window_days（增长统计窗口）完全独立。"
                            "如果用户意图不涉及新项目过滤，不要传此参数。"
                        ),
                    },
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "check_repo_growth",
            "description": (
                "【增长数据】查询单个仓库的 star 增长趋势：当前 star 数、近期增长量和增长率。"
                "仅适合回答「这个项目最近涨了多少 star」「增长趋势怎么样」等增长类问题。"
                "不适合回答「这个项目是做什么的」「支持哪些功能」等功能了解类问题（应使用 describe_project）。"
                f"默认增长窗口为近{TIME_WINDOW_DAYS}天。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "repo": {
                        "type": "string",
                        "description": "仓库全名，格式为 owner/repo，如 vllm-project/vllm",
                    },
                    "time_window_days": {
                        "type": "integer",
                        "description": "增长统计窗口（天）。如用户说近10天/近30天，则传对应值；默认不传时使用系统默认窗口。",
                    },
                },
                "required": ["repo"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "batch_check_growth",
            "description": (
                "【批量增长筛选】对 search_hot_projects/scan_star_range 收集的候选仓库批量计算 star 增长，"
                "筛选满足阈值的候选。通常在搜索/扫描之后、排序之前调用。"
                "不适合查询单个项目。"
                "支持 time_window_days（自定义增长统计窗口）、new_project_days（按创建时间前置过滤）、"
                "force_refresh（跳过缓存强制实时估算）等参数。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "growth_threshold": {
                        "type": "integer",
                        "description": f"增长阈值，默认{STAR_GROWTH_THRESHOLD}",
                        "default": STAR_GROWTH_THRESHOLD,
                    },
                    "new_project_days": {
                        "type": "integer",
                        "description": (
                            "新项目创建时间窗口（天）。指定后先按创建时间过滤，只对N天内创建的项目计算增长。"
                            "例如用户说'近20天内新创建的项目'则传 20。"
                            "与 time_window_days 完全独立：new_project_days 过滤创建时间，time_window_days 决定增长统计区间。"
                            "两者可同时指定。如果用户未提及新项目/新创建，不要传此参数。"
                        ),
                    },
                    "time_window_days": {
                        "type": "integer",
                        "description": (
                            "增长统计窗口（天）。计算最近N天的star增长量。"
                            "例如用户说'近10天热榜'则传 10。与 new_project_days（创建时间过滤）完全独立。"
                        ),
                    },
                    "force_refresh": {
                        "type": "boolean",
                        "description": (
                            "是否强制实时刷新。true=不走checkpoint/growth_cache/DB差值，"
                            "直接重新估算增长并刷新数据库。默认false。"
                        ),
                        "default": False,
                    },
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "rank_candidates",
            "description": (
                "【排序出榜】对 batch_check_growth 筛选后的候选仓库评分排序，输出 Top N 榜单。"
                "comprehensive=综合排名；hot_new=新项目专榜。通常在 batch_check_growth 之后调用。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "top_n": {
                        "type": "integer",
                        "description": (
                            f"返回前N个。comprehensive 默认{HOT_PROJECT_COUNT}；"
                            f"hot_new 默认{HOT_NEW_PROJECT_COUNT}。"
                        ),
                    },
                    "mode": {
                        "type": "string",
                        "enum": ["comprehensive", "hot_new"],
                        "description": (
                            "评分模式。comprehensive=综合排名（增长量+增长率，新项目平滑折扣）；"
                            "hot_new=新项目专榜（仅满足创建时间窗口的新项目，按增长量排序）。"
                            "默认comprehensive。"
                        ),
                        "default": "comprehensive",
                    },
                    "new_project_days": {
                        "type": "integer",
                        "description": (
                            "新项目创建时间窗口（天）。hot_new 模式下用于筛选创建时间在N天以内的项目。"
                            f"默认{NEW_PROJECT_DAYS}天。"
                            "例如用户说'近20天内新创建项目的榜单'则传 20。"
                            "用户明确提到天数时必须传入用户指定的值，不要用默认值覆盖。"
                        ),
                    },
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "describe_project",
            "description": (
                "【项目介绍】获取单个项目的功能介绍和详细描述（基于 README 生成 200-400 字中文摘要）。"
                "适合回答「这个项目是做什么的」「能不能用于某场景」「支持哪些功能/CLI/平台」等功能了解类问题。"
                "当用户问项目功能、兼容性、使用方式、适用场景时，必须优先使用此工具而非 check_repo_growth。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "repo": {
                        "type": "string",
                        "description": "仓库全名，如 vllm-project/vllm",
                    },
                },
                "required": ["repo"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "generate_report",
            "description": (
                "【报告生成】为 rank_candidates 输出的 Top N 项目生成完整 Markdown 报告，保存到 report 目录。"
                "仅在完成搜索→增长→排序全流程后调用。"
            ),
            "parameters": {
                "type": "object",
                "properties": {},
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_db_info",
            "description": (
                "【数据库查询】查询本地 DB 状态概览（项目总数、更新日期）或指定仓库的历史缓存数据。"
                "仅返回本地已存储的信息，不从 GitHub 实时获取。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "repo": {
                        "type": "string",
                        "description": "可选，查询特定仓库信息。不传则返回DB概览。",
                    },
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "fetch_trending",
            "description": (
                "【Trending 浏览】获取 GitHub Trending 页面的热门仓库列表。"
                "用途：1) 直接展示当前 Trending 项目；2) 在热榜工作流中作为候选补充源。"
                "不消耗 GitHub API 配额。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "since": {
                        "type": "string",
                        "enum": ["daily", "weekly", "monthly"],
                        "description": "时间范围，直接浏览 Trending 时默认 weekly",
                        "default": "weekly",
                    },
                    "include_all_periods": {
                        "type": "boolean",
                        "description": (
                            "为 true 时抓取 daily、weekly、monthly 三个 Trending 维度并按仓库去重汇总。"
                            "适合综合排名和新项目排名的候选补充阶段。"
                        ),
                        "default": False,
                    },
                },
            },
        },
    },
]
