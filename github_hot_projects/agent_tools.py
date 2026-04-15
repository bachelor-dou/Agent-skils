"""
Agent Tool 定义
===============
9 个 Tool 函数 + TOOL_SCHEMAS（供 Agent ReAct 循环调用）。

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
    MIN_STAR_FILTER,
    NEW_PROJECT_DAYS,
    SEARCH_KEYWORDS,
    SEARCH_REQUEST_INTERVAL,
    STAR_GROWTH_THRESHOLD,
    STAR_RANGE_MAX,
    STAR_RANGE_MIN,
    TIME_WINDOW_DAYS,
)
from .common.db import save_db, update_db_project, set_growth_cache
from .common.github_api import auto_split_star_range, search_github_repos
from .growth_estimator import (
    GROWTH_ESTIMATION_UNRESOLVED,
    estimate_star_growth_binary,
)
from .common.llm import call_llm_describe
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


# ══════════════════════════════════════════════════════════════
# Tool 实现
# ══════════════════════════════════════════════════════════════


def tool_search_hot_projects(
    token_mgr: TokenManager,
    categories: list[str] | None = None,
    min_stars: int = STAR_RANGE_MIN,
    max_pages: int = 3,
    new_project_days: int | None = None,
) -> dict:
    """
    Tool 1: 按关键词类别搜索 GitHub 热门仓库（并行）。

    使用 TokenWorkerPool + KeywordSearchTask 并行搜索。

    Args:
        token_mgr:        TokenManager 实例
        categories:       搜索类别列表（如 ["AI-Agent", "AI-RAG"]），None 则搜索全部
        min_stars:        最低 star 过滤线
        max_pages:        每个关键词搜索的最大页数
        new_project_days: 新项目判定窗口（天），指定后在搜索查询中加入 created:>=date 过滤

    Returns:
        {"repos": [{"full_name": ..., "star": ..., "description": ...}, ...],
         "total": int, "categories_searched": list}
    """
    from datetime import timedelta

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
                    min_stars_override=min_stars,
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
        if star < min_stars:
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
      Phase 0 — 串行：auto_split_star_range 递归分段（Pool 未启动，token_idx=0）
      Phase 1 — 并行：ScanSegmentTask 提交到 Pool，N Worker 并行扫描

    Args:
        min_star:         最低星数
        max_star:         最高星数
        seen_repos:       已扫描过的仓库集合（用于去重）
        new_project_days: 新项目判定窗口（天），指定后在查询中加入 created:>=date 过滤
    """
    from datetime import timedelta

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

    # ── Phase 0: 串行分段（Pool 未启动，token_idx=0 独占） ──
    segments = auto_split_star_range(
        token_mgr, min_star, max_star, token_idx=0, extra_query=extra_query
    )
    raw_repos: dict[str, dict] = {}

    # ── Phase 1: 并行扫描各子区间 ──
    pool = TokenWorkerPool(token_mgr.tokens)
    pool.start()
    try:
        for seg_idx, (low, high) in enumerate(segments, 1):
            pool.submit(ScanSegmentTask(
                _token_mgr=token_mgr,
                seg_idx=seg_idx,
                low=low,
                high=high,
                total_segments=len(segments),
                created_after=created_after,
                min_stars_override=min_star_filter,
                _raw_repos=raw_repos,
            ))
        pool.wait_all_done()
        pool.drain_results()
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
) -> dict:
    """
    Tool 3: 查询单个仓库近期 star 增长，实时获取项目详情并生成 LLM 描述。

    返回实时数据（不使用 DB 缓存，除了创建时间）：
      - 当前 star、近期增长、语言、简介、创建时间
      - LLM 生成的项目详细描述（README 浓缩摘要）

    Args:
        repo: "owner/repo" 格式
        db:   DB 字典（可选，提供则优先用差值法计算增长）
    """
    parts = repo.split("/", 1)
    if len(parts) != 2:
        return {"error": f"仓库格式错误，应为 owner/repo: {repo}"}

    owner, repo_name = parts

    # 实时获取仓库信息
    items = search_github_repos(
        token_mgr, f"repo:{repo}", token_idx=0, page=1, per_page=1, auto_star_filter=False
    )
    if not items:
        return {"error": f"未找到仓库: {repo}"}

    repo_item = items[0]
    current_star = repo_item.get("stargazers_count", 0)

    # 增长计算
    if db and db.get("valid") and repo in db.get("projects", {}):
        saved_star = db["projects"][repo].get("star", 0)
        growth = current_star - saved_star
        method = "DB差值法"
    else:
        growth = estimate_star_growth_binary(token_mgr, owner, repo_name, current_star, token_idx=0)
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
        description = call_llm_describe(repo, repo_info, html_url)

    return {
        "repo": repo,
        "current_star": current_star,
        "growth": growth_value,
        "growth_status": growth_status,
        "time_window_days": TIME_WINDOW_DAYS,
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
    force_refresh: bool = False,
) -> dict:
    """
    Tool 4: 批量计算仓库增长并筛选候选。

    使用 TokenWorkerPool + CalcGrowthTask 并行计算。
    当 new_project_days 指定时，先按创建时间筛选新项目，只对新项目计算增长。
    当 force_refresh=True 时，跳过 checkpoint/growth_cache/DB 差值，全部走实时估算并刷新 DB。

    Args:
        repos:            仓库列表（含 full_name, star, _raw）
        db:               DB 字典
        growth_threshold: 增长阈值
        new_project_days: 新项目判定窗口（天），None 则不做创建时间筛选（全量计算）
        force_refresh:    是否强制实时刷新（不复用 DB 与缓存）
    """
    from datetime import timedelta

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
    # 确保 Trending 等无 created_at 的仓库也能存入 DB，后续查询可直接命中
    db_projects = db.get("projects", {})
    api_fetched_count = 0
    for fn, info in raw_repos.items():
        if info.get("created_at"):
            continue
        # 1. DB 查询
        db_ca = db_projects.get(fn, {}).get("created_at", "")
        if db_ca:
            info["created_at"] = db_ca
            info["repo_item"]["created_at"] = db_ca
            continue
        # 2. API 调用
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
                    update_db_project(
                        db_projects, fn,
                        info.get("star", repo_item.get("stargazers_count", 0)),
                        repo_item,
                    )
                    api_fetched_count += 1
            time.sleep(SEARCH_REQUEST_INTERVAL)
        except Exception as e:
            logger.warning(f"API 补全 created_at 失败: {fn}, {e}")
    if api_fetched_count:
        logger.info(f"created_at 补全: API 获取 {api_fetched_count} 个")

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
            "force_refresh": force_refresh,
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

    return {
        "candidates": candidate_map,
        "total_checked": len(raw_repos),
        "total_input": len(repos),
        "candidates_count": len(candidate_map),
        "unresolved_sampling_count": growth_ctx["unresolved_count"][0],
        "skipped_by_creation_time": skipped_count,
        "threshold": growth_threshold,
        "force_refresh": force_refresh,
    }


def tool_rank_candidates(
    candidates: dict[str, dict],
    top_n: int = HOT_PROJECT_COUNT,
    mode: str = DEFAULT_SCORE_MODE,
    db: dict | None = None,
    new_project_days: int | None = None,
    prefiltered_new_project_days: int | None = None,
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
    top = step2_rank_and_select(
        candidates, mode=mode, db=db,
        new_project_days=new_project_days,
        prefiltered_new_project_days=prefiltered_new_project_days,
    )[:top_n]

    if mode == "hot_new" and db is not None:
        save_db(db)

    # 将评分持久化到 DB（跨会话可查）
    if db is not None:
        db_projects = db.get("projects", {})
        for name, info in top:
            score = info.get("_score")
            if score is not None:
                set_growth_cache(db_projects, name, info["growth"], score=score)

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
    desc = call_llm_describe(repo, saved, html_url)
    if desc and repo in db_projects:
        db_projects[repo]["desc"] = desc

    return {
        "repo": repo,
        "description": desc or "描述生成失败",
        "source": "LLM生成",
    }


def tool_generate_report(
    top_projects: list[tuple[str, dict]],
    db: dict,
    mode: str = "comprehensive",
) -> dict:
    """
    Tool 7: 生成完整 Markdown 报告。

    调用 report.step3_generate_report 生成报告。
    """
    report_path = step3_generate_report(top_projects, db, mode=mode)
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
    language: str = "",
    spoken_language: str = "",
    include_all_periods: bool = False,
) -> dict:
    """
    Tool 10: 获取 GitHub Trending 仓库列表。

    两种使用路径：
      路径 1 — 直接展示：用户问 Trending 上有什么，默认返回 weekly
      路径 2 — 候选补充：可抓取 daily / weekly / monthly 三档并去重后加入候选池

    Args:
        since:           时间范围 ("daily" | "weekly" | "monthly")，默认 weekly
        language:        编程语言筛选，如 "python"，""=全部
        spoken_language: 自然语言代码，如 "zh"，""=全部
        include_all_periods:
                         为 True 时抓取 daily / weekly / monthly 三档并去重汇总
    """
    from .github_trending import fetch_trending, fetch_trending_all

    normalized_since = since if since in {"daily", "weekly", "monthly"} else "weekly"
    if include_all_periods:
        repos = fetch_trending_all(language=language, spoken_language=spoken_language)
    else:
        repos = fetch_trending(
            since=normalized_since,
            language=language,
            spoken_language=spoken_language,
        )

    if include_all_periods:
        display_repos = [
            {
                "full_name": r["full_name"],
                "star": r["star"],
                "forks": r["forks"],
                "periods": r.get("periods", []),
                "stars_by_period": r.get("stars_by_period", {}),
                "description": r["description"][:200],
                "language": r["language"],
            }
            for r in repos
        ]
    else:
        display_repos = [
            {
                "full_name": r["full_name"],
                "star": r["star"],
                "forks": r["forks"],
                "stars_today": r["stars_today"],
                "description": r["description"][:200],
                "language": r["language"],
            }
            for r in repos
        ]

    result = {
        "repos": display_repos,
        "count": len(display_repos),
        "language": language or "all",
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
                "按关键词类别搜索GitHub热门仓库。可指定搜索类别（如AI-Agent、AI-RAG等），"
                "返回满足star过滤条件的仓库列表。不计算增长，仅搜索。"
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
                    "min_stars": {
                        "type": "integer",
                        "description": f"最低star过滤线，默认{STAR_RANGE_MIN}",
                        "default": STAR_RANGE_MIN,
                    },
                    "max_pages": {
                        "type": "integer",
                        "description": "每个关键词搜索最大页数，默认3",
                        "default": 3,
                    },
                    "new_project_days": {
                        "type": "integer",
                        "description": (
                            "新项目筛选窗口（天）。指定后在搜索查询中加入 created:>=date 过滤，"
                            "只返回该天数内创建的仓库"
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
            "description": "按star范围扫描GitHub仓库，补充关键词搜索未覆盖的热门仓库。",
            "parameters": {
                "type": "object",
                "properties": {
                    "min_star": {
                        "type": "integer",
                        "description": f"最低星数，默认{STAR_RANGE_MIN}",
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
                            "新项目筛选窗口（天）。指定后在查询中加入 created:>=date 过滤，"
                            "只扫描该天数内创建的仓库"
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
                "查询单个GitHub仓库的实时详细信息：当前star、近期增长、语言、简介、创建时间，"
                "并调用LLM生成项目详细描述（README浓缩摘要）。"
                f"增长窗口为近{TIME_WINDOW_DAYS}天。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "repo": {
                        "type": "string",
                        "description": "仓库全名，格式为 owner/repo，如 vllm-project/vllm",
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
                "批量计算多个仓库的star增长并筛选满足阈值的候选。"
                "通常在search_hot_projects之后调用。"
                "指定new_project_days时，先按创建时间筛选新项目再计算增长，大幅减少请求量。"
                "指定force_refresh=true时，强制走实时估算，不复用DB差值和增长缓存。"
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
                            "新项目筛选窗口（天）。指定后只对创建时间在该窗口内的仓库计算增长。"
                            "不传则对所有仓库计算增长（综合排名场景）。"
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
                "对候选仓库评分排序，返回Top N。"
                "支持两种模式：comprehensive（综合排名）和 hot_new（新项目专榜）。"
                "通常在batch_check_growth之后调用。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "top_n": {
                        "type": "integer",
                        "description": f"返回前N个，默认{HOT_PROJECT_COUNT}",
                        "default": HOT_PROJECT_COUNT,
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
                            "hot_new模式下的新项目判定窗口（天）。"
                            f"默认{NEW_PROJECT_DAYS}天。如用户说'近一个月的新项目'则传30。"
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
            "description": "调用LLM为指定项目生成200-400字中文详细描述。",
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
            "description": "为已排序的Top项目生成完整Markdown报告（含LLM描述），保存到report目录。",
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
            "description": "查询DB状态（有效性、项目总数、上次更新日期）或单个仓库的详细信息。",
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
                "获取GitHub Trending页面的热门仓库列表。"
                "两种用途：1) 直接展示当前Trending项目；"
                "2) 将Trending仓库加入候选池进行评分排名。"
                "不消耗GitHub API配额。"
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
                    "language": {
                        "type": "string",
                        "description": "编程语言筛选，如python、javascript，不传则全部",
                    },
                    "spoken_language": {
                        "type": "string",
                        "description": "自然语言代码，如zh（中文）、en（英文），不传则全部",
                    },
                },
            },
        },
    },
]
