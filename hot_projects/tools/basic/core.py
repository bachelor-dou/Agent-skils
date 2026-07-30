"""
tools/basic/core — 基础能力实现（纯函数）
==========================================
被 ranking 复合工具、独立工具(repo_growth/describe_project 等)与 cron 复用。
经 tools/basic/__init__.py 以干净名字导出。

能力清单：
  search_by_keywords / scan_star_range — 候选收集(并行搜索/星段扫描)
  check_repo_growth / batch_check_growth — 单仓库/批量 star 增长
  describe_project — LLM 生成单项目描述
  get_db_info — 查询 DB 状态/仓库信息
  fetch_trending — 获取 GitHub Trending

依赖：infra/concurrency(Task 子类/断点续传)、scoring.py。
"""

import asyncio
import logging
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor
from datetime import datetime, timedelta, timezone

from ...config import (
    MIN_STAR,
    MAX_STAR,
    SEARCH_KEYWORDS,
    STAR_GROWTH_THRESHOLD,
    GROWTH_CALC_DAYS,
    MAX_DYNAMIC_SEARCH_KEYWORDS,
)
from ...infra.db import update_db_project, get_db_age_days
from ...infra.snapshots import (
    SNAPSHOT_ANCHOR_TOLERANCE_DAYS,
    anchor_for_window,
    utc_today,
)
from ...datasource.github.token_pool import AsyncTokenPool, GitHubTokenPool
from ...datasource.github.api import (
    SEARCH_REQUEST_INTERVAL,
    auto_split_star_range,
    build_github_async_client,
    fetch_repo_info,
    fetch_repo_readme_excerpt,
    fetch_repo_recent_commits,
    fetch_repo_recent_releases,
    search_github_repos,
)
from ...infra.llm import batch_condense_descriptions, call_llm_describe
from ..arg_validator import validate_tool_args
from ...infra.concurrency import (
    AsyncTaskDispatcher,
    KeywordSearchTask,
    ScanSegmentTask,
    TrendingPeriodTask,
    _remove_checkpoint,
    _save_checkpoint,
    _resolve_growth,
)

logger = logging.getLogger("hot_projects")


def _normalize_star_range(min_star: int, max_star: int) -> tuple[int, int]:
    """自动修正反向的 star 扫描区间。"""
    low = min_star
    high = max_star
    if high < low:
        low, high = high, low
    return low, high


def _resolve_future_or_default(label: str, future: Future, default):
    """读取并发任务结果，异常时回退默认值。"""
    try:
        result = future.result()
        return default if result is None else result
    except Exception as e:
        logger.warning("describe_project 上下文抓取失败: %s, error=%s", label, e)
        return default


def _default_async_worker_count(token_count: int) -> int:
    """默认协程消费者数量：任务级 token 模式下与 token 数对齐。"""
    return max(1, token_count)


def _resolve_async_worker_count(token_mgr: GitHubTokenPool) -> int:
    """兼容真实 token 池与测试 mock，解析调度器 worker 数。"""
    token_count = getattr(token_mgr, "token_count", None)
    if isinstance(token_count, int):
        return _default_async_worker_count(token_count)
    tokens = getattr(token_mgr, "tokens", [])
    return _default_async_worker_count(len(tokens))


def _resolve_dispatcher_token_pool(token_mgr: GitHubTokenPool) -> AsyncTokenPool:
    """真实运行时复用 GitHubTokenPool；测试 mock 回退为临时 AsyncTokenPool。"""
    if isinstance(token_mgr, AsyncTokenPool):
        return token_mgr
    return AsyncTokenPool(list(getattr(token_mgr, "tokens", [])))


PAGE_COMPENSATION_ROUNDS = 3          # 失败页最多补偿轮数（1 轮时限流期会残留几十页）
PAGE_COMPENSATION_MAX_WAIT = 150.0    # 每轮补偿前等 token 冷却的上限（秒）

# 增长"未决"哨兵：缺这个仓库的历史快照，算不出窗口增长。区别于 -1（出错）和 0（真没涨）：
# 未决的仓库不写 DB、不进候选池，也不当成 0 增长参与排序——否则新接入的仓库会被判死。
GROWTH_ESTIMATION_UNRESOLVED = -2


async def _compensate_failed_pages(
    dispatcher,
    token_pool: AsyncTokenPool,
    tasks: list,
    clone_for_retry,
    format_failure,
    label: str,
) -> None:
    """重跑关键词搜索/星段扫描的失败页，每轮开跑前先等 token 脱离限流冷却。

    只补偿一轮、且紧接着在冷却期内重跑时，任务一命中限流就把剩余页整批丢回失败集，
    补偿等于白跑（2026-07-22/29 两期各残留 80/74 个失败页，约 1800 个仓库没被收集）。
    """
    pending = [task for task in tasks if task.failed_pages]
    for round_no in range(1, PAGE_COMPENSATION_ROUNDS + 1):
        if not pending:
            return
        failed_page_count = sum(len(task.failed_pages) for task in pending)
        retry_tasks = [clone_for_retry(task, list(task.failed_pages), round_no) for task in pending]
        cooldown = min(token_pool.seconds_until_all_cool(), PAGE_COMPENSATION_MAX_WAIT)
        logger.warning(
            f"{label}发现 {failed_page_count} 个失败页，提交 {len(retry_tasks)} 个页级补偿任务"
            f"（第 {round_no}/{PAGE_COMPENSATION_ROUNDS} 轮，先等 token 冷却 {cooldown:.0f}s）。"
        )
        if cooldown > 0:
            await asyncio.sleep(cooldown)

        for task in retry_tasks:
            await dispatcher.submit(task)
        await dispatcher.wait_all_done()
        await dispatcher.drain_results()

        pending = [task for task in retry_tasks if task.failed_pages]

    final_failed = [format_failure(task, page) for task in pending for page in task.failed_pages]
    if final_failed:
        failed_preview = ", ".join(final_failed[:10])
        if len(final_failed) > 10:
            failed_preview += ", ..."
        logger.error(
            f"{label}补偿 {PAGE_COMPENSATION_ROUNDS} 轮后仍有 {len(final_failed)} 个失败页，"
            f"结果可能不完整: {failed_preview}"
        )


def _run_coroutine_sync(coro):
    """在同步上下文执行协程；若当前线程已有事件循环则转到子线程执行。"""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)

    result: dict[str, object] = {}
    error: dict[str, BaseException] = {}

    def _runner() -> None:
        try:
            result["value"] = asyncio.run(coro)
        except BaseException as exc:  # pragma: no cover - defensive passthrough
            error["value"] = exc

    thread = threading.Thread(target=_runner, daemon=True)
    thread.start()
    thread.join()

    if "value" in error:
        raise error["value"]

    return result.get("value")


def trending_repo_to_search_repo(repo: dict) -> dict:
    """将 Trending 返回项转换为内部统一 repo 结构。"""
    full_name = repo["full_name"]
    return {
        "full_name": full_name,
        "star": repo["star"],
        "description": repo.get("description", ""),
        "language": repo.get("language", ""),
        "_raw": {
            "full_name": full_name,
            "stargazers_count": repo["star"],
            "forks_count": repo.get("forks", 0),
            "description": repo.get("description", ""),
            "language": repo.get("language", ""),
            "topics": [],
        },
    }


# ══════════════════════════════════════════════════════════════
# Tool 实现
# ══════════════════════════════════════════════════════════════


def search_by_keywords(
    token_mgr: GitHubTokenPool,
    categories: list[str] | None = None,
    min_star: int = MIN_STAR,
    days_since_created: int | None = None,
    keywords: list[str] | None = None,
) -> dict:
    """
    Tool 1: 按关键词搜索 GitHub 热门仓库（并行）。

    使用 AsyncTaskDispatcher + KeywordSearchTask 并行搜索。

    搜索词来源（取并集、去重）：
      - categories: 预设类别（如 ["AI-Agent","AI-RAG"]）对应的整组关键词；
      - keywords:   LLM 根据用户自然语言挑选/补充的具体搜索词（数量受
                    MAX_DYNAMIC_SEARCH_KEYWORDS 限制，控制 Search 配额）；
      - 两者都未提供时，默认搜索全部预设类别。

    Args:
        token_mgr:        GitHubTokenPool 实例
        categories:       预设类别列表；None 表示不按类别选
        min_star:         项目最低 star 门槛
        days_since_created: 新项目判定窗口（天），指定后查询加 created:>=date 过滤
        keywords:         显式搜索词列表（LLM 补充）
    """
    from datetime import timedelta

    validated = validate_tool_args(
        "search_by_keywords",
        {
            "categories": categories,
            "min_star": min_star,
            "days_since_created": days_since_created,
        },
    )
    categories = validated.get("categories")
    min_star = validated.get("min_star", MIN_STAR)
    days_since_created = validated.get("days_since_created")

    # 显式关键词：去空格、去重（大小写不敏感）、限量
    explicit: list[str] = []
    seen_terms: set[str] = set()
    for kw in (keywords or []):
        if not isinstance(kw, str):
            continue
        term = kw.strip()
        key = term.lower()
        if term and key not in seen_terms:
            seen_terms.add(key)
            explicit.append(term)
    explicit = explicit[:MAX_DYNAMIC_SEARCH_KEYWORDS]

    # 预设来源：选了类别用对应组；否则——有显式词则只搜显式词，无则搜全部类别
    if categories:
        base_dict = {k: v for k, v in SEARCH_KEYWORDS.items() if k in categories}
        if not base_dict and not explicit:
            return {"repos": [], "total": 0, "categories_searched": [],
                    "error": f"未找到匹配类别，可用类别: {list(SEARCH_KEYWORDS.keys())}"}
    elif explicit:
        base_dict = {}
    else:
        base_dict = SEARCH_KEYWORDS

    # 扁平化为 (keyword, category)，跨组去重
    search_terms: list[tuple[str, str]] = []
    for cat, kws in base_dict.items():
        for kw in kws:
            key = kw.strip().lower()
            if key and key not in seen_terms:
                seen_terms.add(key)
                search_terms.append((kw, cat))
    for term in explicit:
        search_terms.append((term, "custom"))

    # 新项目模式：计算创建时间截止日期
    created_after = ""
    if days_since_created is not None:
        cutoff = datetime.now(timezone.utc) - timedelta(days=days_since_created)
        created_after = cutoff.strftime("%Y-%m-%d")

    raw_repos: dict[str, dict] = {}
    total_keywords = len(search_terms)

    # ── 并行搜索：提交 KeywordSearchTask 到异步调度器 ──
    async def _run_keyword_tasks() -> None:
        async with build_github_async_client(timeout_seconds=60.0) as async_client:
            token_pool = _resolve_dispatcher_token_pool(token_mgr)
            dispatcher = AsyncTaskDispatcher(
                token_pool=token_pool,
                worker_count=_resolve_async_worker_count(token_mgr),
            )
            await dispatcher.start()
            try:
                keyword_idx = 0
                keyword_tasks: list[KeywordSearchTask] = []
                for keyword, category in search_terms:
                    keyword_idx += 1
                    task = KeywordSearchTask(
                        _token_mgr=token_mgr,
                        keyword=keyword,
                        category=category,
                        keyword_idx=keyword_idx,
                        total_keywords=total_keywords,
                        created_after=created_after,
                        min_star=min_star,
                        _async_http_client=async_client,
                        _raw_repos=raw_repos,
                    )
                    keyword_tasks.append(task)
                    await dispatcher.submit(task)
                await dispatcher.wait_all_done()
                await dispatcher.drain_results()

                await _compensate_failed_pages(
                    dispatcher,
                    token_pool,
                    keyword_tasks,
                    lambda task, pages, round_no: KeywordSearchTask(
                        _token_mgr=token_mgr,
                        keyword=task.keyword,
                        category=task.category,
                        keyword_idx=task.keyword_idx,
                        total_keywords=task.total_keywords,
                        created_after=task.created_after,
                        min_star=task.min_star,
                        page_numbers=pages,
                        retry_round=round_no,
                        _async_http_client=async_client,
                        _raw_repos=raw_repos,
                    ),
                    lambda task, page: f"{task.keyword}/{task.category}/page={page}",
                    "关键词搜索",
                )
            finally:
                await dispatcher.shutdown()

    _run_coroutine_sync(_run_keyword_tasks())

    # ── 转换为返回格式 ──
    repos: list[dict] = []
    for fn, info in raw_repos.items():
        repo_item = info["repo_item"]
        star = info["star"]
        if star < min_star:
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
        "categories_searched": sorted({cat for _, cat in search_terms}),
        "keywords_searched": total_keywords,
        "_raw_repos": repos,
    }


def scan_star_range(
    token_mgr: GitHubTokenPool,
    min_star: int = MIN_STAR,
    max_star: int = MAX_STAR,
    seen_repos: set[str] | None = None,
    days_since_created: int | None = None,
) -> dict:
    """
    Tool 2: 按 star 范围扫描仓库（并行）。

    使用 AsyncTaskDispatcher + ScanSegmentTask 并行扫描各子区间。

    阶段隔离：
    Phase 0 — 串行：auto_split_star_range 递归分段（主线程优先 token_idx=0，限流时自动切换其他 token）
      Phase 1 — 并行：ScanSegmentTask 提交到 AsyncTaskDispatcher，N Worker 并行扫描

    Args:
        min_star:         项目最低 star 门槛（扫描区间下界）
        max_star:         扫描区间上限
        seen_repos:       已扫描过的仓库集合（用于去重）
        days_since_created: 新项目判定窗口（天），指定后在查询条件中加入 created:>=date 过滤项目
    """
    from datetime import timedelta

    validated = validate_tool_args(
        "scan_star_range",
        {
            "min_star": min_star,
            "max_star": max_star,
            "days_since_created": days_since_created,
        },
    )
    min_star, max_star = _normalize_star_range(
        validated.get("min_star", MIN_STAR),
        validated.get("max_star", MAX_STAR),
    )
    days_since_created = validated.get("days_since_created")

    if seen_repos is None:
        seen_repos = set()

    # 新项目模式：计算创建时间截止日期
    created_after = ""
    extra_query = ""
    if days_since_created is not None:
        cutoff = datetime.now(timezone.utc) - timedelta(days=days_since_created)
        created_after = cutoff.strftime("%Y-%m-%d")
        extra_query = f"created:>={created_after}"

    # ── Phase 0: 串行分段（主线程优先 token_idx=0，必要时自动切换其他 token） ──
    segments = auto_split_star_range(
        token_mgr, min_star, max_star, token_idx=0, extra_query=extra_query
    )
    raw_repos: dict[str, dict] = {}

    # ── Phase 1: 并行扫描各子区间（异步调度器）──
    async def _run_scan_tasks() -> None:
        async with build_github_async_client(timeout_seconds=60.0) as async_client:
            token_pool = _resolve_dispatcher_token_pool(token_mgr)
            dispatcher = AsyncTaskDispatcher(
                token_pool=token_pool,
                worker_count=_resolve_async_worker_count(token_mgr),
            )
            await dispatcher.start()
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
                        min_star=min_star,
                        _async_http_client=async_client,
                        _raw_repos=raw_repos,
                    )
                    segment_tasks.append(task)
                    await dispatcher.submit(task)

                await dispatcher.wait_all_done()
                await dispatcher.drain_results()

                await _compensate_failed_pages(
                    dispatcher,
                    token_pool,
                    segment_tasks,
                    lambda task, pages, round_no: ScanSegmentTask(
                        _token_mgr=token_mgr,
                        seg_idx=task.seg_idx,
                        low=task.low,
                        high=task.high,
                        total_segments=task.total_segments,
                        created_after=task.created_after,
                        min_star=task.min_star,
                        page_numbers=pages,
                        retry_round=round_no,
                        _async_http_client=async_client,
                        _raw_repos=raw_repos,
                    ),
                    lambda task, page: f"stars:{task.low}..{task.high}/page={page}",
                    "区间扫描",
                )
            finally:
                await dispatcher.shutdown()

    _run_coroutine_sync(_run_scan_tasks())

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


def check_repo_growth(
    token_mgr: GitHubTokenPool,
    repo: str,
    db: dict | None = None,
    growth_calc_days: int = GROWTH_CALC_DAYS,
) -> dict:
    """
    Tool 3: 查询单个仓库近期 star 增长，实时获取项目详情并生成 LLM 描述。

    star 数实时取，增长靠每日快照减法：当前 star − T−N 那天快照里的 star。
    快照里没有这个仓库（刚建、或 star 没到每日发现门槛）→ 增长未决，如实报未决。
    DB 仅用于读取已有描述缓存和补充静态元数据。

    Args:
        repo: "owner/repo" 格式
        db:   DB 字典（可选，仅用于读取描述缓存）
        growth_calc_days: 增长统计窗口（天）
    """
    validated = validate_tool_args(
        "check_repo_growth",
        {
            "repo": repo,
            "growth_calc_days": growth_calc_days,
        },
    )
    repo = validated.get("repo", repo)
    growth_calc_days = validated.get("growth_calc_days", GROWTH_CALC_DAYS)

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

    # 增长 = 当前 star − T−N 那天快照里的 star。找不到锚点就如实报未决，
    # 不再回退任何实时估算：stargazers 列表 2026-06-30 起对他人仓库一律 404。
    target_day = utc_today() - timedelta(days=growth_calc_days)
    anchor = anchor_for_window(growth_calc_days)
    anchor_star = anchor.stars.get(repo) if anchor else None

    if anchor_star is None:
        growth = GROWTH_ESTIMATION_UNRESOLVED
        method = "每日快照减法(未决)"
    else:
        # 锚点可能顺延一两天（漏采），按实际天数报口径，否则速率会被算虚高。
        growth_calc_days = anchor.window_days
        growth = current_star - anchor_star
        method = f"每日快照减法（{anchor.day} → 今天，{growth_calc_days} 天）"

    if growth == GROWTH_ESTIMATION_UNRESOLVED:
        growth_value = None
        growth_status = "snapshot_unresolved"
        meets_threshold = False
        growth_warning = (
            f"缺少 {target_day} 前后 {SNAPSHOT_ANCHOR_TOLERANCE_DAYS} 天内的每日快照，"
            "或该仓库当时还未进入快照范围（新建 / star 未达每日发现门槛），无法算出窗口增长。"
        )
    else:
        growth_value = growth
        growth_status = "ok"
        meets_threshold = growth >= STAR_GROWTH_THRESHOLD
        growth_warning = ""

    # 单仓库查询：不读 DB desc，始终实时抓取并生成描述
    html_url = repo_item.get("html_url", f"https://github.com/{repo}")
    repo_info = {
        "gh_desc": repo_item.get("description", ""),
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
        "growth_calc_days": growth_calc_days,
        "method": method,
        "meets_threshold": meets_threshold,
        "warning": growth_warning,
        "language": repo_item.get("language", ""),
        "gh_desc": (repo_item.get("description") or "")[:200],
        "created_at": repo_item.get("created_at", ""),
        "topics": repo_item.get("topics", []),
        "description": description or "描述生成失败",
    }


def batch_check_growth(
    token_mgr: GitHubTokenPool,
    repos: list[dict],
    db: dict,
    growth_threshold: int = STAR_GROWTH_THRESHOLD,
    days_since_created: int | None = None,
    growth_calc_days: int = GROWTH_CALC_DAYS,
    force_refresh: bool = False,
    window_specified: bool = True,
    candidate_log_threshold: int | None = None,
) -> dict:
    """
    Tool 4: 批量计算仓库增长并筛选候选。

    全程零请求：增长由每日快照减法/DB 差值算出（见 _resolve_growth）。
    当 days_since_created 指定时，先按创建时间筛选新项目，只对新项目计算增长。

    增长计算策略：
    - 优先每日快照锚点（T−N 那天的 star），全部仓库共用同一锚点
    - 退而用 DB 差值/窗口内新建/按快照年龄折算
    - 四条路都不成立 → 记未决，不进候选池（不当成 0 增长）
    - force_refresh=True（仅定时脚本）：刷新DB快照 + 启用 checkpoint

    DB写入权限（can_write_db）：
    - 定时脚本 force_refresh=True → 允许刷新DB快照（seeding）
    - 其他场景（force_refresh=False）→ 不刷新快照；desc 不在此写，
      由 ranking 复合工具完成后经 save_db_desc_only 持久化

    Args:
        repos:            仓库列表（含 full_name, star, _raw）
        db:               DB 字典
        growth_threshold: 增长阈值
        days_since_created: 新项目判定窗口（天），None 则不做创建时间筛选（全量计算）
        growth_calc_days: 增长统计窗口（天）
        force_refresh:    定时脚本传入 True 以刷新DB快照；Agent 始终传入 False
        window_specified: 调用方是否显式指定了 growth_calc_days
        candidate_log_threshold: [OK] 候选 日志的展示阈值（仅 growth >= 此值才打印）。
                          None 时回退为 growth_threshold。用于 growth_threshold=0
                          全量收录候选池、但日志只显示达标候选的场景。
    """
    from datetime import timedelta

    validated = validate_tool_args(
        "batch_check_growth",
        {
            "growth_threshold": growth_threshold,
            "days_since_created": days_since_created,
            "growth_calc_days": growth_calc_days,
        },
    )
    growth_threshold = validated.get("growth_threshold", STAR_GROWTH_THRESHOLD)
    days_since_created = validated.get("days_since_created")
    growth_calc_days = validated.get("growth_calc_days", GROWTH_CALC_DAYS)
    # force_refresh 不在 schema 中，由定时脚本内部传递，跳过验证
    window_specified = bool(window_specified)

    # 新项目榜与综合榜走同一套本地定案（详见 _resolve_growth_without_timestamps 的四条路）；
    # is_hot_new 只影响「未指定窗口时是否自动取 DB 年龄」和 checkpoint 是否启用。
    is_hot_new = days_since_created is not None

    # DB写入权限：只有定时刷新模式（force_refresh）才写 DB 快照；Agent/其他通道一律不写。
    can_write_db = force_refresh

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
                min_star=0,
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

    # ── 在 seeding 覆盖前，捕获旧快照（star + refreshed_at）供 DB 差值使用 ──
    # 差值 = current_star − 旧star，且需用旧 refreshed_at 判断窗口匹配；
    # 下面的 seeding 会把快照刷成当前值，因此必须先在这里留底。
    prev_snapshot: dict[str, dict] = {}
    for fn in raw_repos:
        proj = db_projects.get(fn)
        if isinstance(proj, dict) and "star" in proj:
            prev_snapshot[fn] = {
                "star": proj.get("star"),
                "refreshed_at": proj.get("refreshed_at", ""),
            }

    seeded_count = 0
    if can_write_db:
        for fn, info in raw_repos.items():
            repo_item = info.get("repo_item", {})
            if info.get("created_at") and not repo_item.get("created_at"):
                repo_item["created_at"] = info["created_at"]
            update_db_project(db_projects, fn, info.get("star", 0), repo_item)
            seeded_count += 1
        if seeded_count:
            logger.info(f"刷新模式:  当前star数 更新已有{seeded_count} 个项目的 DB 快照基线（refreshed_at=now）。")

    # ── 新项目前置筛选：仅保留创建时间在窗口内的仓库 ──
    skipped_count = 0
    if days_since_created is not None:
        cutoff = datetime.now(timezone.utc) - timedelta(days=days_since_created)
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
            f"新项目前置筛选(<={days_since_created}天): "
            f"原 {len(raw_repos)} 个 → 保留 {len(filtered)} 个, "
            f"跳过 {skipped_count} 个"
        )
        raw_repos = filtered

    candidate_map: dict[str, dict] = {}
    growth_ctx = {
        "checkpoint": None,
        "db_projects": db.get("projects", {}),
        "candidate_map": candidate_map,
        "growth_threshold": growth_threshold,
        # 候选池全量收录（growth_threshold 可为 0 以支持分阶段缓存复用），
        # 但 [OK] 候选 日志仅展示达标候选（candidate_log_threshold），保持"候选=达标"语义。
        "candidate_log_threshold": (
            candidate_log_threshold if candidate_log_threshold is not None else growth_threshold
        ),
        "window_specified": window_specified,
        "growth_calc_days": growth_calc_days,
        "is_hot_new": is_hot_new,
        "prev_snapshot": prev_snapshot,
        # checkpoint 仅用于定时长跑（force_refresh）断点续传；Agent 不碰 checkpoint 文件。
        "use_checkpoint": can_write_db and not is_hot_new,
        "checkpoint_dirty": [False],
    }

    # 增长全靠快照/DB 算术定案，没有请求可发——不再起 AsyncTaskDispatcher。
    checkpoint = _resolve_growth(raw_repos, db, candidate_map, growth_ctx)
    if growth_ctx["checkpoint_dirty"][0]:
        _save_checkpoint(checkpoint)
    _remove_checkpoint()

    db_updated = bool(can_write_db)
    effective_time_window = growth_ctx.get("effective_growth_calc_days", growth_calc_days)

    return {
        "candidates": candidate_map,
        "total_checked": len(raw_repos),
        "total_input": len(repos),
        "candidates_count": len(candidate_map),
        "unresolved_count": growth_ctx.get("unresolved_count", 0),
        "skipped_by_creation_time": skipped_count,
        "threshold": growth_threshold,
        "db_updated": db_updated,
        "seeded_snapshot_count": seeded_count,
        "db_diff_count": growth_ctx.get("db_diff_count", 0),
        "resumed_count": growth_ctx.get("resumed_count", 0),
        "growth_calc_days": effective_time_window,
        "requested_growth_calc_days": growth_calc_days,
    }


def describe_project(repo: str, db: dict, token_mgr: GitHubTokenPool | None = None) -> dict:
    """
    Tool 6: 调用 LLM 为单个项目生成描述。

    Args:
        repo: "owner/repo"
        db:   DB 字典
        token_mgr: 可选，提供后将实时拉取 GitHub API 丰富上下文
    """
    db_projects = db.get("projects", {})
    saved = db_projects.get(repo, {})

    parts = repo.split("/", 1)
    if len(parts) != 2:
        return {"error": f"仓库格式错误，应为 owner/repo: {repo}"}
    owner, repo_name = parts

    existing = str(saved.get("desc", "") or "").strip()

    # 有 desc 直接使用
    if existing:
        return {
            "repo": repo,
            "description": existing,
            "source": "DB缓存",
        }

    # 无 desc，需要重新获取（但不写入 DB，因为这是其他通道）
    if token_mgr is None:
        html_url = f"https://github.com/{repo}"
        desc = call_llm_describe(repo, saved, html_url, detail_level="detailed")
        # 注意：其他通道不写入 DB，只返回结果
        if desc:
            return {
                "repo": repo,
                "description": desc,
                "source": "LLM生成",
                "note": "描述已生成但未写入DB（其他通道只读不写）",
            }
        return {
            "repo": repo,
            "description": "描述生成失败",
            "source": "LLM生成",
        }

    repo_item = fetch_repo_info(token_mgr, owner, repo_name, token_idx=0)
    if not repo_item:
        return {
            "error": f"未找到仓库: {repo}（可能不存在或为私有仓库）",
            "hint": "请确认仓库名，或稍后重试。",
        }

    html_url = repo_item.get("html_url", f"https://github.com/{repo}")

    with ThreadPoolExecutor(max_workers=3) as executor:
        readme_future = executor.submit(
            fetch_repo_readme_excerpt,
            token_mgr,
            owner,
            repo_name,
            0,
        )
        releases_future = executor.submit(
            fetch_repo_recent_releases,
            token_mgr,
            owner,
            repo_name,
            0,
            5,
        )
        commits_future = executor.submit(
            fetch_repo_recent_commits,
            token_mgr,
            owner,
            repo_name,
            0,
            10,
        )

        readme = _resolve_future_or_default("readme", readme_future, {})
        releases = _resolve_future_or_default("releases", releases_future, [])
        commits = _resolve_future_or_default("commits", commits_future, [])

    logger.info(
        "[Tool describe_project] %s 上下文汇总: readme=%s, releases=%d, commits=%d",
        repo,
        bool(readme),
        len(releases),
        len(commits),
    )

    repo_info = {
        "gh_desc": repo_item.get("description", ""),
        "topics": repo_item.get("topics", []),
        "readme_url": f"{html_url}#readme",
        "readme_excerpt": readme.get("text", ""),
        "recent_releases": releases,
        "recent_commits": commits,
    }

    desc = call_llm_describe(repo, repo_info, html_url, detail_level="detailed")

    # 其他通道完全不写 DB（只读不写，包括元数据）
    if desc:
        return {
            "repo": repo,
            "description": desc,
            "source": "LLM生成",
            "note": "其他通道只读不写DB",
            "context_sources": {
                "repo_api": True,
                "readme_excerpt": bool(readme),
                "releases": len(releases),
                "commits": len(commits),
            },
        }
    elif existing:
        return {
            "repo": repo,
            "description": existing,
            "source": "DB缓存(LLM失败回退)",
            "warning": "实时上下文已拉取，但 LLM 生成失败，回退为 DB 缓存（可能是 brief desc）。",
            "context_sources": {
                "repo_api": True,
                "readme_excerpt": bool(readme),
                "releases": len(releases),
                "commits": len(commits),
            },
        }

    return {
        "repo": repo,
        "description": "描述生成失败",
        "source": "LLM生成",
        "context_sources": {
            "repo_api": True,
            "readme_excerpt": bool(readme),
            "releases": len(releases),
            "commits": len(commits),
        },
    }


def get_db_info(db: dict, repo: str | None = None) -> dict:
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

    age_days = get_db_age_days(db)
    return {
        "date": db.get("date", ""),
        "age_days": age_days,  # 由 date 实时计算的 DB 年龄（天）
        "fresh": age_days is not None and 0 <= age_days <= GROWTH_CALC_DAYS,
        "total_projects": len(db.get("projects", {})),
    }


def fetch_trending(
    trending_range: str = "weekly",
) -> dict:
    """
    Tool 9: 获取 GitHub Trending 仓库列表。

    参数说明：
      - "daily"   : 今日热门榜
      - "weekly"  : 本周热门榜（默认）
      - "monthly" : 本月热门榜
      - "all"     : 抓取三档（daily/weekly/monthly）并去重汇总，用于候选池补充

    使用场景：
      - 用户查看 Trending → 默认 "weekly"
      - 综合榜/新项目榜候选补充 → 使用 "all"
      - 用户指定"日榜/周榜/月榜" → 对应 "daily"/"weekly"/"monthly"
    """
    from ...datasource.github.trending import TRENDING_PERIODS, fetch_trending, merge_trending_period_results

    validated = validate_tool_args(
        "fetch_trending",
        {
            "trending_range": trending_range,
        },
    )
    trending_range = validated.get("trending_range", "weekly")

    period_label = {"daily": "今日增长", "weekly": "本周增长", "monthly": "本月增长"}

    if trending_range == "all":
        logger.info("[Tool fetch_trending] trending_range=all，抓取三档并去重")
        period_results: dict[str, list[dict]] = {}

        async def _run_trending_tasks() -> None:
            dispatcher = AsyncTaskDispatcher(
                token_pool=None,
                worker_count=len(TRENDING_PERIODS),
            )
            await dispatcher.start()
            try:
                for period in TRENDING_PERIODS:
                    await dispatcher.submit(
                        TrendingPeriodTask(period=period, _period_results=period_results)
                    )
                await dispatcher.wait_all_done()
                await dispatcher.drain_results()
            finally:
                await dispatcher.shutdown()

        _run_coroutine_sync(_run_trending_tasks())

        repos = merge_trending_period_results(period_results)
        display_repos = [
            {
                "full_name": r["full_name"],
                "star": r["star"],
                "forks": r["forks"],
                "periods": r.get("periods", []),
                "stars_by_period": r.get("stars_by_period", {}),
                "description": r.get("description", ""),
                "language": r["language"],
            }
            for r in repos
        ]
        # 用 LLM 批量浓缩描述
        condensed = batch_condense_descriptions(repos, max_chars=70)
        for i, r in enumerate(display_repos):
            r["description"] = condensed[i]

        return {
            "repos": display_repos,
            "count": len(display_repos),
            "trending_range": "all",
            "periods": ["daily", "weekly", "monthly"],
            "_raw_repos": repos,
        }
    else:
        logger.info(f"[Tool fetch_trending] trending_range={trending_range}，仅抓取该周期")
        repos = fetch_trending(since=trending_range)

        # 用 LLM 批量浓缩描述
        condensed = batch_condense_descriptions(repos, max_chars=70)

        growth_field = period_label.get(trending_range, "增长")
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

        return {
            "repos": display_repos,
            "count": len(display_repos),
            "trending_range": trending_range,
            "_raw_repos": repos,
        }
