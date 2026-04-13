"""
Agent Tool 定义
===============
将热门项目发现的各功能封装为 Agent 可调用的 Tool。

每个 Tool 包含：
  - 函数实现（接收结构化参数，返回结构化结果）
  - OpenAI Function Calling 格式的 schema 定义
  - 人类可读的描述（供 LLM 理解 Tool 功能）

Tool 列表：
  1. search_hot_projects    — 按关键词类别搜索热门仓库
  2. scan_star_range        — 按 star 范围扫描仓库
  3. check_repo_growth      — 查询单个仓库近期增长
  4. batch_check_growth     — 批量计算仓库增长并筛选候选
  5. rank_candidates        — 对候选列表混合评分排序
  6. describe_project       — 调用 LLM 生成单个项目描述
  7. generate_report        — 生成完整 Markdown 报告
  8. get_db_info            — 查询 DB 状态和仓库信息
  9. full_discovery         — 完整执行一次热门项目发现流程
"""

import json
import logging
import time

from .config import (
    GITHUB_TOKENS,
    HOT_PROJECT_COUNT,
    MIN_STAR_FILTER,
    DEFAULT_SCORE_MODE,
    SEARCH_KEYWORDS,
    SEARCH_REQUEST_INTERVAL,
    STAR_GROWTH_THRESHOLD,
    STAR_RANGE_MAX,
    STAR_RANGE_MIN,
    TIME_WINDOW_DAYS,
)
from .db import load_db, save_db
from .github_api import auto_split_star_range, search_github_repos
from .growth_estimator import estimate_star_growth_binary
from .llm import call_llm_describe
from .token_manager import TokenManager

logger = logging.getLogger("discover_hot")


# ══════════════════════════════════════════════════════════════
# Tool 实现
# ══════════════════════════════════════════════════════════════


def tool_search_hot_projects(
    token_mgr: TokenManager,
    categories: list[str] | None = None,
    min_stars: int = MIN_STAR_FILTER,
    max_pages: int = 3,
) -> dict:
    """
    Tool 1: 按关键词类别搜索 GitHub 热门仓库。

    Args:
        token_mgr:  TokenManager 实例
        categories: 搜索类别列表（如 ["AI-Agent", "AI-RAG"]），None 则搜索全部
        min_stars:  最低 star 过滤线
        max_pages:  每个关键词搜索的最大页数

    Returns:
        {"repos": [{"full_name": ..., "star": ..., "description": ...}, ...],
         "total": int, "categories_searched": list}
    """
    if categories:
        keywords_dict = {k: v for k, v in SEARCH_KEYWORDS.items() if k in categories}
        if not keywords_dict:
            return {"repos": [], "total": 0, "categories_searched": [],
                    "error": f"未找到匹配类别，可用类别: {list(SEARCH_KEYWORDS.keys())}"}
    else:
        keywords_dict = SEARCH_KEYWORDS

    seen: set[str] = set()
    repos: list[dict] = []

    for category, keywords in keywords_dict.items():
        for keyword in keywords:
            for page in range(1, max_pages + 1):
                items = search_github_repos(token_mgr, keyword, page=page)
                if not items:
                    break
                for item in items:
                    full_name = item.get("full_name", "")
                    if not full_name or full_name in seen:
                        continue
                    seen.add(full_name)
                    star = item.get("stargazers_count", 0)
                    if star < min_stars:
                        continue
                    repos.append({
                        "full_name": full_name,
                        "star": star,
                        "description": (item.get("description") or "")[:200],
                        "language": item.get("language") or "",
                        "topics": item.get("topics") or [],
                        "_raw": item,  # 保留原始数据供后续使用
                    })
                time.sleep(SEARCH_REQUEST_INTERVAL)
            time.sleep(SEARCH_REQUEST_INTERVAL)

    # 去掉 _raw 再返回给 Agent（减少 token 消耗）
    display_repos = [{k: v for k, v in r.items() if k != "_raw"} for r in repos]
    return {
        "repos": display_repos,
        "total": len(repos),
        "categories_searched": list(keywords_dict.keys()),
        "_raw_repos": repos,  # 内部使用，不序列化给 LLM
    }


def tool_scan_star_range(
    token_mgr: TokenManager,
    min_star: int = STAR_RANGE_MIN,
    max_star: int = STAR_RANGE_MAX,
    seen_repos: set[str] | None = None,
) -> dict:
    """
    Tool 2: 按 star 范围扫描仓库。

    Args:
        min_star:    最低星数
        max_star:    最高星数
        seen_repos:  已扫描过的仓库集合（用于去重）
    """
    if seen_repos is None:
        seen_repos = set()

    segments = auto_split_star_range(token_mgr, min_star, max_star)
    repos: list[dict] = []

    for seg_idx, (low, high) in enumerate(segments, 1):
        query = f"stars:{low}..{high}"
        for page in range(1, 11):
            items = search_github_repos(
                token_mgr, query, page=page, sort="updated", auto_star_filter=False
            )
            if not items:
                break
            for item in items:
                full_name = item.get("full_name", "")
                if not full_name or full_name in seen_repos:
                    continue
                seen_repos.add(full_name)
                star = item.get("stargazers_count", 0)
                if star < MIN_STAR_FILTER:
                    continue
                repos.append({
                    "full_name": full_name,
                    "star": star,
                    "description": (item.get("description") or "")[:200],
                    "language": item.get("language") or "",
                    "_raw": item,
                })
            time.sleep(SEARCH_REQUEST_INTERVAL)

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
    Tool 3: 查询单个仓库近期 star 增长。

    Args:
        repo: "owner/repo" 格式
        db:   DB 字典（可选，提供则优先用差值法）
    """
    parts = repo.split("/", 1)
    if len(parts) != 2:
        return {"error": f"仓库格式错误，应为 owner/repo: {repo}"}

    owner, repo_name = parts

    # 先获取当前 star 数
    items = search_github_repos(
        token_mgr, f"repo:{repo}", page=1, per_page=1, auto_star_filter=False
    )
    if not items:
        return {"error": f"未找到仓库: {repo}"}

    current_star = items[0].get("stargazers_count", 0)

    # 尝试 DB 差值法
    if db and db.get("valid") and repo in db.get("projects", {}):
        saved_star = db["projects"][repo].get("star", 0)
        growth = current_star - saved_star
        method = "DB差值法"
    else:
        growth = estimate_star_growth_binary(token_mgr, owner, repo_name, current_star)
        method = "二分法/采样外推"

    return {
        "repo": repo,
        "current_star": current_star,
        "growth": growth,
        "time_window_days": TIME_WINDOW_DAYS,
        "method": method,
        "meets_threshold": growth >= STAR_GROWTH_THRESHOLD,
    }


def tool_batch_check_growth(
    token_mgr: TokenManager,
    repos: list[dict],
    db: dict,
    growth_threshold: int = STAR_GROWTH_THRESHOLD,
) -> dict:
    """
    Tool 4: 批量计算仓库增长并筛选候选。

    复用 pipeline.batch_growth_calc 的逻辑，保持一致性。

    Args:
        repos:            仓库列表（含 full_name, star, _raw）
        db:               DB 字典
        growth_threshold: 增长阈值
    """
    from .pipeline import batch_growth_calc

    # 构建 raw_repos 格式（batch_growth_calc 的输入）
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

    candidate_map: dict[str, dict] = {}
    batch_growth_calc(token_mgr, raw_repos, db, candidate_map)

    return {
        "candidates": candidate_map,
        "total_checked": len(repos),
        "candidates_count": len(candidate_map),
        "skipped": 0,
        "threshold": growth_threshold,
    }


def tool_rank_candidates(
    candidates: dict[str, dict],
    top_n: int = HOT_PROJECT_COUNT,
    mode: str = DEFAULT_SCORE_MODE,
) -> dict:
    """
    Tool 5: 对候选列表评分排序。

    Args:
        candidates: {full_name: {"growth": int, "star": int, "stars_today": int(可选)}}
        top_n:      取前 N 个
        mode:       评分模式 ("comprehensive" | "hot_new")
    """
    from .pipeline import step2_rank_and_select

    top = step2_rank_and_select(candidates, mode=mode)[:top_n]

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
) -> dict:
    """
    Tool 7: 生成完整 Markdown 报告。

    复用 pipeline.step3_generate_report 的逻辑。
    """
    from .pipeline import step3_generate_report
    report_path = step3_generate_report(top_projects, db)
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


def tool_full_discovery(token_mgr: TokenManager) -> dict:
    """
    Tool 9: 完整执行一次热门项目发现流程（等同原脚本 main()）。

    依次执行搜索 → 增长筛选 → 排序 → 报告。
    """
    from .pipeline import main as pipeline_main
    try:
        pipeline_main()
        return {"status": "completed"}
    except Exception as e:
        logger.error(f"完整发现流程执行失败: {e}")
        return {"status": "failed", "error": str(e)}


def tool_fetch_trending(
    since: str = "daily",
    language: str = "",
    spoken_language: str = "",
) -> dict:
    """
    Tool 10: 获取 GitHub Trending 仓库列表。

    两种使用路径：
      路径 1 — 直接展示：用户问 Trending 上有什么，直接返回列表
      路径 2 — 候选补充：返回的仓库可加入候选池走正常评分流程

    Args:
        since:           时间范围 ("daily" | "weekly" | "monthly")
        language:        编程语言筛选，如 "python"，""=全部
        spoken_language: 自然语言代码，如 "zh"，""=全部
    """
    from .github_trending import fetch_trending

    repos = fetch_trending(since=since, language=language, spoken_language=spoken_language)

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

    return {
        "repos": display_repos,
        "count": len(display_repos),
        "since": since,
        "language": language or "all",
        "_raw_repos": repos,  # 内部使用
    }


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
                        "description": f"最低star过滤线，默认{MIN_STAR_FILTER}",
                        "default": MIN_STAR_FILTER,
                    },
                    "max_pages": {
                        "type": "integer",
                        "description": "每个关键词搜索最大页数，默认3",
                        "default": 3,
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
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "check_repo_growth",
            "description": (
                "查询单个GitHub仓库近期star增长情况。"
                f"返回近{TIME_WINDOW_DAYS}天的star增长数、当前star总数、是否达到热门阈值。"
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
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "growth_threshold": {
                        "type": "integer",
                        "description": f"增长阈值，默认{STAR_GROWTH_THRESHOLD}",
                        "default": STAR_GROWTH_THRESHOLD,
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
                            "hot_new=新项目专榜（仅创建时间<=45天的新项目，按增长量排序）。"
                            "默认comprehensive。"
                        ),
                        "default": "comprehensive",
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
            "name": "full_discovery",
            "description": (
                "完整执行一次热门项目发现流程（搜索→增长计算→排序→LLM描述→报告）。"
                "等同于直接运行原脚本，耗时较长。"
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
                        "description": "时间范围，默认daily",
                        "default": "daily",
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
