"""
参数 Schema
===========
集中管理两类 schema：
1) TOOL_PARAM_SCHEMA：运行时参数类型、范围、默认值
2) TOOL_SCHEMAS：LLM function-calling 协议 schema
"""

from ..common.config import (
    MIN_STAR_FILTER,
    STAR_RANGE_MIN,
    STAR_RANGE_MAX,
    STAR_GROWTH_THRESHOLD,
    TIME_WINDOW_DAYS,
    HOT_PROJECT_COUNT,
    HOT_NEW_PROJECT_COUNT,
    NEW_PROJECT_DAYS,
    SEARCH_KEYWORDS,
)

# 运行时参数 schema：供 validate_tool_args 使用。
# 职责：定义每个 tool 的参数类型、边界和默认值。
TOOL_PARAM_SCHEMA: dict[str, dict] = {
    "search_hot_projects": {
        "categories": {"type": "list_str", "default": None},
        "project_min_star": {"type": "int", "min": 1, "default": MIN_STAR_FILTER},
        "max_pages": {"type": "int", "min": 1, "max": 10, "default": 3},
        "new_project_days": {"type": "int", "min": 1, "default": None},
    },
    "scan_star_range": {
        "min_star": {"type": "int", "min": 1, "default": STAR_RANGE_MIN},
        "max_star": {"type": "int", "min": 1, "default": STAR_RANGE_MAX},
        "new_project_days": {"type": "int", "min": 1, "default": None},
    },
    "check_repo_growth": {
        "repo": {"type": "str", "required": True},
        "time_window_days": {"type": "int", "min": 1, "default": TIME_WINDOW_DAYS},
    },
    "batch_check_growth": {
        "growth_threshold": {"type": "int", "min": 0, "default": STAR_GROWTH_THRESHOLD},
        "time_window_days": {"type": "int", "min": 1, "default": TIME_WINDOW_DAYS},
        "new_project_days": {"type": "int", "min": 1, "default": None},
        "force_refresh": {"type": "bool", "default": False},
    },
    "rank_candidates": {
        "mode": {
            "type": "enum",
            "choices": ["comprehensive", "hot_new"],
            "default": "comprehensive",
        },
        "top_n": {
            "type": "int",
            "min": 1,
            "max": 200,
            "default_by_mode": {
                "comprehensive": HOT_PROJECT_COUNT,
                "hot_new": HOT_NEW_PROJECT_COUNT,
            },
        },
        "new_project_days": {
            "type": "int",
            "min": 1,
            "default_by_mode": {
                "comprehensive": None,
                "hot_new": NEW_PROJECT_DAYS,
            },
        },
    },
    "describe_project": {
        "repo": {"type": "str", "required": True},
    },
    "generate_report": {},
    "get_db_info": {
        "repo": {"type": "str", "default": None},
    },
    "fetch_trending": {
        "since": {
            "type": "enum",
            "choices": ["daily", "weekly", "monthly"],
            "default": "weekly",
        },
        "include_all_periods": {"type": "bool", "default": False},
    },
}


# ══════════════════════════════════════════════════════════════════════════════════════
# Tool Schema 定义（OpenAI Function Calling 格式）供 agent 注册工具时传给模型。
# 职责：定义每个 tool 的名称、描述和 JSON parameters 协议。
# ══════════════════════════════════════════════════════════════════════════════════════

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
