"""
参数 Schema
===========
集中管理两类 schema：
1) TOOL_PARAM_SCHEMA：运行时参数类型、范围、默认值
2) AGENT_TOOL_SCHEMAS：LLM function-calling 协议 schema
"""

from ..config import (
    MIN_STAR,
    MAX_STAR,
    STAR_GROWTH_THRESHOLD,
    GROWTH_CALC_DAYS,
    HOT_PROJECT_COUNT,
    HOT_NEW_PROJECT_COUNT,
    DAYS_SINCE_CREATED,
    SEARCH_KEYWORDS,
)

# 运行时参数 schema：供 validate_tool_args 使用。
# 职责：定义每个 tool 的参数类型、边界和默认值。
TOOL_PARAM_SCHEMA: dict[str, dict] = {
    "search_by_keywords": {
        "categories": {"type": "list_str", "default": None},
        "min_star": {"type": "int", "min": 1, "default": MIN_STAR},
        "days_since_created": {"type": "int", "min": 1, "default": None},
    },
    "scan_star_range": {
        "min_star": {"type": "int", "min": 1, "default": MIN_STAR},
        "max_star": {"type": "int", "min": 1, "default": MAX_STAR},
        "days_since_created": {"type": "int", "min": 1, "default": None},
    },
    "check_repo_growth": {
        "repo": {"type": "str"},
        "growth_calc_days": {"type": "int", "min": 1, "default": GROWTH_CALC_DAYS},
    },
    "batch_check_growth": {
        "growth_threshold": {"type": "int", "min": 0, "default": STAR_GROWTH_THRESHOLD},
        "growth_calc_days": {"type": "int", "min": 1, "default": GROWTH_CALC_DAYS},
        "days_since_created": {"type": "int", "min": 1, "default": None},
    },
}


# ══════════════════════════════════════════════════════════════════════════════════════
# Agent 暴露层工具：复合榜单工具 + 原子工具
#   - 参数校验项并入 TOOL_PARAM_SCHEMA（供 validate_tool_args_strict 按 agent 工具名校验）
#   - LLM function-calling schema 见 AGENT_TOOL_SCHEMAS（供工具注册表）
# ══════════════════════════════════════════════════════════════════════════════════════

_RANK_COMMON_PARAMS = {
    "categories": {"type": "list_str", "default": None},
    "min_star": {"type": "int", "min": 1, "default": MIN_STAR},
    "growth_calc_days": {"type": "int", "min": 1, "default": None},
    "growth_threshold": {"type": "int", "min": 0, "default": STAR_GROWTH_THRESHOLD},
}

TOOL_PARAM_SCHEMA.update({
    "comprehensive_ranking": {
        **_RANK_COMMON_PARAMS,
        "max_star": {"type": "int", "min": 1, "default": MAX_STAR},
        "top_n": {"type": "int", "min": 1, "max": 200, "default": HOT_PROJECT_COUNT},
    },
    "hot_new_ranking": {
        **_RANK_COMMON_PARAMS,
        "max_star": {"type": "int", "min": 1, "default": MAX_STAR},
        "days_since_created": {"type": "int", "min": 1, "default": DAYS_SINCE_CREATED},
        "top_n": {"type": "int", "min": 1, "max": 200, "default": HOT_NEW_PROJECT_COUNT},
    },
    "keyword_ranking": {
        **_RANK_COMMON_PARAMS,
        "keywords": {"type": "list_str", "default": None},
        "topic": {"type": "str", "default": None},
        "top_n": {"type": "int", "min": 1, "max": 200, "default": HOT_PROJECT_COUNT},
    },
    "repo_growth": {
        "repo": {"type": "str"},
        "growth_calc_days": {"type": "int", "min": 1, "default": GROWTH_CALC_DAYS},
    },
    "describe_project": {"repo": {"type": "str"}},
    "repo_profile": {"repo": {"type": "str"}},
    "search_repos": {
        "query": {"type": "str"},
        "top_n": {"type": "int", "min": 1, "max": 20, "default": 5},
        "min_star": {"type": "int", "min": 0, "default": 0},
    },
    "star_trend": {"repo": {"type": "str"}},
    "analyze_report": {
        "name": {"type": "str", "default": None},
        "repo": {"type": "str", "default": None},
    },
    "get_db_info": {"repo": {"type": "str", "default": None}},
    "fetch_trending": {
        # 含 "all"：定时任务/综合榜管线传 "all" 表示日/周/月三榜合一去重；
        # 缺了它会被校验器打回默认 "weekly"，导致只抓周榜（与前面字面定义保持一致）。
        "trending_range": {"type": "enum", "choices": ["daily", "weekly", "monthly", "all"], "default": "weekly"},
    },
})


def _fn(name, description, properties, required=None):
    fn = {"name": name, "description": description,
          "parameters": {"type": "object", "properties": properties}}
    if required:
        fn["parameters"]["required"] = required
    return {"type": "function", "function": fn}


_categories_prop = {
    "type": "array", "items": {"type": "string"},
    "description": f"搜索类别，可选: {list(SEARCH_KEYWORDS.keys())}；不传=全部类别。",
}
_min_star_prop = {"type": "integer", "description": f"最低 star 门槛，默认{MIN_STAR}"}
_growth_calc_days_prop = {"type": "integer", "description": "增长统计窗口（天）；不传则综合/关键词榜用 DB 年龄窗口。与创建时间窗口独立。"}
_growth_threshold_prop = {"type": "integer", "description": f"增长入选阈值，默认{STAR_GROWTH_THRESHOLD}"}

AGENT_TOOL_SCHEMAS = [
    _fn("comprehensive_ranking",
        "【综合热榜·昂贵】跑完整发现流程(搜索+扫描+Trending→增长→排序→报告)输出综合 Top N。执行前请先回显参数并等用户确认『开始』。",
        {
            "categories": _categories_prop,
            "min_star": _min_star_prop,
            "max_star": {"type": "integer", "description": f"星段扫描上限，默认{MAX_STAR}"},
            "growth_calc_days": _growth_calc_days_prop,
            "growth_threshold": _growth_threshold_prop,
            "top_n": {"type": "integer", "description": f"返回前 N，默认{HOT_PROJECT_COUNT}"},
        }),
    _fn("hot_new_ranking",
        "【新项目热榜·昂贵】只看近 days_since_created 天内创建的新项目，按增长排序。执行前先回显参数等用户确认『开始』。",
        {
            "categories": _categories_prop,
            "min_star": _min_star_prop,
            "max_star": {"type": "integer", "description": f"星段扫描上限，默认{MAX_STAR}"},
            "days_since_created": {"type": "integer", "description": f"新项目创建时间窗口（天），默认{DAYS_SINCE_CREATED}"},
            "growth_calc_days": _growth_calc_days_prop,
            "growth_threshold": _growth_threshold_prop,
            "top_n": {"type": "integer", "description": f"返回前 N，默认{HOT_NEW_PROJECT_COUNT}"},
        }),
    _fn("keyword_ranking",
        "【关键词热榜·昂贵】按关键词搜索→增长→排序（不做星段扫描/Trending）。"
        "根据用户自然语言：从系统提示里的「关键词类别参考」挑出相关关键词，并补充未覆盖到的英文搜索词，一起传入 keywords；"
        "也可用 categories 选整组。执行前先回显参数等用户确认『开始』。",
        {
            "keywords": {
                "type": "array", "items": {"type": "string"},
                "description": "要搜索的具体英文关键词列表（从类别参考里挑 + 你的补充），如 [\"vector database\",\"voice assistant llm\"]。",
            },
            "topic": {
                "type": "string",
                "description": "本次搜索方向的简短中文概括，6 个字以内，用于报告标题点明方向，如『向量数据库』『AI语音助手』。",
            },
            "categories": _categories_prop,
            "min_star": _min_star_prop,
            "growth_calc_days": _growth_calc_days_prop,
            "growth_threshold": _growth_threshold_prop,
            "top_n": {"type": "integer", "description": f"返回前 N，默认{HOT_PROJECT_COUNT}"},
        }),
    _fn("repo_growth",
        "【单仓库增长】查单个仓库近期 star 增长。若精确仓库查不到，会返回相似候选供用户选择。",
        {
            "repo": {"type": "string", "description": "owner/repo（如 vllm-project/vllm）；也可只给项目名、拼错、或一句描述，会自动检索匹配，有歧义时返回候选。"},
            "growth_calc_days": {"type": "integer", "description": f"增长统计窗口（天），默认{GROWTH_CALC_DAYS}"},
        },
        required=["repo"]),
    _fn("describe_project",
        "【项目介绍】生成单个仓库的中文功能介绍。精确查不到会返回相似候选供选择。",
        {"repo": {"type": "string", "description": "owner/repo；也可只给项目名、拼错、或一句描述，会自动检索匹配，有歧义时返回候选。"}},
        required=["repo"]),
    _fn("repo_profile",
        "【项目画像取证】一次获取单仓库的原始证据：README 摘录、官方简介、topics、语言、star/forks/issues、"
        "创建/最近推送时间、release 节奏、近期提交（是否活跃维护）。只取证不归纳——功能清单、场景覆盖、"
        "上手方式、优缺点、活跃度判断由你基于返回内容自行提炼。用于了解单个项目或同类项目对比（各调一次）。",
        {"repo": {"type": "string", "description": "owner/repo；也可只给项目名或描述，自动检索匹配，有歧义时返回候选。"}},
        required=["repo"]),
    _fn("search_repos",
        "【按描述找项目】把用户的自然语言需求转成简洁的英文搜索词，去 GitHub 按 star 降序找 Top N 项目。"
        "适合『帮我找个手机远程控制 agent 的项目』这类『找到那个项目』的诉求——即时、轻量、不出榜单、不算增长。"
        "query 用 2-4 个核心英文关键词（可加引号词组），不要堆太多词以免零结果；可选 in: 限定符（默认已含 name/description/readme）。",
        {
            "query": {"type": "string", "description": "GitHub 搜索查询：由用户需求提炼的简洁英文关键词，如 'mobile remote control ai agent'。"},
            "top_n": {"type": "integer", "description": "返回前 N 个，默认 5，最多 20。"},
            "min_star": {"type": "integer", "description": "可选最低 star 门槛，默认 0（不限制）。想只看有名气的可设 1000。"},
        },
        required=["query"]),
    _fn("star_trend",
        "【star 轨迹】从历史周报推导某项目多周的总 star 与排名变化（本地读取，不联网），"
        "用于判断项目在涨/见顶/退烧。仅覆盖曾上过榜的项目；某周未上榜则该周缺点。",
        {"repo": {"type": "string", "description": "owner/repo（如 vllm-project/vllm）；也可只给项目名，按报告内名称匹配。"}},
        required=["repo"]),
    _fn("analyze_report",
        "【报告分析】读取已生成的榜单报告并分析（本地读取，不联网）。"
        "不传 name→列出可用报告；传 name（文件名或『最新』）→返回该报告的项目清单"
        "（排名/仓库/Star/增长/语言/主题），用于整体分析与筛选；"
        "再带 repo=owner/repo→返回该项目在报告中的完整分段内容，用于针对单个项目追问。",
        {
            "name": {"type": "string", "description": "报告文件名（如 2026-07-08.md）或『最新』；不传则列出全部报告。"},
            "repo": {"type": "string", "description": "可选，owner/repo；配合 name 获取该项目在报告中的完整分析分段。"},
        }),
    _fn("get_db_info",
        "【数据库查询】查本地 DB 概览或指定仓库缓存信息（不联网）。",
        {"repo": {"type": "string", "description": "可选，查特定仓库；不传返回概览。"}}),
    _fn("fetch_trending",
        "【Trending】获取 GitHub Trending 列表。",
        {"trending_range": {"type": "string", "enum": ["daily", "weekly", "monthly"],
                            "description": "daily/weekly(默认)/monthly", "default": "weekly"}}),
]
