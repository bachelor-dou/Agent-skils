"""
Prompt 参数 Schema
===================
用于构建面向 LLM 的参数语义上下文表（展示层 schema）。

说明：
- 该文件不做运行时校验；运行时校验以 parsing/schema.py 的 TOOL_PARAM_SCHEMA 为准。
- 本文件通过引用 TOOL_PARAM_SCHEMA 的边界与默认值，减少双维护漂移风险。
"""

from .schema import TOOL_PARAM_SCHEMA

SEARCH_PROJECT_MIN_STAR_SPEC = TOOL_PARAM_SCHEMA["search_hot_projects"]["project_min_star"]
SCAN_MIN_STAR_SPEC = TOOL_PARAM_SCHEMA["scan_star_range"]["min_star"]
SCAN_MAX_STAR_SPEC = TOOL_PARAM_SCHEMA["scan_star_range"]["max_star"]
BATCH_GROWTH_THRESHOLD_SPEC = TOOL_PARAM_SCHEMA["batch_check_growth"]["growth_threshold"]
BATCH_TIME_WINDOW_SPEC = TOOL_PARAM_SCHEMA["batch_check_growth"]["time_window_days"]
BATCH_FORCE_REFRESH_SPEC = TOOL_PARAM_SCHEMA["batch_check_growth"]["force_refresh"]
RANK_TOP_N_SPEC = TOOL_PARAM_SCHEMA["rank_candidates"]["top_n"]
RANK_NEW_PROJECT_DAYS_SPEC = TOOL_PARAM_SCHEMA["rank_candidates"]["new_project_days"]
TRENDING_SINCE_SPEC = TOOL_PARAM_SCHEMA["fetch_trending"]["since"]
TRENDING_INCLUDE_ALL_PERIODS_SPEC = TOOL_PARAM_SCHEMA["fetch_trending"]["include_all_periods"]

# Prompt 展示层 schema：用于生成 markdown 语义表并注入系统提示词。
PROMPT_PARAMETER_SCHEMA = [
    {
        "name": "categories",
        "type": "list[str]",
        "default": "全部类别",
        "description": "搜索的项目方向或主题。",
        "examples": ["AI Agent 方向", "数据库相关"],
    },
    {
        "name": "project_min_star",
        "type": f"int >= {SEARCH_PROJECT_MIN_STAR_SPEC['min']}",
        "default": SEARCH_PROJECT_MIN_STAR_SPEC["default"],
        "description": "关键词搜索时的最低 star 门槛。",
        "examples": ["至少 2000 star", "1000 星以上"],
    },
    {
        "name": "min_star / max_star",
        "type": f"int >= {SCAN_MIN_STAR_SPEC['min']}",
        "default": f"{SCAN_MIN_STAR_SPEC['default']}/{SCAN_MAX_STAR_SPEC['default']}",
        "description": "按 star 区间扫描项目时的上下界。",
        "examples": ["5000 到 20000 star 的项目"],
    },
    {
        "name": "time_window_days",
        "type": f"int >= {BATCH_TIME_WINDOW_SPEC['min']}",
        "default": BATCH_TIME_WINDOW_SPEC["default"],
        "description": "增长统计窗口，表示计算近 N 天的 star 增长。",
        "examples": ["近 10 天热榜", "最近 30 天增长"],
    },
    {
        "name": "new_project_days",
        "type": "int >= 1",
        "default": f"hot_new 默认 {RANK_NEW_PROJECT_DAYS_SPEC['default_by_mode']['hot_new']}，其他场景默认不过滤",
        "description": "创建时间窗口，只看近 N 天内创建的新项目。与增长窗口独立。",
        "examples": ["近 20 天内新创建的项目", "一个月内的新项目"],
    },
    {
        "name": "growth_threshold",
        "type": "int >= 0",
        "default": BATCH_GROWTH_THRESHOLD_SPEC["default"],
        "description": "增长量筛选门槛。",
        "examples": ["增长超过 500", "增长 >= 300"],
    },
    {
        "name": "top_n",
        "type": "int (1-200)",
        "default": (
            f"综合榜 {RANK_TOP_N_SPEC['default_by_mode']['comprehensive']} / "
            f"新项目榜 {RANK_TOP_N_SPEC['default_by_mode']['hot_new']}"
        ),
        "description": "返回数量（最多展示多少个结果）。",
        "examples": ["前 10 名", "榜单前 50"],
    },
    {
        "name": "mode",
        "type": "enum",
        "default": "comprehensive",
        "description": "榜单模式，综合热榜或新项目热榜。",
        "examples": ["综合榜", "新项目榜"],
    },
    {
        "name": "since",
        "type": "enum(daily|weekly|monthly)",
        "default": TRENDING_SINCE_SPEC["default"],
        "description": "查看 Trending 时使用的时间范围。",
        "examples": ["今日热门", "本周 Trending", "本月 Trending"],
    },
    {
        "name": "force_refresh",
        "type": "bool",
        "default": BATCH_FORCE_REFRESH_SPEC["default"],
        "description": "强制实时刷新，跳过已有缓存与差值。",
        "examples": ["实时热榜", "强制刷新"],
    },
    {
        "name": "repo",
        "type": "str",
        "default": "必填时由用户指定",
        "description": "目标仓库，格式为 owner/repo。",
        "examples": ["vllm-project/vllm"],
    },
    {
        "name": "include_all_periods",
        "type": "bool",
        "default": TRENDING_INCLUDE_ALL_PERIODS_SPEC["default"],
        "description": "是否同时抓取日榜、周榜、月榜。通常只在榜单补源时启用。",
        "examples": ["把日榜周榜月榜都补进来"],
    },
]

def _render_prompt_parameter_schema_context() -> str:
    lines = [
        "## 用户可自定义参数语义",
        "",
        "| 参数 | 类型 | 默认值 | 语义说明 | 用户表达示例 |",
        "|------|------|--------|----------|-------------|",
    ]
    for item in PROMPT_PARAMETER_SCHEMA:
        examples = "、".join(item["examples"])
        lines.append(
            f"| {item['name']} | {item['type']} | {item['default']} | {item['description']} | {examples} |"
        )

    lines.extend([
        "",
        "补充规则：",
        "- 用户说“近 7 天/近 10 天/近 30 天热榜”时，默认指增长统计窗口，也就是 time_window_days。",
        "- 只有“30 天内创建的新项目”这类明确创建时间语义，才映射为 new_project_days。",
        "- 榜单型任务中，time_window_days 和 new_project_days 可以同时出现，含义互不覆盖。",
    ])
    return "\n".join(lines)


PROMPT_PARAMETER_SCHEMA_CONTEXT = _render_prompt_parameter_schema_context()