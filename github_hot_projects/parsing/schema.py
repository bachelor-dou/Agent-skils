"""
参数 Schema
===========
每个 Tool 参数的类型、范围、默认值。纯数据定义，不含语义规则。
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
)

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
