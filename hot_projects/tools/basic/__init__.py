"""tools/basic — 基础能力层（纯函数，供上层工具复用）。

被 ranking 复合工具、独立工具(repo_growth/describe_project 等)与 Web 渲染复用；
彼此无强依赖。实现集中在 core.py（搜索/扫描/增长/描述/DB/Trending）、
report.py（报告生成）、scoring.py（评分）、report_parse.py（报告解析）。
"""

from .core import (
    search_by_keywords,
    scan_star_range,
    fetch_trending,
    trending_repo_to_search_repo,
    check_repo_growth,
    batch_check_growth,
    describe_project,
    get_db_info,
)

__all__ = [
    "search_by_keywords",
    "scan_star_range",
    "fetch_trending",
    "trending_repo_to_search_repo",
    "check_repo_growth",
    "batch_check_growth",
    "describe_project",
    "get_db_info",
]
