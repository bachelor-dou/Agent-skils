"""capabilities — 基础工具层（纯函数）。

榜单流水线与原子工具的底层能力；彼此无强依赖，由上层 pipeline / tools 组合。
"""

from .collect import search_by_keywords, scan_star_range, fetch_trending, trending_repo_to_search_repo
from .growth import check_repo_growth, batch_check_growth
from .rank import rank_candidates
from .describe import describe_project, get_db_info, generate_report

__all__ = [
    "search_by_keywords",
    "scan_star_range",
    "fetch_trending",
    "trending_repo_to_search_repo",
    "check_repo_growth",
    "batch_check_growth",
    "rank_candidates",
    "describe_project",
    "get_db_info",
    "generate_report",
]
