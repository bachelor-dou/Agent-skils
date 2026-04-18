"""
Tool 参数规范化
===============
对 LLM 返回的 Tool 参数进行修正、补齐默认值、来源追踪，
并以单条完整日志输出所有生效参数。

关键设计：
  - 综合热榜与新项目榜单的数据均为三源（search + scan + trending）合一，
    采用不同规则筛选：综合榜按 log-score 排序，新项目榜按 created_at 过滤后按增长排序。
  - "近N天"默认指增长统计窗口，而非创建窗口。
"""

import json
import logging

from .intent_detector import is_new_project_workflow, is_comprehensive_ranking
from .param_extractor import (
    extract_time_window_days,
    extract_creation_window_days,
    has_explicit_creation_window,
    extract_top_n,
    has_explicit_top_n,
)
from ..common.config import (
    TIME_WINDOW_DAYS,
    NEW_PROJECT_DAYS,
    STAR_RANGE_MIN,
    STAR_RANGE_MAX,
    STAR_GROWTH_THRESHOLD,
    HOT_PROJECT_COUNT,
    HOT_NEW_PROJECT_COUNT,
)

logger = logging.getLogger("discover_hot")
_MISSING = object()


# ══════════════════════════════════════════════════════════════
# 参数规范化
# ══════════════════════════════════════════════════════════════

def normalize_tool_args(tool_name: str, args: dict, user_message: str) -> dict:
    """修正 LLM 易误判的榜单参数。

    主要修正：
      1. 用户未明确创建窗口时，剥离 LLM 擅自添加的 new_project_days
      2. 根据用户意图修正 rank mode（hot_new ↔ comprehensive）
    """
    normalized = dict(args)
    ranking_tools = {
        "search_hot_projects",
        "scan_star_range",
        "batch_check_growth",
        "rank_candidates",
    }

    if tool_name not in ranking_tools:
        return normalized

    has_creation_request = has_explicit_creation_window(user_message)
    has_new_project_intent = is_new_project_workflow(user_message)

    if "new_project_days" in normalized and not has_creation_request:
        logger.info(
            "[Agent] 当前请求未显式指定新项目创建窗口，忽略 new_project_days=%s。",
            normalized.get("new_project_days"),
        )
        normalized.pop("new_project_days", None)

    if tool_name == "rank_candidates":
        if has_new_project_intent:
            if normalized.get("mode") != "hot_new":
                logger.info("[Agent] 检测到明确新项目意图，将 rank mode 修正为 hot_new。")
            normalized["mode"] = "hot_new"
        elif normalized.get("mode") == "hot_new":
            logger.info("[Agent] 当前请求未显式指定新项目语义，将 rank mode 从 hot_new 修正为 comprehensive。")
            normalized["mode"] = "comprehensive"

    return normalized


def resolve_time_window_days(tool_name: str, args: dict, user_message: str) -> int:
    """解析增长统计窗口。

    优先级：tool_args > 用户消息提取 > 默认值
    """
    time_window_tools = {"check_repo_growth", "batch_check_growth", "generate_report"}
    if tool_name not in time_window_tools:
        return TIME_WINDOW_DAYS

    explicit_days = args.get("time_window_days")
    if isinstance(explicit_days, int) and explicit_days > 0:
        return explicit_days

    extracted_days = extract_time_window_days(user_message)
    if extracted_days is not None:
        return extracted_days

    return TIME_WINDOW_DAYS


def resolve_new_project_days(tool_name: str, args: dict, user_message: str) -> int | None:
    """解析新项目创建窗口。

    仅在 hot_new 工作流 / 明确创建窗口语义下生效。
    """
    hot_new_tools = {
        "search_hot_projects",
        "scan_star_range",
        "batch_check_growth",
        "rank_candidates",
    }
    if tool_name not in hot_new_tools:
        return None

    creation_window = extract_creation_window_days(user_message)
    if creation_window is not None:
        return creation_window

    explicit_days = args.get("new_project_days")
    if isinstance(explicit_days, int) and explicit_days > 0:
        return explicit_days

    if args.get("mode") == "hot_new":
        return NEW_PROJECT_DAYS

    if is_new_project_workflow(user_message):
        return NEW_PROJECT_DAYS

    return None


def resolve_workflow_mode(tool_name: str, args: dict, user_message: str) -> str | None:
    """返回当前 Tool 所处的榜单模式（comprehensive / hot_new / None）。"""
    ranking_tools = {
        "search_hot_projects",
        "scan_star_range",
        "batch_check_growth",
        "rank_candidates",
    }
    if tool_name not in ranking_tools:
        return None

    if tool_name == "rank_candidates":
        return args.get("mode", "comprehensive")

    return "hot_new" if is_new_project_workflow(user_message) else "comprehensive"


# ══════════════════════════════════════════════════════════════
# 来源追踪（用于日志标注每个参数值的来源）
# ══════════════════════════════════════════════════════════════

def workflow_mode_source(effective_mode: str | None, user_message: str) -> str:
    if effective_mode == "hot_new" and is_new_project_workflow(user_message):
        return "prompt"
    if effective_mode == "comprehensive" and is_comprehensive_ranking(user_message):
        return "prompt"
    return "default"


def time_window_source(args: dict, user_message: str) -> str:
    explicit_days = args.get("time_window_days")
    if isinstance(explicit_days, int) and explicit_days > 0:
        return "tool_args"
    if extract_time_window_days(user_message) is not None:
        return "prompt"
    return "default"


def creation_window_source(args: dict, effective_days: int | None, user_message: str) -> str:
    if effective_days is None:
        return "unused"
    if extract_creation_window_days(user_message) is not None:
        return "prompt"
    explicit_days = args.get("new_project_days")
    if isinstance(explicit_days, int) and explicit_days > 0:
        return "tool_args"
    return "default"


def arg_source(args: dict, key: str, effective_value: object = _MISSING) -> str:
    if key not in args:
        return "default"
    if effective_value is _MISSING or args.get(key) == effective_value:
        return "tool_args"
    return "normalized"


def mode_source(raw_args: dict, effective_mode: str, user_message: str) -> str:
    requested_mode = raw_args.get("mode")
    if requested_mode is None:
        return workflow_mode_source(effective_mode, user_message)
    if requested_mode == effective_mode:
        return "tool_args"
    return "normalized"


def top_n_source(raw_args: dict, effective_top_n: int, user_message: str) -> str:
    explicit_top_n = extract_top_n(user_message)
    if explicit_top_n is not None and explicit_top_n == effective_top_n:
        return "prompt"
    requested_top_n = raw_args.get("top_n")
    if requested_top_n is None:
        return "default"
    if requested_top_n == effective_top_n:
        return "tool_args"
    return "normalized"


def force_refresh_source(raw_args: dict, effective_force_refresh: bool) -> str:
    if bool(raw_args.get("force_refresh", False)):
        return "tool_args"
    if effective_force_refresh:
        return "prompt"
    return "default"


# ══════════════════════════════════════════════════════════════
# 综合日志
# ══════════════════════════════════════════════════════════════

def log_effective_tool_params(tool_name: str, params: list[tuple[str, object, str]]) -> None:
    """以单条日志输出 Tool 全部生效参数。

    格式示例：
      [Agent] Tool 生效参数: search_hot_projects | workflow_mode=comprehensive(default) | min_stars=1300(default) | ...
    """
    rendered: list[str] = []
    for key, value, source in params:
        if isinstance(value, (list, dict, tuple)):
            value_text = json.dumps(value, ensure_ascii=False, default=str)
        else:
            value_text = str(value)
        rendered.append(f"{key}={value_text}({source})")
    if rendered:
        logger.info("[Agent] Tool 生效参数: %s | %s", tool_name, " | ".join(rendered))


def log_request_summary(user_message: str) -> None:
    """在新一轮榜单构建时，输出用户请求的完整参数摘要（单条日志）。

    数据源说明：综合热榜与新项目榜单均为三源数据（search + scan + trending）合一，
    采用不同规则筛选。
    """
    mode = "hot_new" if is_new_project_workflow(user_message) else "comprehensive"
    rank_label = "新项目榜" if mode == "hot_new" else "综合榜"
    data_source_label = "search+scan+trending 三源合一"
    growth_window_days = extract_time_window_days(user_message) or TIME_WINDOW_DAYS
    requested_top_n = extract_top_n(user_message)
    effective_top_n = requested_top_n if requested_top_n is not None else (
        HOT_NEW_PROJECT_COUNT if mode == "hot_new" else HOT_PROJECT_COUNT
    )
    creation_window_days = extract_creation_window_days(user_message)
    if creation_window_days is None and mode == "hot_new":
        creation_window_days = NEW_PROJECT_DAYS
    creation_window_text = f"{creation_window_days}天" if creation_window_days is not None else "未启用"
    filter_rule = "created_at 过滤后按增长排序" if mode == "hot_new" else "log-score 综合评分排序"

    logger.info(
        "[Agent] 本轮请求参数: 榜单=%s | 数据源=%s | 筛选规则=%s | "
        "增长窗口=%s天 | 返回数量=%s | 创建窗口=%s | "
        "搜索最小Star=%s | 扫描范围=%s..%s | 增长阈值>=%s",
        rank_label,
        data_source_label,
        filter_rule,
        growth_window_days,
        effective_top_n,
        creation_window_text,
        STAR_RANGE_MIN,
        STAR_RANGE_MIN,
        STAR_RANGE_MAX,
        STAR_GROWTH_THRESHOLD,
    )
