"""
输入解析层 (Parsing Layer)
==========================
负责从用户自然语言 + LLM Tool 参数中提取、校验、规范化所有执行参数。

三个子模块：
  - intent_detector      — 纯函数：识别工作流意图（新项目/综合/实时刷新）
  - param_extractor      — 纯函数：从文本中提取时间窗口、数量等参数
  - tool_arg_normalizer  — 参数规范化 + 来源追踪 + 综合日志
"""

from .intent_detector import (
    is_new_project_workflow,
    is_comprehensive_ranking,
    is_realtime_refresh,
)
from .param_extractor import (
    latest_user_message,
    extract_time_window_days,
    extract_creation_window_days,
    has_explicit_creation_window,
    extract_top_n,
    has_explicit_top_n,
)
from .tool_arg_normalizer import (
    normalize_tool_args,
    resolve_time_window_days,
    resolve_new_project_days,
    resolve_workflow_mode,
    log_effective_tool_params,
    log_request_summary,
    workflow_mode_source,
    time_window_source,
    creation_window_source,
    arg_source,
    mode_source,
    top_n_source,
    force_refresh_source,
)

__all__ = [
    # intent
    "is_new_project_workflow",
    "is_comprehensive_ranking",
    "is_realtime_refresh",
    # param extraction
    "latest_user_message",
    "extract_time_window_days",
    "extract_creation_window_days",
    "has_explicit_creation_window",
    "extract_top_n",
    "has_explicit_top_n",
    # normalization
    "normalize_tool_args",
    "resolve_time_window_days",
    "resolve_new_project_days",
    "resolve_workflow_mode",
    # logging
    "log_effective_tool_params",
    "log_request_summary",
    # source tracking
    "workflow_mode_source",
    "time_window_source",
    "creation_window_source",
    "arg_source",
    "mode_source",
    "top_n_source",
    "force_refresh_source",
]
