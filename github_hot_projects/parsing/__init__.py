"""
输入解析层 (Parsing Layer)
==========================
负责 Tool 参数的类型校验和边界裁剪。

子模块：
  - schema          — 参数 schema 定义（类型、范围、默认值）
  - arg_validator   — 基于 schema 的机械校验 + 工具函数
  - route_helpers   — 路由解析纯函数（JSON 提取、意图/工具名规范化）
"""

from .arg_validator import validate_tool_args, log_validated_params
from .route_helpers import (
    extract_json_object,
    looks_like_structured_confirmation_text,
    normalize_intent_family,
    normalize_specified_params,
    normalize_tool_names,
    normalize_turn_kind,
    ordered_tool_names,
    sanitize_confirmation_fallback,
)
from .schema import TOOL_PARAM_SCHEMA, TOOL_SCHEMAS

__all__ = [
    "validate_tool_args",
    "log_validated_params",
    "extract_json_object",
    "looks_like_structured_confirmation_text",
    "normalize_intent_family",
    "normalize_specified_params",
    "normalize_tool_names",
    "normalize_turn_kind",
    "ordered_tool_names",
    "sanitize_confirmation_fallback",
    "TOOL_PARAM_SCHEMA",
    "TOOL_SCHEMAS",
]
