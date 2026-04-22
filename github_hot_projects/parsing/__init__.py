"""
输入解析层 (Parsing Layer)
==========================
负责 Tool 参数的类型校验和边界裁剪。

子模块：
  - schema          — 参数 schema 定义（类型、范围、默认值）
  - arg_validator   — 基于 schema 的机械校验 + 工具函数
"""

from .arg_validator import validate_tool_args, log_validated_params
from .schema import TOOL_PARAM_SCHEMA, TOOL_SCHEMAS

__all__ = [
    "validate_tool_args",
    "log_validated_params",
    "TOOL_PARAM_SCHEMA",
    "TOOL_SCHEMAS",
]
