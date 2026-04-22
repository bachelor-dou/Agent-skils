"""
Tool 参数校验
=============
基于 schema 做类型校验和边界裁剪。不猜语义，不否决 LLM 的参数。

规则：
  1. LLM 传了的参数：只做类型转换和边界裁剪，不删除
  2. LLM 没传的参数：填默认值
  3. 不在 schema 中的参数：丢弃（不向 Tool 函数传未知参数）
"""

import json
import logging

from .schema import TOOL_PARAM_SCHEMA

logger = logging.getLogger("discover_hot")


def validate_tool_args(tool_name: str, args: dict) -> dict:
    """校验并填充默认值，返回清洁参数。"""
    schema = TOOL_PARAM_SCHEMA.get(tool_name, {})
    result = {}

    for param_name, spec in schema.items():
        if param_name in args:
            result[param_name] = _coerce(args[param_name], spec, result)
        else:
            default = _get_default(spec, result)
            if default is not None:
                result[param_name] = default

    return result


def _get_default(spec: dict, current_args: dict):
    """获取默认值，支持 default_by_mode。"""
    if "default_by_mode" in spec:
        mode = current_args.get("mode", "comprehensive")
        return spec["default_by_mode"].get(mode)
    return spec.get("default")


def _coerce(value, spec: dict, current_args: dict):
    """类型转换和边界裁剪。"""
    vtype = spec.get("type", "str")

    if vtype == "int":
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return _get_default(spec, current_args)
        value = int(value)
        if "min" in spec:
            value = max(value, spec["min"])
        if "max" in spec:
            value = min(value, spec["max"])
        return value

    if vtype == "bool":
        return bool(value)

    if vtype == "enum":
        return value if value in spec.get("choices", []) else _get_default(spec, current_args)

    if vtype == "list_str":
        return value if isinstance(value, list) else _get_default(spec, current_args)

    # str
    return value


def log_validated_params(tool_name: str, raw_args: dict, validated_args: dict) -> None:
    """以单条日志输出 Tool 参数校验结果。"""
    parts: list[str] = []
    schema = TOOL_PARAM_SCHEMA.get(tool_name, {})

    for key, value in validated_args.items():
        if key in raw_args:
            if raw_args[key] == value:
                source = "llm"
            else:
                source = "coerced"
        else:
            source = "default"

        if isinstance(value, (list, dict, tuple)):
            value_text = json.dumps(value, ensure_ascii=False, default=str)
        else:
            value_text = str(value)
        parts.append(f"{key}={value_text}({source})")

    if parts:
        logger.info("[Agent] Tool 参数: %s | %s", tool_name, " | ".join(parts))
