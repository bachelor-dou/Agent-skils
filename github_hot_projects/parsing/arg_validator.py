"""
Tool 参数校验
=============
提供两种模式：
1) 宽松模式（validate_tool_args）：类型转换 + 边界裁剪 + 默认值填充
2) 严格模式（validate_tool_args_strict）：检测 LLM 显式传错参数，返回结构化错误
"""

import json
import logging
from typing import Any

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


def validate_tool_args_strict(tool_name: str, args: dict) -> tuple[dict, list[dict[str, Any]]]:
    """严格校验：仅用于识别 LLM 显式传错的参数。

    返回:
      - validated_args: 通过严格校验并填充默认值后的参数
      - errors: 结构化参数错误列表（为空表示可执行）
    """
    schema = TOOL_PARAM_SCHEMA.get(tool_name, {})
    result: dict[str, Any] = {}
    errors: list[dict[str, Any]] = []

    # 先处理 schema 内参数
    for param_name, spec in schema.items():
        if param_name in args:
            ok, coerced, reason = _coerce_strict(args[param_name], spec, result)
            if not ok:
                errors.append({
                    "param": param_name,
                    "reason": reason,
                    "received": args[param_name],
                })
                continue
            result[param_name] = coerced
        else:
            default = _get_default(spec, result)
            if default is not None:
                result[param_name] = default

    # 显式拒绝未知参数，避免模型幻觉参数被静默吞掉
    known = set(schema.keys())
    for unknown in sorted(set(args.keys()) - known):
        errors.append({
            "param": unknown,
            "reason": "unknown_parameter",
            "received": args[unknown],
        })

    return result, errors


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


def _coerce_strict(value, spec: dict, current_args: dict) -> tuple[bool, Any, str | None]:
    """严格类型校验：不做静默纠偏。"""
    vtype = spec.get("type", "str")

    if vtype == "int":
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return False, None, "expected_integer"
        parsed = int(value)
        if "min" in spec and parsed < spec["min"]:
            return False, None, f"must_be_gte_{spec['min']}"
        if "max" in spec and parsed > spec["max"]:
            return False, None, f"must_be_lte_{spec['max']}"
        return True, parsed, None

    if vtype == "bool":
        if not isinstance(value, bool):
            return False, None, "expected_boolean"
        return True, value, None

    if vtype == "enum":
        choices = spec.get("choices", [])
        if value not in choices:
            return False, None, f"must_be_one_of_{choices}"
        return True, value, None

    if vtype == "list_str":
        if not isinstance(value, list):
            return False, None, "expected_array_of_strings"
        if any(not isinstance(item, str) for item in value):
            return False, None, "expected_array_of_strings"
        return True, value, None

    # str
    if not isinstance(value, str):
        return False, None, "expected_string"
    return True, value, None


def log_validated_params(
    tool_name: str,
    llm_args: dict,
    prepared_args: dict,
    validated_args: dict,
) -> None:
    """以单条日志输出 Tool 参数校验结果。"""
    parts: list[str] = []

    for key, value in validated_args.items():
        if key in llm_args:
            if llm_args[key] == value:
                source = "llm"
            else:
                source = "coerced"
        elif key in prepared_args:
            if prepared_args[key] == value:
                source = "system"
            else:
                source = "system_coerced"
        else:
            source = "default"

        if isinstance(value, (list, dict, tuple)):
            value_text = json.dumps(value, ensure_ascii=False, default=str)
        else:
            value_text = str(value)
        parts.append(f"{key}={value_text}({source})")

    if parts:
        logger.info("[Agent] Tool 参数: %s | %s", tool_name, " | ".join(parts))
