"""
路由解析辅助函数
================
为 agent.py 的意图识别与参数提取阶段提供纯函数工具。

职责：
  - JSON 对象提取：从 LLM 输出文本中安全提取 JSON 结构
  - 文本安全处理：识别并过滤结构化输出，避免将 JSON 作为用户回复
  - 字段规范化：意图别名映射、工具名过滤、参数键名标准化

特点：
  - 无副作用：所有函数均为纯函数，不依赖外部状态
  - 防御式编程：对异常输入返回安全默认值而非抛异常
"""

import json


def extract_json_object(content: str) -> dict | None:
    """
    从文本内容中提取第一个 JSON 对象。

    用于解析 LLM 返回的混合文本（如包含 JSON 的回复）。
    支持两种解析路径：
      1. 直接解析整个文本
      2. 从第一个 '{' 开始截取后解析

    Args:
        content: 待解析的文本内容

    Returns:
        成功提取时返回 dict；否则返回 None
    """
    text = (content or "").strip()
    if not text:
        return None
    for candidate in (text, text[text.find("{"):] if "{" in text else ""):
        if not candidate:
            continue
        try:
            value, _ = json.JSONDecoder().raw_decode(candidate)
        except ValueError:
            continue
        if isinstance(value, dict):
            return value
    return None


def looks_like_structured_confirmation_text(text: str) -> bool:
    """
    判断文本是否为结构化确认输出（而非自然语言回复）。

    用于过滤 LLM 输出中的 JSON/结构化内容，避免将路由结果误当作用户回复。
    检测特征：
      - 包含路由关键字段名（turn_kind、intent_family 等）
      - 以 Markdown 代码块（```）开头
      - 以花括号包围（{...}）

    Args:
        text: 待检测的文本

    Returns:
        True 表示是结构化输出，应过滤；False 表示可能是自然语言
    """
    stripped = (text or "").strip()
    if not stripped:
        return False

    lowered = stripped.lower()
    structured_keys = (
        '"turn_kind"',
        '"intent_family"',
        '"intent_label_zh"',
        '"target_repo"',
        '"specified_params"',
        '"ambiguous_fields"',
        '"confirmation_text_zh"',
    )
    if any(key in lowered for key in structured_keys):
        return True

    return stripped.startswith("```") or (stripped.startswith("{") and stripped.endswith("}"))


def sanitize_confirmation_fallback(content: str, default_message: str) -> str:
    """
    安全处理确认文本：过滤空内容或结构化输出。

    当路由阶段输出为空或 JSON 结构时，返回预设的默认消息，
    防止将技术细节暴露给用户。

    Args:
        content: 原始确认文本
        default_message: 回退时使用的默认消息

    Returns:
        过滤后的安全文本
    """
    stripped = (content or "").strip()
    if not stripped:
        return default_message
    if looks_like_structured_confirmation_text(stripped):
        return default_message
    return stripped


def normalize_intent_family(
    raw_intent: object,
    *,
    intent_aliases: dict[str, str],
    intent_labels: dict[str, str],
) -> str:
    """
    规范化意图名称：别名映射 + 白名单校验。

    将用户/LLM 输出的意图名称转换为系统标准名称：
      1. 空值或非字符串 → "unknown"
      2. 小写化后查找别名映射
      3. 不在白名单内 → "unknown"

    Args:
        raw_intent: 原始意图名称（可能为任意类型）
        intent_aliases: 别名映射表，如 {"comprehensive": "comprehensive_ranking"}
        intent_labels: 有效意图白名单

    Returns:
        规范化后的意图名称
    """
    if not isinstance(raw_intent, str) or not raw_intent.strip():
        return "unknown"
    normalized = raw_intent.strip().lower()
    normalized = intent_aliases.get(normalized, normalized)
    return normalized if normalized in intent_labels else "unknown"


def normalize_tool_names(raw_tools: object, *, allowed_tool_names: set[str]) -> list[str]:
    """
    过滤并规范化工具名称列表。

    处理步骤：
      1. 类型校验：仅接受 list 类型输入
      2. 元素过滤：非字符串、空值、不在白名单的均剔除
      3. 去重保序：保持首次出现的顺序，移除重复项

    Args:
        raw_tools: 原始工具名称列表（可能包含无效值）
        allowed_tool_names: 允许的工具名称白名单

    Returns:
        清洁的工具名称列表（无重复、均在白名单内）
    """
    if not isinstance(raw_tools, list):
        return []
    cleaned: list[str] = []
    for item in raw_tools:
        if not isinstance(item, str):
            continue
        name = item.strip()
        if not name or name not in allowed_tool_names:
            continue
        cleaned.append(name)

    deduped = []
    seen = set()
    for name in cleaned:
        if name in seen:
            continue
        seen.add(name)
        deduped.append(name)
    return deduped


def ordered_tool_names(tool_names: set[str], *, all_tool_names: list[str]) -> list[str]:
    """
    按预定义顺序排列工具名称。

    用于生成稳定的提示词和日志输出，避免因集合遍历顺序不确定
    导致的输出波动。

    Args:
        tool_names: 待排序的工具名称集合
        all_tool_names: 全量工具名称列表（作为排序基准）

    Returns:
        按 all_tool_names 顺序排列的工具列表
    """
    return [name for name in all_tool_names if name in tool_names]


def normalize_turn_kind(raw_turn_kind: object, *, turn_kinds: set[str]) -> str:
    """
    规范化对话轮次类型（turn_kind）。

    将 LLM 输出的轮次类型转换为系统标准值：
      - 空值或非字符串 → "unknown"
      - 不在白名单内 → "unknown"

    Args:
        raw_turn_kind: 原始轮次类型（可能为任意类型）
        turn_kinds: 有效轮次类型白名单

    Returns:
        规范化后的轮次类型
    """
    if not isinstance(raw_turn_kind, str) or not raw_turn_kind.strip():
        return "unknown"
    normalized = raw_turn_kind.strip().lower()
    return normalized if normalized in turn_kinds else "unknown"


def normalize_specified_params(
    raw_params: object,
    *,
    allowed_param_names: set[str],
) -> tuple[dict[str, object], list[str], list[str]]:
    """
    规范化路由提取的参数字典，仅保留标准键名。

    处理步骤：
      1. 类型校验：非 dict 类型直接返回空结果
      2. 构建映射表：
         - canonical_by_lower: 小写化标准键名 → 原始标准键名
         - canonical_by_compact: 去掉下划线的小写键名 → 原始标准键名
      3. 遍历每个键值：
         - 非字符串键 → 丢弃并记录
         - 空键 → 跳过
         - 已匹配标准名 → 直接使用
         - 规范化（小写 + 替换连字符/空格为下划线）后查找映射
         - 不匹配则去掉下划线再查找紧凑映射
         - 仍不匹配 → 丢弃并记录
         - 已有同 canonical 键但值不同 → 丢弃（冲突）并记录
         - 成功映射 → 记录并添加说明

    Args:
        raw_params: 原始参数字典（可能包含非标准键名）
        allowed_param_names: 允许的标准参数键名集合

    Returns:
        tuple 包含三个元素：
          - normalized: 规范化后的参数字典（仅包含标准键名）
          - dropped: 被丢弃的键名列表
          - notes: 映射说明列表（用于调试日志，如 "map:TimeWindow->growth_calc_days(case)"）
    """
    if not isinstance(raw_params, dict):
        return {}, [], []

    canonical_by_lower = {name.lower(): name for name in allowed_param_names}
    canonical_by_compact = {
        lowered.replace("_", ""): canonical
        for lowered, canonical in canonical_by_lower.items()
    }

    normalized: dict[str, object] = {}
    dropped: list[str] = []
    notes: list[str] = []

    for raw_key, value in raw_params.items():
        if not isinstance(raw_key, str):
            dropped.append("<non_string_key>")
            notes.append("drop:non_string_key")
            continue

        key = raw_key.strip()
        if not key:
            continue

        canonical: str | None = None
        reason: str | None = None

        if key in allowed_param_names:
            canonical = key
        else:
            normalized_key = key.lower().replace("-", "_").replace(" ", "_")
            if normalized_key in canonical_by_lower:
                canonical = canonical_by_lower[normalized_key]
                reason = "case"
            else:
                compact = normalized_key.replace("_", "")
                if compact in canonical_by_compact:
                    canonical = canonical_by_compact[compact]
                    reason = "shape"

        if canonical is None:
            dropped.append(key)
            notes.append(f"drop:{key}")
            continue

        if canonical in normalized and normalized[canonical] != value:
            dropped.append(key)
            notes.append(f"conflict:{key}->{canonical}")
            continue

        normalized[canonical] = value
        if canonical != key:
            notes.append(f"map:{key}->{canonical}({reason or 'normalized'})")

    return normalized, dropped, notes
