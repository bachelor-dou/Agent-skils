"""
参数提取
========
纯函数：从用户消息文本 / 对话列表中提取具体参数值。

所有 extract_* 函数接收小写化文本，返回提取到的值或 None。
"""

import re


def latest_user_message(conversation: list[dict]) -> str:
    """返回对话历史中最近一条用户消息的小写文本。"""
    for msg in reversed(conversation):
        if msg.get("role") == "user":
            return (msg.get("content") or "").lower()
    return ""


def extract_time_window_days(text: str) -> int | None:
    """从文本中提取增长统计窗口天数。

    匹配模式：
      - 数值型："近10天"、"最近2周"、"过去3个月"
      - 别名型："近一周"、"近两个月"
    """
    if not text:
        return None

    numeric_patterns = [
        (r"(?:近|最近|过去)\s*(\d+)\s*天", 1),
        (r"(?:近|最近|过去)\s*(\d+)\s*周", 7),
        (r"(?:近|最近|过去)\s*(\d+)\s*个?月", 30),
    ]
    for pattern, multiplier in numeric_patterns:
        match = re.search(pattern, text)
        if match:
            return int(match.group(1)) * multiplier

    alias_patterns = {
        7: [r"近一周", r"最近一周", r"过去一周"],
        14: [r"近两周", r"最近两周", r"过去两周"],
        30: [r"近一个月", r"最近一个月", r"过去一个月", r"近一月", r"最近一月", r"过去一月"],
        60: [r"近两个月", r"最近两个月", r"过去两个月"],
    }
    for days, patterns in alias_patterns.items():
        if any(re.search(pattern, text) for pattern in patterns):
            return days

    return None


def extract_creation_window_days(text: str) -> int | None:
    """从文本中提取新项目创建时间窗口。

    仅对明确的"创建于 / 天内创建"语义生效，
    避免将"近N天"误解为创建窗口。
    """
    if not text:
        return None

    numeric_patterns = [
        (r"(\d+)\s*天内创建", 1),
        (r"(?:近|最近)\s*(\d+)\s*天创建", 1),
        (r"(\d+)\s*周内创建", 7),
        (r"(?:近|最近)\s*(\d+)\s*周创建", 7),
        (r"(\d+)\s*个?月内创建", 30),
        (r"(?:近|最近)\s*(\d+)\s*个?月创建", 30),
    ]
    for pattern, multiplier in numeric_patterns:
        match = re.search(pattern, text)
        if match:
            return int(match.group(1)) * multiplier

    alias_patterns = {
        7: [r"一周内创建", r"(?:近|最近)一周创建"],
        14: [r"两周内创建", r"(?:近|最近)两周创建"],
        30: [r"一个月内创建", r"一月内创建", r"(?:近|最近)一个月创建", r"(?:近|最近)一月创建"],
        60: [r"两个月内创建", r"(?:近|最近)两个月创建"],
    }
    for days, patterns in alias_patterns.items():
        if any(re.search(pattern, text) for pattern in patterns):
            return days

    return None


def has_explicit_creation_window(text: str) -> bool:
    """文本中是否明确提到了新项目创建时间窗口。"""
    return extract_creation_window_days(text) is not None


def extract_top_n(text: str) -> int | None:
    """从文本中提取榜单数量（如 "前50"、"Top 20"、"130个项目"）。"""
    if not text:
        return None

    patterns = [
        r"(?:top|前)\s*(\d+)",
        r"(\d+)\s*个(?:项目|仓库|结果|条)",
        r"(\d+)\s*名",
    ]
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            value = int(match.group(1))
            if value > 0:
                return value
    return None


def has_explicit_top_n(text: str) -> bool:
    """文本中是否明确指定了榜单数量。"""
    return extract_top_n(text) is not None
