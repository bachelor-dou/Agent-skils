"""环境变量解析 —— 纯函数,口径只有一份。

集中在这里是为了让配置里的机密、Web 安全项、以及将来任何读环境的地方共用同一套口径:
逗号分隔怎么切、布尔怎么认,只有一份定义,不会出现某处认 "on" 另一处不认。
"""

from __future__ import annotations

import os

_TRUE_VALUES = frozenset({"1", "true", "yes", "on"})


def csv_list(name: str) -> list[str]:
    """读逗号分隔的环境变量,去空白、丢空项。未设置或全空 → []。"""
    raw = os.environ.get(name, "")
    if not raw:
        return []
    return [part.strip() for part in raw.split(",") if part.strip()]


def truthy(value: object) -> bool:
    """字符串或其它值 → 布尔,口径与环境变量一致。

    裸 `bool()` 会把字符串 `"0"` 判成真,而手写配置里 `"enabled": "0"` 是很自然的写法
    (旁边几行都带引号,顺手就加了)。判反了的后果是「关掉的平台照样被调用」,
    而且不报错,只是账单上多一笔。
    """
    if isinstance(value, str):
        return value.strip().lower() in _TRUE_VALUES
    return bool(value)


def flag(name: str, default: bool = False) -> bool:
    """读布尔环境变量。未设置 → default;设置了但不在真值集合里 → False。"""
    raw = os.environ.get(name)
    return default if raw is None else truthy(raw)


def text(name: str, default: str = "") -> str:
    """读字符串环境变量,去首尾空白。"""
    return os.environ.get(name, default).strip()
