"""LLM 出站 —— 多平台目录、顺序回退、流式。

    api.py       目录归一化 + key 落地(config 只声明 key_env,真值在这里取);一个 Api = 一个平台接入
    protocol.py  线上协议:请求头 / 请求体 / 一次请求(含 SSE)。`if backend ==` 只在这里
    client.py    调用顺序:内部调用顺序回退,网页硬切换不回退

对外只有 `get()`。**提示词不在这一层** —— 那是产品知识,归调用它的工具。
这层只管「怎么把一段消息发出去、怎么在某家挂掉时换一家」。
"""

from __future__ import annotations

from .api import Api, build
from .client import LLMClient
from .protocol import EFFORT_DEFAULT, EFFORT_MEDIUM, EFFORT_OFF, EFFORTS, level

__all__ = ["LLMClient", "Api", "get", "build_from_config",
           "EFFORTS", "EFFORT_OFF", "EFFORT_MEDIUM", "EFFORT_DEFAULT", "level"]

_shared: LLMClient | None = None


def build_from_config() -> LLMClient:
    from ... import config
    return LLMClient(build(config.LLM_MODELS, config.llm_key))


def get() -> LLMClient:
    """进程内共享的客户端。缓存只为省掉重读环境变量,它本身无状态,并发调用安全。

    代价是进程起来后改环境变量不生效 —— 对 CI 和长驻服务都无所谓。
    """
    global _shared
    if _shared is None:
        _shared = build_from_config()
    return _shared
