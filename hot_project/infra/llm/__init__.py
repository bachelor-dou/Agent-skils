"""LLM 出站 —— 多平台目录、顺序回退、流式。

    schemes.py   目录归一化 + key 落地(config 只声明 key_env,真值在这里取)
    wire.py      线上协议:请求头 / 请求体 / 一次请求(含 SSE)。`if backend ==` 只在这里
    client.py    调用顺序:内部调用顺序回退,网页硬切换不回退

对外只有 `get()`。**提示词不在这一层** —— 那是产品知识,归调用它的工具。
这层只管「怎么把一段消息发出去、怎么在某家挂掉时换一家」。
"""

from __future__ import annotations

from .client import LLMClient
from .schemes import Scheme, build

__all__ = ["LLMClient", "Scheme", "get", "build_from_config"]

_shared: LLMClient | None = None


def build_from_config() -> LLMClient:
    from ... import config
    return LLMClient(build(config.LLM_MODELS, config.llm_key))


def get() -> LLMClient:
    """进程内共享的客户端。

    缓存是因为构造要读一遍环境变量,不是因为它有状态 —— 它没有,并发调用安全。
    (代价:进程跑起来之后改环境变量不会生效。这对 CI 和长驻服务都不是问题,
    而 GitHub token 那边要能中途补 token,所以那边就没缓存。)
    """
    global _shared
    if _shared is None:
        _shared = build_from_config()
    return _shared
