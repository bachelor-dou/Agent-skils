"""会话池 —— 每个 session_id 一个 Agent,带 TTL 和上限。

内存版。够用是因为会话本来就是短命的(一小时 TTL),而且换 Redis 要先有第二台机器。
真需要横向扩的那天,替换的是这一个文件。

## 三条不能省的规矩

1. **TTL + 数量上限**。一个 Agent 拿着完整对话历史,几十轮之后是几百 KB。
   没有上限的话,爬虫拿随机 session_id 打几万次就能把内存撑爆。
2. **全局工具锁**。多个会话同时出榜会争同一批 GitHub token,谁都跑不快,还更容易撞限流。
   所以工具执行全局串行 —— 这是有意的取舍,不是漏了并发。
3. **待发回复缓冲**。手机端切后台会断 WebSocket,而 agent 还在跑。跑完发不出去就存着,
   重连时补推 —— 否则用户等了两分钟回来看到的是空白。

`ponytail:` 单进程内存态。多副本部署时会话不粘,用户重连到另一个副本就丢上下文;
真要多副本得把这里换成 Redis。
"""

from __future__ import annotations

import logging
import threading
import time
from datetime import datetime, timezone

from ..agent import Agent, build

logger = logging.getLogger("hot_project")

TTL_SECONDS = 3600
MAX_SESSIONS = 100

# 工具执行全局串行,见模块头部第 2 条。HTTP 路径直接阻塞等,WebSocket 路径带超时
# —— 前端在等一个响应,与其让它无限期转圈,不如 90 秒后明确说「系统繁忙」。
TOOL_LOCK_TIMEOUT = 90
tool_lock = threading.Lock()

_agents: dict[str, tuple[Agent, float]] = {}
_lock = threading.Lock()

_pending: dict[str, list[str]] = {}
_pending_lock = threading.Lock()


def expires_at(when: float | None = None) -> str:
    """会话过期时刻,UTC 时间戳串。前端拿它显示「还有多久要重新开始」。"""
    moment = when if when is not None else time.time() + TTL_SECONDS
    return datetime.fromtimestamp(moment, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _drop(session_id: str) -> None:
    """调用方必须持有 `_lock`。"""
    _agents.pop(session_id, None)
    with _pending_lock:
        _pending.pop(session_id, None)


def get(session_id: str) -> Agent:
    """取或建一个会话的 Agent。线程安全。"""
    now = time.time()
    with _lock:
        for stale in [sid for sid, (_, seen) in _agents.items() if now - seen > TTL_SECONDS]:
            _drop(stale)
            logger.info("会话过期已清理:%s", stale)

        if (entry := _agents.get(session_id)) is not None:
            _agents[session_id] = (entry[0], now)
            return entry[0]

        if len(_agents) >= MAX_SESSIONS:
            oldest = min(_agents, key=lambda sid: _agents[sid][1])
            _drop(oldest)
            logger.info("会话数到上限,淘汰最旧的:%s", oldest)

        agent = build()
        _agents[session_id] = (agent, now)
        logger.info("新建会话:%s", session_id)
        return agent


def drop(session_id: str) -> bool:
    with _lock:
        if session_id not in _agents:
            return False
        _drop(session_id)
        return True


def count() -> int:
    with _lock:
        return len(_agents)


# ── 断线期间的回复 ──────────────────────────────────────────────

def stash(session_id: str, reply: str) -> None:
    with _pending_lock:
        _pending.setdefault(session_id, []).append(reply)


def take(session_id: str) -> list[str]:
    """取走并清空。取走之后发送失败的,调用方要自己 `stash` 回来。"""
    with _pending_lock:
        return _pending.pop(session_id, [])
