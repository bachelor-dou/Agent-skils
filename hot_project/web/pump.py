"""线程 → 队列 → WebSocket 的泵。

同步的 worker 在线程里跑,期间用 `emit(dict)` 发事件;这里把事件实时转成 JSON 推给
websocket,闲得太久就发心跳。前端断开**不中断 worker**(它可能正跑到出榜第三分钟),
只是不再推送 —— 结果照样返回,存不存待发缓冲由调用方定。

事件长什么样、最终结果怎么处理,都是调用方的协议;这里只认「dict 进、JSON 出」。
"""

from __future__ import annotations

import asyncio
import json
import queue
import time
from collections.abc import Callable
from typing import NamedTuple

_DONE = object()


class Outcome(NamedTuple):
    result: str     # worker 的返回值
    alive: bool     # 结束时连接是否还活着(False = 结果没能送出去)


async def drive(websocket, worker: Callable[[Callable[[dict], None]], str],
                *, heartbeat: float, poll: float) -> Outcome:
    """在线程里跑 `worker(emit)`,同时把 emit 出来的事件实时推给 websocket。

    worker 自己兜异常:这里不翻译错误,抛出来什么算什么(调用方的 worker 应当把异常
    变成一条给用户的结果字符串)。
    """
    events: queue.Queue = queue.Queue()
    holder: dict[str, str] = {}

    def runner() -> None:
        try:
            holder["result"] = worker(events.put)
        finally:
            events.put(_DONE)

    task = asyncio.create_task(asyncio.to_thread(runner))
    alive = True
    last_sent = time.time()

    while True:
        finished = False
        try:
            while True:
                item = events.get_nowait()
                if item is _DONE:
                    finished = True
                    break
                if alive:
                    try:
                        await websocket.send_text(json.dumps(item, ensure_ascii=False))
                        last_sent = time.time()
                    except Exception:       # noqa: BLE001
                        alive = False       # 前端走了,不再推,但等 worker 跑完
        except queue.Empty:
            pass

        if finished:
            break
        if alive and time.time() - last_sent >= heartbeat:
            try:
                await websocket.send_text(json.dumps({"type": "heartbeat"}))
                last_sent = time.time()
            except Exception:       # noqa: BLE001
                alive = False
        await asyncio.sleep(poll)

    await task
    return Outcome(holder.get("result", ""), alive)
