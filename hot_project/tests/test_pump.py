"""WebSocket 泵:事件实时转发、断线不打断 worker、闲时心跳。"""

import asyncio
import json
import time

from hot_project.web import pump


class _WS:
    def __init__(self, dead: bool = False):
        self.sent: list[dict] = []
        self.dead = dead

    async def send_text(self, text: str) -> None:
        if self.dead:
            raise RuntimeError("前端走了")
        self.sent.append(json.loads(text))


def test_events_arrive_in_order_and_the_result_comes_back():
    ws = _WS()

    def worker(emit) -> str:
        emit({"type": "progress", "percent": 5})
        emit({"type": "delta", "text": "第一段"})
        return "最终回复"

    got = asyncio.run(pump.drive(ws, worker, heartbeat=60, poll=0.01))

    assert got == pump.Outcome("最终回复", True)
    assert [e["type"] for e in ws.sent] == ["progress", "delta"]


def test_a_dead_connection_does_not_interrupt_the_worker():
    """前端断开只是不再推送;worker(可能正跑到出榜第三分钟)必须跑完,结果照样返回。"""
    finished = []

    def worker(emit) -> str:
        emit({"type": "progress", "percent": 5})
        time.sleep(0.05)
        finished.append(True)
        return "跑完了"

    got = asyncio.run(pump.drive(_WS(dead=True), worker, heartbeat=60, poll=0.01))

    assert got == pump.Outcome("跑完了", False)
    assert finished == [True], "连接断了 worker 也得跑完"


def test_a_quiet_stretch_produces_heartbeats():
    ws = _WS()

    def worker(emit) -> str:
        time.sleep(0.08)        # 一言不发,泵得自己保活
        return "好"

    got = asyncio.run(pump.drive(ws, worker, heartbeat=0, poll=0.01))

    assert got.result == "好"
    assert any(e["type"] == "heartbeat" for e in ws.sent)
