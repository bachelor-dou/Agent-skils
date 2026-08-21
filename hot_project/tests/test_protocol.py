"""聊天线上协议:一轮 WS 对话的信封序列、Busy 的表达、断线补发的形状与顺序。

以前这条组合路径零测试:信封在 `_pump` 里内联拼、补发存裸字符串、Busy 伪装成回复,
前端只好维护第二套解析器。现在协议收进 `web/protocol.py`,这里经同一条缝测组合。
"""

import asyncio
import json

from hot_project import api_server
from hot_project.web import protocol, sessions


class _WS:
    """记录发出的帧;`die_after` 指定发几帧后开始拒收(-1 = 一直活着)。"""

    def __init__(self, die_after: int = -1):
        self.frames: list[dict] = []
        self.die_after = die_after

    async def send_text(self, text: str) -> None:
        if self.die_after >= 0 and len(self.frames) >= self.die_after:
            raise RuntimeError("前端走了")
        self.frames.append(json.loads(text))


def test_a_turn_reaches_the_client_as_envelopes_only():
    """进度、增量、最终回复,前端看到的每一帧都是带 type 的信封。"""
    ws = _WS()

    def run(message, progress, on_delta) -> str:
        progress(50, "干活中")
        on_delta("第一段")
        return "最终回复"

    got = asyncio.run(api_server._pump(ws, "proto-1", "你好", run))

    assert got is not None
    assert [f["type"] for f in ws.frames] == ["progress", "delta", "reply"]
    assert ws.frames[-1]["reply"] == "最终回复"


def test_busy_goes_out_as_an_error_envelope_not_a_fake_reply():
    """系统繁忙是状态不是聊天内容:以前伪装成 reply 混进聊天记录,现在是 error 帧。"""
    ws = _WS()

    def run(message, progress, on_delta) -> str:
        raise sessions.Busy("测试")

    asyncio.run(api_server._pump(ws, "proto-2", "你好", run))

    assert ws.frames[-1]["type"] == "error"
    assert ws.frames[-1]["error"] == api_server.BUSY_DETAIL


def test_a_reply_stashed_while_offline_replays_as_the_same_envelope():
    """断线时攒下的帧和在线推送是同一种形状 —— 前端不需要裸字符串兜底解析器。"""
    dead = _WS(die_after=0)

    def run(message, progress, on_delta) -> str:
        return "迟到的回复"

    got = asyncio.run(api_server._pump(dead, "proto-3", "你好", run))
    assert got is None, "没送出去要返回 None,外层循环据此收尾"

    ws = _WS()
    asyncio.run(protocol.replay(ws, "proto-3"))

    assert [f["type"] for f in ws.frames] == ["reply"]
    assert ws.frames[0]["reply"] == "迟到的回复"
    assert sessions.take("proto-3") == [], "补发成功后缓冲要清空"


def test_a_failed_replay_keeps_the_remaining_frames_in_order():
    """补发到一半又断线:旧实现只塞回当前那帧,后面的整批丢掉 —— 这里钉死不再发生。"""
    for text in ("一", "二", "三"):
        sessions.stash("proto-4", protocol.reply(text))

    flaky = _WS(die_after=1)
    asyncio.run(protocol.replay(flaky, "proto-4"))
    assert [f["reply"] for f in flaky.frames] == ["一"]

    ws = _WS()
    asyncio.run(protocol.replay(ws, "proto-4"))
    assert [f["reply"] for f in ws.frames] == ["二", "三"], "剩余帧一个不少、顺序不变"
