"""聊天线上协议 —— WS 发给前端的每一帧长什么样,全在这里。

四种信封:progress / delta / reply / error(pump 的 heartbeat 是保活机械,不算)。
在线推送和断线补发共用同一种信封文本:`sessions` 的待发缓冲里只允许存这里生成的
帧,重连时 `replay` 原样补推。前端因此只需要一个 JSON 信封解析器,没有裸字符串兜底。
"""

from __future__ import annotations

import json

from . import sessions


def progress(percent: int, label: str) -> dict:
    return {"type": "progress", "percent": percent, "label": label}


def delta(text: str, reset: bool = False) -> dict:
    return {"type": "delta", "text": text, "reset": reset}


def reply(text: str) -> str:
    """最终回复帧,可直接 send_text 的线上文本。"""
    return json.dumps({"type": "reply", "reply": text}, ensure_ascii=False)


def error(message: str) -> str:
    """系统状态帧(繁忙、执行出错)。前端把它渲染成错误,不混进聊天记录。"""
    return json.dumps({"type": "error", "error": message}, ensure_ascii=False)


async def deliver(websocket, session_id: str, frame: str, *, alive: bool = True) -> bool:
    """发一帧;连接已死或发送失败就进待发缓冲,等重连 `replay` 补推。"""
    if alive:
        try:
            await websocket.send_text(frame)
            return True
        except Exception:       # noqa: BLE001 —— 发送失败一律按断线处理
            pass
    sessions.stash(session_id, frame)
    return False


async def replay(websocket, session_id: str) -> None:
    """把断线期间攒下的帧按原顺序补发;发到一半又断,剩下的全部塞回去下次再来。"""
    frames = sessions.take(session_id)
    for i, frame in enumerate(frames):
        try:
            await websocket.send_text(frame)
        except Exception:       # noqa: BLE001
            for rest in frames[i:]:
                sessions.stash(session_id, rest)
            break
