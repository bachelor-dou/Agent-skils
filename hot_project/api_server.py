#!/usr/bin/env python
"""Web / API 服务。

    python -m hot_project.api_server

这个文件只做**路由**:收请求、调 web/{render,sessions,security},把结果变成 HTTP 响应。

**规矩:碰存储或渲染的处理器一律写同步 `def`。** FastAPI 会把同步 `def` 丢进线程池,而
agent 一路是同步的,写成 `async def` 会把事件循环连同 WS 心跳一起锁死几分钟;只有纯内存/
纯配置的接口才留 `async def`。这也是工具能直接 `asyncio.run` 的前提:它永不在事件循环里跑。
"""

from __future__ import annotations

import asyncio
import json
import logging
import queue
import re
import time
from contextlib import asynccontextmanager
from datetime import datetime

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from . import config
from .common import logs
from .infra import llm
from .infra.store import favorites, reports, universe
from .tools import describe
from .web import render, security, sessions

logger = logging.getLogger("hot_project")

PORT = 8001
NO_CACHE = {"Cache-Control": "no-store, no-cache, must-revalidate, max-age=0",
            "Pragma": "no-cache", "Expires": "0"}

REPO_NAME = re.compile(r"^[\w.-]+/[\w.-]+$")
SHORT_DESC_MAX = 60

# WebSocket 空闲时的心跳间隔:反代会掐掉长时间没数据的连接,而出榜要跑几分钟。
WS_HEARTBEAT = 15
# 轮询进度队列的粒度。也是正文流式的最小粒度,所以要细 —— 只在对话进行时空转。
WS_POLL = 0.05


# ── 请求体 ──────────────────────────────────────────────────────

class ChatIn(BaseModel):
    session_id: str = "default"
    message: str
    user_id: str = ""
    model: str = ""
    lite: str = ""          # 子模型 id(「平台id:模型名」);空 = 跟随主模型平台


class ChatOut(BaseModel):
    session_id: str
    reply: str
    session_ttl_seconds: int
    session_expires_at: str


class ModelTestIn(BaseModel):
    model: str = ""
    lite: str = ""


class FavoriteIn(BaseModel):
    user_id: str
    repo: str
    action: str                         # add / remove
    source_report: str = ""
    category: str | None = None         # None = 不改;"" = 归到未分类
    short_desc: str | None = None       # None = 按需自动生成;字符串(含空)= 直接采用


# ── 应用 ────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("API Server 启动。日志:%s", logs.setup(config.LOG_DIR, "web"))
    yield
    logger.info("API Server 关闭。")


app = FastAPI(title="GitHub Hot Projects", version="2.0.0", lifespan=lifespan)
app.mount("/web", StaticFiles(directory=config.WEB_DIR), name="web")
app.add_middleware(CORSMiddleware, **security.cors_options())
app.add_middleware(security.Guard)       # 后注册先执行,所以它在 CORS 之前拦


def _page(name: str, missing: str) -> HTMLResponse:
    try:
        html = render.page(name, {})
    except OSError:
        raise HTTPException(status_code=404, detail=missing) from None
    return HTMLResponse(html, headers=NO_CACHE)


@app.get("/api/status")
async def status():
    return {"status": "running", "active_sessions": sessions.count(),
            "session_ttl_seconds": sessions.TTL_SECONDS,
            "min_star": config.MIN_STAR}


@app.get("/", response_class=HTMLResponse)
@app.get("/chat", response_class=HTMLResponse)
async def chat_page():
    return _page("chat.html", "聊天页面不存在")


# ── 对话 ────────────────────────────────────────────────────────

@app.post("/api/chat", response_model=ChatOut)
def chat(body: ChatIn):
    """同步 `def` 是有意的,见文件头。"""
    logger.info("HTTP 对话:session=%s message=%s", body.session_id, body.message[:120])
    agent = sessions.get(body.session_id)
    # 必须带超时。同步 `def` 跑在全站共享的线程池里,无超时地等这把全局锁会把池占满 ——
    # 之后每个同步接口都无限期排队,整台服务卡死。
    if not sessions.tool_lock.acquire(timeout=sessions.TOOL_LOCK_TIMEOUT):
        logger.warning("HTTP 等执行锁超时:%s", body.session_id)
        raise HTTPException(status_code=503, detail="系统繁忙,请稍后重试。")
    try:
        reply = agent.chat(body.message, user_id=body.user_id,
                           model=body.model, lite=body.lite)
    finally:
        sessions.tool_lock.release()
    return ChatOut(session_id=body.session_id, reply=reply,
                   session_ttl_seconds=sessions.TTL_SECONDS,
                   session_expires_at=sessions.expires_at())


@app.delete("/api/sessions/{session_id}")
async def drop_session(session_id: str):
    if not sessions.drop(session_id):
        raise HTTPException(status_code=404, detail="会话不存在")
    return {"message": f"会话 {session_id} 已清除"}


# ── 模型 ────────────────────────────────────────────────────────

@app.get("/api/models")
async def models():
    """给网页的模型切换器。

    子模型跨平台融合成一个共享池:主/子选择解耦,同名的只留先出现的那个。
    """
    schemes = llm.get().usable()
    pool, seen = [], set()
    for scheme in schemes:
        for name in scheme.lite_models:
            if name not in seen:
                seen.add(name)
                pool.append({"id": f"{scheme.id}:{name}", "label": name.rsplit("/", 1)[-1]})
    return {"models": [{"id": s.id, "label": s.label} for s in schemes],
            "lite_models": pool}


@app.post("/api/models/test")
def test_models(body: ModelTestIn):
    """预检:对选中的主/子模型各发一次极小的真实调用。

    key 配错、额度用完、区域限制都只有真发一次才知道,对话中途才发现意味着用户白等一轮。
    """
    client = llm.get()
    bad: list[str] = []
    if body.model and not client.ping(model_id=body.model):
        bad.append(next((s.label for s in client.usable() if s.id == body.model), body.model))
    if body.lite and not client.ping(lite_id=body.lite):
        bad.append(body.lite.partition(":")[2] or body.lite)
    logger.info("模型预检:model=%s lite=%s → %s",
                body.model or "-", body.lite or "-", bad or "全部可用")
    return {"ok": not bad, "unavailable": bad}


# ── 报告 ────────────────────────────────────────────────────────

def _resolved(name: str) -> str:
    """把 URL 里的名字变成确定的报告文件名,顺带挡住路径穿越。"""
    resolved = reports.resolve_name(name)
    if resolved is None:
        raise HTTPException(status_code=404, detail="报告不存在")
    return resolved


@app.get("/api/reports")
def report_list():
    directory = reports.directory()
    out = []
    for item in reports.listing():
        stat = (directory / item.name).stat()
        out.append({"name": item.name, "title": item.title,
                    "day": str(item.day) if item.day else "",
                    "size": stat.st_size,
                    # 必须是 ISO 字符串:前端 `new Date(v)` 把裸数字当**毫秒**,给秒会错到 1970。
                    "modified_at": datetime.fromtimestamp(stat.st_mtime).isoformat()})
    return {"reports": out}


@app.get("/api/reports/{name}")
def report_markdown(name: str):
    resolved = _resolved(name)
    return {"name": resolved, "content": reports.read(resolved) or ""}


@app.get("/api/reports/{name}/html", response_class=HTMLResponse)
def report_page(name: str):
    resolved = _resolved(name)
    return HTMLResponse(render.report_html(resolved, reports.read(resolved) or ""),
                        headers=NO_CACHE)


@app.delete("/api/reports/{name}")
def report_delete(name: str):
    resolved = _resolved(name)
    if not reports.delete(resolved):
        raise HTTPException(status_code=500, detail="删除失败")
    return {"message": f"报告 {resolved} 已删除", "deleted": resolved}


@app.get("/api/star-trend")
def star_trend(repo: str):
    """报告卡片上「star 走势」按钮的数据源。"""
    if not REPO_NAME.match(repo or ""):
        raise HTTPException(status_code=400, detail="无效的仓库名")
    from .tools.report_tools import star_trend as trend
    from .tools.spec import Ctx
    return trend(Ctx(), {"repo": repo})


# ── 收藏 ────────────────────────────────────────────────────────

@app.get("/api/favorite-tags")
async def favorite_tags():
    return {"tags": list(config.FAVORITE_DEFAULT_TAGS)}


@app.get("/api/favorites")
def favorite_list(user_id: str):
    if not favorites.valid_user_id(user_id):
        raise HTTPException(status_code=400, detail="无效的 user_id")
    counts, total = reports.appearance_counts()
    return {"user_id": user_id, "report_total": total,
            "favorites": [dict(item, report_count=counts.get(item.get("repo", ""), 0),
                               report_total=total)
                          for item in favorites.get(user_id)]}


def _short_desc(repo: str) -> str:
    """收藏卡片上那句中文概要。和 `add_favorite` 工具同一条路径。

    收藏时才生成,不在出报告时给几百个项目预生成。
    """
    gh_desc = (universe.load().get(repo, {}).get("gh_desc") or "").strip()
    if not gh_desc:
        return ""
    return describe.condense([{"full_name": repo, "description": gh_desc}],
                             max_chars=SHORT_DESC_MAX)[0]


@app.post("/api/favorites")
def favorite_update(body: FavoriteIn):
    short = None
    if body.action == "add":
        # 用户手填的(含清空)直接用,不再花一次 LLM
        short = (body.short_desc.strip()[:SHORT_DESC_MAX] if body.short_desc is not None
                 else _short_desc(body.repo) or None)
    try:
        items = favorites.set_favorite(body.user_id, body.repo, body.action,
                                       source_report=body.source_report,
                                       short_desc=short, category=body.category)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    return {"user_id": body.user_id, "favorites": items}


# ── WebSocket ───────────────────────────────────────────────────

@app.websocket("/ws/chat/{session_id}")
async def ws_chat(websocket: WebSocket, session_id: str):
    """实时对话:执行期间推进度 + 心跳 + 正文增量,最后再整段发一次权威全文。

    整段重发不是冗余:增量可能中途断掉,前端要一份完整的做最终渲染和历史存档。
    """
    # 安全中间件对 WebSocket 不生效(starlette 见到非 http scope 直接放行),必须自己问一次
    # —— 否则最该保护的入口是唯一没保护的入口。
    ip = security.client_ip(websocket)
    if verdict := security.check(ip, websocket.url.path):
        logger.warning("WS %s拦截:%s %s", verdict.reason, ip, session_id)
        await websocket.close(code=1008)        # 1008 = policy violation
        return

    await websocket.accept()
    params = websocket.query_params
    user_id, model, lite = (params.get(k, "") for k in ("user_id", "model", "lite"))
    logger.info("WS 已连接:%s(user=%s model=%s lite=%s)",
                session_id, user_id or "-", model or "-", lite or "-")

    # 断线期间攒下的回复,重连补推。发不出去就放回去,下次再试。
    for reply in sessions.take(session_id):
        try:
            await websocket.send_text(reply)
        except Exception:       # noqa: BLE001
            sessions.stash(session_id, reply)
            break

    def run(message: str, progress, on_delta) -> str:
        agent = sessions.get(session_id)
        if not sessions.tool_lock.acquire(timeout=sessions.TOOL_LOCK_TIMEOUT):
            logger.warning("WS 等执行锁超时:%s", session_id)
            return "系统繁忙,请稍后重试。"
        try:
            return agent.chat(message, progress=progress, user_id=user_id,
                              model=model, lite=lite, on_delta=on_delta)
        finally:
            sessions.tool_lock.release()

    try:
        while True:
            message = await websocket.receive_text()
            logger.info("WS 收到:%s %s", session_id, message[:120])
            if await _pump(websocket, session_id, message, run) is None:
                break       # 连接断了,回复已经存起来
    except WebSocketDisconnect:
        logger.info("WS 断开:%s", session_id)


async def _pump(websocket, session_id: str, message: str, run) -> str | None:
    """在线程里跑一轮对话,同时把进度和正文增量实时推出去。

    连接断了**不中断 agent**:它可能正跑在出榜的第三分钟上,掐掉等于白烧一次;让它跑完,
    回复存进待发缓冲。返回最终回复;连接已断、回复已缓存时返回 None。
    """
    events: queue.Queue = queue.Queue()
    DONE = object()
    holder: dict[str, str] = {}

    def worker() -> None:
        try:
            holder["reply"] = run(
                message,
                lambda percent, label: events.put(
                    {"type": "progress", "percent": percent, "label": label}),
                lambda text, reset=False: events.put(
                    {"type": "delta", "text": text, "reset": reset}),
            )
        except Exception as e:      # noqa: BLE001 —— 一轮失败要告诉前端,不能静默吞掉
            logger.exception("WS 对话异常:%s", session_id)
            holder["reply"] = f"处理消息时出错:{e}"
        finally:
            events.put(DONE)

    task = asyncio.create_task(asyncio.to_thread(worker))
    alive = True
    last_sent = time.time()

    while True:
        finished = False
        try:
            while True:
                item = events.get_nowait()
                if item is DONE:
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
        if alive and time.time() - last_sent >= WS_HEARTBEAT:
            try:
                await websocket.send_text(json.dumps({"type": "heartbeat"}))
                last_sent = time.time()
            except Exception:       # noqa: BLE001
                alive = False
        await asyncio.sleep(WS_POLL)

    await task
    reply = holder.get("reply", "")
    logger.info("WS 回复完成:%s(%d 字)", session_id, len(reply))

    if alive:
        try:
            await websocket.send_text(
                json.dumps({"type": "reply", "reply": reply}, ensure_ascii=False))
            return reply
        except Exception:       # noqa: BLE001
            pass
    sessions.stash(session_id, reply)
    return None


def main() -> None:
    import uvicorn
    logs.setup(config.LOG_DIR, "web", console=True)
    uvicorn.run("hot_project.api_server:app", host="0.0.0.0", port=PORT, reload=False)


if __name__ == "__main__":
    main()
