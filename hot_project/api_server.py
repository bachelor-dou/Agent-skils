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
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from . import config
from .common import logs
from .infra import llm
from .infra.data_access import favorites, reports, universe
from .tools import repo_tools
from .web import render, security, sessions

logger = logging.getLogger("hot_project")

PORT = 8001
NO_CACHE = {"Cache-Control": "no-store, no-cache, must-revalidate, max-age=0",
            "Pragma": "no-cache", "Expires": "0"}

REPO_NAME = re.compile(r"^[\w.-]+/[\w.-]+$")

WS_HEARTBEAT = 15
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
app.add_middleware(security.Guard)


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


class DescribeIn(BaseModel):
    repo: str


@app.post("/api/repo-describe")
def repo_describe(body: DescribeIn):
    """报告卡片「刷新介绍」:重跑描述生成并落库,顺带回传当下的 star 与窗口增长。

    star/增长走 `live_growth`(实时 star 减最早快照),和 agent 的 repo_growth 同一份算法,
    所以两边永远说同一个数。它拿不到就只回介绍,不能因此让刷新整个失败。

    和对话共用同一把执行锁,不让并发刷新和出榜互相抢 token。同步 `def`,理由见文件头。
    """
    if not REPO_NAME.match(body.repo or ""):
        raise HTTPException(status_code=400, detail="无效的仓库名")
    from .provider.github import client as github
    from .service import report
    if not (gh := github.shared()).usable:
        raise HTTPException(status_code=503, detail="没有可用的 GitHub token,无法刷新介绍。")
    if not sessions.tool_lock.acquire(timeout=sessions.TOOL_LOCK_TIMEOUT):
        logger.warning("刷新介绍等执行锁超时:%s", body.repo)
        raise HTTPException(status_code=503, detail="系统繁忙,请稍后重试。")
    try:
        desc = report.regenerate(body.repo, gh)
    finally:
        sessions.tool_lock.release()
    if not desc:
        raise HTTPException(status_code=502,
                            detail=f"生成 {body.repo} 的介绍失败(LLM 未配置或全部平台不可用)。")
    try:
        stats = repo_tools.live_growth(body.repo, gh, config.GROWTH_CALC_DAYS)
    except Exception:                       # noqa: BLE001
        logger.warning("刷新 %s 的 star/增长失败,只回介绍。", body.repo, exc_info=True)
        stats = {}
    return {"repo": body.repo, "sections": render.section_payload(desc), **stats}


# ── 收藏 ────────────────────────────────────────────────────────

@app.get("/api/favorites")
def favorite_list(user_id: str):
    if not favorites.valid_user_id(user_id):
        raise HTTPException(status_code=400, detail="无效的 user_id")
    counts, total = reports.appearance_counts()
    return {"user_id": user_id, "report_total": total,
            "favorites": [dict(item, report_count=counts.get(item.get("repo", ""), 0),
                               report_total=total)
                          for item in favorites.get(user_id)]}


@app.post("/api/favorites")
def favorite_update(body: FavoriteIn):
    """概要收藏时才生成,不在出报告时给几百个项目预生成。"""
    short = None
    if body.action == "add":
        already = next((x for x in favorites.get(body.user_id)
                        if x.get("repo") == body.repo), {})
        if body.short_desc is not None:
            short = body.short_desc.strip()[:repo_tools.FAVORITE_DESC_MAX]
        elif not already:
            from .provider.github import client as github
            saved = universe.load().get(body.repo, {})
            short = repo_tools.short_desc(body.repo, saved, github.shared()) or None
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

    连接断开**不中断 agent**(它可能正跑到出榜第三分钟),回复存进待发缓冲。
    返回最终回复;连接已断、回复已缓存时返回 None。
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
