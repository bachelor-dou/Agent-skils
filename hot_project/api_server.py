#!/usr/bin/env python
"""Web / API 服务。

    python -m hot_project.api_server

这个文件只做**路由**:收请求、调 web/{render,sessions,security},把结果变成 HTTP 响应。

**规矩:碰存储或渲染的处理器一律写同步 `def`。** FastAPI 会把同步 `def` 丢进线程池,而
agent 一路是同步的,写成 `async def` 会把事件循环连同 WS 心跳一起锁死几分钟;只有纯内存/
纯配置的接口才留 `async def`。这也是工具能直接 `asyncio.run` 的前提:它永不在事件循环里跑。
"""

from __future__ import annotations

import logging
import re
from contextlib import asynccontextmanager
from datetime import datetime

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from . import config
from .common import logs
from .infra import llm
from .infra.data_access import reports
from .service import favorites as favorite_service
from .service import repo_card
from .web import chat_options, protocol, pump, render, security, sessions, view_model

logger = logging.getLogger("hot_project")

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
    thinking: str = ""      # 思考档位;空 = 用默认档(高)


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
    subcategory: str | None = None      # 二级细分,只在 category 非空时有意义
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

BUSY_DETAIL = "系统繁忙,请稍后重试。"


@app.exception_handler(sessions.Busy)
async def _busy_to_503(request, exc: sessions.Busy) -> JSONResponse:
    """`Busy` → 503 只翻译这一次;WS 不走这里,在自己的闭包里接。"""
    return JSONResponse(status_code=503, content={"detail": BUSY_DETAIL})


def _page(name: str, missing: str, replacements: dict[str, str] | None = None) -> HTMLResponse:
    try:
        html = render.page(name, replacements or {})
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
    # TTL 的唯一真值在 sessions,前端的过期提示从这里注入,别在 JS 里硬编码。
    return _page("chat.html", "聊天页面不存在",
                 {"__SESSION_TTL_SECONDS__": str(sessions.TTL_SECONDS)})


# ── 对话 ────────────────────────────────────────────────────────

@app.post("/api/chat", response_model=ChatOut)
def chat(body: ChatIn):
    """同步 `def` 是有意的,见文件头。"""
    logger.info("HTTP 对话:session=%s message=%s", body.session_id, body.message[:120])
    agent = sessions.get(body.session_id)
    options = chat_options.parse(body.model_dump())
    with sessions.hold_tool_lock(f"HTTP 对话 {body.session_id}"):
        reply = agent.chat(body.message, **options.kwargs())
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
    apis = llm.get().usable()
    pool, seen = [], set()
    for api in apis:
        for name in api.lite_models:
            if name not in seen:
                seen.add(name)
                pool.append({"id": f"{api.id}:{name}", "label": name.rsplit("/", 1)[-1]})
    return {"models": [a.public() for a in apis], "lite_models": pool}


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
    return repo_card.trend(repo)


class DescribeIn(BaseModel):
    repo: str


@app.post("/api/repo-describe")
def repo_describe(body: DescribeIn):
    """报告卡片「刷新介绍」。编排在 `repo_card.refresh`;这里只做校验、锁、状态码、渲染。

    和对话共用同一把执行锁,不让并发刷新和出榜互相抢 token。同步 `def`,理由见文件头。
    """
    if not REPO_NAME.match(body.repo or ""):
        raise HTTPException(status_code=400, detail="无效的仓库名")
    from .provider.github import client as github
    if not (gh := github.shared()).usable:
        raise HTTPException(status_code=503, detail="没有可用的 GitHub token,无法刷新介绍。")
    with sessions.hold_tool_lock(f"刷新介绍 {body.repo}"):
        out = repo_card.refresh(body.repo, gh, config.GROWTH_CALC_DAYS)
    if not (desc := out.pop("desc")):
        raise HTTPException(status_code=502,
                            detail=f"生成 {body.repo} 的介绍失败(LLM 未配置或全部平台不可用)。")
    return {"repo": body.repo, "sections": view_model.section_payload(desc), **out}


# ── 收藏 ────────────────────────────────────────────────────────

@app.get("/api/favorites")
def favorite_list(user_id: str):
    try:
        items, total = favorite_service.listing(user_id)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    return {"user_id": user_id, "report_total": total, "favorites": items}


@app.post("/api/favorites")
def favorite_update(body: FavoriteIn):
    try:
        items, total = favorite_service.update(
            body.user_id, body.repo, body.action, source_report=body.source_report,
            category=body.category, subcategory=body.subcategory,
            short_desc=body.short_desc)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    return {"user_id": body.user_id, "report_total": total, "favorites": items}


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
    # 选项在连接时读一次:切模型或档位由前端重连,所以一条连接的选项是固定的
    options = chat_options.parse(websocket.query_params)
    logger.info("WS 已连接:%s(user=%s model=%s lite=%s 思考=%s)",
                session_id, options.user_id or "-", options.model or "-",
                options.lite or "-", options.effort)

    await protocol.replay(websocket, session_id)

    def run(message: str, progress, on_delta) -> str:
        agent = sessions.get(session_id)
        with sessions.hold_tool_lock(f"WS 对话 {session_id}"):
            return agent.chat(message, progress=progress, on_delta=on_delta,
                              **options.kwargs())

    try:
        while True:
            message = await websocket.receive_text()
            logger.info("WS 收到:%s %s", session_id, message[:120])
            if await _pump(websocket, session_id, message, run) is None:
                break       # 连接断了,回复已经存起来
    except WebSocketDisconnect:
        logger.info("WS 断开:%s", session_id)


async def _pump(websocket, session_id: str, message: str, run) -> str | None:
    """跑一轮对话并实时推进度/增量。线程和心跳的机械在 `web.pump`,帧的形状和补发
    契约在 `web.protocol`;这里只把两者接起来:worker 的产出一律是线上帧。

    返回送达的最终帧;连接已断、帧已进待发缓冲时返回 None。
    """

    def worker(emit) -> str:
        try:
            return protocol.reply(run(
                message,
                lambda percent, label: emit(protocol.progress(percent, label)),
                lambda text, reset=False: emit(protocol.delta(text, reset)),
            ))
        except sessions.Busy:
            return protocol.error(BUSY_DETAIL)
        except Exception as e:      # noqa: BLE001 —— 一轮失败要告诉前端,不能静默吞掉
            logger.exception("WS 对话异常:%s", session_id)
            return protocol.error(f"处理消息时出错:{e}")

    got = await pump.drive(websocket, worker, heartbeat=WS_HEARTBEAT, poll=WS_POLL)
    logger.info("WS 回复完成:%s(%d 字)", session_id, len(got.result))

    if await protocol.deliver(websocket, session_id, got.result, alive=got.alive):
        return got.result
    return None


def main() -> None:
    import uvicorn
    logs.setup(config.LOG_DIR, "web", console=True)
    uvicorn.run("hot_project.api_server:app", host="0.0.0.0",
                port=config.WEB_PORT, reload=False)


if __name__ == "__main__":
    main()
