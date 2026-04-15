"""
API Server — FastAPI Web 服务入口
=================================
将 HotProjectAgent 封装为 HTTP REST API + WebSocket，
提供 Web/手机端对话和报告查询能力。

启动方式：
  # 开发环境
  uvicorn github_hot_projects.api_server:app --host 0.0.0.0 --port 8000 --reload

  # 生产环境（进程挂起）
  nohup uvicorn github_hot_projects.api_server:app --host 0.0.0.0 --port 8000 --workers 1 &

  # 或使用 python -m 启动
  python -m github_hot_projects.api_server

API 接口：
  POST /api/chat          — 发送消息，返回 Agent 回复
  GET  /api/reports        — 获取报告列表
  GET  /api/reports/{name} — 获取单个报告内容
  GET  /api/status         — 服务状态检查
  WS   /ws/chat/{sid}      — WebSocket 实时对话（预留）

依赖：
  pip install fastapi uvicorn
"""

import logging
import os
import glob
import time
import asyncio
import threading
from contextlib import asynccontextmanager

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from .agent import HotProjectAgent
from .common.config import DATA_DIR, LOG_DIR, REPORT_DIR

logger = logging.getLogger("discover_hot")


# ══════════════════════════════════════════════════════════════
# 请求 / 响应 模型
# ══════════════════════════════════════════════════════════════

class ChatRequest(BaseModel):
    session_id: str = "default"
    message: str

class ChatResponse(BaseModel):
    session_id: str
    reply: str


# ══════════════════════════════════════════════════════════════
# 会话管理（内存版 + TTL；后续可替换为 Redis/DB）
# ══════════════════════════════════════════════════════════════

_SESSION_TTL = 3600  # 会话过期时间（秒），默认 1 小时
_MAX_SESSIONS = 100  # 最大会话数，防止内存泄漏

_sessions: dict[str, tuple[HotProjectAgent, float]] = {}  # {sid: (agent, last_access_time)}
_sessions_lock = threading.Lock()


def _cleanup_expired_sessions() -> None:
    """清理过期会话。调用者需持有 _sessions_lock。"""
    now = time.time()
    expired = [sid for sid, (_, ts) in _sessions.items() if now - ts > _SESSION_TTL]
    for sid in expired:
        del _sessions[sid]
        logger.info(f"会话过期已清理: {sid}")


def get_agent(session_id: str) -> HotProjectAgent:
    """获取或创建 Agent 实例（按 session_id 隔离，自带 TTL 清理，线程安全）。"""
    with _sessions_lock:
        _cleanup_expired_sessions()
        if session_id in _sessions:
            agent, _ = _sessions[session_id]
            _sessions[session_id] = (agent, time.time())
            return agent
        if len(_sessions) >= _MAX_SESSIONS:
            # 淘汰最久未访问的会话
            oldest_sid = min(_sessions, key=lambda k: _sessions[k][1])
            del _sessions[oldest_sid]
            logger.info(f"会话数达上限，淘汰最旧: {oldest_sid}")
        agent = HotProjectAgent()
        _sessions[session_id] = (agent, time.time())
        logger.info(f"创建新会话: {session_id}")
        return agent


# ── 全局 Tool 执行锁：防止多会话同时创建 TokenWorkerPool 导致 Token 竞争 ──
_tool_execution_lock = threading.Lock()


# ══════════════════════════════════════════════════════════════
# FastAPI App
# ══════════════════════════════════════════════════════════════

@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期：启动时初始化目录。"""
    os.makedirs(REPORT_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)
    logger.info(f"API Server 启动，数据目录: {DATA_DIR}")
    yield
    logger.info("API Server 关闭")

app = FastAPI(
    title="GitHub Hot Projects Agent API",
    description="基于 ReAct Agent 的 GitHub 热门项目发现服务",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS 配置（允许前端跨域访问）
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 生产环境应限制为具体域名
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ══════════════════════════════════════════════════════════════
# REST API
# ══════════════════════════════════════════════════════════════

@app.get("/api/status")
async def status():
    """服务状态检查。"""
    with _sessions_lock:
        active = len(_sessions)
    return {
        "status": "running",
        "active_sessions": active,
        "data_dir": DATA_DIR,
    }


@app.post("/api/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    """
    对话接口：发送消息给 Agent，返回回复。

    使用同步 def（非 async）：FastAPI 自动将其放入线程池执行，
    避免阻塞事件循环。全局互斥锁防止多会话同时占用 GitHub Token。

    - session_id: 会话标识，同一 session_id 共享对话上下文
    - message:    用户消息（自然语言）
    """
    agent = get_agent(req.session_id)
    with _tool_execution_lock:
        reply = agent.chat(req.message)
    return ChatResponse(session_id=req.session_id, reply=reply)


@app.get("/api/reports")
async def list_reports():
    """获取已生成的报告列表。"""
    os.makedirs(REPORT_DIR, exist_ok=True)
    files = sorted(glob.glob(os.path.join(REPORT_DIR, "*.md")), reverse=True)
    return {
        "reports": [
            {"name": os.path.basename(f), "path": f, "size": os.path.getsize(f)}
            for f in files
        ]
    }


@app.get("/api/reports/{name}")
async def get_report(name: str):
    """获取单个报告内容（Markdown 文本）。"""
    # 防止路径穿越
    if "/" in name or "\\" in name or ".." in name:
        raise HTTPException(status_code=400, detail="无效的报告名称")
    path = os.path.join(REPORT_DIR, name)
    if not os.path.isfile(path):
        raise HTTPException(status_code=404, detail="报告不存在")
    try:
        with open(path, "r", encoding="utf-8") as f:
            content = f.read()
    except IOError:
        raise HTTPException(status_code=500, detail="无法读取报告")
    return {"name": name, "content": content}


@app.delete("/api/sessions/{session_id}")
async def delete_session(session_id: str):
    """清除指定会话（释放内存）。"""
    with _sessions_lock:
        if session_id in _sessions:
            del _sessions[session_id]
            return {"message": f"会话 {session_id} 已清除"}
    raise HTTPException(status_code=404, detail="会话不存在")


# ══════════════════════════════════════════════════════════════
# WebSocket（预留，未来支持流式输出）
# ══════════════════════════════════════════════════════════════

@app.websocket("/ws/chat/{session_id}")
async def ws_chat(websocket: WebSocket, session_id: str):
    """
    WebSocket 实时对话（预留接口）。

    当前实现：接收消息 → 调用 Agent → 返回完整回复。
    使用 asyncio.to_thread 避免阻塞事件循环。
    全局互斥锁防止多会话同时占用 GitHub Token。
    未来：LLM 流式输出时逐 token 推送。
    """
    await websocket.accept()
    agent = get_agent(session_id)

    def _chat_with_lock(message: str) -> str:
        with _tool_execution_lock:
            return agent.chat(message)

    try:
        while True:
            data = await websocket.receive_text()
            reply = await asyncio.to_thread(_chat_with_lock, data)
            await websocket.send_text(reply)
    except WebSocketDisconnect:
        logger.info(f"WebSocket 断开: {session_id}")


# ══════════════════════════════════════════════════════════════
# 直接运行支持: python -m github_hot_projects.api_server
# ══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import uvicorn

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    uvicorn.run(
        "github_hot_projects.api_server:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
    )
