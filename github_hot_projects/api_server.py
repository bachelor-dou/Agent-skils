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
import logging.handlers
import os
import glob
import time
import asyncio
import threading
import collections
import re
from datetime import datetime
from html import escape
from contextlib import asynccontextmanager

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, Request
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from starlette.middleware.base import BaseHTTPMiddleware
from pydantic import BaseModel
import markdown

import hashlib

from .agent import HotProjectAgent
from .common.config import DATA_DIR, LOG_DIR, REPORT_DIR

logger = logging.getLogger("discover_hot")
WEB_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "web")
CHAT_PAGE_PATH = os.path.join(WEB_DIR, "chat.html")
REPORT_PAGE_TEMPLATE_PATH = os.path.join(WEB_DIR, "report.html")
APP_LOG_PATH = ""


def _compute_asset_version() -> str:
    """根据 web/ 目录下所有文件的修改时间生成版本哈希（服务启动时计算一次）。"""
    h = hashlib.md5(usedforsecurity=False)
    for name in sorted(os.listdir(WEB_DIR)):
        fpath = os.path.join(WEB_DIR, name)
        if os.path.isfile(fpath):
            h.update(f"{name}:{os.path.getmtime(fpath)}".encode())
    return h.hexdigest()[:10]


ASSET_VERSION = _compute_asset_version()
PAGE_NO_CACHE_HEADERS = {
    "Cache-Control": "no-store, no-cache, must-revalidate, max-age=0",
    "Pragma": "no-cache",
    "Expires": "0",
}


def setup_app_logging() -> str:
    """配置 API 业务日志：使用 RotatingFileHandler 防止单日志过大，不污染 uvicorn 访问日志输出。"""
    os.makedirs(LOG_DIR, exist_ok=True)
    log_path = os.path.join(
        LOG_DIR,
        f"agent-{datetime.now().strftime('%Y-%m-%d')}.log",
    )

    for handler in list(logger.handlers):
        logger.removeHandler(handler)
        try:
            handler.close()
        except Exception:
            pass

    file_handler = logging.handlers.RotatingFileHandler(
        log_path, maxBytes=50 * 1024 * 1024, backupCount=3, encoding="utf-8",
    )
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    )

    logger.setLevel(logging.INFO)
    logger.addHandler(file_handler)
    logger.propagate = False
    return log_path


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

# 待发回复缓冲：WebSocket 断开期间产生的回复，重连后推送
_pending_replies: dict[str, list[str]] = {}
_pending_replies_lock = threading.Lock()


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


def _validate_report_name(name: str) -> str:
    """校验报告名，防止路径穿越。"""
    if "/" in name or "\\" in name or ".." in name:
        raise HTTPException(status_code=400, detail="无效的报告名称")
    return os.path.join(REPORT_DIR, name)


def _read_report_content(name: str) -> str:
    """读取报告 Markdown 文本。"""
    path = _validate_report_name(name)
    if not os.path.isfile(path):
        raise HTTPException(status_code=404, detail="报告不存在")
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    except IOError as exc:
        raise HTTPException(status_code=500, detail="无法读取报告") from exc


def _load_web_text_asset(path: str) -> str:
    """读取 web 目录中的模板/静态文本资源。"""
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    except OSError as exc:
        raise HTTPException(status_code=500, detail=f"无法加载 Web 资源: {os.path.basename(path)}") from exc


def _render_web_template(path: str, replacements: dict[str, str]) -> str:
    """将占位符模板渲染为最终 HTML（自动包含 __ASSET_VER__）。"""
    document = _load_web_text_asset(path)
    replacements.setdefault("__ASSET_VER__", ASSET_VERSION)
    for placeholder, value in replacements.items():
        document = document.replace(placeholder, value)
    return document


def _build_page_response(path: str, missing_detail: str) -> HTMLResponse:
    """统一返回 no-cache 页面响应，自动替换 __ASSET_VER__ 占位符。"""
    if not os.path.isfile(path):
        raise HTTPException(status_code=404, detail=missing_detail)
    content = _load_web_text_asset(path).replace("__ASSET_VER__", ASSET_VERSION)
    return HTMLResponse(content, headers=PAGE_NO_CACHE_HEADERS)


def _render_report_html(name: str, markdown_text: str) -> str:
    """将 Markdown 报告渲染为移动端友好的 HTML 页面。"""
    lines = markdown_text.splitlines()
    title = next((line[2:].strip() for line in lines if line.startswith("# ")), name)
    summary = next((line[1:].strip() for line in lines if line.startswith(">")), "")
    # 预处理：移除 Markdown 中的原始 HTML 标签，防止 XSS
    sanitized_text = re.sub(r'<(script|iframe|object|embed|form|input|style)[^>]*>.*?</\1>', '', markdown_text, flags=re.DOTALL | re.IGNORECASE)
    sanitized_text = re.sub(r'<(script|iframe|object|embed|form|input|style)[^>]*/?\s*>', '', sanitized_text, flags=re.IGNORECASE)
    sanitized_text = re.sub(r'\bon\w+\s*=', '', sanitized_text, flags=re.IGNORECASE)

    md = markdown.Markdown(
        extensions=["extra", "sane_lists", "toc", "nl2br"],
        output_format="html5",
    )
    article_html = md.convert(sanitized_text)
    toc_html = getattr(md, "toc", "")
    if not toc_html.strip():
        toc_html = '<p class="toc-empty">当前报告暂无可跳转目录。</p>'

    safe_title = escape(title)
    safe_summary = escape(summary or "这是一份由服务器根据 Markdown 报告渲染出的移动端可读网页。")
    safe_name = escape(name)
    return _render_web_template(
        REPORT_PAGE_TEMPLATE_PATH,
        {
            "__REPORT_NAME__": safe_name,
            "__REPORT_TITLE__": safe_title,
            "__REPORT_SUMMARY__": safe_summary,
            "__REPORT_TOC_HTML__": toc_html,
            "__REPORT_ARTICLE_HTML__": article_html,
        },
    )


# ══════════════════════════════════════════════════════════════
# FastAPI App
# ══════════════════════════════════════════════════════════════

@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期：启动时初始化目录。"""
    global APP_LOG_PATH
    os.makedirs(REPORT_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)
    APP_LOG_PATH = setup_app_logging()
    logger.info(f"API Server 启动，数据目录: {DATA_DIR}")
    logger.info(f"API Server 业务日志文件: {APP_LOG_PATH}")
    yield
    logger.info("API Server 关闭")


# ══════════════════════════════════════════════════════════════
# 安全中间件：IP 黑名单 + 速率限制 + 敏感路径拦截
# ══════════════════════════════════════════════════════════════

# 已确认的恶意扫描 IP（从日志分析得出）
_IP_BLACKLIST: set[str] = {
    "104.243.32.126",
    "209.222.101.194",
    "172.232.209.215",
}

# 敏感路径前缀 — 命中即返回 404，不暴露任何信息
_BLOCKED_PATH_PREFIXES: tuple[str, ...] = (
    "/.env", "/.git", "/.well-known/mcp", "/.well-known/agent",
    "/.well-known/ai-plugin", "/v1/models", "/v1/chat/completions",
    "/v1/embeddings", "/api/tags", "/console/api", "/graphql",
    "/debug", "/config", "/_cluster", "/_cat", "/_ml",
    "/admin", "/login", "/swagger", "/internal",
    "/copilot_internal", "/openai/", "/sdapi/",
)

# 速率限制：滑动窗口，每 IP 每分钟最多 _RATE_LIMIT 次请求
_RATE_WINDOW = 60          # 窗口秒数
_RATE_LIMIT = 120          # 窗口内最大请求数
_rate_records: dict[str, collections.deque] = {}
_rate_lock = threading.Lock()


def _get_client_ip(request: Request) -> str:
    """提取客户端真实 IP（支持反代 X-Forwarded-For）。"""
    forwarded = request.headers.get("x-forwarded-for")
    if forwarded:
        return forwarded.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


def _is_rate_limited(ip: str) -> bool:
    """检查 IP 是否超出速率限制。"""
    now = time.time()
    with _rate_lock:
        if ip not in _rate_records:
            _rate_records[ip] = collections.deque()
        dq = _rate_records[ip]
        # 清理过期记录
        while dq and dq[0] < now - _RATE_WINDOW:
            dq.popleft()
        if len(dq) >= _RATE_LIMIT:
            return True
        dq.append(now)
        return False


class SecurityMiddleware(BaseHTTPMiddleware):
    """统一安全中间件：黑名单 → 敏感路径 → 速率限制 → 请求日志。"""

    async def dispatch(self, request: Request, call_next):
        client_ip = _get_client_ip(request)

        # 1. IP 黑名单
        if client_ip in _IP_BLACKLIST:
            logger.warning(f"黑名单拦截: {client_ip} {request.url.path}")
            return JSONResponse(status_code=403, content={"detail": "Forbidden"})

        # 2. 敏感路径拦截
        path = request.url.path.lower()
        if any(path.startswith(p) for p in _BLOCKED_PATH_PREFIXES):
            logger.warning(f"敏感路径拦截: {client_ip} {request.url.path}")
            return JSONResponse(status_code=404, content={"detail": "Not Found"})

        # 3. 速率限制
        if _is_rate_limited(client_ip):
            logger.warning(f"速率限制触发: {client_ip} {request.url.path}")
            return JSONResponse(status_code=429, content={"detail": "Too Many Requests"})

        # 4. 请求日志 + 响应计时
        start = time.time()
        response = await call_next(request)
        duration_ms = (time.time() - start) * 1000
        logger.info(
            "%s %s %s %.0fms %s",
            request.method, request.url.path, client_ip, duration_ms, response.status_code,
        )
        return response


app = FastAPI(
    title="GitHub Hot Projects Agent API",
    description="基于 ReAct Agent 的 GitHub 热门项目发现服务",
    version="1.0.0",
    lifespan=lifespan,
)

if os.path.isdir(WEB_DIR):
    app.mount("/web", StaticFiles(directory=WEB_DIR), name="web")

# CORS 配置（允许前端跨域访问）
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 生产环境应限制为具体域名
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 安全中间件（注册在 CORS 之后，Starlette 中间件栈后注册先执行）
app.add_middleware(SecurityMiddleware)


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
        "log_path": APP_LOG_PATH,
    }


@app.get("/", response_class=FileResponse)
async def index():
    """默认打开移动端聊天页。"""
    return _build_page_response(CHAT_PAGE_PATH, "聊天页面不存在")


@app.get("/chat", response_class=FileResponse)
async def chat_page():
    """提供移动端聊天页静态文件。"""
    return _build_page_response(CHAT_PAGE_PATH, "聊天页面不存在")


@app.post("/api/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    """
    对话接口：发送消息给 Agent，返回回复。

    使用同步 def（非 async）：FastAPI 自动将其放入线程池执行，
    避免阻塞事件循环。全局互斥锁防止多会话同时占用 GitHub Token。

    - session_id: 会话标识，同一 session_id 共享对话上下文
    - message:    用户消息（自然语言）
    """
    try:
        logger.info(
            "HTTP 对话开始: session=%s, message=%s",
            req.session_id,
            req.message[:120],
        )
        agent = get_agent(req.session_id)
        with _tool_execution_lock:
            reply = agent.chat(req.message)
        logger.info(
            "HTTP 对话完成: session=%s, reply_len=%s",
            req.session_id,
            len(reply or ""),
        )
    except SystemExit as exc:
        raise HTTPException(
            status_code=503,
            detail="未配置任何 GitHub Token，无法运行。请设置 GITHUB_TOKENS 环境变量。",
        ) from exc
    return ChatResponse(session_id=req.session_id, reply=reply)


@app.get("/api/reports")
async def list_reports():
    """获取已生成的报告列表。"""
    os.makedirs(REPORT_DIR, exist_ok=True)
    files = sorted(
        glob.glob(os.path.join(REPORT_DIR, "*.md")),
        key=os.path.getmtime,
        reverse=True,
    )
    return {
        "reports": [
            {
                "name": os.path.basename(f),
                "path": f,
                "size": os.path.getsize(f),
                "modified_at": datetime.fromtimestamp(os.path.getmtime(f)).isoformat(),
            }
            for f in files
        ]
    }


@app.get("/api/reports/{name}")
async def get_report(name: str):
    """获取单个报告内容（Markdown 文本）。"""
    content = _read_report_content(name)
    return {"name": name, "content": content}


@app.get("/api/reports/{name}/html", response_class=HTMLResponse)
async def get_report_html(name: str):
    """获取单个报告的 HTML 渲染页面。"""
    content = _read_report_content(name)
    return HTMLResponse(
        content=_render_report_html(name, content),
        headers=PAGE_NO_CACHE_HEADERS,
    )


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
    WebSocket 实时对话。

    当前实现：接收消息 → 调用 Agent → 返回完整回复。
    使用 asyncio.to_thread 避免阻塞事件循环。
    全局互斥锁防止多会话同时占用 GitHub Token。
    支持重连后推送断开期间的待发回复。
    """
    await websocket.accept()
    logger.info("WebSocket 已连接: %s", session_id)

    # 推送断开期间缓存的待发回复
    with _pending_replies_lock:
        pending = _pending_replies.pop(session_id, [])
    for reply in pending:
        try:
            await websocket.send_text(reply)
            logger.info("WebSocket 推送待发回复: session=%s, reply_len=%s", session_id, len(reply))
        except Exception:
            # 推送失败时放回缓冲
            with _pending_replies_lock:
                _pending_replies.setdefault(session_id, []).append(reply)
            break

    def _chat_with_lock(message: str) -> str:
        agent = get_agent(session_id)
        with _tool_execution_lock:
            return agent.chat(message)

    try:
        while True:
            data = await websocket.receive_text()
            logger.info("WebSocket 收到消息: session=%s, message=%s", session_id, data[:120])
            try:
                reply = await asyncio.to_thread(_chat_with_lock, data)
            except SystemExit:
                reply = "未配置任何 GitHub Token，当前只能预览页面与报告渲染效果。请先设置 GITHUB_TOKENS 环境变量后再发起 Agent 对话。"
            except Exception as e:
                logger.error("WebSocket Agent 执行异常: session=%s, error=%s", session_id, e)
                reply = f"处理消息时出现错误：{e}"
            logger.info("WebSocket 回复完成: session=%s, reply_len=%s", session_id, len(reply or ""))
            try:
                await websocket.send_text(reply)
            except Exception:
                # WebSocket 已断开，缓存回复供重连后推送
                logger.info("WebSocket 发送失败，缓存待发回复: session=%s", session_id)
                with _pending_replies_lock:
                    _pending_replies.setdefault(session_id, []).append(reply)
                break
    except WebSocketDisconnect:
        logger.info(f"WebSocket 断开: {session_id}")


# ══════════════════════════════════════════════════════════════
# 直接运行支持: python -m github_hot_projects.api_server
# ══════════════════════════════════════════════════════════════

def main() -> None:
    """统一的 Web/API 服务启动入口。"""
    import uvicorn

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    uvicorn.run(
        "github_hot_projects.api_server:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
    )


if __name__ == "__main__":
    main()
