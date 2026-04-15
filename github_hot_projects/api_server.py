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
from datetime import datetime
from html import escape
from contextlib import asynccontextmanager

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import markdown

from .agent import HotProjectAgent
from .common.config import DATA_DIR, LOG_DIR, REPORT_DIR

logger = logging.getLogger("discover_hot")
WEB_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "web")
CHAT_PAGE_PATH = os.path.join(WEB_DIR, "chat.html")
APP_LOG_PATH = ""


def setup_app_logging() -> str:
    """配置 API 业务日志：仅写入日期文件，不污染 uvicorn 访问日志输出。"""
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

    file_handler = logging.FileHandler(log_path, encoding="utf-8")
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


def _render_report_html(name: str, markdown_text: str) -> str:
        """将 Markdown 报告渲染为移动端友好的 HTML 页面。"""
        lines = markdown_text.splitlines()
        title = next((line[2:].strip() for line in lines if line.startswith("# ")), name)
        summary = next((line[1:].strip() for line in lines if line.startswith(">")), "")
        import re
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
        safe_summary = escape(summary)
        safe_name = escape(name)
        return f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{safe_title}</title>
    <style>
        :root {{
            --bg: #eef5ea;
            --paper: rgba(244, 249, 240, 0.97);
            --ink: #1f2329;
            --muted: #6b7280;
            --brand: #18344e;
            --accent: #d78939;
            --line: rgba(24, 52, 78, 0.12);
            --shadow: 0 18px 55px rgba(24, 52, 78, 0.12);
        }}
        * {{ box-sizing: border-box; }}
        body {{
            margin: 0;
            font-family: "Avenir Next", "Segoe UI Variable", "PingFang SC", "Noto Sans SC", sans-serif;
            color: var(--ink);
            background:
                radial-gradient(circle at top left, rgba(212, 229, 214, 0.82), transparent 28%),
                radial-gradient(circle at top right, rgba(240, 229, 206, 0.72), transparent 24%),
                linear-gradient(180deg, #eef5ea 0%, #e4ede0 100%);
            line-height: 1.75;
            padding: 24px 16px 40px;
        }}
        .page {{ max-width: 980px; margin: 0 auto; }}
        .hero {{
            padding: 28px 24px;
            border-radius: 28px;
            background: linear-gradient(135deg, rgba(24, 52, 78, 0.98), rgba(47, 91, 130, 0.92));
            color: #fff;
            box-shadow: var(--shadow);
        }}
        .eyebrow {{
            display: inline-flex;
            padding: 6px 10px;
            border-radius: 999px;
            background: rgba(255, 255, 255, 0.14);
            font-size: 12px;
            text-transform: uppercase;
            letter-spacing: 0.08em;
        }}
        h1 {{ margin: 14px 0 10px; font-size: clamp(30px, 6vw, 46px); line-height: 1.06; letter-spacing: -0.03em; }}
        .summary {{
            margin-top: 14px;
            padding: 14px 16px;
            border-radius: 18px;
            background: rgba(255, 255, 255, 0.1);
            color: rgba(255, 255, 255, 0.88);
            font-size: 14px;
        }}
        .toolbar {{ display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 10px; margin: 18px 0 0; }}
        .toolbar a {{
            display: inline-flex;
            align-items: center;
            justify-content: center;
            min-height: 42px;
            width: 100%;
            padding: 0 16px;
            border-radius: 999px;
            text-decoration: none;
            font-weight: 700;
            font-size: 14px;
            letter-spacing: 0.02em;
            color: #fff;
            background: rgba(255, 255, 255, 0.12);
            border: 1px solid rgba(255, 255, 255, 0.14);
        }}
        .content {{
            margin-top: 18px;
            padding: 26px 22px;
            border-radius: 28px;
            background: var(--paper);
            border: 1px solid var(--line);
            box-shadow: var(--shadow);
            font-size: 17px;
            line-height: 1.86;
            text-rendering: optimizeLegibility;
            -webkit-font-smoothing: antialiased;
        }}
        .content h1, .content h2, .content h3 {{ color: var(--brand); line-height: 1.25; scroll-margin-top: 24px; }}
        .content h1 {{ font-size: 34px; margin-top: 0; }}
        .content h2 {{ margin-top: 28px; font-size: 24px; padding-bottom: 10px; border-bottom: 1px solid var(--line); }}
        .content h3 {{ margin-top: 20px; font-size: 19px; }}
        .content p {{ margin: 14px 0; word-break: normal; overflow-wrap: break-word; }}
        .content blockquote {{
            margin: 16px 0;
            padding: 14px 16px;
            border-left: 4px solid var(--accent);
            background: rgba(248, 216, 176, 0.22);
            color: #334155;
            border-radius: 0 18px 18px 0;
        }}
        .content hr {{ border: 0; border-top: 1px dashed rgba(24, 52, 78, 0.18); margin: 28px 0; }}
        .content a {{ color: var(--brand); font-weight: 700; text-decoration: none; overflow-wrap: break-word; word-break: normal; }}
        .content a:hover {{ text-decoration: underline; }}
        .content p a, .content li a, .content td a, .content th a {{ display: inline-block; max-width: 100%; overflow-wrap: anywhere; }}
        .content li, .content td, .content th {{ word-break: normal; overflow-wrap: break-word; }}
        .content code {{
            padding: 2px 6px;
            border-radius: 8px;
            background: rgba(24, 52, 78, 0.08);
            font-family: "JetBrains Mono", "Cascadia Mono", monospace;
            font-size: 0.92em;
        }}
        .content pre {{
            overflow: auto;
            padding: 14px;
            border-radius: 18px;
            background: #112132;
            color: #e5edf6;
            line-height: 1.7;
        }}
        .content pre code {{ background: transparent; padding: 0; color: inherit; }}
        .content ul, .content ol {{ padding-left: 20px; }}
        .content li + li {{ margin-top: 8px; }}
        .footer {{ margin-top: 12px; color: var(--muted); font-size: 13px; text-align: center; }}
        @media (max-width: 640px) {{
            body {{ padding: 12px 10px 28px; }}
            .hero {{ padding: 22px 18px; border-radius: 24px; }}
            .content {{ padding: 20px 16px; border-radius: 24px; }}
            .content h2 {{ font-size: 22px; }}
            .content {{ font-size: 16px; line-height: 1.82; }}
        }}
        .toc-overlay {{
            position: fixed;
            inset: 0;
            z-index: 999;
            background: rgba(15, 23, 42, 0.45);
            backdrop-filter: blur(8px);
            display: none;
        }}
        .toc-overlay.active {{ display: flex; justify-content: center; align-items: flex-start; padding-top: 60px; }}
        .toc-panel {{
            background: linear-gradient(180deg, rgba(250, 253, 247, 0.98), rgba(244, 248, 240, 0.98));
            border: 1px solid rgba(24, 52, 78, 0.10);
            border-radius: 26px;
            padding: 24px 20px;
            max-width: 540px;
            width: 90%;
            max-height: 70vh;
            overflow-y: auto;
            box-shadow: var(--shadow);
        }}
        .toc-panel h3 {{ margin: 0; font-size: 18px; color: var(--brand); }}
        .toc-subtitle {{ margin: 8px 0 16px; color: var(--muted); font-size: 13px; line-height: 1.6; }}
        .toc-empty {{ margin: 0; color: var(--muted); line-height: 1.7; }}
        .toc-panel ul {{ list-style: none; padding: 0; margin: 0; }}
        .toc-panel .toc > ul {{ display: grid; gap: 10px; }}
        .toc-panel .toc > ul > li {{
            margin: 0;
            padding: 12px 14px;
            border-radius: 18px;
            background: rgba(216, 229, 214, 0.55);
            border: 1px solid rgba(24, 52, 78, 0.08);
        }}
        .toc-panel li ul {{ margin-top: 8px; padding-left: 12px; border-left: 2px solid rgba(24, 52, 78, 0.12); }}
        .toc-panel li li {{ margin: 4px 0 0; padding-left: 0; }}
        .toc-panel a {{ color: #111827; text-decoration: underline; text-decoration-color: rgba(17, 24, 39, 0.35); text-underline-offset: 3px; font-size: 15px; font-weight: 700; line-height: 1.6; display: block; padding: 6px 8px; border-radius: 12px; transition: background 160ms ease, text-decoration-color 160ms ease; }}
        .toc-panel li li a {{ font-size: 14px; font-weight: 600; color: #1f2937; }}
        .toc-panel a:hover {{ color: #111827; background: rgba(17, 24, 39, 0.04); text-decoration-color: rgba(17, 24, 39, 0.65); }}
    </style>
</head>
<body>
    <div class="page">
        <header class="hero">
            <h1>{safe_title}</h1>
            <div class="summary">{safe_summary or '这是一份由服务器根据 Markdown 报告渲染出的移动端可读网页。'}</div>
            <div class="toolbar">
                <a href="/chat">返回聊天</a>
                <a href="#toc" id="toc-trigger">目录导航</a>
            </div>
        </header>

        <div class="toc-overlay" id="toc-overlay" aria-hidden="true">
            <div class="toc-panel">
                <h3>报告目录</h3>
                <p class="toc-subtitle">点击目录项可直接跳到正文，对应浏览器返回会先回到目录视图。</p>
                {toc_html}
            </div>
        </div>

        <article class="content">
            {article_html}
        </article>

        <div class="footer">GitHub Hot Projects · Mobile Report View</div>
    </div>
    <script>
        (function setupTocNavigation() {{
            const tocOverlay = document.getElementById("toc-overlay");
            const tocTrigger = document.getElementById("toc-trigger");
            const tocHash = "#toc";

            function setOverlayVisible(visible) {{
                tocOverlay.classList.toggle("active", visible);
                tocOverlay.setAttribute("aria-hidden", visible ? "false" : "true");
            }}

            function syncFromHash() {{
                setOverlayVisible(window.location.hash === tocHash);
            }}

            tocTrigger.addEventListener("click", function(event) {{
                event.preventDefault();
                if (window.location.hash !== tocHash) {{
                    history.pushState({{ toc: true }}, "", tocHash);
                }}
                syncFromHash();
            }});

            tocOverlay.addEventListener("click", function(event) {{
                if (event.target !== tocOverlay) {{
                    return;
                }}
                if (window.location.hash === tocHash) {{
                    history.replaceState(null, "", window.location.pathname + window.location.search);
                }}
                syncFromHash();
            }});

            tocOverlay.querySelectorAll('a[href^="#"]').forEach(function(link) {{
                link.addEventListener("click", function() {{
                    setOverlayVisible(false);
                }});
            }});

            window.addEventListener("hashchange", syncFromHash);
            syncFromHash();
        }})();
    </script>
</body>
</html>
"""


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
    if not os.path.isfile(CHAT_PAGE_PATH):
        raise HTTPException(status_code=404, detail="聊天页面不存在")
    return FileResponse(
        CHAT_PAGE_PATH,
        headers={
            "Cache-Control": "no-store, no-cache, must-revalidate, max-age=0",
            "Pragma": "no-cache",
            "Expires": "0",
        },
    )


@app.get("/chat", response_class=FileResponse)
async def chat_page():
    """提供移动端聊天页静态文件。"""
    if not os.path.isfile(CHAT_PAGE_PATH):
        raise HTTPException(status_code=404, detail="聊天页面不存在")
    return FileResponse(
        CHAT_PAGE_PATH,
        headers={
            "Cache-Control": "no-store, no-cache, must-revalidate, max-age=0",
            "Pragma": "no-cache",
            "Expires": "0",
        },
    )


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
    return HTMLResponse(content=_render_report_html(name, content))


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

if __name__ == "__main__":
    import uvicorn

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    uvicorn.run(
        "github_hot_projects.api_server:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
    )
