"""
API Server — FastAPI Web 服务入口
=================================
将 HotProjectAgent 封装为 HTTP REST API + WebSocket，
提供 Web/手机端对话和报告查询能力。

启动方式：
  # 开发环境
  uvicorn hot_projects.api_server:app --host 0.0.0.0 --port 8001 --reload

  # 生产环境（进程挂起）
  nohup uvicorn hot_projects.api_server:app --host 0.0.0.0 --port 8001 --workers 1 &

  # 或使用 python -m 启动（main() 默认 port=8001）
  python -m hot_projects.api_server

API 接口：
  POST /api/chat          — 发送消息，返回 Agent 回复
  GET  /api/reports        — 获取报告列表
  GET  /api/reports/{name} — 获取单个报告内容
  GET  /api/status         — 服务状态检查
  WS   /ws/chat/{sid}      — WebSocket 实时对话（进度流式推送 + 末尾整段回复）

依赖：
  pip install fastapi uvicorn
"""

import logging
import logging.handlers
import os
import glob
import time
import json
import queue as _queue
import asyncio
import threading
import collections
import re
from datetime import datetime, timezone
from html import escape, unescape
from contextlib import asynccontextmanager

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, Request
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from starlette.middleware.base import BaseHTTPMiddleware
from pydantic import BaseModel
import markdown

import hashlib

from .agent import HotProjectAgent, build_agent
from .infra import favorites_store
from .tools.basic.report_parse import parse_structured_report
from .config import (
    DATA_DIR,
    DB_FILE_PATH,
    LOG_DIR,
    REPORT_DIR,
    LLM_MODELS,
    CORS_ALLOWED_ORIGINS,
    CORS_ALLOW_CREDENTIALS,
    SECURITY_IP_BLACKLIST,
    FAVORITE_DEFAULT_TAGS,
)

logger = logging.getLogger("hot_projects")
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
    """配置 API 业务日志：使用 TimedRotatingFileHandler 按日期切换日志文件。"""
    os.makedirs(LOG_DIR, exist_ok=True)
    log_path = os.path.join(LOG_DIR, "web.log")

    for handler in list(logger.handlers):
        logger.removeHandler(handler)
        try:
            handler.close()
        except Exception:
            pass

    # 每天午夜自动切换，保留 7 天备份，备份文件名格式: web.log.2026-04-28
    file_handler = logging.handlers.TimedRotatingFileHandler(
        log_path, when="midnight", interval=1, backupCount=7, encoding="utf-8",
    )
    file_handler.suffix = "%Y-%m-%d"
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
    user_id: str = ""
    model: str = ""
    lite: str = ""  # 子模型 id（"平台id:模型名"）；空=跟随主模型平台

class ChatResponse(BaseModel):
    session_id: str
    reply: str
    session_ttl_seconds: int
    session_expires_at: str


class FavoriteRequest(BaseModel):
    user_id: str
    repo: str
    action: str  # "add" | "remove"
    source_report: str = ""
    category: str | None = None  # 单一分类标签；None=不改动，""=未分类
    short_desc: str | None = None  # 用户手动编辑概要；None=按需自动生成，字符串(含"")=直接采用/清空


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


def _format_session_expiry(expires_at_ts: float | None = None) -> str:
    """返回会话过期时间的 UTC 时间戳字符串。"""
    expires_at = expires_at_ts if expires_at_ts is not None else time.time() + _SESSION_TTL
    return datetime.fromtimestamp(expires_at, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _cleanup_expired_sessions() -> None:
    """清理过期会话。调用者需持有 _sessions_lock。"""
    now = time.time()
    expired = [sid for sid, (_, ts) in _sessions.items() if now - ts > _SESSION_TTL]
    for sid in expired:
        del _sessions[sid]
        with _pending_replies_lock:
            _pending_replies.pop(sid, None)
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
            with _pending_replies_lock:
                _pending_replies.pop(oldest_sid, None)
            logger.info(f"会话数达上限，淘汰最旧: {oldest_sid}")
        agent = build_agent()
        _sessions[session_id] = (agent, time.time())
        logger.info(f"创建新会话: {session_id}")
        return agent


# ── 全局 Tool 执行锁：防止多会话并发触发扫描导致 Token 竞争 ──
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
    content = _render_web_template(
        path,
        {"__SESSION_TTL_SECONDS__": str(_SESSION_TTL)},
    )
    return HTMLResponse(content, headers=PAGE_NO_CACHE_HEADERS)


def _is_safe_report_url(url: str) -> bool:
    """仅允许报告 HTML 中出现安全协议或站内相对链接。"""
    if not url:
        return False

    normalized = unescape(url).strip()
    if not normalized:
        return False

    if normalized.startswith(("#", "/", "./", "../", "//")):
        return True

    compact = re.sub(r"[\x00-\x20]+", "", normalized)
    if compact.startswith(("#", "/", "./", "../", "//")):
        return True

    scheme_match = re.match(r"^([a-zA-Z][a-zA-Z0-9+.-]*):", compact)
    if not scheme_match:
        return True

    return scheme_match.group(1).lower() in {"http", "https", "mailto"}


def _sanitize_report_html_urls(html_text: str) -> str:
    """对渲染后的 HTML 再做一层链接协议白名单过滤。"""
    pattern = re.compile(
        r'(?P<attr>\b(?:href|src))\s*=\s*(?P<quote>["\'])(?P<value>.*?)(?P=quote)',
        re.IGNORECASE,
    )

    def _replace(match: re.Match[str]) -> str:
        attr = match.group("attr")
        quote = match.group("quote")
        value = match.group("value")
        if _is_safe_report_url(value):
            return match.group(0)
        fallback = "#" if attr.lower() == "href" else ""
        return f"{attr}={quote}{fallback}{quote}"

    return pattern.sub(_replace, html_text)


def _slugify_report_anchor(text: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "-", text.lower()).strip("-")
    return slug or "section"


def _split_report_paragraphs(text: str) -> list[str]:
    return [block.strip() for block in re.split(r"\n\s*\n", text) if block.strip()]


def _safe_report_href(url: str) -> str:
    return url if _is_safe_report_url(url) else "#"


def _parse_structured_report(markdown_text: str) -> dict | None:
    """解析结构化报告（共用 tools.basic.report_parse 的实现）。"""
    return parse_structured_report(markdown_text)


def _render_report_stat(label: str, value: str, kind: str = "") -> str:
    class_name = f"repo-stat repo-stat--{kind}" if kind else "repo-stat"
    return (
        f'<div class="{class_name}">'
        '<div class="repo-stat__body">'
        f'<span class="repo-stat__label">{escape(label)}</span>'
        f'<strong class="repo-stat__value">{escape(value)}</strong>'
        '</div>'
        '</div>'
    )


# ── 上期对比：同类报告 diff（蓝色「上新」徽章 + 排名变化） ──

# 报告名 = 日期 + 类型/区间/方向尾缀：2026-07-07.md / 2026-07-07_NEW.md /
# 2026-07-07_KEY_10d.md / 2026-07-07_KEY_向量库.md（方向可含中文）
_REPORT_NAME_RE = re.compile(r"^(?P<date>\d{4}-\d{2}-\d{2})(?P<suffix>.*)\.md$")

# 上一份报告的解析缓存：{path: (mtime, parsed)}
_prev_report_cache: dict[str, tuple[float, dict | None]] = {}


def _load_structured_report_cached(path: str) -> dict | None:
    try:
        mtime = os.path.getmtime(path)
    except OSError:
        return None
    cached = _prev_report_cache.get(path)
    if cached and cached[0] == mtime:
        return cached[1]
    try:
        with open(path, "r", encoding="utf-8") as f:
            parsed = _parse_structured_report(f.read())
    except OSError:
        parsed = None
    _prev_report_cache[path] = (mtime, parsed)
    return parsed


def _title_prefix(parsed: dict | None) -> str:
    """报告标题去掉日期部分（『GitHub 热门项目 — 2026-07-01』→『GitHub 热门项目』）。"""
    if not parsed:
        return ""
    return (parsed.get("title") or "").split("—")[0].strip()


def _find_previous_report(name: str, current_parsed: dict) -> tuple[str, dict] | None:
    """找同类型（同尾缀）且日期更早的最新一份报告，返回 (文件名, 解析结果)。

    旧文件（重命名规则之前生成的）可能类型混叠，故再用标题前缀做一层校验。
    """
    m = _REPORT_NAME_RE.match(name)
    if not m:
        return None
    cur_date, cur_suffix = m.group("date"), m.group("suffix")

    candidates: list[tuple[str, str]] = []
    try:
        entries = os.listdir(REPORT_DIR)
    except OSError:
        return None
    for fname in entries:
        pm = _REPORT_NAME_RE.match(fname)
        if not pm or fname == name:
            continue
        if pm.group("suffix") != cur_suffix or pm.group("date") >= cur_date:
            continue
        candidates.append((pm.group("date"), fname))

    for _, fname in sorted(candidates, reverse=True):
        parsed = _load_structured_report_cached(os.path.join(REPORT_DIR, fname))
        if parsed is None:
            continue
        if _title_prefix(parsed) != _title_prefix(current_parsed):
            continue  # 同名不同类（如旧版关键词榜与综合榜混叠）→ 跳过
        return fname, parsed
    return None


def _build_report_diff(name: str, current_parsed: dict) -> dict | None:
    """返回 {prev_name, prev_ranks, added, removed}；无可对比报告时返回 None。"""
    prev = _find_previous_report(name, current_parsed)
    if prev is None:
        return None
    prev_name, prev_parsed = prev
    prev_ranks = {r["repo"]: r["rank"] for r in prev_parsed.get("repos", [])}
    cur_repos = {r["repo"] for r in current_parsed.get("repos", [])}
    return {
        "prev_name": prev_name,
        "prev_ranks": prev_ranks,
        "added": sum(1 for r in cur_repos if r not in prev_ranks),
        "removed": sum(1 for r in prev_ranks if r not in cur_repos),
    }


# 常见主语言的标识色（侧栏语言圆点），未收录语言回退灰色
_LANG_COLORS = {
    "python": "#3572A5", "typescript": "#3178c6", "javascript": "#f1e05a",
    "go": "#00ADD8", "rust": "#dea584", "c++": "#f34b7d", "c": "#555555",
    "java": "#b07219", "kotlin": "#A97BFF", "swift": "#F05138", "ruby": "#701516",
    "c#": "#178600", "html": "#e34c26", "css": "#563d7c", "shell": "#89e051",
    "jupyter notebook": "#DA5B0B", "dart": "#00B4AB", "php": "#4F5D95",
    "zig": "#ec915c", "lua": "#000080", "vue": "#41b883", "svelte": "#ff3e00",
}


# ── 报告描述与 DB 实时同步 ──
# 报告 .md 里的四段描述是生成当天写死的快照。为让描述随 DB 修正/重刷而更新（star、增长等
# 统计仍保持当时快照），渲染时用 DB 的 desc 覆盖同名项目的对应小节。
# DB 有 29MB，绝大部分是 star 历史；这里只抽取非空 desc 建索引（体量小），并按文件 mtime 缓存，
# 仅在 DB 变更后重载一次。

_DB_DESC_SECTION_TITLES = (
    "项目定位与用途",
    "解决的问题",
    "使用场景",
    "技术架构与特性",
    "核心依赖与生态",
    "已知局限或注意事项",
)

# {db_path: (mtime, {repo: desc})}
_db_desc_cache: dict[str, tuple[float, dict[str, str]]] = {}


def _db_desc_index() -> dict[str, str]:
    """从 Github_DB.json 抽取 {repo: desc}（仅非空），按 mtime 缓存。失败时回退旧缓存/空。"""
    try:
        mtime = os.path.getmtime(DB_FILE_PATH)
    except OSError:
        return {}
    cached = _db_desc_cache.get(DB_FILE_PATH)
    if cached and cached[0] == mtime:
        return cached[1]
    index: dict[str, str] = {}
    try:
        with open(DB_FILE_PATH, "r", encoding="utf-8") as f:
            db = json.load(f)
        for repo, info in (db.get("projects") or {}).items():
            if isinstance(info, dict):
                desc = (info.get("desc") or "").strip()
                if desc:
                    index[repo] = desc
    except (OSError, ValueError) as exc:
        logger.warning("加载 DB 描述索引失败: %s", exc)
        return cached[1] if cached else {}
    _db_desc_cache[DB_FILE_PATH] = (mtime, index)
    logger.info("DB 描述索引已刷新: %d 条非空描述", len(index))
    return index


def _split_db_desc_sections(desc: str) -> dict[str, str]:
    """把 DB 里“标题：内容”分段的 desc 拆成 {标题: 内容}，标题限定为已知小节名。"""
    sections: dict[str, str] = {}
    current = ""
    for raw in desc.splitlines():
        line = raw.strip()
        if not line:
            continue
        matched = ""
        for title in _DB_DESC_SECTION_TITLES:
            for sep in ("：", ":"):
                if line.startswith(f"{title}{sep}"):
                    matched = title
                    sections[title] = line.split(sep, 1)[1].strip()
                    break
            if matched:
                break
        if matched:
            current = matched
            continue
        if current:
            sections[current] = f"{sections[current]} {line}".strip()
    return sections


def _render_structured_report_html(parsed: dict, diff: dict | None = None) -> tuple[str, str]:
    """主从布局：返回 (详情面板集合 article_html, 侧栏项目列表 toc_html)。

    diff 提供时标注上期对比：上期没有的项目挂蓝色「上新」徽章，排名变化挂 ↑/↓。
    """
    article_parts: list[str] = []
    nav_items: list[str] = []
    prev_ranks = diff["prev_ranks"] if diff else None
    desc_index = _db_desc_index()

    for repo in parsed["repos"]:
        repo_name = repo["repo"]
        # 用 DB 最新 desc 覆盖 .md 里写死的描述小节（保持与 DB 同步）
        db_desc = desc_index.get(repo_name)
        if db_desc:
            db_sections = _split_db_desc_sections(db_desc)
            if db_sections:
                for section in repo["sections"]:
                    override = db_sections.get(section["title"])
                    if override:
                        section["content"] = override
        metadata = repo["metadata"]
        repo_link = _safe_report_href(repo.get("link") or f"https://github.com/{repo_name}")
        readme_link = _safe_report_href(f"{repo_link}#readme") if repo_link != "#" else "#"
        anchor = f"repo-{repo['rank']}-{_slugify_report_anchor(repo_name)}"
        topic_values = [item.strip() for item in re.split(r"[，,]", metadata.get("主题标签", "")) if item.strip()]
        growth_label = next((label for label in metadata if "增长" in label), "")
        growth_value = metadata.get(growth_label, "") if growth_label else ""
        language = metadata.get("主语言", "")

        stat_items: list[str] = []
        if metadata.get("总 Star"):
            stat_items.append(_render_report_stat("总 Star", metadata["总 Star"], "star"))
        if growth_label and growth_value:
            stat_items.append(_render_report_stat(growth_label, growth_value, "growth"))
        if language:
            stat_items.append(_render_report_stat("主语言", language, "language"))

        created_value = escape(metadata.get("创建时间", "未知"))
        status_value = metadata.get("项目状态", "")
        created_extra = ""
        if status_value:
            created_extra = f' <span class="repo-stat__tag repo-stat__tag--new" title="{escape(status_value)}">NEW</span>'
        stat_items.append(
            '<div class="repo-stat repo-stat--created">'
            '<div class="repo-stat__body">'
            '<span class="repo-stat__label">创建时间</span>'
            f'<strong class="repo-stat__value">{created_value}{created_extra}</strong>'
            '</div>'
            '</div>'
        )

        section_items: list[str] = []
        for section in repo["sections"]:
            section_anchor = f"{anchor}-{_slugify_report_anchor(section['title'])}"
            paragraphs = _split_report_paragraphs(section["content"]) or ["暂无补充信息，可进入仓库查看 README。"]
            paragraphs_html = "".join(f"<p>{escape(paragraph)}</p>" for paragraph in paragraphs)
            section_items.append(
                '<section class="repo-panel">'
                f'<h3 id="{section_anchor}">{escape(section["title"])}</h3>'
                f'{paragraphs_html}'
                '</section>'
            )

        topics_html = ""
        if topic_values:
            tags_html = "".join(f'<span class="repo-topic">{escape(topic)}</span>' for topic in topic_values[:6])
            topics_html = f'<div class="repo-detail__topics">{tags_html}</div>'

        actions_html = (
            '<div class="repo-detail__actions">'
            f'<a class="repo-action" href="{escape(repo_link)}" target="_blank" rel="noreferrer">打开仓库 ↗</a>'
            f'<a class="repo-action repo-action--ghost" href="{escape(readme_link)}" target="_blank" rel="noreferrer">查看 README</a>'
            f'<button type="button" class="repo-action repo-action--ghost repo-trend-btn" data-repo="{escape(repo_name)}">📈 star 走势</button>'
            '</div>'
            '<div class="repo-trend" hidden></div>'
        )

        # 上期对比：上期没有 → 蓝色「上新」；排名变化 → ↑/↓
        is_fresh = prev_ranks is not None and repo_name not in prev_ranks
        delta_html = ""
        if prev_ranks is not None and not is_fresh:
            delta = prev_ranks[repo_name] - repo["rank"]
            if delta > 0:
                delta_html = f'<span class="repo-detail__delta repo-detail__delta--up" title="较上期上升 {delta} 名">↑{delta}</span>'
            elif delta < 0:
                delta_html = f'<span class="repo-detail__delta repo-detail__delta--down" title="较上期下降 {-delta} 名">↓{-delta}</span>'
        fresh_badge = '<span class="repo-detail__fresh" title="上期报告中没有的项目">上新</span>' if is_fresh else ""

        new_badge = '<span class="repo-detail__new" title="新项目">NEW</span>' if status_value else ""
        fresh_attr = ' data-fresh="1"' if is_fresh else ""
        article_parts.append(
            f'<section class="repo-detail" id="{anchor}" data-repo="{escape(repo_name)}" data-rank="{repo["rank"]}"{fresh_attr}>'
            '<header class="repo-detail__head">'
            f'<span class="repo-detail__rank">#{repo["rank"]}</span>'
            f'{delta_html}'
            f'<h2>{escape(repo_name)}</h2>'
            f'{fresh_badge}'
            f'{new_badge}'
            '</header>'
            f'<div class="repo-detail__stats">{"".join(stat_items)}</div>'
            f'{topics_html}'
            f'<div class="repo-detail__grid">{"".join(section_items)}</div>'
            f'{actions_html}'
            '</section>'
        )

        lang_dot = ""
        if language:
            color = _LANG_COLORS.get(language.strip().lower(), "#8b94a7")
            lang_dot = (
                f'<span class="repo-nav__lang"><i style="background:{color}"></i>{escape(language)}</span>'
            )
        growth_chip = f'<span class="repo-nav__growth">{escape(growth_value)}</span>' if growth_value else ""
        new_chip = '<span class="repo-nav__new">NEW</span>' if status_value else ""
        fresh_chip = '<span class="repo-nav__fresh">上新</span>' if is_fresh else ""
        nav_delta = ""
        if delta_html:
            arrow = "↑" if "--up" in delta_html else "↓"
            nav_delta = (
                f'<span class="repo-nav__delta repo-nav__delta--{"up" if arrow == "↑" else "down"}">'
                f'{arrow}{abs(prev_ranks[repo_name] - repo["rank"])}</span>'
            )
        search_blob = " ".join([repo_name, language] + topic_values).lower()
        nav_items.append(
            f'<a class="repo-nav__item" href="#{anchor}" data-panel="{anchor}" '
            f'data-repo="{escape(repo_name)}" data-search="{escape(search_blob)}"{fresh_attr}>'
            f'<span class="repo-nav__rank">{repo["rank"]}</span>'
            '<span class="repo-nav__body">'
            f'<span class="repo-nav__name">{escape(repo_name)}</span>'
            # 徽章放 meta 行行首：项目名过长被截断时徽章仍可见；收藏 ★ 由 report.js 挂载
            f'<span class="repo-nav__meta">{fresh_chip}{new_chip}{growth_chip}{nav_delta}{lang_dot}</span>'
            '</span>'
            '</a>'
        )

    toc_html = (
        f'<nav class="repo-nav" id="repo-nav">{"".join(nav_items)}</nav>'
        if nav_items else '<p class="toc-empty">当前报告暂无可跳转目录。</p>'
    )
    article_html = "".join(article_parts) if article_parts else '<p>当前报告暂无项目内容。</p>'
    return article_html, toc_html


def _render_summary_chips(summary: str, extra_chips: list[str] | None = None) -> str:
    """把「共 N 个项目 | 窗口: 7 天 | …」形式的摘要拆成头部信息条。

    extra_chips: 已渲染好的附加 chip HTML（如上期对比统计）。
    """
    text = (summary or "").strip()
    extra = "".join(extra_chips or [])
    if not text:
        return f'<div class="hero__chips">{extra}</div>' if extra else ""
    parts = [p.strip() for p in text.split("|") if p.strip()]
    if len(parts) <= 1 and not extra:
        return f'<p class="hero__summary">{escape(text)}</p>'
    chips = "".join(f'<span class="hero__chip">{escape(p)}</span>' for p in parts)
    return f'<div class="hero__chips">{chips}{extra}</div>'


def _render_report_html(name: str, markdown_text: str) -> str:
    """将 Markdown 报告渲染为主从布局的 HTML 页面。"""
    lines = markdown_text.splitlines()
    title = next((line[2:].strip() for line in lines if line.startswith("# ")), name)
    summary = next((line[1:].strip() for line in lines if line.startswith(">")), "")
    structured_report = _parse_structured_report(markdown_text)

    if structured_report is not None:
        diff = _build_report_diff(name, structured_report)
        article_html, toc_html = _render_structured_report_html(structured_report, diff)
        extra_chips = []
        if diff:
            prev_date = diff["prev_name"].rsplit(".", 1)[0]
            extra_chips.append(
                '<span class="hero__chip hero__chip--fresh">'
                f'较上期 {escape(prev_date)}: 上新 {diff["added"]} · 移出 {diff["removed"]}'
                '</span>'
            )
        safe_title = escape(structured_report.get("title") or title)
        summary_html = _render_summary_chips(structured_report.get("summary") or summary, extra_chips)
        safe_name = escape(name)
        return _render_web_template(
            REPORT_PAGE_TEMPLATE_PATH,
            {
                "__REPORT_NAME__": safe_name,
                "__REPORT_TITLE__": safe_title,
                "__REPORT_SUMMARY_HTML__": summary_html,
                "__REPORT_TOC_HTML__": toc_html,
                "__REPORT_ARTICLE_HTML__": article_html,
            },
        )

    # 预处理：移除 Markdown 中的原始 HTML 标签，防止 XSS
    sanitized_text = re.sub(r'<(script|iframe|object|embed|form|input|style)[^>]*>.*?</\1>', '', markdown_text, flags=re.DOTALL | re.IGNORECASE)
    sanitized_text = re.sub(r'<(script|iframe|object|embed|form|input|style)[^>]*/?\s*>', '', sanitized_text, flags=re.IGNORECASE)
    sanitized_text = re.sub(r'\bon\w+\s*=', '', sanitized_text, flags=re.IGNORECASE)

    md = markdown.Markdown(
        extensions=["extra", "sane_lists", "toc", "nl2br"],
        output_format="html5",
    )
    article_html = _sanitize_report_html_urls(md.convert(sanitized_text))
    toc_html = _sanitize_report_html_urls(getattr(md, "toc", ""))
    if not toc_html.strip():
        toc_html = '<p class="toc-empty">当前报告暂无可跳转目录。</p>'

    safe_title = escape(title)
    summary_html = _render_summary_chips(summary or "这是一份由服务器根据 Markdown 报告渲染出的可读网页。")
    safe_name = escape(name)
    return _render_web_template(
        REPORT_PAGE_TEMPLATE_PATH,
        {
            "__REPORT_NAME__": safe_name,
            "__REPORT_TITLE__": safe_title,
            "__REPORT_SUMMARY_HTML__": summary_html,
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

# 已确认的恶意扫描 IP（支持通过环境变量 SECURITY_IP_BLACKLIST 配置）
_IP_BLACKLIST: set[str] = set(SECURITY_IP_BLACKLIST)

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

# 配置防护：CORS wildcard 与 credentials 不能同时开启
_cors_allow_credentials = CORS_ALLOW_CREDENTIALS and "*" not in CORS_ALLOWED_ORIGINS
if CORS_ALLOW_CREDENTIALS and "*" in CORS_ALLOWED_ORIGINS:
    logger.warning(
        "检测到 CORS_ALLOW_CREDENTIALS=true 且 allow_origins 包含 '*'，"
        "已自动降级为 allow_credentials=false 以避免高风险配置。"
    )

# CORS 配置（允许前端跨域访问）
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ALLOWED_ORIGINS,
    allow_credentials=_cors_allow_credentials,
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
        "session_ttl_seconds": _SESSION_TTL,
    }


@app.get("/", response_class=FileResponse)
async def index():
    """默认打开移动端聊天页。"""
    return _build_page_response(CHAT_PAGE_PATH, "聊天页面不存在")


@app.get("/chat", response_class=FileResponse)
async def chat_page():
    """渲染并返回移动端聊天页（读取 HTML 模板并注入占位符，非原样静态文件）。"""
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
            reply = agent.chat(req.message, user_id=req.user_id, model=req.model, lite=req.lite)
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
    return ChatResponse(
        session_id=req.session_id,
        reply=reply,
        session_ttl_seconds=_SESSION_TTL,
        session_expires_at=_format_session_expiry(),
    )


@app.get("/api/models")
async def list_models():
    """返回已配置（有 key 且 enabled）的模型清单，供网页模型切换器渲染。

    - models:      主模型列表（config 已过滤未启用条目）。
    - lite_models: 所有平台子模型融合成的共享池；主/子模型选择解耦，
                   任意主模型可搭配任意平台的子模型（按子模型所属平台调用）。
    """
    configured = [m for m in LLM_MODELS if m.get("key")]
    # 跨平台融合子模型池：按模型名去重，同名只保留先出现的平台（前端展示用；
    # 各平台内部的完整 lite_models 不受影响，仍供内部/定时任务按平台回退）。
    lite_pool: list[dict] = []
    seen_names: set[str] = set()
    for m in configured:
        for name in m["lite_models"]:
            if name in seen_names:
                continue
            seen_names.add(name)
            lite_pool.append({"id": f"{m['id']}:{name}", "label": name.rsplit("/", 1)[-1]})
    return {
        "models": [{"id": m["id"], "label": m["label"]} for m in configured],
        "lite_models": lite_pool,
    }


class ModelTestRequest(BaseModel):
    model: str = ""  # 主模型 id；空=不测
    lite: str = ""   # 子模型 id（"平台id:模型名"）；空=不测


@app.post("/api/models/test")
def test_models(req: ModelTestRequest):
    """模型预检：对选中的主/子模型各发一次极小的真实调用，验证可用性。

    前端在切换模型后、主循环启动前调用；返回 unavailable 非空则取消使用并提示。
    同步 def 走线程池，不阻塞事件循环。
    """
    from .infra.llm_client import get_client

    client = get_client()
    logger.info("模型预检开始: model=%s, lite=%s", req.model or "-", req.lite or "-")
    unavailable: list[str] = []
    if req.model:
        ok = client.test_model(model_id=req.model)
        label = next((m["label"] for m in LLM_MODELS if m["id"] == req.model), req.model)
        if ok:
            logger.info("模型预检通过: 主模型 %s (%s)", req.model, label)
        else:
            logger.warning("模型预检不可用: 主模型 %s (%s)", req.model, label)
            unavailable.append(label)
    if req.lite:
        ok = client.test_model(lite_id=req.lite)
        lite_name = req.lite.partition(":")[2] or req.lite
        if ok:
            logger.info("模型预检通过: 子模型 %s", req.lite)
        else:
            logger.warning("模型预检不可用: 子模型 %s", req.lite)
            unavailable.append(lite_name)
    if unavailable:
        logger.warning("模型预检结果: 不可用 %s", unavailable)
    else:
        logger.info("模型预检结果: 全部可用")
    return {"ok": not unavailable, "unavailable": unavailable}


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


@app.delete("/api/reports/{name}")
async def delete_report(name: str):
    """删除指定报告（本地文件）。"""
    # 安全检查：防止路径注入
    if not name.endswith(".md") or "/" in name or ".." in name:
        raise HTTPException(status_code=400, detail="无效的报告名称")

    report_path = os.path.join(REPORT_DIR, name)
    if not os.path.exists(report_path):
        raise HTTPException(status_code=404, detail="报告不存在")

    try:
        os.remove(report_path)
        return {"message": f"报告 {name} 已删除", "deleted": name}
    except OSError as e:
        raise HTTPException(status_code=500, detail=f"删除失败: {e}")


@app.get("/api/star-trend")
async def star_trend_api(repo: str):
    """返回某项目的多周 star 轨迹（供报告卡片的「star 走势」按钮）。"""
    from .tools.tool.star_trend import star_trend
    if not re.match(r"^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$", repo or ""):
        raise HTTPException(status_code=400, detail="无效的仓库名")
    return star_trend(repo)


@app.get("/api/favorite-tags")
async def favorite_tags():
    """收藏分类的预置标签（前端点 ★ 时下方可选，用户仍可自定义）。"""
    return {"tags": list(FAVORITE_DEFAULT_TAGS)}


def _report_appearance_counts() -> tuple[collections.Counter, int]:
    """返回 (每个项目上过多少期定时周报, 周报总期数)；同一期内重复出现只算一次。

    只数无尾缀的 {日期}.md —— 那是 cron 的标准周报。带尾缀的（_NEW 新项目榜、_KEY 关键词榜、
    _10d 自定义窗口）都是按需跑出来的，计入会让「上榜次数」随临时查询虚高。
    总期数与计数走同一次遍历，保证分子分母口径一致（都只认解析成功的周报）。
    走 _load_structured_report_cached（按 mtime 缓存解析结果），故只有新报告需要重解析。
    """
    counts: collections.Counter = collections.Counter()
    total = 0
    for path in glob.glob(os.path.join(REPORT_DIR, "*.md")):
        matched = _REPORT_NAME_RE.match(os.path.basename(path))
        if not matched or matched.group("suffix"):
            continue
        parsed = _load_structured_report_cached(path)
        if parsed:
            total += 1
            counts.update({r["repo"] for r in parsed.get("repos", []) if r.get("repo")})
    return counts, total


@app.get("/api/favorites")
async def list_favorites(user_id: str):
    """获取用户全局收藏清单（附带「上榜期数 / 周报总期数」）。"""
    if not favorites_store.valid_user_id(user_id):
        raise HTTPException(status_code=400, detail="无效的 user_id")
    counts, total = _report_appearance_counts()
    return {
        "user_id": user_id,
        "report_total": total,
        "favorites": [
            dict(x, report_count=counts.get(x.get("repo", ""), 0), report_total=total)
            for x in favorites_store.get_favorites(user_id)
        ],
    }


def _favorite_short_desc(repo: str) -> str:
    """收藏用一句话中文概要：从 DB 的 GitHub 原始描述实时浓缩；无描述则空。

    与 add_favorite 工具同一套逻辑（复用 _make_short_desc），保证网页 ★ 收藏
    与 Agent 收藏得到一致的中文概要。短描述仅收藏展示用，故在收藏时按需生成，
    不在 cron/报告阶段为大量不入收藏的项目预生成。
    """
    from .infra.db import load_db
    from .tools.tool.add_favorite import _make_short_desc
    proj = load_db().get("projects", {}).get(repo, {})
    gh_desc = proj.get("gh_desc", "")
    return _make_short_desc(repo, gh_desc) if gh_desc else ""


@app.post("/api/favorites")
async def update_favorite(req: FavoriteRequest):
    """添加 / 取消收藏（全局，按 user_id 存储）。"""
    from starlette.concurrency import run_in_threadpool
    short_desc = None
    if req.action == "add":
        if req.short_desc is not None:
            short_desc = req.short_desc.strip()[:60]  # 用户手动编辑（含 "" 清空），不再走 LLM
        else:
            short_desc = (await run_in_threadpool(_favorite_short_desc, req.repo)) or None
    try:
        items = favorites_store.set_favorite(
            req.user_id, req.repo, req.action, source_report=req.source_report,
            short_desc=short_desc, category=req.category,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"user_id": req.user_id, "favorites": items}


@app.delete("/api/sessions/{session_id}")
async def delete_session(session_id: str):
    """清除指定会话（释放内存）。"""
    with _sessions_lock:
        if session_id in _sessions:
            del _sessions[session_id]
            with _pending_replies_lock:
                _pending_replies.pop(session_id, None)
            return {"message": f"会话 {session_id} 已清除"}
    raise HTTPException(status_code=404, detail="会话不存在")


# ══════════════════════════════════════════════════════════════
# WebSocket 实时对话
#   执行期间流式推送 进度(progress) + 心跳(heartbeat) + 正文增量(delta)；末尾再以单条
#   reply 整段返回作为权威全文（前端据此做最终渲染、断线待发缓存、会话历史）。
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
    user_id = websocket.query_params.get("user_id", "")
    model = websocket.query_params.get("model", "")
    lite = websocket.query_params.get("lite", "")
    logger.info("WebSocket 已连接: %s (user=%s, model=%s, lite=%s)",
                session_id, user_id or "-", model or "-", lite or "-")

    # 推送断开期间缓存的待发回复（纯文本：前端按非信封消息当最终回复处理）
    with _pending_replies_lock:
        pending = _pending_replies.pop(session_id, [])
    for reply in pending:
        try:
            await websocket.send_text(reply)
            logger.info("WebSocket 推送待发回复: session=%s, reply_len=%s", session_id, len(reply))
        except Exception:
            with _pending_replies_lock:
                _pending_replies.setdefault(session_id, []).append(reply)
            break

    def _chat_with_lock(message: str, progress_cb, delta_cb) -> str:
        agent = get_agent(session_id)
        logger.info("WebSocket 尝试获取执行锁: session=%s", session_id)
        acquired = _tool_execution_lock.acquire(timeout=90)
        if not acquired:
            logger.warning("WebSocket 获取执行锁超时: session=%s", session_id)
            return "系统繁忙，请稍后重试。"
        logger.info("WebSocket 已获取执行锁: session=%s", session_id)
        try:
            return agent.chat(message, progress_cb=progress_cb, user_id=user_id,
                              model=model, lite=lite, delta_cb=delta_cb)
        finally:
            _tool_execution_lock.release()
            logger.info("WebSocket 已释放执行锁: session=%s", session_id)

    try:
        while True:
            data = await websocket.receive_text()
            logger.info("WebSocket 收到消息: session=%s, message=%s", session_id, data[:120])
            reply = await _run_chat_with_progress(websocket, session_id, data, _chat_with_lock)
            if reply is None:
                break  # WS 断开且最终回复已缓存
    except WebSocketDisconnect:
        logger.info(f"WebSocket 断开: {session_id}")


# 执行期间无新进度时的心跳间隔（秒），防止反代/网关掐断空闲连接
_WS_HEARTBEAT_SECONDS = 15
# 进度队列轮询间隔（秒）：也是正文流式增量推送的最小粒度。
# 调细到 50ms 让逐 token 流式更连贯（对话进行时才空转，开销可忽略）。
_WS_POLL_SECONDS = 0.05


async def _run_chat_with_progress(websocket, session_id, message, chat_fn):
    """在线程里跑 agent.chat，同时把进度/心跳实时推给前端，最后发最终回复。

    返回最终回复文本；若 WS 期间断开且回复已缓存到待发缓冲，返回 None。
    """
    progress_queue: _queue.Queue = _queue.Queue()
    _DONE = object()
    holder: dict[str, str] = {}

    def progress_cb(percent: int, label: str) -> None:
        progress_queue.put({"type": "progress", "percent": percent, "label": label})

    def delta_cb(text: str, reset: bool = False) -> None:
        # 最终回答的正文增量：入同一队列，随现有轮询实时推给前端（粒度约 _WS_POLL_SECONDS）。
        # reset=True 表示新一轮正文开始，前端应清掉上一轮流出的过渡正文。
        progress_queue.put({"type": "delta", "text": text, "reset": reset})

    def worker() -> None:
        try:
            holder["reply"] = chat_fn(message, progress_cb, delta_cb)
        except SystemExit:
            holder["reply"] = ("未配置任何 GitHub Token，当前只能预览页面与报告渲染效果。"
                               "请先设置 GITHUB_TOKENS 环境变量后再发起 Agent 对话。")
        except Exception as e:  # noqa: BLE001
            logger.error("WebSocket Agent 执行异常: session=%s, error=%s", session_id, e)
            holder["reply"] = f"处理消息时出现错误：{e}"
        finally:
            progress_queue.put(_DONE)

    worker_task = asyncio.create_task(asyncio.to_thread(worker))
    ws_alive = True
    last_send = time.time()

    while True:
        drained_done = False
        try:
            while True:
                item = progress_queue.get_nowait()
                if item is _DONE:
                    drained_done = True
                    break
                if ws_alive:
                    try:
                        await websocket.send_text(json.dumps(item, ensure_ascii=False))
                        last_send = time.time()
                    except Exception:
                        ws_alive = False  # 前端断开，停止推送，但等 worker 跑完
        except _queue.Empty:
            pass

        if drained_done:
            break

        if ws_alive and time.time() - last_send >= _WS_HEARTBEAT_SECONDS:
            try:
                await websocket.send_text(json.dumps({"type": "heartbeat"}))
                last_send = time.time()
            except Exception:
                ws_alive = False
        await asyncio.sleep(_WS_POLL_SECONDS)

    await worker_task
    reply = holder.get("reply", "")
    logger.info("WebSocket 回复完成: session=%s, reply_len=%s", session_id, len(reply or ""))

    if ws_alive:
        try:
            await websocket.send_text(json.dumps({"type": "reply", "reply": reply}, ensure_ascii=False))
            return reply
        except Exception:
            ws_alive = False

    # WS 已断开：缓存纯文本回复，供重连后推送
    logger.info("WebSocket 发送失败，缓存待发回复: session=%s", session_id)
    with _pending_replies_lock:
        _pending_replies.setdefault(session_id, []).append(reply)
    return None


# ══════════════════════════════════════════════════════════════
# 直接运行支持: python -m hot_projects.api_server
# ══════════════════════════════════════════════════════════════

def main() -> None:
    """统一的 Web/API 服务启动入口。"""
    import uvicorn

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    uvicorn.run(
        "hot_projects.api_server:app",
        host="0.0.0.0",
        port=8001,
        reload=False,
    )


if __name__ == "__main__":
    main()
