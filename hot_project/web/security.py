"""安全中间件 —— IP 黑名单、敏感路径、限速、请求日志。

这台机器暴露在公网上,日志里最多的不是用户,是扫描器:探 `/.env`、探
`/v1/chat/completions`(把它当成一个能白嫖的 OpenAI 代理)、探 `/admin`。

对这些一律回 404 而不是 403 —— 403 等于告诉对方「这里有东西但你没权限」,
而 404 什么都没说。这不是深度防御,是不给对方任何反馈。
"""

from __future__ import annotations

import collections
import logging
import threading
import time

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse

from .. import config

logger = logging.getLogger("hot_project")

# 命中即 404。前缀匹配,小写比较。
BLOCKED_PREFIXES = (
    "/.env", "/.git", "/.well-known/mcp", "/.well-known/agent",
    "/.well-known/ai-plugin", "/v1/models", "/v1/chat/completions",
    "/v1/embeddings", "/api/tags", "/console/api", "/graphql",
    "/debug", "/config", "/_cluster", "/_cat", "/_ml",
    "/admin", "/login", "/swagger", "/internal",
    "/copilot_internal", "/openai/", "/sdapi/",
)

# 滑动窗口限速:每 IP 每分钟这么多次。正常用户一次对话也就十几个请求
# (页面 + 静态资源 + 一次 WS),120 留了很宽的余量。
RATE_WINDOW = 60
RATE_LIMIT = 120

_hits: dict[str, collections.deque] = {}
_lock = threading.Lock()


def client_ip(request) -> str:
    """取真实 IP。走了反代所以要认 X-Forwarded-For 的第一段。"""
    if forwarded := request.headers.get("x-forwarded-for"):
        return forwarded.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


def rate_limited(ip: str) -> bool:
    now = time.time()
    with _lock:
        window = _hits.setdefault(ip, collections.deque())
        while window and window[0] < now - RATE_WINDOW:
            window.popleft()
        if len(window) >= RATE_LIMIT:
            return True
        window.append(now)
        return False


class Guard(BaseHTTPMiddleware):
    async def dispatch(self, request, call_next):
        ip = client_ip(request)

        if ip in config.SECURITY_IP_BLACKLIST:
            logger.warning("黑名单拦截:%s %s", ip, request.url.path)
            return JSONResponse(status_code=403, content={"detail": "Forbidden"})

        path = request.url.path.lower()
        if path.startswith(BLOCKED_PREFIXES):
            logger.warning("敏感路径拦截:%s %s", ip, request.url.path)
            return JSONResponse(status_code=404, content={"detail": "Not Found"})

        if rate_limited(ip):
            logger.warning("限速触发:%s %s", ip, request.url.path)
            return JSONResponse(status_code=429, content={"detail": "Too Many Requests"})

        started = time.time()
        response = await call_next(request)
        logger.info("%s %s %s %.0fms %s", request.method, request.url.path, ip,
                    (time.time() - started) * 1000, response.status_code)
        return response


def cors_options() -> dict:
    """CORS 参数,顺手挡住一个高危组合。

    `allow_origins=["*"]` 配上 `allow_credentials=True` 意味着任何网站都能带着用户的
    cookie 调这里的接口。浏览器其实会拒绝这个组合,但中间件不会报错,配错的人也就
    不会发现 —— 所以这里明确降级并留一行警告。
    """
    credentials = config.CORS_ALLOW_CREDENTIALS
    if credentials and "*" in config.CORS_ALLOWED_ORIGINS:
        logger.warning("CORS 配置里 allow_origins 含 '*' 且开了 credentials,"
                       "已自动降级为 allow_credentials=false。")
        credentials = False
    return {"allow_origins": list(config.CORS_ALLOWED_ORIGINS),
            "allow_credentials": credentials,
            "allow_methods": ["*"], "allow_headers": ["*"]}
