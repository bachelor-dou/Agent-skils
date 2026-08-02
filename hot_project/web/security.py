"""安全中间件 —— IP 黑名单、敏感路径、限速、请求日志。

机器暴露在公网上,日志里最多的是扫描器。敏感路径一律回 404 而不是 403 ——
403 等于告诉对方「这里有东西但你没权限」,404 什么都没说。
"""

from __future__ import annotations

import collections
import logging
import threading
import time
from typing import NamedTuple

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

# 滑动窗口限速:每 IP 每分钟这么多次。
RATE_WINDOW = 60
RATE_LIMIT = 120

_hits: dict[str, collections.deque] = {}
_lock = threading.Lock()

# 表超过这个规模才去扫过期键:正常只有几个 IP,只有被伪造 X-Forwarded-For 灌进来时才需要清理。
_SWEEP_THRESHOLD = 256


def client_ip(request) -> str:
    """取真实 IP。走了反代所以要认 X-Forwarded-For 的第一段。

    这个头可伪造,限速和黑名单都能被换头绕过 —— 有意的取舍(只认 socket 地址的话,架了
    反代后全站流量会算在网关一个 IP 上)。要收紧就加可信代理白名单。
    """
    if forwarded := request.headers.get("x-forwarded-for"):
        return forwarded.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


def rate_limited(ip: str) -> bool:
    """这个 IP 该不该被挡。挡住返回 True,同时**不**计入本次(挡下的请求不续命窗口)。

    键必须按过期删:`X-Forwarded-For` 可伪造,每换一个值就是一个新键,不回收会被撑到几百万条。
    """
    now = time.time()
    with _lock:
        window = _hits.setdefault(ip, collections.deque())
        while window and window[0] < now - RATE_WINDOW:
            window.popleft()
        if len(window) >= RATE_LIMIT:
            return True
        window.append(now)
        # 同时清除整窗都过期的键。判据不能是"deque 空了" —— 伪造 IP 是一次性的,永远不会
        # 回来把自己 popleft 清空;所以看最后一次命中是否已出窗(刚 append 过的当前键不会误删)。
        # ponytail: 每次触发时扫全表,O(n)。n 是"表里现存的键数",正常是个位数;
        # 真要扛住百万级扫描,得换成分桶轮转过期。
        if len(_hits) > _SWEEP_THRESHOLD:
            stale = now - RATE_WINDOW
            for key in [k for k, w in _hits.items() if not w or w[-1] < stale]:
                del _hits[key]
        return False


class Verdict(NamedTuple):
    status: int
    detail: str
    reason: str


def check(ip: str, path: str) -> Verdict | None:
    """三条规则:黑名单 → 敏感路径 → 限速。放行返回 None。

    单独成函数是因为**中间件管不到 WebSocket**(starlette 见到非 http scope 直接转交下一层),
    而 `/ws/chat/{id}` 是唯一能驱动 agent、真会花钱的入口,必须自己调一次。
    """
    if ip in config.SECURITY_IP_BLACKLIST:
        return Verdict(403, "Forbidden", "黑名单")
    if path.lower().startswith(BLOCKED_PREFIXES):
        return Verdict(404, "Not Found", "敏感路径")
    if rate_limited(ip):
        return Verdict(429, "Too Many Requests", "限速")
    return None


class Guard(BaseHTTPMiddleware):
    async def dispatch(self, request, call_next):
        ip = client_ip(request)

        if verdict := check(ip, request.url.path):
            logger.warning("%s拦截:%s %s", verdict.reason, ip, request.url.path)
            return JSONResponse(status_code=verdict.status,
                                content={"detail": verdict.detail})

        started = time.time()
        response = await call_next(request)
        logger.info("%s %s %s %.0fms %s", request.method, request.url.path, ip,
                    (time.time() - started) * 1000, response.status_code)
        return response


def cors_options() -> dict:
    """CORS 参数,同时拦截一个高危组合。

    `allow_origins=["*"]` 配 `allow_credentials=True` = 任何网站都能带用户 cookie 调接口;
    中间件不报错,所以这里明确降级并留一行警告。
    """
    credentials = config.CORS_ALLOW_CREDENTIALS
    if credentials and "*" in config.CORS_ALLOWED_ORIGINS:
        logger.warning("CORS 配置里 allow_origins 含 '*' 且开了 credentials,"
                       "已自动降级为 allow_credentials=false。")
        credentials = False
    return {"allow_origins": list(config.CORS_ALLOWED_ORIGINS),
            "allow_credentials": credentials,
            "allow_methods": ["*"], "allow_headers": ["*"]}
