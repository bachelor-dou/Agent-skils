"""GitHub 出站请求 —— 发一次,把结果或失败原因还回去。**不重试。**

重试是任务池的事。旧包在三个层次上各重试一遍:`api.py` 内部 3 次循环、dispatcher 命中
异常后重排、外面再套 3 轮「页级补偿」。三套叠起来的后果是一次限流最多能放大成 27 次请求,
而限流恰恰是「请求太多」引起的 —— 越撞越退不出来。这里只发一次。

每个函数都要一张租约。租约从哪来是调用方的事(任务池的 worker 借好递进来),
本模块不认识 token 池,也不知道池里有几个 token。
"""

from __future__ import annotations

import json
import logging
import time
from typing import Any

import httpx

from ...infra.exceptions import RateLimitError, RetryableError, TokenInvalidError
from .tokens import Lease

logger = logging.getLogger("hot_project")

SEARCH_URL = "https://api.github.com/search/repositories"
GRAPHQL_URL = "https://api.github.com/graphql"

# 一次 GraphQL 查询塞多少个仓库别名。实测 100 别名 = 1 个 GraphQL 点,
# 7.8 万仓库全量约 780 点,而 12 个 token 每小时共 6 万点 —— 配额是零头。
#
# **勿上调。** 实测 200 别名会 HTTP 200 + 无 errors + 全字段 null:看起来成功,
# 拿回来的却是一批空值。要是当成「这批仓库都没了」,一次退化就能抹掉整批基线。
BATCH_SIZE = 100

# 限流响应没给 reset 时刻时,默认冷却多久。
DEFAULT_COOLDOWN = 60.0


def build_client(timeout: float = 90.0) -> httpx.AsyncClient:
    """建一个可跨请求复用的连接池。

    复用很要紧:全量快照约 780 次 POST,每次重建连接要多付一次 TLS 握手。
    """
    return httpx.AsyncClient(
        timeout=timeout,
        limits=httpx.Limits(max_connections=100, max_keepalive_connections=100),
    )


def _reset_at(headers) -> float:
    """从响应头推算限流何时解除。"""
    for key, absolute in (("X-RateLimit-Reset", True), ("Retry-After", False)):
        raw = headers.get(key)
        if not raw:
            continue
        try:
            value = float(raw)
        except (TypeError, ValueError):
            continue
        when = value if absolute else time.time() + value
        if when > time.time():
            return when
    return time.time() + DEFAULT_COOLDOWN


def _classify(resp: httpx.Response) -> None:
    """HTTP 状态 → 异常。**全项目只有这一处做这个翻译。**

    分类决定了任务池怎么处置(见 `infra/tasks/pool.py`),所以它必须只有一个版本:
    401 记 strike、403/429 冷却 token、5xx 退避重试、4xx 当场失败不重排。
    """
    code = resp.status_code
    if code == 200:
        return
    if code == 401:
        raise TokenInvalidError(f"HTTP 401: {resp.text[:200]}")
    if code in (403, 429):
        raise RateLimitError(_reset_at(resp.headers))
    if code >= 500:
        raise RetryableError(f"HTTP {code}: {resp.text[:200]}")
    raise RuntimeError(f"HTTP {code}: {resp.text[:200]}")      # 4xx:请求本身有问题,重试无用


async def _send(coro) -> httpx.Response:
    """把网络层异常翻译成 `RetryableError`,让任务池按瞬时故障处理。"""
    try:
        return await coro
    except httpx.HTTPError as e:
        raise RetryableError(f"{type(e).__name__}: {e}") from e


async def search_page(
    client: httpx.AsyncClient,
    lease: Lease,
    query: str,
    *,
    page: int = 1,
    per_page: int = 100,
    sort: str = "stars",
    order: str = "desc",
) -> list[dict[str, Any]]:
    """搜一页。返回这一页的原始条目;空列表表示没有更多了。

    `query` 是**完整的** Search 语法串(含 `stars:>=N` 之类的限定),本函数不替调用方拼 ——
    旧包在这里偷偷追加 `stars:>=MIN_STAR`,于是调用方看到的查询和真正发出去的不是一回事。
    """
    resp = await _send(client.get(
        SEARCH_URL,
        headers=lease.rest_headers,
        params={"q": query, "sort": sort, "order": order,
                "per_page": per_page, "page": page},
    ))
    if resp.status_code == 422:
        # 分页越界(Search 只给前 1000 条)或查询语法不合法。两者都不该重试。
        logger.debug("搜索 422,当作没有更多:q=%r page=%d", query, page)
        return []
    _classify(resp)
    return resp.json().get("items", [])


async def search_count(client: httpx.AsyncClient, lease: Lease, query: str) -> int:
    """只问命中多少条。用于星段拆分:超过 1000 就得把区间切开。"""
    resp = await _send(client.get(
        SEARCH_URL, headers=lease.rest_headers, params={"q": query, "per_page": 1},
    ))
    if resp.status_code == 422:
        return 0
    _classify(resp)
    return int(resp.json().get("total_count", 0))


def _star_query(names: list[str]) -> str:
    """把一批 owner/repo 拼成别名查询。

    owner 与 name 用 `json.dumps` 转义(仓库名里出现引号会把查询拼坏),
    别名用序号而不是仓库名 —— GraphQL 别名不允许出现 `-`、`.` 这些字符。
    """
    parts = []
    for i, full_name in enumerate(names):
        owner, _, repo = full_name.partition("/")
        parts.append(
            f"r{i}: repository(owner:{json.dumps(owner)}, name:{json.dumps(repo)})"
            " { stargazerCount }"
        )
    return "query{" + "\n".join(parts) + "}"


async def fetch_stars(
    client: httpx.AsyncClient, lease: Lease, names: list[str]
) -> dict[str, int] | None:
    """批量取 star。

    返回 dict —— **键缺失意味着 GitHub 确认查不到那个仓库**(已删除/改名/转私有)。
    这个语义是淘汰判定的依据,所以不能把「我们没问到」混进来。

    返回 None —— 整批全 null 的退化响应(见 `BATCH_SIZE` 上方)。调用方应对半拆开重来,
    **绝不可当成「这批仓库都没了」**:真实的删除是零星的,不会整批同时发生。
    """
    resp = await _send(client.post(
        GRAPHQL_URL, headers=lease.graphql_headers, json={"query": _star_query(names)},
    ))
    _classify(resp)

    payload = resp.json()
    data = payload.get("data")
    if not isinstance(data, dict):
        # GraphQL 的限流不走 HTTP 403,而是 200 + errors 里写 RATE_LIMITED。
        errors = str(payload.get("errors", ""))[:200]
        if "RATE_LIMITED" in errors:
            raise RateLimitError(time.time() + DEFAULT_COOLDOWN)
        raise RetryableError(f"GraphQL 响应没有 data:{errors}")

    stars: dict[str, int] = {}
    for i, full_name in enumerate(names):
        node = data.get(f"r{i}")
        if isinstance(node, dict) and isinstance(node.get("stargazerCount"), int):
            stars[full_name] = node["stargazerCount"]

    # errors 和 data 可以并存:个别仓库 NOT_FOUND 是常态,不能因此丢掉整批。
    if not stars and len(names) > 1:
        return None
    return stars
