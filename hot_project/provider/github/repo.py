"""单仓库资料 —— REST `/repos/{owner}/{repo}` 那几个端点。

对外只有一个动作:`profile(name, want=...)` 给一个仓库的资料包。

**为什么是「资料包」而不是四个函数。** 旧包里 describe_project、报告生成、repo_profile
各自抄了一遍「抓 info、抓 readme、抓 commits、拼进一个 dict」,三份抄得还不一样:
describe 用 ThreadPoolExecutor 并发抓、报告生成串行抓且只抓两样、repo_profile 又是另一套。
真正会变的是「这次要哪几样」,那就让它成为参数。

**限流轮换不在这里写。** 每次请求借一张租约,撞限流时租约在归还时冷却那个 token,
重试自然会拿到另一个 —— 旧包那个 80 行的 `_with_token_rotation` 整个不需要了。
"""

from __future__ import annotations

import asyncio
import base64
import binascii
import logging
from typing import Any

import httpx

from ...infra.exceptions import RateLimitError, RetryableError, TokenInvalidError
from . import client as gh
from .tokens import CORE, SEARCH, TokenPool

logger = logging.getLogger("hot_project")

API = "https://api.github.com"

README_MAX_CHARS = 4000
RELEASES = 5
COMMITS = 10
COMMIT_MESSAGE_MAX = 140

# 一次请求最多重试几次。单仓库抓取是交互路径(用户在等),不像每日快照那样可以慢慢磨,
# 所以比任务池的预算小。
ATTEMPTS = 3

ALL = ("info", "readme", "releases", "commits")


async def _get(client: httpx.AsyncClient, pool: TokenPool, path: str,
               *, params: dict | None = None) -> Any | None:
    """请求一个仓库端点。404/422 → None(没有这个东西,不是故障)。

    每次尝试重新借租约:上一次撞限流的那个 token 已经在归还时被冷却了,
    重借拿到的必然是另一张。
    """
    url = f"{API}{path}"
    for attempt in range(ATTEMPTS):
        try:
            async with pool.lease(CORE) as lease:
                resp = await client.get(url, headers=lease.rest_headers, params=params)
                if resp.status_code == 200:
                    try:
                        return resp.json()
                    except ValueError:
                        logger.warning("仓库端点返回的不是 JSON:%s", url)
                        return None
                if resp.status_code in (404, 422):
                    return None
                if resp.status_code == 401:
                    raise TokenInvalidError(f"HTTP 401: {resp.text[:200]}")
                if resp.status_code in (403, 429):
                    raise RateLimitError(gh._reset_at(resp.headers))
                raise RetryableError(f"HTTP {resp.status_code}: {resp.text[:200]}")
        except (RateLimitError, RetryableError, httpx.RequestError) as e:
            if attempt == ATTEMPTS - 1:
                logger.warning("仓库端点放弃:%s(%s)", url, e)
                return None
    return None


async def info(client, pool, name: str) -> dict | None:
    """仓库元信息。返回的结构和 Search API 的 item 兼容。"""
    return await _get(client, pool, f"/repos/{name}")


async def readme(client, pool, name: str, max_chars: int = README_MAX_CHARS) -> dict:
    """README 摘录。没有 README、解不开 base64 → 空 dict。"""
    data = await _get(client, pool, f"/repos/{name}/readme")
    if not isinstance(data, dict):
        return {}
    raw = data.get("content") or ""
    if not isinstance(raw, str) or not raw:
        return {}
    if str(data.get("encoding", "")).lower() == "base64":
        try:
            text = base64.b64decode(raw.replace("\n", ""), validate=False).decode(
                "utf-8", errors="ignore")
        except (ValueError, binascii.Error):
            return {}
    else:
        text = raw
    text = text.strip()
    if not text:
        return {}
    return {"text": text[:max_chars], "truncated": len(text) > max_chars,
            "sha": data.get("sha", ""), "path": data.get("path", "")}


async def releases(client, pool, name: str, limit: int = RELEASES) -> list[dict]:
    data = await _get(client, pool, f"/repos/{name}/releases",
                      params={"per_page": limit, "page": 1})
    if not isinstance(data, list):
        return []
    return [
        {"tag_name": it.get("tag_name", ""), "name": it.get("name", ""),
         "published_at": it.get("published_at", ""),
         "prerelease": bool(it.get("prerelease")), "draft": bool(it.get("draft"))}
        for it in data if isinstance(it, dict)
    ]


async def commits(client, pool, name: str, limit: int = COMMITS) -> list[dict]:
    data = await _get(client, pool, f"/repos/{name}/commits",
                      params={"per_page": limit, "page": 1})
    if not isinstance(data, list):
        return []
    out: list[dict] = []
    for it in data:
        detail = it.get("commit") if isinstance(it, dict) else None
        if not isinstance(detail, dict):
            continue
        author = detail.get("author") if isinstance(detail.get("author"), dict) else {}
        first_line = str(detail.get("message") or "").strip().splitlines()[:1]
        out.append({"sha": it.get("sha", ""), "date": author.get("date", ""),
                    "message": (first_line[0] if first_line else "")[:COMMIT_MESSAGE_MAX]})
    return out


_FETCHERS = {"info": info, "readme": readme, "releases": releases, "commits": commits}


async def profile(client, pool, name: str, want: tuple[str, ...] = ALL) -> dict[str, Any]:
    """一个仓库的资料包。要哪几样由 `want` 决定,几样并发抓。

    某一样失败只让那一样缺席,不拖垮整包 —— 一个仓库没有 releases 是常态,
    不该让描述生成整个失败。
    """
    keys = [k for k in want if k in _FETCHERS]
    results = await asyncio.gather(
        *(_FETCHERS[k](client, pool, name) for k in keys), return_exceptions=True)
    out: dict[str, Any] = {}
    for key, value in zip(keys, results):
        if isinstance(value, BaseException):
            logger.warning("抓 %s 的 %s 失败:%s", name, key, value)
            continue
        if value:
            out[key] = value
    return out


async def profiles(client, pool, names: list[str],
                   want: tuple[str, ...] = ALL) -> dict[str, dict]:
    """一批仓库的资料包。并发度由 token 池的容量自然限制(借不到租约就等)。"""
    packs = await asyncio.gather(*(profile(client, pool, n, want) for n in names))
    return dict(zip(names, packs))


async def search(client, pool, query: str, *, limit: int = 5,
                 sort: str = "stars") -> list[dict]:
    """按关键词搜仓库,取前 `limit` 个。用于仓库名消歧和 `search_repos` 工具。

    走 SEARCH 配速而不是 CORE:搜索是另一本限额账(30 次/分),混用会撞墙。
    """
    async with pool.lease(SEARCH) as lease:
        return (await gh.search_page(client, lease, query, page=1,
                                     per_page=limit, sort=sort))[:limit]
