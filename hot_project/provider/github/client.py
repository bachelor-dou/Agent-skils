"""同步的 GitHub 客户端 —— 本包对外的唯一入口,工具层不必知道 asyncio 的存在。

    gh = GitHub()
    pack = gh.profile("langchain-ai/langchain", want=("info", "readme"))

本模块在栈顶:它组合 `request`(出站原语)、`repo`、`tasks`、`trending`、`tokens`。所以别把
它和 `request` 合成一个模块 —— `repo` 和 `tasks` 都 import `request`,合并即循环 import。

一次调用 = 一个事件循环 = 一个 httpx 客户端。可以直接 `asyncio.run`,是因为工具永远跑在
工作线程上而不在循环里(`/api/chat` 是同步 `def`,WebSocket 走 `asyncio.to_thread`,
cron 本来就是同步的)。真要在协程里用,直接 await 底下的 async 函数。
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from ... import config
from ...infra import tasks
from ...infra.exceptions import RetryableError
from . import repo as repo_api
from . import request as gh
from . import collect
from . import trending as trending_api
from .tokens import CORE, SEARCH, TokenPool

logger = logging.getLogger("hot_project")

SEARCH_WORKERS = 12

_PACES = {collect.SEARCH_TOKEN: SEARCH, collect.CORE_TOKEN: CORE}


class GitHub:
    """同步客户端。token 池随实例走,一个进程共享一个就够。"""

    def __init__(self, pool: TokenPool | None = None) -> None:
        self.pool = pool or TokenPool(config.github_tokens())

    @property
    def usable(self) -> bool:
        """一个 token 都没有时,调用方该退化而不是抛 —— 本地跑没配 token 是常见情况。"""
        return self.pool.capacity > 0

    def _run(self, make_coro):
        """开一个循环、一个客户端,跑完关掉。"""
        async def main():
            client = gh.build_client()
            try:
                return await make_coro(client)
            finally:
                await client.aclose()

        return asyncio.run(main())

    def info(self, name: str) -> dict | None:
        return self._run(lambda c: repo_api.info(c, self.pool, name))

    def profile(self, name: str, want: tuple[str, ...] = repo_api.ALL) -> dict[str, Any]:
        return self._run(lambda c: repo_api.profile(c, self.pool, name, want))

    def profiles(self, names: list[str],
                 want: tuple[str, ...] = repo_api.ALL) -> dict[str, dict]:
        """一次循环抓完一批。报告生成用它,比逐个抓快一个数量级。"""
        if not names:
            return {}
        return self._run(lambda c: repo_api.profiles(c, self.pool, names, want))

    def search(self, query: str, *, limit: int = 5, sort: str = "stars") -> list[dict]:
        return self._run(lambda c: repo_api.search(c, self.pool, query,
                                                   limit=limit, sort=sort))

    def stars(self, names: list[str]) -> collect.Harvest:
        """批量取**当前** star。榜单的被减数走这里,和每日采集是同一条路。

        `Harvest` 把「取到 / 确认查不到 / 这次没问到」分开,后两者不能混:一次限流高峰就能
        让上万个活仓库集体从榜上消失。
        """
        if not names:
            return collect.Harvest()
        if self.pool.capacity < 1:
            logger.error("所有 token 都已失效,取不到当前 star。")
            return collect.Harvest()

        async def harvest(client):
            sink = collect.Harvest()
            async with tasks.TaskPool(
                lanes={collect.GRAPHQL_LANE: collect.GRAPHQL_WORKERS},
                leaser=lambda kind: self.pool.lease(_PACES[kind]),
            ) as pool:
                for group in collect.batches(names):
                    pool.submit(collect.StarBatch(sink, client, group))
                await pool.join()
            return sink

        return self._run(harvest)

    def trending(self, period: str = trending_api.DEFAULT_PERIOD) -> list[dict]:
        """Trending 不吃 token,所以没有租约 —— 它抓的是普通网页。"""
        try:
            return self._run(lambda c: trending_api.fetch_trending(c, period))
        except RetryableError as e:
            logger.warning("Trending(%s) 抓取失败:%s", period, e)
            return []

    def keyword_sweep(self, words: list[str], min_star: int) -> dict[str, dict]:
        """按一批关键词并发搜,合并去重。关键词榜的候选来源。

        工具层唯一还要真扫 GitHub 的地方 —— 快照只有 star 数,推不出"这个仓库和向量库有关"。
        """
        if not words:
            return {}

        if self.pool.capacity < 1:
            logger.error("所有 token 都已失效,关键词搜索无法进行。")
            return {}

        async def sweep(client):
            sink = collect.Discovered()
            async with tasks.TaskPool(
                lanes={collect.SEARCH_LANE: min(SEARCH_WORKERS, self.pool.capacity)},
                leaser=lambda kind: self.pool.lease(_PACES[kind]),
            ) as pool:
                for word in words:
                    pool.submit(collect.KeywordPage(sink, client, word, min_star))
                await pool.join()
            if sink.failures:
                logger.warning("关键词搜索有 %d 处失败,结果可能不全。", len(sink.failures))
            return sink.repos

        return self._run(sweep)


_shared: GitHub | None = None


def shared() -> GitHub:
    """进程内共享的那一份。token 池有状态(冷却、占用、401 计数),必须只有一份。"""
    global _shared
    if _shared is None:
        _shared = GitHub()
    return _shared
