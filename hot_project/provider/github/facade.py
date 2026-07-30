"""给同步调用方的门面 —— 工具层不必知道 asyncio 的存在。

    gh = GitHub()
    pack = gh.profile("langchain-ai/langchain", want=("info", "readme"))

一次调用 = 一个事件循环 = 一个 httpx 客户端。听起来浪费,但这条路上的调用都是交互式的
(用户在等一个仓库的资料),连接建立那 100ms 淹没在 LLM 的几秒里;而**批量**入口
(`profiles`)一次 `asyncio.run` 抓完全部,该省的地方省到了。

**为什么可以直接 `asyncio.run`。** 整个服务端里工具永远跑在工作线程上,不在事件循环里:
FastAPI 的 `/api/chat` 是有意写成同步 `def`(框架自动丢进线程池),WebSocket 那条路
显式用 `asyncio.to_thread`,cron 脚本本来就是同步的。所以旧包那个「已在循环里就另起
一个线程跑」的 `_run_coroutine_sync` 不需要 —— 那 24 行是为一种不存在的情况准备的。
真有一天要在协程里用,直接 `await` 底下的 async 函数就是了,它们都在。
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from ... import config
from ...infra import tasks
from ...infra.exceptions import RetryableError
from . import client as gh
from . import repo as repo_api
from . import tasks as gh_tasks
from . import trending as trending_api
from .tokens import SEARCH, TokenPool

logger = logging.getLogger("hot_project")

# 关键词榜一次最多几路并发搜。和每日快照那边一个口径:每个 token 有独立的
# 30 次/分钟额度,配速由 token 池管,这里只是别开出比 token 还多的 worker。
SEARCH_WORKERS = 12


class GitHub:
    """同步门面。token 池随实例走,一个进程共享一个就够。"""

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

    def trending(self, period: str = trending_api.DEFAULT_PERIOD) -> list[dict]:
        """Trending 不吃 token,所以没有租约 —— 它抓的是普通网页。"""
        try:
            return self._run(lambda c: trending_api.fetch_trending(c, period))
        except RetryableError as e:
            logger.warning("Trending(%s) 抓取失败:%s", period, e)
            return []

    def keyword_sweep(self, words: list[str], min_star: int) -> dict[str, dict]:
        """按一批关键词并发搜,合并去重。关键词榜的候选来源。

        这是工具层唯一还要真扫一遍 GitHub 的地方:按关键词挑出来的仓库集合无法从快照
        推导 —— 快照只有 star 数,没有「这个仓库和向量数据库有关」这种信息。
        """
        if not words:
            return {}

        async def sweep(client):
            sink = gh_tasks.Discovered()
            async with tasks.TaskPool(
                lanes={gh_tasks.SEARCH_LANE: min(SEARCH_WORKERS, self.pool.capacity)},
                leaser=lambda kind: self.pool.lease(SEARCH),
            ) as pool:
                for word in words:
                    pool.submit(gh_tasks.KeywordPage(sink, client, word, min_star))
                await pool.join()
            if sink.failures:
                logger.warning("关键词搜索有 %d 处失败,结果可能不全。", len(sink.failures))
            return sink.repos

        return self._run(sweep)


_shared: GitHub | None = None


def get() -> GitHub:
    """进程内共享的门面。token 池有状态(冷却、占用、401 计数),必须只有一份。"""
    global _shared
    if _shared is None:
        _shared = GitHub()
    return _shared
