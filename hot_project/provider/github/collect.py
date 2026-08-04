"""GitHub 侧的具体任务类型。

每个类只回答两件事:走哪条道、这一次请求做什么。分页与拆段一律靠**派生**(`ctx.submit`),
不在任务内部循环 —— 一个任务 = 一次请求,理由见 `infra/tasks/task.py`。

关键词搜索、星段扫描、Trending 三个阶段互不依赖,可一次性全提交,最后按 `full_name` 去重。
Trending 不吃 token 也不受 Search 限额,走 free 道。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import httpx

from ...infra.tasks import Ctx, Task
from . import request as gh
from .tokens import Lease
from .trending import fetch_trending

logger = logging.getLogger("hot_project")

SEARCH_LANE = "search"
GRAPHQL_LANE = "graphql"
FREE_LANE = "free"

GRAPHQL_WORKERS = 6

SEARCH_TOKEN = "search"
CORE_TOKEN = "core"

MAX_PAGES = 10
PER_PAGE = 100
SEARCH_CAP = MAX_PAGES * PER_PAGE

KEYWORD_SOURCE = "kw:"
SEGMENT_SOURCE = "segment"
TRENDING_SOURCE = "trending"


# ──────────────────────────────────────────────────────────
# 收集:关键词 / 星段 / Trending
# ──────────────────────────────────────────────────────────


@dataclass
class Discovered:
    """三个阶段共用的收集箱。按 full_name 去重,先到先得。

    `sources` 刻意不与 `repos` 同口径:一个仓库被十个关键词搜到,十个来源里都得有它,
    否则「某关键词抢到几个」只反映并发下谁先跑完。
    """

    repos: dict[str, dict[str, Any]] = field(default_factory=dict)
    failures: list[str] = field(default_factory=list)
    sources: dict[str, set[str]] = field(default_factory=dict)

    def add(self, items: list[dict[str, Any]], source: str = "") -> None:
        names = {n for item in items if (n := item.get("full_name"))}
        if source:
            self.sources.setdefault(source, set()).update(names)
        for item in items:
            name = item.get("full_name")
            if name and name not in self.repos:
                self.repos[name] = item


class _SearchTask(Task):
    """搜索类任务的共同部分:一页、够满就派生下一页。"""

    lane = SEARCH_LANE
    needs_token = True
    token_kind = SEARCH_TOKEN

    query: str
    page: int

    def __init__(self, sink: Discovered, client: httpx.AsyncClient) -> None:
        self.sink = sink
        self.client = client

    async def run(self, ctx: Ctx) -> list[dict[str, Any]]:
        lease: Lease = ctx.token
        items = await gh.search_page(
            self.client, lease, self.query, page=self.page, per_page=PER_PAGE
        )
        if len(items) == PER_PAGE and self.page < MAX_PAGES:
            ctx.submit(self._next_page())
        return items

    def _next_page(self) -> _SearchTask:
        raise NotImplementedError

    @property
    def source(self) -> str:
        """记账用的来源名。翻页要合并到同一个来源下,所以不带页码。"""
        return self.query

    def on_done(self, result: list[dict[str, Any]]) -> None:
        self.sink.add(result, self.source)

    def on_error(self, err: BaseException) -> None:
        self.sink.failures.append(f"{self!r}: {err}")
        logger.warning("%r 最终失败:%s", self, err)


class KeywordPage(_SearchTask):
    """一个关键词的一页。"""

    def __init__(self, sink: Discovered, client: httpx.AsyncClient,
                 word: str, min_star: int, page: int = 1) -> None:
        super().__init__(sink, client)
        self.word, self.min_star, self.page = word, min_star, page
        self.query = f"{word} stars:>={min_star}"

    @property
    def source(self) -> str:
        return f"{KEYWORD_SOURCE}{self.word}"

    def _next_page(self) -> KeywordPage:
        return KeywordPage(self.sink, self.client, self.word, self.min_star, self.page + 1)

    def __repr__(self) -> str:
        return f"KeywordPage({self.word!r}, p{self.page})"


class SegmentPage(_SearchTask):
    """一个 star 区间的一页。"""

    def __init__(self, sink: Discovered, client: httpx.AsyncClient,
                 lo: int, hi: int, page: int = 1) -> None:
        super().__init__(sink, client)
        self.lo, self.hi, self.page = lo, hi, page
        self.query = f"stars:{lo}..{hi}"

    @property
    def source(self) -> str:
        return SEGMENT_SOURCE     # 星段切分是自适应的,合成一个来源才好和关键词比

    def _next_page(self) -> SegmentPage:
        return SegmentPage(self.sink, self.client, self.lo, self.hi, self.page + 1)

    def __repr__(self) -> str:
        return f"SegmentPage({self.lo}..{self.hi}, p{self.page})"


class SegmentProbe(Task):
    """探一个 star 区间有多少条:装得下就直接翻页,装不下就对半劈开再探。

    「装得下」= 命中数 ≤ 1000,即 Search API 一个查询能翻到的极限,超过就只能把区间切细。
    """

    lane = SEARCH_LANE
    needs_token = True
    token_kind = SEARCH_TOKEN

    def __init__(self, sink: Discovered, client: httpx.AsyncClient,
                 lo: int, hi: int) -> None:
        self.sink, self.client = sink, client
        self.lo, self.hi = lo, hi

    async def run(self, ctx: Ctx) -> None:
        count = await gh.search_count(self.client, ctx.token, f"stars:{self.lo}..{self.hi}")
        if count == 0:
            return

        if count <= SEARCH_CAP or self.lo >= self.hi:
            ctx.submit(SegmentPage(self.sink, self.client, self.lo, self.hi))
            return

        mid = (self.lo + self.hi) // 2
        ctx.submit(SegmentProbe(self.sink, self.client, self.lo, mid))
        ctx.submit(SegmentProbe(self.sink, self.client, mid + 1, self.hi))

    def on_error(self, err: BaseException) -> None:
        self.sink.failures.append(f"{self!r}: {err}")
        logger.warning("%r 最终失败,这一段整段漏掉:%s", self, err)

    def __repr__(self) -> str:
        return f"SegmentProbe({self.lo}..{self.hi})"


class TrendingPage(Task):
    """抓一个周期的 Trending 榜。不吃 token,也不受 Search 限额 —— 纯 HTML。"""

    lane = FREE_LANE
    needs_token = False

    def __init__(self, sink: Discovered, client: httpx.AsyncClient, period: str) -> None:
        self.sink, self.client, self.period = sink, client, period

    async def run(self, ctx: Ctx) -> list[dict[str, Any]]:
        return await fetch_trending(self.client, self.period)

    def on_done(self, result: list[dict[str, Any]]) -> None:
        self.sink.add(result, TRENDING_SOURCE)
        logger.info("Trending(%s):%d 个仓库。", self.period, len(result))

    def on_error(self, err: BaseException) -> None:
        self.sink.failures.append(f"{self!r}: {err}")

    def __repr__(self) -> str:
        return f"TrendingPage({self.period})"


# ──────────────────────────────────────────────────────────
# 采集:批量取 star
# ──────────────────────────────────────────────────────────


@dataclass
class Harvest:
    """采集结果。三个集合的区别是淘汰判定的全部依据,不能混。

        stars    成功取到的 star
        missing  采集成功、但 GitHub 明确查不到 —— 删库/改名/转私有,该淘汰
        failed   压根没问到(限流、超时、重试耗尽)—— **绝不能当成 missing**
    """

    stars: dict[str, int] = field(default_factory=dict)
    missing: set[str] = field(default_factory=set)
    failed: set[str] = field(default_factory=set)


class StarBatch(Task):
    """取一批仓库的 star。整批退化成 null 就对半拆开重来。

    拆分靠派生而不是递归调用:递归会让一个 worker 抱着 token 把整棵子树跑完。
    """

    lane = GRAPHQL_LANE
    needs_token = True
    token_kind = CORE_TOKEN

    def __init__(self, sink: Harvest, client: httpx.AsyncClient, names: list[str]) -> None:
        self.sink, self.client, self.names = sink, client, names

    async def run(self, ctx: Ctx) -> dict[str, int] | None:
        stars = await gh.fetch_stars(self.client, ctx.token, self.names)
        if stars is None:
            logger.warning("%d 个仓库整批为 null,拆半重试(疑似查询过大退化)。", len(self.names))
            mid = len(self.names) // 2
            if mid == 0:
                raise RetryableError(f"{self.names} 返回 null 且无法再拆分")
            ctx.submit(StarBatch(self.sink, self.client, self.names[:mid]))
            ctx.submit(StarBatch(self.sink, self.client, self.names[mid:]))
        return stars

    def on_done(self, result: dict[str, int] | None) -> None:
        if result is None:
            return                          # 已拆分,结果由两个子任务记
        self.sink.stars.update(result)
        self.sink.missing.update(set(self.names) - result.keys())

    def on_error(self, err: BaseException) -> None:
        self.sink.failed.update(self.names)
        logger.warning("%d 个仓库采集失败,本次缺席(不计入淘汰):%s", len(self.names), err)

    def __repr__(self) -> str:
        return f"StarBatch({len(self.names)} 个)"


def batches(names: list[str], size: int = gh.BATCH_SIZE) -> list[list[str]]:
    return [names[i:i + size] for i in range(0, len(names), size)]
