"""任务池:分道、并发度、派生、重试、完成判定。

其中两条是**性能**断言,不是行为断言 —— 分道这个设计的全部理由就在它俩身上:

    test_lanes_do_not_block_each_other        搜索和 GraphQL 能真的同时跑
    test_concurrency_equals_worker_count      一条道的并发就是它的 worker 数,不打折

它们要是变红,结构还在、速度没了,而那种退化不会有任何测试自己报警。
"""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager

import pytest

from hot_project.infra.exceptions import RateLimitError, RetryableError
from hot_project.infra.tasks import Task, TaskPool


class Record(Task):
    """跑一下就记一笔,用来观察调度。"""

    lane = "free"

    def __init__(self, name: str, log: list[str], delay: float = 0.0) -> None:
        self.name, self.log, self.delay = name, log, delay
        self.result = None
        self.error = None

    async def run(self, ctx):
        if self.delay:
            await asyncio.sleep(self.delay)
        self.log.append(self.name)
        return self.name

    def on_done(self, result):
        self.result = result

    def on_error(self, err):
        self.error = err

    def __repr__(self) -> str:
        return f"Record({self.name})"


async def test_runs_and_reports_result():
    log: list[str] = []
    task = Record("a", log)

    async with TaskPool({"free": 2}) as pool:
        pool.submit(task)
        await pool.join()

    assert log == ["a"]
    assert task.result == "a"
    assert pool.stats["done"] == 1


async def test_unknown_lane_fails_at_submit():
    """写错道名要在提交处当场炸,不能默默排进某条道再慢慢发现。"""
    class Astray(Record):
        lane = "typo"

    async with TaskPool({"free": 1}) as pool:
        with pytest.raises(KeyError, match="typo"):
            pool.submit(Astray("x", []))


# ──────────────────────────────────────────────────────────
# 并发 —— 分道设计的两条性能断言
# ──────────────────────────────────────────────────────────


class Gate(Task):
    """卡在闸门上,直到测试放行。用来精确观察「同时有几个在跑」。"""

    def __init__(self, lane: str, started: asyncio.Event, release: asyncio.Event,
                 counter: list[int]) -> None:
        self.lane = lane                      # 实例级覆盖类属性,便于一个类测多条道
        self.started, self.release, self.counter = started, release, counter

    async def run(self, ctx):
        self.counter[0] += 1
        self.started.set()
        await self.release.wait()
        self.counter[0] -= 1


async def test_concurrency_equals_worker_count():
    """一条道开 N 个 worker,就该有 N 个任务同时在跑 —— 不多也不少。

    少了说明有隐藏的串行点(整个分道设计就白做了);多了说明并发失控。
    """
    release = asyncio.Event()
    live = [0]
    peak = 0

    async with TaskPool({"free": 4}) as pool:
        for _ in range(10):
            pool.submit(Gate("free", asyncio.Event(), release, live))
        for _ in range(20):                   # 让 worker 有机会全部起来
            await asyncio.sleep(0)
            peak = max(peak, live[0])
        release.set()
        await pool.join()

    assert peak == 4, f"并发是 {peak},不等于 worker 数 4"


async def test_lanes_do_not_block_each_other():
    """一条道被占满,另一条道必须照常跑。

    这是分道存在的理由。单队列 + 信号量的版本会在这里挂掉:worker 从同一条队列取到
    上限已满的那类任务后堵住,别的类型就没人做了。而实测里搜索(受限额)和 GraphQL
    (2026-07-30 那轮 779 批一次限流都没有)本来就该同时跑 —— 差着十分钟。
    """
    release = asyncio.Event()
    live = [0]
    log: list[str] = []

    async with TaskPool({"slow": 2, "fast": 2}) as pool:
        for _ in range(6):                    # 远超 slow 道的 2 个 worker
            pool.submit(Gate("slow", asyncio.Event(), release, live))

        quick = Record("quick", log)
        quick.lane = "fast"
        pool.submit(quick)

        await asyncio.wait_for(_until(lambda: log == ["quick"]), timeout=2)

        release.set()
        await pool.join()

    assert quick.result == "quick"


async def _until(pred, step: float = 0.005) -> None:
    while not pred():
        await asyncio.sleep(step)


# ──────────────────────────────────────────────────────────
# 派生
# ──────────────────────────────────────────────────────────


class Chain(Task):
    """跑起来再生一个后继,直到深度耗尽 —— 模拟分页与星段拆分。"""

    lane = "free"

    def __init__(self, depth: int, log: list[str]) -> None:
        self.depth, self.log = depth, log

    async def run(self, ctx):
        self.log.append(f"d{self.depth}")
        if self.depth > 0:
            ctx.submit(Chain(self.depth - 1, self.log))


async def test_join_waits_for_derived_tasks():
    """`join()` 不能在父任务完成时就返回 —— 它派生的子任务还没跑。

    「所有队列都空了」这个判据在这里正好失效:父任务执行中队列确实是空的,
    而它下一行就要塞进一个子任务。旧包的星段拆分正是这个形状。
    """
    log: list[str] = []

    async with TaskPool({"free": 1}) as pool:
        pool.submit(Chain(3, log))
        await pool.join()

    assert log == ["d3", "d2", "d1", "d0"]


async def test_derived_tasks_can_cross_lanes():
    """发现阶段的任务要能往采集道里塞活 —— 否则又得回到「按阶段串行」。"""
    log: list[str] = []

    class Spawner(Task):
        lane = "a"

        async def run(self, ctx):
            follower = Record("crossed", log)
            follower.lane = "b"
            ctx.submit(follower)

    async with TaskPool({"a": 1, "b": 1}) as pool:
        pool.submit(Spawner())
        await pool.join()

    assert log == ["crossed"]


# ──────────────────────────────────────────────────────────
# 失败与重试
# ──────────────────────────────────────────────────────────


class Flaky(Task):
    lane = "free"
    max_retries = 3

    def __init__(self, fails: int, exc: BaseException) -> None:
        self.fails, self.exc = fails, exc
        self.runs = 0
        self.result = None
        self.error = None

    async def run(self, ctx):
        self.runs += 1
        if self.runs <= self.fails:
            raise self.exc
        return "finally ok"

    def on_done(self, result):
        self.result = result

    def on_error(self, err):
        self.error = err


async def test_transient_failure_is_retried():
    task = Flaky(2, RetryableError("网络抖了一下"))

    async with TaskPool({"free": 1}) as pool:
        pool.submit(task)
        await pool.join()

    assert task.runs == 3 and task.result == "finally ok"


async def test_retries_are_capped():
    """网络长期不通时必须收尾,不能无限自旋。"""
    task = Flaky(99, RetryableError("一直不通"))

    async with TaskPool({"free": 1}) as pool:
        pool.submit(task)
        await pool.join()

    assert task.runs == task.max_retries + 1
    assert isinstance(task.error, RetryableError)
    assert pool.stats["failed"] == 1


class Scripted(Task):
    """按剧本依次抛出指定异常,剧本走完就成功。"""

    lane = "free"
    max_retries = 3

    def __init__(self, script: list[BaseException]) -> None:
        self.script = list(script)
        self.runs = 0
        self.result = None
        self.error = None

    async def run(self, ctx):
        self.runs += 1
        if self.script:
            raise self.script.pop(0)
        return "finally ok"

    def on_done(self, result):
        self.result = result

    def on_error(self, err):
        self.error = err


async def test_rate_limit_does_not_consume_the_retry_budget():
    """限流重排**不计**重试次数,而且不能悄悄累积。

    限流是外部节流,不是这个任务有问题。要是计入,一轮限流高峰就能把所有任务的重试额度
    烧光,接着它们在真正的瞬时故障面前一次都不重试 —— 而限流高峰恰恰是网络最不稳的时候。

    所以剧本是「先连撞 5 次限流(已超 max_retries=3),再来 2 次瞬时故障」:
    额度没被限流吃掉的话,这 2 次仍在预算内,任务最终成功。
    """
    task = Scripted(
        [RateLimitError(reset_at=0.0)] * 5 + [RetryableError("抖了一下")] * 2
    )

    async with TaskPool({"free": 1}) as pool:
        pool.submit(task)
        await pool.join()

    assert task.result == "finally ok", f"限流吃掉了重试额度:{task.error!r}"
    assert task.runs == 8


async def test_programming_error_fails_immediately():
    """bug 重排一万次还是 bug,只会刷屏。"""
    task = Flaky(99, ValueError("拼错了字段名"))

    async with TaskPool({"free": 1}) as pool:
        pool.submit(task)
        await pool.join()

    assert task.runs == 1 and isinstance(task.error, ValueError)


async def test_callback_blowing_up_does_not_hang_join():
    """回调抛异常必须照常减计数,否则 `join()` 永远不返回 —— 整轮卡死。"""
    class BadCallback(Record):
        def on_done(self, result):
            raise RuntimeError("回调自己炸了")

    async with TaskPool({"free": 1}) as pool:
        pool.submit(BadCallback("x", []))
        await asyncio.wait_for(pool.join(), timeout=2)


# ──────────────────────────────────────────────────────────
# 租约注入 —— 本包不认识 token 池
# ──────────────────────────────────────────────────────────


async def test_token_is_leased_per_task_and_returned():
    """要 token 的任务由 worker 借好递进去,任务自己不碰池子。

    每个任务各借各还(而不是一个任务攥着 token 跑完多页):旧包 KeywordSearch 中位持有
    4.42 秒,大半在等配速,别的任务只能干看着。
    """
    live = [0]
    peak = [0]

    @asynccontextmanager
    async def leaser(kind: str):
        live[0] += 1
        peak[0] = max(peak[0], live[0])
        try:
            yield f"lease:{kind}"
        finally:
            live[0] -= 1

    seen: list[str] = []

    class NeedsToken(Task):
        lane = "free"
        needs_token = True
        token_kind = "search"

        async def run(self, ctx):
            seen.append(ctx.token)

    async with TaskPool({"free": 2}, leaser=leaser) as pool:
        for _ in range(4):
            pool.submit(NeedsToken())
        await pool.join()

    assert seen == ["lease:search"] * 4
    assert live[0] == 0, "有租约没还"
    assert peak[0] <= 2, "借出的租约超过了 worker 数"


async def test_task_without_token_never_touches_the_leaser():
    """不吃 token 的任务(Trending 抓 HTML、本地计算)不该白占一张租约。"""
    calls = []

    @asynccontextmanager
    async def leaser(kind: str):
        calls.append(kind)
        yield None

    async with TaskPool({"free": 1}, leaser=leaser) as pool:
        pool.submit(Record("free-of-charge", []))
        await pool.join()

    assert calls == []


async def test_needing_a_token_without_a_leaser_is_an_error():
    class NeedsToken(Record):
        needs_token = True

    task = NeedsToken("x", [])
    async with TaskPool({"free": 1}) as pool:
        pool.submit(task)
        await pool.join()

    assert isinstance(task.error, RuntimeError)
