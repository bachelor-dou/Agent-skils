"""token 池:租约归还、限流冷却、401 strikes、配速、动态容量。

全部用可控时钟(`FakeClock`),所以没有一条测试会真的 sleep —— 401 冷却是 60 秒、
限流冷却按 GitHub 的 reset 时刻算,真等就没法跑了。
"""

from __future__ import annotations

import asyncio
import threading
import time

import pytest

from hot_project.infra.exceptions import RateLimitError, TokenInvalidError
from hot_project.provider.github.tokens import (
    CORE,
    SEARCH,
    AllTokensInvalid,
    Pace,
    TokenPool,
)


class FakeClock:
    """手动推进的时钟。池只从 `time_fn` 取时间,所以这里说几点就是几点。"""

    def __init__(self) -> None:
        self.t = 1_000_000.0

    def __call__(self) -> float:
        return self.t

    def advance(self, seconds: float) -> None:
        self.t += seconds


@pytest.fixture
def clock():
    return FakeClock()


def make_pool(clock, count=3, **kw) -> TokenPool:
    return TokenPool([f"tok{i}" for i in range(count)], time_fn=clock, **kw)


def _index_of(lease) -> int:
    return lease._index


# ──────────────────────────────────────────────────────────
# 1. 租约自动归还
# ──────────────────────────────────────────────────────────


async def test_lease_returns_token_on_success(clock):
    pool = make_pool(clock, count=1)

    async with pool.lease() as lease:
        assert "Authorization" in lease.rest_headers

    async with pool.lease():        # 只有一个 token,借得到就说明上一次归还了
        pass


async def test_unrelated_exception_returns_token_without_burning_it(clock):
    """网络抖动不是 token 的错:必须归还,且不能标记失效或冷却。"""
    pool = make_pool(clock, count=1)

    with pytest.raises(RuntimeError, match="网络炸了"):
        async with pool.lease():
            raise RuntimeError("网络炸了")

    assert pool.capacity == 1
    async with pool.lease():        # 立刻还能借到 = 没被冷却
        pass


async def test_exception_propagates(clock):
    """池不吞异常:重试与否是任务池的事。"""
    pool = make_pool(clock, count=1)

    with pytest.raises(TokenInvalidError):
        async with pool.lease():
            raise TokenInvalidError("401")


# ──────────────────────────────────────────────────────────
# 2. 限流冷却
# ──────────────────────────────────────────────────────────


async def test_rate_limited_token_is_skipped_until_reset(clock):
    pool = make_pool(clock, count=2, recovery_buffer=3.0)

    with pytest.raises(RateLimitError):
        async with pool.lease() as lease:
            burned = _index_of(lease)
            raise RateLimitError(reset_at=clock.t + 100)

    for _ in range(4):
        async with pool.lease() as lease:
            assert _index_of(lease) != burned

    clock.advance(100 + 3.0)        # reset + recovery_buffer
    seen = set()
    for _ in range(6):
        async with pool.lease() as lease:
            seen.add(_index_of(lease))
    assert burned in seen, "冷却到期后没有回归调度"


async def test_lease_waits_instead_of_failing_when_all_cooling(clock):
    """全部在冷却时 `lease()` 要等,不能报错 —— 限流是常态。"""
    pool = make_pool(clock, count=1, recovery_buffer=0.0)

    with pytest.raises(RateLimitError):
        async with pool.lease():
            raise RateLimitError(reset_at=clock.t + 30)

    waiter = asyncio.create_task(_take_one(pool))
    await asyncio.sleep(0)
    assert not waiter.done(), "应当在等待而不是立刻拿到或报错"

    await _skip_ahead(pool, clock, 30)
    assert await asyncio.wait_for(waiter, timeout=2) == 0


async def test_waiter_wakes_itself_on_real_clock():
    """没有任何人 notify 时,等待者必须靠自己的定时器到点醒来。

    这条用**真**时钟(冷却 50 毫秒):假时钟测不到它 —— 假时钟下是测试在 notify,
    真正的定时器分支反而被绕过了。少了这条,把 `wait_for` 的 timeout 去掉也能全绿,
    而线上的表现是所有 worker 在第一次限流时永久挂死。
    """
    pool = TokenPool(["only"], recovery_buffer=0.0)

    with pytest.raises(RateLimitError):
        async with pool.lease():
            raise RateLimitError(reset_at=time.time() + 0.05)

    async with pool.lease() as lease:      # 无人唤醒,只能靠定时器
        assert _index_of(lease) == 0


def test_a_pool_survives_being_used_by_a_second_event_loop():
    """`client` 是同步客户端,每个方法各起一个 `asyncio.run` —— 池必须扛得住换循环。

    `asyncio.Condition` 绑死在第一个 await 它的循环上,于是第二次调用只要有人**真的**
    等锁就 `RuntimeError: bound to a different event loop`,而且池从此报废。所以这里
    刻意让 worker 数超过 token 数,逼出 `_cond.wait()`;不这么写两次都会"成功"。

    刻意不加 async 标记:这条测的就是「起了两个循环」,不能待在测试自己那个循环里。
    """
    pool = TokenPool(["a", "b"])

    async def contend():
        async def once(_):
            async with pool.lease(SEARCH):
                await asyncio.sleep(0)
        await asyncio.gather(*(once(i) for i in range(4)))   # 4 抢 2

    for _ in range(3):
        asyncio.run(contend())
    assert pool.stats["waits"] > 0, "没走到 wait(),这条测试就什么都没测到"


def test_two_live_loops_sharing_one_pool_is_an_error_not_a_silent_free_for_all():
    """换锁的前提是同一时刻只有一个存活的循环。真有两个并存就说明用法变了,
    此时静默换锁会悄悄废掉互斥(两个循环各拿一把锁,同一个 token 会被借出两次)。
    """
    pool = TokenPool(["a", "b"])
    errors: list[str] = []

    def in_thread() -> None:
        async def hold():
            async with pool.lease(SEARCH):
                await asyncio.sleep(0.3)        # 拖住,让两个循环真正重叠
        try:
            asyncio.run(hold())
        except RuntimeError as e:
            errors.append(str(e))

    threads = [threading.Thread(target=in_thread) for _ in range(2)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert any("两个存活的事件循环" in e for e in errors)


async def _take_one(pool, pace: Pace = CORE) -> int:
    async with pool.lease(pace) as lease:
        return _index_of(lease)


async def _skip_ahead(pool: TokenPool, clock: FakeClock, seconds: float) -> None:
    """推进假时钟,并叫醒池里的等待者重新扫描。

    为什么要显式叫醒:池算出「还要等 30 秒」之后交给真的 `asyncio.wait_for` 去等,
    而这 30 秒是**假时钟**上的 30 秒 —— 真定时器不会因为我们改了假时钟就提前触发。
    等待者被 notify 唤醒后会重扫,此时假时钟已经过了冷却点,于是拿到 token。

    「没人叫醒时它自己到点醒」是另一条路,由 `test_waiter_wakes_itself_on_real_clock`
    用真时钟 + 极短冷却覆盖。
    """
    clock.advance(seconds)
    async with pool._cond:
        pool._cond.notify_all()


# ──────────────────────────────────────────────────────────
# 3. 401 strikes —— 旧包两处 bug 的回归锁
# ──────────────────────────────────────────────────────────


async def test_single_401_does_not_burn_the_token(clock):
    """一次瞬时 401 绝不能永久失效。

    旧包 `cron_daily_star_snapshot.py:168` 与 `api.py:400` 都在 401 上直接 mark_invalid,
    于是每日快照跑一半遇到一次鉴权抖动就少一个 token,烧几个就再也跑不完 7.8 万个仓库。
    """
    pool = make_pool(clock, count=1)

    with pytest.raises(TokenInvalidError):
        async with pool.lease():
            raise TokenInvalidError("瞬时 401")

    assert pool.capacity == 1, "一次 401 就把 token 烧了"

    clock.advance(61)               # 冷却过去
    async with pool.lease():
        pass


async def test_consecutive_401s_eventually_invalidate(clock):
    pool = make_pool(clock, count=1, auth_fail_strikes=3, auth_fail_cooldown=60.0)

    for attempt in range(3):
        with pytest.raises(TokenInvalidError):
            async with pool.lease():
                raise TokenInvalidError(f"401 #{attempt + 1}")
        clock.advance(61)

    assert pool.capacity == 0
    with pytest.raises(AllTokensInvalid):
        async with pool.lease():
            pass


async def test_a_success_resets_the_strike_counter(clock):
    """「连续」是这条规则的全部意义:中间成功一次就重新数。"""
    pool = make_pool(clock, count=1, auth_fail_strikes=3, auth_fail_cooldown=60.0)

    for _ in range(2):
        with pytest.raises(TokenInvalidError):
            async with pool.lease():
                raise TokenInvalidError("401")
        clock.advance(61)

    async with pool.lease():        # 成功一次 → 计数清零
        pass

    for _ in range(2):              # 再来两次也不该失效(需要连续 3 次)
        with pytest.raises(TokenInvalidError):
            async with pool.lease():
                raise TokenInvalidError("401")
        clock.advance(61)

    assert pool.capacity == 1, "成功归还没有清零连续计数"


# ──────────────────────────────────────────────────────────
# 4. 配速
# ──────────────────────────────────────────────────────────


async def test_same_token_respects_the_pace_interval(clock):
    """同一 token 两次搜索之间必须满 2.1 秒 —— 跨调用延续,不靠调用点自己 sleep。"""
    pool = make_pool(clock, count=1)

    assert await _take_one(pool, SEARCH) == 0

    waiter = asyncio.create_task(_take_one(pool, SEARCH))
    await asyncio.sleep(0)
    assert not waiter.done(), "同一 token 立刻又发了一次搜索"

    await _skip_ahead(pool, clock, SEARCH.interval)
    assert await asyncio.wait_for(waiter, timeout=2) == 0


async def test_pace_is_per_token(clock):
    """3 个 token 能连着发 3 次搜索,互不等待 —— 当前口径,沿用旧包。

    ⚠️ 实测很可能说明这个口径是错的(墙在来源 IP 上,不在账号配额上,证据见
    `tokens.py` 的 `SEARCH` 上方)。等新每日快照跑几轮拿到数据再决定要不要改成全局配速;
    真改了,这条测试的方向要跟着反过来。
    """
    pool = make_pool(clock, count=3)

    seen = {await _take_one(pool, SEARCH) for _ in range(3)}

    assert seen == {0, 1, 2}, f"配速被算成了全局的,只用到 {seen}"


async def test_pace_is_per_kind(clock):
    """Search 的 2.1 秒不该拖住 GraphQL —— 两者在 GitHub 那边是分开的限额。

    这也是「搜索和快照采集能同时跑」的前提。
    """
    pool = make_pool(clock, count=1)

    await _take_one(pool, SEARCH)
    assert await asyncio.wait_for(_take_one(pool, CORE), timeout=2) == 0


# ──────────────────────────────────────────────────────────
# 5. 动态容量 —— 锁死「不需要 max_concurrency 数字」这个决定
# ──────────────────────────────────────────────────────────


async def test_capacity_tracks_tokens_without_a_configured_number(clock):
    pool = make_pool(clock, count=3, auth_fail_strikes=1)
    assert pool.capacity == 3

    with pytest.raises(TokenInvalidError):
        async with pool.lease():
            raise TokenInvalidError("401")
    assert pool.capacity == 2, "失效后容量没有立刻下降"

    assert await pool.add(["tok-new"]) == 1
    assert pool.capacity == 3

    assert await pool.add(["tok-new"]) == 0, "重复的 token 不该重复加入"


async def test_concurrent_leases_are_bounded_by_token_count(clock):
    """3 个 token → 最多 3 个并发租约,第 4 个必须等。"""
    pool = make_pool(clock, count=3)
    held = asyncio.Event()
    released = asyncio.Event()
    active = 0
    peak = 0

    async def hold() -> None:
        nonlocal active, peak
        async with pool.lease():
            active += 1
            peak = max(peak, active)
            if active == 3:
                held.set()
            await released.wait()
            active -= 1

    holders = [asyncio.create_task(hold()) for _ in range(3)]
    await asyncio.wait_for(held.wait(), timeout=2)

    fourth = asyncio.create_task(_take_one(pool))
    await asyncio.sleep(0)
    assert not fourth.done(), "借出的租约数超过了 token 数"

    released.set()
    await asyncio.wait_for(fourth, timeout=2)
    await asyncio.gather(*holders)
    assert peak == 3


async def test_adding_a_token_wakes_a_blocked_waiter(clock):
    """所有 token 都在长冷却时,补一个 token 应当立刻放行等待者(而不是干等到冷却结束)。"""
    pool = make_pool(clock, count=1, recovery_buffer=0.0)

    with pytest.raises(RateLimitError):
        async with pool.lease():
            raise RateLimitError(reset_at=clock.t + 3600)

    waiter = asyncio.create_task(_take_one(pool))
    await asyncio.sleep(0)
    assert not waiter.done()

    await pool.add(["fresh"])
    assert await asyncio.wait_for(waiter, timeout=2) == 1


# ──────────────────────────────────────────────────────────
# 其他
# ──────────────────────────────────────────────────────────


def test_empty_pool_is_rejected(clock):
    for bad in ([], ["", "   "]):
        with pytest.raises(ValueError, match="至少需要一个 token"):
            TokenPool(bad, time_fn=clock)


async def test_lease_never_exposes_the_secret_itself(clock):
    """租约只给请求头。token 字符串拿不到,就不会被日志或异常意外带出。"""
    pool = TokenPool(["s3cret"], time_fn=clock)

    async with pool.lease() as lease:
        assert not hasattr(lease, "token")
        assert not hasattr(lease, "secret")
        assert "s3cret" in lease.rest_headers["Authorization"]
        assert repr(lease).find("s3cret") == -1, "repr 里泄了 token"


async def test_seconds_until_any_ready(clock):
    pool = make_pool(clock, count=1, recovery_buffer=0.0)
    assert pool.seconds_until_any_ready() == 0.0

    with pytest.raises(RateLimitError):
        async with pool.lease():
            raise RateLimitError(reset_at=clock.t + 45)

    assert pool.seconds_until_any_ready() == pytest.approx(45, abs=0.01)
