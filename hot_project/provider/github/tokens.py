"""GitHub token 池 —— 对外只有一个动作:借一张租约。

拿到租约就意味着这个 token 同时满足三件事:**没被别人借走、不在限流冷却里、距它自己上次
同类请求已满最小间隔**。退出 `async with` 自动归还,并按抛出的异常类型自动记账。

索引只存在于池内部:异常 → 记账动作的映射只在 `Lease.__aexit__` 一处,调用方伸不进手来。
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import NamedTuple

from ...infra.exceptions import RateLimitError, TokenInvalidError

logger = logging.getLogger("hot_project")


class Pace(NamedTuple):
    """一类请求的配速:同一个 token 连续两次这类请求的最小间隔。

    必须按类分开 —— GitHub 的限额本身是分开的,让 GraphQL 去等 Search 的间隔会把批量取 star
    拖慢一个数量级。
    """
    key: str
    interval: float


# Search API 每 token 30 次/分 —— 但这不是我们撞到的墙:实测被拒时 remaining 还剩 21/30。
# 真正的天花板是二级限流,按**来源 IP** 计,12 张独立账号的 token 共用一个额度,所以全体
# 在同一秒里一起撞墙(retry-after 60)。总量约 100 次/分(12 token × 9 次)就到顶。
#
# **这个旋钮控制不了撞墙,别再拿它调。** 三轮实测:2.1s 撞墙周期 65s、发到 108 次才撞;
# 8.0s 周期反而拉到 93s、只发到 36 次就撞。放慢它撞得更早,可见墙不是按请求速率算的
# (真正的成因未知,GitHub 也明说「may encounter a secondary rate limit for undisclosed
# reasons」)。既然撞是必然,就该在罚站之间尽量多干活 —— 配速小反而吞吐高。
#
# 2.5s 是取的保守值:再小虽然更快,但被限流期间继续加压有封号风险,不值得。
#
# 间隔必须按 token 跨调用延续,不能只在页与页之间 sleep:多数搜索不到 3 页就结束、尾部
# 没有 sleep,token 立刻被下一个任务借走并马上发第一页,页间 sleep 再大也拦不住这种突发。
SEARCH = Pace("search", 2.5)

# GraphQL / REST 走 core 限额 5000 点/小时,批量取 star 是 100 个仓库一次请求,配额是零头,
# 所以间隔为 0 —— 保留这个 Pace 只是为了让两条路径同形。
CORE = Pace("core", 0.0)


# 401 的处置:连续这么多次才永久失效,否则按瞬时故障冷却。
# 任意一次成功归还清零计数 —— 「连续」是这条规则的全部意义所在。
AUTH_FAIL_STRIKES = 3
AUTH_FAIL_COOLDOWN = 60.0

# 限流恢复后再多等一会儿:reset 时刻和实际放行之间有抖动,掐着点重发会再吃一个 403,
# 而一个 403 的代价是整轮冷却。
RECOVERY_BUFFER = 3.0


@dataclass(slots=True)
class _Token:
    secret: str
    in_use: bool = False
    invalid: bool = False
    available_at: float = 0.0                       # 限流冷却到期时刻
    next_at: dict[str, float] = field(default_factory=dict)   # 各类请求的配速下一可发时刻
    auth_fails: int = 0                             # 连续 401 次数(成功归还即清零)
    rate_limit_hits: int = 0
    used_seq: int = 0                               # 上次被借出的序号,用于轮转(见 _pick)
    last_error: str = ""

    def ready_at(self, pace: Pace) -> float:
        """这个 token 最早能为 `pace` 这类请求所用的时刻。"""
        return max(self.available_at, self.next_at.get(pace.key, 0.0))


class Lease:
    """一次租约。只给请求头,不给 token 字符串 —— 让它没法被日志或异常顺手带出去。"""

    __slots__ = ("_pool", "_index")

    def __init__(self, pool: TokenPool, index: int) -> None:
        self._pool = pool
        self._index = index

    @property
    def rest_headers(self) -> dict[str, str]:
        return {
            "Authorization": f"token {self._pool._secret(self._index)}",
            "Accept": "application/vnd.github.v3+json",
        }

    @property
    def graphql_headers(self) -> dict[str, str]:
        return {
            "Authorization": f"bearer {self._pool._secret(self._index)}",
            "Content-Type": "application/json",
        }


class AllTokensInvalid(RuntimeError):
    """所有 token 都永久失效了 —— 等下去也不会有转机,只能让调用方停。"""


class TokenPool:
    """协程安全的 token 池。容量 = 当前未失效的 token 数,会随失效/新增实时变化。"""

    def __init__(
        self,
        secrets: list[str],
        *,
        time_fn: Callable[[], float] | None = None,
        auth_fail_strikes: int = AUTH_FAIL_STRIKES,
        auth_fail_cooldown: float = AUTH_FAIL_COOLDOWN,
        recovery_buffer: float = RECOVERY_BUFFER,
    ) -> None:
        cleaned = [s.strip() for s in secrets if s and s.strip()]
        if not cleaned:
            raise ValueError("token 池至少需要一个 token(设置 GITHUB_TOKENS)")

        self._tokens = [_Token(secret=s) for s in cleaned]
        # 先建好只是让字段存在;真正绑到哪个循环由 `_bind_to_running_loop` 在首次借出时决定。
        self._cond = asyncio.Condition()
        self._cond_loop: asyncio.AbstractEventLoop | None = None
        self._now = time_fn or time.time
        self._strikes = max(1, auth_fail_strikes)
        self._auth_cooldown = max(0.0, auth_fail_cooldown)
        self._recovery_buffer = max(0.0, recovery_buffer)
        self._seq = 0                   # 单调递增的借出序号,_pick 靠它做轮转
        self.stats = {"leases": 0, "waits": 0, "waited_seconds": 0.0,
                      "rate_limited": 0, "invalidated": 0}
        logger.info("token 池初始化:%d 个 token。", len(self._tokens))

    # ── 容量 ──────────────────────────────────────────────

    @property
    def capacity(self) -> int:
        """还能同时借出多少张租约(未失效的 token 数)。

        任务侧的并发度读它而不是写死的 `max_concurrency` —— 401 三振或新增 token 后它立刻变。
        """
        return sum(1 for t in self._tokens if not t.invalid)

    async def add(self, secrets: list[str]) -> int:
        """加入新 token(已在池里的忽略),返回实际加入数。用于运行期补 token。"""
        known = {t.secret for t in self._tokens}
        fresh = [s.strip() for s in secrets if s and s.strip() and s.strip() not in known]
        if not fresh:
            return 0
        self._bind_to_running_loop()
        async with self._cond:
            self._tokens.extend(_Token(secret=s) for s in fresh)
            self._cond.notify_all()      # 可能有 worker 正因「全都在冷却」而挂着
        logger.info("token 池新增 %d 个,现有容量 %d。", len(fresh), self.capacity)
        return len(fresh)

    # ── 借出 ──────────────────────────────────────────────

    @asynccontextmanager
    async def lease(self, pace: Pace = CORE) -> AsyncIterator[Lease]:
        """借一张租约:`async with pool.lease(SEARCH) as lease:`。

        没有可用 token 时**等待**而不是报错 —— 限流是常态。只有全部 token 永久失效才抛
        `AllTokensInvalid`。下面这段 try 是全代码库唯一决定「一次失败怎么记账」的地方。
        """
        lease = await self._acquire(pace)
        index = lease._index
        try:
            yield lease
        except RateLimitError as e:
            await self._on_rate_limited(index, e.reset_at, str(e))
            raise
        except TokenInvalidError as e:
            await self._on_auth_failed(index, str(e))
            raise
        except BaseException:
            # 不是 token 的错,只归还。但也不清零 401 计数:那次请求没成功,不算「健康」的证据。
            await self._on_released(index, healthy=False)
            raise
        else:
            await self._on_released(index, healthy=True)

    def _bind_to_running_loop(self) -> None:
        """把 `_cond` 绑到当前这个事件循环上,必要时换一把新的。

        `asyncio.Condition` 绑死在第一个 await 它的循环上,而 `facade` 每个方法各起一个
        `asyncio.run` —— 不换锁,第二次调用只要有人真的等锁就会崩,且池从此报废。而换锁安全的
        前提是同一时刻只有一个存活的循环,两个并存时静默换锁会废掉互斥,所以那时直接报错。
        """
        loop = asyncio.get_running_loop()
        if self._cond_loop is loop:
            return
        if self._cond_loop is not None and not self._cond_loop.is_closed():
            raise RuntimeError(
                "同一个 token 池被两个存活的事件循环同时使用 —— 换锁会废掉互斥。"
                "请让这些调用共用一个事件循环,或各用一个池。"
            )
        self._cond = asyncio.Condition()
        self._cond_loop = loop

    async def _acquire(self, pace: Pace) -> Lease:
        self._bind_to_running_loop()
        async with self._cond:
            while True:
                now = self._now()
                index = self._pick(now, pace)
                if index is not None:
                    token = self._tokens[index]
                    token.in_use = True
                    self._seq += 1
                    token.used_seq = self._seq
                    if pace.interval > 0:
                        token.next_at[pace.key] = now + pace.interval
                    self.stats["leases"] += 1
                    return Lease(self, index)

                if self.capacity == 0:
                    raise AllTokensInvalid(
                        f"{len(self._tokens)} 个 token 全部永久失效,无法继续。"
                    )

                self.stats["waits"] += 1
                wait = self._earliest_ready(now, pace)
                if wait is None:
                    # 全被借走了 —— 没有确定的到期时刻,只能等某个租约归还时来叫醒。
                    await self._cond.wait()
                    continue

                self.stats["waited_seconds"] += wait
                try:
                    # 等冷却/配速到期,但也接受被提前叫醒(有人归还了、或新增了 token)。
                    await asyncio.wait_for(self._cond.wait(), timeout=wait)
                except TimeoutError:
                    pass

    def _pick(self, now: float, pace: Pace) -> int | None:
        """挑一个此刻就能用的 token:在够格的里面选**最久没用过**的那个。

        不能按 `ready_at` 排序 —— 限流过的 token 的 `available_at` 很大,冷却结束后仍会永远
        排在后面;也不能取第一个够格的 —— 0 号会一直被选中、卡在自己的配速上,后面的闲着。
        """
        best: int | None = None
        for index, token in enumerate(self._tokens):
            if token.invalid or token.in_use or token.ready_at(pace) > now:
                continue
            if best is None or token.used_seq < self._tokens[best].used_seq:
                best = index
        return best

    def _earliest_ready(self, now: float, pace: Pace) -> float | None:
        """还要等多久才有 token 可用。全部在被使用中(没有确定到期时刻)则返回 None。"""
        waits = [
            t.ready_at(pace) - now
            for t in self._tokens
            if not t.invalid and not t.in_use
        ]
        return min(waits) if waits else None

    # ── 记账(只由 Lease.__aexit__ 调用)──────────────────

    def _secret(self, index: int) -> str:
        return self._tokens[index].secret

    async def _on_released(self, index: int, *, healthy: bool) -> None:
        # 先同步清掉 in_use,再去抢锁:归还路径本身就跑在异常里(见 lease 的
        # except BaseException),取消要是打在下面那个 await 上,in_use 就永远清不掉 ——
        # token 被漏掉、_pick 永不返回它,若它是最后一张,等待者全挂死、join() 永不返回。
        self._tokens[index].in_use = False
        async with self._cond:
            token = self._tokens[index]
            if healthy:
                token.auth_fails = 0
            # notify_all 而不是 notify:等待者的条件各不相同(要的 pace 不一样),被叫醒的那个
            # _pick 不到就把这次唤醒白吃掉,而挂在**无限** wait() 里的那个没被叫到。
            self._cond.notify_all()

    async def _on_rate_limited(self, index: int, reset_at: float, reason: str) -> None:
        self._tokens[index].in_use = False      # 同步清,理由见 _on_released
        async with self._cond:
            token = self._tokens[index]
            token.rate_limit_hits += 1
            token.last_error = reason
            token.available_at = max(token.available_at, reset_at + self._recovery_buffer)
            self.stats["rate_limited"] += 1
            logger.info("token#%d 限流,%.0fs 后恢复(%s)。",
                        index, token.available_at - self._now(), reason)
            self._cond.notify_all()      # 理由见 _on_released

    async def _on_auth_failed(self, index: int, reason: str) -> None:
        """401:连续 `strikes` 次才永久失效,否则只冷却。这是唯一的失效入口。"""
        self._tokens[index].in_use = False      # 同步清,理由见 _on_released
        async with self._cond:
            token = self._tokens[index]
            token.auth_fails += 1
            token.last_error = reason

            if token.auth_fails >= self._strikes:
                token.invalid = True
                token.available_at = float("inf")
                self.stats["invalidated"] += 1
                logger.warning("token#%d 连续 %d 次 401,永久失效(容量 → %d)。",
                               index, token.auth_fails, self.capacity)
                # 失效改变了「全部失效 → 抛错」这个出口条件,必须叫醒**所有**等待者,否则最后
                # 一个 token 失效时已挂起的 worker 会永久错过出口而死锁。
                self._cond.notify_all()
            else:
                token.available_at = max(token.available_at,
                                         self._now() + self._auth_cooldown)
                logger.warning("token#%d 命中 401(第 %d/%d 次),冷却 %.0fs 后重试。",
                               index, token.auth_fails, self._strikes, self._auth_cooldown)
                self._cond.notify_all()      # 理由见 _on_released

    # ── 只读观测 ──────────────────────────────────────────

    def seconds_until_any_ready(self, pace: Pace = CORE) -> float:
        """最快多久会有 token 可用(0 = 现在就有)。

        给页级补偿用:还有 token 在冷却时立刻重跑失败页会再撞限流,一撞剩余页整批回失败集。
        """
        now = self._now()
        waits = [t.ready_at(pace) - now for t in self._tokens if not t.invalid]
        return max(0.0, min(waits, default=0.0))
