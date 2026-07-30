"""GitHub token 池 —— 对外只有一个动作:借一张租约。

    async with pool.lease(SEARCH) as lease:
        resp = await client.get(url, headers=lease.rest_headers)

拿到租约就意味着这个 token 同时满足三件事:**没被别人借走、不在限流冷却里、
距它自己上次同类请求已经满了最小间隔**。退出 `async with` 自动归还,而且按抛出的异常
类型自动记账。

## 为什么是租约,而不是「给我一个 token 索引」

旧池的接口是 `idx = await pool.acquire()` + `pool.get_token(idx)` + `await pool.release(idx)`,
索引一路传给调用方,于是**每个调用方都得自己决定异常怎么记账**。同一次 401 长出了四种
处置,其中两处直接永久失效(见 `infra/exceptions.py` 头部)。那不是有人写错了,是接口
把「必须每处都做对的判断」交给了每一处。

租约把索引关在池内部。异常 → 记账动作的映射只存在于 `Lease.__aexit__` 一个地方,
调用方连做错的手都伸不进来。

## 顺带消灭的三样东西

- **`max_concurrency` 数字**:池的容量就是可用 token 数。想要「并发度等于 token 数」的任务
  直接一人借一张租约,拿不到就等 —— 不需要谁去传一个必须和 token 数保持同步的常量。
- **四处散落的同步 `time.sleep(SEARCH_REQUEST_INTERVAL)`**(`api.py:777`、
  `tools/basic/core.py:711` 等):配速是 token 的属性,不是调用点的属性。现在它是
  `lease()` 能否成立的第三个条件。
- **跨事件循环重绑 Condition**:那是旧包分阶段多次 `asyncio.run` 逼出来的补丁。
  新设计一个进程一个 `asyncio.run`,不需要。
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

    按类分开计数,因为 GitHub 的限额本身是分开的 —— 让 GraphQL 去等 Search 的 2.1 秒
    会把批量取 star 拖慢一个数量级。
    """
    key: str
    interval: float


# Search API 限额 30 次/分/token。取 2.1s(略高于 2.0s)留余量 → 每 token ≈28.5/min。
# 必须按 token 跨调用延续,不能只在页与页之间 sleep:多数关键词搜索不到 3 页、遇空页就
# 结束(尾部没有 sleep),token 立刻被下一个任务借走并马上发第一页 —— 这种跨任务边界的
# 突发,页间 sleep 再大也拦不住。沿用旧包的口径。
#
# ⚠️ 这个模型很可能不对,但**先不动**,等新每日快照真跑几轮再按数据调。已测到的反证:
#
#   07-30 11:00     11:00:26–29 三秒内 token 0..11 全部撞限流,静默 60s,下一分钟原样重演。
#                   一轮 176 次限流、171 次全池空转。
#   07-29 DEBUG     每个 token 每分钟只领到 ≤6 个搜索任务(约 15 次请求),**只用掉自己
#                   30/min 主限额的一半**;而 12 个 token 分属 12 个独立账号。
#
# 账号配额没被碰到却全体撞墙 → 墙在**来源(IP)**那一侧,聚合上限实测约 120 次/分钟
# ≈ 2 req/s,且与 token 数无关。真要修就是把配速从「每 token」改成「全局」,并且区分
# 主限额耗尽(冷却该 token)与二级限流(降全局速率)。留到快照跑顺之后再做。
SEARCH = Pace("search", 2.1)

# GraphQL / REST 走 core 限额 5000 点/小时,而批量取 star 是 100 个仓库一次请求。
# 实测 779 批 600 秒、一次限流都没有,所以间隔为 0 —— 保留这个 Pace 只是为了让两条路径同形。
CORE = Pace("core", 0.0)


# 401 的处置:连续这么多次才永久失效,否则按瞬时故障冷却。
# 任意一次成功归还清零计数 —— 「连续」是这条规则的全部意义所在。
AUTH_FAIL_STRIKES = 3
AUTH_FAIL_COOLDOWN = 60.0

# 限流恢复后再多等一会儿:GitHub 给的 reset 时刻和它实际放行之间有抖动,
# 掐着点重发会再吃一个 403,而一个 403 的代价是整轮冷却。
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
        self._cond = asyncio.Condition()
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
        """还能同时借出多少张租约。

        任务侧要「并发度等于 token 数」时读它,而不是读一个写死的 `max_concurrency`:
        某个 token 401 三振出局后容量立刻少一,新增 token 后立刻多一,不需要谁去同步。
        """
        return sum(1 for t in self._tokens if not t.invalid)

    async def add(self, secrets: list[str]) -> int:
        """加入新 token(已在池里的忽略),返回实际加入数。

        用在运行期补 token:CI 里 12 个 token 烧掉几个之后,可以不重启进程补上。
        """
        known = {t.secret for t in self._tokens}
        fresh = [s.strip() for s in secrets if s and s.strip() and s.strip() not in known]
        if not fresh:
            return 0
        async with self._cond:
            self._tokens.extend(_Token(secret=s) for s in fresh)
            self._cond.notify_all()      # 可能有 worker 正因「全都在冷却」而挂着
        logger.info("token 池新增 %d 个,现有容量 %d。", len(fresh), self.capacity)
        return len(fresh)

    # ── 借出 ──────────────────────────────────────────────

    @asynccontextmanager
    async def lease(self, pace: Pace = CORE) -> AsyncIterator[Lease]:
        """借一张租约:`async with pool.lease(SEARCH) as lease:`。

        没有可用 token 时**等待**而不是报错 —— 限流是常态,报错只会让每个调用方各自再实现
        一遍重试。只有全部 token 永久失效才抛 `AllTokensInvalid`(那种情况等下去没有转机)。

        下面这段 try 是**全代码库唯一决定「一次失败怎么记账」的地方**。旧包把这个判断散给了
        四个捕获点,其中两个选了永久失效。这里没有第二个入口,也就没有第二种答案。
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
            # 网络抖动、解析失败、取消 —— 都不是 token 的错,只归还。
            # 但也不清零 401 计数:那次请求毕竟没成功,不能拿它当「健康」的证据。
            await self._on_released(index, healthy=False)
            raise
        else:
            await self._on_released(index, healthy=True)

    async def _acquire(self, pace: Pace) -> Lease:
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

        为什么不是「第一个够格的」:固定从头扫会让 0 号一直被选中、一直卡在自己的配速间隔上,
        后面的反而闲着。

        为什么不是「`ready_at` 最早的」:限流过一次的 token 的 `available_at` 是个很大的数,
        按它排序会让这个 token 在冷却结束之后仍然永远排在没限流过的后面 ——
        `test_rate_limited_token_is_skipped_until_reset` 就是这么发现的。
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
        async with self._cond:
            token = self._tokens[index]
            token.in_use = False
            if healthy:
                token.auth_fails = 0
            self._cond.notify()          # 只叫醒一个,避免惊群

    async def _on_rate_limited(self, index: int, reset_at: float, reason: str) -> None:
        async with self._cond:
            token = self._tokens[index]
            token.in_use = False
            token.rate_limit_hits += 1
            token.last_error = reason
            token.available_at = max(token.available_at, reset_at + self._recovery_buffer)
            self.stats["rate_limited"] += 1
            logger.info("token#%d 限流,%.0fs 后恢复。", index, token.available_at - self._now())
            self._cond.notify()

    async def _on_auth_failed(self, index: int, reason: str) -> None:
        """401:连续 `strikes` 次才永久失效,否则只冷却。

        这条规则本身旧池就有,坏在有两个调用方绕过它直接 `mark_invalid`。现在没有别的入口。
        """
        async with self._cond:
            token = self._tokens[index]
            token.in_use = False
            token.auth_fails += 1
            token.last_error = reason

            if token.auth_fails >= self._strikes:
                token.invalid = True
                token.available_at = float("inf")
                self.stats["invalidated"] += 1
                logger.warning("token#%d 连续 %d 次 401,永久失效(容量 → %d)。",
                               index, token.auth_fails, self.capacity)
                # 失效会改变「全部失效 → 抛错」这个出口条件,必须叫醒**所有**等待者,
                # 否则最后一个 token 失效时,已挂起的 worker 会永久错过出口而死锁。
                self._cond.notify_all()
            else:
                token.available_at = max(token.available_at,
                                         self._now() + self._auth_cooldown)
                logger.warning("token#%d 命中 401(第 %d/%d 次),冷却 %.0fs 后重试。",
                               index, token.auth_fails, self._strikes, self._auth_cooldown)
                self._cond.notify()

    # ── 只读观测 ──────────────────────────────────────────

    def seconds_until_any_ready(self, pace: Pace = CORE) -> float:
        """最快多久会有 token 可用(0 = 现在就有)。

        给页级补偿用:还有 token 在冷却时立刻重跑失败页会再撞一次限流,
        一撞就把剩余页整批丢回失败集,整轮补偿等于白跑。
        """
        now = self._now()
        waits = [t.ready_at(pace) - now for t in self._tokens if not t.invalid]
        return max(0.0, min(waits, default=0.0))
