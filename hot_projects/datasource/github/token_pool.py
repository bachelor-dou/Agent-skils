"""
协程调度用的异步 Token 池。

该模块统一管理 Token 生命周期：
- 借出 / 归还
- 限流冷却
- 失效剔除
- 最早恢复等待
"""

from __future__ import annotations

import asyncio
import logging
import sys
import time
from dataclasses import dataclass
from typing import Callable

from ...config import GITHUB_TOKENS

logger = logging.getLogger("hot_projects")

# 401 鉴权失败处理（池内部常量，基本不变动，故不放入 config）：
#   连续命中 AUTH_FAIL_STRIKES 次 401 才永久失效；否则按瞬时故障冷却，
#   冷却期间 token 抛回池中（不被 worker 持有），到点自动回归；任意成功 release 清零计数。
AUTH_FAIL_STRIKES = 3
AUTH_FAIL_COOLDOWN_SECONDS = 60.0

# 每 token 的 Search API 主动配速间隔（池内部常量，由 GitHub 固定的 30 次/分/token 推导，
# 不是可调偏好，故不放入 config）。token 是"按任务"借还的，多数关键词搜索 <3 页、遇空页
# 就 break（无尾部 sleep），token 立刻被下一个任务重新借走并马上发第一页——这种跨任务边界
# 的突发，页间 sleep 再大也拦不住。按 token 维护"下次可发搜索的时刻"（跨任务延续）才压得住。
# 取 2.1s（略高于 30/min 的 2.0s）留余量 → 每 token ≈28.5/min，12 token 聚合 ≈342/min。
SEARCH_TOKEN_MIN_INTERVAL = 2.1


@dataclass(slots=True)
class _TokenState:
    token: str
    in_use: bool = False
    invalid: bool = False
    available_at: float = 0.0
    rate_limited_count: int = 0
    auth_fail_count: int = 0
    last_error: str = ""


class AsyncTokenPool:
    """协程安全的 Token 池，支持冷却与失效处理。"""

    def __init__(
        self,
        tokens: list[str],
        recovery_buffer_seconds: float = 3.0,
        time_fn: Callable[[], float] | None = None,
        health_degrade_threshold: int = 3,
        health_penalty_seconds: float = 2.0,
        wait_log_interval_seconds: float = 10.0,
        auth_fail_strikes: int = AUTH_FAIL_STRIKES,
        auth_fail_cooldown_seconds: float = AUTH_FAIL_COOLDOWN_SECONDS,
        search_min_interval: float = 0.0,
    ) -> None:
        normalized = [t.strip() for t in tokens if t and t.strip()]
        if not normalized:
            raise ValueError("AsyncTokenPool requires at least one token")

        self._states: list[_TokenState] = [_TokenState(token=t) for t in normalized]
        # 按 token 的 Search API 配速：每个 token 下一次允许发搜索请求的最早时刻。
        self._search_min_interval = max(0.0, search_min_interval)
        self._search_next_at: list[float] = [0.0] * len(normalized)
        self._condition = asyncio.Condition()
        self._condition_loop: asyncio.AbstractEventLoop | None = None
        self._recovery_buffer_seconds = max(0.0, recovery_buffer_seconds)
        self._time_fn = time_fn or time.time
        self._health_degrade_threshold = max(1, health_degrade_threshold)
        self._health_penalty_seconds = max(0.0, health_penalty_seconds)
        self._auth_fail_strikes = max(1, auth_fail_strikes)
        self._auth_fail_cooldown_seconds = max(0.0, auth_fail_cooldown_seconds)
        self._wait_log_interval_seconds = max(0.0, wait_log_interval_seconds)
        self._last_wait_log_at = 0.0
        self._last_wait_log_key: tuple[str, int] | None = None
        self._metrics = {
            "acquire_wait_count": 0,
            "acquire_wait_total_seconds": 0.0,
            "rate_limited_total": 0,
            "invalid_total": 0,
        }

    @property
    def token_count(self) -> int:
        return len(self._states)

    def get_token(self, token_idx: int) -> str:
        """按索引返回 token 字符串，用于构造请求头。"""
        self._validate_token_idx(token_idx)
        return self._states[token_idx].token

    async def acquire(self) -> int:
        """获取一个当前可用的 token 索引。

        Raises:
            RuntimeError: 当所有 token 都失效时抛出。
        """
        condition = self._ensure_condition_for_current_loop()
        async with condition:
            while True:
                now = self._time_fn()
                idx = self._find_available_token_idx(now)
                if idx is not None:
                    self._states[idx].in_use = True
                    return idx

                if not self._has_non_invalid_tokens():
                    raise RuntimeError("All tokens are invalid")

                wait_seconds = self._compute_wait_seconds(now)
                if wait_seconds is None:
                    self._log_wait_state("busy")
                    self._metrics["acquire_wait_count"] += 1
                    await condition.wait()
                    continue

                if wait_seconds <= 0:
                    self._metrics["acquire_wait_count"] += 1
                    await condition.wait()
                    continue

                try:
                    self._log_wait_state("cooldown", wait_seconds)
                    self._metrics["acquire_wait_count"] += 1
                    self._metrics["acquire_wait_total_seconds"] += wait_seconds
                    await asyncio.wait_for(condition.wait(), timeout=wait_seconds)
                except asyncio.TimeoutError:
                    # Wake up and re-scan availability.
                    pass

    async def throttle_search(self, token_idx: int) -> None:
        """按 token 主动配速 Search API：在发起搜索请求前调用。

        GitHub Search 限额是 30 次/分/token，且限额窗口不随任务边界重置。此处按 token 维护
        「下一次允许发搜索的时刻」，同一 token 两次搜索至少间隔 search_min_interval 秒。
        调用方必须已独占该 token（A 模式任务级持有，或 B 模式 acquire 之后），因此无需加锁：
        同一 token_idx 不会有并发协程同时进入。主动等待可避免撞 429 后被罚 60s 到点冷却，
        总时长通常反而更短。
        """
        if self._search_min_interval <= 0:
            return
        self._validate_token_idx(token_idx)
        now = self._time_fn()
        next_at = self._search_next_at[token_idx]
        if next_at > now:
            await asyncio.sleep(next_at - now)
            now = self._time_fn()
        self._search_next_at[token_idx] = now + self._search_min_interval

    async def release(self, token_idx: int) -> None:
        """释放已借出的 token。"""
        self._validate_token_idx(token_idx)
        condition = self._ensure_condition_for_current_loop()
        async with condition:
            state = self._states[token_idx]
            state.in_use = False
            state.auth_fail_count = 0  # 成功归还视为该 token 正常，清零 401 连续计数
            condition.notify()  # 只唤醒 1 个等待者，避免惊群

    async def mark_rate_limited(self, token_idx: int, reset_time: float, reason: str = "") -> None:
        """命中限流后写入冷却时间并释放 token。

        reset_time 为 GitHub X-RateLimit-Reset 的 epoch 秒级时间戳。
        """
        self._validate_token_idx(token_idx)
        condition = self._ensure_condition_for_current_loop()
        async with condition:
            self._apply_rate_limited_state(token_idx, reset_time, reason)
            self._metrics["rate_limited_total"] += 1
            condition.notify()  # 唤醒 1 个等待者重新扫描可用 token

    async def mark_invalid(self, token_idx: int, reason: str = "") -> None:
        """将 token 永久标记为失效并移出后续调度。"""
        self._validate_token_idx(token_idx)
        condition = self._ensure_condition_for_current_loop()
        async with condition:
            self._apply_invalid_state(token_idx, reason)
            self._metrics["invalid_total"] += 1
            # 失效会改变 acquire() 的 "全部失效→抛错" 出口条件，必须唤醒所有等待者，
            # 否则在最后一个 token 失效时，已挂起的 worker 会永久错过该出口而死锁。
            condition.notify_all()

    async def mark_auth_failed(self, token_idx: int, reason: str = "") -> None:
        """处理 401：默认按瞬时故障冷却，连续多次才永久失效。

        GitHub 可能对有效 token 返回瞬时 401（二级限流/鉴权抖动/连接半坏）。
        若一律永久失效，会因一次抖动把所有有效 token 踢光。此处：
          - 连续 auth_fail_count < strikes：冷却 cooldown 秒后自动回归。
          - 达到 strikes：才永久失效。
        任意一次成功 release 会清零计数。无论哪种分支都 notify_all，
        让挂起的 worker 重新判定可用性/失效出口。
        """
        self._validate_token_idx(token_idx)
        condition = self._ensure_condition_for_current_loop()
        async with condition:
            state = self._states[token_idx]
            state.auth_fail_count += 1
            state.last_error = reason or state.last_error
            if state.auth_fail_count >= self._auth_fail_strikes:
                self._apply_invalid_state(token_idx, reason)
                self._metrics["invalid_total"] += 1
                logger.warning(
                    "token#%d 连续 %d 次 401，永久失效。",
                    token_idx, state.auth_fail_count,
                )
            else:
                state.in_use = False
                state.available_at = max(
                    state.available_at,
                    self._time_fn() + self._auth_fail_cooldown_seconds,
                )
                logger.warning(
                    "token#%d 命中 401（第 %d/%d 次），冷却 %.0fs 后重试。",
                    token_idx, state.auth_fail_count, self._auth_fail_strikes,
                    self._auth_fail_cooldown_seconds,
                )
            condition.notify_all()

    def record_rate_limited(self, token_idx: int, reset_time: float, reason: str = "") -> None:
        """同步阶段写回限流状态。

        仅用于主线程串行链路（例如分段探测）把观测到的限流写回统一 token 池。
        """
        self._validate_token_idx(token_idx)
        self._apply_rate_limited_state(token_idx, reset_time, reason)
        self._metrics["rate_limited_total"] += 1

    def record_invalid(self, token_idx: int, reason: str = "") -> None:
        """同步阶段写回失效状态。"""
        self._validate_token_idx(token_idx)
        self._apply_invalid_state(token_idx, reason)
        self._metrics["invalid_total"] += 1

    def record_auth_failed(self, token_idx: int, reason: str = "") -> None:
        """同步阶段处理 401：与异步 mark_auth_failed 一致——连续 strikes 次才永久失效，
        否则按瞬时故障冷却，避免主线程串行链路（分段探测等）被一次瞬时 401 永久踢除有效 token。

        注意：同步路径无成功 release 的清零钩子，strikes 在本次进程内只增不减；
        但这些链路都很短（Phase0 分段 / 单仓查询），累计到永久失效的风险很低。
        """
        self._validate_token_idx(token_idx)
        state = self._states[token_idx]
        state.auth_fail_count += 1
        state.last_error = reason or state.last_error
        if state.auth_fail_count >= self._auth_fail_strikes:
            self._apply_invalid_state(token_idx, reason)
            self._metrics["invalid_total"] += 1
            logger.warning(
                "token#%d 连续 %d 次 401，永久失效（同步路径）。",
                token_idx, state.auth_fail_count,
            )
        else:
            state.in_use = False
            state.available_at = max(
                state.available_at,
                self._time_fn() + self._auth_fail_cooldown_seconds,
            )
            logger.warning(
                "token#%d 命中 401（第 %d/%d 次，同步路径），冷却 %.0fs 后重试。",
                token_idx, state.auth_fail_count, self._auth_fail_strikes,
                self._auth_fail_cooldown_seconds,
            )

    def _ensure_condition_for_current_loop(self) -> asyncio.Condition:
        loop = asyncio.get_running_loop()
        if self._condition_loop is loop:
            return self._condition

        # scheduled_update 会分阶段多次 asyncio.run；同一 token 池跨事件循环复用时，
        # 需要为新 loop 重新绑定 condition，避免后续 wait/notify 命中 loop 绑定错误。
        self._condition = asyncio.Condition()
        self._condition_loop = loop
        return self._condition

    def _validate_token_idx(self, token_idx: int) -> None:
        if token_idx < 0 or token_idx >= len(self._states):
            raise IndexError(f"token_idx out of range: {token_idx}")

    def _apply_rate_limited_state(self, token_idx: int, reset_time: float, reason: str = "") -> None:
        state = self._states[token_idx]
        cooldown_until = reset_time + self._recovery_buffer_seconds
        if state.rate_limited_count + 1 >= self._health_degrade_threshold:
            degrade_steps = state.rate_limited_count + 1 - self._health_degrade_threshold + 1
            cooldown_until += degrade_steps * self._health_penalty_seconds
        state.available_at = max(state.available_at, cooldown_until)
        state.in_use = False
        state.rate_limited_count += 1
        state.last_error = reason or state.last_error

    def _apply_invalid_state(self, token_idx: int, reason: str = "") -> None:
        state = self._states[token_idx]
        state.invalid = True
        state.in_use = False
        state.available_at = float("inf")
        state.last_error = reason or state.last_error

    def _log_wait_state(self, state: str, wait_seconds: float | None = None) -> None:
        bucket = -1 if wait_seconds is None else int(wait_seconds)
        key = (state, bucket)
        now = self._time_fn()
        if (
            self._last_wait_log_key == key
            and self._wait_log_interval_seconds > 0
            and now - self._last_wait_log_at < self._wait_log_interval_seconds
        ):
            return

        self._last_wait_log_key = key
        self._last_wait_log_at = now

        if state == "busy":
            logger.info("当前无空闲 token，等待其他任务释放 token。")
            return

        if wait_seconds is not None:
            logger.info("当前无可用 token，最早 %.2fs 后恢复。", wait_seconds)

    def _find_available_token_idx(self, now: float) -> int | None:
        for idx, state in enumerate(self._states):
            if state.invalid or state.in_use:
                continue
            if state.available_at <= now:
                return idx
        return None

    def seconds_until_all_cool(self) -> float:
        """所有未失效 token 都脱离限流冷却所需的秒数（0 = 当前已全部可用）。

        供页级补偿使用：只要还有 token 在冷却，立刻重跑失败页会再撞一次限流，
        任务一命中限流就把剩余页整批丢回失败集，整轮补偿等于白跑。
        """
        now = self._time_fn()
        waits = [s.available_at - now for s in self._states if not s.invalid]
        return max(0.0, max(waits, default=0.0))

    def _has_non_invalid_tokens(self) -> bool:
        return any(not s.invalid for s in self._states)

    def _compute_wait_seconds(self, now: float) -> float | None:
        earliest: float | None = None
        for state in self._states:
            if state.invalid or state.in_use:
                continue
            if earliest is None or state.available_at < earliest:
                earliest = state.available_at

        if earliest is None:
            return None

        return earliest - now


class GitHubTokenPool(AsyncTokenPool):
    """统一 GitHub Token 池：兼具调度状态与请求头构建能力。"""

    def __init__(
        self,
        tokens: list[str] | None = None,
        recovery_buffer_seconds: float = 3.0,
        time_fn: Callable[[], float] | None = None,
        health_degrade_threshold: int = 3,
        health_penalty_seconds: float = 2.0,
    ) -> None:
        source_tokens = GITHUB_TOKENS if tokens is None else tokens
        normalized = [t.strip() for t in source_tokens if t and t.strip()]
        if not normalized:
            logger.error("未配置任何 GitHub Token，无法运行。请设置 GITHUB_TOKENS 环境变量。")
            sys.exit(1)

        super().__init__(
            tokens=normalized,
            recovery_buffer_seconds=recovery_buffer_seconds,
            time_fn=time_fn,
            health_degrade_threshold=health_degrade_threshold,
            health_penalty_seconds=health_penalty_seconds,
            search_min_interval=SEARCH_TOKEN_MIN_INTERVAL,
        )
        logger.info("GitHubTokenPool 初始化: 共 %d 个 token 可用。", len(normalized))

    @property
    def tokens(self) -> list[str]:
        """返回当前 token 列表，兼容历史调用方。"""
        return [state.token for state in self._states]

    def get_rest_headers(self, token_idx: int) -> dict[str, str]:
        """REST API 通用请求头。"""
        return {
            "Authorization": f"token {self.get_token(token_idx)}",
            "Accept": "application/vnd.github.v3+json",
        }

    def get_star_headers(self, token_idx: int) -> dict[str, str]:
        """REST stargazers 请求头（返回 starred_at 时间戳）。"""
        return {
            "Authorization": f"token {self.get_token(token_idx)}",
            "Accept": "application/vnd.github.v3.star+json",
        }

    def get_graphql_headers(self, token_idx: int) -> dict[str, str]:
        """GraphQL API 请求头。"""
        return {
            "Authorization": f"bearer {self.get_token(token_idx)}",
            "Content-Type": "application/json",
        }
