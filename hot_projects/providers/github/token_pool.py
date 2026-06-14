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

logger = logging.getLogger("discover_hot")


@dataclass(slots=True)
class _TokenState:
    token: str
    in_use: bool = False
    invalid: bool = False
    available_at: float = 0.0
    rate_limited_count: int = 0
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
    ) -> None:
        normalized = [t.strip() for t in tokens if t and t.strip()]
        if not normalized:
            raise ValueError("AsyncTokenPool requires at least one token")

        self._states: list[_TokenState] = [_TokenState(token=t) for t in normalized]
        self._condition = asyncio.Condition()
        self._condition_loop: asyncio.AbstractEventLoop | None = None
        self._recovery_buffer_seconds = max(0.0, recovery_buffer_seconds)
        self._time_fn = time_fn or time.time
        self._health_degrade_threshold = max(1, health_degrade_threshold)
        self._health_penalty_seconds = max(0.0, health_penalty_seconds)
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

    async def release(self, token_idx: int) -> None:
        """释放已借出的 token。"""
        self._validate_token_idx(token_idx)
        condition = self._ensure_condition_for_current_loop()
        async with condition:
            state = self._states[token_idx]
            state.in_use = False
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
            # token 永久失效，不唤醒等待者（不会释放出可用 token）

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

    async def snapshot(self) -> list[dict[str, object]]:
        """返回状态快照，用于日志与测试。"""
        condition = self._ensure_condition_for_current_loop()
        async with condition:
            return [
                {
                    "token_idx": idx,
                    "in_use": state.in_use,
                    "invalid": state.invalid,
                    "available_at": state.available_at,
                    "rate_limited_count": state.rate_limited_count,
                    "health_penalty_seconds": self._health_penalty_seconds,
                    "last_error": state.last_error,
                }
                for idx, state in enumerate(self._states)
            ]

    async def metrics(self) -> dict[str, float]:
        """返回 Token 池观测指标。"""
        condition = self._ensure_condition_for_current_loop()
        async with condition:
            return dict(self._metrics)

    async def earliest_available_delay(self) -> float | None:
        """返回距离最早可恢复 token 的等待秒数。

        当没有“空闲且处于恢复期”的 token 时返回 None。
        """
        condition = self._ensure_condition_for_current_loop()
        async with condition:
            now = self._time_fn()
            delay = self._compute_wait_seconds(now)
            return None if delay is None else max(0.0, delay)

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
