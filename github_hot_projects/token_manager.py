"""
GitHub Token 轮换管理器
=======================
管理多个 GitHub Personal Access Token 的轮换、限流恢复与并行分配。

设计要点：
  - round-robin 轮换：每次请求后自动切到下一个 token
  - 限流感知：读取 X-RateLimit-Remaining，剩余 <10 时主动标记
  - 403 恢复：记录 Reset 时间，等待最早恢复的 token
  - 线程安全：Lock 保护 current_idx / reset_times 的读写
"""

import logging
import sys
import time
import threading
from datetime import datetime

import requests

from .config import GITHUB_TOKENS

logger = logging.getLogger("discover_hot")


class TokenManager:
    """
    GitHub Token 轮换管理器（线程安全）。

    使用方式::

        mgr = TokenManager()
        idx = mgr.acquire_token()
        resp = requests.get(url, headers=mgr.get_rest_headers(idx))
        mgr.release_token(idx, resp)
    """

    def __init__(self) -> None:
        self.tokens: list[str] = [t.strip() for t in GITHUB_TOKENS if t and t.strip()]
        if not self.tokens:
            logger.error("未配置任何 GitHub Token，无法运行。请设置 GITHUB_TOKENS 环境变量。")
            sys.exit(1)

        self._lock = threading.Lock()
        self.current_idx: int = 0
        # 每个 token 的限流恢复 Unix 时间戳（0 表示未被限流）
        self.reset_times: dict[int, float] = {i: 0.0 for i in range(len(self.tokens))}
        logger.info(f"TokenManager 初始化: 共 {len(self.tokens)} 个 token 可用。")

    # ────────── 获取 / 归还 token ──────────

    def acquire_token(self) -> int:
        """获取当前可用的 token 索引（线程安全）。"""
        with self._lock:
            return self.current_idx

    def release_token(self, token_idx: int, response: requests.Response | None = None) -> None:
        """归还 token 并根据响应头更新 rate limit 信息。"""
        with self._lock:
            self.current_idx = token_idx
            if response is not None:
                self._update_rate_info_locked(response)

    # ────────── 请求头构建 ──────────

    def get_rest_headers(self, token_idx: int | None = None) -> dict[str, str]:
        """REST API 通用请求头。"""
        idx = token_idx if token_idx is not None else self.current_idx
        return {
            "Authorization": f"token {self.tokens[idx]}",
            "Accept": "application/vnd.github.v3+json",
        }

    def get_star_headers(self, token_idx: int | None = None) -> dict[str, str]:
        """REST stargazers 请求头（返回 starred_at 时间戳）。"""
        idx = token_idx if token_idx is not None else self.current_idx
        return {
            "Authorization": f"token {self.tokens[idx]}",
            "Accept": "application/vnd.github.v3.star+json",
        }

    def get_graphql_headers(self, token_idx: int | None = None) -> dict[str, str]:
        """GraphQL API 请求头。"""
        idx = token_idx if token_idx is not None else self.current_idx
        return {
            "Authorization": f"bearer {self.tokens[idx]}",
            "Content-Type": "application/json",
        }

    # ────────── 限流处理 ──────────

    def update_rate_info(self, response: requests.Response) -> None:
        """从响应头更新 rate limit 信息（线程安全）。"""
        with self._lock:
            self._update_rate_info_locked(response)

    def handle_rate_limit(self, response: requests.Response, token_idx: int | None = None) -> None:
        """
        处理 HTTP 403 限流响应。

        记录当前 token 的 Reset 时间并尝试切换到下一个可用 token；
        若所有 token 均被限流，则阻塞等待最早恢复的 token。
        """
        with self._lock:
            if token_idx is not None:
                self.current_idx = token_idx
            reset_str = response.headers.get("X-RateLimit-Reset")
            if reset_str:
                self.reset_times[self.current_idx] = float(reset_str)
            logger.warning(f"Token#{self.current_idx} 触发 403 限流。")
            self._switch_token_locked()

    # ────────── 内部方法（调用时已持有 _lock）──────────

    def _update_rate_info_locked(self, response: requests.Response) -> None:
        """更新 rate limit 并执行 round-robin 轮换。"""
        remaining_str = response.headers.get("X-RateLimit-Remaining")
        reset_str = response.headers.get("X-RateLimit-Reset")
        if remaining_str is not None:
            remaining = int(remaining_str)
            if remaining < 10:
                logger.warning(f"Token#{self.current_idx} 剩余请求数={remaining}，标记限流。")
                if reset_str:
                    self.reset_times[self.current_idx] = float(reset_str)
        self._rotate_token_locked()

    def _rotate_token_locked(self) -> None:
        """round-robin 轮换到下一个未被限流的 token。"""
        if len(self.tokens) <= 1:
            return
        prev_idx = self.current_idx
        now = time.time()
        # 依次尝试后续 token
        for offset in range(1, len(self.tokens)):
            candidate = (self.current_idx + offset) % len(self.tokens)
            if self.reset_times.get(candidate, 0) <= now:
                self.current_idx = candidate
                break
        if self.current_idx != prev_idx:
            logger.debug(f"Token 轮换: #{prev_idx} → #{self.current_idx}")

    def _switch_token_locked(self) -> None:
        """切换到下一个可用 token；全部耗尽时进入等待。"""
        now = time.time()
        for offset in range(1, len(self.tokens)):
            candidate = (self.current_idx + offset) % len(self.tokens)
            if self.reset_times.get(candidate, 0) <= now:
                self.current_idx = candidate
                logger.info(f"已切换到 Token#{self.current_idx}")
                return
        # 所有 token 均被限流 → 等待恢复
        self._wait_for_reset()

    def _wait_for_reset(self) -> None:
        """
        所有 token 均被限流时，释放锁后 sleep 到最早恢复时间。

        注意：此方法在持有 _lock 时被调用，
        sleep 前释放锁以避免阻塞其他线程，醒来后重新获取锁。
        """
        while True:
            now = time.time()
            earliest_idx = min(self.reset_times, key=self.reset_times.get)  # type: ignore[arg-type]
            earliest_reset = self.reset_times[earliest_idx]
            remaining = earliest_reset - now + 5  # +5s 余量

            if remaining <= 0:
                # 已有 token 恢复
                for idx, rst in self.reset_times.items():
                    if rst <= now:
                        self.current_idx = idx
                        logger.info(f"Token#{idx} 限流已恢复，继续执行。")
                        return

            # 释放锁后 sleep，避免阻塞其他线程
            self._lock.release()
            try:
                logger.info(
                    f"所有 {len(self.tokens)} 个 token 均被限流。"
                    f"Token#{earliest_idx} 预计 {int(remaining)}s 后恢复 "
                    f"({datetime.fromtimestamp(earliest_reset).strftime('%H:%M:%S')})，等待中..."
                )
                sleep_chunk = min(max(remaining, 1), 30)
                time.sleep(sleep_chunk)
            finally:
                self._lock.acquire()
