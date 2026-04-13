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
        self._condition = threading.Condition(self._lock)
        self.current_idx: int = 0
        # 每个 token 的限流恢复 Unix 时间戳（0 表示未被限流）
        self.reset_times: dict[int, float] = {i: 0.0 for i in range(len(self.tokens))}
        logger.info(f"TokenManager 初始化: 共 {len(self.tokens)} 个 token 可用。")

    # ────────── 获取 / 归还 token ──────────

    def acquire_token(self) -> int:
        """
        获取可用 token 索引（线程安全，round-robin + 限流感知）。

        每次调用返回不同 token（round-robin 递增），被限流的 token 自动跳过。
        所有 token 均被限流时，阻塞等待最早恢复的 token 再返回。
        """
        with self._condition:
            while True:
                now = time.time()
                for offset in range(len(self.tokens)):
                    candidate = (self.current_idx + offset) % len(self.tokens)
                    if self.reset_times.get(candidate, 0) <= now:
                        # 找到可用 token，推进 current_idx 供下次调用
                        self.current_idx = (candidate + 1) % len(self.tokens)
                        return candidate
                # 所有 token 被限流 → 等待最早恢复
                earliest_idx = min(self.reset_times, key=self.reset_times.get)  # type: ignore[arg-type]
                earliest_reset = self.reset_times[earliest_idx]
                remaining = earliest_reset - now + 5  # +5s 余量
                if remaining <= 0:
                    continue
                logger.info(
                    f"acquire_token: 所有 {len(self.tokens)} 个 token 限流，"
                    f"等待 {int(remaining)}s 后恢复 "
                    f"({datetime.fromtimestamp(earliest_reset).strftime('%H:%M:%S')})"
                )
                self._condition.wait(timeout=min(remaining, 30))

    def release_token(self, token_idx: int, response: requests.Response | None = None) -> None:
        """归还 token 并根据响应头更新指定 token 的 rate limit 信息。"""
        with self._lock:
            if response is not None:
                self._update_rate_info_locked(token_idx, response)

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

    def update_rate_info(self, response: requests.Response, token_idx: int | None = None) -> None:
        """从响应头更新 rate limit 信息（线程安全）。"""
        with self._lock:
            idx = token_idx if token_idx is not None else self.current_idx
            self._update_rate_info_locked(idx, response)

    def handle_rate_limit(self, response: requests.Response, token_idx: int | None = None) -> None:
        """
        处理 HTTP 403 限流响应。

        记录被限流 token 的 Reset 时间戳。下次 acquire_token 会自动
        跳过该 token 或等待恢复，无需在此处切换/阻塞。
        """
        with self._lock:
            idx = token_idx if token_idx is not None else self.current_idx
            reset_str = response.headers.get("X-RateLimit-Reset")
            if reset_str:
                self.reset_times[idx] = float(reset_str)
            logger.warning(f"Token#{idx} 触发 403 限流。")

    # ────────── 内部方法（调用时已持有 _lock）──────────

    def _update_rate_info_locked(self, token_idx: int, response: requests.Response) -> None:
        """更新指定 token 的 rate limit 信息（调用时已持有 _lock）。"""
        remaining_str = response.headers.get("X-RateLimit-Remaining")
        reset_str = response.headers.get("X-RateLimit-Reset")
        if remaining_str is not None:
            remaining = int(remaining_str)
            if remaining < 10:
                logger.warning(f"Token#{token_idx} 剩余请求数={remaining}，标记限流。")
                if reset_str:
                    self.reset_times[token_idx] = float(reset_str)
