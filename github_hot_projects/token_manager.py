"""
GitHub Token 管理器
====================
管理多个 GitHub Personal Access Token，提供请求头构建。

Token 与 Worker 绑定（一对一），不再做内部轮换/限流。
限流和失效由 WorkerPool 的 Worker 自行处理。
"""

import logging
import sys

from .config import GITHUB_TOKENS

logger = logging.getLogger("discover_hot")


class TokenManager:
    """
    GitHub Token 管理器。

    职责：
      - 持有 token 列表
      - 根据 token_idx 构建请求头

    使用方式::

        mgr = TokenManager()
        headers = mgr.get_rest_headers(token_idx=0)
        resp = requests.get(url, headers=headers)
    """

    def __init__(self) -> None:
        self.tokens: list[str] = [t.strip() for t in GITHUB_TOKENS if t and t.strip()]
        if not self.tokens:
            logger.error("未配置任何 GitHub Token，无法运行。请设置 GITHUB_TOKENS 环境变量。")
            sys.exit(1)
        logger.info(f"TokenManager 初始化: 共 {len(self.tokens)} 个 token 可用。")

    # ────────── 请求头构建 ──────────

    def get_rest_headers(self, token_idx: int) -> dict[str, str]:
        """REST API 通用请求头。"""
        return {
            "Authorization": f"token {self.tokens[token_idx]}",
            "Accept": "application/vnd.github.v3+json",
        }

    def get_star_headers(self, token_idx: int) -> dict[str, str]:
        """REST stargazers 请求头（返回 starred_at 时间戳）。"""
        return {
            "Authorization": f"token {self.tokens[token_idx]}",
            "Accept": "application/vnd.github.v3.star+json",
        }

    def get_graphql_headers(self, token_idx: int) -> dict[str, str]:
        """GraphQL API 请求头。"""
        return {
            "Authorization": f"bearer {self.tokens[token_idx]}",
            "Content-Type": "application/json",
        }
