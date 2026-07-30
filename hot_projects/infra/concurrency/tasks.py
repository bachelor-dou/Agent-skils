"""
Task 子类定义
==============
定义搜索 / 扫描 / Trending 相关的 Task 子类，即所有需要发 HTTP 请求的并行工作。

Task 子类（继承 task_base.Task）由 tools/basic/core.py 中的能力函数创建并提交到 AsyncTaskDispatcher。
增长计算不在此列：它是纯本地算术（快照减法 / DB 差值），见 task_help._resolve_growth。
"""

import logging
import time
from dataclasses import dataclass, field
from typing import Any

from ...config import MIN_STAR
from ..exceptions import RateLimitError, TokenInvalidError
from ...datasource.github.api import (
    SEARCH_REQUEST_INTERVAL,
    search_github_repos,
    async_search_github_repos,
)
from ...datasource.github.trending import fetch_trending
from .task_base import Task

logger = logging.getLogger("hot_projects")


def _remaining_pages_from(pages: list[int], current_page: int) -> list[int]:
    return [page for page in pages if page >= current_page]


def _record_token_issue(token_mgr: Any, token_idx: int | None, exc: Exception) -> None:
    if token_idx is None or token_mgr is None:
        return

    if isinstance(exc, RateLimitError):
        recorder = getattr(token_mgr, "record_rate_limited", None)
        if callable(recorder):
            recorder(token_idx, exc.reset_time, str(exc))
        return

    if isinstance(exc, TokenInvalidError):
        # 401 优先走 strikes/冷却（与异步一致），避免瞬时 401 永久踢除有效 token。
        recorder = getattr(token_mgr, "record_auth_failed", None)
        if callable(recorder):
            recorder(token_idx, str(exc))
            return
        recorder = getattr(token_mgr, "record_invalid", None)
        if callable(recorder):
            recorder(token_idx, str(exc))


async def _record_token_issue_async(token_mgr: Any, token_idx: int | None, exc: Exception) -> None:
    if token_idx is None or token_mgr is None:
        return

    if isinstance(exc, RateLimitError):
        marker = getattr(token_mgr, "mark_rate_limited", None)
        if callable(marker):
            await marker(token_idx, exc.reset_time, str(exc))
            return
        _record_token_issue(token_mgr, token_idx, exc)
        return

    if isinstance(exc, TokenInvalidError):
        # 401 优先走 strikes/冷却（与增长阶段一致），避免瞬时 401 永久踢除有效 token；
        # 仅当池不支持该接口时回退到永久失效。
        marker = getattr(token_mgr, "mark_auth_failed", None)
        if callable(marker):
            await marker(token_idx, str(exc))
            return
        marker = getattr(token_mgr, "mark_invalid", None)
        if callable(marker):
            await marker(token_idx, str(exc))
            return
        _record_token_issue(token_mgr, token_idx, exc)


# ══════════════════════════════════════════════════════════════
# Task 子类定义（搜索 / 扫描 / 增长计算）
# ══════════════════════════════════════════════════════════════


@dataclass
class KeywordSearchTask(Task):
    """关键词搜索任务：搜索单个关键词的多页结果。"""

    # 类常量：每个关键词搜索的最大页数
    MAX_PAGES: int = 3

    needs_github_token: bool = True  # A模式: True（任务级持有）; B模式: 改回 False（请求级借还）
    keyword: str = ""
    category: str = ""
    keyword_idx: int = 0
    total_keywords: int = 0
    created_after: str = ""
    min_star: int = 0  # 最低 star 过滤阈值，0 则使用默认 MIN_STAR
    page_numbers: list[int] | None = None
    retry_round: int = 0
    _async_http_client: Any = field(default=None, repr=False)
    _raw_repos: dict = field(default=None, repr=False)
    failed_pages: list[int] = field(default_factory=list, init=False, repr=False)

    def execute(self, token_idx: int | None) -> list[dict]:
        self.failed_pages = []
        retry_suffix = f", retry={self.retry_round}" if self.retry_round else ""
        page_suffix = f", pages={self.page_numbers}" if self.page_numbers else ""
        logger.debug(
            f"[{self.keyword_idx}/{self.total_keywords}] 搜索: "
            f"'{self.keyword}' (类别: {self.category}{retry_suffix}{page_suffix})"
        )
        collected: list[dict] = []
        star_threshold = self.min_star if self.min_star else MIN_STAR
        query = self.keyword
        if self.created_after:
            query = f"{query} created:>={self.created_after}"

        pages = self.page_numbers if self.page_numbers is not None else list(range(1, self.MAX_PAGES + 1))
        stop_on_empty = self.page_numbers is None

        for page in pages:
            try:
                items = search_github_repos(
                    self._token_mgr,
                    query,
                    token_idx,
                    page=page,
                    min_star=star_threshold,
                )
            except (RateLimitError, TokenInvalidError) as exc:
                _record_token_issue(self._token_mgr, token_idx, exc)
                remaining_pages = _remaining_pages_from(pages, page)
                self.failed_pages.extend(remaining_pages)
                logger.warning(
                    f"[{self.keyword_idx}/{self.total_keywords}] 搜索: '{self.keyword}' "
                    f"(类别: {self.category}), page={page} 命中 {exc.__class__.__name__}，"
                    f"剩余页转入补偿: {remaining_pages}。"
                )
                return collected
            if items is None:
                self.failed_pages.append(page)
                failure_action = "加入补偿队列" if self.retry_round == 0 else "补偿后仍失败"
                logger.warning(
                    f"[{self.keyword_idx}/{self.total_keywords}] 搜索: '{self.keyword}' "
                    f"(类别: {self.category}), page={page} 连续失败，{failure_action}。"
                )
                continue
            if not items:
                if stop_on_empty:
                    break
                continue
            for repo_item in items:
                full_name = repo_item.get("full_name", "")
                if not full_name:
                    continue
                collected.append({
                    "full_name": full_name,
                    "star": repo_item.get("stargazers_count", 0),
                    "repo_item": repo_item,
                    "created_at": repo_item.get("created_at", ""),
                })
            time.sleep(SEARCH_REQUEST_INTERVAL)

        return collected

    async def execute_async(self, token_idx: int | None) -> list[dict]:
        self.failed_pages = []
        token_suffix = f", token={token_idx}" if token_idx is not None else ""
        retry_suffix = f", retry={self.retry_round}" if self.retry_round else ""
        page_suffix = f", pages={self.page_numbers}" if self.page_numbers else ""
        logger.debug(
            f"[{self.keyword_idx}/{self.total_keywords}] 搜索: "
            f"'{self.keyword}' (类别: {self.category}{token_suffix}{retry_suffix}{page_suffix})"
        )
        collected: list[dict] = []
        star_threshold = self.min_star if self.min_star else MIN_STAR
        query = self.keyword
        if self.created_after:
            query = f"{query} created:>={self.created_after}"

        pages = self.page_numbers if self.page_numbers is not None else list(range(1, self.MAX_PAGES + 1))
        stop_on_empty = self.page_numbers is None

        for page in pages:
            try:
                # A模式：任务级 token 持有。
                items = await async_search_github_repos(
                    self._token_mgr,
                    query,
                    token_idx,
                    page=page,
                    min_star=star_threshold,
                    client=self._async_http_client,
                )
            except (RateLimitError, TokenInvalidError) as exc:
                await _record_token_issue_async(self._token_mgr, token_idx, exc)
                remaining_pages = _remaining_pages_from(pages, page)
                self.failed_pages.extend(remaining_pages)
                logger.warning(
                    f"[{self.keyword_idx}/{self.total_keywords}] 搜索: '{self.keyword}' "
                    f"(类别: {self.category}{token_suffix}), page={page} 命中 {exc.__class__.__name__}，"
                    f"剩余页转入补偿: {remaining_pages}。"
                )
                return collected
            # // B模式：请求级 token 借还备份。
            # // items = await async_search_github_repos(
            # //     self._token_mgr,
            # //     query,
            # //     None,
            # //     page=page,
            # //     min_star=star_threshold,
            # //     client=self._async_http_client,
            # // )
            if items is None:
                self.failed_pages.append(page)
                failure_action = "加入补偿队列" if self.retry_round == 0 else "补偿后仍失败"
                logger.warning(
                    f"[{self.keyword_idx}/{self.total_keywords}] 搜索: '{self.keyword}' "
                    f"(类别: {self.category}{token_suffix}), page={page} 连续失败，{failure_action}。"
                )
                continue
            if not items:
                if stop_on_empty:
                    break
                continue
            for repo_item in items:
                full_name = repo_item.get("full_name", "")
                if not full_name:
                    continue
                collected.append({
                    "full_name": full_name,
                    "star": repo_item.get("stargazers_count", 0),
                    "repo_item": repo_item,
                    "created_at": repo_item.get("created_at", ""),
                })
            # 页间配速已由 async_search_github_repos 内的 token_mgr.throttle_search（按 token、
            # 跨任务延续）统一处理，这里不再逐页 sleep（否则与 throttle 重复）。

        return collected

    def on_result(self, result: list[dict]) -> None:
        if not result or self._raw_repos is None:
            return
        for repo in result:
            fn = repo["full_name"]
            if fn not in self._raw_repos:
                self._raw_repos[fn] = {
                    "star": repo["star"],
                    "repo_item": repo["repo_item"],
                    "created_at": repo["created_at"],
                }

    def __str__(self) -> str:
        return f"KeywordSearch({self.keyword})"


@dataclass
class ScanSegmentTask(Task):
    """Star 区间扫描任务：扫描单个子区间的多页结果。"""

    needs_github_token: bool = True  # A模式: True（任务级持有）; B模式: 改回 False（请求级借还）
    seg_idx: int = 0
    low: int = 0
    high: int = 0
    total_segments: int = 0
    created_after: str = ""
    min_star: int = 0  # 最低 star 过滤阈值，0 则使用默认 MIN_STAR
    page_numbers: list[int] | None = None
    retry_round: int = 0
    _async_http_client: Any = field(default=None, repr=False)
    _raw_repos: dict = field(default=None, repr=False)
    failed_pages: list[int] = field(default_factory=list, init=False, repr=False)

    def execute(self, token_idx: int | None) -> list[dict]:
        self.failed_pages = []
        query = f"stars:{self.low}..{self.high}"
        if self.created_after:
            query = f"{query} created:>={self.created_after}"
        retry_suffix = f", retry={self.retry_round}" if self.retry_round else ""
        page_suffix = f", pages={self.page_numbers}" if self.page_numbers else ""
        logger.debug(
            f"  子区间 {self.seg_idx}/{self.total_segments}: "
            f"{query}{retry_suffix}{page_suffix}"
        )
        collected: list[dict] = []
        star_threshold = self.min_star if self.min_star else MIN_STAR
        pages = self.page_numbers if self.page_numbers is not None else list(range(1, 11))
        stop_on_empty = self.page_numbers is None

        for page in pages:
            try:
                items = search_github_repos(
                    self._token_mgr,
                    query,
                    token_idx,
                    page=page,
                    sort="updated",
                    min_star=0,
                )
            except (RateLimitError, TokenInvalidError) as exc:
                _record_token_issue(self._token_mgr, token_idx, exc)
                remaining_pages = _remaining_pages_from(pages, page)
                self.failed_pages.extend(remaining_pages)
                token_suffix = f", token={token_idx}" if token_idx is not None else ""
                logger.warning(
                    f"  子区间 {self.seg_idx}/{self.total_segments}: {query}, "
                    f"page={page}{token_suffix} 命中 {exc.__class__.__name__}，"
                    f"剩余页转入补偿: {remaining_pages}。"
                )
                return collected
            if items is None:
                self.failed_pages.append(page)
                token_suffix = f", token={token_idx}" if token_idx is not None else ""
                failure_action = "加入补偿队列" if self.retry_round == 0 else "补偿后仍失败"
                logger.warning(
                    f"  子区间 {self.seg_idx}/{self.total_segments}: {query}, "
                    f"page={page}{token_suffix} 连续失败，{failure_action}。"
                )
                continue
            if not items:
                if stop_on_empty:
                    break
                continue
            for repo_item in items:
                full_name = repo_item.get("full_name", "")
                if not full_name:
                    continue
                current_star = repo_item.get("stargazers_count", 0)
                if current_star < star_threshold:
                    continue
                collected.append({
                    "full_name": full_name,
                    "star": current_star,
                    "repo_item": repo_item,
                    "created_at": repo_item.get("created_at", ""),
                })
            time.sleep(SEARCH_REQUEST_INTERVAL)

        return collected

    async def execute_async(self, token_idx: int | None) -> list[dict]:
        self.failed_pages = []
        query = f"stars:{self.low}..{self.high}"
        if self.created_after:
            query = f"{query} created:>={self.created_after}"
        token_suffix = f" (token={token_idx})" if token_idx is not None else ""
        retry_suffix = f", retry={self.retry_round}" if self.retry_round else ""
        page_suffix = f", pages={self.page_numbers}" if self.page_numbers else ""
        logger.debug(
            f"  子区间 {self.seg_idx}/{self.total_segments}: "
            f"{query}{token_suffix}{retry_suffix}{page_suffix}"
        )
        collected: list[dict] = []
        star_threshold = self.min_star if self.min_star else MIN_STAR
        pages = self.page_numbers if self.page_numbers is not None else list(range(1, 11))
        stop_on_empty = self.page_numbers is None

        for page in pages:
            try:
                # A模式：任务级 token 持有。
                items = await async_search_github_repos(
                    self._token_mgr,
                    query,
                    token_idx,
                    page=page,
                    sort="updated",
                    min_star=0,
                    client=self._async_http_client,
                )
            except (RateLimitError, TokenInvalidError) as exc:
                await _record_token_issue_async(self._token_mgr, token_idx, exc)
                remaining_pages = _remaining_pages_from(pages, page)
                self.failed_pages.extend(remaining_pages)
                token_suffix = f", token={token_idx}" if token_idx is not None else ""
                logger.warning(
                    f"  子区间 {self.seg_idx}/{self.total_segments}: {query}, "
                    f"page={page}{token_suffix} 命中 {exc.__class__.__name__}，"
                    f"剩余页转入补偿: {remaining_pages}。"
                )
                return collected
            # // B模式：请求级 token 借还备份。
            # // items = await async_search_github_repos(
            # //     self._token_mgr,
            # //     query,
            # //     None,
            # //     page=page,
            # //     sort="updated",
            # //     min_star=0,
            # //     client=self._async_http_client,
            # // )
            if items is None:
                self.failed_pages.append(page)
                token_suffix = f", token={token_idx}" if token_idx is not None else ""
                failure_action = "加入补偿队列" if self.retry_round == 0 else "补偿后仍失败"
                logger.warning(
                    f"  子区间 {self.seg_idx}/{self.total_segments}: {query}, "
                    f"page={page}{token_suffix} 连续失败，{failure_action}。"
                )
                continue
            if not items:
                if stop_on_empty:
                    break
                continue
            for repo_item in items:
                full_name = repo_item.get("full_name", "")
                if not full_name:
                    continue
                current_star = repo_item.get("stargazers_count", 0)
                if current_star < star_threshold:
                    continue
                collected.append({
                    "full_name": full_name,
                    "star": current_star,
                    "repo_item": repo_item,
                    "created_at": repo_item.get("created_at", ""),
                })
            # 页间配速已由 async_search_github_repos 内的 token_mgr.throttle_search（按 token、
            # 跨任务延续）统一处理，这里不再逐页 sleep（否则与 throttle 重复）。

        return collected

    def on_result(self, result: list[dict]) -> None:
        if not result or self._raw_repos is None:
            return
        for repo in result:
            fn = repo["full_name"]
            if fn not in self._raw_repos:
                self._raw_repos[fn] = {
                    "star": repo["star"],
                    "repo_item": repo["repo_item"],
                    "created_at": repo["created_at"],
                }

    def __str__(self) -> str:
        pages = f", pages={self.page_numbers}" if self.page_numbers is not None else ""
        retry = f", retry={self.retry_round}" if self.retry_round else ""
        return f"ScanSegment({self.low}..{self.high}{pages}{retry})"


@dataclass
class TrendingPeriodTask(Task):
    """Trending 单周期抓取任务：抓取一个 since 周期的榜单。"""

    period: str = "weekly"
    _period_results: dict[str, list[dict]] | None = field(default=None, repr=False)

    def execute(self, token_idx: int | None) -> tuple[str, list[dict]]:
        return self.period, fetch_trending(since=self.period)

    def on_result(self, result: tuple[str, list[dict]]) -> None:
        if self._period_results is None:
            return
        period, repos = result
        self._period_results[period] = repos

    def __str__(self) -> str:
        return f"TrendingPeriod({self.period})"


