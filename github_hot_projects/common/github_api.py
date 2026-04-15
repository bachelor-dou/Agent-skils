"""
GitHub API 封装
===============
封装 GitHub REST / GraphQL API 的底层调用，包括：
  - 仓库搜索（Search API）
  - Star 范围自动分段
  - Stargazers 分页查询（REST，返回 starred_at）
  - Stargazers 批量查询（GraphQL，游标翻页）

所有函数接收 token_idx 参数（由 Worker 绑定），不再内部 acquire/release。
限流（403/429）和 Token 失效（401）通过抛异常交由 Worker 处理。
"""

import logging
import time
from datetime import datetime, timezone

import requests

from .config import (
    MIN_STAR_FILTER,
    SEARCH_REQUEST_INTERVAL,
)
from .token_manager import TokenManager
from .exceptions import RateLimitError, TokenInvalidError

logger = logging.getLogger("discover_hot")

# ──────────────────────────────────────────────────────────────
# Star 范围自动分段 — 常量
# ──────────────────────────────────────────────────────────────
_GITHUB_SEARCH_MAX_PER_QUERY = 1000  # GitHub Search API 单次查询结果上限
_SEGMENT_MAX_RESULTS = 800            # 单个子区间允许的最大结果数（留余量）
_SEGMENT_MIN_STAR_SPAN = 50           # 最小星数跨度，避免过度细分


# ══════════════════════════════════════════════════════════════
# 内部工具：检查响应状态并抛异常
# ══════════════════════════════════════════════════════════════


def _check_response(resp: requests.Response, token_idx: int) -> None:
    """检查响应状态码，401/403/429 抛出对应异常。"""
    if resp.status_code == 401:
        raise TokenInvalidError(token_idx, f"HTTP 401: {resp.text[:200]}")
    if resp.status_code in (403, 429):
        reset_str = resp.headers.get("X-RateLimit-Reset", "0")
        raise RateLimitError(token_idx, float(reset_str))


# ══════════════════════════════════════════════════════════════
# 仓库搜索
# ══════════════════════════════════════════════════════════════


def search_github_repos(
    token_mgr: TokenManager,
    query: str,
    token_idx: int,
    page: int = 1,
    per_page: int = 100,
    sort: str = "stars",
    order: str = "desc",
    auto_star_filter: bool = True,
) -> list[dict] | None:
    """
    调用 GitHub Search API 搜索仓库（3 次重试）。

    Returns:
        仓库列表，成功但无数据返回 []，3 次网络异常全失败返回 None。

    Raises:
        TokenInvalidError: Token 失效 (401)
        RateLimitError:    Token 限流 (403/429)
    """
    q = f"{query} stars:>={MIN_STAR_FILTER}" if auto_star_filter else query
    url = "https://api.github.com/search/repositories"
    params = {"q": q, "sort": sort, "order": order, "per_page": per_page, "page": page}
    headers = token_mgr.get_rest_headers(token_idx)

    for attempt in range(3):
        try:
            resp = requests.get(url, headers=headers, params=params, timeout=30)
            _check_response(resp, token_idx)
            if resp.status_code == 200:
                try:
                    return resp.json().get("items", [])
                except (ValueError, KeyError):
                    logger.error(f"搜索响应 JSON 解析失败: query='{q}', page={page}, attempt={attempt + 1}")
                    continue
            elif resp.status_code == 422:
                logger.warning(f"搜索参数无效: query='{q}', page={page}, status=422")
                return []
            else:
                logger.warning(f"搜索异常: query='{q}', status={resp.status_code}")
                time.sleep(5 * 2 ** attempt)
        except (TokenInvalidError, RateLimitError):
            raise
        except requests.RequestException as e:
            logger.error(f"搜索请求异常: query='{q}', error={e}")
            time.sleep(5 * 2 ** attempt)

    logger.warning(f"搜索 '{q}' page={page} 经 3 次重试仍失败，跳过。")
    return None


# ══════════════════════════════════════════════════════════════
# Star 范围自动分段
# ══════════════════════════════════════════════════════════════


def get_search_total_count(token_mgr: TokenManager, query: str, token_idx: int) -> int:
    """
    获取搜索查询的 total_count（不拉取 items）。

    Raises:
        TokenInvalidError, RateLimitError
    """
    url = "https://api.github.com/search/repositories"
    params = {"q": query, "per_page": 1, "page": 1}
    headers = token_mgr.get_rest_headers(token_idx)

    for attempt in range(3):
        try:
            resp = requests.get(url, headers=headers, params=params, timeout=30)
            _check_response(resp, token_idx)
            if resp.status_code == 200:
                try:
                    return resp.json().get("total_count", 0)
                except (ValueError, KeyError):
                    logger.error(f"total_count 响应 JSON 解析失败: query='{query}', attempt={attempt + 1}")
                    continue
            else:
                time.sleep(3 * 2 ** attempt)
        except (TokenInvalidError, RateLimitError):
            raise
        except requests.RequestException:
            time.sleep(3 * 2 ** attempt)

    logger.warning(f"获取 total_count 失败: query='{query}'，视为 0。")
    return 0


def auto_split_star_range(
    token_mgr: TokenManager,
    low: int,
    high: int,
    token_idx: int,
    max_results: int = _SEGMENT_MAX_RESULTS,
    min_span: int = _SEGMENT_MIN_STAR_SPAN,
    extra_query: str = "",
) -> list[tuple[int, int]]:
    """
    递归自动分段：将 [low, high] 星数范围拆成若干子区间，
    使每个子区间的 total_count <= max_results。

    在 WorkerPool 启动前由主线程调用，使用固定 token_idx。

    Args:
        extra_query: 附加查询条件（如 "created:>=2026-03-10"），会与 stars 条件合并

    Raises:
        TokenInvalidError, RateLimitError（主线程需处理）
    """
    if high - low <= min_span:
        return [(low, high)]

    query = f"stars:{low}..{high}"
    if extra_query:
        query = f"{query} {extra_query}"
    total = get_search_total_count(token_mgr, query, token_idx)
    time.sleep(SEARCH_REQUEST_INTERVAL)

    if total <= max_results:
        logger.info(f"  区间 stars:{low}..{high} → total_count={total}，无需细分。")
        return [(low, high)]

    mid = (low + high) // 2
    logger.info(
        f"  区间 stars:{low}..{high} → total_count={total}，"
        f"细分 → [{low}..{mid}] + [{mid + 1}..{high}]"
    )
    left = auto_split_star_range(token_mgr, low, mid, token_idx, max_results, min_span, extra_query)
    right = auto_split_star_range(token_mgr, mid + 1, high, token_idx, max_results, min_span, extra_query)
    return left + right


# ══════════════════════════════════════════════════════════════
# REST Stargazers 分页查询
# ══════════════════════════════════════════════════════════════


def get_stargazers_page(
    token_mgr: TokenManager,
    owner: str,
    repo: str,
    page: int,
    token_idx: int,
    per_page: int = 100,
) -> list[dict] | None:
    """
    获取指定仓库 stargazers 的第 page 页（3 次重试）。

    Returns:
        [{"starred_at": ..., "user": {...}}, ...] 或 None（失败/不可访问）

    Raises:
        TokenInvalidError, RateLimitError
    """
    url = f"https://api.github.com/repos/{owner}/{repo}/stargazers"
    params = {"per_page": per_page, "page": page}
    headers = token_mgr.get_star_headers(token_idx)

    for attempt in range(3):
        try:
            resp = requests.get(url, headers=headers, params=params, timeout=30)
            _check_response(resp, token_idx)
            if resp.status_code == 200:
                try:
                    return resp.json()
                except ValueError:
                    logger.error(f"stargazers 响应 JSON 解析失败: {owner}/{repo} page={page}")
                    return None
            elif resp.status_code == 422:
                return None
            else:
                logger.debug(
                    f"stargazers 请求失败: {owner}/{repo} page={page}, "
                    f"status={resp.status_code}"
                )
                time.sleep(2 * 2 ** attempt)
        except (TokenInvalidError, RateLimitError):
            raise
        except requests.RequestException as e:
            logger.debug(f"stargazers 请求异常: {owner}/{repo} page={page}, {e}")
            time.sleep(2 * 2 ** attempt)

    return None


# ══════════════════════════════════════════════════════════════
# GraphQL Stargazers 批量查询
# ══════════════════════════════════════════════════════════════


def graphql_stargazers_batch(
    token_mgr: TokenManager,
    owner: str,
    repo: str,
    token_idx: int,
    last: int = 100,
    before: str | None = None,
) -> tuple[list[datetime], str | None]:
    """
    单次 GraphQL 请求获取一批 stargazers（从最新往前翻页）。

    Raises:
        TokenInvalidError, RateLimitError
    """
    query_str = """
    query($owner: String!, $name: String!, $last: Int!, $before: String) {
      repository(owner: $owner, name: $name) {
        stargazers(last: $last, orderBy: {field: STARRED_AT, direction: ASC}, before: $before) {
          edges {
            starredAt
            cursor
          }
        }
      }
    }
    """
    variables: dict = {"owner": owner, "name": repo, "last": last}
    if before:
        variables["before"] = before

    headers = token_mgr.get_graphql_headers(token_idx)

    for attempt in range(3):
        try:
            resp = requests.post(
                "https://api.github.com/graphql",
                headers=headers,
                json={"query": query_str, "variables": variables},
                timeout=30,
            )
            _check_response(resp, token_idx)
            if resp.status_code == 200:
                try:
                    data = resp.json()
                except ValueError:
                    logger.error(f"GraphQL 响应 JSON 解析失败: {owner}/{repo}")
                    return [], None

                if "errors" in data:
                    logger.warning(f"GraphQL 返回错误: {owner}/{repo}, {data['errors']}")
                    return [], None

                repo_data = data.get("data", {}).get("repository")
                if not repo_data:
                    return [], None

                edges = repo_data.get("stargazers", {}).get("edges", [])
                timestamps: list[datetime] = []
                first_cursor: str | None = None

                for e in edges:
                    t = _parse_starred_at(e.get("starredAt", ""))
                    if t:
                        timestamps.append(t)
                    if first_cursor is None:
                        first_cursor = e.get("cursor")

                return timestamps, first_cursor
            else:
                time.sleep(3 * 2 ** attempt)
        except (TokenInvalidError, RateLimitError):
            raise
        except requests.RequestException:
            time.sleep(3 * 2 ** attempt)

    return [], None


# ──────────────────────────────────────────────────────────────
# 工具函数
# ──────────────────────────────────────────────────────────────


def _parse_starred_at(ts: str) -> datetime | None:
    """解析 starred_at 时间戳字符串为 UTC datetime。"""
    if not ts:
        return None
    try:
        return datetime.fromisoformat(ts.replace("Z", "+00:00"))
    except (ValueError, AttributeError):
        return None


def parse_starred_at_from_entry(entry: dict) -> datetime | None:
    """解析 REST stargazer 条目中的 starred_at 时间戳。"""
    return _parse_starred_at(entry.get("starred_at", ""))
