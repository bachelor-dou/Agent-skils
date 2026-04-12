"""
GitHub API 封装
===============
封装 GitHub REST / GraphQL API 的底层调用，包括：
  - 仓库搜索（Search API）
  - Star 范围自动分段
  - Stargazers 分页查询（REST，返回 starred_at）
  - Stargazers 批量查询（GraphQL，游标翻页）

所有函数均为线程安全（通过 TokenManager 管理并发 token）。
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

logger = logging.getLogger("discover_hot")

# ──────────────────────────────────────────────────────────────
# Star 范围自动分段 — 常量
# ──────────────────────────────────────────────────────────────
_GITHUB_SEARCH_MAX_PER_QUERY = 1000  # GitHub Search API 单次查询结果上限
_SEGMENT_MAX_RESULTS = 800            # 单个子区间允许的最大结果数（留余量）
_SEGMENT_MIN_STAR_SPAN = 50           # 最小星数跨度，避免过度细分


# ══════════════════════════════════════════════════════════════
# 仓库搜索
# ══════════════════════════════════════════════════════════════


def search_github_repos(
    token_mgr: TokenManager,
    query: str,
    page: int = 1,
    per_page: int = 100,
    sort: str = "stars",
    order: str = "desc",
    auto_star_filter: bool = True,
) -> list[dict]:
    """
    调用 GitHub Search API 搜索仓库（线程安全，3 次重试）。

    Args:
        token_mgr:        TokenManager 实例
        query:            搜索关键词或完整查询字符串
        page:             页码（1-based）
        per_page:         每页结果数（最大 100）
        sort:             排序字段 ("stars" | "updated" | "forks")
        order:            排序方向 ("desc" | "asc")
        auto_star_filter: 是否自动追加 stars:>={MIN_STAR_FILTER}

    Returns:
        仓库列表 [{"full_name": ..., "stargazers_count": ..., ...}, ...]
    """
    q = f"{query} stars:>={MIN_STAR_FILTER}" if auto_star_filter else query
    url = "https://api.github.com/search/repositories"
    params = {"q": q, "sort": sort, "order": order, "per_page": per_page, "page": page}

    for attempt in range(3):
        token_idx = token_mgr.acquire_token()
        try:
            resp = requests.get(
                url, headers=token_mgr.get_rest_headers(token_idx),
                params=params, timeout=30,
            )
            if resp.status_code == 200:
                token_mgr.release_token(token_idx, resp)
                try:
                    return resp.json().get("items", [])
                except (ValueError, KeyError):
                    logger.error(f"搜索响应 JSON 解析失败: query='{q}', page={page}")
                    return []
            elif resp.status_code in (403, 429):
                token_mgr.handle_rate_limit(resp, token_idx)
                continue
            elif resp.status_code == 422:
                token_mgr.release_token(token_idx)
                logger.warning(f"搜索参数无效: query='{q}', page={page}, status=422")
                return []
            else:
                token_mgr.release_token(token_idx)
                logger.warning(f"搜索异常: query='{q}', status={resp.status_code}")
                time.sleep(5)
        except requests.RequestException as e:
            token_mgr.release_token(token_idx)
            logger.error(f"搜索请求异常: query='{q}', error={e}")
            time.sleep(5)

    logger.warning(f"搜索 '{q}' page={page} 经 3 次重试仍失败，跳过。")
    return []


# ══════════════════════════════════════════════════════════════
# Star 范围自动分段
# ══════════════════════════════════════════════════════════════


def get_search_total_count(token_mgr: TokenManager, query: str) -> int:
    """
    获取搜索查询的 total_count（不拉取 items），
    用于判断区间内仓库数是否超过 API 返回上限。
    """
    url = "https://api.github.com/search/repositories"
    params = {"q": query, "per_page": 1, "page": 1}

    for attempt in range(3):
        token_idx = token_mgr.acquire_token()
        try:
            resp = requests.get(
                url, headers=token_mgr.get_rest_headers(token_idx),
                params=params, timeout=30,
            )
            if resp.status_code == 200:
                token_mgr.release_token(token_idx, resp)
                try:
                    return resp.json().get("total_count", 0)
                except (ValueError, KeyError):
                    logger.error(f"total_count 响应 JSON 解析失败: query='{query}'")
                    return 0
            elif resp.status_code in (403, 429):
                token_mgr.handle_rate_limit(resp, token_idx)
                continue
            else:
                token_mgr.release_token(token_idx)
                time.sleep(3)
        except requests.RequestException:
            token_mgr.release_token(token_idx)
            time.sleep(3)

    logger.warning(f"获取 total_count 失败: query='{query}'，视为 0。")
    return 0


def auto_split_star_range(
    token_mgr: TokenManager,
    low: int,
    high: int,
    max_results: int = _SEGMENT_MAX_RESULTS,
    min_span: int = _SEGMENT_MIN_STAR_SPAN,
) -> list[tuple[int, int]]:
    """
    递归自动分段：将 [low, high] 星数范围拆成若干子区间，
    使每个子区间的 total_count <= max_results。

    策略：
      1. 查询区间 total_count，若 <= max_results → 不再细分
      2. 若 > max_results 且跨度 > min_span → 从中点劈开，递归
      3. 若跨度 <= min_span → 不再细分（防止无限递归）
    """
    if high - low <= min_span:
        return [(low, high)]

    query = f"stars:{low}..{high}"
    total = get_search_total_count(token_mgr, query)
    time.sleep(SEARCH_REQUEST_INTERVAL)

    if total <= max_results:
        logger.info(f"  区间 stars:{low}..{high} → total_count={total}，无需细分。")
        return [(low, high)]

    mid = (low + high) // 2
    logger.info(
        f"  区间 stars:{low}..{high} → total_count={total}，"
        f"细分 → [{low}..{mid}] + [{mid + 1}..{high}]"
    )
    left = auto_split_star_range(token_mgr, low, mid, max_results, min_span)
    right = auto_split_star_range(token_mgr, mid + 1, high, max_results, min_span)
    return left + right


# ══════════════════════════════════════════════════════════════
# REST Stargazers 分页查询
# ══════════════════════════════════════════════════════════════


def get_stargazers_page(
    token_mgr: TokenManager,
    owner: str,
    repo: str,
    page: int,
    per_page: int = 100,
) -> list[dict] | None:
    """
    获取指定仓库 stargazers 的第 page 页（线程安全，3 次重试）。

    使用 star media type，返回含 starred_at 时间戳的列表。
    stargazers 按 starred_at 升序排列（page 1 最老，page N 最新）。

    Returns:
        [{"starred_at": "2026-04-01T...", "user": {...}}, ...] 或 None（失败/不可访问）
    """
    url = f"https://api.github.com/repos/{owner}/{repo}/stargazers"
    params = {"per_page": per_page, "page": page}

    for attempt in range(3):
        token_idx = token_mgr.acquire_token()
        try:
            resp = requests.get(
                url, headers=token_mgr.get_star_headers(token_idx),
                params=params, timeout=30,
            )
            if resp.status_code == 200:
                token_mgr.release_token(token_idx, resp)
                try:
                    return resp.json()
                except ValueError:
                    logger.error(f"stargazers 响应 JSON 解析失败: {owner}/{repo} page={page}")
                    return None
            elif resp.status_code in (403, 429):
                token_mgr.handle_rate_limit(resp, token_idx)
                continue
            elif resp.status_code == 422:
                # 超大仓库分页限制，返回 None 触发降级
                token_mgr.release_token(token_idx)
                return None
            else:
                token_mgr.release_token(token_idx)
                logger.debug(
                    f"stargazers 请求失败: {owner}/{repo} page={page}, "
                    f"status={resp.status_code}"
                )
                time.sleep(2)
        except requests.RequestException as e:
            token_mgr.release_token(token_idx)
            logger.debug(f"stargazers 请求异常: {owner}/{repo} page={page}, {e}")
            time.sleep(2)

    return None


# ══════════════════════════════════════════════════════════════
# GraphQL Stargazers 批量查询
# ══════════════════════════════════════════════════════════════


def graphql_stargazers_batch(
    token_mgr: TokenManager,
    owner: str,
    repo: str,
    last: int = 100,
    before: str | None = None,
) -> tuple[list[datetime], str | None]:
    """
    单次 GraphQL 请求获取一批 stargazers（从最新往前翻页）。

    使用 GraphQL 变量参数化查询，避免注入风险。
    通过 last + before 游标实现从新到老翻页。

    Args:
        token_mgr: TokenManager 实例
        owner:     仓库所有者
        repo:      仓库名
        last:      每批获取条数（最大 100）
        before:    上一页的游标（None 表示从最新开始）

    Returns:
        (timestamps 列表, 上一页游标)
        timestamps 按时间升序排列（单批次内有序，跨批次需 .sort()）
        失败时返回 ([], None)
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

    for attempt in range(3):
        token_idx = token_mgr.acquire_token()
        try:
            resp = requests.post(
                "https://api.github.com/graphql",
                headers=token_mgr.get_graphql_headers(token_idx),
                json={"query": query_str, "variables": variables},
                timeout=30,
            )
            if resp.status_code == 200:
                token_mgr.release_token(token_idx, resp)
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
            elif resp.status_code in (403, 429):
                token_mgr.handle_rate_limit(resp, token_idx)
                continue
            else:
                token_mgr.release_token(token_idx)
                time.sleep(3)
        except requests.RequestException:
            token_mgr.release_token(token_idx)
            time.sleep(3)

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
