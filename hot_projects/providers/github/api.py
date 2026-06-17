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

import base64
import binascii
import asyncio
import logging
import time
from datetime import datetime
from email.utils import parsedate_to_datetime

import requests

try:
    import httpx
except ImportError:  # pragma: no cover - handled at runtime when async path is used
    httpx = None

from ...config import (
    MIN_STAR,
    SEARCH_REQUEST_INTERVAL,
)
from .token_pool import GitHubTokenPool
from ...infra.exceptions import RateLimitError, RetryableError, TokenInvalidError

logger = logging.getLogger("discover_hot")

# stargazers 瞬时故障（网络异常 / 5xx）处理：先原地快速重试，仍失败则抛 RetryableError
# 交由调度器释放 token 并重排队，避免把瞬时故障误当“大仓库”降级到昂贵的 GraphQL 采样，
# 也避免在原地退避 sleep 期间长期占用 token。422（REST 分页上限=超大仓库）仍走采样。
STARGAZER_TRANSIENT_FAST_RETRIES = 2
STARGAZER_REQUEUE_BACKOFF_SECONDS = 2.0

# ──────────────────────────────────────────────────────────────
# Star 范围自动分段 — 常量
# ──────────────────────────────────────────────────────────────
_GITHUB_SEARCH_MAX_PER_QUERY = 1000  # GitHub Search API 单次查询结果上限
_SEGMENT_MAX_RESULTS = 800            # 单个子区间允许的最大结果数（留余量）
_SEGMENT_MIN_STAR_SPAN = 50           # 最小星数跨度，避免过度细分
_UNKNOWN_TOTAL_COUNT_FALLBACK = _GITHUB_SEARCH_MAX_PER_QUERY + 1
_DEFAULT_RATE_LIMIT_COOLDOWN_SECONDS = 60.0


# ══════════════════════════════════════════════════════════════
# 内部工具：检查响应状态并抛异常
# ══════════════════════════════════════════════════════════════


def _resolve_rate_limit_reset(headers) -> float:
    now = time.time()

    reset_str = headers.get("X-RateLimit-Reset")
    if reset_str is not None:
        try:
            reset_time = float(reset_str)
            if reset_time > now:
                return reset_time
        except (TypeError, ValueError):
            pass

    retry_after = headers.get("Retry-After")
    if retry_after:
        try:
            delay_seconds = float(retry_after)
            if delay_seconds > 0:
                return now + delay_seconds
        except (TypeError, ValueError):
            try:
                retry_after_at = parsedate_to_datetime(retry_after).timestamp()
                if retry_after_at > now:
                    return retry_after_at
            except (TypeError, ValueError, IndexError, OverflowError):
                pass

    return now + _DEFAULT_RATE_LIMIT_COOLDOWN_SECONDS


def _check_response(resp: requests.Response, token_idx: int) -> None:
    """检查响应状态码，401/403/429 抛出对应异常。"""
    if resp.status_code == 401:
        raise TokenInvalidError(token_idx, f"HTTP 401: {resp.text[:200]}")
    if resp.status_code in (403, 429):
        raise RateLimitError(token_idx, _resolve_rate_limit_reset(resp.headers))


def _check_response_async(resp, token_idx: int) -> None:
    """异步响应状态检查，保持与同步链路一致的异常语义。"""
    if resp.status_code == 401:
        raise TokenInvalidError(token_idx, f"HTTP 401: {resp.text[:200]}")
    if resp.status_code in (403, 429):
        raise RateLimitError(token_idx, _resolve_rate_limit_reset(resp.headers))


def _build_async_client(timeout_seconds: float = 60.0):
    if httpx is None:
        raise RuntimeError("httpx is required for async GitHub API calls. Install httpx>=0.27.0")
    limits = httpx.Limits(max_connections=100, max_keepalive_connections=100)
    return httpx.AsyncClient(timeout=timeout_seconds, limits=limits)


def build_github_async_client(timeout_seconds: float = 60.0):
    """构建可跨多个 GitHub 请求复用的异步客户端。"""
    return _build_async_client(timeout_seconds=timeout_seconds)


def _format_request_error(err: Exception) -> str:
    parts: list[str] = []

    detail = str(err).strip()
    if detail:
        parts.append(f"{err.__class__.__name__}: {detail}")
    else:
        parts.append(err.__class__.__name__)

    request = getattr(err, "request", None)
    url = getattr(request, "url", None) if request is not None else None
    host = getattr(url, "host", "") if url is not None else ""
    if host:
        parts.append(f"host={host}")

    cause = getattr(err, "__cause__", None) or getattr(err, "__context__", None)
    if cause is not None:
        cause_detail = str(cause).strip()
        cause_text = (
            f"{cause.__class__.__name__}: {cause_detail}"
            if cause_detail
            else cause.__class__.__name__
        )
        if cause_text != parts[0]:
            parts.append(f"cause={cause_text}")

    return ", ".join(parts)


# ══════════════════════════════════════════════════════════════
# 仓库搜索
# ══════════════════════════════════════════════════════════════


def search_github_repos(
    token_mgr: GitHubTokenPool,
    query: str,
    token_idx: int,
    page: int = 1,
    per_page: int = 100,
    sort: str = "stars",
    order: str = "desc",
    min_star: int | None = None,
    worker_idx: int | None = None,
) -> list[dict] | None:
    """
    调用 GitHub Search API 搜索仓库（3 次重试）。

    Args:
        min_star: 最低 star 过滤阈值。
            - None: 使用默认 MIN_STAR
            - > 0: 使用指定值
            - = 0: 不添加 star 过滤（query 已包含范围或查特定仓库）

    Returns:
        仓库列表，成功但无数据返回 []，3 次网络异常全失败返回 None。

    Raises:
        TokenInvalidError: Token 失效 (401)
        RateLimitError:    Token 限流 (403/429)
    """
    if min_star is None:
        star_threshold = MIN_STAR
    elif min_star > 0:
        star_threshold = min_star
    else:
        # min_star <= 0: 不添加 star 过滤
        star_threshold = None

    if star_threshold is not None:
        q = f"{query} stars:>={star_threshold}"
    else:
        q = query
    url = "https://api.github.com/search/repositories"
    params = {"q": q, "sort": sort, "order": order, "per_page": per_page, "page": page}
    headers = token_mgr.get_rest_headers(token_idx)
    caller = f"worker={worker_idx}, token={token_idx}" if worker_idx is not None else f"token={token_idx}"

    for attempt in range(3):
        attempt_no = attempt + 1
        try:
            resp = requests.get(url, headers=headers, params=params, timeout=60)
            _check_response(resp, token_idx)
            if resp.status_code == 200:
                try:
                    return resp.json().get("items", [])
                except (ValueError, KeyError):
                    logger.error(
                        f"搜索响应 JSON 解析失败: query='{q}', {caller}, "
                        f"page={page}, attempt={attempt_no}/3"
                    )
                    continue
            elif resp.status_code == 422:
                logger.warning(
                    f"搜索参数无效: query='{q}', {caller}, page={page}, status=422"
                )
                return []
            else:
                logger.warning(
                    f"搜索异常: query='{q}', {caller}, page={page}, "
                    f"attempt={attempt_no}/3, status={resp.status_code}"
                )
                time.sleep(5 * 2 ** attempt)
        except (TokenInvalidError, RateLimitError):
            raise
        except requests.RequestException as e:
            logger.error(
                f"搜索请求异常: query='{q}', {caller}, page={page}, "
                f"attempt={attempt_no}/3, error={e}"
            )
            time.sleep(5 * 2 ** attempt)

    logger.warning(
        f"搜索 '{q}' {caller}, page={page} 经 3 次重试仍失败，返回 None 由调用方决定后续处理。"
    )
    return None


async def async_search_github_repos(
    token_mgr: GitHubTokenPool,
    query: str,
    token_idx: int | None,
    page: int = 1,
    per_page: int = 100,
    sort: str = "stars",
    order: str = "desc",
    min_star: int | None = None,
    worker_idx: int | None = None,
    client=None,
) -> list[dict] | None:
    """异步调用 GitHub Search API（3 次重试）。

    与同步版本保持相同返回值和异常语义：
      - TokenInvalidError (401)
      - RateLimitError (403/429)

        B模式：请求级 token 借还。
        当 token_idx 为 None 时，表示调用方选择请求级模式：
            1. 每次 HTTP 请求前临时 acquire token
            2. 请求成功或普通异常后立刻 release
            3. 限流/失效在 helper 内写回 token 池状态
    """
    if min_star is None:
        star_threshold = MIN_STAR
    elif min_star > 0:
        star_threshold = min_star
    else:
        star_threshold = None

    if star_threshold is not None:
        q = f"{query} stars:>={star_threshold}"
    else:
        q = query

    url = "https://api.github.com/search/repositories"
    params = {"q": q, "sort": sort, "order": order, "per_page": per_page, "page": page}
    owns_client = client is None
    async_client = client
    if async_client is None:
        async_client = _build_async_client(timeout_seconds=60.0)

    try:
        for attempt in range(3):
            attempt_no = attempt + 1
            attempt_token_idx = token_idx
            borrowed_token = False
            try:
                # B模式入口：调用方未绑定任务级 token 时，在单次请求前临时借用。
                if attempt_token_idx is None:
                    attempt_token_idx = await token_mgr.acquire()
                    borrowed_token = True

                headers = token_mgr.get_rest_headers(attempt_token_idx)
                caller = (
                    f"worker={worker_idx}, token={attempt_token_idx}"
                    if worker_idx is not None
                    else f"token={attempt_token_idx}"
                )
                resp = await async_client.get(url, headers=headers, params=params)
                _check_response_async(resp, attempt_token_idx)

                if resp.status_code == 200:
                    try:
                        # B模式成功路径：请求结束后立即释放 token。
                        if borrowed_token:
                            await token_mgr.release(attempt_token_idx)
                        return resp.json().get("items", [])
                    except (ValueError, KeyError):
                        logger.error(
                            f"异步搜索响应 JSON 解析失败: query='{q}', {caller}, "
                            f"page={page}, attempt={attempt_no}/3"
                        )
                        if borrowed_token:
                            await token_mgr.release(attempt_token_idx)
                        continue

                if resp.status_code == 422:
                    logger.warning(
                        f"异步搜索参数无效: query='{q}', {caller}, page={page}, status=422"
                    )
                    if borrowed_token:
                        await token_mgr.release(attempt_token_idx)
                    return []

                logger.warning(
                    f"异步搜索异常: query='{q}', {caller}, page={page}, "
                    f"attempt={attempt_no}/3, status={resp.status_code}"
                )
                if borrowed_token:
                    await token_mgr.release(attempt_token_idx)
                await asyncio.sleep(5 * 2 ** attempt)

            except RateLimitError as e:
                if borrowed_token:
                    # B模式限流路径：记录 cooldown 并释放 token 给池统一调度。
                    await token_mgr.mark_rate_limited(attempt_token_idx, e.reset_time, str(e))
                    continue
                raise
            except TokenInvalidError as e:
                if borrowed_token:
                    # B模式失效路径：在 helper 内剔除坏 token，避免上层 task 感知细节。
                    await token_mgr.mark_invalid(attempt_token_idx, str(e))
                    continue
                raise
            except Exception as e:
                if httpx is not None and isinstance(e, httpx.RequestError):
                    if borrowed_token:
                        await token_mgr.release(attempt_token_idx)
                    logger.warning(
                        f"异步搜索请求异常: query='{q}', {caller}, page={page}, "
                        f"attempt={attempt_no}/3, error={_format_request_error(e)}"
                    )
                    await asyncio.sleep(5 * 2 ** attempt)
                    continue
                if borrowed_token:
                    await token_mgr.release(attempt_token_idx)
                raise

        logger.error(
            f"异步搜索 '{q}' {caller}, page={page} 经 3 次重试仍失败，返回 None 由调用方决定后续处理。"
        )
        return None
    finally:
        if owns_client and hasattr(async_client, "aclose"):
            await async_client.aclose()


async def async_get_stargazers_page(
    token_mgr: GitHubTokenPool,
    owner: str,
    repo: str,
    page: int,
    token_idx: int | None,
    per_page: int = 100,
    client=None,
) -> list[dict] | None:
    """异步获取指定仓库 stargazers 的第 page 页（3 次重试）。

    B模式：请求级 token 借还。
    与异步 Search helper 一致，当 token_idx 为 None 时，每次 stargazers
    请求前临时 acquire，请求结束后立即 release。
    """
    url = f"https://api.github.com/repos/{owner}/{repo}/stargazers"
    params = {"per_page": per_page, "page": page}

    owns_client = client is None
    async_client = client or _build_async_client(timeout_seconds=60.0)

    try:
        for attempt in range(3):
            attempt_token_idx = token_idx
            borrowed_token = False
            try:
                # B模式入口：每次 stargazers 请求前临时借 token。
                if attempt_token_idx is None:
                    attempt_token_idx = await token_mgr.acquire()
                    borrowed_token = True

                headers = token_mgr.get_star_headers(attempt_token_idx)
                _diag_req_t0 = time.time()  # [DIAG-P0] 仅诊断
                resp = await async_client.get(url, headers=headers, params=params)
                logger.debug(
                    "[DIAG-REQ] stargazers %s/%s page=%s token=%s attempt=%s elapsed=%.2fs status=%s",
                    owner, repo, page, attempt_token_idx, attempt + 1,
                    time.time() - _diag_req_t0, resp.status_code,
                )
                _check_response_async(resp, attempt_token_idx)
                if resp.status_code == 200:
                    try:
                        # B模式成功路径：页请求完成后立即归还 token。
                        if borrowed_token:
                            await token_mgr.release(attempt_token_idx)
                        return resp.json()
                    except ValueError:
                        logger.error("异步 stargazers 响应 JSON 解析失败: %s/%s page=%s", owner, repo, page)
                        if borrowed_token:
                            await token_mgr.release(attempt_token_idx)
                        return None
                if resp.status_code == 422:
                    # 422 = REST stargazers 翻页超过上限（约 4 万 star 的超大仓库），
                    # 这是真正需要 GraphQL 采样的信号，返回 None 让上层降级采样。
                    if borrowed_token:
                        await token_mgr.release(attempt_token_idx)
                    return None
                if resp.status_code >= 500:
                    # 5xx 视为瞬时服务端故障：快速重试，耗尽后重排队（不原地占 token 退避）。
                    if borrowed_token:
                        await token_mgr.release(attempt_token_idx)
                    if attempt < STARGAZER_TRANSIENT_FAST_RETRIES:
                        await asyncio.sleep(2 * 2 ** attempt)
                        continue
                    raise RetryableError(
                        time.time() + STARGAZER_REQUEUE_BACKOFF_SECONDS,
                        f"stargazers {owner}/{repo} page={page} server {resp.status_code}",
                    )
                logger.debug(
                    "异步 stargazers 请求失败: %s/%s page=%s, status=%s",
                    owner,
                    repo,
                    page,
                    resp.status_code,
                )
                if borrowed_token:
                    await token_mgr.release(attempt_token_idx)
                await asyncio.sleep(2 * 2 ** attempt)
            except RateLimitError as e:
                if borrowed_token:
                    # B模式限流路径：helper 内写回 cooldown，任务层无需关心 token 状态。
                    await token_mgr.mark_rate_limited(attempt_token_idx, e.reset_time, str(e))
                    continue
                raise
            except TokenInvalidError as e:
                if borrowed_token:
                    await token_mgr.mark_invalid(attempt_token_idx, str(e))
                    continue
                raise
            except Exception as e:
                if httpx is not None and isinstance(e, httpx.RequestError):
                    if borrowed_token:
                        await token_mgr.release(attempt_token_idx)
                    logger.debug(
                        "异步 stargazers 请求异常: %s/%s page=%s, %s",
                        owner,
                        repo,
                        page,
                        _format_request_error(e),
                    )
                    logger.debug(
                        "[DIAG-REQ] stargazers %s/%s page=%s token=%s attempt=%s elapsed=%.2fs status=EXC:%s",
                        owner, repo, page, attempt_token_idx, attempt + 1,
                        time.time() - _diag_req_t0, type(e).__name__,
                    )
                    # 网络异常：快速重试，耗尽后重排队，而非原地长时间退避占着 token。
                    if attempt < STARGAZER_TRANSIENT_FAST_RETRIES:
                        await asyncio.sleep(2 * 2 ** attempt)
                        continue
                    raise RetryableError(
                        time.time() + STARGAZER_REQUEUE_BACKOFF_SECONDS,
                        f"stargazers {owner}/{repo} page={page} network {type(e).__name__}",
                    )
                if borrowed_token:
                    await token_mgr.release(attempt_token_idx)
                raise

        return None
    finally:
        if owns_client and hasattr(async_client, "aclose"):
            await async_client.aclose()


# ══════════════════════════════════════════════════════════════
# 单仓库信息获取（REST /repos API）
# ══════════════════════════════════════════════════════════════


def fetch_repo_info(
    token_mgr: GitHubTokenPool,
    owner: str,
    repo_name: str,
    token_idx: int = 0,
) -> dict | None:
    """
    通过 /repos/{owner}/{repo} 直接获取仓库信息（无 Search API 限制）。

    Returns:
        仓库信息字典（与 Search API items 结构兼容），失败返回 None。

    Raises:
        TokenInvalidError, RateLimitError
    """
    url = f"https://api.github.com/repos/{owner}/{repo_name}"
    headers = token_mgr.get_rest_headers(token_idx)

    for attempt in range(3):
        try:
            resp = requests.get(url, headers=headers, timeout=60)
            _check_response(resp, token_idx)
            if resp.status_code == 200:
                return resp.json()
            if resp.status_code == 404:
                return None
            time.sleep(3 * 2 ** attempt)
        except (TokenInvalidError, RateLimitError):
            raise
        except requests.RequestException as e:
            logger.error("获取仓库信息失败: %s/%s, attempt=%d, error=%s", owner, repo_name, attempt + 1, e)
            time.sleep(3 * 2 ** attempt)

    return None


def _fetch_repo_endpoint_json(
    token_mgr: GitHubTokenPool,
    url: str,
    token_idx: int = 0,
    params: dict | None = None,
    accept: str | None = None,
) -> dict | list | None:
    """请求单个仓库相关 REST 接口，返回 JSON（404/422 视为无数据）。"""
    headers = token_mgr.get_rest_headers(token_idx)
    if accept:
        headers = dict(headers)
        headers["Accept"] = accept

    for attempt in range(3):
        try:
            resp = requests.get(url, headers=headers, params=params, timeout=60)
            _check_response(resp, token_idx)
            if resp.status_code == 200:
                try:
                    return resp.json()
                except ValueError:
                    logger.warning("仓库接口 JSON 解析失败: %s", url)
                    return None
            if resp.status_code in (404, 422):
                return None
            time.sleep(2 * 2 ** attempt)
        except (TokenInvalidError, RateLimitError):
            raise
        except requests.RequestException as e:
            logger.warning(
                "仓库接口请求异常: %s, attempt=%d, error=%s",
                url,
                attempt + 1,
                e,
            )
            time.sleep(2 * 2 ** attempt)

    return None


def fetch_repo_readme_excerpt(
    token_mgr: GitHubTokenPool,
    owner: str,
    repo_name: str,
    token_idx: int = 0,
    max_chars: int = 4000,
) -> dict:
    """获取 README 摘录（文本可能被截断）。"""
    url = f"https://api.github.com/repos/{owner}/{repo_name}/readme"
    data = _fetch_repo_endpoint_json(token_mgr, url, token_idx=token_idx)
    if not isinstance(data, dict):
        return {}

    text = ""
    content = data.get("content", "")
    encoding = str(data.get("encoding", "")).lower()
    if isinstance(content, str) and content:
        if encoding == "base64":
            raw = content.replace("\n", "")
            try:
                decoded = base64.b64decode(raw, validate=False)
                text = decoded.decode("utf-8", errors="ignore")
            except (ValueError, binascii.Error):
                text = ""
        else:
            text = content

    text = text.strip()
    if not text:
        return {}

    excerpt = text[:max_chars]
    return {
        "text": excerpt,
        "truncated": len(text) > max_chars,
        "sha": data.get("sha", ""),
        "path": data.get("path", ""),
    }


def fetch_repo_recent_releases(
    token_mgr: GitHubTokenPool,
    owner: str,
    repo_name: str,
    token_idx: int = 0,
    per_page: int = 5,
) -> list[dict]:
    """获取近期 release 元信息。"""
    url = f"https://api.github.com/repos/{owner}/{repo_name}/releases"
    data = _fetch_repo_endpoint_json(
        token_mgr,
        url,
        token_idx=token_idx,
        params={"per_page": per_page, "page": 1},
    )
    if not isinstance(data, list):
        return []

    releases: list[dict] = []
    for item in data:
        if not isinstance(item, dict):
            continue
        releases.append(
            {
                "tag_name": item.get("tag_name", ""),
                "name": item.get("name", ""),
                "published_at": item.get("published_at", ""),
                "prerelease": bool(item.get("prerelease", False)),
                "draft": bool(item.get("draft", False)),
            }
        )
    return releases


def fetch_repo_recent_commits(
    token_mgr: GitHubTokenPool,
    owner: str,
    repo_name: str,
    token_idx: int = 0,
    per_page: int = 10,
) -> list[dict]:
    """获取默认分支近期提交摘要。"""
    url = f"https://api.github.com/repos/{owner}/{repo_name}/commits"
    data = _fetch_repo_endpoint_json(
        token_mgr,
        url,
        token_idx=token_idx,
        params={"per_page": per_page, "page": 1},
    )
    if not isinstance(data, list):
        return []

    commits: list[dict] = []
    for item in data:
        if not isinstance(item, dict):
            continue
        commit_info = item.get("commit")
        if not isinstance(commit_info, dict):
            continue
        author_info = commit_info.get("author")
        if not isinstance(author_info, dict):
            author_info = {}
        message = str(commit_info.get("message") or "").strip().splitlines()[0:1]
        commits.append(
            {
                "sha": item.get("sha", ""),
                "date": author_info.get("date", ""),
                "message": (message[0] if message else "")[:140],
            }
        )
    return commits


# ══════════════════════════════════════════════════════════════
# Star 范围自动分段
# ══════════════════════════════════════════════════════════════


def get_search_total_count(token_mgr: GitHubTokenPool, query: str, token_idx: int) -> int:
    """
    获取搜索查询的 total_count（不拉取 items）。

    Raises:
        TokenInvalidError, RateLimitError
    """
    url = "https://api.github.com/search/repositories"
    params = {"q": query, "per_page": 1, "page": 1}
    headers = token_mgr.get_rest_headers(token_idx)

    last_error: str = ""

    for attempt in range(3):
        try:
            resp = requests.get(url, headers=headers, params=params, timeout=60)
            _check_response(resp, token_idx)
            if resp.status_code == 200:
                try:
                    return resp.json().get("total_count", 0)
                except (ValueError, KeyError) as e:
                    last_error = _format_request_error(e)
                    logger.error(
                        "total_count 响应 JSON 解析失败: query='%s', attempt=%s, error=%s",
                        query,
                        attempt + 1,
                        last_error,
                    )
                    continue
            else:
                time.sleep(3 * 2 ** attempt)
        except (TokenInvalidError, RateLimitError):
            raise
        except requests.RequestException as e:
            last_error = _format_request_error(e)
            time.sleep(3 * 2 ** attempt)

    if not last_error:
        last_error = "unknown"

    logger.warning(
        "获取 total_count 失败: query='%s'，按保守上界 %s 处理，error=%s",
        query,
        _UNKNOWN_TOTAL_COUNT_FALLBACK,
        last_error,
    )
    return _UNKNOWN_TOTAL_COUNT_FALLBACK


def _get_search_total_count_with_fallback(
    token_mgr: GitHubTokenPool,
    query: str,
    preferred_token_idx: int,
) -> tuple[int, int]:
    """优先使用指定 token 获取 total_count，限流或失效时自动尝试其他 token。"""
    token_count = len(getattr(token_mgr, "tokens", []))
    if token_count <= 1:
        return get_search_total_count(token_mgr, query, preferred_token_idx), preferred_token_idx

    token_order = [preferred_token_idx] + [
        idx for idx in range(token_count) if idx != preferred_token_idx
    ]
    earliest_rate_limit: tuple[int, float] | None = None
    last_token_invalid: TokenInvalidError | None = None

    for token_idx in token_order:
        try:
            total = get_search_total_count(token_mgr, query, token_idx)
            return total, token_idx
        except RateLimitError as exc:
            if hasattr(token_mgr, "record_rate_limited"):
                token_mgr.record_rate_limited(token_idx, exc.reset_time, str(exc))
            logger.warning(
                "total_count 查询命中限流: query='%s', token=%s, reset=%s，尝试其他 token。",
                query,
                token_idx,
                int(exc.reset_time),
            )
            if earliest_rate_limit is None or exc.reset_time < earliest_rate_limit[1]:
                earliest_rate_limit = (token_idx, exc.reset_time)
        except TokenInvalidError as exc:
            if hasattr(token_mgr, "record_invalid"):
                token_mgr.record_invalid(token_idx, str(exc))
            logger.warning(
                "total_count 查询 token 失效: query='%s', token=%s，尝试其他 token。",
                query,
                token_idx,
            )
            last_token_invalid = exc

    if earliest_rate_limit is not None:
        token_idx, reset_time = earliest_rate_limit
        raise RateLimitError(token_idx=token_idx, reset_time=reset_time)
    if last_token_invalid is not None:
        raise last_token_invalid
    return get_search_total_count(token_mgr, query, preferred_token_idx), preferred_token_idx


def auto_split_star_range(
    token_mgr: GitHubTokenPool,
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

    在 WorkerPool 启动前由主线程调用，优先使用给定 token_idx；
    若该 token 限流或失效，会自动尝试其他可用 token。

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
    total, active_token_idx = _get_search_total_count_with_fallback(token_mgr, query, token_idx)
    time.sleep(SEARCH_REQUEST_INTERVAL)

    if total <= max_results:
        logger.debug(f"  区间 stars:{low}..{high} → total_count={total}，无需细分。")
        return [(low, high)]

    mid = (low + high) // 2
    logger.debug(
        f"  区间 stars:{low}..{high} → total_count={total}，"
        f"细分 → [{low}..{mid}] + [{mid + 1}..{high}]"
    )
    left = auto_split_star_range(token_mgr, low, mid, active_token_idx, max_results, min_span, extra_query)
    right = auto_split_star_range(token_mgr, mid + 1, high, active_token_idx, max_results, min_span, extra_query)
    return left + right


# ══════════════════════════════════════════════════════════════
# REST Stargazers 分页查询
# ══════════════════════════════════════════════════════════════


def get_stargazers_page(
    token_mgr: GitHubTokenPool,
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
            resp = requests.get(url, headers=headers, params=params, timeout=60)
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
    token_mgr: GitHubTokenPool,
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
                timeout=60,
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


async def async_graphql_stargazers_batch(
    token_mgr: GitHubTokenPool,
    owner: str,
    repo: str,
    token_idx: int | None,
    last: int = 100,
    before: str | None = None,
    client=None,
) -> tuple[list[datetime], str | None]:
    """异步获取一批 GraphQL stargazers 数据。"""
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

    owns_client = client is None
    async_client = client or _build_async_client(timeout_seconds=60.0)

    try:
        for attempt in range(3):
            attempt_token_idx = token_idx
            borrowed_token = False
            try:
                if attempt_token_idx is None:
                    attempt_token_idx = await token_mgr.acquire()
                    borrowed_token = True

                headers = token_mgr.get_graphql_headers(attempt_token_idx)
                _diag_req_t0 = time.time()  # [DIAG-P0] 仅诊断
                resp = await async_client.post(
                    "https://api.github.com/graphql",
                    headers=headers,
                    json={"query": query_str, "variables": variables},
                )
                logger.debug(
                    "[DIAG-REQ] graphql %s/%s token=%s attempt=%s elapsed=%.2fs status=%s",
                    owner, repo, attempt_token_idx, attempt + 1,
                    time.time() - _diag_req_t0, resp.status_code,
                )
                _check_response_async(resp, attempt_token_idx)
                if resp.status_code == 200:
                    try:
                        data = resp.json()
                    except ValueError:
                        logger.error("异步 GraphQL 响应 JSON 解析失败: %s/%s", owner, repo)
                        if borrowed_token:
                            await token_mgr.release(attempt_token_idx)
                        return [], None

                    if "errors" in data:
                        logger.warning("异步 GraphQL 返回错误: %s/%s, %s", owner, repo, data["errors"])
                        if borrowed_token:
                            await token_mgr.release(attempt_token_idx)
                        return [], None

                    repo_data = data.get("data", {}).get("repository")
                    if not repo_data:
                        if borrowed_token:
                            await token_mgr.release(attempt_token_idx)
                        return [], None

                    edges = repo_data.get("stargazers", {}).get("edges", [])
                    timestamps: list[datetime] = []
                    first_cursor: str | None = None

                    for entry in edges:
                        ts = _parse_starred_at(entry.get("starredAt", ""))
                        if ts:
                            timestamps.append(ts)
                        if first_cursor is None:
                            first_cursor = entry.get("cursor")

                    if borrowed_token:
                        await token_mgr.release(attempt_token_idx)
                    return timestamps, first_cursor

                if borrowed_token:
                    await token_mgr.release(attempt_token_idx)
                await asyncio.sleep(3 * 2 ** attempt)
            except RateLimitError as e:
                if borrowed_token:
                    await token_mgr.mark_rate_limited(attempt_token_idx, e.reset_time, str(e))
                    continue
                raise
            except TokenInvalidError as e:
                if borrowed_token:
                    await token_mgr.mark_invalid(attempt_token_idx, str(e))
                    continue
                raise
            except Exception as e:
                if httpx is not None and isinstance(e, httpx.RequestError):
                    if borrowed_token:
                        await token_mgr.release(attempt_token_idx)
                    logger.debug(
                        "异步 GraphQL 请求异常: %s/%s, %s",
                        owner,
                        repo,
                        _format_request_error(e),
                    )
                    logger.debug(
                        "[DIAG-REQ] graphql %s/%s token=%s attempt=%s elapsed=%.2fs status=EXC:%s",
                        owner, repo, attempt_token_idx, attempt + 1,
                        time.time() - _diag_req_t0, type(e).__name__,
                    )
                    await asyncio.sleep(3 * 2 ** attempt)
                    continue
                if borrowed_token:
                    await token_mgr.release(attempt_token_idx)
                raise

        return [], None
    finally:
        if owns_client and hasattr(async_client, "aclose"):
            await async_client.aclose()


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
