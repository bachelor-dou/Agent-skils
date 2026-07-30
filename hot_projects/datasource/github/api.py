"""
GitHub API 封装
===============
封装 GitHub REST / GraphQL API 的底层调用，包括：
  - 仓库搜索（Search API）
  - Star 范围自动分段
  - 单仓库信息 / README / 提交 / release

所有函数接收 token_idx 参数（由 Worker 绑定），不再内部 acquire/release。
限流（403/429）和 Token 失效（401）通过抛异常交由 Worker 处理。

例外是没有 Worker 调度的同步单仓库请求（仓库信息 / README / 提交 / release / tree）
和 total_count 查询：它们的 token_idx 只是首选，撞限流会自己顺延到其他 token，
只有全部 token 都不可用时才抛出异常。见 _with_token_rotation。
"""

import base64
import binascii
import asyncio
import logging
import time
from email.utils import parsedate_to_datetime
from typing import Any, Callable

import requests

try:
    import httpx
except ImportError:  # pragma: no cover - handled at runtime when async path is used
    httpx = None

from ...config import MIN_STAR
from .token_pool import GitHubTokenPool
from ...infra.exceptions import RateLimitError, TokenInvalidError

logger = logging.getLogger("hot_projects")

# Search API 相邻请求的最小间隔（秒）。搜索/扫描/详情补全三条链路共用，
# 都在本模块这一层发请求，所以定义在这里而不是 config——它是限速实现细节，不是可调策略。
SEARCH_REQUEST_INTERVAL = 1.3

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


def _with_token_rotation(
    token_mgr: GitHubTokenPool,
    attempt: Callable[[int], Any],
    *,
    what: str,
    preferred_token_idx: int = 0,
) -> tuple[Any, int]:
    """依次换 token 执行 attempt(token_idx)，限流/失效就顺延到下一个。

    同步 REST 链路没有 AsyncTokenPool.acquire 那样的调度器帮它挑 token——调用方全都写死
    token_idx=0，于是所有单仓库请求（报告抓 README/提交、Agent 各单仓库工具）都挤在
    第一个 token 上，另外十几个闲着。额度耗尽后 report.py 只是 except+warning，
    仓库静默失去素材、LLM 只能靠元数据编描述：不崩，但整篇报告的质量一起悄悄塌。

    attempt 必须自己完成一次完整请求（含网络抖动重试），并按老约定抛
    RateLimitError / TokenInvalidError；其它返回值（含 404 的 None）都算成功，不触发换 token。

    Returns:
        (attempt 的返回值, 实际用成的 token 序号)

    Raises:
        RateLimitError:    全部 token 都限流时抛 reset 最早的那个，让调用方知道最短等多久。
        TokenInvalidError: 没有任何 token 限流、但有 token 失效时抛最后一个。
    """
    token_count = len(getattr(token_mgr, "tokens", []))
    if token_count <= 1:
        # 单 token（含测试里的桩）没有可换的，直接透传，异常语义与从前完全一致。
        return attempt(preferred_token_idx), preferred_token_idx

    if hasattr(token_mgr, "rest_token_order"):
        order = token_mgr.rest_token_order(preferred_token_idx)
    else:  # 测试桩只提供 tokens/get_rest_headers
        order = [preferred_token_idx] + [i for i in range(token_count) if i != preferred_token_idx]
    earliest_rate_limit: tuple[int, float] | None = None
    last_invalid: TokenInvalidError | None = None

    for token_idx in order:
        try:
            return attempt(token_idx), token_idx
        except RateLimitError as exc:
            if hasattr(token_mgr, "record_rate_limited"):
                token_mgr.record_rate_limited(token_idx, exc.reset_time, str(exc))
            logger.warning(
                "%s 命中限流: token=%s, reset=%s，顺延到下一个 token。",
                what, token_idx, int(exc.reset_time),
            )
            if earliest_rate_limit is None or exc.reset_time < earliest_rate_limit[1]:
                earliest_rate_limit = (token_idx, exc.reset_time)
        except TokenInvalidError as exc:
            # 401 走 strikes/冷却而非永久失效，避免瞬时 401 把有效 token 踢光（与异步一致）。
            if hasattr(token_mgr, "record_auth_failed"):
                token_mgr.record_auth_failed(token_idx, str(exc))
            elif hasattr(token_mgr, "record_invalid"):
                token_mgr.record_invalid(token_idx, str(exc))
            logger.warning("%s token 失效: token=%s，顺延到下一个 token。", what, token_idx)
            last_invalid = exc

    if earliest_rate_limit is not None:
        token_idx, reset_time = earliest_rate_limit
        raise RateLimitError(token_idx=token_idx, reset_time=reset_time)
    raise last_invalid  # 循环必然以两类异常之一收尾，否则上面已 return


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

                # 按 token 主动配速，压在 Search API 30/min 之下，避免撞 429 后被罚长冷却。
                await token_mgr.throttle_search(attempt_token_idx)
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

    token_idx 只是首选：限流/失效会自动顺延到其他 token（见 _with_token_rotation）。

    Returns:
        仓库信息字典（与 Search API items 结构兼容），失败或 404 返回 None。

    Raises:
        TokenInvalidError, RateLimitError：全部 token 都不可用时才抛。
    """
    url = f"https://api.github.com/repos/{owner}/{repo_name}"

    def _attempt(idx: int) -> dict | None:
        headers = token_mgr.get_rest_headers(idx)
        for attempt in range(3):
            try:
                resp = requests.get(url, headers=headers, timeout=60)
                _check_response(resp, idx)
                if resp.status_code == 200:
                    return resp.json()
                if resp.status_code == 404:
                    return None
                time.sleep(3 * 2 ** attempt)
            except (TokenInvalidError, RateLimitError):
                raise
            except requests.RequestException as e:
                logger.error("获取仓库信息失败: %s/%s, attempt=%d, error=%s",
                             owner, repo_name, attempt + 1, e)
                time.sleep(3 * 2 ** attempt)
        return None

    result, _ = _with_token_rotation(
        token_mgr, _attempt,
        what=f"仓库信息 {owner}/{repo_name}", preferred_token_idx=token_idx,
    )
    return result


def _fetch_repo_endpoint_json(
    token_mgr: GitHubTokenPool,
    url: str,
    token_idx: int = 0,
    params: dict | None = None,
    accept: str | None = None,
) -> dict | list | None:
    """请求单个仓库相关 REST 接口，返回 JSON（404/422 视为无数据）。

    readme / releases / commits / tree 四个抓取函数都走这里，所以 token 轮换只需接在这一处。
    token_idx 是首选，限流/失效自动顺延（见 _with_token_rotation）。
    """
    def _attempt(idx: int) -> dict | list | None:
        headers = token_mgr.get_rest_headers(idx)
        if accept:
            headers = dict(headers)
            headers["Accept"] = accept
        for attempt in range(3):
            try:
                resp = requests.get(url, headers=headers, params=params, timeout=60)
                _check_response(resp, idx)
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
                logger.warning("仓库接口请求异常: %s, attempt=%d, error=%s", url, attempt + 1, e)
                time.sleep(2 * 2 ** attempt)
        return None

    result, _ = _with_token_rotation(
        token_mgr, _attempt, what=f"仓库接口 {url}", preferred_token_idx=token_idx,
    )
    return result


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


# 目录树线索中要跳过的噪音路径/扩展名（依赖、构建产物、多媒体资源）
_TREE_NOISE_DIRS = (
    "node_modules/", "vendor/", "third_party/", "dist/", "build/",
    ".github/", ".git/", "assets/", "static/img", "__pycache__/",
)
_TREE_NOISE_EXTS = (
    ".png", ".jpg", ".jpeg", ".gif", ".svg", ".ico", ".webp",
    ".mp4", ".mp3", ".wav", ".woff", ".woff2", ".ttf", ".lock",
)


def fetch_repo_tree_paths(
    token_mgr: GitHubTokenPool,
    owner: str,
    repo_name: str,
    token_idx: int = 0,
    max_paths: int = 80,
) -> list[str]:
    """获取默认分支目录树的文件路径清单（仅路径名，不取内容）。

    用途：README 过于简陋时，docs/、examples/ 等文件名本身就是功能覆盖线索。
    一次 API 请求；剔除依赖/构建产物/多媒体噪音后按路径排序截断。
    """
    url = f"https://api.github.com/repos/{owner}/{repo_name}/git/trees/HEAD"
    data = _fetch_repo_endpoint_json(
        token_mgr, url, token_idx=token_idx, params={"recursive": "1"},
    )
    if not isinstance(data, dict):
        return []

    paths: list[str] = []
    for entry in data.get("tree", []):
        if not isinstance(entry, dict) or entry.get("type") != "blob":
            continue
        path = str(entry.get("path") or "")
        if not path:
            continue
        lowered = path.lower()
        if any(seg in lowered for seg in _TREE_NOISE_DIRS):
            continue
        if lowered.endswith(_TREE_NOISE_EXTS):
            continue
        paths.append(path)

    paths.sort()
    return paths[:max_paths]


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
    return _with_token_rotation(
        token_mgr,
        lambda idx: get_search_total_count(token_mgr, query, idx),
        what=f"total_count 查询 query='{query}'",
        preferred_token_idx=preferred_token_idx,
    )


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
