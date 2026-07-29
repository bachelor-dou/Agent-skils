"""全量 star 计数采集：批量 GraphQL 只取 stargazerCount，供每日快照使用。

成本实测（2026-07-29）：100 个别名/次查询 = 1 个 GraphQL 点，5.3 万仓库全量 526 次请求、
526 点；12 个 token 每小时共 6 万点，占用不到 1%。并发 8 时约 4 分钟。
"""

import asyncio
import json
import logging
import time

from .api import _build_async_client, _check_response_async
from .token_pool import GitHubTokenPool
from ...config import SNAPSHOT_BATCH_SIZE, SNAPSHOT_CONCURRENCY
from ...infra.exceptions import RateLimitError, TokenInvalidError

logger = logging.getLogger("hot_projects")

_MAX_ATTEMPTS = 3
_GRAPHQL_URL = "https://api.github.com/graphql"


def _build_query(names: list[str]) -> str:
    """把一批 owner/repo 拼成别名查询。owner/name 用 json.dumps 转义，别名用序号保证合法。"""
    aliases = []
    for i, full_name in enumerate(names):
        owner, _, repo = full_name.partition("/")
        aliases.append(
            f"r{i}: repository(owner:{json.dumps(owner)}, name:{json.dumps(repo)})"
            " { stargazerCount }"
        )
    return "query{" + "\n".join(aliases) + "}"


async def _fetch_batch(
    client, token_pool: GitHubTokenPool, names: list[str]
) -> dict[str, int] | None:
    """取一批仓库的 star 数。

    返回 dict = 成功（缺失的键代表那几个仓库确实没了/改名了）；
    返回 None = 整批全 null 的退化响应，调用方应对半拆分重试，绝不可当成「仓库都没了」。
    重试耗尽仍失败则抛异常，由调用方计入失败批次——半途失败不能伪装成 0 增长。
    """
    query = _build_query(names)
    last_error: Exception | None = None

    for attempt in range(_MAX_ATTEMPTS):
        token_idx = await token_pool.acquire()
        try:
            resp = await client.post(
                _GRAPHQL_URL,
                headers=token_pool.get_graphql_headers(token_idx),
                json={"query": query},
            )
            _check_response_async(resp, token_idx)
            if resp.status_code != 200:
                await token_pool.release(token_idx)
                last_error = RuntimeError(f"HTTP {resp.status_code}")
                await asyncio.sleep(2 ** attempt)
                continue

            payload = resp.json()
            data = payload.get("data")
            # errors 与 data 可以并存：个别仓库 NOT_FOUND 是常态，不能因此丢掉整批。
            if not isinstance(data, dict):
                await token_pool.release(token_idx)
                errs = str(payload.get("errors", ""))[:200]
                if "RATE_LIMITED" in errs:
                    raise RateLimitError(token_idx, time.time() + 60)
                last_error = RuntimeError(f"GraphQL 无 data: {errs}")
                await asyncio.sleep(2 ** attempt)
                continue

            stars: dict[str, int] = {}
            for i, full_name in enumerate(names):
                node = data.get(f"r{i}")
                if isinstance(node, dict) and isinstance(node.get("stargazerCount"), int):
                    stars[full_name] = node["stargazerCount"]

            await token_pool.release(token_idx)
            if not stars and len(names) > 1:
                # 实测 200 个别名会 HTTP 200 + 无 errors + 全字段 null。若把它当成
                # 「这批仓库都没了」，一次批次退化就能让整批基线消失，故拆分重试。
                # 真实的删除/改名是零星的，不会整批同时发生。
                logger.warning("批次 %d 个仓库全部为 null，拆分重试（疑似查询过大退化）。", len(names))
                return None
            return stars

        except RateLimitError as e:
            await token_pool.mark_rate_limited(token_idx, e.reset_time, str(e))
            last_error = e
        except TokenInvalidError as e:
            await token_pool.mark_invalid(token_idx, str(e))
            last_error = e
        except Exception as e:  # noqa: BLE001 — 网络类异常统一退避重试
            await token_pool.release(token_idx)
            last_error = e
            await asyncio.sleep(2 ** attempt)

    raise RuntimeError(f"批次 {len(names)} 个仓库重试 {_MAX_ATTEMPTS} 次仍失败: {last_error}")


async def _collect_chunk(
    client, token_pool: GitHubTokenPool, names: list[str], failed: list[str]
) -> dict[str, int]:
    """采集一批，遇全 null 退化则对半拆分；单个仍为 null 才认定该仓库真的取不到。"""
    try:
        stars = await _fetch_batch(client, token_pool, names)
    except Exception as e:  # noqa: BLE001 — 记为失败批次，不阻塞其余批次
        logger.error("批次采集失败（%d 个仓库将缺席本次快照）: %s", len(names), e)
        failed.extend(names)
        return {}

    if stars is not None:
        return stars
    if len(names) == 1:
        return {}

    mid = len(names) // 2
    left = await _collect_chunk(client, token_pool, names[:mid], failed)
    right = await _collect_chunk(client, token_pool, names[mid:], failed)
    return {**left, **right}


async def collect_star_snapshot(
    token_pool: GitHubTokenPool,
    full_names: list[str],
    batch_size: int = SNAPSHOT_BATCH_SIZE,
    concurrency: int = SNAPSHOT_CONCURRENCY,
    progress_cb=None,
) -> tuple[dict[str, int], list[str]]:
    """采集全部仓库的当前 star 数。

    Returns:
        ({full_name: star}, 采集失败的 full_name 列表)。
        不在返回 dict 里、也不在失败列表里的，是 GitHub 明确查不到（已删除/改名）。
    """
    if not full_names:
        return {}, []

    batches = [full_names[i:i + batch_size] for i in range(0, len(full_names), batch_size)]
    failed: list[str] = []
    stars: dict[str, int] = {}
    done = 0
    sem = asyncio.Semaphore(concurrency)
    client = _build_async_client(timeout_seconds=90.0)

    async def run(batch: list[str]) -> dict[str, int]:
        nonlocal done
        async with sem:
            got = await _collect_chunk(client, token_pool, batch, failed)
        done += 1
        if progress_cb and done % 50 == 0:
            progress_cb(done, len(batches))
        return got

    try:
        for got in await asyncio.gather(*(run(b) for b in batches)):
            stars.update(got)
    finally:
        await client.aclose()

    return stars, failed
