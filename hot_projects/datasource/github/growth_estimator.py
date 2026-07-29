"""
Star 增长估算器（执行层 · 增长组件）
======================================
三条估算路径（按优先级递降）：

架构定位：
  执行层独立组件，由 tools/basic/core.check_repo_growth() 和 tasks/task.py CalcGrowthTask 调用。

  A. DB 差值法   — DB 有效 + 已有仓库 → current_star - db_star（0 次请求）
  B. REST 二分法 — 新仓库/DB 无效 → stargazers 分页二分查找窗口边界（~5-10 次请求）
    C. 采样外推法  — REST 返回 422 → GraphQL 采 3000 条 + 分段加权速率外推（~30 次请求）

本模块实现路径 B 和 C。路径 A 在 tasks/task_help.py 的 _submit_growth_tasks 中直接计算。
"""

import logging
import math
import asyncio
import time
from datetime import datetime, timedelta, timezone

try:
    import httpx
except ImportError:  # pragma: no cover - runtime guarded in async path
    httpx = None

from ...config import (
    MAX_GRAPHQL_SAMPLING_BATCHES,
    MAX_BINARY_SEARCH_DEPTH,
    STAR_GROWTH_THRESHOLD,
    GROWTH_CALC_DAYS,
)
from .api import (
    get_stargazers_page,
    graphql_stargazers_batch,
    parse_starred_at_from_entry,
    async_get_stargazers_page,
    async_graphql_stargazers_batch,
)
from .token_pool import GitHubTokenPool
from ...infra.exceptions import RetryableError

logger = logging.getLogger("hot_projects")

GROWTH_ESTIMATION_UNRESOLVED = -2

# GitHub 自 2026-06-30 起把 stargazers 列表限权给仓库的 admin/collaborator
# （changelog: 2026-06-30-upcoming-access-restrictions-to-public-api-endpoints-and-ui-views）：
# 非协作者读 REST /stargazers 得 404、GraphQL stargazers 得空 edges，二分法与采样外推同时失效。
# 我们跟踪的是他人仓库，换 token 或加 scope 都拿不到，逐仓库硬试只会空转
# （2026-07-29 一轮 4906 次 REST + 1636 次 GraphQL 全部无效，白跑 37 分钟，还把 token 推向
# 二级限流拖累搜索阶段），故连续失败到阈值即熔断。
# 不永久锁死：留一个重探窗口，GitHub 若放宽策略可自动回到时间戳路径。
TIMESTAMP_PATH_STRIKE_LIMIT = 5
TIMESTAMP_PATH_RETRY_AFTER_SECONDS = 1800.0

_timestamp_path_strikes = 0
_timestamp_path_disabled_until = 0.0


def timestamp_path_unavailable() -> bool:
    """star 时间戳路径当前是否处于熔断状态。"""
    return time.time() < _timestamp_path_disabled_until


def reset_timestamp_path_state() -> None:
    """清空熔断状态（供测试与手工重试使用）。"""
    global _timestamp_path_strikes, _timestamp_path_disabled_until
    _timestamp_path_strikes = 0
    _timestamp_path_disabled_until = 0.0


def _note_timestamp_result(got_timestamps: bool) -> None:
    """记录一次时间戳采集结果，连续失败到阈值则熔断。"""
    global _timestamp_path_strikes, _timestamp_path_disabled_until
    if got_timestamps:
        reset_timestamp_path_state()
        return

    _timestamp_path_strikes += 1
    if _timestamp_path_strikes >= TIMESTAMP_PATH_STRIKE_LIMIT and not timestamp_path_unavailable():
        _timestamp_path_disabled_until = time.time() + TIMESTAMP_PATH_RETRY_AFTER_SECONDS
        logger.error(
            "[GROWTH] 连续 %s 个仓库拿不到 star 时间戳（REST stargazers 404 / GraphQL 空 edges），"
            "判定该路径不可用，%.0f 分钟内不再实时估算，增长改由 DB 快照差值/折算兜底。",
            TIMESTAMP_PATH_STRIKE_LIMIT,
            TIMESTAMP_PATH_RETRY_AFTER_SECONDS / 60,
        )


def _create_growth_async_client():
    if httpx is None:
        raise RuntimeError("httpx is required for async growth estimation. Install httpx>=0.27.0")
    return httpx.AsyncClient(timeout=60.0)


def _count_entries_since_cutoff(entries: list[dict], cutoff: datetime) -> int:
    """统计页面中位于窗口期内的 star 记录数量。"""
    floor_time = datetime.min.replace(tzinfo=timezone.utc)
    return sum(
        1 for entry in entries
        if (parse_starred_at_from_entry(entry) or floor_time) >= cutoff
    )


def _estimate_growth_from_sampling_timestamps(
    owner: str,
    repo: str,
    timestamps: list[datetime],
    cutoff: datetime,
    growth_calc_days: int,
) -> int:
    """基于采样时间戳估算窗口增长（同步/异步共用核心逻辑）。"""
    _note_timestamp_result(len(timestamps) >= 2)
    if len(timestamps) < 2:
        logger.warning(
            f"  [GROWTH] {owner}/{repo} 采样数据不足: "
            f"仅获得 {len(timestamps)} 条有效时间戳，跳过本轮增长写入。"
        )
        return GROWTH_ESTIMATION_UNRESOLVED

    all_timestamps = sorted(timestamps)
    in_window = sum(1 for ts in all_timestamps if ts >= cutoff)

    oldest_ts = all_timestamps[0]
    if oldest_ts < cutoff:
        logger.info(
            f"  [GROWTH] {owner}/{repo} 采样精确: {len(all_timestamps)} 条采样, "
            f"窗口内 {in_window} 条"
        )
        return in_window

    window_seconds = growth_calc_days * 86400
    time_span = (all_timestamps[-1] - all_timestamps[0]).total_seconds()

    if time_span <= 0:
        return len(all_timestamps)

    segment_size = 100
    segment_rates: list[float] = []
    for i in range(0, len(all_timestamps), segment_size):
        end = min(i + segment_size, len(all_timestamps))
        if end - i < 2:
            continue
        seg_ts = all_timestamps[i:end]
        seg_span = (seg_ts[-1] - seg_ts[0]).total_seconds()
        if seg_span <= 0:
            continue
        segment_rates.append((end - i) / seg_span)

    if not segment_rates:
        rate_per_second = len(all_timestamps) / time_span
    else:
        sorted_rates = sorted(segment_rates)
        median_rate = sorted_rates[len(sorted_rates) // 2]
        max_rate = sorted_rates[-1]

        if len(segment_rates) >= 2 and max_rate > median_rate * 3:
            rate_per_second = median_rate
            logger.debug(
                f"  [GROWTH] {owner}/{repo} 检测到异常段速率 "
                f"(max={max_rate:.4f} > 3×median={median_rate:.4f})，使用中位数"
            )
        else:
            total_weight = 0.0
            weighted_rate = 0.0
            for i, seg_rate in enumerate(segment_rates):
                weight = i + 1
                weighted_rate += seg_rate * weight
                total_weight += weight
            rate_per_second = (
                weighted_rate / total_weight
                if total_weight > 0
                else len(all_timestamps) / time_span
            )

    coverage = time_span / window_seconds
    if coverage < 0.3:
        min_rate = min(segment_rates) if segment_rates else rate_per_second
        estimated = in_window + int(min_rate * (window_seconds - time_span))
        logger.warning(
            f"  [GROWTH] {owner}/{repo} 外推覆盖率低({coverage:.0%}), "
            f"使用保守估计: {len(all_timestamps)} 条采样, "
            f"跨度 {time_span:.0f}s → estimated={estimated}"
        )
        return estimated

    estimated = int(rate_per_second * window_seconds)
    logger.info(
        f"  [GROWTH] {owner}/{repo} 采样外推: {len(all_timestamps)} 条, "
        f"跨度 {time_span:.0f}s, {len(segment_rates)} 段, "
        f"速率={rate_per_second:.3f}/s → estimated={estimated}"
    )
    return estimated


def estimate_star_growth_binary(
    token_mgr: GitHubTokenPool, owner: str, repo: str, total_stars: int,
    token_idx: int = 0,
    growth_calc_days: int = GROWTH_CALC_DAYS,
) -> int:
    """同步增长估算入口：把瞬时故障（网络/5xx → RetryableError）收敛为 unresolved。

    同步链没有调度器重排机制，因此瞬时故障不再被误当“大仓库”降级采样，
    而是返回 GROWTH_ESTIMATION_UNRESOLVED，由调用方按“暂不可确定”处理。
    """
    try:
        return _estimate_star_growth_binary_impl(
            token_mgr, owner, repo, total_stars,
            token_idx=token_idx, growth_calc_days=growth_calc_days,
        )
    except RetryableError as e:
        logger.warning(f"  [GROWTH] {owner}/{repo} 瞬时故障无法确定增长，标记 unresolved：{e}")
        return GROWTH_ESTIMATION_UNRESOLVED


def _estimate_star_growth_binary_impl(
    token_mgr: GitHubTokenPool, owner: str, repo: str, total_stars: int,
    token_idx: int = 0,
    growth_calc_days: int = GROWTH_CALC_DAYS,
) -> int:
    """
    使用 REST stargazers API + 二分法，估算近指定窗口的 star 增量。

    原理：
    ────────────────────────────────────────────
    GitHub stargazers 接口按 starred_at 升序排列（page 1 最老，page N 最新）。
    支持 ?page=N 直接跳到任意页 → 天然适合二分查找。

    步骤：
      1. 快速检查：取最后一页，看该页最老记录
         - 最老记录不在窗口内 → 窗口边界在该页内，直接精确计数（1 次请求）
         - 在窗口内 → 需向前找，进入二分
        2. 二分法：lo=1, hi=total_pages
            - 取 mid 页末条 starred_at 与 cutoff 比较
            - >= cutoff → hi=mid   （该页已触及窗口，边界在 mid 或更前）
            - < cutoff  → lo=mid+1 （整页都在窗口外，边界在更后）
         - 最大深度 MAX_BINARY_SEARCH_DEPTH(20)
      3. growth = (total_pages - boundary_page) × 100 + 边界页内窗口期计数
      4. 降级：REST 返回 422 → 采样外推法

    请求数估算：
      - 增长 1000 (10页)  → ~5 次
      - 增长 10000 (100页) → ~8 次
      - 增长 50000 (500页) → ~10 次
    """
    if total_stars < STAR_GROWTH_THRESHOLD:
        return 0
    if timestamp_path_unavailable():
        return GROWTH_ESTIMATION_UNRESOLVED

    per_page = 100
    total_pages = math.ceil(total_stars / per_page)
    cutoff = datetime.now(timezone.utc) - timedelta(days=growth_calc_days)

    # ── 快速检查：最新一页 ──
    last_page_data = get_stargazers_page(token_mgr, owner, repo, total_pages, token_idx, per_page)
    if last_page_data is None:
        # REST 无法访问最后一页（超大仓库），降级为采样外推
        logger.warning(
            f"  [GROWTH] {owner}/{repo} 最后一页(page={total_pages})不可访问，降级为采样外推。"
        )
        return estimate_by_sampling(token_mgr, owner, repo, token_idx, growth_calc_days=growth_calc_days)

    if not last_page_data:
        return 0

    _note_timestamp_result(True)
    # 最新一页的第一条（该页中最老的）
    oldest_on_last = parse_starred_at_from_entry(last_page_data[0])
    if oldest_on_last and oldest_on_last < cutoff:
        # 窗口边界在最后一页内，直接精确计数
        count = _count_entries_since_cutoff(last_page_data, cutoff)
        logger.info(f"  [GROWTH] {owner}/{repo} 窗口边界在最后一页: growth={count}")
        return count

    # ── 二分法查找窗口边界页 ──
    lo, hi = 1, total_pages
    actual_depth = 0
    consecutive_failures = 0

    for depth in range(MAX_BINARY_SEARCH_DEPTH):
        if lo >= hi:
            break
        actual_depth = depth + 1
        mid = (lo + hi) // 2

        page_data = get_stargazers_page(token_mgr, owner, repo, mid, token_idx, per_page)
        if page_data is None:
            logger.warning(
                f"  [GROWTH] {owner}/{repo} page={mid} 不可访问，降级为采样外推。"
            )
            return estimate_by_sampling(token_mgr, owner, repo, token_idx, growth_calc_days=growth_calc_days)

        if not page_data:
            consecutive_failures += 1
            if consecutive_failures >= 3:
                logger.warning(
                    f"  [GROWTH] {owner}/{repo} 连续 {consecutive_failures} 页空数据，降级为采样外推。"
                )
                return estimate_by_sampling(token_mgr, owner, repo, token_idx, growth_calc_days=growth_calc_days)
            lo = mid + 1
            continue

        last_entry_time = parse_starred_at_from_entry(page_data[-1])
        if last_entry_time is None:
            consecutive_failures += 1
            if consecutive_failures >= 3:
                logger.warning(
                    f"  [GROWTH] {owner}/{repo} 连续 {consecutive_failures} 次无法解析时间戳，降级为采样外推。"
                )
                return estimate_by_sampling(token_mgr, owner, repo, token_idx, growth_calc_days=growth_calc_days)
            lo = mid + 1
            continue

        consecutive_failures = 0
        if last_entry_time >= cutoff:
            hi = mid   # 该页已出现窗口内记录 → 边界在 mid 或更前面
        else:
            lo = mid + 1

        # 限速由调用方统一处理，无需额外 sleep

    # ── 精确计数边界页 ──
    boundary_page = get_stargazers_page(token_mgr, owner, repo, lo, token_idx, per_page)
    within_on_boundary = 0
    if boundary_page:
        within_on_boundary = _count_entries_since_cutoff(boundary_page, cutoff)

    full_pages_after = total_pages - lo
    growth = full_pages_after * per_page + within_on_boundary

    logger.info(
        f"  [GROWTH] {owner}/{repo} 二分法完成: 边界页={lo}/{total_pages}, "
        f"growth={growth} (深度={actual_depth})"
    )
    return growth


def estimate_by_sampling(
    token_mgr: GitHubTokenPool, owner: str, repo: str,
    token_idx: int = 0,
    growth_calc_days: int = GROWTH_CALC_DAYS,
) -> int:
    """
    采样外推法（增强版）：多批次 GraphQL 游标翻页采集 ~3000 条 star，
    分段计算速率并识别加速趋势，外推窗口期增量。

    用于 REST 分页无法覆盖的超大仓库。

    优化策略：
            1. 多批次采集：GraphQL last+before，最多 30 批 × 100 条 = 3000 条
            2. 提前中断：采样跨越窗口边界（cutoff）时停止
            3. 分段速率：按 100 条一段，越新的段权重越高（线性加权 1,2,...,n）
            4. 外推：rate × window_seconds = 整个窗口的预估增长
    """
    cutoff = datetime.now(timezone.utc) - timedelta(days=growth_calc_days)
    all_timestamps: list[datetime] = []
    cursor: str | None = None
    max_batches = MAX_GRAPHQL_SAMPLING_BATCHES

    for _ in range(max_batches):
        ts_batch, cursor = graphql_stargazers_batch(
            token_mgr, owner, repo, token_idx, last=100, before=cursor
        )
        if not ts_batch:
            break

        all_timestamps.extend(ts_batch)

        # 提前中断：最早一条已在窗口外
        if ts_batch[0] < cutoff:
            break
        if cursor is None:
            break

        time.sleep(0.5)

    return _estimate_growth_from_sampling_timestamps(
        owner,
        repo,
        all_timestamps,
        cutoff,
        growth_calc_days,
    )


async def estimate_star_growth_binary_async(
    token_mgr: GitHubTokenPool,
    owner: str,
    repo: str,
    total_stars: int,
    token_idx: int | None = None,
    growth_calc_days: int = GROWTH_CALC_DAYS,
    client=None,
) -> int:
    """异步版本的 REST 二分法增长估算。"""
    if total_stars < STAR_GROWTH_THRESHOLD:
        return 0
    if timestamp_path_unavailable():
        return GROWTH_ESTIMATION_UNRESOLVED

    per_page = 100
    total_pages = math.ceil(total_stars / per_page)
    cutoff = datetime.now(timezone.utc) - timedelta(days=growth_calc_days)

    owns_client = client is None
    async_client = client or _create_growth_async_client()

    try:
        last_page_data = await async_get_stargazers_page(
            token_mgr,
            owner,
            repo,
            total_pages,
            token_idx,
            per_page,
            client=async_client,
        )
        if last_page_data is None:
            logger.warning(
                f"  [GROWTH] {owner}/{repo} 最后一页(page={total_pages})不可访问，降级为采样外推。"
            )
            return await estimate_by_sampling_async(
                token_mgr,
                owner,
                repo,
                token_idx,
                growth_calc_days=growth_calc_days,
                client=async_client,
            )

        if not last_page_data:
            return 0

        _note_timestamp_result(True)
        oldest_on_last = parse_starred_at_from_entry(last_page_data[0])
        if oldest_on_last and oldest_on_last < cutoff:
            count = _count_entries_since_cutoff(last_page_data, cutoff)
            logger.info(f"  [GROWTH] {owner}/{repo} 窗口边界在最后一页: growth={count}")
            return count

        lo, hi = 1, total_pages
        actual_depth = 0
        consecutive_failures = 0

        for depth in range(MAX_BINARY_SEARCH_DEPTH):
            if lo >= hi:
                break
            actual_depth = depth + 1
            mid = (lo + hi) // 2

            page_data = await async_get_stargazers_page(
                token_mgr,
                owner,
                repo,
                mid,
                token_idx,
                per_page,
                client=async_client,
            )
            if page_data is None:
                logger.warning(
                    f"  [GROWTH] {owner}/{repo} page={mid} 不可访问，降级为采样外推。"
                )
                return await estimate_by_sampling_async(
                    token_mgr,
                    owner,
                    repo,
                    token_idx,
                    growth_calc_days=growth_calc_days,
                    client=async_client,
                )

            if not page_data:
                consecutive_failures += 1
                if consecutive_failures >= 3:
                    logger.warning(
                        f"  [GROWTH] {owner}/{repo} 连续 {consecutive_failures} 页空数据，降级为采样外推。"
                    )
                    return await estimate_by_sampling_async(
                        token_mgr,
                        owner,
                        repo,
                        token_idx,
                        growth_calc_days=growth_calc_days,
                        client=async_client,
                    )
                lo = mid + 1
                continue

            last_entry_time = parse_starred_at_from_entry(page_data[-1])
            if last_entry_time is None:
                consecutive_failures += 1
                if consecutive_failures >= 3:
                    logger.warning(
                        f"  [GROWTH] {owner}/{repo} 连续 {consecutive_failures} 次无法解析时间戳，降级为采样外推。"
                    )
                    return await estimate_by_sampling_async(
                        token_mgr,
                        owner,
                        repo,
                        token_idx,
                        growth_calc_days=growth_calc_days,
                        client=async_client,
                    )
                lo = mid + 1
                continue

            consecutive_failures = 0
            if last_entry_time >= cutoff:
                hi = mid
            else:
                lo = mid + 1

        boundary_page = await async_get_stargazers_page(
            token_mgr,
            owner,
            repo,
            lo,
            token_idx,
            per_page,
            client=async_client,
        )
        within_on_boundary = 0
        if boundary_page:
            within_on_boundary = _count_entries_since_cutoff(boundary_page, cutoff)

        full_pages_after = total_pages - lo
        growth = full_pages_after * per_page + within_on_boundary

        logger.info(
            f"  [GROWTH] {owner}/{repo} 二分法完成: 边界页={lo}/{total_pages}, "
            f"growth={growth} (深度={actual_depth})"
        )
        return growth
    finally:
        if owns_client and hasattr(async_client, "aclose"):
            await async_client.aclose()


async def estimate_by_sampling_async(
    token_mgr: GitHubTokenPool,
    owner: str,
    repo: str,
    token_idx: int | None = None,
    growth_calc_days: int = GROWTH_CALC_DAYS,
    client=None,
) -> int:
    """异步版本的 GraphQL 采样外推。"""
    cutoff = datetime.now(timezone.utc) - timedelta(days=growth_calc_days)
    all_timestamps: list[datetime] = []
    cursor: str | None = None
    max_batches = MAX_GRAPHQL_SAMPLING_BATCHES

    owns_client = client is None
    async_client = client or _create_growth_async_client()

    try:
        for _ in range(max_batches):
            ts_batch, cursor = await async_graphql_stargazers_batch(
                token_mgr,
                owner,
                repo,
                token_idx,
                last=100,
                before=cursor,
                client=async_client,
            )
            if not ts_batch:
                break

            all_timestamps.extend(ts_batch)

            if ts_batch[0] < cutoff:
                break
            if cursor is None:
                break

            await asyncio.sleep(0.5)

        return _estimate_growth_from_sampling_timestamps(
            owner,
            repo,
            all_timestamps,
            cutoff,
            growth_calc_days,
        )
    finally:
        if owns_client and hasattr(async_client, "aclose"):
            await async_client.aclose()
