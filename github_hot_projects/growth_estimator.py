"""
Star 增长估算器
===============
三条估算路径（按优先级递降）：

  A. DB 差值法   — DB 有效 + 已有仓库 → current_star - db_star（0 次请求）
  B. REST 二分法 — 新仓库/DB 无效 → stargazers 分页二分查找窗口边界（~5-10 次请求）
  C. 采样外推法  — REST 返回 422 → GraphQL 采 2000 条 + 分段加权速率外推（~20 次请求）

本模块实现路径 B 和 C。路径 A 在 pipeline.py 中直接计算。
"""

import logging
import math
import time
from datetime import datetime, timedelta, timezone

from .config import (
    MAX_BINARY_SEARCH_DEPTH,
    STAR_GROWTH_THRESHOLD,
    TIME_WINDOW_DAYS,
)
from .github_api import (
    get_stargazers_page,
    graphql_stargazers_batch,
    parse_starred_at_from_entry,
)
from .token_manager import TokenManager

logger = logging.getLogger("discover_hot")


def estimate_star_growth_binary(
    token_mgr: TokenManager, owner: str, repo: str, total_stars: int,
    token_idx: int = 0,
) -> int:
    """
    使用 REST stargazers API + 二分法，估算近 TIME_WINDOW_DAYS 天的 star 增量。

    原理：
    ────────────────────────────────────────────
    GitHub stargazers 接口按 starred_at 升序排列（page 1 最老，page N 最新）。
    支持 ?page=N 直接跳到任意页 → 天然适合二分查找。

    步骤：
      1. 快速检查：取最后一页，看该页最老记录
         - 最老记录不在窗口内 → 窗口边界在该页内，直接精确计数（1 次请求）
         - 在窗口内 → 需向前找，进入二分
      2. 二分法：lo=1, hi=total_pages
         - 取 mid 页首条 starred_at 与 cutoff 比较
         - >= cutoff → hi=mid   （边界在更前面）
         - < cutoff  → lo=mid+1 （边界在更后面）
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

    per_page = 100
    total_pages = math.ceil(total_stars / per_page)
    cutoff = datetime.now(timezone.utc) - timedelta(days=TIME_WINDOW_DAYS)

    # ── 快速检查：最新一页 ──
    last_page_data = get_stargazers_page(token_mgr, owner, repo, total_pages, token_idx, per_page)
    if last_page_data is None:
        # REST 无法访问最后一页（超大仓库），降级为采样外推
        logger.info(
            f"  [GROWTH] {owner}/{repo} 最后一页(page={total_pages})不可访问，降级为采样外推。"
        )
        return estimate_by_sampling(token_mgr, owner, repo, token_idx)

    if not last_page_data:
        return 0

    # 最新一页的第一条（该页中最老的）
    oldest_on_last = parse_starred_at_from_entry(last_page_data[0])
    if oldest_on_last and oldest_on_last < cutoff:
        # 窗口边界在最后一页内，直接精确计数
        count = sum(
            1 for e in last_page_data
            if (parse_starred_at_from_entry(e) or datetime.min.replace(tzinfo=timezone.utc)) >= cutoff
        )
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
            logger.info(
                f"  [GROWTH] {owner}/{repo} page={mid} 不可访问，降级为采样外推。"
            )
            return estimate_by_sampling(token_mgr, owner, repo, token_idx)

        if not page_data:
            consecutive_failures += 1
            if consecutive_failures >= 3:
                logger.info(
                    f"  [GROWTH] {owner}/{repo} 连续 {consecutive_failures} 页空数据，降级为采样外推。"
                )
                return estimate_by_sampling(token_mgr, owner, repo, token_idx)
            lo = mid + 1
            continue

        first_entry_time = parse_starred_at_from_entry(page_data[0])
        if first_entry_time is None:
            consecutive_failures += 1
            if consecutive_failures >= 3:
                logger.info(
                    f"  [GROWTH] {owner}/{repo} 连续 {consecutive_failures} 次无法解析时间戳，降级为采样外推。"
                )
                return estimate_by_sampling(token_mgr, owner, repo, token_idx)
            lo = mid + 1
            continue

        consecutive_failures = 0
        if first_entry_time >= cutoff:
            hi = mid   # 整页最老的都在窗口内 → 边界在更前面
        else:
            lo = mid + 1

        # 限速由 TokenManager 统一处理，无需额外 sleep

    # ── 精确计数边界页 ──
    boundary_page = get_stargazers_page(token_mgr, owner, repo, lo, token_idx, per_page)
    within_on_boundary = 0
    if boundary_page:
        within_on_boundary = sum(
            1 for e in boundary_page
            if (parse_starred_at_from_entry(e) or datetime.min.replace(tzinfo=timezone.utc)) >= cutoff
        )

    full_pages_after = total_pages - lo
    growth = full_pages_after * per_page + within_on_boundary

    logger.info(
        f"  [GROWTH] {owner}/{repo} 二分法完成: 边界页={lo}/{total_pages}, "
        f"growth={growth} (深度={actual_depth})"
    )
    return growth


def estimate_by_sampling(
    token_mgr: TokenManager, owner: str, repo: str,
    token_idx: int = 0,
) -> int:
    """
    采样外推法（增强版）：多批次 GraphQL 游标翻页采集 ~2000 条 star，
    分段计算速率并识别加速趋势，外推窗口期增量。

    用于 REST 分页无法覆盖的超大仓库。

    优化策略：
      1. 多批次采集：GraphQL last+before，最多 20 批 × 100 条 = 2000 条
      2. 提前中断：采样跨越窗口边界（cutoff）时停止
      3. 分段速率：按 100 条一段，越新的段权重越高（线性加权 1,2,...,n）
      4. 外推：rate × window_seconds = 整个窗口的预估增长
    """
    cutoff = datetime.now(timezone.utc) - timedelta(days=TIME_WINDOW_DAYS)
    all_timestamps: list[datetime] = []
    cursor: str | None = None
    max_batches = 20

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

    if len(all_timestamps) < 2:
        return 0

    all_timestamps.sort()

    # ── 精确计数：窗口内的 star 总数 ──
    in_window = sum(1 for t in all_timestamps if t >= cutoff)

    oldest_ts = all_timestamps[0]
    if oldest_ts < cutoff:
        # 采样跨越窗口边界 → 窗口内部分就是精确值
        logger.info(
            f"  [GROWTH] {owner}/{repo} 采样精确: {len(all_timestamps)} 条采样, "
            f"窗口内 {in_window} 条"
        )
        return in_window

    # ── 全部采样都在窗口内 → 分段加权速率外推 ──
    window_seconds = TIME_WINDOW_DAYS * 86400
    time_span = (all_timestamps[-1] - all_timestamps[0]).total_seconds()

    if time_span <= 0:
        return len(all_timestamps)

    # 分段计算速率（100 条一段），越新的段权重越高
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
        # 异常值处理：如果最高段速率超过中位数 3 倍，使用中位数
        sorted_rates = sorted(segment_rates)
        median_rate = sorted_rates[len(sorted_rates) // 2]
        max_rate = sorted_rates[-1]

        if len(segment_rates) >= 2 and max_rate > median_rate * 3:
            rate_per_second = median_rate
            logger.info(
                f"  [GROWTH] {owner}/{repo} 检测到异常段速率 "
                f"(max={max_rate:.4f} > 3×median={median_rate:.4f})，使用中位数"
            )
        else:
            # 线性加权：i=0 最老段权重 1，i=n-1 最新段权重 n
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

    # 覆盖率检查：采样时间跨度 / 窗口期
    coverage = time_span / window_seconds
    if coverage < 0.3:
        # 采样只覆盖不到 30% 窗口期，外推不可靠 → 保守估计
        # 窗口内已采样部分直接计数，窗口外未覆盖部分按最低段速率估算
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
