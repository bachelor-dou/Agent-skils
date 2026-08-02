#!/usr/bin/env python
"""每日任务:发现新仓库 → 给 DB 里每个仓库记一份当天 star → 清理旧快照 → 淘汰。

star 时间戳已被 GitHub 限权,增长只能靠「今天的 star − T−N 那天快照里的 star」,所以这个
脚本漏一天是补不回来的。它因此被设计成幂等:当天已有快照就直接退出,可以每小时触发一次。

失败策略各不相同:发现失败记一笔继续跑(DB 明天能补,快照不能,绝不能拖累采集);采集
部分失败照常落盘,缺席的仓库不进淘汰判定;覆盖率不足则中止**不落盘** —— 宁可缺一天锚点
(可顺延到邻近快照),也不写一份可能有系统性错误、事后发现不了的基线。

用法:
    .venv/bin/python -m hot_project.cron_daily_snapshot [--limit N] [--force]
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
import time
from typing import NamedTuple

from . import config
from .common import logs
from .common.timeutil import utc_today
from .infra.store import snapshots, universe
from .infra.tasks import TaskPool
from .provider.github import request as gh
from .provider.github import tasks as gh_tasks
from .provider.github import tokens as gh_tokens
from .provider.github import trending

logger = logging.getLogger("hot_project")

# ── 采集参数:实测定死的,不是可调旋钮,所以不进 config ──
#
# GraphQL 并发在 `gh_tasks.GRAPHQL_WORKERS`(榜单实时取 star 走同一条道,只能有一处)。
# Search 侧按 token 数走:每个 token 有独立的 30 次/分钟额度,配速由 token 池管。
SEARCH_WORKERS_CAP = 12
# Trending 只有三个周期,再多 worker 也没任务可领。
FREE_WORKERS = 3


def _make_pool(tokens: gh_tokens.TokenPool, search_workers: int) -> TaskPool:
    """把 token 池接到任务池上 —— 任务只说自己要哪种 token,不认识 token 池,配速在这里定。"""
    paces = {gh_tasks.SEARCH_TOKEN: gh_tokens.SEARCH, gh_tasks.CORE_TOKEN: gh_tokens.CORE}
    return TaskPool(
        lanes={
            gh_tasks.SEARCH_LANE: search_workers,
            gh_tasks.GRAPHQL_LANE: gh_tasks.GRAPHQL_WORKERS,
            gh_tasks.FREE_LANE: FREE_WORKERS,
        },
        leaser=lambda kind: tokens.lease(paces[kind]),
    )


# ──────────────────────────────────────────────────────────
# 第 2 步 发现
# ──────────────────────────────────────────────────────────
async def discover(
    tokens: gh_tokens.TokenPool, client, min_star: int, max_star: int
) -> gh_tasks.Discovered:
    """三阶段收集,一次性全提交 —— 关键词、星段、Trending 无先后关系,分开跑只让 token 闲着。"""
    sink = gh_tasks.Discovered()
    words = [w for group in config.SEARCH_KEYWORDS.values() for w in group]

    async with _make_pool(tokens, min(SEARCH_WORKERS_CAP, tokens.capacity)) as pool:
        for word in words:
            pool.submit(gh_tasks.KeywordPage(sink, client, word, min_star))
        pool.submit(gh_tasks.SegmentProbe(sink, client, min_star, max_star))
        for period in trending.PERIODS:
            pool.submit(gh_tasks.TrendingPage(sink, client, period))
        await pool.join()

    logger.info(
        "发现完成:%d 个关键词 + 星段 %d..%d + Trending,共扫到 %d 个仓库,失败 %d 处。",
        len(words), min_star, max_star, len(sink.repos), len(sink.failures),
    )
    return sink


def register(found: dict[str, dict], min_star: int) -> list[str]:
    """把新仓库写进 DB。只留 star 和 created_at,展示字段等真上榜了再取。

    created_at 是判定「新项目」的依据;Trending 条目没有它,留空按老项目算,偏保守的那一侧。
    """
    records = {}
    for name, item in found.items():
        star = item.get("star") or item.get("stargazers_count") or 0
        if star < min_star:
            continue
        records[name] = {"star": star, "created_at": item.get("created_at", "")}
    return universe.insert_discovered(records)


TOP_SOURCES_LOGGED = 15


def phase_yield(sources: dict[str, set[str]], fresh: set[str]) -> tuple[int, int, int]:
    """三个发现阶段(关键词 / 星段 / Trending)各命中多少个新仓库。

    一个仓库可被多阶段搜到,计数刻意重叠、相加可大于去重总数:要的是「每个阶段单独能捞到多少」。
    """
    def hit(match) -> int:
        seen: set[str] = set()
        for src, names in sources.items():
            if match(src):
                seen |= names
        return len(seen & fresh)

    return (
        hit(lambda s: s.startswith(gh_tasks.KEYWORD_SOURCE)),
        hit(lambda s: s == gh_tasks.SEGMENT_SOURCE),
        hit(lambda s: s == gh_tasks.TRENDING_SOURCE),
    )


def log_yield(found: gh_tasks.Discovered, fresh: set[str]) -> None:
    """三个阶段各给 DB 带来多少新仓库,再点名产出最高的来源。只写日志,不影响流程。

    用来找白跑的关键词:判据要跨多天看,单天 0 新增很正常。
    """
    scored = sorted(((len(names & fresh), src) for src, names in found.sources.items()),
                    reverse=True)
    if not scored:
        return
    words = [(n, s) for n, s in scored if s.startswith(gh_tasks.KEYWORD_SOURCE)]
    barren = sum(1 for n, _ in words if n == 0)

    kw, seg, trend = phase_yield(found.sources, fresh)
    logger.info("三阶段新增(命中新仓库数,可重叠):关键词 %d,星段 %d,Trending %d;去重后共 %d。",
                kw, seg, trend, len(fresh))
    logger.info("发现来源产出(新入库数),前 %d:%s", TOP_SOURCES_LOGGED,
                ", ".join(f"{s}={n}" for n, s in scored[:TOP_SOURCES_LOGGED]))
    logger.info("关键词 %d 个,其中今天 0 新增 %d 个。", len(words), barren)


# ──────────────────────────────────────────────────────────
# 第 4 步 采集
# ──────────────────────────────────────────────────────────
async def collect(
    tokens: gh_tokens.TokenPool, client, names: list[str]
) -> gh_tasks.Harvest:
    """给每个仓库取一次当天的 star。"""
    sink = gh_tasks.Harvest()
    groups = gh_tasks.batches(names)
    logger.info("待采集 %d 个仓库,分 %d 批,并发 %d。",
                len(names), len(groups), gh_tasks.GRAPHQL_WORKERS)

    started = time.time()
    async with _make_pool(tokens, 1) as pool:
        for group in groups:
            pool.submit(gh_tasks.StarBatch(sink, client, group))
        await pool.join()

    logger.info(
        "采集完成:取到 %d,GitHub 查不到 %d,没问到 %d,耗时 %.0fs。",
        len(sink.stars), len(sink.missing), len(sink.failed), time.time() - started,
    )
    return sink


# ──────────────────────────────────────────────────────────
# 第 7 步 淘汰
#
# DB 不能只增:废弃仓库会一直占着每天的采集量和 git 历史。
#
# 只有两条规则:GitHub 确认查不到(改名/删库/转私有),或者 star 掉到门槛以下。没有宽限期、
# 没有对收藏/已上榜仓库的保护,因为**淘汰是可逆的** —— 再涨回门槛之上,下一次发现阶段会
# 把它重新收进来。
# ──────────────────────────────────────────────────────────
class Eviction(NamedTuple):
    """该删哪些,以及为什么删 —— 分开列出,日志才能看出这轮是正常代谢还是采集出了事。"""

    missing: list[str]      # GitHub 确认查不到
    too_small: list[str]    # star 掉到门槛以下

    @property
    def all(self) -> list[str]:
        return sorted({*self.missing, *self.too_small})

    def __len__(self) -> int:
        return len(self.all)


def decide(
    tracked: set[str],
    stars: dict[str, int],
    confirmed_missing: set[str],
    *,
    star_floor: int,
) -> Eviction:
    """算出这一轮该淘汰谁。纯函数,不碰盘。

    `confirmed_missing` 必须由调用方从成功响应里显式传入,既不在 `stars` 也不在它里面的
    一律不动 —— 把「没问到」当「查不到」,一次限流高峰就能删掉上万个活仓库。
    """
    return Eviction(
        missing=sorted(confirmed_missing & tracked),
        too_small=sorted(
            name for name, star in stars.items()
            if name in tracked and star < star_floor
        ),
    )


# 一轮最多允许因「查不到」删掉多少:占库的这个比例,或这个绝对条数,取大的那个。
#
# `decide` 保证不了**确认查不到本身是错的**:GitHub 一次「200 + data 全 null」的事故会被
# `StarBatch` 拆成几万个「确认查不到」。正常代谢每天几十个,闸门在正常日子里不会响;宁可
# 多留一天垃圾,也不要在一次事故里把库删空(连 LLM 写过的 desc 一起没了)。
MAX_MISSING_RATIO = 0.01
MAX_MISSING_FLOOR = 200


def retire(tracked: set[str], harvest: gh_tasks.Harvest, star_floor: int) -> list[str]:
    plan = decide(tracked, harvest.stars, harvest.missing, star_floor=star_floor)

    ceiling = max(MAX_MISSING_FLOOR, int(len(tracked) * MAX_MISSING_RATIO))
    if len(plan.missing) > ceiling:
        logger.error(
            "放弃本轮「查不到」淘汰:%d 个超过上限 %d(库里共 %d 个)。"
            "这个量级不是仓库集体消失,是采集出了系统性问题 —— 先查采集,库保持原样。",
            len(plan.missing), ceiling, len(tracked),
        )
        # star 掉到门槛以下那部分照常执行:它来自**成功测到**的 star 值,不受影响。
        # 不用 `_replace`:本类重写了 `__len__`,而 `_replace` 会拿 `len()` 自检字段个数,
        # 于是必然 TypeError。
        plan = Eviction(missing=[], too_small=plan.too_small)

    if not plan:
        logger.info("淘汰检查完成:没有仓库需要移除。")
        return []

    universe.evict(set(plan.all))
    logger.info(
        "淘汰 %d 个仓库:GitHub 查不到 %d,star 掉到 %d 以下 %d。",
        len(plan), len(plan.missing), star_floor, len(plan.too_small),
    )
    return plan.all


# ──────────────────────────────────────────────────────────
async def run(args: argparse.Namespace) -> int:
    today = utc_today()

    # ── 1. 幂等 ──────────────────────────────────────────
    if not args.force and args.limit <= 0 and snapshots.already_written(today):
        logger.info("当天(%s)快照已存在,跳过。加 --force 强制重采。", today)
        return 0

    secrets = config.github_tokens()
    if not secrets:
        logger.error("没有配置 GitHub token(设置 GITHUB_TOKENS)。")
        return 1
    tokens = gh_tokens.TokenPool(secrets)

    client = gh.build_client()
    try:
        # ── 2. 发现 ──────────────────────────────────────
        if args.skip_discovery or args.limit > 0:
            logger.info("跳过发现阶段。")
        else:
            try:
                found = await discover(tokens, client, config.MIN_STAR, config.MAX_STAR)
                fresh = register(found.repos, config.MIN_STAR)
                logger.info("其中 %d 个是 DB 里没有的,已入库。", len(fresh))
                log_yield(found, set(fresh))
            except Exception as e:      # noqa: BLE001 —— 发现失败绝不能拖累采集,见文件头
                logger.exception("发现阶段失败,本次只采集 DB 现有项目(明天重试):%s", e)

        # ── 3. 读 DB ─────────────────────────────────────
        tracked = set(universe.load())
        if not tracked:
            logger.error("DB 里没有任何项目,无从采集。")
            return 1
        names = sorted(tracked)
        if args.limit > 0:
            names = names[: args.limit]
            logger.info("调试模式:只采前 %d 个仓库,不落盘。", len(names))

        # ── 4. 采集 ──────────────────────────────────────
        harvest = await collect(tokens, client, names)
    finally:
        await client.aclose()

    if args.limit > 0:
        logger.info("调试模式结束,未写入快照。取到 %d 个。", len(harvest.stars))
        return 0

    # ── 5-6. 覆盖率闸门 + 写快照 ─────────────────────────
    # 闸门在 `snapshots.save` 里(覆盖率不足就不产生文件)。这里不再重复判断,以免两处阈值分叉。
    if snapshots.save(
        today, harvest.stars, not_found=sorted(harvest.missing), expected=len(names),
        throttle={"hits": tokens.stats["rate_limited"],
                  "waited_seconds": round(tokens.stats["waited_seconds"], 1)},
    ) is None:
        logger.error("本次未落盘,明天重试(锚点可顺延到邻近快照)。")
        return 1

    # 传 today:不传的话 prune 自己再取一次 UTC 日期,跨 UTC 零点的那一轮会拿 D 写快照、
    # 却按 D+1 算截断点,把本该留着的 D−35 那份删掉。
    dropped = snapshots.prune(args.keep_days, today=today)
    if dropped:
        logger.info("清理过期快照 %d 份:%s ~ %s", len(dropped), dropped[0], dropped[-1])

    # ── 7. 淘汰 ──────────────────────────────────────────
    if not args.skip_evict:
        retire(tracked, harvest, config.MIN_STAR)
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description="每日 star 快照")
    p.add_argument("--limit", type=int, default=0, help="只采前 N 个仓库(调试用,不落盘)")
    p.add_argument("--force", action="store_true", help="当天已有快照也重采")
    p.add_argument("--skip-discovery", action="store_true", help="跳过发现阶段")
    p.add_argument("--skip-evict", action="store_true", help="跳过淘汰阶段")
    p.add_argument("--keep-days", type=int, default=config.SNAPSHOT_KEEP_DAYS)
    args = p.parse_args()

    # 早挡一次:0 会连今天那份一起删光,而快照重算不回来。prune 里也有同样的检查,
    # 但那时已经采集了两小时。
    if args.keep_days < 1:
        p.error(f"--keep-days 至少为 1,收到 {args.keep_days}(0 会删光全部快照)")

    log_path = logs.setup(config.LOG_DIR, "snapshot", day=utc_today())
    logger.info("=" * 70)
    logger.info("【每日 star 快照】日志:%s", log_path)
    code = asyncio.run(run(args))
    logger.info("=" * 70)
    return code


if __name__ == "__main__":
    sys.exit(main())
