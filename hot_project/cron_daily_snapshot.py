#!/usr/bin/env python
"""每日任务:发现新仓库 → 给 DB 里每个仓库记一份当天 star → 清理旧快照 → 淘汰。

GitHub 2026-06-30 起把 stargazers 列表限权给 admin/collaborator,star 时间戳对他人仓库
全部失效,二分法和采样外推同时报废。唯一还能还原任意窗口增长的办法是**自己每天记一份计数**:

    增长 = 今天的 star − T−N 那天快照里的 star

整件事的本质是把「实时拉时间戳」换成「快照存量」。所以这个脚本漏一天是补不回来的 ——
今天无从得知昨天的 star 数。它因此被设计成幂等:当天已有快照就一个请求都不发直接退出,
于是可以每小时触发一次,机器在一天里任意一小时活着就够了,而不必恰好在某一分钟活着。

## 七步,其中三道屏障

    1. 幂等检查
    2. 发现:三阶段收集 star ≥ MIN_STAR 的仓库并入库   ← 屏障:新仓库须先入库
    3. 读 DB 全量仓库名
    4. 采集:GraphQL 批量取 star                        ← 屏障:全部完成才能算覆盖率
    5. 覆盖率闸门
    6. 写快照 + 清理过期                                ← 屏障:淘汰要读含今天在内的快照
    7. 淘汰

屏障之所以是屏障:第 2 步漏一个新仓库,它今天就没有基线,以后任何窗口都算不出它的增长;
第 4 步没跑完就算覆盖率,得出的数字必然偏低,会误触第 5 步的闸门。

## 每步的失败策略不同,必须各自声明

    发现失败      记一笔继续跑。DB 是单调累积的,今天漏的明天补 —— 但快照补不回来,
                  所以绝不能让发现的失败把采集也拖下水。
    采集部分失败  照常落盘,只要覆盖率够。缺席的仓库不进淘汰判定。
    覆盖率不足    中止,**不落盘**。宁可缺一天锚点(可顺延到邻近快照),也不写一份
                  可能有系统性错误的基线 —— 错的基线会让整个窗口的增长全错,且事后发现不了。

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
from .provider.github import client as gh
from .provider.github import tasks as gh_tasks
from .provider.github import tokens as gh_tokens
from .provider.github import trending

logger = logging.getLogger("hot_project")

# ── 采集参数:实测定死的,不是可调旋钮,所以不进 config ──
#
# GraphQL 并发压到 4 的原因不是配额(全量 780 点是 12 token 每小时 6 万点的零头),
# 而是 GitHub 的二级限流 —— 它按请求速率 + IP 计,与 token 数无关,而 Actions 是共享 IP。
# 实测(2026-07-29):并发 3 顺畅无限流;并发 8 时整批 token 被限、反复等 60s 重置。
# 多放 token 不会更快,只会加剧请求爆发。
GRAPHQL_WORKERS = 4
# Search 侧的 worker 数按 token 数走:每个 token 有独立的 30 次/分钟额度,配速由 token 池管。
SEARCH_WORKERS_CAP = 12
# Trending 只有三个周期,再多 worker 也没任务可领。
FREE_WORKERS = 3


def _make_pool(tokens: gh_tokens.TokenPool, search_workers: int) -> TaskPool:
    """把 token 池接到任务池上。

    任务只说自己要哪种 token(字符串),不认识 token 池;这里是唯一知道
    「search 那种要按 2.1 秒配速」的地方。
    """
    paces = {gh_tasks.SEARCH_TOKEN: gh_tokens.SEARCH, gh_tasks.CORE_TOKEN: gh_tokens.CORE}
    return TaskPool(
        lanes={
            gh_tasks.SEARCH_LANE: search_workers,
            gh_tasks.GRAPHQL_LANE: GRAPHQL_WORKERS,
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
    """三阶段收集,一次性全提交。

    关键词、星段、Trending 之间没有先后关系,分开跑只是让 token 轮流闲着。一起提交后
    Trending(不吃 token)和搜索天然重叠,星段探测按层展开,整段的墙钟时间由最慢的一路决定。
    """
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
    """把新仓库写进 DB。

    只留 star 和 created_at:展示字段(描述/话题/README 链接)占旧 DB 体积的 74%,
    而一个仓库要等涨过上榜门槛、真进了报告才用得上它们。观测层不为此付存储。

    created_at 是判定「新项目」的依据(创建时间落在窗口内 = 它的 star 全是本窗口涨的),
    Trending 抓来的条目没有这个字段,拿不到就留空 —— 留空按老项目算,只认窗口内的差值,
    是偏保守的那一侧。
    """
    records = {}
    for name, item in found.items():
        star = item.get("star") or item.get("stargazers_count") or 0
        if star < min_star:
            continue
        records[name] = {"star": star, "created_at": item.get("created_at", "")}
    return universe.insert_discovered(records)


TOP_SOURCES_LOGGED = 15


def log_yield(found: gh_tasks.Discovered, fresh: set[str]) -> None:
    """每个发现来源各带来了多少个新仓库。只写日志,不影响流程。

    这份账是用来回答「333 个关键词里哪些在白跑」的:一个关键词连续多天 0 新增,
    说明它搜到的东西星段扫描早就覆盖了,砍掉能省下十几分钟和几十次限流。
    判据要跨多天看 —— 单天 0 新增很正常(那天恰好没有新项目撞上这个词)。
    """
    scored = sorted(((len(names & fresh), src) for src, names in found.sources.items()),
                    reverse=True)
    if not scored:
        return
    words = [(n, s) for n, s in scored if s.startswith(gh_tasks.KEYWORD_SOURCE)]
    barren = [s.removeprefix(gh_tasks.KEYWORD_SOURCE) for n, s in words if n == 0]

    logger.info("发现来源产出(新入库数),前 %d:%s", TOP_SOURCES_LOGGED,
                ", ".join(f"{s}={n}" for n, s in scored[:TOP_SOURCES_LOGGED]))
    logger.info("关键词 %d 个,其中今天 0 新增的 %d 个:%s",
                len(words), len(barren), " ".join(sorted(barren)) or "无")


# ──────────────────────────────────────────────────────────
# 第 4 步 采集
# ──────────────────────────────────────────────────────────
async def collect(
    tokens: gh_tokens.TokenPool, client, names: list[str]
) -> gh_tasks.Harvest:
    """给每个仓库取一次当天的 star。"""
    sink = gh_tasks.Harvest()
    groups = gh_tasks.batches(names)
    logger.info("待采集 %d 个仓库,分 %d 批,并发 %d。", len(names), len(groups), GRAPHQL_WORKERS)

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
# DB 不能只增:掉到几十星的废弃仓库会一直占着每天的采集量(采集次数和仓库数成正比)和
# git 历史。
#
# 只有两条规则:GitHub 确认查不到(改名/删库/转私有,都无可挽回 —— 转私有即便仓库还活着,
# 我们也再取不到任何信息),或者 star 掉到门槛以下。没有宽限期、没有对收藏/已上榜/新建
# 仓库的保护,因为**淘汰是可逆的**:观测门槛已经压到 500 星,一个仓库要是改名后又涨回门槛
# 之上,下一次发现阶段会把它重新收进来。为不可逆的操作设保护是对的,为可逆的操作设保护
# 只是把简单的事弄复杂。
# ──────────────────────────────────────────────────────────
class Eviction(NamedTuple):
    """该删哪些,以及为什么删 —— 分开列出是为了日志能一眼看出这轮是正常代谢还是出事了。

    某天 `missing` 突然从几十涨到几万,那不是仓库集体消失,是采集出了系统性问题。
    """

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

    唯一必须小心的地方是「GitHub 确认查不到」和「我们这次没问到」的区别 ——
    它们在快照里长得一模一样(都是缺这个键),后果却差着一个数量级:

        确认查不到    采集成功返回了,只是这个键不在结果里   →  它真的没了
        没问到        整批限流/超时失败,压根没拿到回答       →  它可能好端端的

    把后者当成前者,一次限流高峰就能删掉成千上万个活仓库。所以本函数**不接受**「快照里
    没有就是没了」这种推断,调用方必须显式传入 `confirmed_missing`,那份名单只能来自
    成功的响应。既不在 `stars` 也不在 `confirmed_missing` 的,一律不动。
    """
    return Eviction(
        missing=sorted(confirmed_missing & tracked),
        too_small=sorted(
            name for name, star in stars.items()
            if name in tracked and star < star_floor
        ),
    )


def retire(tracked: set[str], harvest: gh_tasks.Harvest, star_floor: int) -> list[str]:
    plan = decide(tracked, harvest.stars, harvest.missing, star_floor=star_floor)
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
    # 闸门在 store 里(`snapshots.save` 覆盖率不足就不产生文件)。这里不再判一次:
    # 两处阈值早晚会分叉,而写快照的那一侧才是知道「什么样的快照能落盘」的地方。
    if snapshots.save(
        today, harvest.stars, not_found=sorted(harvest.missing), expected=len(names),
        throttle={"hits": tokens.stats["rate_limited"],
                  "waited_seconds": round(tokens.stats["waited_seconds"], 1)},
    ) is None:
        logger.error("本次未落盘,明天重试(锚点可顺延到邻近快照)。")
        return 1

    dropped = snapshots.prune(args.keep_days)
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

    log_path = logs.setup(config.LOG_DIR, "snapshot", day=utc_today())
    logger.info("=" * 70)
    logger.info("【每日 star 快照】日志:%s", log_path)
    code = asyncio.run(run(args))
    logger.info("=" * 70)
    return code


if __name__ == "__main__":
    sys.exit(main())
