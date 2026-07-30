#!/usr/bin/env python
"""每日任务：发现新仓库收进 DB → 按 DB 全量采当天 star 快照 → 清理过期快照 → 淘汰长期掉队的仓库。

GitHub 2026-06-30 起把 stargazers 列表限权给 admin/collaborator，star 时间戳对他人仓库
全部失效（二分法、采样外推同时报废）。唯一还能还原任意窗口增长的办法是自己每天记一份计数：
增长 = 当前 star − T−N 那天快照里的 star。整件事的本质是把「实时拉时间戳」换成「快照存量」。

本任务只管一件事：DB 这个「观测宇宙」有多宽。门槛只有一条 MIN_STAR——涨过它就收进来
开始记快照，长期掉到它以下就淘汰。谁能出榜是周报按 STAR_GROWTH_THRESHOLD（窗口内涨幅）
另外判的，与本任务无关，所以这里不需要第二条 star 线。
压低门槛的意义：仓库涨过门槛那天没有一个窗口前的快照，增长只能算「未决」而被剔出排名
（上期 1761 个）。提前几周收进来，等它涨到够格出榜时基线已经存好。

收集三阶段（关键词搜索 + 星段扫描 + Trending）也从每周挪到了每天，且是唯一还在扫的地方：
周报的候选池已改成直接读今天的快照（见 ranking._collect_from_snapshot）。出榜因此不再和
收集绑在一次运行里，今天限流漏掉的明天补上（以前收集失败就直接损失当期数据）。

DB 不是只增：连续 DB_EVICT_GRACE_DAYS 份快照都掉到发现门槛以下的仓库会被淘汰，
否则采集量（与仓库数成正比）和 git 历史会被废弃仓库一直拖着。上过榜的、被收藏的、
近 DB_EVICT_PROTECT_NEW_DAYS 天新建的一律不动——这三类删了就再也回不来。

采集逻辑直接写在本文件里（不另立模块）：只有这个定时任务会用到它，
而快照的「读取侧」在 infra/snapshots.py——那部分是周报算增长锚点用的，属于主项目。

用法：
    cd /root/code/Agent-skils/hot_projects
    /root/code/Agent-skils/.venv/bin/python3 cron_daily_star_snapshot.py
"""
# ============================================================
# 部署为定时任务 —— 现已跑在 GitHub Actions（.github/workflows/daily-star-snapshot.yml），
# 本机不必再配 crontab（本机也采会和 CI 的同名快照文件撞车）。
#
# 无论跑在哪，都建议每小时触发一次而不是每天一次：
#   本脚本幂等，当天已有快照就立刻退出、一个请求都不发，所以按小时跑几乎零成本。
#   好处是机器在一天里任意一小时活着就能拿到当天快照，而不必恰好在某一分钟活着。
#   漏掉的那天补不回来（今天无从得知昨天的 star 数），容错只能靠多给自己机会。
#
# 日志：logs/YYYY-MM/snapshot-YYYY-MM-DD.log
# ============================================================
import argparse
import asyncio
import json
import logging
import os
import sys
import time
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from hot_projects.config import (
    DB_EVICT_GRACE_DAYS,
    DB_EVICT_PROTECT_NEW_DAYS,
    LOG_DIR,
    MAX_STAR,
    MIN_STAR,
    SNAPSHOT_KEEP_DAYS,
)
from hot_projects.datasource.github.api import _build_async_client, _check_response_async
from hot_projects.datasource.github.token_pool import GitHubTokenPool
from hot_projects.infra.db import evict_stale_projects, insert_new_projects, load_db
from hot_projects.infra.favorites_store import all_favorited_repos
from hot_projects.infra.exceptions import RateLimitError, TokenInvalidError
from hot_projects.infra.snapshots import (
    available_dates,
    load_snapshot,
    prune_snapshots,
    save_snapshot,
    utc_today,
)

logger = logging.getLogger("hot_projects")

# ── 采集参数：都是实测定死的，不是可调旋钮，所以不放 config ──
# 每次 GraphQL 查询塞多少个仓库别名。实测 100 别名/次 = 1 个 GraphQL 点，
# 5.2 万仓库全量 526 点、约 4 分钟。**勿上调**：200 会 HTTP 200 + 全字段 null 静默退化，
# 看起来成功、实际拿回一批空值，直接污染当天基线。
SNAPSHOT_BATCH_SIZE = 100
# 并发上限，实际并发 = min(本值, token 数)。
# 压到 4 的原因：瓶颈不是配额（全量 526 点是配额零头），而是 GitHub 的二级限流——
# 它按请求速率/并发 + IP 计，与 token 数无关，而 Actions 是共享 IP，比住宅 IP 更易触发。
# 实测（2026-07-29，Actions）：并发 3 顺畅无限流；并发 8（12 token 时的旧固定值）整批 token
# 被限、反复等 ~60s 重置。所以宁低勿高，多放 token 不会更快，只会加剧请求爆发。
SNAPSHOT_MAX_CONCURRENCY = 4
# 采集覆盖率低于此值就拒绝落盘：宁可缺一天锚点（可顺延到邻近快照），
# 也不写入一份可能有系统性错误的基线——错的基线会让整个窗口的增长全错，且无法事后发现。
SNAPSHOT_MIN_COVERAGE = 0.5

_MAX_ATTEMPTS = 3
_GRAPHQL_URL = "https://api.github.com/graphql"


# ──────────────────────────────────────────────────────────────
# 采集：批量 GraphQL 只取 stargazerCount
#   成本实测（2026-07-29）：100 个别名/次 = 1 个 GraphQL 点，5.3 万仓库全量 526 次请求、
#   526 点；12 个 token 每小时共 6 万点，占用不到 1%。
# ──────────────────────────────────────────────────────────────
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
    concurrency: int | None = None,
    progress_cb=None,
) -> tuple[dict[str, int], list[str]]:
    """采集全部仓库的当前 star 数。

    concurrency=None（默认）时按 token 数自适应：min(SNAPSHOT_MAX_CONCURRENCY, token 数)。
    worker 不会多于 token，避免刷「等待 token」日志；token 多则自动跑满到安全上限。

    Returns:
        ({full_name: star}, 采集失败的 full_name 列表)。
        不在返回 dict 里、也不在失败列表里的，是 GitHub 明确查不到（已删除/改名）。
    """
    if not full_names:
        return {}, []

    if concurrency is None:
        concurrency = min(SNAPSHOT_MAX_CONCURRENCY, token_pool.token_count)
    concurrency = max(1, concurrency)

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


# ──────────────────────────────────────────────────────────────
# 发现：把 star >= MIN_STAR 的新仓库收进 DB
# ──────────────────────────────────────────────────────────────
def discover_and_register(token_pool: GitHubTokenPool) -> tuple[int, int]:
    """跑三阶段收集（关键词搜索 + 星段扫描 + Trending），新仓库插进 DB。

    直接复用榜单流水线的 _collect：三个阶段的顺序、去重、自动分段（把星段切到每段 <1000 条
    以绕开 Search API 的返回上限）、多轮页面补偿、token 池限流处理全在里面，
    都是踩过坑才对的逻辑，不该为了「脚本独立」再写一遍。

    收集从每周挪到每天的意义：出榜不再和收集绑在一次运行里。今天限流漏掉的仓库明天补上，
    DB 单调累积；而以前收集失败就直接损失当期数据（上期漏了约 1800 个）。

    Returns:
        (扫到的仓库数, 实际插入 DB 的新仓库数)
    """
    from hot_projects.datasource.github.provider import GitHubProvider
    from hot_projects.tools.tool.ranking import _collect

    provider = GitHubProvider(token_pool)
    repos = _collect(provider, "comprehensive", {
        "min_star": MIN_STAR,
        "max_star": MAX_STAR,
    })

    # 只留 star 和 created_at：展示字段（gh_desc/topics/readme_url 占 DB 体积的 74%）
    # 等它涨过上榜门槛、真的进了报告再由报告流程补，不为观测层白付存储。
    records = {}
    for r in repos:
        fn = r.get("full_name")
        if not fn:
            continue
        raw = r.get("_raw") or {}
        records[fn] = {
            "star": r.get("star", 0),
            "created_at": raw.get("created_at", ""),
        }

    inserted = insert_new_projects(records)
    logger.info(
        "发现完成: 三阶段共扫到 %d 个仓库（star >= %d），其中 %d 个是 DB 里没有的，已插入。",
        len(records), MIN_STAR, inserted,
    )
    return len(records), inserted


# ──────────────────────────────────────────────────────────────
# 淘汰：把长期掉出发现门槛的仓库从 DB 移除
# ──────────────────────────────────────────────────────────────
def evict_from_db(grace_days: int) -> list[str]:
    """连续 grace_days 份快照里 star 都低于 MIN_STAR 的仓库，从 DB 移除。

    不做淘汰的话 DB 只增不减：掉到几十星的废弃仓库会一直占着每天的快照采集
    （采集量和仓库数成正比）和 git 历史。判定只读已有快照，不额外存状态。

    Returns:
        被移除的仓库名（已排序）。
    """
    recent = [
        stars for stars in (load_snapshot(d) for d in available_dates()[-grace_days:])
        if stars is not None
    ]
    removed = evict_stale_projects(
        recent,
        star_floor=MIN_STAR,
        grace_days=grace_days,
        protect_new_days=DB_EVICT_PROTECT_NEW_DAYS,
        keep=all_favorited_repos(),
    )
    if removed:
        preview = ", ".join(removed[:10]) + (", ..." if len(removed) > 10 else "")
        logger.info(
            "淘汰 %d 个仓库（连续 %d 份快照 star < %d，且未上过榜/未被收藏/非近 %d 天新建）: %s",
            len(removed), grace_days, MIN_STAR, DB_EVICT_PROTECT_NEW_DAYS, preview,
        )
    elif len(recent) >= grace_days:
        logger.info("淘汰检查完成: 没有仓库连续 %d 天低于 %d 星。", grace_days, MIN_STAR)
    return removed


# ──────────────────────────────────────────────────────────────
# 任务入口
# ──────────────────────────────────────────────────────────────
def setup_logging() -> str:
    now = datetime.now()
    month_dir = os.path.join(LOG_DIR, now.strftime("%Y-%m"))
    os.makedirs(month_dir, exist_ok=True)
    log_path = os.path.join(month_dir, f"snapshot-{now.strftime('%Y-%m-%d')}.log")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.FileHandler(log_path, encoding="utf-8"), logging.StreamHandler()],
    )
    # httpx 每个请求打一行 INFO，全量 526 批会把日志淹掉，只留警告。
    logging.getLogger("httpx").setLevel(logging.WARNING)
    return log_path


def main() -> int:
    parser = argparse.ArgumentParser(description="每日 star 快照")
    parser.add_argument("--keep-days", type=int, default=SNAPSHOT_KEEP_DAYS,
                        help=f"快照保留天数（默认 {SNAPSHOT_KEEP_DAYS}）")
    parser.add_argument("--min-coverage", type=float, default=SNAPSHOT_MIN_COVERAGE,
                        help=f"覆盖率低于此值拒绝落盘（默认 {SNAPSHOT_MIN_COVERAGE}）")
    parser.add_argument("--limit", type=int, default=0,
                        help="只采集前 N 个仓库（调试用，>0 时不落盘）")
    parser.add_argument("--force", action="store_true",
                        help="当天已有快照时也重新采集并覆盖")
    parser.add_argument("--skip-discovery", action="store_true",
                        help=f"跳过 star>={MIN_STAR} 的发现阶段，只采集 DB 现有项目")
    parser.add_argument("--skip-evict", action="store_true",
                        help="跳过 DB 淘汰阶段")
    parser.add_argument("--evict-grace-days", type=int, default=DB_EVICT_GRACE_DAYS,
                        help=f"连续多少份快照低于 {MIN_STAR} 星才淘汰（默认 {DB_EVICT_GRACE_DAYS}）")
    args = parser.parse_args()

    log_path = setup_logging()
    logger.info("=" * 70)
    logger.info("【每日 star 快照】日志: %s", log_path)

    # 幂等：当天已有快照就立刻退出，一个请求都不发。
    # 这样才能把定时任务设成每小时跑——机器在一天里任意一小时活着就能拿到当天的快照，
    # 而不是必须在某一分钟活着。漏掉的那天是补不回来的（今天无从得知昨天的 star 数），
    # 所以容错只能靠「多给自己机会」，不能靠事后补采。
    today = utc_today()
    if not args.force and args.limit <= 0 and load_snapshot(today) is not None:
        logger.info("当天（%s）快照已存在，跳过本次采集。加 --force 可强制重采。", today)
        return 0

    token_pool = GitHubTokenPool()

    # 发现失败不能阻断采集：DB 里已有的仓库照采，漏的明天补（每天都跑，自愈）。
    # 反过来不行——快照漏一天是补不回来的（今天无从得知昨天的 star 数）。
    if args.skip_discovery or args.limit > 0:
        logger.info("跳过发现阶段。")
    else:
        try:
            discover_and_register(token_pool)
        except Exception as e:  # noqa: BLE001 — 发现是增量累积，失败不该让当天丢快照
            logger.error("发现阶段失败，本次只采集 DB 现有项目（明天会重试）: %s", e)

    # DB 即观测宇宙：发现阶段刚插入的新仓库从这一刻起就有基线了。
    names = sorted(load_db().get("projects", {}))
    if not names:
        logger.error("DB 里没有任何项目，无从采集。")
        return 1
    if args.limit > 0:
        names = names[: args.limit]
        logger.info("调试模式：只采集前 %d 个仓库，不落盘。", len(names))

    # 算一次、既用于日志也传给采集，免得日志里的数和实际并发对不上。
    concurrency = max(1, min(SNAPSHOT_MAX_CONCURRENCY, token_pool.token_count))
    logger.info(
        "待采集 %d 个仓库，token %d 个，并发 %d（min(上限 %d, token 数)）。",
        len(names), token_pool.token_count, concurrency, SNAPSHOT_MAX_CONCURRENCY,
    )

    started = time.time()
    stars, failed = asyncio.run(collect_star_snapshot(
        token_pool, names, concurrency=concurrency,
        progress_cb=lambda done, total: logger.info("  进度 %d/%d 批", done, total),
    ))
    elapsed = time.time() - started

    coverage = len(stars) / len(names)
    missing = len(names) - len(stars) - len(failed)
    logger.info(
        "采集完成: %d/%d（覆盖率 %.1f%%），失败 %d，GitHub 查不到 %d，耗时 %.0fs。",
        len(stars), len(names), coverage * 100, len(failed), max(missing, 0), elapsed,
    )

    if args.limit > 0:
        logger.info("调试模式结束，未写入快照。")
        return 0

    if coverage < args.min_coverage:
        logger.error(
            "覆盖率 %.1f%% 低于下限 %.1f%%，拒绝落盘——宁可缺一天锚点（可顺延到邻近快照），"
            "也不写入一份可能有系统性错误的基线。",
            coverage * 100, args.min_coverage * 100,
        )
        return 1

    path = save_snapshot(today, stars)
    logger.info("快照已写入: %s（%.1fMB）", path, os.path.getsize(path) / 1e6)

    removed = prune_snapshots(args.keep_days)
    if removed:
        logger.info("清理过期快照 %d 份: %s ~ %s", len(removed), removed[0], removed[-1])
    kept = available_dates()
    logger.info("现存快照 %d 份: %s ~ %s", len(kept), kept[0], kept[-1])

    if not args.skip_evict:
        evict_from_db(args.evict_grace_days)

    logger.info("=" * 70)
    return 0


if __name__ == "__main__":
    sys.exit(main())
