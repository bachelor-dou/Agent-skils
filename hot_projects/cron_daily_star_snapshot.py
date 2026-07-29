#!/usr/bin/env python
"""每日 star 快照任务：拉全量 star 计数存成当天快照，并清理过期快照。

窗口增长的基线来源。GitHub 2026-06-30 起把 stargazers 列表限权给 admin/collaborator，
star 时间戳对他人仓库全部失效（二分法、采样外推同时报废），只能自己每天记一份计数。

用法：
    cd /root/code/Agent-skils/hot_projects
    /root/code/Agent-skils/.venv/bin/python3 cron_daily_star_snapshot.py
"""
# ============================================================
# 部署为定时任务（cron）—— 建议每小时跑一次，而不是每天一次：
#   本脚本幂等，当天已有快照就立刻退出、一个请求都不发，所以按小时跑几乎零成本。
#   好处是机器在一天里任意一小时活着就能拿到当天快照，而不必恰好在某一分钟活着。
#   漏掉的那天补不回来（今天无从得知昨天的 star 数），容错只能靠多给自己机会。
#   起点定在 08:00，早于周三 13:40 的榜单任务，保证出榜前当天快照已就位。
#
#   0 8-23 * * * . /root/.hot_projects.env && cd /root/code/Agent-skils/hot_projects && /root/code/Agent-skils/.venv/bin/python3 cron_daily_star_snapshot.py
#
# 日志：logs/YYYY-MM/snapshot-YYYY-MM-DD.log
# ============================================================
import argparse
import asyncio
import logging
import os
import sys
import time
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from hot_projects.config import (
    LOG_DIR,
    SNAPSHOT_KEEP_DAYS,
    SNAPSHOT_MIN_COVERAGE,
)
from hot_projects.datasource.github.star_snapshot import collect_star_snapshot
from hot_projects.datasource.github.token_pool import GitHubTokenPool
from hot_projects.infra.db import load_db
from hot_projects.infra.snapshots import (
    available_dates,
    load_snapshot,
    prune_snapshots,
    save_snapshot,
    utc_today,
)

logger = logging.getLogger("hot_projects")


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
    args = parser.parse_args()

    log_path = setup_logging()
    logger.info("=" * 70)
    logger.info("【每日 star 快照】日志: %s", log_path)

    # 幂等：当天已有快照就立刻退出，一个请求都不发。
    # 这样才能把 cron 设成每小时跑——机器在一天里任意一小时活着就能拿到当天的快照，
    # 而不是必须在 12:00 那一分钟活着。漏掉的那天是补不回来的（今天无从得知昨天的 star 数），
    # 所以容错只能靠「多给自己机会」，不能靠事后补采。
    today = utc_today()
    if not args.force and args.limit <= 0 and load_snapshot(today) is not None:
        logger.info("当天（%s）快照已存在，跳过本次采集。加 --force 可强制重采。", today)
        return 0

    names = sorted(load_db().get("projects", {}))
    if not names:
        logger.error("DB 里没有任何项目，无从采集。")
        return 1
    if args.limit > 0:
        names = names[: args.limit]
        logger.info("调试模式：只采集前 %d 个仓库，不落盘。", len(names))

    token_pool = GitHubTokenPool()
    logger.info("待采集 %d 个仓库，token %d 个。", len(names), token_pool.token_count)

    started = time.time()
    stars, failed = asyncio.run(collect_star_snapshot(
        token_pool, names,
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
    logger.info("=" * 70)
    return 0


if __name__ == "__main__":
    sys.exit(main())
