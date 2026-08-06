#!/usr/bin/env python
"""每周综合榜 —— 实时取 star 减快照基线出榜、写报告、推一条微信摘要。

    python -m hot_project.cron_weekly_report --top-n 100

候选名单就是 DB 全库,不再自己扫 GitHub 找新仓库:每日任务已经用同一个 `MIN_STAR` 扫过一遍。
慢的只有两步 —— 实时取全库当前 star,和给 Top N 逐个调 LLM 写介绍。

**它不依赖当天那次快照。** 当前 star 是现取的,基线取窗口内最早那份快照,所以这个任务和
每日任务只共享历史快照、没有时序耦合:今天的采集没跑成,周报照样出得来,数字也不会缺掉
「采集时刻 → 出榜时刻」之间涨的那些 star。
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys

from . import config
from .common import logs
from .common.timeutil import format_day, utc_today
from .infra import notify
from .infra.data_access import reports
from .provider.github import client as github
from .provider.github import request as gh_request
from .provider.github import trending
from .service import ranking
from .service import report as report_tool

logger = logging.getLogger("hot_project")

TOP_IN_PUSH = 5
TRENDING_PERIODS = ("weekly", "monthly")


def previous_report(current: str) -> tuple[str, reports.Report] | None:
    """找上一期**同类型**的报告(后缀相同、日期更早)。没有返回 None。

    必须按后缀配对:拿 `_NEW` 那份当综合榜的上期,「上新 / 移出」全是噪音。
    """
    day = reports.day_of(current)
    if day is None:
        return None
    suffix = current[len(str(day)):]

    earlier = sorted(
        (item for item in reports.listing()
         if item.day is not None and item.day < day
         and item.name[len(str(item.day)):] == suffix),
        key=lambda item: item.day, reverse=True,
    )
    for item in earlier:
        if (report := reports.load(item.name)) is not None:
            return item.name, report
    return None


def push(result: dict) -> None:
    """推一条微信摘要:Top5 + 较上期的上新/移出。没配 key 就静默跳过。"""
    ranked = result.get("ranked") or []
    path = result.get("report_path") or ""
    name = path.rsplit("/", 1)[-1]

    parts = [f"综合榜已更新,共 {len(ranked)} 个项目。"]
    if (previous := previous_report(name)) is not None:
        prev_name, prev = previous
        current = {n for n, _ in ranked}
        before = {e.repo for e in prev.entries}
        parts.append(f"较上期({prev_name}):上新 {len(current - before)} · "
                     f"移出 {len(before - current)}")
    if ranked:
        parts.append("**Top5**\n" + "\n".join(
            f"{i}. {n} (+{info['growth']:,}, {info['star']:,}★)"
            for i, (n, info) in enumerate(ranked[:TOP_IN_PUSH], 1)))
    parts.append(f"报告:{name}")
    notify.send(f"GitHub 周报 {format_day(utc_today())}", "\n\n".join(parts))


async def _fetch_trending(period: str) -> list[dict]:
    client = gh_request.build_client()
    try:
        return await trending.fetch_trending(client, period)
    finally:
        await client.aclose()


def attach_trending(result: dict) -> None:
    """报告尾部接周榜和月榜两段 Trending 对照。

    两个周期各自独立:月榜抓挂了不该连累已经写好的周榜,所以每个周期单独 try 单独记日志。
    任何失败都只记日志 —— 附栏绝不能搞砸周报。
    """
    path = result.get("report_path")
    if not path:
        return
    for period in TRENDING_PERIODS:
        try:
            rows = asyncio.run(_fetch_trending(period))
        except Exception:       # noqa: BLE001
            logger.exception("Trending %s 抓取失败,本期没有这段对照附栏。", period)
            continue
        if not rows:
            logger.warning("Trending %s 解析出 0 条,跳过这段附栏。", period)
            continue
        if report_tool.append_trending(path, rows, result["ranked"],
                                       gh=github.shared(), period=period):
            logger.info("Trending 对照附栏已追加(%s):%d 条。", period, len(rows))
        else:
            logger.error("Trending 对照附栏(%s)写入失败,报告正文不受影响。", period)


def run(args: argparse.Namespace) -> int:
    logger.info("周报开始:top_n=%d,窗口 %d 天,最低 star %d,增长阈值 %d。",
                args.top_n, args.growth_days, config.MIN_STAR,
                config.STAR_GROWTH_THRESHOLD)

    result = ranking.run(
        mode="comprehensive", min_star=config.MIN_STAR,
        growth_threshold=config.STAR_GROWTH_THRESHOLD,
        growth_days=args.growth_days, top_n=args.top_n,
        do_report=not args.no_report, gh=github.shared(),
    )

    if not result["ranked"]:
        logger.error("一个项目都没出榜。漏斗:%s", result["funnel"])
        return 1
    if not args.no_report and not result.get("report_path"):
        logger.error("榜单算出来了但报告没写成(看上面的落盘错误)。")
        return 1
    if not args.no_report and not args.no_trending:
        attach_trending(result)

    logger.info("周报完成:%d 个项目,报告 %s", len(result["ranked"]),
                result.get("report_path") or "(未生成)")
    if not args.no_push:
        push(result)
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description="每周综合榜")
    p.add_argument("--top-n", type=int, default=config.HOT_PROJECT_COUNT)
    p.add_argument("--growth-days", type=int, default=config.GROWTH_CALC_DAYS,
                   help="增长窗口(天);基线取窗口内最早那份快照,快照不齐时实际窗口会更短")
    p.add_argument("--no-report", action="store_true", help="只算榜单不写报告(快)")
    p.add_argument("--no-push", action="store_true", help="不推微信")
    p.add_argument("--no-trending", action="store_true", help="不追加 Trending 对照附栏")
    args = p.parse_args()

    log_path = logs.setup(config.LOG_DIR, "weekly", day=utc_today())
    logger.info("=" * 70)
    logger.info("【每周综合榜】日志:%s", log_path)
    try:
        code = run(args)
    except Exception:       # noqa: BLE001 —— cron 里要留下完整栈,别只给一行退出码
        logger.exception("周报异常终止")
        code = 1
    logger.info("=" * 70)
    return code


if __name__ == "__main__":
    sys.exit(main())
