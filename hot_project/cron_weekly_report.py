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
import logging
import sys

from . import config
from .common import logs
from .common.timeutil import format_day, utc_today
from .core import report_parse
from .infra import notify
from .infra.store import reports
from .provider.github import client as github
from .tools import ranking

logger = logging.getLogger("hot_project")

TOP_IN_PUSH = 5


def previous_report(current: str) -> tuple[str, report_parse.Report] | None:
    """找上一期**同类型**的报告(后缀相同、日期更早)。没有返回 None。

    必须按后缀配对:拿 `_NEW` 那份当综合榜的上期,「上新 / 移出」全是噪音。
    """
    day = reports.day_of(current)
    if day is None:
        return None
    suffix = current[len(str(day)):]

    # 必须显式按日期排,不能拿 `listing()` 的第一条 —— 那是按**修改时间**倒序的,而 CI
    # 每次全新 checkout,所有报告 mtime 几乎相同、先后随机。挑错上期,推送里的
    # 「上新 N · 移出 M」就全是错的,收到的人没有任何办法看出来。
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
