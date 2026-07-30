#!/usr/bin/env python
"""每周综合榜 —— 读每日快照出榜、写报告、推一条微信摘要。

    python -m hot_project.cron_weekly_report --top-n 100

## 它现在做的事比以前少得多

旧版在这里跑三阶段扫描(333 个关键词 + 星段 + Trending),一轮几十分钟、两百次限流。
现在候选池就是今天的快照:每日任务已经用同一个 `MIN_STAR` 扫过一遍、并给 DB 里每个仓库
采了当天 star。周报再扫一遍是几十分钟换零个新候选 —— 不在 DB/快照里的仓库拿不到锚点,
增长必然未决,本来就出不了榜。

所以整条链路只有「写介绍」那一步慢(Top N 逐个调 LLM),其余全是内存里的字典运算。

**前提是每日快照跑成功了。** 今天的快照缺失时这里直接报错退出,不去兜底扫一遍:
那个兜底路径一年用不上一次,却要养着整套三阶段代码。快照缺了就先补跑每日任务,
它本来就是幂等的。

## 旧版删掉的两百行

`log_update_summary` 逐字段 diff 新旧 DB(refreshed_at 变了几个、gh_desc 空转非空几个)。
那是 DB 整体覆写年代的产物 —— 那时候一次 `save_db` 会动几十个字段,不 diff 没人知道
改了什么。现在每次写库都是一个事务,函数名就写着它能动哪些字段,而且各自记了改动条数。
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
from .provider.github import facade
from .tools import ranking

logger = logging.getLogger("hot_project")

TOP_IN_PUSH = 5


def previous_report(current: str) -> tuple[str, report_parse.Report] | None:
    """找上一期**同类型**的报告(后缀相同、日期更早)。没有返回 None。

    按后缀配对是必须的:综合榜要和综合榜比。拿 `_NEW` 那份当上期,算出来的
    「上新 / 移出」全是噪音 —— 两张榜本来就没几个重合的。
    """
    day = reports.day_of(current)
    suffix = current[len(str(day)):] if day else ""
    for item in reports.listing():        # listing 按修改时间倒序,这里要按日期找
        if item.day is None or day is None or item.day >= day:
            continue
        if item.name[len(str(item.day)):] != suffix:
            continue
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
        do_report=not args.no_report, gh=facade.get(),
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
                   help="增长窗口(天);当天缺快照会自动顺延到邻近那天")
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
