"""单仓库卡片的数据源:star 轨迹、实时增长、刷新介绍。

报告页项目卡片的按钮(「star 走势」「刷新介绍」)和 agent 的单仓库工具
(star_trend / repo_growth)共用这里的实现 —— 「网页和 agent 永远说同一个数」
靠的是只有这一份代码,不是靠两边记得去 import 同一个函数。

HTTP 路由和工具契约只做各自的边界事:仓库名校验、HTTP 状态码映射、args 提取、
确认守卫;取数、算增长、落库的编排都在这。
"""

from __future__ import annotations

import logging

from ..common.timeutil import age_days
from ..infra.data_access import reports, snapshots
from . import growth as growth_calc
from . import report

logger = logging.getLogger("hot_project")


def trend(repo: str) -> dict:
    """从历次报告拼出一个项目的 star 轨迹。只覆盖上过榜的周;没上榜的那周就缺点。"""
    series = []
    for item, parsed in reports.load_all():
        entry = parsed.find(repo)
        if entry is None:
            continue
        series.append({"date": str(item.day), "repo": entry.repo, "rank": entry.rank,
                       "star": reports.number_of(entry.metadata.get("总 Star", "")),
                       "growth": reports.growth_of(entry.metadata)})

    if not series:
        return {"repo": repo, "points": 0,
                "message": "这个项目没在历史报告里出现过,给不出 star 轨迹。"}

    first, last = series[0]["star"], series[-1]["star"]
    return {
        "repo": series[-1]["repo"], "points": len(series),
        "span": f"{series[0]['date']} → {series[-1]['date']}",
        "star_change": (last - first) if first is not None and last is not None else None,
        "series": series,
        "hint": "series 按时间升序,star=当期总 star、rank=当期排名;据此判断在涨/见顶/退烧。"
                "缺的那期表示当期未上榜。",
    }


def live_growth(name: str, gh, days: int) -> dict:
    """实时 star 减窗口内最早那份快照。agent 的 repo_growth 和报告页刷新按钮共用这一份。

    `star=None` 是取不到当前 star,`growth=None` 是缺基线**算不出**(不是涨了 0);
    `growth_calc_days` 是实际跨度,可能小于请求的 `days`。
    """
    info = gh.info(name) or {}
    star = info.get("stargazers_count")
    out = {"star": star, "growth": None, "growth_basis": "", "growth_calc_days": days}
    if star is None:
        return out

    base = snapshots.earliest_in_window(days)
    base_star, base_days = base.get(name, info.get("id"))
    result = growth_calc.resolve(star, base_star, base_days,
                                 age_days(info.get("created_at", "")), base.span)
    if result is not None:
        out.update(growth=result.value, growth_basis=result.basis,
                   growth_calc_days=result.window_days)
    return out


def refresh(name: str, gh, days: int) -> dict:
    """重跑介绍生成并落库,顺带回传当下的 star 与窗口增长。

    star/增长走 `live_growth`,和 agent 的 repo_growth 同一份算法,所以两边永远说同一个数;
    它拿不到就只回介绍,不能因此让刷新整个失败。介绍生成失败时 `desc` 为空串,怎么报错
    (HTTP 502 还是工具 error dict)由调用方定。
    """
    desc = report.regenerate(name, gh)
    if not desc:
        return {"desc": ""}
    try:
        stats = live_growth(name, gh, days)
    except Exception:       # noqa: BLE001 —— 刷新按钮的主产品是介绍
        logger.warning("刷新 %s 的 star/增长失败,只回介绍。", name, exc_info=True)
        stats = {}
    return {"desc": desc, **stats}
