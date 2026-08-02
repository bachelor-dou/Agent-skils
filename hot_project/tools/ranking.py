"""榜单流水线 —— 综合 / 新项目 / 关键词三榜共用。

    候选名单 → 实时 star → 增长 → 阈值 → 爆发探针 → 打分 → 报告

**当前 star 一律实时取,不读当天的快照。** 读当天快照会把两个定时任务焊死在一起:榜单的
数字锚死在每日采集那一刻,采集到出榜之间涨的 star 凭空消失,而且当天采集没跑成榜单就出不来。
快照只负责一件事 —— 给出窗口内**最早**那天的基线。

两条不变量:

- **达标才进内存。** 增长低于阈值的当场丢:它既出不了榜,也不参与探针和排序。7.8 万个候选
  收到几百个,省下的是几十 MB 和之后每一步的全表遍历。
- **算不出增长的不参与。** 窗口内没有任何快照测到过它、创建时间又不在窗口内 → 这轮它不
  出现。记 0 的话它会以「一点没涨」的身份进排名,永远出不了榜,也没人看得出原因。
"""

from __future__ import annotations

import logging

from .. import config
from ..common.timeutil import age_days
from ..core import growth as growth_calc
from ..core import scoring
from ..infra.store import snapshots, universe
from . import report as report_tool

logger = logging.getLogger("hot_project")

MODES = ("comprehensive", "hot_new", "keyword")


class Funnel(dict):
    """每一层剩下多少。出榜数少于预期时,唯一能回答「卡在哪一层」的东西。"""

    def line(self) -> str:
        return " → ".join(f"{k} {v}" for k, v in self.items())


def _emit(progress, percent: int, label: str) -> None:
    """回传进度。回调自己抛异常绝不能影响流水线 —— 那只是给前端画进度条的。"""
    if progress is None:
        return
    try:
        progress(percent, label)
    except Exception:       # noqa: BLE001
        logger.debug("进度回调异常已忽略", exc_info=True)


def current_stars(gh, names: list[str]) -> dict[str, int]:
    """实时取这批仓库此刻的 star。批量走任务池 + token 池,和每日采集同一条路。

    只取 `stars`:「查不到」和「没问到」在这里都只意味着它这轮排不了名,两者的区别只对淘汰判定有意义。
    """
    if gh is None or not getattr(gh, "usable", False):
        logger.error("没有可用的 GitHub token,取不到当前 star,本轮无从算起。")
        return {}
    harvest = gh.stars(names)
    logger.info("实时取 star:请求 %d 个,取到 %d,GitHub 查不到 %d,没问到 %d。",
                len(names), len(harvest.stars), len(harvest.missing), len(harvest.failed))
    return harvest.stars


def qualify(stars: dict[str, int], meta: dict[str, dict], base: snapshots.Baseline,
            *, min_star: int, threshold: int) -> tuple[dict[str, dict], int]:
    """边算边筛:算一个增长,达标才留。返回 `(候选池, 缺基线算不出的个数)`。

    低于阈值的当场丢 —— 先建全量表再过滤的话,那张表七万多条、几十 MB,建出来只为下一行扔掉。
    """
    pool: dict[str, dict] = {}
    unresolved = 0
    for name, star in stars.items():
        if star < min_star:
            continue
        created = meta.get(name, {}).get("created_at", "")
        result = growth_calc.resolve(star, base.stars.get(name), base.days.get(name),
                                     age_days(created), base.span)
        if result is None:
            unresolved += 1
            continue
        if result.value < threshold:
            continue
        pool[name] = {"star": star, "growth": result.value,
                      "window_days": result.window_days, "created_at": created}
    return pool, unresolved


def recent(pool: dict[str, dict], days: int) -> int:
    """爆发探针:把最近几天的增长写回每个候选。返回名义天数。

    缺快照时什么都不写 —— 探针不加成即可,绝不能让出榜失败。
    """
    if not pool:
        return days
    base = snapshots.earliest_in_window(days)
    if base.oldest is None:
        logger.info("最近 %d 天内没有快照,本轮跳过爆发加成。", days)
        return days

    hit = 0
    for name, info in pool.items():
        anchor = base.stars.get(name)
        # 掉星的跳过:负的最近增长喂进加速比是没有意义的输入。
        if anchor is None or info["star"] < anchor:
            continue
        info["recent_growth"] = info["star"] - anchor
        info["recent_days"] = base.days.get(name, base.span)
        hit += 1
    logger.info("爆发探针基线 %s(%d 天):%d/%d 个候选可算。",
                base.oldest, base.span, hit, len(pool))
    return base.span


def run(*, mode: str = "comprehensive", min_star: int = config.MIN_STAR,
        growth_threshold: int = config.STAR_GROWTH_THRESHOLD,
        growth_days: int = config.GROWTH_CALC_DAYS,
        created_days: int | None = None, top_n: int | None = None,
        topic: str | None = None, do_report: bool = True,
        gh=None, progress=None, pool: dict[str, dict] | None = None) -> dict:
    """跑一轮榜单。返回排名结果 + 漏斗。

    `pool` 给了就只在那批里排,否则排 DB 全库。它只提供名单和 `created_at`,star 一律现取。
    """
    _emit(progress, 5, "读取候选名单…")
    meta = universe.load() if pool is None else pool
    names = sorted(meta)
    collected = len(names)

    _emit(progress, 15, "实时取当前 star…")
    stars = current_stars(gh, names)

    _emit(progress, 50, "计算增长,筛选达标候选…")
    base = snapshots.earliest_in_window(growth_days)
    window, unresolved = growth_days, 0
    qualified: dict[str, dict] = {}
    if base.oldest is None:
        logger.error("最近 %d 天内一份快照都没有,算不了增长(每日任务挂了?)。", growth_days)
    else:
        window = base.span
        if window != growth_days:
            logger.warning("窗口内最早的快照是 %s:实际窗口 %d 天而非请求的 %d 天,全程按实际算。",
                           base.oldest, window, growth_days)
        qualified, unresolved = qualify(stars, meta, base,
                                        min_star=min_star, threshold=growth_threshold)
    logger.info("达标候选(增长 >= %d):%d 个;另有 %d 个缺基线算不出增长。",
                growth_threshold, len(qualified), unresolved)

    if mode == "hot_new":
        window_created = created_days if created_days is not None else config.DAYS_SINCE_CREATED
        qualified = {
            n: i for n, i in qualified.items()
            if (age := age_days(i.get("created_at", ""))) is not None and age <= window_created
        }
        logger.info("新项目窗口(<= %d 天):剩 %d 个。", window_created, len(qualified))

    _emit(progress, 60, "爆发探针…")
    recent_window = config.RECENT_GROWTH_DAYS
    boosted = 0
    if mode != "hot_new" and qualified:
        recent_window = recent(qualified, config.RECENT_GROWTH_DAYS)
        boosted = sum(1 for i in qualified.values()
                      if (rg := i.get("recent_growth")) is not None and i["growth"] > 0
                      and rg / i["recent_days"] > i["growth"] / i["window_days"])
        logger.info("爆发加成生效 %d 个。", boosted)

    _emit(progress, 70, "排序…")
    weights = scoring.Weights(window_days=window, recent_days=recent_window,
                              alpha=config.BURST_ALPHA, cap=config.BURST_CAP)
    rank_mode = scoring.HOT_NEW if mode == "hot_new" else scoring.COMPREHENSIVE
    ordered = scoring.rank(qualified, rank_mode, weights)
    ranked = ordered[:top_n] if top_n else ordered

    if top_n and len(ranked) < top_n:
        logger.warning("达标候选不够:要 %d 个,只有 %d 个(名单 %d,取到 star %d,达标 %d)。",
                       top_n, len(ranked), collected, len(stars), len(qualified))

    funnel = Funnel(名单=collected, 取到star=len(stars), 达标=len(qualified),
                    爆发加成=boosted, 出榜=len(ranked))
    logger.info("漏斗:%s", funnel.line())

    result = {"mode": mode, "ranked": ranked, "growth_days": window,
              "recent_days": recent_window, "funnel": dict(funnel)}

    if do_report:
        _emit(progress, 75, "生成报告…")
        result["report_path"] = report_tool.generate(
            ranked, mode=mode, growth_days=window, growth_threshold=growth_threshold,
            min_star=min_star, created_days=created_days, topic=topic, gh=gh,
            progress=lambda frac, label: _emit(progress, 75 + int(24 * frac), label),
        )
    _emit(progress, 100, "完成")
    return result
