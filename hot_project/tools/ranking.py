"""榜单流水线 —— 综合 / 新项目 / 关键词三榜共用。

    候选池 → 增长 → 阈值 → 爆发探针 → 打分 → 报告

**整条链路零 GitHub 请求**(报告里生成描述除外)。旧版在这里跑三阶段扫描,一轮几十分钟、
撞两百次限流。现在候选池就是今天的快照,增长是查表相减 —— 因为一个不在 DB/快照里的仓库
根本拿不到锚点,增长必然未决、进不了榜,扫它是几十分钟换零个候选。

## 窗口只有一个来源

请求方说「近 7 天」,但真正生效的是锚点的实际跨度(缺快照会顺延)。两者不一致时一律
以锚点为准,并且报告标题里写实际天数。旧代码在这件事上有两套值并存,导致新仓库被误判
出局(见 `core/growth.py`)。

## 未决不是零

算不出增长的仓库直接不进候选池,而不是记 0 —— 记 0 它会以「一点没涨」的身份参与排名,
既永远出不了榜,也没人看得出真实原因是缺基线。漏斗里单独报这个数。
"""

from __future__ import annotations

import json
import logging

from .. import config
from ..common.timeutil import age_days, utc_today
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


def candidates(min_star: int) -> dict[str, dict]:
    """候选池 = 今天的快照 ∩ star 达标。零请求。

    star 取快照而不是 DB 里的值:DB 的 star 只在每日任务里刷新,而增长的减数(锚点)
    是快照,被减数必须来自同一条链,否则窗口两端口径不一致。

    **不套 max_star**:那是星段扫描的分段上限(避免浪费 Search 页),对读库毫无意义,
    照搬会把两万星以上的仓库整批挡在榜外。
    """
    stars = snapshots.load_stars(utc_today())
    if not stars:
        logger.error("今天的快照还没落盘,榜单无从算起(每日任务挂了?)。")
        return {}
    saved = universe.load()
    pool = {
        name: {"star": star, "created_at": saved.get(name, {}).get("created_at", "")}
        for name, star in stars.items() if star >= min_star
    }
    logger.info("候选池:%d 个(快照共 %d,star >= %d),零请求。",
                len(pool), len(stars), min_star)
    return pool


def growths(pool: dict[str, dict], window_days: int) -> tuple[dict[str, dict], int, str]:
    """给候选池算增长。返回 `(算得出来的候选, 实际窗口天数, 口径摘要)`。

    算不出来的直接不在返回值里 —— 见模块头部「未决不是零」。
    """
    anchor = snapshots.anchor_for_window(window_days)
    if anchor is None:
        logger.error("找不到 T−%d 天附近的快照,本轮算不了增长。", window_days)
        return {}, window_days, "无锚点"
    if anchor.window_days != window_days:
        logger.warning("锚点顺延到 %s:实际窗口 %d 天而非请求的 %d 天,全程按实际天数。",
                       anchor.day, anchor.window_days, window_days)

    tally = growth_calc.resolve_all(
        stars={n: info["star"] for n, info in pool.items()},
        anchor_stars=anchor.stars,
        ages={n: age_days(info.get("created_at", "")) for n, info in pool.items()},
        window_days=anchor.window_days,
    )
    decided = {
        name: {**pool[name], "growth": value}
        for name, value in tally.decided.items()
    }
    logger.info("增长口径:%s。", tally.summary())
    return decided, anchor.window_days, tally.summary()


def recent(pool: dict[str, dict], days: int) -> tuple[dict[str, int], int]:
    """爆发探针:最近几天的增长,同样是锚点相减。返回 `(增长表, 实际天数)`。

    以前靠实时二分法逐个仓库发请求(上期三分钟,而 2026-06 后 stargazers 全部 404,
    结果必然为空)。现在是一次查表:零请求,而且拿到的是真实测得的短窗口速率,
    不再是把周均速原样折算过来(那样加速比恒为 1,探针纯空转)。

    缺快照时返回空表 —— 探针不加成即可,绝不能让出榜失败。
    """
    if not pool:
        return {}, days
    anchor = snapshots.anchor_for_window(days)
    if anchor is None:
        logger.info("没有 T−%d 天附近的快照,本轮跳过爆发加成。", days)
        return {}, days
    out = {
        name: info["star"] - base
        for name, info in pool.items()
        if isinstance(base := anchor.stars.get(name), int) and info["star"] >= base
    }
    logger.info("爆发探针锚点 %s(%d 天):%d/%d 个候选可算。",
                anchor.day, anchor.window_days, len(out), len(pool))
    return out, anchor.window_days


def run(*, mode: str = "comprehensive", min_star: int = config.MIN_STAR,
        growth_threshold: int = config.STAR_GROWTH_THRESHOLD,
        growth_days: int = config.GROWTH_CALC_DAYS,
        created_days: int | None = None, top_n: int | None = None,
        topic: str | None = None, do_report: bool = True,
        gh=None, progress=None, pool: dict[str, dict] | None = None) -> dict:
    """跑一轮榜单。返回排名结果 + 漏斗。

    `pool` 给了就用给的(关键词榜按搜索结果筛过一遍),否则取今天的整份快照。
    """
    _emit(progress, 5, "读取候选池…")
    if pool is None:
        pool = candidates(min_star)
    collected = len(pool)

    _emit(progress, 20, "计算增长…")
    scored_pool, window, basis = growths(pool, growth_days)

    _emit(progress, 45, "筛选达标候选…")
    qualified = {n: i for n, i in scored_pool.items() if i["growth"] >= growth_threshold}
    logger.info("达标候选(增长 >= %d):%d 个。", growth_threshold, len(qualified))

    if mode == "hot_new":
        window_created = created_days if created_days is not None else config.DAYS_SINCE_CREATED
        qualified = {
            n: i for n, i in qualified.items()
            if (age := age_days(i.get("created_at", ""))) is not None and age <= window_created
        }
        logger.info("新项目窗口(<= %d 天):剩 %d 个。", window_created, len(qualified))

    _emit(progress, 55, "爆发探针…")
    recent_window = config.RECENT_GROWTH_DAYS
    boosted = 0
    if mode != "hot_new" and qualified:
        probed, recent_window = recent(qualified, config.RECENT_GROWTH_DAYS)
        for name, value in probed.items():
            qualified[name]["recent_growth"] = value
        if window > 0 and recent_window > 0:
            boosted = sum(1 for i in qualified.values()
                          if (rg := i.get("recent_growth")) is not None and i["growth"] > 0
                          and rg / recent_window > i["growth"] / window)
        logger.info("爆发探针:%d 个候选可算,加成生效 %d 个。", len(probed), boosted)

    _emit(progress, 65, "排序…")
    weights = scoring.Weights(window_days=window, recent_days=recent_window,
                              alpha=config.BURST_ALPHA, cap=config.BURST_CAP)
    rank_mode = scoring.HOT_NEW if mode == "hot_new" else scoring.COMPREHENSIVE
    ordered = scoring.rank(qualified, rank_mode, weights)
    ranked = ordered[:top_n] if top_n else ordered

    if top_n and len(ranked) < top_n:
        logger.warning("达标候选不够:要 %d 个,只有 %d 个(候选池 %d,达标 %d)。",
                       top_n, len(ranked), collected, len(qualified))

    funnel = Funnel(收集=collected, 增长可算=len(scored_pool), 达标=len(qualified),
                    爆发加成=boosted, 出榜=len(ranked))
    logger.info("漏斗:%s(%s)", funnel.line(), basis)

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
