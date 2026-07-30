"""打分排序 —— 纯公式,不读 DB 不发请求。

    base  = (log(1+增长) × 1000 + log2(1+增长率) × 3000) × 折扣
    score = base × 爆发加成

各项在干什么:

    log(1+增长)      绝对热度。取对数是因为 star 增长的分布跨几个数量级,
                     不压一下的话榜单永远是那几个巨无霸。
    log2(1+增长率)   爆发力,增长率 = 增长 / 总 star。让小项目的暴涨也能上榜。
    折扣             增长率 > 0.5 时线性打折(最多 -15%),压住低基数虚高
                     —— 100 星涨到 200 的「翻倍」和 10000 星涨 1 万不是一回事。
    爆发加成         最近几天的速率显著高于整窗平均时的乘法加成,见 `burst_boost`。

新项目榜(`hot_new`)不用这套公式,直接按增长量降序:那张榜比的就是「谁涨得多」,
再叠一层增长率会让创建三天涨 200 星的项目压过创建三十天涨 3000 星的。

**这层不许读 DB。** 旧版打分函数里有一行 `db.get("projects", ...)` 用来补
`created_at` —— 一个纯公式伸手去读数据库,于是想验证一次排名要先造一个 DB。
补数据是调用方的事,这里只认送进来的候选。
"""

from __future__ import annotations

import math
from typing import NamedTuple

COMPREHENSIVE = "comprehensive"
HOT_NEW = "hot_new"

_RATE_KNEE = 0.5        # 增长率超过这里开始打折
_MAX_DISCOUNT = 0.15    # 最多打掉 15%


class Weights(NamedTuple):
    """爆发加成的三个旋钮。默认值由 config 提供,调用方传进来。"""

    window_days: int        # 主窗口实际天数(锚点跨度,不是请求值)
    recent_days: int        # 爆发探针的实际天数(同样是锚点跨度)
    alpha: float            # 加成强度
    cap: float              # 加速比的封顶,防止一个异常值把榜单掀翻


def burst_boost(growth: int, recent_growth: int | None, w: Weights) -> float:
    """最近爆发加成:乘法、封顶、不反向惩罚。

    加速比 = 最近速率 / 整窗平均速率。大于 1 说明这几天在提速,给加成;
    小于 1 不扣分 —— 「前期涨得猛、最近平了」的项目本来就该靠基础分排,
    再罚一次等于对同一件事收两遍税。

    `recent_growth` 缺失时返回 1.0。缺快照的日子探针整个失效,那时候榜单该退化成
    纯基础分,而不是全场一起挨罚(那样榜单顺序其实没变,只是分数集体缩水,
    看日志的人却会以为打分坏了)。

    **两个分母必须都是锚点的实际天数。** 主窗口那侧修正了、探针这侧没修正的话,
    5 天的增量除以请求的 3 天,速率虚高 67%,凭空造出一场「爆发」。
    """
    if recent_growth is None or recent_growth < 0 or growth <= 0:
        return 1.0
    if w.window_days <= 0 or w.recent_days <= 0:
        return 1.0
    avg_rate = growth / w.window_days
    if avg_rate <= 0:
        return 1.0
    acceleration = (recent_growth / w.recent_days) / avg_rate
    return 1.0 + w.alpha * min(max(acceleration - 1.0, 0.0), w.cap)


def score(item: dict, mode: str, w: Weights) -> float:
    """一个候选的分数。`item` 要有 `growth` 和 `star`,`recent_growth` 可选。"""
    growth = item["growth"]
    star = item["star"]
    if mode != COMPREHENSIVE:
        return float(growth)
    if star <= 0:
        return float(growth)        # 0 星仓库算不出增长率,退化成纯增长量

    rate = growth / star
    base = math.log(1 + growth) * 1000 + math.log2(1 + rate) * 3000
    discount = (1.0 - _MAX_DISCOUNT * min((rate - _RATE_KNEE) / _RATE_KNEE, 1.0)
                if rate > _RATE_KNEE else 1.0)
    return base * discount * burst_boost(growth, item.get("recent_growth"), w)


def rank(candidates: dict[str, dict], mode: str, w: Weights) -> list[tuple[str, dict]]:
    """按分数降序排全部候选,顺手把 `_score` 写回每个候选(日志和调试要看)。

    不截断 Top N:截断是调用方的事,而且同一份排序结果常要按不同 top_n 复用。
    """
    for info in candidates.values():
        info["_score"] = score(info, mode, w)
    return sorted(candidates.items(), key=lambda kv: kv[1]["_score"], reverse=True)
