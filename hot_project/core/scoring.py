"""打分排序 —— **纯公式,不许读 DB**,补 `created_at` 之类是调用方的事。

    base  = log(1+增长) × 1000 + log2(1+增长率) × 1200
    score = base × 爆发加成

各项衡量什么:

    log(1+增长)      绝对热度
    log2(1+增长率)   爆发力,增长率 = 增长 / 总 star,让小项目的暴涨也能上榜
    爆发加成         最近速率高于整窗平均时的乘法加成,见 `burst_boost`

`1000 / 1200` 这一对配平的是两项在候选池里的**离散度**(不是量级 —— 全场同加一个常数
不改名次),目标是热度 : 增长率 ≈ 6.5 : 3.5。改偏好就动这对系数,按
`w2 = 1000 × (3.5/6.5) × σ(log(1+增长)) / σ(log2(1+增长率))` 重算。

**σ 必须在真候选池上量** —— 也就是过了 `STAR_GROWTH_THRESHOLD`(窗口增长 ≥ 1000)之后
那几十个,不是全库。阈值以下的仓库绝对增长跨度大得多,拿它们算 σ 会把热度项的话语权
高估一倍(同一个系数在全库上看是 7:3,在真候选池上是 4:6)。窗口越长增长率越大,σ 也
越大,所以系数还随窗口漂:实测 2 天窗口要 1231、7 天要 1112,1200 是两者的折中。

新项目榜(`hot_new`)不用这套公式,直接按增长量降序。

**这层不许读 DB。** 补 `created_at` 之类是调用方的事,这里只认送进来的候选。
"""

from __future__ import annotations

import math
from typing import NamedTuple

COMPREHENSIVE = "comprehensive"
HOT_NEW = "hot_new"


class Weights(NamedTuple):
    """爆发加成的三个旋钮。默认值由 config 提供,调用方传进来。"""

    window_days: int        # 全局名义窗口(最早那份快照的跨度),逐仓没给天数时的退路
    recent_days: int        # 探针的全局名义天数,同上
    alpha: float            # 加成强度
    cap: float              # 加速比的封顶,防止一个异常值把榜单掀翻


def burst_boost(growth: int, recent_growth: int | None, w: Weights, *,
                window_days: int | None = None, recent_days: int | None = None) -> float:
    """最近爆发加成:加速比 = 最近速率 / 整窗平均速率,大于 1 给加成,小于 1 不扣分。

    `recent_growth` 缺失时返回 1.0 —— 探针失效时榜单该退化成纯基础分,而不是全场一起挨罚。
    **两个分母必须都是各自基线的实际天数**(逐仓给,缺了才退回 `w` 里的全局值),否则速率
    虚高,凭空造出一场「爆发」:晚进库的仓库基线只有 3 天,按 7 天摊就是 2.3 倍的假加速。
    """
    if recent_growth is None or recent_growth < 0 or growth <= 0:
        return 1.0
    span = window_days or w.window_days
    recent_span = recent_days or w.recent_days
    if span <= 0 or recent_span <= 0:
        return 1.0
    avg_rate = growth / span
    if avg_rate <= 0:
        return 1.0
    acceleration = (recent_growth / recent_span) / avg_rate
    return 1.0 + w.alpha * min(max(acceleration - 1.0, 0.0), w.cap)


def score(item: dict, mode: str, w: Weights) -> float:
    """一个候选的分数。`item` 要有 `growth` 和 `star`;`recent_growth` 与两个天数可选。"""
    growth = item["growth"]
    star = item["star"]
    if mode != COMPREHENSIVE:
        return float(growth)
    if star <= 0:
        return float(growth)        # 0 星仓库算不出增长率,退化成纯增长量

    # 两处 max 都是必须的:`growth` 可以是负的(`growth.py` 刻意保留负增长),而
    # `log(1 + growth)` 对 growth ≤ −1 抛 math domain error —— 一个掉星的仓库就能掀翻
    # 整轮排名,而它本该只是得分很低。掉光全部 star 时 rate 到 −1,log2 那侧同理。
    rate = max(growth / star, 0.0)
    base = math.log1p(max(growth, 0)) * 1000 + math.log2(1 + rate) * 1200
    return base * burst_boost(growth, item.get("recent_growth"), w,
                              window_days=item.get("window_days"),
                              recent_days=item.get("recent_days"))


def rank(candidates: dict[str, dict], mode: str, w: Weights) -> list[tuple[str, dict]]:
    """按分数降序排全部候选,同时把 `_score` 写回每个候选(日志和调试要看)。

    不截断 Top N:同一份排序结果常要按不同 top_n 复用。
    """
    for info in candidates.values():
        info["_score"] = score(info, mode, w)
    return sorted(candidates.items(), key=lambda kv: kv[1]["_score"], reverse=True)
