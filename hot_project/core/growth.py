"""窗口内的 star 增长 —— 纯算术,不读盘不联网。

两条规则:窗口内最早那份快照里有它就相减;没有、但创建时间落在窗口内,那它现在的 star 全
是这个窗口涨的。两条都不成立(老仓库,而窗口内没有任何一份快照测到过它)就是**算不出来**,
`resolve` 返回 None、这一轮它不出现。

**算不出来不等于涨了 0。** 记 0 它会以「一点没涨」的身份进排名,永远出不了榜,而真实原因
(缺基线)没人看得见。返回 None 是为了让调用方想误用都用不了。

`window_days` 逐仓不同:基线是「窗口内最早测到它的那天」,而各仓进库时间不同。凡是按天数
折算速率的地方(爆发加成)都必须用各自的实际天数 —— 拿全局窗口去除,晚进库的仓库速率会虚高。
"""

from __future__ import annotations

from typing import NamedTuple

ANCHOR = "anchor"                 # 和窗口内最早那份快照相减
NEW_IN_WINDOW = "new_in_window"   # 窗口内新建,没有基线可减


class Growth(NamedTuple):
    """一个仓库的窗口增长、这个数怎么来的、以及它是按几天算出来的。"""

    value: int
    basis: str
    window_days: int


def resolve(
    current_star: int,
    anchor_star: int | None,
    anchor_days: int | None,
    age_days: float | None,
    window_days: int,
) -> Growth | None:
    """算一个仓库的窗口增长。算不出来返回 None —— 是「不参与」,不是零增长。

    `anchor_star=None` = 窗口内没快照测到过它,`age_days=None` = 不知道创建时间。
    增长可以是负的,不夹到零:抹平会让「掉了 300 星」和「一点没涨」再也分不开。
    """
    if anchor_star is not None:
        return Growth(current_star - anchor_star, ANCHOR, anchor_days or window_days)
    if age_days is not None and age_days <= window_days:
        # 有尾无头:窗口开始时它还不存在,所以现在的 star 全是这段时间涨的。
        # 天数取实际年龄而非整窗 —— 否则一个三天大的仓库日均速率被摊薄成三分之一。
        # 兜底到 1:今天刚建的仓库年龄不足一天,取 0 会让下游按天折算速率时除零。
        return Growth(current_star, NEW_IN_WINDOW, max(1, round(age_days)))
    return None
