"""窗口内的 star 增长 —— 纯算术,不读盘不联网。

GitHub 2026-06-30 起把 stargazers 列表限权,star 时间戳对他人仓库全部失效,二分法和采样
外推同时报废。剩下唯一能还原任意窗口增长的办法是自己每天记一份计数,增长就是两个数相减。

## 只有两条规则,其余一律未决

    1. 锚点相减      锚点快照里有这个仓库  →  当前 star − 锚点 star      精确
    2. 窗口内新建    锚点里没有,但创建时间落在窗口内(有尾无头)
                     →  当前 star 全都是这个窗口涨的                    精确
    3. 其余          未决

**「未决」不是零增长。** 它是「这个仓库我们还算不出来」:不进候选池、不写 DB,等快照攒够
下一轮自然就算出来了。当成零增长的话它会以「涨了 0」的身份参与排名,永远出不了榜,
而且没人知道它其实只是缺一份基线。

规则 1 优先于规则 2:两者本不该同时成立(窗口内新建的仓库在锚点那天还不存在),真同时
成立说明 `created_at` 和快照对不上,那就信快照 —— 实测数据比元数据可靠。

## 窗口天数只有一个来源:锚点的实际跨度

旧代码在这里有个会让榜单少人的 bug。锚点顺延时(请求 7 天、当天缺快照顺延到 T−9),
它把 `window_days` 改成 9 只用于报告口径,规则 2 里比 `created_at` 却仍用请求值 7。于是
一个 8 天前创建的仓库:锚点里没有(9 天前它还不存在),`8 <= 7` 又为假,两条规则都落空,
被记成未决剔出排名 —— 可它本该精确算出来(锚点窗口实际 9 天,仓库才 8 天大,全部 star
都是这 9 天涨的)。而缺快照顺延的日子,恰恰正是新仓库最多的日子。

所以本模块**不接受**「请求的窗口」这个参数,只接受锚点带来的实际跨度。
`infra.store.snapshots.Anchor` 把天数和 star 表绑在一个 NamedTuple 里,就是为了让
「忘记用实际跨度」这件事在类型上做不到。
"""

from __future__ import annotations

from typing import NamedTuple

ANCHOR = "anchor"                 # 锚点相减
NEW_IN_WINDOW = "new_in_window"   # 窗口内新建
UNDECIDED = "undecided"           # 算不出来


class Growth(NamedTuple):
    """一个仓库在某窗口内的增长,以及这个数是怎么来的。

    `basis` 不是给日志看的装饰:排名漏斗要按它分类计数,好回答「这期少了 1700 个候选,
    是因为快照不够还是因为它们真没涨」。旧代码那个字段叫 `db_diff`,是早就删掉的算法
    留下的名字,数出来的东西和名字对不上。
    """

    value: int | None
    basis: str

    @property
    def decided(self) -> bool:
        return self.value is not None


UNRESOLVED = Growth(None, UNDECIDED)


def resolve(
    current_star: int,
    anchor_star: int | None,
    age_days: float | None,
    window_days: int,
) -> Growth:
    """算一个仓库的窗口增长。

    Args:
        current_star: 现在的 star 数。
        anchor_star:  锚点快照里的 star;`None` 表示那天的快照里没有这个仓库。
        age_days:     仓库创建至今多少天;`None` 表示不知道(DB 里 created_at 为空)。
        window_days:  **锚点的实际跨度**,不是请求的窗口。

    增长可以是负的(有人取消 star),不夹到零:排名自己会把它筛掉,而抹平会让
    「掉了 300 星」和「一点没涨」在数据上再也分不开。
    """
    if anchor_star is not None:
        return Growth(current_star - anchor_star, ANCHOR)
    if age_days is not None and age_days <= window_days:
        return Growth(current_star, NEW_IN_WINDOW)
    return UNRESOLVED


class Tally(NamedTuple):
    """一批仓库算下来,各种来源各有多少 —— 排名漏斗直接用它报数。"""

    growths: dict[str, Growth]

    def count(self, basis: str) -> int:
        return sum(1 for g in self.growths.values() if g.basis == basis)

    @property
    def decided(self) -> dict[str, int]:
        """只保留算得出来的,值是增长数。未决的不在里面 —— 调用方拿不到就不会误用成 0。"""
        return {n: g.value for n, g in self.growths.items() if g.value is not None}

    def summary(self) -> str:
        return (f"锚点相减 {self.count(ANCHOR)},窗口内新建 {self.count(NEW_IN_WINDOW)},"
                f"未决 {self.count(UNDECIDED)}")


def resolve_all(
    stars: dict[str, int],
    anchor_stars: dict[str, int],
    ages: dict[str, float | None],
    window_days: int,
) -> Tally:
    """批量版。`ages` 里查不到的按「不知道创建时间」处理。"""
    return Tally({
        name: resolve(star, anchor_stars.get(name), ages.get(name), window_days)
        for name, star in stars.items()
    })
