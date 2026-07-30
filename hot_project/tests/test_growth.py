"""增长口径的守卫。

纯算术,没有 I/O,所以这些测试跑起来是毫秒级的 —— 这正是把它放进 `core/` 换来的东西。
"""

from hot_project.core import growth


def test_anchor_subtraction_is_just_subtraction():
    assert growth.resolve(1500, anchor_star=1200, age_days=400, window_days=7) == (300, growth.ANCHOR)


def test_repo_created_inside_the_window_counts_all_its_stars():
    """有尾无头:锚点那天它还不存在,所以现在的 star 全是这个窗口涨的。"""
    got = growth.resolve(800, anchor_star=None, age_days=3, window_days=7)
    assert got == (800, growth.NEW_IN_WINDOW)


def test_old_repo_missing_from_the_anchor_is_undecided_not_zero():
    """未决 ≠ 涨了 0。

    当成 0 的话它会以「一点没涨」的身份进排名、永远出不了榜,而真正的原因(缺基线)
    没人看得见。所以这里必须是 None,让调用方拿不到值、想误用都用不了。
    """
    got = growth.resolve(5000, anchor_star=None, age_days=900, window_days=7)
    assert got.value is None
    assert got.basis == growth.UNDECIDED
    assert not got.decided


def test_unknown_creation_date_is_undecided():
    """DB 里 created_at 为空 → 判不了规则 2,老实认未决。"""
    assert growth.resolve(5000, None, age_days=None, window_days=7).value is None


def test_a_deferred_anchor_widens_the_window_for_new_repos_too():
    """旧代码在这里丢候选:请求 7 天,当天缺快照顺延到 T−9,一个 8 天大的仓库两头落空。

    锚点里没有(9 天前它还不存在),`8 <= 7` 又为假 → 被记成未决剔出排名。可它本该精确
    算出来:窗口实际 9 天,仓库才 8 天大,全部 star 都是这 9 天涨的。
    而缺快照顺延的日子,恰恰正是新仓库最多的日子。

    新实现里窗口只有一个来源 —— 锚点的实际跨度,所以传进来的就是 9。
    """
    assert growth.resolve(600, None, age_days=8, window_days=9).basis == growth.NEW_IN_WINDOW
    assert growth.resolve(600, None, age_days=8, window_days=7).basis == growth.UNDECIDED


def test_a_repo_exactly_as_old_as_the_window_still_counts():
    """边界含等号:窗口开始那天创建的仓库,锚点快照里同样没有它。"""
    assert growth.resolve(100, None, age_days=7, window_days=7).basis == growth.NEW_IN_WINDOW


def test_measured_snapshot_beats_declared_creation_date():
    """两条规则本不该同时成立;真同时成立说明元数据不对,那就信快照。"""
    got = growth.resolve(1000, anchor_star=900, age_days=2, window_days=7)
    assert got == (100, growth.ANCHOR)


def test_losing_stars_stays_negative():
    """不夹到零:抹平会让「掉了 300 星」和「一点没涨」在数据上再也分不开。"""
    assert growth.resolve(700, anchor_star=1000, age_days=400, window_days=7).value == -300


def test_batch_reports_where_each_number_came_from():
    tally = growth.resolve_all(
        stars={"a/one": 1500, "b/two": 800, "c/three": 5000},
        anchor_stars={"a/one": 1200},
        ages={"a/one": 400, "b/two": 3, "c/three": 900},
        window_days=7,
    )
    assert tally.decided == {"a/one": 300, "b/two": 800}     # 未决的不在里面
    assert tally.count(growth.ANCHOR) == 1
    assert tally.count(growth.NEW_IN_WINDOW) == 1
    assert tally.count(growth.UNDECIDED) == 1


def test_batch_tolerates_missing_ages():
    tally = growth.resolve_all({"a/one": 10}, {}, {}, window_days=7)
    assert tally.count(growth.UNDECIDED) == 1
