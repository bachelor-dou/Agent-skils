"""增长口径的守卫。

纯算术,没有 I/O,所以这些测试跑起来是毫秒级的 —— 这正是把它放进 `core/` 换来的东西。
"""

from hot_project.core import growth


def test_baseline_subtraction_is_just_subtraction():
    got = growth.resolve(1500, anchor_star=1200, anchor_days=7, age_days=400, window_days=7)
    assert got == (300, growth.ANCHOR, 7)


def test_each_repo_reports_the_days_its_own_baseline_covers():
    """晚进库的仓库基线只有 3 天。按全局 7 天折算速率,它的日均会虚高一倍多,
    爆发加成于是凭空多给一档 —— 所以天数必须跟着每个仓库自己走。"""
    got = growth.resolve(1500, anchor_star=1200, anchor_days=3, age_days=400, window_days=7)
    assert got == (300, growth.ANCHOR, 3)


def test_repo_created_inside_the_window_counts_all_its_stars():
    """有尾无头:窗口开始时它还不存在,所以现在的 star 全是这个窗口涨的。"""
    got = growth.resolve(800, anchor_star=None, anchor_days=None, age_days=3, window_days=7)
    assert got == (800, growth.NEW_IN_WINDOW, 3)


def test_a_brand_new_repo_never_divides_by_zero_days():
    """今天刚建的仓库年龄不足一天。天数取 0 会让后面的速率计算除零。"""
    got = growth.resolve(50, None, None, age_days=0.4, window_days=7)
    assert got.window_days == 1


def test_a_repo_older_than_the_window_is_not_new_in_it():
    """窗口 5 天、仓库 6.8 天大:它在窗口开始时就存在了,没基线就是算不出来。"""
    assert growth.resolve(600, None, None, age_days=6.8, window_days=5) is None


def test_an_old_repo_without_a_baseline_is_left_out_not_zeroed():
    """算不出来 ≠ 涨了 0。

    当成 0 的话它会以「一点没涨」的身份进排名、永远出不了榜,而真正的原因(缺基线)
    没人看得见。返回 None 是为了让调用方想误用都用不了。
    """
    assert growth.resolve(5000, None, None, age_days=900, window_days=7) is None


def test_unknown_creation_date_is_left_out_too():
    """DB 里 created_at 为空 → 判不了规则 2,老实认算不出来。"""
    assert growth.resolve(5000, None, None, age_days=None, window_days=7) is None


def test_a_repo_exactly_as_old_as_the_window_still_counts():
    """边界含等号:窗口开始那天创建的仓库,最早那份快照里同样没有它。"""
    got = growth.resolve(100, None, None, age_days=7, window_days=7)
    assert got.basis == growth.NEW_IN_WINDOW


def test_measured_snapshot_beats_declared_creation_date():
    """两条规则本不该同时成立;真同时成立说明元数据不对,那就信快照。"""
    got = growth.resolve(1000, anchor_star=900, anchor_days=7, age_days=2, window_days=7)
    assert got == (100, growth.ANCHOR, 7)


def test_losing_stars_stays_negative():
    """不夹到零:抹平会让「掉了 300 星」和「一点没涨」在数据上再也分不开。"""
    got = growth.resolve(700, anchor_star=1000, anchor_days=7, age_days=400, window_days=7)
    assert got.value == -300
