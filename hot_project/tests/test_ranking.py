"""打分公式与榜单流水线。

流水线整条不发请求,所以这里连 mock 网络都不需要 —— 只把「今天的快照」和「DB」
换成内存里的字典。这正是把候选池从三阶段扫描改成读快照买到的东西:
上一版要测一轮排名得先有 12 个 token。
"""

from datetime import timedelta

import pytest

from hot_project import config
from hot_project.common.timeutil import utc_today
from hot_project.core import scoring
from hot_project.infra.store import snapshots
from hot_project.tools import ranking

W = scoring.Weights(window_days=7, recent_days=3, alpha=0.5, cap=2.0)


# ── 打分公式 ───────────────────────────────────────────────────────

def test_more_growth_scores_higher():
    small = scoring.score({"growth": 100, "star": 10000}, scoring.COMPREHENSIVE, W)
    big = scoring.score({"growth": 5000, "star": 10000}, scoring.COMPREHENSIVE, W)
    assert big > small


def test_a_small_project_growing_fast_can_beat_a_giant_growing_slowly():
    """增长率那一项存在的全部理由。没有它,榜单永远是那几个巨无霸。"""
    nimble = scoring.score({"growth": 900, "star": 1200}, scoring.COMPREHENSIVE, W)
    giant = scoring.score({"growth": 1000, "star": 200000}, scoring.COMPREHENSIVE, W)
    assert nimble > giant


def test_an_absurd_growth_rate_gets_discounted():
    """100 星涨到 300 的「三倍」和 10000 星涨 2 万不是一回事。"""
    plain = {"growth": 400, "star": 800}          # 率 0.5,正好在拐点上,不打折
    absurd = {"growth": 1200, "star": 800}        # 率 1.5,打满 15%
    ratio = (scoring.score(absurd, scoring.COMPREHENSIVE, W)
             / scoring.score(plain, scoring.COMPREHENSIVE, W))
    # 涨三倍的量只换来 1.21 倍的分。这个数要卡死:松成「小于 1.6」的话,
    # 折扣整个删掉(比值 1.43)也照样通过。
    assert ratio == pytest.approx(1.43 * 0.85, rel=0.02)


def test_hot_new_ranks_purely_by_growth():
    """新项目榜比的就是谁涨得多。叠增长率会让三天涨 200 压过三十天涨 3000。"""
    assert scoring.score({"growth": 3000, "star": 50000}, scoring.HOT_NEW, W) == 3000.0
    assert scoring.score({"growth": 200, "star": 210}, scoring.HOT_NEW, W) == 200.0


# ── 爆发加成 ───────────────────────────────────────────────────────

def test_speeding_up_earns_a_boost():
    #  7 天涨 700(每天 100),最近 3 天涨 600(每天 200)→ 加速比 2
    assert scoring.burst_boost(700, 600, W) == pytest.approx(1.5)


def test_slowing_down_is_not_punished():
    """「前期猛、最近平」本来就该靠基础分排;再罚一次是对同一件事收两遍税。"""
    assert scoring.burst_boost(700, 30, W) == 1.0


def test_one_freak_number_cannot_flip_the_board():
    huge = scoring.burst_boost(7, 100000, W)
    assert huge == pytest.approx(1.0 + W.alpha * W.cap)


def test_no_probe_data_means_no_boost_not_a_penalty():
    """缺快照的日子探针整个失效。全场一起挨罚的话顺序其实没变,
    只是分数集体缩水,看日志的人会以为打分坏了。"""
    assert scoring.burst_boost(700, None, W) == 1.0


def test_the_boost_uses_the_real_anchor_span_on_both_sides():
    """主窗口修正了、探针没修正的话,5 天的增量除以请求的 3 天,凭空造出一场爆发。"""
    honest = scoring.burst_boost(700, 500, W._replace(recent_days=5))
    inflated = scoring.burst_boost(700, 500, W._replace(recent_days=3))
    assert inflated > honest


def test_ranking_writes_the_score_back_for_the_logs():
    ordered = scoring.rank({"a/x": {"growth": 10, "star": 100}}, scoring.COMPREHENSIVE, W)
    assert ordered[0][1]["_score"] > 0


# ── 流水线 ────────────────────────────────────────────────────────

@pytest.fixture
def world(monkeypatch):
    """把「今天的快照」「锚点」「DB」换成内存字典。"""
    state = {"today": {}, "anchors": {}, "db": {}}

    def load_stars(day):
        return state["today"] if day == utc_today() else {}

    def anchor_for_window(days):
        entry = state["anchors"].get(days)
        if entry is None:
            return None
        stars, actual = entry
        return snapshots.Anchor(day=utc_today() - timedelta(days=actual),
                                stars=stars, window_days=actual)

    monkeypatch.setattr(snapshots, "load_stars", load_stars)
    monkeypatch.setattr(snapshots, "anchor_for_window", anchor_for_window)
    monkeypatch.setattr(ranking.universe, "load", lambda: state["db"])
    return state


def _created(days_ago: int) -> str:
    return (utc_today() - timedelta(days=days_ago)).isoformat()


def test_the_candidate_pool_is_todays_snapshot_filtered_by_star(world):
    world["today"] = {"a/big": 5000, "b/small": 100}
    pool = ranking.candidates(min_star=500)
    assert set(pool) == {"a/big"}


def test_no_snapshot_today_means_no_ranking_rather_than_a_wrong_one(world):
    assert ranking.candidates(min_star=500) == {}


def test_a_repo_with_no_anchor_and_no_creation_date_drops_out_of_the_pool(world):
    """未决不是零增长:记 0 它会以「一点没涨」的身份进排名,永远出不了榜,
    而真实原因(缺基线)没人看得见。"""
    world["today"] = {"a/known": 1000, "b/mystery": 9000}
    world["anchors"][7] = ({"a/known": 400}, 7)
    decided, window, _ = ranking.growths(
        {"a/known": {"star": 1000, "created_at": _created(900)},
         "b/mystery": {"star": 9000, "created_at": ""}}, 7)
    assert set(decided) == {"a/known"}
    assert decided["a/known"]["growth"] == 600
    assert window == 7


def test_a_deferred_anchor_widens_the_window_everywhere(world):
    """请求 7 天但只有 T−9 的快照:窗口就是 9 天,一个 8 天大的新仓库该被算出来。"""
    world["anchors"][7] = ({}, 9)
    decided, window, _ = ranking.growths(
        {"a/new": {"star": 600, "created_at": _created(8)}}, 7)
    assert window == 9
    assert decided["a/new"]["growth"] == 600


def test_without_an_anchor_nothing_is_decided(world):
    decided, _, basis = ranking.growths({"a/x": {"star": 1, "created_at": ""}}, 7)
    assert decided == {} and basis == "无锚点"


def test_the_burst_probe_reports_the_real_span(world):
    world["anchors"][config.RECENT_GROWTH_DAYS] = ({"a/x": 100}, 5)
    probed, days = ranking.recent({"a/x": {"star": 400}}, config.RECENT_GROWTH_DAYS)
    assert probed == {"a/x": 300} and days == 5


def test_a_missing_probe_anchor_never_fails_the_ranking(world):
    probed, days = ranking.recent({"a/x": {"star": 400}}, 3)
    assert probed == {} and days == 3


def test_a_repo_that_lost_stars_is_skipped_by_the_probe(world):
    """探针只做加成。负的最近增长喂进加速比是没有意义的输入。"""
    world["anchors"][3] = ({"a/x": 500}, 3)
    probed, _ = ranking.recent({"a/x": {"star": 400}}, 3)
    assert probed == {}


def test_a_full_run_ranks_and_reports_the_funnel(world):
    world["today"] = {"a/hot": 5000, "b/flat": 3000, "c/tiny": 100, "d/mystery": 8000}
    world["anchors"][7] = ({"a/hot": 3000, "b/flat": 2990, "c/tiny": 50}, 7)
    world["anchors"][config.RECENT_GROWTH_DAYS] = ({"a/hot": 3500}, 3)
    world["db"] = {n: {"created_at": _created(500)} for n in world["today"]}

    out = ranking.run(min_star=500, growth_threshold=1000, growth_days=7,
                      do_report=False)

    assert [n for n, _ in out["ranked"]] == ["a/hot"]       # b 只涨 10,c 星太低
    assert out["funnel"] == {"收集": 3, "增长可算": 2, "达标": 1,
                             "爆发加成": 1, "出榜": 1}
    assert out["growth_days"] == 7


def test_hot_new_keeps_only_recently_created_repos(world):
    world["today"] = {"a/old": 5000, "b/fresh": 4000}
    world["anchors"][7] = ({}, 7)       # 都不在锚点里 → 只能靠创建时间判定
    world["db"] = {"a/old": {"created_at": _created(900)},
                   "b/fresh": {"created_at": _created(3)}}

    out = ranking.run(mode="hot_new", min_star=500, growth_threshold=1000,
                      growth_days=7, created_days=30, do_report=False)
    assert [n for n, _ in out["ranked"]] == ["b/fresh"]
    assert out["ranked"][0][1]["growth"] == 4000        # 有尾无头:全部 star 都是增长
