from datetime import timedelta

import pytest

from hot_projects.config import MIN_STAR
from hot_projects.infra import snapshots
from hot_projects.tools.tool.ranking import run_ranking, RankingCache


@pytest.fixture(autouse=True)
def _isolate_snapshots(monkeypatch, tmp_path):
    """把快照目录指到 tmp：否则候选池会读到工作区里真实的今日快照，
    测试结果就随本机有没有跑过每日任务而变。默认空目录 = 无今日快照 = 走 API 收集。"""
    monkeypatch.setattr(snapshots, "SNAPSHOT_DIR", str(tmp_path / "snapshots"))


def _write_today_snapshot(stars: dict[str, int]) -> None:
    snapshots.save_snapshot(snapshots.utc_today(), stars)


class FakeProvider:
    def __init__(self):
        self.calls = []
        self.growth_repos: list[dict] = []

    def search_by_keywords(self, **kw):
        self.calls.append("search")
        return {"_raw_repos": [{"full_name": "a/b", "star": 1500, "_raw": {}}]}

    def scan_star_range(self, **kw):
        self.calls.append("scan")
        return {"_raw_repos": []}

    def fetch_trending(self, trending_range):
        self.calls.append("trending")
        return {"_raw_repos": []}

    def batch_growth(self, repos, db, **kw):
        self.calls.append("growth")
        self.growth_repos = list(repos)
        return {"candidates": {"a/b": {"growth": 900, "star": 1500, "created_at": ""}},
                "growth_calc_days": 7}


def _patch_rank(monkeypatch, spy: dict | None = None):
    import hot_projects.tools.tool.ranking as P

    def fake_rank(cand, **kw):
        if spy is not None:
            spy.update(kw)
        return list(cand.items())

    monkeypatch.setattr(P, "step2_rank_and_select", fake_rank)


def test_full_run_executes_all_stages(monkeypatch):
    _patch_rank(monkeypatch)
    p = FakeProvider()
    db = {"valid": False, "projects": {}}
    out = run_ranking(p, mode="comprehensive",
                      params={"min_star": 1200, "max_star": 45000, "growth_calc_days": 7,
                              "growth_threshold": 800, "top_n": 10},
                      db=db, cache=RankingCache(), do_report=False)
    assert out["candidates_count"] == 1
    assert "search" in p.calls and "scan" in p.calls and "trending" in p.calls and "growth" in p.calls


def test_threshold_change_skips_collect_and_growth(monkeypatch):
    _patch_rank(monkeypatch)
    p = FakeProvider()
    cache = RankingCache()
    db = {"valid": False, "projects": {}}
    base = {"min_star": 1200, "max_star": 45000, "growth_calc_days": 7, "top_n": 10}

    run_ranking(p, mode="comprehensive", params={**base, "growth_threshold": 800},
                db=db, cache=cache, do_report=False)
    first = list(p.calls)
    run_ranking(p, mode="comprehensive", params={**base, "growth_threshold": 500},
                db=db, cache=cache, do_report=False)
    # 仅阈值变化：不应再次 search/scan/trending/growth
    assert p.calls == first


def test_threshold_500_includes_more(monkeypatch):
    _patch_rank(monkeypatch)
    p = FakeProvider()
    # growth=900 的候选：阈值 800 入选；阈值 1000 落选
    cache = RankingCache()
    db = {"valid": False, "projects": {}}
    base = {"min_star": 1200, "max_star": 45000, "growth_calc_days": 7, "top_n": 10}
    out_hi = run_ranking(p, mode="comprehensive", params={**base, "growth_threshold": 1000},
                         db=db, cache=cache, do_report=False)
    out_lo = run_ranking(p, mode="comprehensive", params={**base, "growth_threshold": 800},
                         db=db, cache=cache, do_report=False)
    assert out_hi["candidates_count"] == 0
    assert out_lo["candidates_count"] == 1


def test_progress_cb_emits_monotonic_to_100(monkeypatch):
    _patch_rank(monkeypatch)
    p = FakeProvider()
    db = {"valid": False, "projects": {}}
    events = []
    run_ranking(p, mode="comprehensive",
                params={"min_star": 1200, "max_star": 45000, "growth_calc_days": 7,
                        "growth_threshold": 800, "top_n": 10},
                db=db, cache=RankingCache(), do_report=False,
                progress_cb=lambda pct, label: events.append((pct, label)))
    pcts = [pct for pct, _ in events]
    assert pcts, "应至少回传一次进度"
    assert pcts == sorted(pcts), "进度百分比应单调不降"
    assert pcts[-1] == 100, "最终应回传 100%"
    assert all(0 <= pct <= 100 for pct in pcts)


def test_keyword_mode_skips_scan_and_trending(monkeypatch):
    _patch_rank(monkeypatch)
    p = FakeProvider()
    db = {"valid": False, "projects": {}}
    run_ranking(p, mode="keyword",
                params={"min_star": 1200, "growth_calc_days": 7, "growth_threshold": 800, "top_n": 10},
                db=db, cache=RankingCache(), do_report=False)
    assert "search" in p.calls
    assert "scan" not in p.calls and "trending" not in p.calls


def test_comprehensive_reads_today_snapshot_instead_of_scanning(monkeypatch):
    """有今日快照时候选池直接来自快照：不发 search/scan/trending，star 取快照值。"""
    _patch_rank(monkeypatch)
    _write_today_snapshot({"a/b": 1500, "c/d": 3000, "low/one": 400})
    p = FakeProvider()
    db = {"valid": False, "projects": {"a/b": {"created_at": "2020-01-01T00:00:00Z", "star": 900}}}

    run_ranking(p, mode="comprehensive",
                params={"min_star": MIN_STAR, "max_star": 45000, "growth_calc_days": 7,
                        "growth_threshold": 800, "top_n": 10},
                db=db, cache=RankingCache(), do_report=False)

    assert p.calls == ["growth"], f"不该再扫 GitHub，实际调用: {p.calls}"
    pool = {r["full_name"]: r for r in p.growth_repos}
    assert set(pool) == {"a/b", "c/d"}, "低于 min_star 的应被剔除"
    # star 必须是快照的 1500 而不是 DB 里上周的 900：锚点是快照，被减数也得是快照
    assert pool["a/b"]["star"] == 1500
    assert pool["a/b"]["_raw"]["created_at"] == "2020-01-01T00:00:00Z"


def test_snapshot_pool_ignores_max_star(monkeypatch):
    """max_star 是星段扫描的分段上限，读快照时不该拿它截断候选（否则超大仓库整批出不了榜）。"""
    _patch_rank(monkeypatch)
    _write_today_snapshot({"huge/repo": 90000})
    p = FakeProvider()

    run_ranking(p, mode="comprehensive",
                params={"min_star": MIN_STAR, "max_star": 20000, "growth_calc_days": 7,
                        "growth_threshold": 800, "top_n": 10},
                db={"valid": False, "projects": {}}, cache=RankingCache(), do_report=False)

    assert [r["full_name"] for r in p.growth_repos] == ["huge/repo"]


def test_falls_back_to_scan_when_db_universe_cannot_answer(monkeypatch):
    """两种读快照不成立的情形都必须回退 API：min_star 低于 DB 收录下沿；keyword 模式。"""
    _patch_rank(monkeypatch)
    _write_today_snapshot({"a/b": 1500})

    low = FakeProvider()
    run_ranking(low, mode="comprehensive",
                params={"min_star": MIN_STAR - 1, "max_star": 45000, "growth_calc_days": 7,
                        "growth_threshold": 800, "top_n": 10},
                db={"valid": False, "projects": {}}, cache=RankingCache(), do_report=False)
    assert "scan" in low.calls, "DB 没收 MIN_STAR 以下那一档，读库会静默少给候选"

    kw = FakeProvider()
    run_ranking(kw, mode="keyword",
                params={"min_star": MIN_STAR, "growth_calc_days": 7,
                        "growth_threshold": 800, "top_n": 10},
                db={"valid": False, "projects": {}}, cache=RankingCache(), do_report=False)
    assert "search" in kw.calls, "关键词/类别筛选没法从 DB 读（新仓库连 topics 都没存）"


def test_delayed_probe_anchor_window_reaches_scoring(monkeypatch):
    """探针锚点顺延时，实际天数必须一路传到打分，否则会凭空造出"爆发"。

    造一个只有 T−5 快照的场景（T−3/T−4 漏采），请求窗口仍是 RECENT_GROWTH_DAYS=3：
      正确（按实际 5 天）：recent_rate = 500/5 = 100 < 主窗口 900/7 ≈ 128 → 不加成
      错误（按请求 3 天）：recent_rate = 500/3 ≈ 167 > 128 → 误判为爆发
    所以 boost_applied 能区分对错；同时直接检查 scoring 收到的 recent_growth_days。
    """
    from hot_projects.config import RECENT_GROWTH_DAYS
    spy: dict = {}
    _patch_rank(monkeypatch, spy)
    _write_today_snapshot({"a/b": 1500})
    snapshots.save_snapshot(snapshots.utc_today() - timedelta(days=5), {"a/b": 1000})

    out = run_ranking(FakeProvider(), mode="comprehensive",
                      params={"min_star": MIN_STAR, "max_star": 45000, "growth_calc_days": 7,
                              "growth_threshold": 800, "top_n": 10},
                      db={"valid": False, "projects": {}}, cache=RankingCache(), do_report=False)

    assert spy["recent_growth_days"] == 5, (
        f"打分必须拿到锚点的实际天数 5，而不是请求的 {RECENT_GROWTH_DAYS}"
    )
    assert out["funnel"]["recent_probe"] == 1, "探针本身应该算出了这个候选"
    assert out["funnel"]["boost_applied"] == 0, (
        "按实际 5 天算，近期速率低于整窗均速，不该判定为爆发"
    )


def test_counts_distinguish_growth_pool_from_thresholded_candidates(monkeypatch, caplog):
    _patch_rank(monkeypatch)
    p = FakeProvider()
    db = {"valid": False, "projects": {}}

    caplog.set_level("WARNING", logger="hot_projects")
    out = run_ranking(
        p, mode="comprehensive",
        params={"min_star": 1200, "max_star": 45000, "growth_calc_days": 7,
                "growth_threshold": 1000, "top_n": 2},
        db=db, cache=RankingCache(), do_report=False,
    )

    assert out["growth_candidates_count"] == 1
    assert out["candidates_count"] == 0
    assert out["returned_count"] == 0
    assert "达标候选不足 requested_top_n=2 returned=0 candidates=0 growth_pool=1" in caplog.text
