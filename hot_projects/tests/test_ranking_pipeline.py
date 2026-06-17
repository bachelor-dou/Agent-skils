from hot_projects.pipeline.ranking_pipeline import run_ranking
from hot_projects.pipeline.cache import RankingCache


class FakeProvider:
    def __init__(self):
        self.calls = []

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
        return {"candidates": {"a/b": {"growth": 900, "star": 1500, "created_at": ""}},
                "growth_calc_days": 7}


def _patch_rank(monkeypatch):
    import hot_projects.pipeline.ranking_pipeline as P
    monkeypatch.setattr(P, "step2_rank_and_select", lambda cand, **kw: list(cand.items()))


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
