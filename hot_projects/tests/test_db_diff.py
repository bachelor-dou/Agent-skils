from datetime import datetime, timezone, timedelta

from hot_projects.infra.db import is_project_window_match
from hot_projects.infra.concurrency.task_help import _resolve_growth


def _ts(days_ago, hours=0):
    dt = datetime.now(timezone.utc) - timedelta(days=days_ago, hours=hours)
    return dt.strftime("%Y-%m-%dT%H:%M:%SZ")


# ── 项目级窗口匹配（5h 容差）──

def test_window_match_within_tolerance():
    assert is_project_window_match(_ts(7), 7, 5) is True        # 正好 7 天
    assert is_project_window_match(_ts(7, 4), 7, 5) is True      # 7天4小时，偏差 < 5h
    assert is_project_window_match(_ts(6, 20), 7, 5) is True     # 6天20小时，偏差 4h


def test_window_match_out_of_tolerance():
    assert is_project_window_match(_ts(7, 6), 7, 5) is False     # 偏差 6h > 5h
    assert is_project_window_match(_ts(5), 7, 5) is False        # 偏差 2 天
    assert is_project_window_match(_ts(10), 7, 5) is False       # 偏差 3 天


def test_window_match_invalid_input():
    assert is_project_window_match("", 7, 5) is False
    assert is_project_window_match("not-a-date", 7, 5) is False


# ── _resolve_growth 的定案/未决分流 ──

def _ctx(prev_snapshot, *, can_write_db=False, use_checkpoint=False, window=7):
    return {
        "candidate_map": {},
        "growth_threshold": 200,
        "can_write_db": can_write_db,
        "window_specified": True,
        "growth_calc_days": window,
        "is_hot_new": False,
        "prev_snapshot": prev_snapshot,
        "use_checkpoint": use_checkpoint,
        "checkpoint_dirty": [False],
    }


def test_diff_used_for_matching_project():
    fn = "a/b"
    refreshed = _ts(7)
    db = {"valid": True, "date": _ts(7)[:10], "projects": {fn: {"star": 1000, "refreshed_at": refreshed}}}
    raw_repos = {fn: {"star": 1300, "repo_item": {}, "created_at": ""}}
    candidate_map = {}
    ctx = _ctx({fn: {"star": 1000, "refreshed_at": refreshed}})
    ctx["candidate_map"] = candidate_map

    _resolve_growth(raw_repos, db, candidate_map, ctx)

    assert fn in candidate_map and candidate_map[fn]["growth"] == 300  # 1300-1000
    assert ctx["unresolved_count"] == 0


def test_scaled_diff_when_snapshot_age_within_band():
    # GitHub 停供 star 时间戳后实时估算已不可用，快照年龄在窗口 [0.4, 2.0] 倍内改为线性折算，
    # 否则这批项目（2026-07-29 一期约 1461 个）会整批出不了榜。
    fn = "a/b"
    refreshed = _ts(3)  # 3 天快照 / 窗口 7 天 → 比值 0.43，落在折算区间
    db = {"valid": True, "date": _ts(7)[:10], "projects": {fn: {"star": 1000, "refreshed_at": refreshed}}}
    raw_repos = {fn: {"star": 1300, "repo_item": {}, "created_at": ""}}
    candidate_map = {}
    ctx = _ctx({fn: {"star": 1000, "refreshed_at": refreshed}})
    ctx["candidate_map"] = candidate_map

    _resolve_growth(raw_repos, db, candidate_map, ctx)

    assert candidate_map[fn]["growth"] == 700  # 300 × 7/3
    assert ctx["unresolved_count"] == 0


def test_scaled_diff_for_stale_snapshot():
    # 漏采一周的项目：快照 14 天、窗口 7 天 → 增量摊薄一半，不会虚增。
    fn = "a/b"
    refreshed = _ts(14)
    db = {"valid": True, "date": _ts(7)[:10], "projects": {fn: {"star": 1000, "refreshed_at": refreshed}}}
    raw_repos = {fn: {"star": 1800, "repo_item": {}, "created_at": ""}}
    candidate_map = {}
    ctx = _ctx({fn: {"star": 1000, "refreshed_at": refreshed}})
    ctx["candidate_map"] = candidate_map

    _resolve_growth(raw_repos, db, candidate_map, ctx)

    assert candidate_map[fn]["growth"] == 400  # 800 × 7/14
    assert ctx["unresolved_count"] == 0


def test_unresolved_when_snapshot_too_fresh_to_scale():
    fn = "a/b"
    refreshed = _ts(0, hours=12)  # 半天快照 / 窗口 7 天 → 折算要放大 14 倍，拒绝
    db = {"valid": True, "date": _ts(7)[:10], "projects": {fn: {"star": 1000, "refreshed_at": refreshed}}}
    raw_repos = {fn: {"star": 1300, "repo_item": {}, "created_at": ""}}
    candidate_map = {}
    ctx = _ctx({fn: {"star": 1000, "refreshed_at": refreshed}})
    ctx["candidate_map"] = candidate_map

    _resolve_growth(raw_repos, db, candidate_map, ctx)

    assert fn not in candidate_map
    assert ctx["unresolved_count"] == 1  # 无实时兜底，记未决


def test_fresh_repo_growth_equals_all_stars():
    # 窗口内新建的仓库：全部 star 都是窗口内涨的，无需任何请求就能精确定增长。
    fn = "a/b"
    db = {"valid": True, "date": _ts(7)[:10], "projects": {}}
    raw_repos = {fn: {"star": 1300, "repo_item": {}, "created_at": _ts(3)}}
    candidate_map = {}
    ctx = _ctx({})  # 冷启动，无快照
    ctx["candidate_map"] = candidate_map

    _resolve_growth(raw_repos, db, candidate_map, ctx)

    assert candidate_map[fn]["growth"] == 1300
    assert ctx["unresolved_count"] == 0


def test_diff_used_when_db_invalid_but_window_matches():
    # 差值有效性改为逐项判定：即使顶层 db["valid"]=False（由静态 DATA_EXPIRE_DAYS 驱动），
    # 只要项目快照 refreshed_at 与窗口相差 ≤5h，仍走差值（修复 D1：长窗口被整库误判过期）。
    fn = "a/b"
    refreshed = _ts(7)
    db = {"valid": False, "date": _ts(7)[:10], "projects": {fn: {"star": 1000, "refreshed_at": refreshed}}}
    raw_repos = {fn: {"star": 1300, "repo_item": {}, "created_at": ""}}
    candidate_map = {}
    ctx = _ctx({fn: {"star": 1000, "refreshed_at": refreshed}})
    ctx["candidate_map"] = candidate_map

    _resolve_growth(raw_repos, db, candidate_map, ctx)

    assert fn in candidate_map and candidate_map[fn]["growth"] == 300  # 1300-1000，逐项窗口匹配
    assert ctx["unresolved_count"] == 0  # 不再因 db.valid=False 退回实时


def test_unresolved_when_not_in_prev_snapshot():
    fn = "a/b"
    db = {"valid": True, "date": _ts(7)[:10], "projects": {}}
    # 无快照且非窗口内新建（创建于 400 天前）→ 本地算不出，且没有实时兜底，只能记未决。
    raw_repos = {fn: {"star": 1300, "repo_item": {}, "created_at": _ts(400)}}
    candidate_map = {}
    ctx = _ctx({})  # 冷启动：项目不在快照
    ctx["candidate_map"] = candidate_map

    _resolve_growth(raw_repos, db, candidate_map, ctx)

    assert fn not in candidate_map
    assert ctx["unresolved_count"] == 1
