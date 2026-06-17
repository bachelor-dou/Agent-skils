from datetime import datetime, timezone, timedelta

from hot_projects.infra.db import is_project_window_match
from hot_projects.infra.concurrency.task_help import _submit_growth_tasks


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


# ── _submit_growth_tasks 的差值/实时分流 ──

class _Pool:
    def __init__(self):
        self.submitted = []

    def submit(self, task):
        self.submitted.append(task)


def _ctx(prev_snapshot, *, can_write_db=False, use_checkpoint=False, window=7):
    return {
        "pending_created_at": {},
        "candidate_map": {},
        "growth_threshold": 200,
        "use_realtime_growth": False,
        "can_write_db": can_write_db,
        "window_specified": True,
        "growth_calc_days": window,
        "is_hot_new": False,
        "prev_snapshot": prev_snapshot,
        "use_checkpoint": use_checkpoint,
        "unresolved_count": [0],
        "checkpoint_dirty": [False],
        "completed_since_save": [0],
    }


def test_diff_used_for_matching_project():
    fn = "a/b"
    refreshed = _ts(7)
    db = {"valid": True, "date": _ts(7)[:10], "projects": {fn: {"star": 1000, "refreshed_at": refreshed}}}
    raw_repos = {fn: {"star": 1300, "repo_item": {}, "created_at": ""}}
    candidate_map = {}
    ctx = _ctx({fn: {"star": 1000, "refreshed_at": refreshed}})
    ctx["candidate_map"] = candidate_map
    pool = _Pool()

    _submit_growth_tasks(pool, None, raw_repos, db, candidate_map, ctx)

    assert fn in candidate_map and candidate_map[fn]["growth"] == 300  # 1300-1000
    assert pool.submitted == []  # 没有提交实时任务


def test_realtime_when_window_mismatch():
    fn = "a/b"
    refreshed = _ts(3)  # 才 3 天，窗口 7 → 不匹配
    db = {"valid": True, "date": _ts(7)[:10], "projects": {fn: {"star": 1000, "refreshed_at": refreshed}}}
    raw_repos = {fn: {"star": 1300, "repo_item": {}, "created_at": ""}}
    candidate_map = {}
    ctx = _ctx({fn: {"star": 1000, "refreshed_at": refreshed}})
    ctx["candidate_map"] = candidate_map
    pool = _Pool()

    _submit_growth_tasks(pool, None, raw_repos, db, candidate_map, ctx)

    assert fn not in candidate_map
    assert len(pool.submitted) == 1  # 回退实时任务


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
    pool = _Pool()

    _submit_growth_tasks(pool, None, raw_repos, db, candidate_map, ctx)

    assert fn in candidate_map and candidate_map[fn]["growth"] == 300  # 1300-1000，逐项窗口匹配
    assert pool.submitted == []  # 不再因 db.valid=False 退回实时


def test_realtime_when_not_in_prev_snapshot():
    fn = "a/b"
    db = {"valid": True, "date": _ts(7)[:10], "projects": {}}
    raw_repos = {fn: {"star": 1300, "repo_item": {}, "created_at": ""}}
    candidate_map = {}
    ctx = _ctx({})  # 冷启动：项目不在快照
    ctx["candidate_map"] = candidate_map
    pool = _Pool()

    _submit_growth_tasks(pool, None, raw_repos, db, candidate_map, ctx)

    assert fn not in candidate_map
    assert len(pool.submitted) == 1
