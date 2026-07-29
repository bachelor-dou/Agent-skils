"""每日 star 快照的自检：锚点选取、按日期清理、全 null 批次护栏、锚点优先级。

这三处坏掉都是静默的（选错锚点 → 窗口悄悄变长；清理越界 → 锚点凭空消失；
全 null 被当成真删除 → 整批基线归零），所以各留一条会失败的断言。
"""

import asyncio
from datetime import date, datetime, timedelta, timezone

import pytest

from hot_projects.datasource.github import star_snapshot
from hot_projects.infra import snapshots
from hot_projects.infra.concurrency.task_help import _resolve_growth_without_timestamps


@pytest.fixture
def snap_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(snapshots, "SNAPSHOT_DIR", str(tmp_path))
    return tmp_path


def _ts(days_ago):
    return (datetime.now(timezone.utc) - timedelta(days=days_ago)).strftime("%Y-%m-%dT%H:%M:%SZ")


# ── 存储与锚点选取 ──

def test_roundtrip(snap_dir):
    day = date(2026, 7, 22)
    snapshots.save_snapshot(day, {"a/b": 100, "c/d": 200})
    assert snapshots.load_snapshot(day) == {"a/b": 100, "c/d": 200}
    assert snapshots.available_dates() == [day]


def test_refuse_empty_snapshot(snap_dir):
    # 空快照会被当成「全仓库掉到 0」，宁可缺一天锚点也不能落盘。
    with pytest.raises(ValueError):
        snapshots.save_snapshot(date(2026, 7, 22), {})


def test_anchor_picks_nearest_within_tolerance(snap_dir):
    for day, star in ((date(2026, 7, 20), 10), (date(2026, 7, 23), 30)):
        snapshots.save_snapshot(day, {"a/b": star})
    # target 7-22：7-23 差 1 天、7-20 差 2 天 → 取 7-23
    found = snapshots.find_anchor(date(2026, 7, 22), tolerance_days=2)
    assert found is not None and found[0] == date(2026, 7, 23)


def test_anchor_tie_prefers_earlier(snap_dir):
    for day in (date(2026, 7, 21), date(2026, 7, 23)):
        snapshots.save_snapshot(day, {"a/b": 1})
    # 两边都差 1 天 → 取较早那天，结果不随文件枚举顺序漂移
    found = snapshots.find_anchor(date(2026, 7, 22), tolerance_days=2)
    assert found is not None and found[0] == date(2026, 7, 21)


def test_anchor_none_beyond_tolerance(snap_dir):
    snapshots.save_snapshot(date(2026, 7, 10), {"a/b": 1})
    assert snapshots.find_anchor(date(2026, 7, 22), tolerance_days=2) is None


def test_prune_keeps_cutoff_day(snap_dir):
    today = date(2026, 7, 29)
    for delta in (0, 35, 36):
        snapshots.save_snapshot(today - timedelta(days=delta), {"a/b": 1})
    removed = snapshots.prune_snapshots(keep_days=35, today=today)
    # 边界当天（正好 35 天）必须留下，只删更早的
    assert removed == [today - timedelta(days=36)]
    assert snapshots.available_dates() == [today - timedelta(days=35), today]


# ── 全 null 批次护栏 ──

class _FakePool:
    def __init__(self):
        self.released = 0

    def token_count(self):
        return 1

    async def acquire(self):
        return 0

    async def release(self, idx):
        self.released += 1

    def get_graphql_headers(self, idx):
        return {}


class _FakeResp:
    status_code = 200

    def __init__(self, payload):
        self._payload = payload
        self.headers = {}

    def json(self):
        return self._payload


class _FakeClient:
    """别名数 > max_ok 时模仿 GitHub 的退化响应：HTTP 200、无 errors、全字段 null。"""

    def __init__(self, stars: dict[str, int], max_ok: int):
        self.stars = stars
        self.max_ok = max_ok
        self.calls = []

    async def post(self, url, headers=None, json=None):
        query = json["query"]
        names = [line.split('owner:"')[1].split('"')[0] + "/" +
                 line.split('name:"')[1].split('"')[0]
                 for line in query.splitlines() if "repository(" in line]
        self.calls.append(len(names))
        if len(names) > self.max_ok:
            return _FakeResp({"data": {f"r{i}": None for i in range(len(names))}})
        return _FakeResp({"data": {
            f"r{i}": ({"stargazerCount": self.stars[n]} if n in self.stars else None)
            for i, n in enumerate(names)
        }})

    async def aclose(self):
        return None


def _collect(client, names, batch_size):
    pool = _FakePool()
    return asyncio.run(star_snapshot.collect_star_snapshot(
        pool, names, batch_size=batch_size, concurrency=2,
    )), client.calls


def test_all_null_batch_splits_instead_of_dropping(monkeypatch):
    names = [f"o{i}/r{i}" for i in range(8)]
    stars = {n: 100 + i for i, n in enumerate(names)}
    client = _FakeClient(stars, max_ok=2)
    monkeypatch.setattr(star_snapshot, "_build_async_client", lambda **kw: client)

    (got, failed), calls = _collect(client, names, batch_size=8)
    # 8 → 全 null → 拆到 4 → 仍全 null → 拆到 2 → 成功。一个仓库都不能丢。
    assert got == stars, "全 null 批次被误当成「仓库都没了」，基线会整批归零"
    assert failed == []
    assert max(calls) == 8 and min(calls) == 2


def test_single_null_treated_as_genuinely_missing(monkeypatch):
    names = ["o0/r0", "gone/repo"]
    client = _FakeClient({"o0/r0": 50}, max_ok=99)
    monkeypatch.setattr(star_snapshot, "_build_async_client", lambda **kw: client)

    (got, failed), _ = _collect(client, names, batch_size=2)
    # 零星 null 是真的删除/改名，不该触发拆分，也不算失败
    assert got == {"o0/r0": 50}
    assert failed == []


# ── 锚点在增长计算里的优先级 ──

def test_anchor_beats_db_paths():
    resolved = _resolve_growth_without_timestamps(
        "a/b", current_star=1500, created_at=_ts(400),
        prev={"star": 1000, "refreshed_at": _ts(7)}, time_window=7,
        anchor_stars={"a/b": 1200},
    )
    # 锚点更精确（全仓库同窗口），必须优先于 DB 差值的 1500−1000
    assert resolved == (300, "快照")


def test_falls_back_when_repo_absent_from_anchor():
    resolved = _resolve_growth_without_timestamps(
        "a/b", current_star=1500, created_at=_ts(400),
        prev={"star": 1000, "refreshed_at": _ts(7)}, time_window=7,
        anchor_stars={"other/repo": 1},
    )
    assert resolved == (500, "DB")


def test_new_repo_absent_from_anchor_uses_full_stars():
    # 3 天前建的仓库不可能出现在 7 天前的快照里 → 落到「窗口内新建」
    resolved = _resolve_growth_without_timestamps(
        "new/repo", current_star=800, created_at=_ts(3),
        prev=None, time_window=7, anchor_stars={"other/repo": 1},
    )
    assert resolved == (800, "窗口内新建")


# ── 漏采几天：锚点顺延后统计口径必须跟着改 ──

class _Pool:
    def __init__(self):
        self.submitted = []

    def submit(self, task):
        self.submitted.append(task)


def _growth_ctx(window=7):
    return {
        "pending_created_at": {},
        "candidate_map": {},
        "growth_threshold": 200,
        "use_realtime_growth": False,
        "can_write_db": False,
        "window_specified": True,
        "growth_calc_days": window,
        "is_hot_new": False,
        "prev_snapshot": {},
        "use_checkpoint": False,
        "unresolved_count": [0],
        "checkpoint_dirty": [False],
        "completed_since_save": [0],
    }


def test_delayed_anchor_rewrites_effective_window(snap_dir, monkeypatch):
    """漏采两天后锚点顺延到 T−9：增长是 9 天的，统计口径必须变成 9 天。

    不改的话打分按 7 天算速率，而增长实际跨了 9 天，速率虚高 29%。
    """
    from hot_projects.infra.concurrency import task_help

    monkeypatch.setattr(task_help, "SNAPSHOT_ANCHOR_TOLERANCE_DAYS", 2)
    today = snapshots.utc_today()
    snapshots.save_snapshot(today - timedelta(days=9), {"a/b": 1000})

    fn = "a/b"
    raw_repos = {fn: {"star": 1900, "repo_item": {}, "created_at": _ts(400)}}
    candidate_map = {}
    ctx = _growth_ctx(window=7)
    ctx["candidate_map"] = candidate_map

    task_help._submit_growth_tasks(
        _Pool(), None, raw_repos, {"projects": {}}, candidate_map, ctx,
    )

    # prev_snapshot 为空，900 只可能来自锚点差值 1900−1000
    assert candidate_map[fn]["growth"] == 900
    assert ctx["effective_growth_calc_days"] == 9, "锚点顺延后仍按 7 天算速率，打分会虚高"


def test_exact_anchor_keeps_requested_window(snap_dir, monkeypatch):
    from hot_projects.infra.concurrency import task_help

    monkeypatch.setattr(task_help, "SNAPSHOT_ANCHOR_TOLERANCE_DAYS", 2)
    snapshots.save_snapshot(snapshots.utc_today() - timedelta(days=7), {"a/b": 1000})

    ctx = _growth_ctx(window=7)
    candidate_map = {}
    ctx["candidate_map"] = candidate_map
    task_help._submit_growth_tasks(
        _Pool(), None, {"a/b": {"star": 1900, "repo_item": {}, "created_at": _ts(400)}},
        {"projects": {}}, candidate_map, ctx,
    )
    assert ctx["effective_growth_calc_days"] == 7
