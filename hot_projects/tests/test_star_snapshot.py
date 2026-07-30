"""每日 star 快照的自检：锚点选取、按日期清理、全 null 批次护栏、锚点优先级。

这三处坏掉都是静默的（选错锚点 → 窗口悄悄变长；清理越界 → 锚点凭空消失；
全 null 被当成真删除 → 整批基线归零），所以各留一条会失败的断言。
"""

import asyncio
import json
from datetime import date, datetime, timedelta, timezone

import pytest

# 采集逻辑合并在定时任务脚本里（只有它用），读取侧才在 infra/snapshots.py
from hot_projects import cron_daily_star_snapshot as star_snapshot
from hot_projects.infra import db, snapshots
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


# ── 每日发现写 DB：仅插入语义。破了会静默污染每周报告的 DB 差分兜底 ──

@pytest.fixture
def db_file(tmp_path, monkeypatch):
    """把 DB 路径指到临时文件，并预置一个「已有仓库」。"""
    from hot_projects.infra import db as db_mod
    path = tmp_path / "Github_DB.json"
    path.write_text(json.dumps({
        "date": "2026-07-01",
        "projects": {
            "old/repo": {"star": 5000, "created_at": "2020-01-01T00:00:00Z",
                         "refreshed_at": "2026-07-22T00:00:00Z", "gh_desc": "原有描述"},
        },
    }, ensure_ascii=False), encoding="utf-8")
    monkeypatch.setattr(db_mod, "DB_FILE_PATH", str(path))
    return path


def _read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def test_insert_only_adds_missing(db_file):
    from hot_projects.infra.db import insert_new_projects
    n = insert_new_projects({
        "new/repo": {"star": 600, "created_at": "2026-07-01T00:00:00Z"},
        "old/repo": {"star": 9999, "created_at": "1999-01-01T00:00:00Z"},
    })
    assert n == 1
    assert _read(db_file)["projects"]["new/repo"]["star"] == 600


def test_insert_never_touches_existing(db_file):
    """已有仓库的 star/refreshed_at/gh_desc 一个字都不能动。

    每周报告用项目级 refreshed_at 判断 DB 差值是否匹配本次窗口。每日任务若顺手刷新它，
    这条兜底路径就变成「看起来很新、其实没重新测过」——错得静默且无从发现。
    """
    from hot_projects.infra.db import insert_new_projects
    insert_new_projects({"old/repo": {"star": 9999, "created_at": "1999-01-01T00:00:00Z"}})
    kept = _read(db_file)["projects"]["old/repo"]
    assert kept == {"star": 5000, "created_at": "2020-01-01T00:00:00Z",
                    "refreshed_at": "2026-07-22T00:00:00Z", "gh_desc": "原有描述"}


def test_insert_does_not_bump_db_date(db_file):
    """顶层 date 不能被改：get_db_age_days 靠它推断窗口，每天改一次会让它恒为 0。"""
    from hot_projects.infra.db import insert_new_projects
    insert_new_projects({"new/repo": {"star": 600, "created_at": ""}})
    assert _read(db_file)["date"] == "2026-07-01"


def test_seeding_without_forks_keeps_stored_value():
    """repo_item 里没有 forks_count 时不能把已存的 forks 抹成 0。

    周报的候选池取自每日快照，只带 star + created_at；seeding 走的还是 update_db_project，
    如果照写 0 就会把全库 forks 清零（周报 diff 会报成「五万个仓库 forks 下降」）。
    """
    from hot_projects.infra.db import update_db_project
    projects = {"old/repo": {"star": 5000, "forks": 321, "gh_desc": "原有描述"}}
    update_db_project(projects, "old/repo", 5200, {"created_at": "2020-01-01T00:00:00Z"})
    assert projects["old/repo"]["forks"] == 321
    assert projects["old/repo"]["star"] == 5200, "star 仍必须刷新"

    # 真带 forks_count 时照旧覆写（含降到 0 这种真实下降）
    update_db_project(projects, "old/repo", 5200, {"forks_count": 400})
    assert projects["old/repo"]["forks"] == 400


# ── 爆发探针改成快照减法 ──

def test_burst_probe_is_snapshot_subtraction(snap_dir):
    from hot_projects.tools.tool import ranking
    day = snapshots.utc_today() - timedelta(days=3)
    snapshots.save_snapshot(day, {"a/fast": 1000, "b/slow": 2000})
    got, window = ranking._calc_recent_growth(
        {"a/fast": {"star": 4000}, "b/slow": {"star": 2050}}, recent_days=3,
    )
    assert got == {"a/fast": 3000, "b/slow": 50}
    assert window == 3, "锚点正好是 T−3 时实际窗口就等于请求窗口"


def test_burst_probe_skips_repos_absent_from_anchor(snap_dir):
    """锚点那天还没收进来的新仓库要跳过，不能当成 base=0。

    当成 0 的话增长会等于它的总 star，凭空拿到一个巨大的爆发加成。
    """
    from hot_projects.tools.tool import ranking
    snapshots.save_snapshot(snapshots.utc_today() - timedelta(days=3), {"old/repo": 500})
    got, _ = ranking._calc_recent_growth(
        {"brand/new": {"star": 8000}, "old/repo": {"star": 700}}, recent_days=3,
    )
    assert got == {"old/repo": 200}


def test_burst_probe_returns_empty_without_snapshot(snap_dir):
    """没有 T−3 附近的快照时只是不加成，绝不能让出榜失败。"""
    from hot_projects.tools.tool import ranking
    assert ranking._calc_recent_growth({"a/b": {"star": 100}}, recent_days=3) == ({}, 3)


def test_burst_probe_reports_delayed_anchor_window(snap_dir):
    """锚点顺延时必须回报实际天数，不能让调用方按请求的 3 天算速率。

    这是"同一条规则在三处各写一遍"留下的真 bug：主窗口那侧修正了、探针这侧漏了，
    于是 5 天的增量被除以 3、速率虚高 67%，acceleration 凭空 >1，爆发加成误判。
    现在实际天数跟着数据一起返回，调用方没机会忘。
    """
    from hot_projects.tools.tool import ranking
    # 只有 T−5 的快照（T−3、T−4 漏采），容差 2 天内会顺延到它
    snapshots.save_snapshot(snapshots.utc_today() - timedelta(days=5), {"a/b": 1000})
    got, window = ranking._calc_recent_growth({"a/b": {"star": 1500}}, recent_days=3)
    assert got == {"a/b": 500}
    assert window == 5, "拿的是 5 天前的锚点，就必须如实报 5 天"


# ── 全 null 批次护栏 ──

class _FakePool:
    def __init__(self, token_count=1):
        self.released = 0
        self._token_count = token_count

    @property
    def token_count(self):
        return self._token_count

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


def test_concurrency_defaults_to_token_count(monkeypatch):
    """concurrency=None 时并发跟着 token 数走：worker 不多于 token，不刷「等待」日志。"""
    monkeypatch.setattr(star_snapshot, "SNAPSHOT_MAX_CONCURRENCY", 12)
    seen = {}
    real_sem = asyncio.Semaphore

    def spy(n):
        seen["n"] = n
        return real_sem(n)

    monkeypatch.setattr(star_snapshot.asyncio, "Semaphore", spy)

    names = [f"o{i}/r{i}" for i in range(4)]
    client = _FakeClient({n: 1 for n in names}, max_ok=99)
    monkeypatch.setattr(star_snapshot, "_build_async_client", lambda **kw: client)

    pool = _FakePool(token_count=3)  # 3 个 token
    asyncio.run(star_snapshot.collect_star_snapshot(pool, names, batch_size=1))
    assert seen["n"] == 3, "并发应等于 token 数（3），而非上限 12"


def test_concurrency_capped_at_max(monkeypatch):
    monkeypatch.setattr(star_snapshot, "SNAPSHOT_MAX_CONCURRENCY", 4)
    seen = {}
    real_sem = asyncio.Semaphore
    monkeypatch.setattr(star_snapshot.asyncio, "Semaphore",
                        lambda n: seen.update(n=n) or real_sem(n))

    names = [f"o{i}/r{i}" for i in range(6)]
    client = _FakeClient({n: 1 for n in names}, max_ok=99)
    monkeypatch.setattr(star_snapshot, "_build_async_client", lambda **kw: client)

    pool = _FakePool(token_count=50)  # 贴了 50 个 token
    asyncio.run(star_snapshot.collect_star_snapshot(pool, names, batch_size=1))
    assert seen["n"] == 4, "token 再多也不该超过安全上限 4（二级限流按并发/IP 计）"


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

def _growth_ctx(window=7):
    return {
        "candidate_map": {},
        "growth_threshold": 200,
        "can_write_db": False,
        "window_specified": True,
        "growth_calc_days": window,
        "is_hot_new": False,
        "prev_snapshot": {},
        "use_checkpoint": False,
        "checkpoint_dirty": [False],
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

    task_help._resolve_growth(raw_repos, {"projects": {}}, candidate_map, ctx)

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
    task_help._resolve_growth(
        {"a/b": {"star": 1900, "repo_item": {}, "created_at": _ts(400)}},
        {"projects": {}}, candidate_map, ctx,
    )
    assert ctx["effective_growth_calc_days"] == 7


def test_hot_new_board_also_resolves_from_anchor(snap_dir, monkeypatch):
    """新项目榜必须一起走本地定案。

    它原先整体跳过差值、全部交给实时二分法；实时那条路删掉后若还排除它，
    新项目榜会一个候选都出不来（全部记未决），且是静默的。
    """
    from hot_projects.infra.concurrency import task_help

    monkeypatch.setattr(task_help, "SNAPSHOT_ANCHOR_TOLERANCE_DAYS", 2)
    snapshots.save_snapshot(snapshots.utc_today() - timedelta(days=7), {"a/b": 1000})

    ctx = _growth_ctx(window=7)
    ctx["is_hot_new"] = True
    candidate_map = {}
    ctx["candidate_map"] = candidate_map
    task_help._resolve_growth(
        {"a/b": {"star": 1900, "repo_item": {}, "created_at": _ts(400)}},
        {"projects": {}}, candidate_map, ctx,
    )

    assert candidate_map["a/b"]["growth"] == 900
    assert ctx["unresolved_count"] == 0


# ── 单仓库增长工具（Agent 的 repo_growth）也改成快照减法 ──

def _stub_repo_api(monkeypatch, core, star=1900):
    monkeypatch.setattr(core, "fetch_repo_info", lambda *a, **k: {
        "stargazers_count": star, "html_url": "https://github.com/a/b",
        "description": "", "language": "", "topics": [], "created_at": _ts(400),
    })
    monkeypatch.setattr(core, "call_llm_describe", lambda *a, **k: "desc")


def test_repo_growth_tool_subtracts_anchor(snap_dir, monkeypatch):
    """锚点顺延到 T−9 时，返回的窗口口径必须是 9 天而不是请求的 7 天。

    报 7 天会让调用方按 900/7 理解速率，而这 900 实际跨了 9 天，虚高 29%。
    """
    from hot_projects.tools.basic import core

    snapshots.save_snapshot(snapshots.utc_today() - timedelta(days=9), {"a/b": 1000})
    _stub_repo_api(monkeypatch, core)

    out = core.check_repo_growth(None, "a/b", growth_calc_days=7)

    assert out["growth"] == 900
    assert out["growth_status"] == "ok"
    assert out["growth_calc_days"] == 9


def test_repo_growth_tool_reports_unresolved_without_anchor(snap_dir, monkeypatch):
    """缺快照要如实报未决：返回 0 会让调用方把"没测过"当成"没涨"。"""
    from hot_projects.tools.basic import core

    _stub_repo_api(monkeypatch, core)

    out = core.check_repo_growth(None, "a/b", growth_calc_days=7)

    assert out["growth"] is None
    assert out["growth_status"] == "snapshot_unresolved"
    assert out["meets_threshold"] is False


# ── DB 淘汰：这条路径会删数据，保护名单和 grace 期各留一条断言 ──

def _stale(projects, snaps, *, floor=500, protect_new_days=90, keep=frozenset()):
    return sorted(db._stale_project_names(projects, snaps, floor, protect_new_days, keep))


def test_evict_needs_full_grace_window(tmp_path, monkeypatch):
    """快照份数不够 grace 天时一个都不能删。

    刚接入时只有一两份快照，此时"低于门槛"完全无法区分"长期掉队"和"今天刚好抖一下"。
    走完整的 evict_stale_projects（而非纯判定函数），因为这道闸就在写盘那一层；
    DB 路径指到 tmp_path，闸门若失效会删掉这份临时 DB 而不是真库。
    """
    fake_db = tmp_path / "Github_DB.json"
    fake_db.write_text(json.dumps(
        {"projects": {"a/b": {"star": 100, "created_at": _ts(400)}}}), encoding="utf-8")
    monkeypatch.setattr(db, "DB_FILE_PATH", str(fake_db))

    assert db.evict_stale_projects([{"a/b": 100}] * 3, star_floor=500,
                                   grace_days=7, protect_new_days=90) == []
    assert "a/b" in json.loads(fake_db.read_text(encoding="utf-8"))["projects"]


def test_evict_only_when_below_floor_every_day():
    old = {"created_at": _ts(400)}
    projects = {"dead/repo": dict(old), "dipped/repo": dict(old)}
    snaps = [{"dead/repo": 120, "dipped/repo": 120}] * 6
    # 只有一天回到门槛以上，就不算长期掉队——500 线附近的抖动不该触发删除。
    snaps.append({"dead/repo": 120, "dipped/repo": 640})

    assert _stale(projects, snaps) == ["dead/repo"]


def test_evict_skips_repo_missing_from_any_snapshot():
    """任一天读数缺失就不判：把"没采到"当成"掉下去了"会误删活跃仓库。"""
    projects = {"a/b": {"star": 9000, "created_at": _ts(400)}}
    snaps = [{"a/b": 120}] * 6 + [{}]   # 最后一天漏采

    assert _stale(projects, snaps) == []


def test_evict_protects_ranked_favorited_and_new():
    snaps = [{n: 100 for n in ("ranked/r", "fav/r", "new/r", "plain/r")}] * 7
    projects = {
        "ranked/r": {"created_at": _ts(400), "desc": "上过榜，有 LLM 描述"},
        "fav/r": {"created_at": _ts(400)},
        "new/r": {"created_at": _ts(30)},      # 近 90 天新建，还没长起来
        "plain/r": {"created_at": _ts(400)},
    }

    assert _stale(projects, snaps, keep={"fav/r"}) == ["plain/r"]
