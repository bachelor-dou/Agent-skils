"""数据层:真实数据的读取自检 + 四类写入拦截。

读那几条打**真实数据**(`data/Github_DB.json` 与 `data/snapshots/*.json.gz`):路径搬家、
REPO_ROOT 推导错位这类事故的表现都是「读到空」,而拿构造数据测永远发现不了。
写入那几条一律在临时目录上跑,绝不碰真数据。

2026-07-30 之前这里还有三条「与旧包读取层逐条对照」的测试(DB / 快照 / 收藏,逐字段比)。
它们是重构期的脚手架:证明新读取层和旧的看到的东西一模一样。切 CI、数据搬到 `data/`
之后旧包不再持有数据,对照没有了对象,连同 `_compare.py` 一起删掉。
"""

from __future__ import annotations

import gzip
import json
import threading
from datetime import timedelta
from pathlib import Path

import pytest

from hot_project import config
from hot_project.infra.store import atomic, favorites, snapshots, universe
from hot_project.infra.store.atomic import StoreReadError

# ──────────────────────────────────────────────────────────
# 真实数据的读取自检
# ──────────────────────────────────────────────────────────


def test_the_live_database_reads_back_as_projects():
    """读得到、且不是空的。搬完家第一件要确认的事。"""
    projects = universe.load()
    assert len(projects) > 1000, f"只读到 {len(projects)} 个仓库 —— 路径是不是指错了?"
    sample = next(iter(projects.values()))
    assert "star" in sample


def test_every_live_snapshot_parses():
    days = snapshots.available_dates()
    assert days, "读不到任何真实快照 —— 路径是不是指错了?"
    for day in days:
        stars = snapshots.load_stars(day)
        assert stars, f"{day} 的快照读出来是空的"


def test_the_live_favorites_read_back():
    """收藏是唯一由人手工攒出来的数据,丢了没有任何东西能重新生成它。"""
    assert favorites.all_repos(), "读不到任何收藏 —— 路径是不是指错了?"


def test_anchor_reports_the_real_window():
    """锚点的 window_days 必须是它到今天的真实天数,不是请求的天数。"""
    days = snapshots.available_dates()
    newest = days[-1]
    anchor = snapshots.anchor_for_window((snapshots.utc_today() - newest).days)
    assert anchor is not None
    assert anchor.window_days == (snapshots.utc_today() - anchor.day).days


# ──────────────────────────────────────────────────────────
# 写入:一律在临时目录
# ──────────────────────────────────────────────────────────


@pytest.fixture
def sandbox(tmp_path, monkeypatch):
    """把全部读写路径改到临时目录。返回一个能预置 DB 内容的函数。"""
    data = tmp_path / "data"
    (data / "snapshots").mkdir(parents=True)
    for name, value in {
        "DATA_DIR": data,
        "DB_PATH": data / "Github_DB.json",
        "FAVORITES_PATH": data / "favorites.json",
        "SNAPSHOT_DIR": data / "snapshots",
    }.items():
        monkeypatch.setattr(config, name, value)

    def seed(db: dict) -> Path:
        path = config.DB_PATH
        path.write_text(json.dumps(db, ensure_ascii=False), encoding="utf-8")
        return path

    return seed


def _read_db() -> dict:
    return json.loads(config.DB_PATH.read_text(encoding="utf-8"))


# ── 拦截 1:发现任务只能写自己那两个字段 ──


@pytest.mark.parametrize("field", ["gh_desc", "desc", "forks", "language"])
def test_discover_rejects_foreign_fields(sandbox, field):
    sandbox({"projects": {}})
    with pytest.raises(ValueError, match="不许写字段"):
        universe.insert_discovered({"a/b": {"star": 600, field: "x"}})
    assert _read_db()["projects"] == {}, "被拒的写入不该留下痕迹"


def test_discover_does_not_touch_top_level_date(sandbox):
    """顶层 date 是旧包报告推断窗口的依据,每日任务改一次就会让它恒为 0。

    新设计里没有任何函数能写它 —— 这条测试钉住「插入不会顺手改它」。
    """
    sandbox({"date": "2026-07-01", "projects": {}})
    universe.insert_discovered({"a/b": {"star": 600, "created_at": "2026-01-01T00:00:00Z"}})
    assert _read_db()["date"] == "2026-07-01"


def test_reading_preserves_fields_this_version_knows_nothing_about(sandbox):
    """`load()` 是每一次写回的输入,所以它丢掉的字段等于从盘上永久删掉。

    DB 里攒着 LLM 生成的描述之类不可再生的东西。归一化成「我认识的那几个字段」看着无害,
    实际是下一次每日任务就把它们全抹了 —— 而且没有任何报错。
    """
    sandbox({"projects": {"a/b": {"star": 900, "desc": "花钱生成的描述",
                                  "未来字段": [1, 2]}}})

    assert universe.load()["a/b"] == {"star": 900, "desc": "花钱生成的描述",
                                      "未来字段": [1, 2]}

    universe.insert_discovered({"c/d": {"star": 600}})
    assert _read_db()["projects"]["a/b"]["未来字段"] == [1, 2], "写回抹掉了不认识的字段"


def test_discover_only_inserts_new(sandbox):
    """已有条目一律不碰:发现阶段的粗字段不该盖掉后来补的完整数据。"""
    sandbox({"projects": {"old/repo": {"star": 900, "desc": "已有描述", "forks": 12}}})

    inserted = universe.insert_discovered({
        "old/repo": {"star": 5000, "created_at": "2020-01-01T00:00:00Z"},
        "new/repo": {"star": 700, "created_at": "2026-07-01T00:00:00Z"},
    })

    projects = _read_db()["projects"]
    assert inserted == ["new/repo"]
    assert projects["old/repo"] == {"star": 900, "desc": "已有描述", "forks": 12}
    assert projects["new/repo"]["star"] == 700


# ── 拦截 2:展示字段仅补空,缺 forks 不得写 0(曾真实抹库)──


def test_display_refresh_missing_forks_keeps_stored_value(sandbox):
    """周报候选池改读快照后 repo_item 里没有 forks_count,旧代码照写 0 抹掉了全库 fork 数。

    「缺」必须区别于「0」:没给这个键 = 这次没查到,保持原值。
    """
    sandbox({"projects": {"a/b": {"star": 900, "forks": 456}}})

    universe.refresh_display({"a/b": {"language": "Python"}})   # 故意不给 forks

    record = _read_db()["projects"]["a/b"]
    assert record["forks"] == 456, "forks 被抹掉了"
    assert record["language"] == "Python"


def test_display_refresh_never_overwrites_existing(sandbox):
    sandbox({"projects": {"a/b": {"star": 900, "language": "Rust", "gh_desc": "原始简介"}}})

    universe.refresh_display({"a/b": {"language": "Python", "gh_desc": "新简介",
                                      "topics": ["ai"]}})

    record = _read_db()["projects"]["a/b"]
    assert record["language"] == "Rust"
    assert record["gh_desc"] == "原始简介"
    assert record["topics"] == ["ai"], "空缺的字段仍然要补上"


def test_display_refresh_rejects_foreign_fields(sandbox):
    sandbox({"projects": {"a/b": {"star": 1}}})
    with pytest.raises(ValueError, match="不许写字段"):
        universe.refresh_display({"a/b": {"star": 999}})


# ── 拦截 3:star 刷新只动 star,且只动已存在的仓库 ──


def test_refresh_stars_ignores_unknown_repos(sandbox):
    sandbox({"projects": {"a/b": {"star": 100, "desc": "x"}}})

    updated = universe.refresh_stars({"a/b": 150, "never/seen": 9999})

    projects = _read_db()["projects"]
    assert updated == 1
    assert projects["a/b"] == {"star": 150, "desc": "x"}
    assert "never/seen" not in projects


def test_no_change_writes_nothing(sandbox):
    """star 没变就别重写:主库 30MB,白写一次要 0.8 秒还会在 git 里留个空改动。"""
    path = sandbox({"projects": {"a/b": {"star": 100}}})
    before = path.stat().st_mtime_ns

    assert universe.refresh_stars({"a/b": 100}) == 0
    assert path.stat().st_mtime_ns == before, "无变化却重写了文件"


# ── 拦截 4:读不出来就绝不覆盖 ──


def test_corrupt_db_is_never_overwritten(sandbox):
    """截断的 JSON:必须抛,且原文件字节数不变。

    旧 save_db 在这里把读失败当成 `{}` 然后照写,一次 JSON 截断就清空 5 万条记录。
    """
    path = sandbox({"projects": {"a/b": {"star": 100}}})
    path.write_text('{"projects": {"a/b": {"star": 10', encoding="utf-8")
    original = path.read_bytes()

    for call in (
        lambda: universe.insert_discovered({"c/d": {"star": 600}}),
        lambda: universe.refresh_stars({"a/b": 200}),
        lambda: universe.refresh_display({"a/b": {"language": "Go"}}),
        lambda: universe.write_descriptions({"a/b": {"desc": "x"}}),
        lambda: universe.evict({"a/b"}),
        lambda: universe.load(),
    ):
        with pytest.raises(StoreReadError):
            call()

    assert path.read_bytes() == original, "损坏的文件被覆盖了"


def test_caller_exception_abandons_the_write(sandbox):
    path = sandbox({"projects": {"a/b": {"star": 100}}})
    original = path.read_bytes()

    with pytest.raises(RuntimeError, match="boom"):
        with atomic.transaction(path) as tx:
            tx.data["projects"]["a/b"]["star"] = 999
            raise RuntimeError("boom")

    assert path.read_bytes() == original


# ── 快照:拒写空的与低覆盖的 ──


def test_snapshot_rejects_empty(sandbox):
    day = snapshots.utc_today()
    assert snapshots.save(day, {}, not_found=[], expected=100) is None
    assert not list(config.SNAPSHOT_DIR.iterdir())


def test_snapshot_rejects_low_coverage(sandbox):
    """半份快照落盘后会被当成锚点,把「没测到」算成「掉到 0」,整批虚假负增长。"""
    day = snapshots.utc_today()
    stars = {f"o/r{i}": 100 for i in range(40)}

    assert snapshots.save(day, stars, not_found=[], expected=100) is None
    assert not list(config.SNAPSHOT_DIR.iterdir()), "低覆盖的快照落盘了"


def test_snapshot_roundtrip_with_meta(sandbox):
    day = snapshots.utc_today()
    stars = {f"o/r{i}": 100 + i for i in range(90)}

    path = snapshots.save(day, stars, not_found=["gone/repo"], expected=100)

    assert path is not None
    assert snapshots.load_stars(day) == stars
    assert snapshots.load_not_found(day) == ["gone/repo"]


def test_snapshot_reads_legacy_flat_format(sandbox):
    """历史快照是扁平的 {"owner/repo": star},必须照样能当锚点。"""
    day = snapshots.utc_today()
    with gzip.open(snapshots.path_of(day), "wt", encoding="utf-8") as f:
        json.dump({"a/b": 123}, f)

    assert snapshots.load_stars(day) == {"a/b": 123}
    assert snapshots.load_not_found(day) == []


def test_corrupt_snapshot_is_treated_as_missing(sandbox):
    """快照可替代(顺延到邻近那天),所以损坏按缺失处理 —— 和 DB 相反。"""
    day = snapshots.utc_today()
    snapshots.path_of(day).write_bytes(b"not gzip at all")
    assert snapshots.load_stars(day) is None


def test_prune_keeps_recent_and_drops_old(sandbox):
    today = snapshots.utc_today()
    for offset in (0, 3, 40):
        with gzip.open(snapshots.path_of(today - timedelta(days=offset)),
                       "wt", encoding="utf-8") as f:
            json.dump({"stars": {"a/b": 1}}, f)

    removed = snapshots.prune(keep_days=35, today=today)

    assert removed == [today - timedelta(days=40)]
    assert snapshots.load_stars(today) is not None
    assert snapshots.load_stars(today - timedelta(days=3)) is not None


def test_prune_only_deletes_snapshots(sandbox):
    """目录里除了快照还可能有 .lock、.tmp 之类的东西,清理不该顺手带走它们。"""
    stale = snapshots.utc_today() - timedelta(days=400)
    with gzip.open(snapshots.path_of(stale), "wt", encoding="utf-8") as f:
        json.dump({"stars": {"a/b": 1}}, f)
    (config.SNAPSHOT_DIR / "2026-01-01.json.gz.lock").write_text("", encoding="utf-8")

    assert snapshots.prune(keep_days=35) == [stale]
    assert (config.SNAPSHOT_DIR / "2026-01-01.json.gz.lock").exists()


def test_a_snapshot_written_today_makes_the_run_idempotent(sandbox):
    """幂等是每日脚本敢每小时触发一次的全部依据。

    GitHub 的 schedule 会漂、会静默跳过,所以一天给自己 24 次机会;当天已有快照就秒退、
    一个请求都不发。这条一坏,就是一天跑 24 遍全量采集。
    """
    today = snapshots.utc_today()
    assert not snapshots.already_written(today)

    snapshots.save(today, {"a/b": 2}, not_found=[], expected=1)
    assert snapshots.already_written(today)


# ── 收藏:并发 add 不能丢 ──


def test_concurrent_favorites_do_not_lose_updates(sandbox):
    """旧实现读完就放锁,两个并发请求会互相抹掉。现在读改写在同一把排他锁里。"""
    repos = [f"owner/repo{i}" for i in range(8)]
    errors: list[BaseException] = []

    def add(repo: str) -> None:
        try:
            favorites.set_favorite("tester", repo, "add")
        except BaseException as e:      # noqa: BLE001 —— 线程里的异常要带回主线程
            errors.append(e)

    threads = [threading.Thread(target=add, args=(r,)) for r in repos]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not errors, errors
    assert {item["repo"] for item in favorites.get("tester")} == set(repos)


def test_favorites_rejects_bad_input(sandbox):
    for bad in (("x", "owner/repo", "add"),            # user_id 太短
                ("tester", "not-a-repo", "add"),       # repo 格式不对
                ("tester", "owner/repo", "sideways")):  # action 不认识
        with pytest.raises(ValueError):
            favorites.set_favorite(*bad)
