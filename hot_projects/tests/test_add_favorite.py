"""add_favorite 工具测试：DB 补全 + 一句话概要 + 按用户落盘收藏。"""

import hot_projects.infra.favorites_store as store
import hot_projects.tools.tool.add_favorite as AF
from hot_projects.tools.tool.add_favorite import add_favorite_handler


class _Prov:
    token_mgr = object()

    def repo_info(self, repo):
        # resolve_repo 走精确命中路径
        from hot_projects.datasource.base import Repo
        return Repo(repo, 1234)


class _State:
    active_repo = None


class _Ctx:
    def __init__(self, db, user_id):
        self.provider = _Prov()
        self.db = db
        self.state = _State()
        self.user_id = user_id


def _patch_store(monkeypatch, tmp_path):
    fp = str(tmp_path / "favorites.json")
    monkeypatch.setattr(store, "FAVORITES_FILE_PATH", fp)
    return fp


def test_add_fetches_db_and_stores_short_desc(monkeypatch, tmp_path):
    _patch_store(monkeypatch, tmp_path)
    fetched = {"n": 0}

    def _fake_fetch(tm, o, r):
        fetched["n"] += 1
        return {"stargazers_count": 999, "description": "an OCR toolkit",
                "language": "Python", "topics": ["ocr"], "forks_count": 10,
                "created_at": "2024-01-01T00:00:00Z"}

    monkeypatch.setattr(AF, "fetch_repo_info", _fake_fetch)
    monkeypatch.setattr(AF, "save_db", lambda db: None)
    monkeypatch.setattr(AF, "batch_condense_descriptions", lambda repos, max_chars=60: ["OCR 工具库"])

    db = {"projects": {}}
    ctx = _Ctx(db, "alice")
    out = add_favorite_handler(ctx, {"repo": "a/b"})

    assert out["ok"] is True and out["repo"] == "a/b"
    assert out["short_desc"] == "OCR 工具库"
    assert fetched["n"] == 1                      # DB 缺失 → 拉了一次
    assert db["projects"]["a/b"]["star"] == 999   # 已入库
    favs = store.get_favorites("alice")
    assert favs[0]["repo"] == "a/b" and favs[0]["short_desc"] == "OCR 工具库"


def test_existing_db_project_not_refetched(monkeypatch, tmp_path):
    _patch_store(monkeypatch, tmp_path)
    monkeypatch.setattr(AF, "fetch_repo_info",
                        lambda *a: (_ for _ in ()).throw(AssertionError("不应联网")))
    monkeypatch.setattr(AF, "batch_condense_descriptions", lambda repos, max_chars=60: ["缓存概要"])

    db = {"projects": {"a/b": {"star": 5, "short_desc": "cached desc"}}}
    ctx = _Ctx(db, "bob")
    out = add_favorite_handler(ctx, {"repo": "a/b"})
    assert out["ok"] is True and out["short_desc"] == "缓存概要"


def test_not_logged_in_blocks(monkeypatch, tmp_path):
    _patch_store(monkeypatch, tmp_path)
    ctx = _Ctx({"projects": {}}, "")   # 无 user_id
    out = add_favorite_handler(ctx, {"repo": "a/b"})
    assert "error" in out and "登录" in out["error"]


def test_registry_contains_add_favorite():
    from hot_projects.tools.registry import build_default_registry
    names = {s["function"]["name"] for s in build_default_registry().schemas()}
    assert "add_favorite" in names
