"""收藏存储 + API 测试。"""

import importlib

import pytest
from fastapi.testclient import TestClient

import hot_projects.infra.favorites_store as store


@pytest.fixture
def fresh_store(tmp_path, monkeypatch):
    path = str(tmp_path / "favorites.json")
    monkeypatch.setattr(store, "FAVORITES_FILE_PATH", path)
    return store


# ── store 单元 ──

def test_add_remove_and_dedup(fresh_store):
    fresh_store.set_favorite("blue", "a/b", "add", source_report="2026-07-01.md")
    fresh_store.set_favorite("blue", "a/b", "add")  # 幂等
    items = fresh_store.get_favorites("blue")
    assert [x["repo"] for x in items] == ["a/b"]
    assert items[0]["source_report"] == "2026-07-01.md"

    fresh_store.set_favorite("blue", "a/b", "remove")
    assert fresh_store.get_favorites("blue") == []


def test_user_isolation(fresh_store):
    fresh_store.set_favorite("user1", "a/b", "add")
    fresh_store.set_favorite("user2", "c/d", "add")
    assert [x["repo"] for x in fresh_store.get_favorites("user1")] == ["a/b"]
    assert [x["repo"] for x in fresh_store.get_favorites("user2")] == ["c/d"]


def test_invalid_inputs(fresh_store):
    with pytest.raises(ValueError):
        fresh_store.set_favorite("x", "a/b", "add")          # user_id 太短
    with pytest.raises(ValueError):
        fresh_store.set_favorite("blue", "no-slash", "add")  # repo 非法
    with pytest.raises(ValueError):
        fresh_store.set_favorite("blue", "a/b", "bad")       # action 非法


def test_newest_first(fresh_store):
    fresh_store.set_favorite("blue", "a/b", "add")
    fresh_store.set_favorite("blue", "c/d", "add")
    assert [x["repo"] for x in fresh_store.get_favorites("blue")] == ["c/d", "a/b"]


# ── 分类标签 ──

def test_category_stored_and_cleaned(fresh_store):
    fresh_store.set_favorite("blue", "a/b", "add", category="  效率  标签 ")
    item = fresh_store.get_favorites("blue")[0]
    assert item["category"] == "效率 标签"  # 折叠空白 + 去首尾

    # 非空白控制字符（\x00 \x07 \x1f \x7f）应被剔除，不残留不可见字符
    assert fresh_store.clean_category("效\x00率\x1f工\x7f具") == "效 率 工 具"

    # 超长截断到 MAX_CATEGORY_LEN
    fresh_store.set_favorite("blue", "c/d", "add", category="x" * 40)
    cat = next(x for x in fresh_store.get_favorites("blue") if x["repo"] == "c/d")["category"]
    assert len(cat) == fresh_store.MAX_CATEGORY_LEN


def test_category_none_keeps_existing_str_overwrites(fresh_store):
    fresh_store.set_favorite("blue", "a/b", "add", category="工具")
    fresh_store.set_favorite("blue", "a/b", "add")  # category=None → 不改动
    assert fresh_store.get_favorites("blue")[0]["category"] == "工具"

    fresh_store.set_favorite("blue", "a/b", "add", category="效率")  # 显式覆盖
    assert fresh_store.get_favorites("blue")[0]["category"] == "效率"

    fresh_store.set_favorite("blue", "a/b", "add", category="")  # 显式清空 → 未分类
    assert fresh_store.get_favorites("blue")[0]["category"] == ""


def test_new_favorite_defaults_to_empty_category(fresh_store):
    fresh_store.set_favorite("blue", "a/b", "add")
    assert fresh_store.get_favorites("blue")[0]["category"] == ""


# ── API 集成 ──

@pytest.fixture
def client(tmp_path, monkeypatch):
    path = str(tmp_path / "favorites.json")
    monkeypatch.setattr(store, "FAVORITES_FILE_PATH", path)
    import hot_projects.api_server as api
    importlib.reload  # noqa: B018  (确保引用存在)
    return TestClient(api.app)


def test_api_add_list_remove(client):
    r = client.post("/api/favorites", json={"user_id": "blue", "repo": "a/b", "action": "add"})
    assert r.status_code == 200
    assert r.json()["favorites"][0]["repo"] == "a/b"

    r = client.get("/api/favorites", params={"user_id": "blue"})
    assert r.status_code == 200
    assert [x["repo"] for x in r.json()["favorites"]] == ["a/b"]

    r = client.post("/api/favorites", json={"user_id": "blue", "repo": "a/b", "action": "remove"})
    assert r.json()["favorites"] == []


def test_api_rejects_bad_user(client):
    assert client.get("/api/favorites", params={"user_id": "x"}).status_code == 400
    r = client.post("/api/favorites", json={"user_id": "x", "repo": "a/b", "action": "add"})
    assert r.status_code == 400


def test_api_category_roundtrip(client):
    r = client.post("/api/favorites",
                    json={"user_id": "blue", "repo": "a/b", "action": "add", "category": "工具"})
    assert r.status_code == 200
    assert r.json()["favorites"][0]["category"] == "工具"


def test_api_favorite_tags(client):
    from hot_projects.config import FAVORITE_DEFAULT_TAGS
    r = client.get("/api/favorite-tags")
    assert r.status_code == 200
    assert r.json()["tags"] == list(FAVORITE_DEFAULT_TAGS)
