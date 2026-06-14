from hot_projects.providers.base import Repo
from hot_projects.tools.atomic_tools import (
    repo_growth_handler, describe_project_handler, get_db_info_handler, fetch_trending_handler,
)


class _Prov:
    def __init__(self, info=None, similar=None, token_mgr=None):
        self._info = info
        self._similar = similar or []
        self.token_mgr = token_mgr

    def repo_info(self, repo):
        return self._info

    def repo_growth(self, repo, growth_calc_days):
        return {"repo": repo, "growth": 123}

    def fetch_trending(self, trending_range):
        return {"trending_range": trending_range, "repos": []}

    def search_similar(self, name, limit=5):
        return self._similar


class _State:
    active_repo = None


class _Ctx:
    def __init__(self, prov):
        self.provider = prov
        self.db = {"projects": {}}
        self.state = _State()


def test_exact_hit_returns_growth():
    ctx = _Ctx(_Prov(info=Repo("a/b", 1500)))
    out = repo_growth_handler(ctx, {"repo": "a/b"})
    assert out["growth"] == 123
    assert ctx.state.active_repo == "a/b"


def test_miss_returns_candidates():
    ctx = _Ctx(_Prov(info=None, similar=[Repo("x/vllm", 1300), Repo("y/vllm", 1200)]))
    out = repo_growth_handler(ctx, {"repo": "vllm"})
    assert out.get("disambiguation") is True
    assert "x/vllm" in [c["full_name"] for c in out["candidates"]]


def test_miss_no_similar_returns_error():
    ctx = _Ctx(_Prov(info=None, similar=[]))
    out = repo_growth_handler(ctx, {"repo": "zzz"})
    assert "error" in out


def test_get_db_info():
    ctx = _Ctx(_Prov())
    ctx.db = {"valid": True, "date": "2026-06-10", "projects": {"a/b": {"star": 1}}}
    assert get_db_info_handler(ctx, {})["total_projects"] == 1


def test_fetch_trending_passthrough():
    ctx = _Ctx(_Prov())
    assert fetch_trending_handler(ctx, {"trending_range": "daily"})["trending_range"] == "daily"
