"""search_repos 按描述找项目工具测试。"""

from hot_projects.datasource.base import Repo
from hot_projects.tools.tool.search_repos import search_repos_handler


class _Prov:
    def __init__(self, results=None):
        self._results = results or []
        self.last_query = None
        self.last_top_n = None
        self.last_min_star = None

    def search_top_repos(self, query, top_n=5, min_star=0):
        self.last_query = query
        self.last_top_n = top_n
        self.last_min_star = min_star
        return self._results[:top_n]


class _State:
    active_repo = None


class _Ctx:
    def __init__(self, prov):
        self.provider = prov
        self.db = {"projects": {}}
        self.state = _State()


def test_returns_ranked_results():
    prov = _Prov([
        Repo("a/agent-remote", 5000, description="control agent from phone", language="Python"),
        Repo("b/mobile-ctl", 1200, description="mobile control", language="Go"),
    ])
    out = search_repos_handler(_Ctx(prov), {"query": "mobile remote agent", "top_n": 5})
    assert out["count"] == 2
    assert out["results"][0]["repo"] == "a/agent-remote"
    assert out["results"][0]["rank"] == 1
    assert out["results"][0]["url"] == "https://github.com/a/agent-remote"


def test_appends_in_qualifier_by_default():
    prov = _Prov([Repo("a/b", 10)])
    search_repos_handler(_Ctx(prov), {"query": "remote agent"})
    assert "in:name,description,readme" in prov.last_query


def test_respects_explicit_in_qualifier():
    prov = _Prov([Repo("a/b", 10)])
    search_repos_handler(_Ctx(prov), {"query": "remote agent in:name"})
    # 已含 in: 则不重复追加
    assert prov.last_query.count("in:") == 1


def test_empty_query_errors():
    out = search_repos_handler(_Ctx(_Prov()), {"query": "  "})
    assert "error" in out


def test_no_results_message():
    out = search_repos_handler(_Ctx(_Prov([])), {"query": "zzz nonexistent"})
    assert out["count"] == 0
    assert "message" in out


def test_top_n_passed_through():
    prov = _Prov([Repo(f"o/r{i}", i) for i in range(10)])
    out = search_repos_handler(_Ctx(prov), {"query": "x", "top_n": 3, "min_star": 1000})
    assert prov.last_top_n == 3
    assert prov.last_min_star == 1000
    assert out["count"] == 3


def test_registry_has_search_repos():
    from hot_projects.tools.registry import build_default_registry
    names = {s["function"]["name"] for s in build_default_registry().schemas()}
    assert "search_repos" in names
