"""repo_profile 合并取证工具测试（功能证据 + 维护活跃度一次返回）。"""

import hot_projects.tools.tool.repo_profile as P
from hot_projects.datasource.base import Repo
from hot_projects.tools.tool.repo_profile import repo_profile_handler


_INFO = {
    "html_url": "https://github.com/a/b",
    "description": "an OCR toolkit",
    "language": "Python",
    "topics": ["ocr", "deep-learning"],
    "stargazers_count": 5000,
    "forks_count": 300,
    "open_issues_count": 42,
    "created_at": "2024-01-01T00:00:00Z",
    "pushed_at": "2026-07-06T00:00:00Z",
    "license": {"spdx_id": "MIT"},
    "archived": False,
}


def _patch_common(monkeypatch, readme_text, truncated=False):
    monkeypatch.setattr(P, "fetch_repo_info", lambda tm, o, r: dict(_INFO))
    monkeypatch.setattr(P, "fetch_repo_readme_excerpt",
                        lambda tm, o, r, max_chars: {"text": readme_text, "truncated": truncated})
    monkeypatch.setattr(P, "fetch_repo_recent_releases",
                        lambda tm, o, r: [{"tag_name": "v1.0", "published_at": "2026-07-01"}])
    monkeypatch.setattr(P, "fetch_repo_recent_commits",
                        lambda tm, o, r, per_page=10: [{"sha": "abc", "date": "2026-07-06", "message": "fix"}])


def test_profile_rich_readme_has_all_signals(monkeypatch):
    _patch_common(monkeypatch, "x" * 2000, truncated=True)
    called = {"tree": False}
    monkeypatch.setattr(P, "fetch_repo_tree_paths", lambda tm, o, r: called.update(tree=True) or [])

    out = P._profile("a/b", token_mgr=object())
    assert out["star"] == 5000 and out["license"] == "MIT"
    assert out["pushed_at"] == "2026-07-06T00:00:00Z"
    assert out["recent_releases"][0]["tag_name"] == "v1.0"
    assert len(out["recent_commits"]) == 1
    assert out["readme_truncated"] is True
    assert "structure_hint" not in out
    assert called["tree"] is False  # README 充足时不多花目录请求


def test_profile_thin_readme_adds_tree_hint(monkeypatch):
    _patch_common(monkeypatch, "简陋")
    monkeypatch.setattr(P, "fetch_repo_tree_paths",
                        lambda tm, o, r: ["docs/languages.md", "examples/table_ocr.py"])
    out = P._profile("a/b", token_mgr=object())
    assert out["structure_hint"] == ["docs/languages.md", "examples/table_ocr.py"]


class _Prov:
    token_mgr = object()

    def repo_info(self, repo):
        return Repo(repo, 5000)


class _State:
    active_repo = None


class _Ctx:
    provider = _Prov()
    db = {"projects": {}}
    state = _State()


def test_handler_resolves_and_dispatches(monkeypatch):
    _patch_common(monkeypatch, "x" * 2000)
    ctx = _Ctx()
    out = repo_profile_handler(ctx, {"repo": "a/b"})
    assert out["repo"] == "a/b"
    assert ctx.state.active_repo == "a/b"


def test_registry_contains_repo_profile():
    from hot_projects.tools.registry import build_default_registry
    names = {s["function"]["name"] for s in build_default_registry().schemas()}
    assert "repo_profile" in names
    assert "repo_overview" not in names and "repo_activity" not in names
