from unittest.mock import patch

from hot_projects.providers.github.provider import GitHubProvider


def test_search_similar_returns_repos():
    fake_items = [
        {"full_name": "vllm-project/vllm", "stargazers_count": 30000},
        {"full_name": "x/vllm-fork", "stargazers_count": 1300},
    ]
    with patch("hot_projects.providers.github.provider.search_github_repos", return_value=fake_items):
        p = GitHubProvider(token_mgr=object())
        repos = p.search_similar("vllm", limit=5)
    assert repos[0].full_name == "vllm-project/vllm"
    assert len(repos) == 2


def test_search_similar_empty_on_none():
    with patch("hot_projects.providers.github.provider.search_github_repos", return_value=None):
        p = GitHubProvider(token_mgr=object())
        assert p.search_similar("nope") == []


def test_repo_info_hit_and_miss():
    with patch("hot_projects.providers.github.provider.fetch_repo_info",
               return_value={"full_name": "a/b", "stargazers_count": 1500}):
        p = GitHubProvider(token_mgr=object())
        r = p.repo_info("a/b")
        assert r is not None and r.star == 1500
    with patch("hot_projects.providers.github.provider.fetch_repo_info", return_value=None):
        p = GitHubProvider(token_mgr=object())
        assert p.repo_info("a/b") is None
    # 非法格式
    assert GitHubProvider(token_mgr=object()).repo_info("nofmt") is None
