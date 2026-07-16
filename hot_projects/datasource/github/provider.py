"""GitHubProvider：把 GitHub 下层实现封装为统一 Provider 接口。

本类是 GitHub 专属细节的边界：编排层只通过 Provider 接口与 Repo 交互。
search_by_keywords / scan_star_range / repo_growth / batch_growth / fetch_trending
接线到 tools/basic 纯函数。
"""

from ..base import Provider, Repo
from .api import search_github_repos, fetch_repo_info
from ...tools.basic import (
    search_by_keywords as _search_by_keywords,
    scan_star_range as _scan_star_range,
    fetch_trending as _fetch_trending,
    check_repo_growth as _check_repo_growth,
    batch_check_growth as _batch_check_growth,
)


class GitHubProvider(Provider):
    def __init__(self, token_mgr):
        self.token_mgr = token_mgr

    def search_similar(self, name: str, limit: int = 5) -> list[Repo]:
        items = search_github_repos(
            self.token_mgr, name, token_idx=0, page=1, per_page=limit, min_star=0
        ) or []
        return [Repo.from_github(it) for it in items[:limit]]

    def search_top_repos(self, query: str, top_n: int = 5, min_star: int = 0) -> list[Repo]:
        items = search_github_repos(
            self.token_mgr, query, token_idx=0, page=1,
            per_page=min(max(top_n, 1), 50), sort="stars", order="desc", min_star=min_star,
        ) or []
        return [Repo.from_github(it) for it in items[:top_n]]

    def repo_info(self, repo: str) -> Repo | None:
        parts = repo.split("/", 1)
        if len(parts) != 2:
            return None
        item = fetch_repo_info(self.token_mgr, parts[0], parts[1], token_idx=0)
        return Repo.from_github(item) if item else None

    def search_by_keywords(self, categories, min_star, days_since_created, keywords=None) -> dict:
        return _search_by_keywords(
            self.token_mgr, categories=categories, min_star=min_star,
            days_since_created=days_since_created, keywords=keywords,
        )

    def scan_star_range(self, min_star, max_star, seen_repos, days_since_created) -> dict:
        return _scan_star_range(
            self.token_mgr, min_star=min_star, max_star=max_star,
            seen_repos=seen_repos, days_since_created=days_since_created,
        )

    def repo_growth(self, repo: str, growth_calc_days: int) -> dict:
        return _check_repo_growth(
            self.token_mgr, repo=repo, db=None, growth_calc_days=growth_calc_days,
        )

    def batch_growth(self, repos, db, **kwargs) -> dict:
        return _batch_check_growth(self.token_mgr, repos, db, **kwargs)

    def fetch_trending(self, trending_range: str) -> dict:
        return _fetch_trending(trending_range=trending_range)
