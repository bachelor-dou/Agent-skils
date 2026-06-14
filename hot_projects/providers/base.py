"""数据源 Provider 接口 + 归一化 Repo 模型（多平台边界）。

编排层（capabilities/pipeline/agent/tools）只依赖本模块的 Provider 接口与 Repo 模型，
平台细节（GitHub REST/GraphQL/stargazers/trending/token 限流）封装在各 Provider 实现内。
未来新增平台 = 新增一个 Provider 实现，编排层零改动。
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field


@dataclass
class Repo:
    """跨平台归一化仓库模型。"""

    full_name: str
    star: int
    description: str = ""
    language: str = ""
    topics: list[str] = field(default_factory=list)
    created_at: str = ""
    forks: int = 0
    raw: dict = field(default_factory=dict)

    @classmethod
    def from_github(cls, item: dict) -> "Repo":
        """从 GitHub API item（或内部 repo dict）构造归一化 Repo。"""
        return cls(
            full_name=item.get("full_name", ""),
            star=item.get("stargazers_count", item.get("star", 0)),
            description=(item.get("description") or "")[:500],
            language=item.get("language") or "",
            topics=item.get("topics") or [],
            created_at=item.get("created_at", "") or "",
            forks=item.get("forks_count", item.get("forks", 0)),
            raw=item,
        )


class Provider(ABC):
    """平台数据源接口。编排层只依赖本接口与 Repo。"""

    @abstractmethod
    def search_by_keywords(self, categories, min_star, days_since_created, keywords=None) -> dict:
        """按关键词搜索（预设类别 + LLM 补充的 keywords 取并集）；返回含 `_raw_repos` 的结果 dict。"""

    @abstractmethod
    def scan_star_range(self, min_star, max_star, seen_repos, days_since_created) -> dict:
        """按 star 区间扫描；返回含 `_raw_repos` 的结果 dict。"""

    @abstractmethod
    def repo_info(self, repo: str) -> "Repo | None":
        """精确获取单仓库信息；不存在返回 None。"""

    @abstractmethod
    def repo_growth(self, repo: str, growth_calc_days: int) -> dict:
        """单仓库近期增长。"""

    @abstractmethod
    def batch_growth(self, repos, db, **kwargs) -> dict:
        """批量增长计算并筛候选。"""

    @abstractmethod
    def fetch_trending(self, trending_range: str) -> dict:
        """获取 Trending。"""

    @abstractmethod
    def search_similar(self, name: str, limit: int = 5) -> "list[Repo]":
        """模糊搜索相似仓库（单仓库消歧用）。"""
