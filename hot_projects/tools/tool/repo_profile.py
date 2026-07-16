"""repo_profile 工具：单仓库画像取证（功能/场景 + 维护活跃度，一次返回）。

只取原始证据、不做 LLM 归纳——品类判断、优缺点、场景覆盖等由主模型基于返回内容自行提炼。
典型用途：了解"这项目能干什么/还在维护吗"、以及同类项目对比（对每个项目各调一次）。

Token 预算：结果会被 Agent 截断到 8000 字符，故 README 摘录上限 5000、近期提交 5 条、
目录线索仅在 README 简陋时附带，保证正常不触发截断。
"""

from ..basic.resolve import resolve_repo
from ...datasource.github.api import (
    fetch_repo_info,
    fetch_repo_readme_excerpt,
    fetch_repo_recent_commits,
    fetch_repo_recent_releases,
    fetch_repo_tree_paths,
)

_README_MAX_CHARS = 5000
# README 短于该值视为"简陋"，追加目录文件名线索补充功能覆盖信息
_README_THIN_CHARS = 1000


def _profile(repo: str, token_mgr) -> dict:
    owner, name = repo.split("/", 1)
    info = fetch_repo_info(token_mgr, owner, name) or {}
    readme = fetch_repo_readme_excerpt(token_mgr, owner, name, max_chars=_README_MAX_CHARS)
    readme_text = readme.get("text", "")
    license_info = info.get("license") or {}

    out = {
        "repo": repo,
        "html_url": info.get("html_url") or f"https://github.com/{repo}",
        "description": info.get("description") or "",
        "language": info.get("language") or "",
        "topics": info.get("topics") or [],
        "star": info.get("stargazers_count", 0),
        "forks": info.get("forks_count", 0),
        "open_issues": info.get("open_issues_count", 0),
        "created_at": info.get("created_at") or "",
        "pushed_at": info.get("pushed_at") or "",
        "license": license_info.get("spdx_id") or "",
        "archived": bool(info.get("archived", False)),
        "readme_excerpt": readme_text,
        "readme_truncated": bool(readme.get("truncated", False)),
        # 维护活跃信号
        "recent_releases": fetch_repo_recent_releases(token_mgr, owner, name),
        "recent_commits": fetch_repo_recent_commits(token_mgr, owner, name, per_page=5),
    }

    if len(readme_text) < _README_THIN_CHARS:
        # README 信息不足：目录文件名（docs/、examples/ 等）本身就是功能清单线索
        out["structure_hint"] = fetch_repo_tree_paths(token_mgr, owner, name)
        out["structure_hint_note"] = "README 内容较少，附目录文件清单作为功能覆盖线索。"
    return out


def repo_profile_handler(ctx, args: dict) -> dict:
    repo, payload = resolve_repo(ctx, args.get("repo"))
    if payload is not None:
        return payload
    ctx.state.active_repo = repo
    return _profile(repo, ctx.provider.token_mgr)
