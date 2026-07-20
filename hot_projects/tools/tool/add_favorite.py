"""add_favorite 工具：与用户确认后，把某项目加入其收藏。

流程：resolve_repo 消歧 → DB 无此项目则实时拉 GitHub 信息补全并入库（字段与内部项目一致）
→ 生成一句话中文概要（GitHub 描述浓缩，无描述则留空）→ 写入该用户收藏。
未登录（无有效 user_id）时直接提示，不做任何写入。
"""

from ..basic.resolve import resolve_repo
from ...datasource.github.api import fetch_repo_info
from ...infra import favorites_store
from ...infra.db import save_db, update_db_project
from ...infra.llm import batch_condense_descriptions

_SHORT_DESC_MAX = 60


def _make_short_desc(repo: str, gh_desc: str) -> str:
    """GitHub 描述浓缩为不超过 _SHORT_DESC_MAX 字的中文概要；无描述则空。"""
    gh_desc = (gh_desc or "").strip()
    if not gh_desc:
        return ""
    condensed = batch_condense_descriptions(
        [{"full_name": repo, "description": gh_desc}], max_chars=_SHORT_DESC_MAX
    )
    return (condensed[0] if condensed else gh_desc)[:_SHORT_DESC_MAX]


def add_favorite_handler(ctx, args: dict) -> dict:
    user_id = getattr(ctx, "user_id", "") or ""
    if not favorites_store.valid_user_id(user_id):
        return {"error": "当前会话未登录，无法收藏。请在网页右上角登录后重试。"}

    repo, payload = resolve_repo(ctx, args.get("repo"))
    if payload is not None:
        return payload
    ctx.state.active_repo = repo

    db_projects = ctx.db.setdefault("projects", {})
    proj = db_projects.get(repo)
    if proj is None:
        owner, name = repo.split("/", 1)
        info = fetch_repo_info(ctx.provider.token_mgr, owner, name)
        if not info:
            return {"error": f"未找到仓库 {repo}，无法收藏。"}
        update_db_project(db_projects, repo, info.get("stargazers_count", 0), info)
        save_db(ctx.db)
        proj = db_projects.get(repo, {})

    short_desc = _make_short_desc(repo, proj.get("short_desc", ""))

    try:
        favorites_store.set_favorite(user_id, repo, "add", short_desc=short_desc)
    except ValueError as exc:
        return {"error": str(exc)}

    return {"ok": True, "repo": repo, "short_desc": short_desc,
            "message": f"已将 {repo} 加入你的收藏。"}
