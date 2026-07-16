"""describe_project 工具：LLM 生成单个仓库的中文功能介绍。"""

from ..basic import describe_project as _describe_project
from ..basic.resolve import resolve_repo


def describe_project_handler(ctx, args: dict) -> dict:
    repo, payload = resolve_repo(ctx, args.get("repo"))
    if payload is not None:
        return payload
    ctx.state.active_repo = repo
    return _describe_project(repo=repo, db=ctx.db, token_mgr=ctx.provider.token_mgr)
