"""get_db_info 工具：查本地 DB 概览或指定仓库缓存信息（不联网）。"""

from ..basic import get_db_info as _get_db_info


def get_db_info_handler(ctx, args: dict) -> dict:
    return _get_db_info(db=ctx.db, repo=args.get("repo"))
