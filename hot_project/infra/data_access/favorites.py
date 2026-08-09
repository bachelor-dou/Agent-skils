"""用户收藏 —— 按 user_id 存清单。

    data/favorites.json
    {"users": {"<user_id>": [{"repo": "owner/name",
                              "favorited_at": "YYYY-MM-DDTHH:MM:SSZ",
                              "source_report": "2026-07-01.md",
                              "short_desc": "一句话概要",
                              "category": "效率",
                              "subcategory": "文档"}]}}

分类是两级的:`subcategory` 只在 `category` 之下有意义,离开父分类就没有归属,所以
`category` 被清空时 `subcategory` 一并清掉(见 `set_favorite`)。

收藏是**全局**的:某项目一旦被收藏,在任何包含它的报告里都显示为已收藏。

整个读-改-写必须在同一把排他锁里(`transaction`):分成「读完放锁 → 改 → 再拿锁写」的话,
两个并发收藏请求会各读到同一份旧数据,后写的抹掉前一个。

收藏**不是淘汰的保护名单**(淘汰只看 GitHub 查不到、star 低于门槛),但收藏记录本身从不
因淘汰而删除,所以用户的收藏不会丢。
"""

from __future__ import annotations

import re

from ... import config
from ...common.timeutil import stamp
from ._file_io import read_json, transaction

USER_ID_RE = re.compile(r"^[A-Za-z0-9_-]{3,32}$")
REPO_RE = re.compile(r"^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$")
_CTRL_RE = re.compile(r"[\x00-\x1f\x7f]")

MAX_PER_USER = 500
MAX_CATEGORY_LEN = 20


def valid_user_id(user_id: str) -> bool:
    return bool(user_id and USER_ID_RE.match(user_id))


def valid_repo(repo: str) -> bool:
    return bool(repo and REPO_RE.match(repo))


def clean_category(category: str) -> str:
    """规整分类标签:剔除控制字符、折叠空白、截断长度;空则返回 ''(未分类)。

    控制字符换成空格再折叠,不是直接删 —— 直接删会把 "a\\x00b" 粘成 "ab"。
    """
    if not category:
        return ""
    return " ".join(_CTRL_RE.sub(" ", str(category)).split())[:MAX_CATEGORY_LEN]


def _empty() -> dict:
    return {"users": {}}


def _users_of(data: dict) -> dict:
    users = data.get("users")
    return users if isinstance(users, dict) else {}


def get(user_id: str) -> list[dict]:
    """该用户的收藏清单(新收藏在前)。非法 user_id 返回空表。"""
    if not valid_user_id(user_id):
        return []
    data = read_json(config.FAVORITES_PATH, default=_empty())
    items = _users_of(data).get(user_id, [])
    return items if isinstance(items, list) else []


def all_repos() -> set[str]:
    """所有用户收藏过的仓库全名,跨用户合并。"""
    data = read_json(config.FAVORITES_PATH, default=_empty())
    return {
        item["repo"]
        for items in _users_of(data).values() if isinstance(items, list)
        for item in items if isinstance(item, dict) and item.get("repo")
    }


def set_favorite(user_id: str, repo: str, action: str, *,
                 source_report: str = "", short_desc: str | None = None,
                 category: str | None = None,
                 subcategory: str | None = None) -> list[dict]:
    """add / remove 单个收藏,返回更新后的清单。非法输入抛 `ValueError`。

    `short_desc`、`category`、`subcategory` 同语义:`None` = 不改动(新增时存空串),
    字符串(含 `""`)= 覆盖。父分类被清空时子分类跟着清空。
    """
    if not valid_user_id(user_id):
        raise ValueError("invalid user_id")
    if not valid_repo(repo):
        raise ValueError("invalid repo")
    if action not in ("add", "remove"):
        raise ValueError("invalid action")

    with transaction(config.FAVORITES_PATH, default=_empty()) as tx:
        if not isinstance(tx.data, dict):
            tx.data = _empty()
        users = tx.data.setdefault("users", {})
        items = [x for x in users.get(user_id, []) if isinstance(x, dict)]

        if action == "remove":
            items = [x for x in items if x.get("repo") != repo]
        else:
            existing = next((x for x in items if x.get("repo") == repo), None)
            if existing is not None:          # 幂等:重复 add 只补概要/分类
                if short_desc is not None:
                    existing["short_desc"] = short_desc
                if category is not None:
                    existing["category"] = clean_category(category)
                if subcategory is not None:
                    existing["subcategory"] = clean_category(subcategory)
                if not existing.get("category"):
                    existing["subcategory"] = ""
            else:
                if len(items) >= MAX_PER_USER:
                    raise ValueError("favorites limit reached")
                cat = clean_category(category or "")
                items.insert(0, {
                    "repo": repo,
                    "favorited_at": stamp(),
                    "source_report": source_report or "",
                    "short_desc": short_desc or "",
                    "category": cat,
                    "subcategory": clean_category(subcategory or "") if cat else "",
                })

        users[user_id] = items
        return items
