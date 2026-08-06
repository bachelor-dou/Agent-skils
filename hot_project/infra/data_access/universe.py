"""观测宇宙(`Github_DB.json`)—— 我们在盯着的全部仓库,以及每个仓库的元信息。

    {"projects": {"owner/repo": {star, created_at, language,
                                 topics, gh_desc, desc, desc_updated_at}}}

读取侧原样保留盘上出现的任何未知字段(见 `load`),旧记录里残留的字段不会被抹掉。

写入侧**按意图开口子**:每个写入函数只认自己那一组字段,写别人的字段直接抛。下面四个
常量就是代码里的强制依据,不是注释 —— 一个无条件覆写就足以把全库某个字段清零。
"""

from __future__ import annotations

import logging

from ... import config
from ._file_io import Tx, read_json, transaction

logger = logging.getLogger("hot_project")

# ── 字段归属 ──
DISCOVER_FIELDS = frozenset({"star", "created_at", "id"})

DISPLAY_FIELDS = frozenset({"language", "topics", "gh_desc"})

DESC_FIELDS = frozenset({"desc", "desc_updated_at"})

STAR_FIELD = frozenset({"star"})

ID_FIELD = frozenset({"id"})

GH_DESC_MAX = 500   # gh_desc 截断长度


def _empty() -> dict:
    return {"projects": {}}


def load() -> dict[str, dict]:
    """读全部仓库记录。**原样返回盘上的字段**,不做归一化、不丢未知键。

    文件不存在返回空字典;文件损坏抛 `StoreReadError`,绝不在这里被静默成空 DB。
    """
    db = read_json(config.DB_PATH, default=_empty())
    projects = db.get("projects")
    return projects if isinstance(projects, dict) else {}


def _check_fields(records: dict[str, dict], allowed: frozenset[str], who: str) -> None:
    for name, info in records.items():
        if not isinstance(info, dict):
            raise TypeError(f"{who}: {name} 的记录不是 dict")
        extra = set(info) - allowed
        if extra:
            raise ValueError(
                f"{who} 不许写字段 {sorted(extra)}({name})—— "
                f"它只拥有 {sorted(allowed)}。要新增字段请先在字段归属表里给它找个主人。"
            )


def _open_projects(tx: Tx) -> dict[str, dict]:
    if not isinstance(tx.data, dict):
        raise TypeError("DB 根不是 dict")
    projects = tx.data.setdefault("projects", {})
    if not isinstance(projects, dict):
        raise TypeError("DB projects 不是 dict")
    return projects


def _commit(tx: Tx, changed: int, what: str) -> int:
    if not changed:
        tx.abort()
        return 0
    logger.info("DB %s:%d 条。", what, changed)
    return changed


def insert_discovered(records: dict[str, dict]) -> list[str]:
    """每日发现:把还不在 DB 里的仓库插进来,**已有条目一律不碰**。返回真正插进去的名字。

    「不碰已有」是正确性:发现阶段拿的是搜索粗字段,覆盖会盖掉后来补的完整数据。
    更新 star 走 `refresh_stars`。判重在事务内,所以「哪些是新的」只有这里知道。
    判重先看名字、再看 `id`:同一个 id 换了名字 → 改挂新名,不插重复。
    """
    if not records:
        return []
    _check_fields(records, DISCOVER_FIELDS, "每日发现")

    with transaction(config.DB_PATH, default=_empty()) as tx:
        projects = _open_projects(tx)
        known_ids = {info["id"]: name for name, info in projects.items()
                     if isinstance(info, dict) and isinstance(info.get("id"), int)}
        inserted, renamed = [], 0
        for name, info in records.items():
            if name in projects:
                continue
            rid = info.get("id")
            old = known_ids.get(rid)
            if old is not None:
                projects[name] = projects.pop(old)
                projects[name]["id"] = rid
                known_ids[rid] = name
                renamed += 1
                continue
            projects[name] = dict(info)
            if isinstance(rid, int):
                known_ids[rid] = name
            inserted.append(name)
        if renamed:
            logger.info("DB 发现阶段按 id 识别改名:%d 条已改挂新名。", renamed)
        _commit(tx, len(inserted) + renamed, "新增仓库")
        return inserted


def refresh_stars(stars: dict[str, int]) -> int:
    """覆写已有仓库的 star。不认识的仓库跳过(要入库请走 `insert_discovered`)。

    **目前没有生产调用方** —— 当前 star 由快照负责。和 `insert_discovered` 分开是因为语义
    相反,合成 upsert 就等于取消了那条「不碰已有」的保护。
    """
    if not stars:
        return 0

    with transaction(config.DB_PATH, default=_empty()) as tx:
        projects = _open_projects(tx)
        updated = 0
        for name, star in stars.items():
            info = projects.get(name)
            if not isinstance(info, dict) or info.get("star") == star:
                continue
            info["star"] = star
            updated += 1
        return _commit(tx, updated, "star 刷新")


def refresh_display(items: dict[str, dict]) -> int:
    """出榜前补展示字段,**仅补空**:已有值一概不动。

    缺的字段不用占位 —— 缺 = 这次没查到,空串 = 明确是空的。「拿缺失当 0 写」曾把全库 fork 数清零。
    """
    if not items:
        return 0
    _check_fields(items, DISPLAY_FIELDS, "展示字段刷新")

    with transaction(config.DB_PATH, default=_empty()) as tx:
        projects = _open_projects(tx)
        filled = 0
        for name, info in items.items():
            target = projects.get(name)
            if not isinstance(target, dict):
                continue
            touched = False
            for key, value in info.items():
                if not value or target.get(key):
                    continue
                target[key] = value[:GH_DESC_MAX] if key == "gh_desc" else value
                touched = True
            filled += touched
        return _commit(tx, filled, "展示字段补空")


def write_descriptions(descs: dict[str, dict]) -> int:
    """写 LLM 描述。仓库不在 DB 里就跳过 —— 描述是给榜单用的,而榜单候选只出自 DB。"""
    if not descs:
        return 0
    _check_fields(descs, DESC_FIELDS, "描述写入")

    with transaction(config.DB_PATH, default=_empty()) as tx:
        projects = _open_projects(tx)
        changed = 0
        for name, info in descs.items():
            target = projects.get(name)
            if not isinstance(target, dict) or not info.get("desc"):
                continue
            if all(target.get(k) == v for k, v in info.items()):
                continue
            target.update(info)
            changed += 1
        return _commit(tx, changed, "描述更新")


def set_ids(ids: dict[str, int]) -> int:
    """给已有仓库写入 GitHub databaseId。不认识的名字跳过,已一致的不算变更。"""
    if not ids:
        return 0

    with transaction(config.DB_PATH, default=_empty()) as tx:
        projects = _open_projects(tx)
        updated = 0
        for name, rid in ids.items():
            info = projects.get(name)
            if (not isinstance(info, dict) or not isinstance(rid, int)
                    or info.get("id") == rid):
                continue
            info["id"] = rid
            updated += 1
        return _commit(tx, updated, "databaseId 回写")


def apply_renames(renames: dict[str, tuple[str, int | None]]) -> int:
    """按改名表把 DB 收敛到规范名:旧名记录并入新名后删除旧名。返回变更条数。

    - 新名已在库 → 以新名为主,旧名里有、新名缺的字段补过去(别丢之前生成的 LLM 描述),删旧名。
    - 新名不在库 → 直接把旧名记录改挂到新名。
    历史快照一律不动,基线按 databaseId 对齐。
    """
    if not renames:
        return 0

    with transaction(config.DB_PATH, default=_empty()) as tx:
        projects = _open_projects(tx)
        changed = 0
        for old, (new, rid) in renames.items():
            if new == old or old not in projects:
                continue
            src = projects.pop(old)
            dst = projects.get(new)
            if isinstance(dst, dict):
                for key, value in src.items():      # 新名缺的字段才补,不覆盖已有值
                    dst.setdefault(key, value)
            else:
                projects[new] = src
            if isinstance(rid, int):
                projects[new]["id"] = rid
            changed += 1
        return _commit(tx, changed, "改名归并(旧名→规范新名)")


def evict(names: set[str] | frozenset[str]) -> list[str]:
    """删除仓库,返回真正删掉的名字(已排序)。

    该删谁由 `core/evict.py` 判定,这里只执行。没有宽限期和保护名单:重新涨过门槛会被再收回来。
    """
    if not names:
        return []

    with transaction(config.DB_PATH, default=_empty()) as tx:
        projects = _open_projects(tx)
        removed = sorted(n for n in names if n in projects)
        for name in removed:
            del projects[name]
        _commit(tx, len(removed), "淘汰")
        return removed
