"""观测宇宙(`Github_DB.json`)—— 我们在盯着的全部仓库,以及每个仓库的元信息。

    {"projects": {"owner/repo": {star, created_at, forks, language,
                                 topics, gh_desc, readme_url,
                                 desc, desc_updated_at}}}

## 和旧结构的两处删减

旧结构还有顶层 `date` 和项目级 `refreshed_at`,它们的**全部**消费方都在「DB 差值」那条
增长回退路径上(`task_help._resolve_growth` 拿 `refreshed_at` 判窗口匹配、
`get_db_age_days` 拿 `date` 反推窗口)。增长改成快照相减之后那条路径整体删除,
这两个字段就没有读者了,所以不再写。

读取侧仍原样保留盘上出现的任何未知字段(见 `load`),所以旧记录里残留的 `refreshed_at`
不会被抹掉,新旧读取层也能逐字段比对 —— 这是 P1 对照脚本能做到「零差异」的前提。

## 字段归属:每个字段只有一个写入者

旧包只有一个 `save_db` 什么都能写,于是出过一次抹库:周报的候选池改成读快照后,
`repo_item` 里不再有 `forks_count`,而 `update_db_project` 无条件写 `forks = 0`,
一次运行把所有仓库的 fork 数清零。修法当时是在那一处加 `if forks is not None`,
但下一个无条件覆写还是照样能加进来。

这里改成**按意图开口子**:每个写入函数只认自己那一组字段,写别人的字段直接抛。
表在下面,并且下面这四个常量就是代码里的强制依据,不是注释。
"""

from __future__ import annotations

import logging

from ... import config
from .atomic import Tx, read_json, transaction

logger = logging.getLogger("hot_project")

# ── 字段归属:谁能写什么 ──
#
# 每日发现:仓库第一次进 DB 时带的两个字段。created_at 之后永不改写(它是判定新项目的依据,
# 改了就等于伪造仓库年龄);star 由下面的 refresh_stars 每天覆写。
DISCOVER_FIELDS = frozenset({"star", "created_at"})

# 展示字段:出榜时报告要用。一律**仅补空** —— 已有值不动,因为 GitHub 那边改简介/换语言时
# 我们没有必要跟着抖,而且这些值可能来自更完整的单仓库查询,不该被批量搜索的粗结果盖掉。
DISPLAY_FIELDS = frozenset({"forks", "language", "topics", "gh_desc", "readme_url"})

# LLM 描述:唯一允许反复覆写的一组(超过 DESC_REFRESH_DAYS 天会重新生成)。
DESC_FIELDS = frozenset({"desc", "desc_updated_at"})

# 每日快照:唯一每天覆写的字段。
STAR_FIELD = frozenset({"star"})

GH_DESC_MAX = 500   # gh_desc 截断长度,和旧包一致


def _empty() -> dict:
    return {"projects": {}}


def load() -> dict[str, dict]:
    """读全部仓库记录。**原样返回盘上的字段**,不做归一化、不丢未知键。

    文件不存在返回空字典;文件损坏抛 `StoreReadError`(由调用方决定停还是继续,
    但绝不会在这里被静默成空 DB —— 那是旧包出过的事故)。
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

    「不碰已有」不是优化而是正确性:发现阶段拿到的是搜索结果里的粗字段,
    顺手覆盖已有条目会用粗数据盖掉后来补的完整数据。想更新 star 请走 `refresh_stars`。

    返回名字而不只是条数,是因为「哪些是新的」只有这里知道(判重发生在事务内),
    而调用方要拿它给各个发现来源记账。在外面再读一次库来算差集,是白读 30MB。
    """
    if not records:
        return []
    _check_fields(records, DISCOVER_FIELDS, "每日发现")

    with transaction(config.DB_PATH, default=_empty()) as tx:
        projects = _open_projects(tx)
        inserted = []
        for name, info in records.items():
            if name in projects:
                continue
            projects[name] = dict(info)
            inserted.append(name)
        _commit(tx, len(inserted), "新增仓库")
        return inserted


def refresh_stars(stars: dict[str, int]) -> int:
    """每日快照:覆写已有仓库的 star。不认识的仓库跳过(要入库请走 `insert_discovered`)。

    分成两个函数而不是一次写完,是因为两者的语义正好相反:一个只准新增、一个只准更新。
    合成一个「upsert」就等于把 `insert_discovered` 那条「不碰已有」的保护取消掉了。
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

    传进来的每条只需给你手上有的字段,缺的不用占位 —— 「缺」和「空」在这里是两回事:
    缺 = 这次没查到,保持原值;空字符串 = 明确是空的,也保持原值(仅补空的语义)。
    正是「拿缺失当 0 写下去」造成过一次 fork 数全库清零。
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


def evict(names: set[str] | frozenset[str]) -> list[str]:
    """删除仓库,返回真正删掉的名字(已排序)。

    该删谁由 `core/evict.py` 的纯函数判定 —— 这里只执行。两条规则(GitHub 查不到、
    star 低于门槛)都只看**一次**快照就能定,所以没有宽限期、没有保护名单:
    仓库若改名后重新涨过门槛,下一次发现会照常把它收回来,描述也会重新生成。
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
