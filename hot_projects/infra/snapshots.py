"""每日 star 快照：一天一个 gzip 文件，内容为 {"owner/repo": star}。

GitHub 2026-06-30 起把 stargazers *列表* 限权给仓库的 admin/collaborator，star 时间戳
（REST /stargazers、GraphQL stargazers.edges）对他人仓库全部失效，二分法与采样外推同时报废。
star *计数*（stargazerCount）不受影响，所以窗口增长改由「每天存一份计数」还原：
增长 = 当前 star − T−N 那天快照里的 star。

为什么不并入 Github_DB.json 的项目记录：
  排名需要的是「某一天 × 全部仓库」，正好是按天文件的形状——读一个 0.8MB 的 gz 就够。
  把 {日期: star} 挂到每个项目下是转置布局，为取 T−7 那一天得加载解析 35 天全量。
  实测代价：主库 30MB→83MB、每次 save 序列化 0.8s→2.7s（save_db 全程持排他锁），
  且 api_server 为读单个项目的 gh_desc 也要吞下整库。
"""

import glob
import gzip
import json
import logging
import os
from datetime import date, datetime, timedelta, timezone
from typing import NamedTuple

from ..config import DATA_DIR

logger = logging.getLogger("hot_projects")

SNAPSHOT_DIR = os.path.join(DATA_DIR, "snapshots")
_DATE_FMT = "%Y-%m-%d"
_SUFFIX = ".json.gz"

# 锚点日期与 T−N 的最大允许偏差（天），find_anchor 的默认容差。
# 每天都跑时恒为 0；漏跑一两天就顺延到邻近快照，且全部仓库共用同一锚点，
# 窗口长度一致，相对排名不受影响。定在这里而不是 config：它是锚点选取规则的一部分，
# 由本模块的 find_anchor 定义语义，三个读取侧调用方（排名、探针、单仓库工具）共用。
SNAPSHOT_ANCHOR_TOLERANCE_DAYS = 2


def utc_today() -> date:
    """UTC 日期。快照日期必须与 DB 的 refreshed_at（UTC）同基准，否则窗口会差一天。"""
    return datetime.now(timezone.utc).date()


def snapshot_path(day: date) -> str:
    return os.path.join(SNAPSHOT_DIR, f"{day.strftime(_DATE_FMT)}{_SUFFIX}")


def save_snapshot(day: date, stars: dict[str, int]) -> str:
    """原子写入当天快照（先 .tmp 再 os.replace，避免半截文件被当成锚点读走）。"""
    if not stars:
        raise ValueError("拒绝写入空快照：它会被当成「全仓库掉到 0」，污染整个窗口的增长。")
    os.makedirs(SNAPSHOT_DIR, exist_ok=True)
    path = snapshot_path(day)
    tmp = path + ".tmp"
    with gzip.open(tmp, "wt", encoding="utf-8") as f:
        json.dump(stars, f, ensure_ascii=False, separators=(",", ":"))
    os.replace(tmp, path)
    return path


def load_snapshot(day: date) -> dict[str, int] | None:
    """读取某天快照，缺失或损坏都返回 None（交由调用方另选锚点）。"""
    path = snapshot_path(day)
    if not os.path.exists(path):
        return None
    try:
        with gzip.open(path, "rt", encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError, EOFError) as e:
        logger.warning("快照 %s 读取失败，按缺失处理: %s", path, e)
        return None
    return data if isinstance(data, dict) and data else None


def available_dates() -> list[date]:
    """已落盘的快照日期，升序。"""
    days: list[date] = []
    for path in glob.glob(os.path.join(SNAPSHOT_DIR, f"*{_SUFFIX}")):
        stem = os.path.basename(path)[: -len(_SUFFIX)]
        try:
            days.append(datetime.strptime(stem, _DATE_FMT).date())
        except ValueError:
            continue
    return sorted(days)


def find_anchor(target: date, tolerance_days: int) -> tuple[date, dict[str, int]] | None:
    """取离 target 最近、且相差不超过 tolerance_days 的可用快照。

    锚点一轮只挑一次、全部仓库共用：即使实际日期偏离 target 一两天，所有仓库的窗口长度
    也完全一致，相对排名不受影响——这正是它比逐仓库 refreshed_at 更稳的地方（后者每个
    仓库窗口各不相同，只能靠折算近似）。
    并列时取较早那天，保证结果与文件枚举顺序无关。
    """
    for day in sorted(
        (d for d in available_dates() if abs((d - target).days) <= tolerance_days),
        key=lambda d: (abs((d - target).days), d),
    ):
        stars = load_snapshot(day)
        if stars:
            return day, stars
    return None


class Anchor(NamedTuple):
    """某窗口的锚点：日期、star 表、以及它到今天的**实际**天数。

    window_days 必须和 stars 一起返回、不能让调用方按「自己请求的窗口」去算：
    漏采时 find_anchor 会顺延一两天，实际窗口就比请求的长。三个读取侧
    （排名主窗口、爆发探针、单仓库工具）本来各自算这一步，其中爆发探针漏了修正——
    3 天窗口拿到 5 天的增量却仍除以 3，速率虚高 67%、爆发加成误判。
    把天数绑在数据上是唯一能让"忘记修正"不再可能的形状。
    """
    day: date
    stars: dict[str, int]
    window_days: int


def anchor_for_window(
    days: int, tolerance_days: int = SNAPSHOT_ANCHOR_TOLERANCE_DAYS
) -> Anchor | None:
    """取 T−days 的锚点快照。没有可用快照返回 None（调用方各自决定怎么退化）。"""
    found = find_anchor(utc_today() - timedelta(days=days), tolerance_days)
    if found is None:
        return None
    day, stars = found
    return Anchor(day, stars, (utc_today() - day).days)


def prune_snapshots(keep_days: int, today: date | None = None) -> list[date]:
    """删掉早于 today − keep_days 的快照，返回被删日期。

    按日期截断而非「保留最近 N 份」：漏跑几天时「最近 N 份」会一路留到 N+ 天前，
    按日期截断始终是真正的 keep_days 天视野。
    """
    cutoff = (today or utc_today()) - timedelta(days=keep_days)
    removed: list[date] = []
    for day in available_dates():
        if day >= cutoff:
            continue
        try:
            os.remove(snapshot_path(day))
            removed.append(day)
        except OSError as e:
            logger.warning("快照 %s 删除失败: %s", day, e)
    return removed
