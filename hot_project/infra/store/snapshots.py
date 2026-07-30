"""每日 star 快照 —— 一天一个 gzip 文件,是全部增长计算的唯一数据源。

    data/snapshots/2026-07-30.json.gz
    {"meta": {"coverage": 0.998, "not_found": ["owner/gone", ...]},
     "stars": {"owner/repo": 12345, ...}}

## 为什么增长只能靠它

GitHub 2026-06-30 起把 stargazers **列表**限权给仓库的 admin/collaborator,star 时间戳
(REST /stargazers、GraphQL stargazers.edges)对他人仓库全部失效,二分法和采样外推同时报废。
star **计数**(stargazerCount)不受影响,所以窗口增长改由「每天存一份计数」还原:

    增长 = 今天的 star − T−N 那天快照里的 star

## 为什么不并进 Github_DB.json

排名要的是「某一天 × 全部仓库」,正好是按天一个文件的形状 —— 读一个 0.8MB 的 gz 就够。
把 `{日期: star}` 挂到每个项目下是转置布局,为取 T−7 那一天得加载解析 35 天全量。
实测代价:主库 30MB→83MB、每次保存序列化 0.8s→2.7s(全程持排他锁),
而且 api_server 为读单个项目的 gh_desc 也要吞下整库。

## meta 是新加的

旧格式是扁平的 `{"owner/repo": star}`,读到的时候无法区分「这个仓库那天没测到」和
「那天它真的不在宇宙里」。`not_found` 记下 GitHub 明确查不到的名字(改名/删库/转私有),
淘汰判定直接用它;`coverage` 记下实际测到的比例,低于下限就拒绝落盘。
读取侧兼容旧扁平格式(见 `load_stars`),所以历史快照照样能当锚点。
"""

from __future__ import annotations

import gzip
import json
import logging
from datetime import date
from pathlib import Path
from typing import NamedTuple

from ... import config
from ...common.timeutil import days_between, format_day, parse_day, shift_days, utc_today
from .atomic import write_whole

logger = logging.getLogger("hot_project")

_SUFFIX = ".json.gz"

# 锚点日期与 T−N 的最大允许偏差(天)。每天都跑时恒为 0;漏跑一两天就顺延到邻近快照。
# 顺延不会歪曲排名:锚点一轮只挑一次、**全部仓库共用**,窗口长度一致,相对次序不受影响。
# 定在这里而不是 config:它是锚点选取规则的一部分,由本模块定义语义。
ANCHOR_TOLERANCE_DAYS = 2

# 覆盖率下限:实际测到的仓库数 ÷ 应测数。低于它拒绝落盘。
# 实测正常批次覆盖率 99.8%,掉到一半以下只可能是限流打崩了或 token 集体失效 ——
# 这种半份快照一旦落盘,会被后续几天当成锚点读走,把「没测到」算成「掉到 0」,
# 于是整批仓库出现巨额虚假负增长。宁可当天没有快照(锚点自动顺延一天)也不要错的。
MIN_COVERAGE = 0.5


def _filename(day: date) -> str:
    return f"{format_day(day)}{_SUFFIX}"


def path_of(day: date) -> Path:
    return config.SNAPSHOT_DIR / _filename(day)


def save(day: date, stars: dict[str, int], *, not_found: list[str],
         expected: int, throttle: dict | None = None) -> Path | None:
    """写当天快照。覆盖率不足则**不产生文件**并返回 None。

    Args:
        stars:    实际测到的 {full_name: star}
        not_found: GitHub 明确查不到的名字(改名/删库/转私有),供淘汰判定用
        expected: 本次应测的仓库总数,用来算覆盖率
        throttle: 这一轮撞了多少次限流、等了多久。存进去是因为几个月后看到一份
                  覆盖率偏低的快照时,「那天限流很重」和「代码有 bug」得能分开;
                  并跑期还要靠它认出「限流较重的那一天」。
    """
    if not stars:
        logger.error("拒绝写入空快照:它会被当成「全仓库掉到 0」,污染整个窗口的增长。")
        return None

    coverage = len(stars) / expected if expected > 0 else 0.0
    if coverage < MIN_COVERAGE:
        logger.error(
            "拒绝写入快照 %s:覆盖率 %.1f%%(%d/%d)低于下限 %.0f%%。"
            "半份快照会在之后几天被当成锚点,把「没测到」算成「掉到 0」。",
            day, coverage * 100, len(stars), expected, MIN_COVERAGE * 100,
        )
        return None

    meta: dict = {"coverage": round(coverage, 5), "not_found": sorted(not_found)}
    if throttle:
        meta["throttle"] = throttle
    payload = {"meta": meta, "stars": stars}

    def _write(tmp: Path) -> None:
        with gzip.open(tmp, "wt", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, separators=(",", ":"))

    path = path_of(day)
    write_whole(path, _write)
    logger.info("快照 %s 已写入:%d 个仓库,覆盖率 %.1f%%,查不到 %d 个。",
                day, len(stars), coverage * 100, len(not_found))
    return path


def _load_raw(day: date) -> dict | None:
    path = path_of(day)
    if not path.exists():
        return None
    try:
        with gzip.open(path, "rt", encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError, EOFError) as e:
        # 快照损坏按缺失处理(而不是抛):锚点是可替代的,顺延到邻近那天即可,
        # 不该让一个坏文件把整轮排名打断。DB 损坏则相反 —— 它无可替代,必须抛。
        logger.warning("快照 %s 读取失败,按缺失处理: %s", path, e)
        return None
    return data if isinstance(data, dict) and data else None


def load_stars(day: date) -> dict[str, int] | None:
    """读某天的 star 表。兼容旧扁平格式;缺失或损坏返回 None。"""
    data = _load_raw(day)
    if data is None:
        return None
    stars = data.get("stars") if "stars" in data else data   # 旧格式就是扁平表本身
    return stars if isinstance(stars, dict) and stars else None


def already_written(day: date) -> bool:
    """今天这份快照写过没有?每日脚本靠它做幂等。

    幂等是那个脚本敢每小时触发一次的全部依据(GitHub 的 schedule 会漂、会静默跳过,
    所以一天给自己 24 次机会),当天已有快照就秒退、一个请求都不发。
    """
    return path_of(day).exists()


def load_not_found(day: date) -> list[str]:
    """读某天 GitHub 查不到的名字。旧格式没有这项,返回空表。"""
    data = _load_raw(day)
    if data is None:
        return []
    meta = data.get("meta")
    names = meta.get("not_found") if isinstance(meta, dict) else None
    return names if isinstance(names, list) else []


def available_dates() -> list[date]:
    """已落盘的快照日期,升序。

    只扫**读目录**:锚点全都是历史数据,而历史数据的权威副本在读目录。
    过渡期写目录(影子)里只有今天那一份,不参与锚点选取 —— 每日脚本算增长用的是
    手上刚拿到的 star,不需要把自己刚写的文件再读回来。
    """
    days: list[date] = []
    directory = config.SNAPSHOT_DIR
    if not directory.is_dir():
        return days
    for path in directory.glob(f"*{_SUFFIX}"):
        day = parse_day(path.name[: -len(_SUFFIX)])
        if day is not None:
            days.append(day)
    return sorted(days)


class Anchor(NamedTuple):
    """某窗口的锚点:日期、star 表、以及它到今天的**实际**天数。

    `window_days` 必须和 `stars` 一起返回,不能让调用方拿「自己请求的窗口」去算:
    漏采时锚点会顺延一两天,实际窗口就比请求的长。旧包三个读取侧各自算这一步,
    其中爆发探针漏了修正 —— 3 天窗口拿到 5 天的增量却仍除以 3,速率虚高 67%、
    爆发加成误判。把天数绑在数据上是唯一能让「忘记修正」不再可能的形状。
    """
    day: date
    stars: dict[str, int]
    window_days: int


def _anchor_at(day: date) -> Anchor | None:
    stars = load_stars(day)
    return None if stars is None else Anchor(day, stars, days_between(utc_today(), day))


def anchor_for_window(days: int, tolerance: int = ANCHOR_TOLERANCE_DAYS) -> Anchor | None:
    """取 T−days 的锚点:离目标日最近、且偏差不超过 tolerance 的那份快照。

    并列时取较早那天,保证结果与文件枚举顺序无关。找不到返回 None ——
    要不要退到更短的窗口由调用方决定(见 `oldest_anchor`),这里不替它做主。
    """
    target = shift_days(utc_today(), -days)
    for day in sorted(
        (d for d in available_dates() if abs((d - target).days) <= tolerance),
        key=lambda d: (abs((d - target).days), d),
    ):
        anchor = _anchor_at(day)
        if anchor is not None:
            return anchor
    return None


def oldest_anchor(max_days: int) -> Anchor | None:
    """退化用:取现存最早、但不早于 max_days 天前的那份快照。

    用在快照还没攒够的时候 —— 请求 7 天窗口而手上只有 3 天,与其算不出增长(整批记未决、
    榜单一个候选都没有),不如按真实的 3 天窗口给出增长,`window_days` 会如实说明是 3 天。
    上限 max_days 是为了不让它一路退到 35 天前去。
    """
    floor = shift_days(utc_today(), -max_days)
    for day in available_dates():          # 升序,第一个合格的就是最早的
        if day < floor:
            continue
        anchor = _anchor_at(day)
        if anchor is not None and anchor.window_days > 0:
            return anchor
    return None


def prune(keep_days: int, today: date | None = None) -> list[date]:
    """删掉早于 today − keep_days 的快照,返回被删日期。

    按日期截断而非「保留最近 N 份」:漏跑几天时「最近 N 份」会一路留到 N+ 天前,
    按日期截断始终是真正的 keep_days 天视野。

    只认 `*.json.gz` 且日期能解析的文件 —— 同目录下的锁文件、半成品不该被顺手带走。
    """
    directory = config.SNAPSHOT_DIR
    if not directory.is_dir():
        return []
    cutoff = shift_days(today or utc_today(), -keep_days)
    removed: list[date] = []
    for path in sorted(directory.glob(f"*{_SUFFIX}")):
        day = parse_day(path.name[: -len(_SUFFIX)])
        if day is None or day >= cutoff:
            continue
        try:
            path.unlink()
            removed.append(day)
        except OSError as e:
            logger.warning("快照 %s 删除失败: %s", day, e)
    return removed
