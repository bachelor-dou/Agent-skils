"""每日 star 快照 —— 一天一个 gzip 文件,是全部增长计算的唯一数据源。

    data/snapshots/2026-07-30.json.gz
    {"meta": {"coverage": 0.998, "not_found": ["owner/gone", ...]},
     "stars": {"owner/repo": 12345, ...}}

GitHub 已把 star 时间戳限权给仓库 admin,二分法和采样外推都报废,窗口增长只能靠
「实时 star − 窗口内最早那份快照里的 star」。快照只出**基线**这一件事,当前值一律实时取
(见 `tools/ranking.py`)。`not_found` 是 GitHub 明确查不到的名字
(改名/删库/转私有),淘汰判定直接用它;读取侧兼容没有 meta 的旧扁平格式(见 `load_stars`)。
"""

from __future__ import annotations

import gzip
import json
import logging
import zlib
from datetime import date
from pathlib import Path
from typing import NamedTuple

from ... import config
from ...common.timeutil import days_between, format_day, parse_day, shift_days, utc_today
from ._file_io import write_whole

logger = logging.getLogger("hot_project")

_SUFFIX = ".json.gz"

MIN_COVERAGE = 0.5


def _filename(day: date) -> str:
    return f"{format_day(day)}{_SUFFIX}"


def path_of(day: date) -> Path:
    return config.SNAPSHOT_DIR / _filename(day)


def save(day: date, stars: dict[str, int], *, not_found: list[str],
         expected: int, throttle: dict | None = None) -> Path | None:
    """写当天快照。覆盖率不足则**不产生文件**并返回 None。

    `not_found` 是 GitHub 明确查不到的名字(改名/删库/转私有),供淘汰判定;`expected` 是
    本次应测总数,用来算覆盖率;`throttle` 记这轮限流次数与等待,事后好和代码 bug 区分。
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

    if (existing := _coverage_of(day)) is not None and existing > coverage:
        logger.warning(
            "放弃写入快照 %s:盘上那份覆盖率 %.1f%% 比这次的 %.1f%% 更高,保留原文件。",
            day, existing * 100, coverage * 100,
        )
        return path_of(day)

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
    except (OSError, zlib.error, json.JSONDecodeError, EOFError, UnicodeDecodeError) as e:
        logger.warning("快照 %s 读取失败,按缺失处理: %s", path, e)
        return None
    return data if isinstance(data, dict) and data else None


def _coverage_of(day: date) -> float | None:
    """盘上那份快照的覆盖率。文件不存在、读不出、或是没有 meta 的旧扁平格式都返回 None。"""
    data = _load_raw(day)
    meta = data.get("meta") if isinstance(data, dict) else None
    value = meta.get("coverage") if isinstance(meta, dict) else None
    return float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else None


def load_stars(day: date) -> dict[str, int] | None:
    """读某天的 star 表。兼容旧扁平格式;缺失或损坏返回 None。"""
    data = _load_raw(day)
    if data is None:
        return None
    stars = data.get("stars") if "stars" in data else data   # 旧格式就是扁平表本身
    return stars if isinstance(stars, dict) and stars else None


def already_written(day: date) -> bool:
    """今天这份快照写过没有?每日脚本靠它做幂等 —— GitHub 的 schedule 会漂、会静默跳过,
    所以那个脚本每小时触发一次,当天已有快照就秒退、一个请求都不发。
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
    """已落盘的快照日期,升序。"""
    days: list[date] = []
    directory = config.SNAPSHOT_DIR
    if not directory.is_dir():
        return days
    for path in directory.glob(f"*{_SUFFIX}"):
        day = parse_day(path.name[: -len(_SUFFIX)])
        if day is not None:
            days.append(day)
    return sorted(days)


class Baseline(NamedTuple):
    """窗口内每个仓库**最早**被测到的 star,以及那天到今天的实际天数。

    `days` 必须逐仓给:拿全局天数去除会让晚进库的仓库日均速率虚高一倍多。
    `span` 是最早那份快照的跨度,用作全轮名义窗口(报告标题、判「窗口内新建」)。
    """

    stars: dict[str, int]
    days: dict[str, int]
    oldest: date | None
    span: int


def earliest_in_window(days: int, today: date | None = None) -> Baseline:
    """一趟扫出窗口内每个仓库最早被测到的 star。

    取「最早的一份」而不是「正好 T−N 那天」:晚进库的仓库按它算出的是**实测下界**,比整个
    丢掉强,漏采几天时 `span` 也会如实说明实际跨度。今天那份不作基线(窗口 0 天,增长恒为 0)。
    """
    now = today or utc_today()
    floor = shift_days(now, -days)
    stars: dict[str, int] = {}
    spans: dict[str, int] = {}
    oldest: date | None = None

    for day in available_dates():           # 升序,所以先落进表里的就是最早的那次
        span = days_between(now, day)
        if span < 1 or day < floor:
            continue
        snapshot = load_stars(day)
        if snapshot is None:
            continue
        if oldest is None:
            oldest = day
        for name, star in snapshot.items():
            if name not in stars:
                stars[name] = star
                spans[name] = span

    return Baseline(stars, spans, oldest,
                    days_between(now, oldest) if oldest is not None else days)


def prune(keep_days: int, today: date | None = None) -> list[date]:
    """删掉早于 today − keep_days 的快照,返回被删日期。

    按日期截断而非「保留最近 N 份」:漏跑几天时后者会一路留到 N+ 天前。只认 `*.json.gz`,
    锁文件和半成品不动。`keep_days < 1` 拒绝:0 会把今天在内的快照删光,而快照重算不回来。
    """
    if keep_days < 1:
        raise ValueError(f"keep_days 至少为 1,收到 {keep_days} —— 这会删光全部快照")
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
