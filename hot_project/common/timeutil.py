"""时间 —— 全项目只有这一处知道时间长什么样。

项目里有两种时间字符串,它们不能混用:

    GitHub 时间戳   "2026-07-30T08:15:00Z"   仓库的 created_at、收藏时刻
    日期串          "2026-07-30"             快照文件名、描述刷新日期、报告文件名

**一律 UTC。** 混用本地时区会让窗口差一天:快照按 UTC 日期命名,而「今天」若按本地算,
东八区下午跑的任务会去找一个还不存在的锚点。旧包在 `snapshots.utc_today()` 的注释里
专门叮嘱过这件事,但每个用到时间的模块仍各写一遍 —— 现在没有第二处可写。

解析失败一律返回 None 而不是抛:这些串大多来自 GitHub 或历史数据文件,
格式意外时该由调用方决定是跳过还是回退,不该让一条脏数据打断整轮。
"""

from __future__ import annotations

from datetime import date, datetime, timedelta, timezone

STAMP_FMT = "%Y-%m-%dT%H:%M:%SZ"    # GitHub 风格时间戳
DAY_FMT = "%Y-%m-%d"                # 日期串

SECONDS_PER_DAY = 86400


def utc_now() -> datetime:
    """当前 UTC 时间(带 tzinfo)。"""
    return datetime.now(timezone.utc)


def utc_today() -> date:
    """当前 UTC 日期。"""
    return utc_now().date()


def stamp(when: datetime | None = None) -> str:
    """格式化成 GitHub 风格时间戳。默认取现在。"""
    return (when or utc_now()).strftime(STAMP_FMT)


def parse_stamp(text: str) -> datetime | None:
    """解析 GitHub 风格时间戳。空串或格式不对返回 None。"""
    if not text:
        return None
    try:
        return datetime.strptime(text, STAMP_FMT).replace(tzinfo=timezone.utc)
    except ValueError:
        return None


def format_day(when: date) -> str:
    return when.strftime(DAY_FMT)


def parse_day(text: str) -> date | None:
    """解析 "YYYY-MM-DD"。空串或格式不对返回 None。"""
    if not text:
        return None
    try:
        return datetime.strptime(text, DAY_FMT).date()
    except ValueError:
        return None


def parse_moment(text: str) -> datetime | None:
    """把**任意一种**时间串解析成时刻。都不认返回 None。

    两种形状都得认,因为它们在同一个字段位置上混着出现:仓库的 `created_at` 来自 GitHub,
    是完整时间戳;而我们自己写的 `desc_updated_at` 是日期串。只认前者的话,
    「描述过期了吗」永远答不上来 —— 不报错,只是缓存的描述再也不刷新。

    日期串按当天 00:00 UTC 算。这会让年龄最多虚增不到一天,而所有用它的判断
    (创建于窗口内、描述超过 60 天)容忍度都远大于一天。
    """
    return parse_stamp(text) or (
        datetime.combine(day, datetime.min.time(), tzinfo=timezone.utc)
        if (day := parse_day((text or "")[:10])) else None
    )


def age_days(text: str, *, now: datetime | None = None) -> float | None:
    """这个时刻距今多少天(小数)。无法解析返回 None。

    返回小数而不是整数:判断「创建于窗口内」时,0.5 天和 0 天是两回事 ——
    取整会把今天凌晨创建的仓库和昨天创建的算成同一天。
    """
    parsed = parse_moment(text)
    if parsed is None:
        return None
    return ((now or utc_now()) - parsed).total_seconds() / SECONDS_PER_DAY


def days_between(later: date, earlier: date) -> int:
    """两个日期相差几天(later − earlier)。可以为负。"""
    return (later - earlier).days


def shift_days(when: date, days: int) -> date:
    """日期加减天数。`days` 为负即往前推。"""
    return when + timedelta(days=days)
